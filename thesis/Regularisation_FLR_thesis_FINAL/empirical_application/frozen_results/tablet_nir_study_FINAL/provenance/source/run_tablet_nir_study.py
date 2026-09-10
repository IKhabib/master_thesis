#!/usr/bin/env python3
"""Run the frozen empirical study of the 2002 tablet NIR data.

Stages are intentionally explicit:

* ``audit`` validates data identity and structure without fitting;
* ``development`` runs only calibration/validation analyses;
* ``full`` repeats the development analysis, freezes all resampling indices,
  and then performs the designated benchmark evaluation.

Every stage refuses an existing output directory and records a success marker.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib
import numpy as np
import pandas as pd
import scipy
import threadpoolctl

import tablet_nir_core as study_core


CODE_FILES = (
    "four_method_core.py",
    "tablet_nir_core.py",
    "run_tablet_nir_study.py",
    "make_tablet_nir_figures.py",
    "test_tablet_nir_core.py",
    "TABLET_NIR_README.md",
    "requirements-spectroscopy.txt",
)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialise {type(value).__name__}")


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_text(
        path,
        json.dumps(value, indent=2, sort_keys=True, default=_json_default) + "\n",
    )


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    pd.DataFrame(list(rows)).to_csv(temporary, index=False)
    temporary.replace(path)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    required = {"data", "methods", "estimation", "development_evaluation", "benchmark", "randomness", "variants"}
    missing = required - set(config)
    if missing:
        raise ValueError(f"configuration is missing {sorted(missing)}")
    if tuple(config["methods"]) != study_core.EXPECTED_METHODS:
        raise ValueError("configured method names/order differ from the frozen core")
    primary = [variant for variant in config["variants"] if variant["role"] == "primary"]
    if len(primary) != 1:
        raise ValueError("exactly one primary spectral variant is required")
    return config


def _sha256_text(path: Path) -> str:
    return study_core.sha256_file(path)


def _git_metadata(root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() if result.returncode == 0 else ""

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "status_porcelain": run("status", "--short"),
        "remote_origin": run("remote", "get-url", "origin"),
    }


def _software_metadata(thread_limit: int) -> dict[str, Any]:
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pandas": pd.__version__,
        "matplotlib": matplotlib.__version__,
        "threadpoolctl": threadpoolctl.__version__,
        "thread_limit": int(thread_limit),
        "threadpools_at_capture": threadpoolctl.threadpool_info(),
        "numpy_build_configuration": getattr(np.__config__, "CONFIG", {}),
    }


def _initialise_output(
    output_root: Path,
    config_path: Path,
    protocol_path: Path,
    config: Mapping[str, Any],
    source_root: Path,
    git_metadata: Mapping[str, Any],
    invocation: Mapping[str, Any],
) -> None:
    if output_root.exists():
        raise FileExistsError(
            f"output directory already exists and will not be overwritten: {output_root}"
        )
    output_root.mkdir(parents=True)
    (output_root / "provenance").mkdir()
    shutil.copy2(
        protocol_path,
        output_root / "provenance" / "SPECTROSCOPY_PROTOCOL.md",
    )
    resolved_config_path = output_root / "provenance" / "configuration.json"
    _atomic_json(resolved_config_path, config)
    _atomic_json(output_root / "provenance" / "run_invocation.json", invocation)
    source_snapshot = output_root / "provenance" / "source"
    source_snapshot.mkdir()
    for name in CODE_FILES:
        source_path = source_root / name
        if not source_path.is_file():
            raise FileNotFoundError(
                f"required empirical source file is absent: {source_path}"
            )
        shutil.copy2(source_path, source_snapshot / name)
    shutil.copy2(config_path, source_snapshot / "spectroscopy_config.json")
    shutil.copy2(protocol_path, source_snapshot / "SPECTROSCOPY_PROTOCOL.md")
    code_checksums = {
        name: _sha256_text(source_root / name)
        for name in CODE_FILES
    }
    for name, expected in code_checksums.items():
        archived = _sha256_text(source_snapshot / name)
        if archived != expected:
            raise IOError(f"source snapshot checksum mismatch for {name}")
    _atomic_json(
        output_root / "provenance" / "code_checksums.json",
        {
            "protocol_sha256": _sha256_text(protocol_path),
            "configuration_source_sha256": _sha256_text(config_path),
            "configuration_resolved_sha256": _sha256_text(resolved_config_path),
            "source_files": code_checksums,
            "git": dict(git_metadata),
        },
    )


def _add_metadata(row: Mapping[str, Any], **metadata: Any) -> dict[str, Any]:
    return {**metadata, **dict(row)}


def _fold_assignment_rows(
    data: study_core.RegressionData,
    folds: Sequence[np.ndarray],
    **metadata: Any,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for fold_number, indices in enumerate(folds, start=1):
        for index in np.asarray(indices, dtype=int):
            result.append(
                {
                    **metadata,
                    "fold": fold_number,
                    "data_index_zero_based": int(index),
                    "tablet_id": str(data.tablet_ids[index]),
                    "source_split": str(data.source_split[index]),
                    "source_row_one_based": int(data.source_row[index]),
                }
            )
    return result


def _prediction_rows(
    fits: Sequence[study_core.ProcedureFit],
    training_data: study_core.RegressionData,
    evaluation_data: study_core.RegressionData,
    evaluation_indices: np.ndarray,
    **metadata: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    prediction_records: list[dict[str, Any]] = []
    metric_records: list[dict[str, Any]] = []
    evaluation_indices = np.asarray(evaluation_indices, dtype=int)
    for fit in fits:
        prediction = fit.predict(evaluation_data.X)
        evaluation_prediction_finite = bool(np.all(np.isfinite(prediction)))
        evaluation_status = (
            fit.status
            if fit.status != "ok"
            else ("ok" if evaluation_prediction_finite else "nonfinite_prediction")
        )
        metrics = study_core.prediction_metrics(evaluation_data.y, prediction)
        metric_records.append(
            {
                **metadata,
                "method": fit.method,
                "fit_status": fit.status,
                "evaluation_status": evaluation_status,
                "evaluation_prediction_finite": evaluation_prediction_finite,
                "selected_components": fit.selected_components,
                **metrics,
            }
        )
        for local, original in enumerate(evaluation_indices):
            observed = float(evaluation_data.y[local])
            predicted = float(prediction[local])
            prediction_records.append(
                {
                    **metadata,
                    "method": fit.method,
                    "fit_status": fit.status,
                    "evaluation_status": evaluation_status,
                    "evaluation_prediction_finite": evaluation_prediction_finite,
                    "selected_components": fit.selected_components,
                    "evaluation_index_zero_based": int(original),
                    "tablet_id": str(evaluation_data.tablet_ids[local]),
                    "source_split": str(evaluation_data.source_split[local]),
                    "source_row_one_based": int(evaluation_data.source_row[local]),
                    "observed_assay": observed,
                    "prediction": predicted,
                    "residual_prediction_minus_observation": predicted - observed,
                    "training_observations": int(len(training_data.y)),
                }
            )
    return prediction_records, metric_records


def _fit_artifact_rows(
    fits: Sequence[study_core.ProcedureFit],
    wavelengths: np.ndarray,
    **metadata: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    summaries: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    coefficients: list[dict[str, Any]] = []
    for fit in fits:
        summaries.append(_add_metadata(study_core.fit_summary_record(fit), **metadata))
        curves.extend(
            _add_metadata(row, **metadata)
            for row in study_core.fit_curve_records(fit)
        )
        coefficients.extend(
            _add_metadata(row, **metadata)
            for row in study_core.coefficient_records(fit, wavelengths)
        )
    return summaries, curves, coefficients


def _inner_folds_for_training(
    data: study_core.RegressionData,
    config: Mapping[str, Any],
    *stream_parts: int,
) -> list[np.ndarray]:
    random_config = config["randomness"]
    return study_core.make_source_balanced_folds(
        data.y,
        data.source_split,
        int(config["estimation"]["inner_folds"]),
        [
            int(random_config["master_seed"]),
            int(random_config["inner_stream_code"]),
            *stream_parts,
        ],
    )


def _run_outer_plan(
    data: study_core.RegressionData,
    spec: study_core.PreprocessSpec,
    config: Mapping[str, Any],
    outer_plans: Sequence[tuple[int, Sequence[np.ndarray]]],
    *,
    analysis: str,
    stream_tag: int,
) -> dict[str, list[dict[str, Any]]]:
    predictions: list[dict[str, Any]] = []
    fold_metrics: list[dict[str, Any]] = []
    fit_summaries: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    coefficients: list[dict[str, Any]] = []
    outer_assignments: list[dict[str, Any]] = []
    inner_assignments: list[dict[str, Any]] = []

    total_fits = sum(len(folds) for _, folds in outer_plans)
    completed = 0
    for repeat, outer_folds in outer_plans:
        study_core.validate_folds(outer_folds, len(data.y))
        outer_assignments.extend(
            _fold_assignment_rows(
                data,
                outer_folds,
                analysis=analysis,
                repeat=repeat,
                assignment_level="outer_validation",
                index_scope="development_combined",
            )
        )
        for outer_fold, validation in enumerate(outer_folds, start=1):
            train = study_core.training_indices(len(data.y), validation)
            training_data = data.subset(train)
            validation_data = data.subset(validation)
            inner_folds = _inner_folds_for_training(
                training_data, config, stream_tag, repeat, outer_fold
            )
            inner_assignments.extend(
                _fold_assignment_rows(
                    training_data,
                    inner_folds,
                    analysis=analysis,
                    repeat=repeat,
                    outer_fold=outer_fold,
                    assignment_level="inner_validation",
                    index_scope="outer_training_local",
                )
            )
            fits = study_core.fit_all_procedures(
                training_data, spec, inner_folds, config["estimation"]
            )
            metadata = {
                "analysis": analysis,
                "variant": spec.name,
                "variant_role": spec.role,
                "instrument": spec.instrument,
                "repeat": repeat,
                "outer_fold": outer_fold,
                "evaluation_index_scope": "development_combined",
            }
            pred, metrics = _prediction_rows(
                fits,
                training_data,
                validation_data,
                np.asarray(validation, dtype=int),
                **metadata,
            )
            summary, curve, coefficient = _fit_artifact_rows(
                fits, data.wavelengths, **metadata
            )
            predictions.extend(pred)
            fold_metrics.extend(metrics)
            fit_summaries.extend(summary)
            curves.extend(curve)
            coefficients.extend(coefficient)
            completed += 1
            print(
                f"[{analysis}] completed outer fit {completed}/{total_fits} "
                f"(repeat {repeat}, fold {outer_fold})",
                flush=True,
            )

    return {
        "predictions": predictions,
        "fold_metrics": fold_metrics,
        "fit_summaries": fit_summaries,
        "curves": curves,
        "coefficients": coefficients,
        "outer_assignments": outer_assignments,
        "inner_assignments": inner_assignments,
    }


def _aggregate_prediction_metrics(
    prediction_rows: Sequence[Mapping[str, Any]],
    group_columns: Sequence[str],
) -> list[dict[str, Any]]:
    frame = pd.DataFrame(prediction_rows)
    rows: list[dict[str, Any]] = []
    if frame.empty:
        return rows
    for keys, group in frame.groupby(list(group_columns), sort=False, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        metadata = dict(zip(group_columns, keys, strict=True))
        rows.append(
            {
                **metadata,
                **study_core.prediction_metrics(
                    group["observed_assay"].to_numpy(),
                    group["prediction"].to_numpy(),
                ),
                "observations": int(len(group)),
                "failed_predictions": int(
                    np.count_nonzero(~np.isfinite(group["prediction"].to_numpy()))
                ),
            }
        )
    return rows


def run_development(
    archive_path: Path,
    output_root: Path,
    config: Mapping[str, Any],
) -> None:
    primary_mapping = next(
        variant for variant in config["variants"] if variant["role"] == "primary"
    )
    spec = study_core.PreprocessSpec.from_mapping(primary_mapping)
    data = study_core.load_regression_data(
        archive_path,
        config["data"],
        instrument=spec.instrument,
        splits=("calibrate", "validate"),
    )
    dev_config = config["development_evaluation"]
    random_config = config["randomness"]
    repeats = int(dev_config["outer_repeats"])
    outer_count = int(dev_config["outer_folds"])
    random_plans: list[tuple[int, Sequence[np.ndarray]]] = []
    for repeat in range(1, repeats + 1):
        folds = study_core.make_source_balanced_folds(
            data.y,
            data.source_split,
            outer_count,
            [
                int(random_config["master_seed"]),
                int(random_config["outer_stream_code"]),
                repeat,
            ],
        )
        random_plans.append((repeat, folds))
    random_results = _run_outer_plan(
        data,
        spec,
        config,
        random_plans,
        analysis="development_repeated_nested_cv",
        stream_tag=1,
    )

    blocked_folds = study_core.make_source_contiguous_folds(
        data.source_split, int(dev_config["blocked_outer_folds"])
    )
    blocked_results = _run_outer_plan(
        data,
        spec,
        config,
        [(1, blocked_folds)],
        analysis="development_source_contiguous_cv",
        stream_tag=2,
    )

    combined = {
        key: random_results[key] + blocked_results[key]
        for key in random_results
    }

    if bool(dev_config["run_original_split_sensitivity"]):
        calibration = study_core.load_regression_data(
            archive_path,
            config["data"],
            instrument=spec.instrument,
            splits=("calibrate",),
        )
        validation = study_core.load_regression_data(
            archive_path,
            config["data"],
            instrument=spec.instrument,
            splits=("validate",),
        )
        inner_folds = _inner_folds_for_training(calibration, config, 3, 1, 1)
        combined["inner_assignments"].extend(
            _fold_assignment_rows(
                calibration,
                inner_folds,
                analysis="original_split_sensitivity",
                repeat=1,
                outer_fold=1,
                assignment_level="inner_validation",
                index_scope="calibration_split_local",
            )
        )
        fits = study_core.fit_all_procedures(
            calibration, spec, inner_folds, config["estimation"]
        )
        metadata = {
            "analysis": "original_split_sensitivity",
            "variant": spec.name,
            "variant_role": spec.role,
            "instrument": spec.instrument,
            "repeat": 1,
            "outer_fold": 1,
            "evaluation_index_scope": "validation_split_local",
        }
        pred, metrics = _prediction_rows(
            fits,
            calibration,
            validation,
            np.arange(len(validation.y), dtype=int),
            **metadata,
        )
        summary, curve, coefficient = _fit_artifact_rows(
            fits, calibration.wavelengths, **metadata
        )
        combined["predictions"].extend(pred)
        combined["fold_metrics"].extend(metrics)
        combined["fit_summaries"].extend(summary)
        combined["curves"].extend(curve)
        combined["coefficients"].extend(coefficient)
        print("[original_split_sensitivity] completed calibration-to-validation fit", flush=True)

    repeat_metrics = _aggregate_prediction_metrics(
        combined["predictions"],
        ("analysis", "variant", "instrument", "repeat", "method"),
    )
    destination = output_root / "development"
    _atomic_csv(destination / "predictions.csv", combined["predictions"])
    _atomic_csv(destination / "fold_metrics.csv", combined["fold_metrics"])
    _atomic_csv(destination / "repeat_metrics.csv", repeat_metrics)
    _atomic_csv(destination / "fit_diagnostics.csv", combined["fit_summaries"])
    _atomic_csv(destination / "tuning_and_numerical_curves.csv", combined["curves"])
    _atomic_csv(destination / "selected_coefficients.csv", combined["coefficients"])
    _atomic_csv(
        output_root / "resampling" / "development_outer_assignments.csv",
        combined["outer_assignments"],
    )
    _atomic_csv(
        output_root / "resampling" / "development_inner_assignments.csv",
        combined["inner_assignments"],
    )
    _atomic_json(
        destination / "_SUCCESS.json",
        {
            "status": "passed",
            "benchmark_variables_loaded": False,
            "primary_variant": spec.name,
            "random_outer_repeats": repeats,
            "random_outer_folds": outer_count,
            "blocked_outer_folds": int(dev_config["blocked_outer_folds"]),
            "method_fit_count": int(len(combined["fit_summaries"])),
            "prediction_row_count": int(len(combined["predictions"])),
        },
    )


def _safe_key(*parts: str) -> str:
    text = "__".join(parts)
    return "".join(character if character.isalnum() else "_" for character in text)


def _declared_contrasts(
    benchmark_config: Mapping[str, Any],
    methods: Sequence[str],
) -> list[tuple[str, str, str, str]]:
    declared = [
        (
            str(contrast["label"]),
            str(contrast["left"]),
            str(contrast["right"]),
            str(contrast["scope"]),
        )
        for contrast in benchmark_config["contrasts"]
    ]
    method_set = set(methods)
    if any(
        left not in method_set
        or right not in method_set
        or left == right
        or scope not in {"focal", "secondary", "exploratory"}
        for _, left, right, scope in declared
    ):
        raise ValueError("configured benchmark contrasts are invalid")
    if sum(scope == "focal" for _, _, _, scope in declared) != 1:
        raise ValueError("exactly one focal benchmark contrast is required")
    used = {(left, right) for _, left, right, _ in declared}
    used |= {(right, left) for left, right in used}
    for label, left, right in study_core.all_pairwise_contrasts(methods):
        if (left, right) not in used:
            declared.append((label, left, right, "exploratory"))
    return declared


def _freeze_bootstrap_indices(
    output_root: Path,
    config: Mapping[str, Any],
    tablet_ids: np.ndarray,
) -> dict[str, np.ndarray]:
    benchmark_config = config["benchmark"]
    random_config = config["randomness"]
    replicates = int(benchmark_config["bootstrap_replicates"])
    sample_size = len(tablet_ids)
    matrices: dict[str, np.ndarray] = {
        "iid": study_core.make_bootstrap_indices(
            sample_size,
            replicates,
            [
                int(random_config["master_seed"]),
                int(random_config["bootstrap_stream_code"]),
            ],
        )
    }
    for block_length in benchmark_config["moving_block_lengths"]:
        block = int(block_length)
        matrices[f"moving_block_{block}"] = study_core.make_bootstrap_indices(
            sample_size,
            replicates,
            [
                int(random_config["master_seed"]),
                int(random_config["block_bootstrap_stream_code"]),
                block,
            ],
            block_length=block,
        )
    _atomic_npz(
        output_root / "resampling" / "benchmark_bootstrap_indices.npz",
        tablet_ids=np.asarray(tablet_ids, dtype="U32"),
        **matrices,
    )
    return matrices


def run_benchmark(
    archive_path: Path,
    output_root: Path,
    config: Mapping[str, Any],
    bootstrap_indices: Mapping[str, np.ndarray],
) -> None:
    primary_variant = next(
        variant for variant in config["variants"] if variant["role"] == "primary"
    )
    primary_spec = study_core.PreprocessSpec.from_mapping(primary_variant)
    primary_development = study_core.load_regression_data(
        archive_path,
        config["data"],
        instrument=primary_spec.instrument,
        splits=("calibrate", "validate"),
    )
    random_config = config["randomness"]
    final_inner_folds = study_core.make_source_balanced_folds(
        primary_development.y,
        primary_development.source_split,
        int(config["estimation"]["inner_folds"]),
        [
            int(random_config["master_seed"]),
            int(random_config["final_inner_stream_code"]),
        ],
    )
    final_assignment_rows = _fold_assignment_rows(
        primary_development,
        final_inner_folds,
        analysis="final_development_fit",
        repeat=1,
        assignment_level="inner_validation",
        index_scope="development_combined",
    )
    _atomic_csv(
        output_root / "resampling" / "final_inner_assignments.csv",
        final_assignment_rows,
    )
    primary_benchmark = study_core.load_regression_data(
        archive_path,
        config["data"],
        instrument=primary_spec.instrument,
        splits=("test",),
    )
    expected_test_ids = np.array(
        [
            f"test:{row:03d}"
            for row in range(1, int(config["data"]["expected_counts"]["test"]) + 1)
        ]
    )
    if not np.array_equal(primary_benchmark.tablet_ids, expected_test_ids):
        raise ValueError("benchmark IDs differ from the pre-frozen resampling plan")

    prediction_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    fit_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []
    primary_predictions: dict[str, np.ndarray] = {}
    variant_predictions: dict[str, dict[str, np.ndarray]] = {}
    point_metrics: dict[str, dict[str, dict[str, float]]] = {}

    for position, variant_mapping in enumerate(config["variants"], start=1):
        spec = study_core.PreprocessSpec.from_mapping(variant_mapping)
        development = study_core.load_regression_data(
            archive_path,
            config["data"],
            instrument=spec.instrument,
            splits=("calibrate", "validate"),
        )
        benchmark = study_core.load_regression_data(
            archive_path,
            config["data"],
            instrument=spec.instrument,
            splits=("test",),
        )
        if not np.array_equal(development.tablet_ids, primary_development.tablet_ids):
            raise ValueError("development tablet order differs across variants")
        if not np.array_equal(benchmark.tablet_ids, primary_benchmark.tablet_ids):
            raise ValueError("benchmark tablet order differs across variants")
        if not np.array_equal(benchmark.y, primary_benchmark.y):
            raise ValueError("benchmark responses differ across instruments")
        fits = study_core.fit_all_procedures(
            development, spec, final_inner_folds, config["estimation"]
        )
        metadata = {
            "analysis": "designated_benchmark",
            "variant": spec.name,
            "variant_role": spec.role,
            "instrument": spec.instrument,
            "evaluation_index_scope": "test_split_local",
        }
        pred, metrics = _prediction_rows(
            fits,
            development,
            benchmark,
            np.arange(len(benchmark.y), dtype=int),
            **metadata,
        )
        summary, curves, coefficients = _fit_artifact_rows(
            fits, development.wavelengths, **metadata
        )
        prediction_rows.extend(pred)
        metric_rows.extend(metrics)
        fit_rows.extend(summary)
        curve_rows.extend(curves)
        coefficient_rows.extend(coefficients)
        variant_predictions[spec.name] = {
            fit.method: fit.predict(benchmark.X) for fit in fits
        }
        point_metrics[spec.name] = {
            method: study_core.prediction_metrics(benchmark.y, values)
            for method, values in variant_predictions[spec.name].items()
        }
        if spec.role == "primary":
            primary_predictions = variant_predictions[spec.name]
        print(
            f"[designated_benchmark] completed variant {position}/{len(config['variants'])}: {spec.name}",
            flush=True,
        )

    if set(primary_predictions) != set(study_core.EXPECTED_METHODS):
        raise RuntimeError("primary benchmark predictions are incomplete")

    confidence = float(config["benchmark"]["bootstrap_confidence"])
    contrast_definitions = _declared_contrasts(
        config["benchmark"], config["methods"]
    )
    bootstrap_summary: list[dict[str, Any]] = []
    bootstrap_draw_archive: dict[str, np.ndarray] = {}
    bootstrap_draw_key_map: dict[str, dict[str, str]] = {}
    for variant_name, predictions in variant_predictions.items():
        for scheme, indices in bootstrap_indices.items():
            draws = study_core.bootstrap_metric_draws(
                primary_benchmark.y, predictions, indices
            )
            for name, values in draws.items():
                method, metric = name.split("|", maxsplit=1)
                key = _safe_key(variant_name, scheme, method, metric)
                bootstrap_draw_archive[key] = values
                bootstrap_draw_key_map[key] = {
                    "variant": variant_name,
                    "scheme": scheme,
                    "method": method,
                    "metric": metric,
                }
            rows = study_core.bootstrap_summary_rows(
                draws,
                point_metrics[variant_name],
                confidence=confidence,
                scheme=scheme,
                declared_contrasts=[
                    (label, left, right)
                    for label, left, right, _ in contrast_definitions
                ],
            )
            scope_by_label = {
                label: scope for label, _, _, scope in contrast_definitions
            }
            for row in rows:
                row["variant"] = variant_name
                row["variant_role"] = next(
                    variant["role"]
                    for variant in config["variants"]
                    if variant["name"] == variant_name
                )
                row["contrast_scope"] = (
                    scope_by_label.get(row["contrast_label"], "not_applicable")
                    if row["result_type"] == "method_contrast"
                    else "not_applicable"
                )
            bootstrap_summary.extend(rows)
        print(
            f"[bootstrap] completed all paired schemes for {variant_name}",
            flush=True,
        )

    destination = output_root / "benchmark"
    _atomic_csv(destination / "predictions.csv", prediction_rows)
    _atomic_csv(destination / "metrics.csv", metric_rows)
    _atomic_csv(destination / "fit_diagnostics.csv", fit_rows)
    _atomic_csv(destination / "tuning_and_numerical_curves.csv", curve_rows)
    _atomic_csv(destination / "selected_coefficients.csv", coefficient_rows)
    _atomic_csv(destination / "bootstrap_summary.csv", bootstrap_summary)
    _atomic_npz(
        destination / "bootstrap_metric_draws.npz", **bootstrap_draw_archive
    )
    _atomic_json(
        destination / "bootstrap_metric_draw_key_map.json",
        bootstrap_draw_key_map,
    )
    _atomic_json(
        destination / "_SUCCESS.json",
        {
            "status": "passed",
            "benchmark_label": config["benchmark"]["label"],
            "tablets": int(len(primary_benchmark.y)),
            "variants": [variant["name"] for variant in config["variants"]],
            "methods": list(config["methods"]),
            "bootstrap_schemes": list(bootstrap_indices),
            "bootstrap_replicates_per_scheme": int(
                config["benchmark"]["bootstrap_replicates"]
            ),
        },
    )


def _write_manifest(output_root: Path) -> None:
    excluded = {"artifact_manifest.json", "checksums.sha256"}
    entries: list[dict[str, Any]] = []
    for path in sorted(item for item in output_root.rglob("*") if item.is_file()):
        relative = path.relative_to(output_root).as_posix()
        if relative in excluded:
            continue
        entries.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": study_core.sha256_file(path),
            }
        )
    manifest = {
        "schema_version": "1.0.0",
        "file_count_excluding_manifest": len(entries),
        "files": entries,
    }
    _atomic_json(output_root / "artifact_manifest.json", manifest)
    lines = [f"{entry['sha256']}  {entry['path']}" for entry in entries]
    lines.append(
        f"{study_core.sha256_file(output_root / 'artifact_manifest.json')}  artifact_manifest.json"
    )
    _atomic_text(output_root / "checksums.sha256", "\n".join(lines) + "\n")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("audit", "development", "full"),
        default="development",
    )
    parser.add_argument("--archive", type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("spectroscopy_config.json"),
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path(__file__).with_name("SPECTROSCOPY_PROTOCOL.md"),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.self_test_only:
        study_core.run_self_tests()
        print("tablet NIR empirical self-tests passed", flush=True)
        return 0
    if args.archive is None or args.output is None:
        raise SystemExit("--archive and --output are required unless --self-test-only is used")
    config_path = args.config.resolve()
    protocol_path = args.protocol.resolve()
    archive_path = args.archive.resolve()
    output_root = args.output.resolve()
    source_root = Path(__file__).resolve().parent
    config = _load_config(config_path)
    if not protocol_path.is_file():
        raise FileNotFoundError(protocol_path)

    started = time.perf_counter()
    git_metadata = _git_metadata(source_root)
    invocation = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "argv": [sys.executable, *sys.argv],
        "cwd": str(Path.cwd().resolve()),
        "stage": args.stage,
        "archive": str(archive_path),
        "config": str(config_path),
        "protocol": str(protocol_path),
        "output": str(output_root),
    }
    _initialise_output(
        output_root,
        config_path,
        protocol_path,
        config,
        source_root,
        git_metadata,
        invocation,
    )
    thread_limit = int(config["estimation"]["thread_limit"])
    try:
        with threadpoolctl.threadpool_limits(limits=thread_limit):
            _atomic_json(
                output_root / "provenance" / "software_environment.json",
                _software_metadata(thread_limit),
            )
            include_benchmark = args.stage == "full"
            bootstrap_indices: dict[str, np.ndarray] | None = None
            if include_benchmark:
                expected_test_ids = np.array(
                    [
                        f"test:{row:03d}"
                        for row in range(
                            1,
                            int(config["data"]["expected_counts"]["test"]) + 1,
                        )
                    ]
                )
                bootstrap_indices = _freeze_bootstrap_indices(
                    output_root, config, expected_test_ids
                )
                print("[resampling] benchmark bootstrap indices frozen", flush=True)
            audit = study_core.audit_archive(
                archive_path,
                config["data"],
                include_benchmark=include_benchmark,
            )
            _atomic_json(output_root / "audit" / "data_quality.json", audit)
            _atomic_json(
                output_root / "audit" / "_SUCCESS.json",
                {"status": "passed", "benchmark_variables_loaded": include_benchmark},
            )
            print(f"[audit] passed ({args.stage} stage)", flush=True)
            if args.stage in ("development", "full"):
                run_development(archive_path, output_root, config)
                print("[development] archive complete", flush=True)
            if args.stage == "full":
                assert bootstrap_indices is not None
                run_benchmark(
                    archive_path, output_root, config, bootstrap_indices
                )
                print("[benchmark] archive complete", flush=True)
                from make_tablet_nir_figures import make_figures

                make_figures(output_root, archive_path, config)
                print("[figures] thesis-ready tables and figures complete", flush=True)

        _atomic_json(
            output_root / "_SUCCESS.json",
            {
                "status": "passed",
                "stage": args.stage,
                "elapsed_seconds": float(time.perf_counter() - started),
                "protocol_sha256": _sha256_text(protocol_path),
                "configuration_source_sha256": _sha256_text(config_path),
                "configuration_resolved_sha256": _sha256_text(
                    output_root / "provenance" / "configuration.json"
                ),
            },
        )
        _write_manifest(output_root)
    except BaseException as exc:
        _atomic_json(
            output_root / "_FAILED.json",
            {
                "status": "failed",
                "stage": args.stage,
                "elapsed_seconds": float(time.perf_counter() - started),
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            },
        )
        raise
    print(f"completed {args.stage} stage in {time.perf_counter() - started:.2f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
