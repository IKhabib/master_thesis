

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import re
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats

import four_method_core as core


COEFFICIENT_INDICES = (0, 1, 2)
BETA_POINTS = (0.00, 0.25, 0.50, 0.75, 1.00)
EXPECTED_METHODS = (
    "CG-FPLS-code",
    "Raw FPLS",
    "Arnoldi FPLS",
    "FPCR",
)
SHORT_LABELS = {
    "CG-FPLS-code": "CG code",
    "Raw FPLS": "Raw FPLS",
    "Arnoldi FPLS": "Arnoldi FPLS",
    "FPCR": "FPCR",
}
METHOD_COLORS = {
    "CG-FPLS-code": "#0072B2",
    "Raw FPLS": "#D55E00",
    "Arnoldi FPLS": "#009E73",
    "FPCR": "#CC79A7",
}
BIAS_COLOR = "#D55E00"
VARIANCE_COLOR = "#0072B2"
NOISE_COLOR = "#777777"


@dataclass(frozen=True)
class FrozenInputs:
    configuration: Dict[str, object]
    method_names: Tuple[str, ...]
    model_names: Tuple[str, ...]
    selected_components: np.ndarray
    ise: np.ndarray
    mspe: np.ndarray
    method_instability: np.ndarray


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _is_complete_pdf(path: Path) -> bool:
    try:
        data = path.read_bytes()
    except OSError:
        return False
    return _pdf_bytes_are_complete(data)


def _pdf_bytes_are_complete(data: bytes) -> bool:
    if len(data) < 100 or not data.startswith(b"%PDF-"):
        return False
    if b"%%EOF" not in data[-2048:]:
        return False
    matches = list(
        re.finditer(rb"startxref\s+(\d+)\s+%%EOF", data[-4096:])
    )
    if not matches:
        return False
    xref_offset = int(matches[-1].group(1))
    if not 0 <= xref_offset < len(data):
        return False
    xref_prefix = data[xref_offset : xref_offset + 64]
    return bool(
        xref_prefix.startswith(b"xref")
        or re.match(rb"\d+\s+\d+\s+obj", xref_prefix)
    )


def _write_pdf_bytes(path: Path, data: bytes) -> None:
    if not _pdf_bytes_are_complete(data):
        raise OSError(f"incomplete PDF generated for {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.stem}.",
        suffix=".pdf",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        if not _is_complete_pdf(temporary):
            raise OSError(f"incomplete PDF write: {path}")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        buffer = io.BytesIO()
        fig.savefig(buffer, format="pdf", bbox_inches="tight")
        _write_pdf_bytes(path, buffer.getvalue())
    finally:
        plt.close(fig)


def _save_multipage(path: Path, figures: Iterable[plt.Figure]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure_list = list(figures)
    if not figure_list:
        raise ValueError(f"cannot create an empty PDF: {path}")
    try:
        buffer = io.BytesIO()
        with PdfPages(buffer) as pdf:
            for fig in figure_list:
                pdf.savefig(fig, bbox_inches="tight")
        _write_pdf_bytes(path, buffer.getvalue())
    finally:
        for fig in figure_list:
            plt.close(fig)


def _resolve_inputs(args: argparse.Namespace) -> Tuple[Path, Path]:
    results_dir = Path(args.results_dir)
    raw_path = Path(args.raw_results) if args.raw_results else results_dir / "raw_results.npz"
    config_path = (
        Path(args.configuration)
        if args.configuration
        else results_dir / "configuration.json"
    )
    if not raw_path.is_file():
        raise FileNotFoundError(raw_path)
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    return raw_path, config_path


def load_frozen_inputs(
    raw_path: Path,
    config_path: Path,
    replications: Optional[int],
) -> FrozenInputs:
    with config_path.open("r", encoding="utf-8") as handle:
        configuration = json.load(handle)
    with np.load(raw_path, allow_pickle=False) as archive:
        required = {
            "method_names",
            "model_names",
            "selected_components",
            "ise",
            "mspe",
            "method_instability",
        }
        missing = required.difference(archive.files)
        if missing:
            raise KeyError(f"raw archive is missing {sorted(missing)}")
        method_names = tuple(archive["method_names"].astype(str).tolist())
        model_names = tuple(archive["model_names"].astype(str).tolist())
        selected = np.asarray(archive["selected_components"], dtype=int)
        ise = np.asarray(archive["ise"], dtype=float)
        mspe = np.asarray(archive["mspe"], dtype=float)
        instability = np.asarray(archive["method_instability"], dtype=bool)

    if method_names != EXPECTED_METHODS:
        raise ValueError(
            "unexpected methods in frozen archive:\n"
            f"  found: {method_names}\n  expected: {EXPECTED_METHODS}"
        )
    if model_names != ("Model 1", "Model 2", "Model 3"):
        raise ValueError(f"unexpected models: {model_names}")
    if not (
        selected.shape == ise.shape == mspe.shape == instability.shape
        and selected.shape[0] == len(model_names)
        and selected.shape[2] == len(method_names)
    ):
        raise ValueError("frozen arrays have incompatible shapes")
    archived_replications = selected.shape[1]
    use_replications = archived_replications if replications is None else replications
    if use_replications < 2 or use_replications > archived_replications:
        raise ValueError(
            f"replications must be in [2, {archived_replications}]"
        )

    configured = int(configuration["replications_per_model"])
    if configured != archived_replications:
        raise ValueError(
            "configuration and raw archive disagree on replication count"
        )
    configured_methods = tuple(configuration.get("methods", ()))
    if configured_methods and configured_methods != method_names:
        raise ValueError("configuration and raw archive disagree on method order")

    sl = slice(0, use_replications)
    return FrozenInputs(
        configuration=configuration,
        method_names=method_names,
        model_names=model_names,
        selected_components=selected[:, sl].copy(),
        ise=ise[:, sl].copy(),
        mspe=mspe[:, sl].copy(),
        method_instability=instability[:, sl].copy(),
    )


def _point_values(curves: np.ndarray, grid: np.ndarray, points: np.ndarray) -> np.ndarray:
    curves = np.asarray(curves, dtype=float)
    return np.asarray(
        [np.interp(points, grid, curve) for curve in curves], dtype=float
    )


def _selected_beta_matrix(
    y: np.ndarray,
    X: np.ndarray,
    K: np.ndarray,
    r: np.ndarray,
    selected: np.ndarray,
    configuration: Mapping[str, object],
    *,
    include_raw: bool,
) -> Tuple[np.ndarray, bool]:
    """Evaluate all methods at archived component counts."""
    m_max = int(configuration["maximum_components"])
    n, T = X.shape
    cg_path = core.cg_fpls_path(r, K, m_max)
    raw_path = (
        core.raw_fpls_path(y, X, K, r, m_max) if include_raw else None
    )
    stable_path = core.arnoldi_fpls_path(
        y,
        X,
        K,
        r,
        m_max,
        rank_tolerance=float(configuration["arnoldi_rank_tolerance"]),
    )
    fpcr_path = core.fpcr_spectral_path(
        r, K, min(m_max, n - 1, T - 1)
    )

    cg_index = EXPECTED_METHODS.index("CG-FPLS-code")
    raw_index = EXPECTED_METHODS.index("Raw FPLS")
    stable_index = EXPECTED_METHODS.index("Arnoldi FPLS")
    fpcr_index = EXPECTED_METHODS.index("FPCR")
    cg_m = int(selected[cg_index])
    raw_m = int(selected[raw_index])
    stable_m = int(selected[stable_index])
    fpcr_m = int(selected[fpcr_index])
    if min(cg_m, raw_m, stable_m, fpcr_m) < 1:
        raise ValueError("all archived component counts must be positive")

    beta = np.zeros((len(EXPECTED_METHODS), T), dtype=float)
    beta[cg_index] = cg_path[:, cg_m - 1]
    if not stable_path.valid[stable_m - 1]:
        raise RuntimeError("archived Arnoldi component is invalid during replay")
    if fpcr_m > fpcr_path.shape[1]:
        raise RuntimeError("archived FPCR count exceeds replay path")
    raw_valid = bool(
        include_raw
        and raw_path is not None
        and raw_path.valid[raw_m - 1]
        and np.all(np.isfinite(raw_path.beta[:, raw_m - 1]))
    )
    if raw_valid and raw_path is not None:
        beta[raw_index] = raw_path.beta[:, raw_m - 1]
    else:
        beta[raw_index] = np.nan
    beta[stable_index] = stable_path.beta[:, stable_m - 1]
    beta[fpcr_index] = fpcr_path[:, fpcr_m - 1]
    return beta, raw_valid


def _replay_one_model(
    model_index: int,
    configuration: Mapping[str, object],
    selected_components: np.ndarray,
    raw_stable_mask: np.ndarray,
    stored_ise: np.ndarray,
    stored_mspe: np.ndarray,
) -> Dict[str, np.ndarray | int | float | str]:
    """Replay one independent model stream; safe to run in a worker process."""
    n = int(configuration["n"])
    J = int(configuration["basis_size_J"])
    T = int(configuration["grid_size_T"])
    noise_sd = float(configuration["noise_sd"])
    seed = int(configuration["seed"])
    replications = selected_components.shape[0]
    method_count = selected_components.shape[1]
    grid = np.linspace(0.0, 1.0, T)
    points = np.asarray(BETA_POINTS, dtype=float)
    basis = core.create_cosine_basis(grid, J)
    models = core.make_model_specs(J, basis)
    model = models[model_index]
    seed_sequence = np.random.SeedSequence(seed).spawn(len(models))[model_index]
    rng = np.random.default_rng(seed_sequence)
    sqrt_eigenvalues = np.sqrt(model.eigenvalues)
    basis_pinv = np.linalg.pinv(basis.T)

    coefficient_samples = np.full(
        (replications, method_count, len(COEFFICIENT_INDICES)), dtype=float
        , fill_value=np.nan
    )
    point_samples = np.full(
        (replications, method_count, len(BETA_POINTS)), dtype=float
        , fill_value=np.nan
    )
    sum_beta = np.zeros((method_count, T), dtype=float)
    sumsq_beta = np.zeros_like(sum_beta)
    sum_prediction_coordinates = np.zeros((method_count, J), dtype=float)
    sumsq_prediction_coordinates = np.zeros_like(sum_prediction_coordinates)
    body_sum_beta = np.zeros_like(sum_beta)
    body_sumsq_beta = np.zeros_like(sum_beta)
    body_sum_prediction_coordinates = np.zeros_like(sum_prediction_coordinates)
    body_sumsq_prediction_coordinates = np.zeros_like(sum_prediction_coordinates)
    all_count = np.zeros(method_count, dtype=int)
    body_count = 0
    raw_replay_failure_count = 0
    replay_ise = np.full((replications, method_count), np.nan, dtype=float)
    replay_mspe = np.full_like(replay_ise, np.nan)

    for replication in range(replications):
        scores = rng.normal(size=(n, J))
        X = (scores * sqrt_eigenvalues) @ basis
        y = X @ model.beta / T + rng.normal(0.0, noise_sd, n)
        K = X.T @ X / (n * T)
        r = X.T @ y / n

        beta, raw_valid = _selected_beta_matrix(
            y,
            X,
            K,
            r,
            selected_components[replication],
            configuration,
            include_raw=bool(raw_stable_mask[replication]),
        )
        if raw_stable_mask[replication] and not raw_valid:
            raw_replay_failure_count += 1
        coefficients = beta @ basis_pinv.T
        coefficient_samples[replication] = coefficients[
            :, COEFFICIENT_INDICES
        ]
        point_samples[replication] = _point_values(beta, grid, points)
        prediction_coordinates = beta @ basis.T / T
        valid_methods = np.all(np.isfinite(beta), axis=1)
        sum_beta[valid_methods] += beta[valid_methods]
        sumsq_beta[valid_methods] += beta[valid_methods] ** 2
        sum_prediction_coordinates[valid_methods] += prediction_coordinates[
            valid_methods
        ]
        sumsq_prediction_coordinates[valid_methods] += prediction_coordinates[
            valid_methods
        ] ** 2
        all_count += valid_methods.astype(int)
        paired_raw_stable = bool(raw_stable_mask[replication] and raw_valid)
        if paired_raw_stable:
            body_sum_beta += beta
            body_sumsq_beta += beta * beta
            body_sum_prediction_coordinates += prediction_coordinates
            body_sumsq_prediction_coordinates += prediction_coordinates ** 2
            body_count += 1

        replay_ise[replication] = np.mean(
            (beta - model.beta[None, :]) ** 2, axis=1
        )

        # Preserve the original RNG order: folds are drawn before the test set.
        rng.permutation(n)
        test_scores = rng.normal(size=(n, J))
        X_test = (test_scores * sqrt_eigenvalues) @ basis
        y_test = X_test @ model.beta / T + rng.normal(0.0, noise_sd, n)
        predictions = X_test @ beta.T / T
        replay_mspe[replication] = np.mean(
            (y_test[:, None] - predictions) ** 2, axis=0
        )

        if replication == 0 or (replication + 1) % max(1, replications // 10) == 0:
            print(
                f"{model.name}: replay {replication + 1}/{replications}",
                flush=True,
            )

    true_coefficients = basis_pinv @ model.beta
    true_points = np.interp(points, grid, model.beta)
    true_prediction_coordinates = basis @ model.beta / T
    abs_ise = np.abs(replay_ise - stored_ise)
    abs_mspe = np.abs(replay_mspe - stored_mspe)
    rel_ise = abs_ise / np.maximum(np.abs(stored_ise), 1.0)
    rel_mspe = abs_mspe / np.maximum(np.abs(stored_mspe), 1.0)
    return {
        "model_index": model_index,
        "model_name": model.name,
        "coefficient_samples": coefficient_samples,
        "point_samples": point_samples,
        "sum_beta": sum_beta,
        "sumsq_beta": sumsq_beta,
        "sum_prediction_coordinates": sum_prediction_coordinates,
        "sumsq_prediction_coordinates": sumsq_prediction_coordinates,
        "body_sum_beta": body_sum_beta,
        "body_sumsq_beta": body_sumsq_beta,
        "body_sum_prediction_coordinates": body_sum_prediction_coordinates,
        "body_sumsq_prediction_coordinates": body_sumsq_prediction_coordinates,
        "all_count": all_count,
        "body_count": body_count,
        "raw_replay_failure_count": raw_replay_failure_count,
        "replay_ise": replay_ise,
        "replay_mspe": replay_mspe,
        "true_beta": model.beta,
        "true_coefficients": true_coefficients,
        "true_points": true_points,
        "true_prediction_coordinates": true_prediction_coordinates,
        "eigenvalues": model.eigenvalues,
        "max_abs_ise_error": float(np.nanmax(abs_ise)),
        "max_relative_ise_error": float(np.nanmax(rel_ise)),
        "max_abs_mspe_error": float(np.nanmax(abs_mspe)),
        "max_relative_mspe_error": float(np.nanmax(rel_mspe)),
    }


def replay_selected_estimates(
    frozen: FrozenInputs,
    jobs: int,
) -> Dict[str, np.ndarray]:
    model_count, replications, method_count = frozen.selected_components.shape
    raw_index = frozen.method_names.index("Raw FPLS")
    raw_stable = ~frozen.method_instability[:, :, raw_index]
    kwargs = [
        (
            model_index,
            frozen.configuration,
            frozen.selected_components[model_index],
            raw_stable[model_index],
            frozen.ise[model_index],
            frozen.mspe[model_index],
        )
        for model_index in range(model_count)
    ]

    results: List[Optional[Dict[str, object]]] = [None] * model_count
    if jobs == 1:
        for item in kwargs:
            result = _replay_one_model(*item)
            results[int(result["model_index"])] = result
    else:
        with ProcessPoolExecutor(max_workers=jobs) as executor:
            futures = {
                executor.submit(_replay_one_model, *item): item[0]
                for item in kwargs
            }
            for future in as_completed(futures):
                result = future.result()
                results[int(result["model_index"])] = result

    if any(result is None for result in results):
        raise RuntimeError("one or more replay workers did not return")
    complete = [result for result in results if result is not None]
    stacked: Dict[str, np.ndarray] = {}
    array_keys = (
        "coefficient_samples",
        "point_samples",
        "sum_beta",
        "sumsq_beta",
        "sum_prediction_coordinates",
        "sumsq_prediction_coordinates",
        "body_sum_beta",
        "body_sumsq_beta",
        "body_sum_prediction_coordinates",
        "body_sumsq_prediction_coordinates",
        "replay_ise",
        "replay_mspe",
        "true_beta",
        "true_coefficients",
        "true_points",
        "true_prediction_coordinates",
        "eigenvalues",
    )
    for key in array_keys:
        stacked[key] = np.stack(
            [np.asarray(result[key]) for result in complete], axis=0
        )
    stacked["body_count"] = np.asarray(
        [int(result["body_count"]) for result in complete], dtype=int
    )
    stacked["all_count"] = np.stack(
        [np.asarray(result["all_count"], dtype=int) for result in complete],
        axis=0,
    )
    stacked["raw_replay_failure_count"] = np.asarray(
        [int(result["raw_replay_failure_count"]) for result in complete],
        dtype=int,
    )
    for key in (
        "max_abs_ise_error",
        "max_relative_ise_error",
        "max_abs_mspe_error",
        "max_relative_mspe_error",
    ):
        stacked[key] = np.asarray(
            [float(result[key]) for result in complete], dtype=float
        )
    stacked["method_names"] = np.asarray(frozen.method_names)
    stacked["model_names"] = np.asarray(frozen.model_names)
    stacked["coefficient_indices"] = np.asarray(COEFFICIENT_INDICES, dtype=int)
    stacked["beta_points"] = np.asarray(BETA_POINTS, dtype=float)
    stacked["replications"] = np.asarray(replications, dtype=int)
    return stacked


def _validate_replay(
    cache: Mapping[str, np.ndarray], frozen: FrozenInputs
) -> Dict[str, object]:
    replay_ise = np.asarray(cache["replay_ise"], dtype=float)
    replay_mspe = np.asarray(cache["replay_mspe"], dtype=float)
    abs_ise = np.abs(replay_ise - frozen.ise)
    abs_mspe = np.abs(replay_mspe - frozen.mspe)
    scaled_ise = abs_ise / np.maximum(np.abs(frozen.ise), 1.0)
    scaled_mspe = abs_mspe / np.maximum(np.abs(frozen.mspe), 1.0)
    per_method: List[Dict[str, object]] = []
    passed = True
    for method_index, method in enumerate(frozen.method_names):
        # Raw normal equations are intentionally platform-sensitive.  A looser
        # replay tolerance is used only for that diagnostic implementation;
        # every stabilized/CG/FPCR estimate must reproduce much more tightly.
        tolerance = 5e-3 if method == "Raw FPLS" else 5e-9
        method_ise = scaled_ise[:, :, method_index]
        method_mspe = scaled_mspe[:, :, method_index]
        finite_ise = method_ise[np.isfinite(method_ise)]
        finite_mspe = method_mspe[np.isfinite(method_mspe)]
        if len(finite_ise) == 0 or len(finite_mspe) == 0:
            raise RuntimeError(f"no replayed risks available for {method}")
        method_max_ise = float(np.max(finite_ise))
        method_max_mspe = float(np.max(finite_mspe))
        method_passed = bool(
            method_max_ise <= tolerance and method_max_mspe <= tolerance
        )
        passed = passed and method_passed
        per_method.append(
            {
                "method": method,
                "tolerance": tolerance,
                "max_scaled_ise_error": method_max_ise,
                "max_scaled_mspe_error": method_max_mspe,
                "replayed_replications": int(len(finite_ise)),
                "passed": method_passed,
            }
        )
    max_abs_ise = float(np.nanmax(abs_ise))
    max_rel_ise = float(np.nanmax(scaled_ise))
    max_abs_mspe = float(np.nanmax(abs_mspe))
    max_rel_mspe = float(np.nanmax(scaled_mspe))
    validation = {
        "passed": passed,
        "criterion": (
            "maximum absolute error divided by max(abs(archived value), 1); "
            "tolerance 5e-9 for stabilized/CG/FPCR methods and 5e-3 for the "
            "intentionally platform-sensitive raw normal equations"
        ),
        "max_absolute_ise_error": max_abs_ise,
        "max_relative_ise_error": max_rel_ise,
        "max_absolute_mspe_error": max_abs_mspe,
        "max_relative_mspe_error": max_rel_mspe,
        "per_method": per_method,
        "raw_replay_failure_count_by_model": cache[
            "raw_replay_failure_count"
        ].astype(int).tolist(),
    }
    if not passed:
        raise RuntimeError(
            "deterministic replay does not match the frozen archive: "
            + ", ".join(
                f"{row['method']} ISE={row['max_scaled_ise_error']:.3g} "
                f"MSPE={row['max_scaled_mspe_error']:.3g}"
                for row in per_method
                if not row["passed"]
            )
        )
    return validation


def _save_cache(path: Path, cache: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **cache)


def _load_cache(
    path: Path,
    frozen: FrozenInputs,
) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        cache = {key: np.asarray(archive[key]) for key in archive.files}
    required = {
        "coefficient_samples",
        "point_samples",
        "sum_beta",
        "sumsq_beta",
        "sum_prediction_coordinates",
        "sumsq_prediction_coordinates",
        "body_sum_beta",
        "body_sumsq_beta",
        "body_sum_prediction_coordinates",
        "body_sumsq_prediction_coordinates",
        "body_count",
        "all_count",
        "raw_replay_failure_count",
        "replay_ise",
        "replay_mspe",
        "true_beta",
        "true_coefficients",
        "true_points",
        "true_prediction_coordinates",
        "eigenvalues",
        "max_abs_ise_error",
        "max_relative_ise_error",
        "max_abs_mspe_error",
        "max_relative_mspe_error",
        "method_names",
        "model_names",
        "coefficient_indices",
        "beta_points",
        "replications",
    }
    missing = required.difference(cache)
    if missing:
        raise KeyError(f"replay cache is missing {sorted(missing)}")
    if tuple(cache["method_names"].astype(str)) != frozen.method_names:
        raise ValueError("replay cache has different methods")
    if tuple(cache["model_names"].astype(str)) != frozen.model_names:
        raise ValueError("replay cache has different models")
    if int(cache["replications"]) != frozen.selected_components.shape[1]:
        raise ValueError("replay cache has a different replication count")
    return cache


def _empirical_variance(sum_values: np.ndarray, sumsq_values: np.ndarray, count: int) -> np.ndarray:
    if count < 1:
        raise ValueError("variance count must be positive")
    mean = sum_values / count
    variance = sumsq_values / count - mean * mean
    tolerance = 100.0 * np.finfo(float).eps * np.maximum(
        np.abs(sumsq_values / count), 1.0
    )
    variance = np.where(
        (variance < 0.0) & (variance >= -tolerance), 0.0, variance
    )
    if np.any(variance < 0.0):
        raise FloatingPointError("negative empirical variance beyond roundoff")
    return variance


def build_bias_variance_rows(
    frozen: FrozenInputs,
    cache: Mapping[str, np.ndarray],
) -> List[Dict[str, object]]:
    replications = frozen.selected_components.shape[1]
    raw_index = frozen.method_names.index("Raw FPLS")
    rows: List[Dict[str, object]] = []

    for subset in ("all replications", "paired raw-stable subset"):
        if subset == "all replications":
            counts = cache["all_count"].astype(int)
            sum_beta = cache["sum_beta"]
            sumsq_beta = cache["sumsq_beta"]
            sum_pred = cache["sum_prediction_coordinates"]
            sumsq_pred = cache["sumsq_prediction_coordinates"]
        else:
            counts = np.repeat(
                cache["body_count"].astype(int)[:, None],
                len(frozen.method_names),
                axis=1,
            )
            sum_beta = cache["body_sum_beta"]
            sumsq_beta = cache["body_sumsq_beta"]
            sum_pred = cache["body_sum_prediction_coordinates"]
            sumsq_pred = cache["body_sumsq_prediction_coordinates"]

        for model_index, model_name in enumerate(frozen.model_names):
            true_beta = cache["true_beta"][model_index]
            true_pred = cache["true_prediction_coordinates"][model_index]
            eigenvalues = cache["eigenvalues"][model_index]
            if subset == "all replications":
                metric_mask = np.ones(replications, dtype=bool)
            else:
                metric_mask = np.isfinite(
                    cache["replay_ise"][model_index, :, raw_index]
                )

            for method_index, method in enumerate(frozen.method_names):
                count = int(counts[model_index, method_index])
                decomposition_available = not (
                    subset == "all replications" and method == "Raw FPLS"
                )
                if decomposition_available:
                    beta_mean = sum_beta[model_index, method_index] / count
                    beta_variance = _empirical_variance(
                        sum_beta[model_index, method_index],
                        sumsq_beta[model_index, method_index],
                        count,
                    )
                    slope_squared_bias = float(
                        np.mean((beta_mean - true_beta) ** 2)
                    )
                    slope_variance = float(np.mean(beta_variance))
                    slope_mse = slope_squared_bias + slope_variance

                    pred_mean = sum_pred[model_index, method_index] / count
                    pred_variance_by_coordinate = _empirical_variance(
                        sum_pred[model_index, method_index],
                        sumsq_pred[model_index, method_index],
                        count,
                    )
                    prediction_squared_bias = float(
                        np.sum(eigenvalues * (pred_mean - true_pred) ** 2)
                    )
                    prediction_variance = float(
                        np.sum(eigenvalues * pred_variance_by_coordinate)
                    )
                    replay_values = cache["replay_ise"][
                        model_index, metric_mask, method_index
                    ]
                    replay_values = replay_values[np.isfinite(replay_values)]
                    replay_mean_ise = float(np.mean(replay_values))
                else:
                    slope_squared_bias = math.nan
                    slope_variance = math.nan
                    slope_mse = math.nan
                    prediction_squared_bias = math.nan
                    prediction_variance = math.nan
                    replay_mean_ise = math.nan
                noise_variance = float(frozen.configuration["noise_sd"]) ** 2
                expected_mspe = float(
                    prediction_squared_bias
                    + prediction_variance
                    + noise_variance
                )
                stored_mean_ise = float(
                    np.mean(
                        frozen.ise[model_index, metric_mask, method_index]
                    )
                )
                stored_mean_mspe = float(
                    np.mean(
                        frozen.mspe[model_index, metric_mask, method_index]
                    )
                )
                rows.append(
                    {
                        "subset": subset,
                        "model": model_name,
                        "method": method,
                        "decomposition_available": decomposition_available,
                        "decomposition_scope": (
                            subset
                            if decomposition_available
                            else "not recoverable for archived unstable raw fits"
                        ),
                        "archived_replications_in_scope": int(
                            np.count_nonzero(metric_mask)
                        ),
                        "decomposition_replications": count,
                        "slope_squared_bias": slope_squared_bias,
                        "slope_variance": slope_variance,
                        "slope_mse": slope_mse,
                        "archived_mean_ise": stored_mean_ise,
                        "replayed_mean_ise": replay_mean_ise,
                        "slope_decomposition_minus_replayed_ise": (
                            slope_mse - replay_mean_ise
                        ),
                        "prediction_squared_bias": prediction_squared_bias,
                        "prediction_variance": prediction_variance,
                        "irreducible_noise_variance": noise_variance,
                        "expected_test_mspe": expected_mspe,
                        "archived_mean_test_mspe": stored_mean_mspe,
                        "expected_minus_archived_test_mspe": (
                            expected_mspe - stored_mean_mspe
                        ),
                        "variance_denominator": (
                            count if decomposition_available else ""
                        ),
                    }
                )
    return rows


def _distribution_summary(
    values: np.ndarray,
    true_value: float,
) -> Dict[str, float]:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if len(finite) < 2:
        raise ValueError("distribution summary needs at least two finite values")
    mean = float(np.mean(finite))
    variance = float(np.var(finite, ddof=0))
    return {
        "true": true_value,
        "mean": mean,
        "bias": mean - true_value,
        "variance": variance,
        "standard_deviation": math.sqrt(variance),
        "median": float(np.median(finite)),
        "p025": float(np.quantile(finite, 0.025)),
        "p975": float(np.quantile(finite, 0.975)),
        "minimum": float(np.min(finite)),
        "maximum": float(np.max(finite)),
        "skewness": float(stats.skew(finite, bias=False)),
        "excess_kurtosis": float(stats.kurtosis(finite, fisher=True, bias=False)),
    }


def build_coefficient_rows(
    frozen: FrozenInputs,
    cache: Mapping[str, np.ndarray],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    samples = cache["coefficient_samples"]
    for model_index, model_name in enumerate(frozen.model_names):
        for method_index, method in enumerate(frozen.method_names):
            for local_index, coefficient_index in enumerate(COEFFICIENT_INDICES):
                values = samples[model_index, :, method_index, local_index]
                summary = _distribution_summary(
                    values,
                    float(cache["true_coefficients"][model_index, coefficient_index]),
                )
                rows.append(
                    {
                        "model": model_name,
                        "method": method,
                        "distribution_scope": (
                            "archived numerically stable raw fits"
                            if method == "Raw FPLS"
                            else "all replications"
                        ),
                        "replications": int(np.count_nonzero(np.isfinite(values))),
                        "coefficient": coefficient_index + 1,
                        **summary,
                    }
                )
    return rows


def build_point_rows(
    frozen: FrozenInputs,
    cache: Mapping[str, np.ndarray],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    samples = cache["point_samples"]
    for model_index, model_name in enumerate(frozen.model_names):
        for method_index, method in enumerate(frozen.method_names):
            for point_index, point in enumerate(BETA_POINTS):
                values = samples[model_index, :, method_index, point_index]
                summary = _distribution_summary(
                    values,
                    float(cache["true_points"][model_index, point_index]),
                )
                rows.append(
                    {
                        "model": model_name,
                        "method": method,
                        "distribution_scope": (
                            "archived numerically stable raw fits"
                            if method == "Raw FPLS"
                            else "all replications"
                        ),
                        "replications": int(np.count_nonzero(np.isfinite(values))),
                        "s": point,
                        **summary,
                    }
                )
    return rows


def build_tail_rows(frozen: FrozenInputs) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for model_index, model_name in enumerate(frozen.model_names):
        for method_index, method in enumerate(frozen.method_names):
            for metric_name, metric in (("ISE", frozen.ise), ("MSPE", frozen.mspe)):
                values = metric[model_index, :, method_index]
                q1, median, q3 = np.quantile(values, (0.25, 0.50, 0.75))
                upper_whisker = q3 + 1.5 * (q3 - q1)
                rows.append(
                    {
                        "model": model_name,
                        "method": method,
                        "metric": metric_name,
                        "mean": float(np.mean(values)),
                        "q1": float(q1),
                        "median": float(median),
                        "q3": float(q3),
                        "p90": float(np.quantile(values, 0.90)),
                        "p99": float(np.quantile(values, 0.99)),
                        "maximum": float(np.max(values)),
                        "upper_whisker_1.5_iqr": float(upper_whisker),
                        "upper_outlier_count": int(np.count_nonzero(values > upper_whisker)),
                        "upper_outlier_rate": float(np.mean(values > upper_whisker)),
                        "numerical_instability_count": int(
                            np.count_nonzero(
                                frozen.method_instability[
                                    model_index, :, method_index
                                ]
                            )
                        ),
                        "numerical_instability_rate": float(
                            np.mean(
                                frozen.method_instability[
                                    model_index, :, method_index
                                ]
                            )
                        ),
                    }
                )
    return rows


def _robust_limits(data: Sequence[np.ndarray], lower: float = 0.005, upper: float = 0.995) -> Tuple[float, float]:
    combined = np.concatenate(
        [np.asarray(values, dtype=float)[np.isfinite(values)] for values in data]
    )
    lo, hi = np.quantile(combined, (lower, upper))
    if not np.isfinite(lo) or not np.isfinite(hi):
        raise ValueError("nonfinite robust limits")
    if hi <= lo:
        margin = max(abs(float(lo)) * 0.05, 1e-6)
        return float(lo - margin), float(hi + margin)
    margin = 0.04 * (hi - lo)
    return float(lo - margin), float(hi + margin)


def _hist_panel(
    ax: plt.Axes,
    values: np.ndarray,
    true_value: float,
    color: str,
    xlim: Tuple[float, float],
    bins: int,
) -> None:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    inside = finite[(finite >= xlim[0]) & (finite <= xlim[1])]
    outside = len(finite) - len(inside)
    ax.hist(
        inside,
        bins=bins,
        density=True,
        color=color,
        edgecolor="white",
        linewidth=0.35,
        alpha=0.78,
    )
    mean_value = float(np.mean(finite))
    variance = float(np.var(finite, ddof=0))
    ax.axvline(true_value, color="#B2182B", linewidth=1.6, label="True")
    if xlim[0] <= mean_value <= xlim[1]:
        ax.axvline(
            mean_value,
            color="#1B7837",
            linestyle="--",
            linewidth=1.4,
            label="Mean",
        )
    else:
        side = "left" if mean_value < xlim[0] else "right"
        ax.text(
            0.04 if side == "left" else 0.96,
            0.94,
            f"mean outside {side}",
            transform=ax.transAxes,
            ha="left" if side == "left" else "right",
            va="top",
            fontsize=6.8,
            color="#1B7837",
        )
    annotation = (
        f"bias={mean_value - true_value:.3g}\n"
        f"var={variance:.3g}"
    )
    if outside:
        annotation += f"\n{outside} outside display"
    ax.text(
        0.98,
        0.94,
        annotation,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.8,
        bbox={
            "boxstyle": "round,pad=0.18",
            "facecolor": "white",
            "edgecolor": "#CCCCCC",
            "alpha": 0.84,
        },
    )
    ax.set_xlim(*xlim)
    ax.grid(True, axis="y", alpha=0.18)


def plot_coefficient_histograms(
    frozen: FrozenInputs,
    cache: Mapping[str, np.ndarray],
    output_dir: Path,
    bins: int,
) -> None:
    method_indices = list(range(len(frozen.method_names)))
    samples = cache["coefficient_samples"]
    figures: List[plt.Figure] = []
    for model_index, model_name in enumerate(frozen.model_names):
        fig, axes = plt.subplots(
            len(COEFFICIENT_INDICES),
            len(method_indices),
            figsize=(18.5, 10.2),
            squeeze=False,
        )
        for local_index, coefficient_index in enumerate(COEFFICIENT_INDICES):
            row_data = [
                samples[model_index, :, method_index, local_index]
                for method_index in method_indices
            ]
            xlim = _robust_limits(row_data)
            true_value = float(
                cache["true_coefficients"][model_index, coefficient_index]
            )
            for column, method_index in enumerate(method_indices):
                method = frozen.method_names[method_index]
                ax = axes[local_index, column]
                _hist_panel(
                    ax,
                    row_data[column],
                    true_value,
                    METHOD_COLORS[method],
                    xlim,
                    bins,
                )
                if local_index == 0:
                    ax.set_title(SHORT_LABELS[method], fontweight="bold")
                if column == 0:
                    ax.set_ylabel(
                        rf"Density of $\widehat{{b}}_{{{coefficient_index + 1}}}$"
                    )
                if local_index == len(COEFFICIENT_INDICES) - 1:
                    ax.set_xlabel(rf"$\widehat{{b}}_{{{coefficient_index + 1}}}$")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=2,
            frameon=False,
            bbox_to_anchor=(0.5, 0.965),
        )
        fig.suptitle(
            f"{model_name}: distributions of the first three cosine coefficients",
            fontsize=15,
            fontweight="bold",
            y=0.995,
        )
        fig.text(
            0.5,
            0.01,
            (
                "Axes show the central 99% pooled range; omitted numerical tails are counted. "
                "Raw-FPLS panels use archived fits not flagged as numerically unstable."
            ),
            ha="center",
            fontsize=8.5,
        )
        fig.tight_layout(rect=(0.02, 0.035, 0.99, 0.94))
        figures.append(fig)
    _save_multipage(
        output_dir / "basis_coefficient_histograms.pdf", figures
    )


def plot_beta_point_histograms(
    frozen: FrozenInputs,
    cache: Mapping[str, np.ndarray],
    output_dir: Path,
    bins: int,
) -> None:
    method_indices = list(range(len(frozen.method_names)))
    samples = cache["point_samples"]
    figures: List[plt.Figure] = []
    for model_index, model_name in enumerate(frozen.model_names):
        fig, axes = plt.subplots(
            len(BETA_POINTS),
            len(method_indices),
            figsize=(18.5, 15.8),
            squeeze=False,
        )
        for point_index, point in enumerate(BETA_POINTS):
            row_data = [
                samples[model_index, :, method_index, point_index]
                for method_index in method_indices
            ]
            xlim = _robust_limits(row_data)
            true_value = float(cache["true_points"][model_index, point_index])
            for column, method_index in enumerate(method_indices):
                method = frozen.method_names[method_index]
                ax = axes[point_index, column]
                _hist_panel(
                    ax,
                    row_data[column],
                    true_value,
                    METHOD_COLORS[method],
                    xlim,
                    bins,
                )
                if point_index == 0:
                    ax.set_title(SHORT_LABELS[method], fontweight="bold")
                if column == 0:
                    ax.set_ylabel(rf"Density at $s={point:.2f}$")
                if point_index == len(BETA_POINTS) - 1:
                    ax.set_xlabel(rf"$\widehat{{\beta}}({point:.2f})$")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=2,
            frameon=False,
            bbox_to_anchor=(0.5, 0.973),
        )
        fig.suptitle(
            f"{model_name}: pointwise distributions of the selected slope estimate",
            fontsize=15,
            fontweight="bold",
            y=0.995,
        )
        fig.text(
            0.5,
            0.008,
            (
                "Axes show the central 99% pooled range; omitted numerical tails are counted. "
                "Raw-FPLS panels use archived fits not flagged as numerically unstable."
            ),
            ha="center",
            fontsize=8.5,
        )
        fig.tight_layout(rect=(0.02, 0.025, 0.99, 0.955))
        figures.append(fig)
    _save_multipage(
        output_dir / "beta_point_histograms.pdf", figures
    )


def _rows_for_subset(
    rows: Sequence[Mapping[str, object]], subset: str
) -> Dict[Tuple[str, str], Mapping[str, object]]:
    return {
        (str(row["model"]), str(row["method"])): row
        for row in rows
        if row["subset"] == subset
    }


def plot_bias_variance(
    frozen: FrozenInputs,
    rows: Sequence[Mapping[str, object]],
    subset: str,
    filename: str,
    output_dir: Path,
) -> None:
    lookup = _rows_for_subset(rows, subset)
    methods = list(frozen.method_names)
    fig, axes = plt.subplots(2, 3, figsize=(18.2, 9.3), squeeze=False)
    x = np.arange(len(methods), dtype=float)
    width = 0.22
    floor = 1e-12

    for model_index, model_name in enumerate(frozen.model_names):
        model_rows = [lookup[(model_name, method)] for method in methods]
        slope_bias = np.asarray(
            [float(row["slope_squared_bias"]) for row in model_rows]
        )
        slope_variance = np.asarray(
            [float(row["slope_variance"]) for row in model_rows]
        )
        slope_total = np.asarray(
            [
                float(row["slope_mse"])
                if bool(row["decomposition_available"])
                else float(row["archived_mean_ise"])
                for row in model_rows
            ]
        )
        ax = axes[0, model_index]
        ax.bar(
            x - width / 2,
            np.maximum(slope_bias, floor),
            width,
            color=BIAS_COLOR,
            label="Squared bias" if model_index == 0 else None,
        )
        ax.bar(
            x + width / 2,
            np.maximum(slope_variance, floor),
            width,
            color=VARIANCE_COLOR,
            label="Variance" if model_index == 0 else None,
        )
        ax.scatter(
            x,
            np.maximum(slope_total, floor),
            marker="D",
            s=24,
            color="#111111",
            label="Total" if model_index == 0 else None,
            zorder=4,
        )
        ax.set_yscale("log")
        if subset == "all replications":
            model_title = model_name
        else:
            model_title = (
                f"{model_name}\n"
                f"N={int(model_rows[0]['decomposition_replications']):,} paired replications"
            )
        ax.set_title(model_title, fontweight="bold")
        ax.set_ylabel("Slope risk" if model_index == 0 else "")
        ax.grid(True, axis="y", which="both", alpha=0.22)

        prediction_bias = np.asarray(
            [float(row["prediction_squared_bias"]) for row in model_rows]
        )
        prediction_variance = np.asarray(
            [float(row["prediction_variance"]) for row in model_rows]
        )
        noise = np.asarray(
            [float(row["irreducible_noise_variance"]) for row in model_rows]
        )
        prediction_total = np.asarray(
            [
                float(row["expected_test_mspe"])
                if bool(row["decomposition_available"])
                else float(row["archived_mean_test_mspe"])
                for row in model_rows
            ]
        )
        ax = axes[1, model_index]
        ax.bar(
            x - width,
            np.maximum(prediction_bias, floor),
            width,
            color=BIAS_COLOR,
            label="Squared bias" if model_index == 0 else None,
        )
        ax.bar(
            x,
            np.maximum(prediction_variance, floor),
            width,
            color=VARIANCE_COLOR,
            label="Variance" if model_index == 0 else None,
        )
        ax.bar(
            x + width,
            np.maximum(noise, floor),
            width,
            color=NOISE_COLOR,
            label="Noise" if model_index == 0 else None,
        )
        ax.scatter(
            x,
            np.maximum(prediction_total, floor),
            marker="D",
            s=24,
            color="#111111",
            label="Total" if model_index == 0 else None,
            zorder=4,
        )
        ax.set_yscale("log")
        ax.set_ylabel("Prediction risk" if model_index == 0 else "")
        ax.grid(True, axis="y", which="both", alpha=0.22)
        if subset == "all replications":
            raw_position = methods.index("Raw FPLS")
            for row_index in range(2):
                axes[row_index, model_index].text(
                    raw_position,
                    0.035,
                    "archived total\nonly",
                    transform=axes[row_index, model_index].get_xaxis_transform(),
                    ha="center",
                    va="bottom",
                    fontsize=6.4,
                    color="#7A1F1F",
                )
        for row_index in range(2):
            axes[row_index, model_index].set_xticks(x)
            axes[row_index, model_index].set_xticklabels(
                [SHORT_LABELS[method] for method in methods],
                rotation=30,
                ha="right",
                fontsize=8,
            )

    handles_top, labels_top = axes[0, 0].get_legend_handles_labels()
    handles_bottom, labels_bottom = axes[1, 0].get_legend_handles_labels()
    unique: Dict[str, object] = {}
    for handle, label in zip(handles_top + handles_bottom, labels_top + labels_bottom):
        unique.setdefault(label, handle)
    title_suffix = (
        f"all {frozen.selected_components.shape[1]:,} replications"
        if subset == "all replications"
        else "paired subset with numerically stable Raw FPLS"
    )
    fig.suptitle(
        f"Bias-variance decomposition: {title_suffix}",
        fontsize=15,
        fontweight="bold",
        y=0.995,
    )
    fig.text(
        0.5,
        0.012,
        (
            "Bars show components on a logarithmic scale; black diamonds show their arithmetic sum. "
            "Full Raw-FPLS totals come from the frozen archive because unstable slope directions are backend-dependent."
            if subset == "all replications"
            else "Bars show components on a logarithmic scale; black diamonds show their arithmetic sum."
        ),
        ha="center",
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0.02, 0.12, 0.99, 0.91))
    legend = fig.legend(
        unique.values(),
        unique.keys(),
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.045),
    )
    legend.set_zorder(1000)
    _save_figure(fig, output_dir / filename)


def plot_coefficient_qq(
    frozen: FrozenInputs,
    cache: Mapping[str, np.ndarray],
    output_dir: Path,
) -> None:
    method_indices = list(range(len(frozen.method_names)))
    samples = cache["coefficient_samples"]
    figures: List[plt.Figure] = []
    for model_index, model_name in enumerate(frozen.model_names):
        fig, axes = plt.subplots(
            len(COEFFICIENT_INDICES),
            len(method_indices),
            figsize=(17.7, 10.4),
            squeeze=False,
        )
        for row_index, coefficient_index in enumerate(COEFFICIENT_INDICES):
            for column, method_index in enumerate(method_indices):
                method = frozen.method_names[method_index]
                values = samples[model_index, :, method_index, row_index]
                values = np.sort(values[np.isfinite(values)])
                mean = float(np.mean(values))
                sd = float(np.std(values, ddof=0))
                if sd <= 0.0 or not np.isfinite(sd):
                    raise FloatingPointError("Q-Q standard deviation is invalid")
                standardized = (values - mean) / sd
                probabilities = (
                    np.arange(1, len(values) + 1, dtype=float) - 0.5
                ) / len(values)
                theoretical = stats.norm.ppf(probabilities)
                ax = axes[row_index, column]
                ax.scatter(
                    theoretical,
                    standardized,
                    s=4,
                    alpha=0.38,
                    color=METHOD_COLORS[method],
                    linewidths=0,
                    rasterized=True,
                )
                lower = min(float(theoretical[0]), float(standardized[0]))
                upper = max(float(theoretical[-1]), float(standardized[-1]))
                ax.plot(
                    [lower, upper],
                    [lower, upper],
                    color="#444444",
                    linestyle="--",
                    linewidth=1.0,
                )
                ax.set_xlim(lower, upper)
                ax.set_ylim(lower, upper)
                ax.grid(True, alpha=0.18)
                if row_index == 0:
                    ax.set_title(SHORT_LABELS[method], fontweight="bold")
                if column == 0:
                    ax.set_ylabel(
                        rf"Observed $\widehat{{b}}_{{{coefficient_index + 1}}}$ quantiles"
                    )
                if row_index == len(COEFFICIENT_INDICES) - 1:
                    ax.set_xlabel("Standard normal quantiles")
                ax.text(
                    0.04,
                    0.94,
                    f"skew={stats.skew(values, bias=False):.2g}\n"
                    f"ex.kurt={stats.kurtosis(values, fisher=True, bias=False):.2g}",
                    transform=ax.transAxes,
                    va="top",
                    fontsize=6.8,
                    bbox={
                        "boxstyle": "round,pad=0.16",
                        "facecolor": "white",
                        "edgecolor": "#CCCCCC",
                        "alpha": 0.84,
                    },
                )
        fig.suptitle(
            f"{model_name}: normal Q-Q plots for the first three cosine coefficients",
            fontsize=15,
            fontweight="bold",
            y=0.995,
        )
        fig.text(
            0.5,
            0.008,
            "Raw-FPLS Q-Q panels use archived fits not flagged as numerically unstable.",
            ha="center",
            fontsize=8.3,
        )
        fig.tight_layout(rect=(0.02, 0.025, 0.99, 0.965))
        figures.append(fig)
    _save_multipage(
        output_dir / "coefficient_normal_qq.pdf", figures
    )


def plot_babii_style_boxplots(
    frozen: FrozenInputs,
    output_dir: Path,
) -> None:
    method_indices = list(range(len(frozen.method_names)))
    methods = [frozen.method_names[index] for index in method_indices]
    raw_index = frozen.method_names.index("Raw FPLS")
    raw_display_index = method_indices.index(raw_index) + 1
    fig, axes = plt.subplots(2, 3, figsize=(18.0, 9.1), squeeze=False)

    for model_index, model_name in enumerate(frozen.model_names):
        for row_index, (metric, metric_label) in enumerate(
            ((frozen.ise, "Integrated squared error"), (frozen.mspe, "Test MSPE"))
        ):
            ax = axes[row_index, model_index]
            data = [metric[model_index, :, index] for index in method_indices]
            artists = ax.boxplot(
                data,
                showfliers=True,
                whis=1.5,
                patch_artist=True,
                flierprops={
                    "marker": ".",
                    "markersize": 2.0,
                    "markerfacecolor": "#555555",
                    "markeredgecolor": "#555555",
                    "alpha": 0.30,
                },
                medianprops={"color": "#111111", "linewidth": 1.1},
            )
            for patch, method in zip(artists["boxes"], methods):
                patch.set_facecolor(METHOD_COLORS[method])
                patch.set_alpha(0.72)

            unstable = frozen.method_instability[model_index, :, raw_index]
            raw_values = metric[model_index, :, raw_index]
            unstable_indices = np.flatnonzero(unstable)
            if len(unstable_indices):
                ax.scatter(
                    np.full(len(unstable_indices), raw_display_index),
                    raw_values[unstable_indices],
                    marker="D",
                    s=20,
                    facecolors="none",
                    edgecolors="#B2182B",
                    linewidths=0.75,
                    zorder=4,
                    label="Raw instability flag" if model_index == 0 and row_index == 0 else None,
                )
            max_index = int(np.argmax(raw_values))
            ax.text(
                0.03,
                0.97,
                f"raw unstable: {len(unstable_indices)}/{len(raw_values)}\n"
                f"raw max: {raw_values[max_index]:.3g} (rep {max_index + 1})",
                transform=ax.transAxes,
                va="top",
                fontsize=7.0,
                bbox={
                    "boxstyle": "round,pad=0.20",
                    "facecolor": "white",
                    "edgecolor": "#BBBBBB",
                    "alpha": 0.88,
                },
            )
            ax.set_yscale("log")
            ax.set_xticks(np.arange(1, len(methods) + 1))
            ax.set_xticklabels(
                [SHORT_LABELS[method] for method in methods],
                rotation=30,
                ha="right",
                fontsize=8,
            )
            ax.grid(True, axis="y", which="both", alpha=0.23)
            if row_index == 0:
                ax.set_title(model_name, fontweight="bold")
            if model_index == 0:
                ax.set_ylabel(metric_label)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper right",
            frameon=False,
            bbox_to_anchor=(0.985, 0.985),
        )
    fig.suptitle(
        "Estimation and prediction accuracy with complete numerical tails",
        fontsize=15,
        fontweight="bold",
        y=0.995,
    )
    fig.text(
        0.5,
        0.012,
        "Box-and-whisker summaries use 1.5 IQR whiskers; all outliers are shown on logarithmic axes.",
        ha="center",
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0.02, 0.035, 0.99, 0.965))
    _save_figure(
        fig, output_dir / "babii_style_error_boxplots_with_tails.pdf"
    )


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlepad": 7.0,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create thesis figures from the four-method simulation."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("fpls_four_method_results"),
        help="directory containing raw_results.npz and configuration.json",
    )
    parser.add_argument(
        "--raw-results",
        type=Path,
        help="optional explicit raw_results.npz path",
    )
    parser.add_argument(
        "--configuration",
        type=Path,
        help="optional explicit configuration.json path",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("fpls_four_method_graphs"),
    )
    parser.add_argument(
        "--replications",
        type=int,
        help="use the first R archived replications; default uses all",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="independent model replay workers (1 to 3)",
    )
    parser.add_argument(
        "--bins", type=int, default=30, help="histogram bins"
    )
    parser.add_argument(
        "--force-replay",
        action="store_true",
        help="ignore an existing replay cache",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 1 <= args.jobs <= 3:
        raise ValueError("jobs must be between 1 and 3")
    if args.bins < 10:
        raise ValueError("bins must be at least 10")
    raw_path, config_path = _resolve_inputs(args)
    frozen = load_frozen_inputs(raw_path, config_path, args.replications)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / "selected_beta_diagnostics_cache.npz"

    print("=" * 72)
    print("THESIS FIGURES FOR THE FOUR-METHOD SIMULATION")
    print("=" * 72)
    print(f"Raw archive: {raw_path}")
    print(f"Configuration: {config_path}")
    print(f"Replications/model: {frozen.selected_components.shape[1]}")

    if cache_path.is_file() and not args.force_replay:
        print(f"Loading verified replay cache: {cache_path}")
        cache = _load_cache(cache_path, frozen)
    else:
        print(
            "Reconstructing selected slope estimates from archived component counts..."
        )
        cache = replay_selected_estimates(frozen, args.jobs)
        _save_cache(cache_path, cache)
        print(f"Saved replay cache: {cache_path}")

    validation = _validate_replay(cache, frozen)
    with (output_dir / "replay_validation.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(validation, handle, indent=2)
        handle.write("\n")
    print(
        "Replay validation passed: "
        f"max relative ISE error={validation['max_relative_ise_error']:.3g}, "
        f"MSPE error={validation['max_relative_mspe_error']:.3g}"
    )

    bias_variance_rows = build_bias_variance_rows(frozen, cache)
    coefficient_rows = build_coefficient_rows(frozen, cache)
    point_rows = build_point_rows(frozen, cache)
    tail_rows = build_tail_rows(frozen)
    _write_csv(output_dir / "bias_variance_decomposition.csv", bias_variance_rows)
    _write_csv(output_dir / "coefficient_distribution_summary.csv", coefficient_rows)
    _write_csv(output_dir / "beta_point_distribution_summary.csv", point_rows)
    _write_csv(output_dir / "tail_summary.csv", tail_rows)

    _apply_style()
    plot_coefficient_histograms(frozen, cache, output_dir, args.bins)
    plot_beta_point_histograms(frozen, cache, output_dir, args.bins)
    plot_bias_variance(
        frozen,
        bias_variance_rows,
        "all replications",
        "bias_variance_decomposition.pdf",
        output_dir,
    )
    plot_bias_variance(
        frozen,
        bias_variance_rows,
        "paired raw-stable subset",
        "bias_variance_decomposition_raw_stable_subset.pdf",
        output_dir,
    )
    plot_coefficient_qq(frozen, cache, output_dir)
    plot_babii_style_boxplots(frozen, output_dir)

    print("\nCreated:")
    for path in sorted(output_dir.iterdir()):
        if path.is_file():
            print(f"  {path.name}")


if __name__ == "__main__":
    main()
