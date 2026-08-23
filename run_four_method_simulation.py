"""Run the clean four-method functional-regression simulation.

This program writes a frozen archive for exactly four estimators and exactly
three controlled simulation setups.  Figure creation is handled downstream by
the matched programs in the same folder.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

import four_method_core as core


def _finite_values(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=float)
    return array[np.isfinite(array)]


def _safe_mean(values: Iterable[float]) -> float:
    finite = _finite_values(values)
    return float(np.mean(finite)) if len(finite) else math.nan


def _safe_median(values: Iterable[float]) -> float:
    finite = _finite_values(values)
    return float(np.median(finite)) if len(finite) else math.nan


def _safe_quantile(values: Iterable[float], probability: float) -> float:
    finite = _finite_values(values)
    return float(np.quantile(finite, probability)) if len(finite) else math.nan


def _safe_max(values: Iterable[float]) -> float:
    finite = _finite_values(values)
    return float(np.max(finite)) if len(finite) else math.nan


def _standard_error(values: Iterable[float]) -> float:
    finite = _finite_values(values)
    if len(finite) < 2:
        return math.nan
    return float(np.std(finite, ddof=1) / np.sqrt(len(finite)))


def _extreme_tail_mask(values: np.ndarray, multiple: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    median = _safe_median(values)
    if not np.isfinite(median) or median <= 0.0:
        return np.zeros(values.shape, dtype=bool)
    return finite & (values > multiple * median)


def _write_csv_atomic(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    fieldnames: Optional[Sequence[str]] = None,
) -> None:
    if not rows and not fieldnames:
        raise ValueError(f"cannot infer CSV fields for {path}")
    fields = list(fieldnames) if fieldnames else list(rows[0].keys())
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json_atomic(path: Path, value: Mapping[str, object]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_npz_atomic(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_arguments(
    *,
    replications: int,
    sample_size: int,
    basis_size: int,
    grid_size: int,
    maximum_components: int,
    folds: int,
    noise_sd: float,
    tau: float,
    delta: float,
    rank_tolerance: float,
    extreme_tail_multiple: float,
) -> None:
    if replications < 2:
        raise ValueError("replications must be at least two")
    if sample_size < 3 or basis_size < 5 or grid_size < basis_size:
        raise ValueError("require n >= 3, J >= 5, and T >= J")
    if maximum_components < 1:
        raise ValueError("maximum_components must be positive")
    if not 2 <= folds <= sample_size:
        raise ValueError("folds must be between two and n")
    if not np.isfinite(noise_sd) or noise_sd < 0.0:
        raise ValueError("noise_sd must be finite and nonnegative")
    if tau <= 1.0 or not 0.0 < delta < 1.0:
        raise ValueError("tau must exceed one and delta must lie in (0,1)")
    if not np.isfinite(rank_tolerance) or rank_tolerance <= 0.0:
        raise ValueError("rank_tolerance must be positive and finite")
    if extreme_tail_multiple <= 1.0:
        raise ValueError("extreme_tail_multiple must exceed one")


def run_simulation(
    *,
    replications: int,
    sample_size: int,
    basis_size: int,
    grid_size: int,
    maximum_components: int,
    folds: int,
    noise_sd: float,
    tau: float,
    delta: float,
    seed: int,
    rank_tolerance: float,
    extreme_tail_multiple: float,
    output_dir: Path,
) -> Dict[str, object]:
    """Run and archive the complete four-method Monte Carlo experiment."""
    _validate_arguments(
        replications=replications,
        sample_size=sample_size,
        basis_size=basis_size,
        grid_size=grid_size,
        maximum_components=maximum_components,
        folds=folds,
        noise_sd=noise_sd,
        tau=tau,
        delta=delta,
        rank_tolerance=rank_tolerance,
        extreme_tail_multiple=extreme_tail_multiple,
    )
    output_dir = Path(output_dir)
    protected = (output_dir / "raw_results.npz", output_dir / "configuration.json")
    if any(path.exists() for path in protected):
        raise FileExistsError(
            f"refusing to overwrite an existing simulation archive in {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    grid = np.linspace(0.0, 1.0, grid_size)
    basis = core.create_cosine_basis(grid, basis_size)
    models = core.make_model_specs(basis_size, basis)
    methods = core.METHODS
    seed_sequences = np.random.SeedSequence(seed).spawn(len(models))

    shape = (len(models), replications, len(methods))
    selected_components = np.zeros(shape, dtype=int)
    ise = np.full(shape, np.nan)
    mspe = np.full(shape, np.nan)
    method_instability = np.zeros(shape, dtype=bool)
    replication_rows: List[Dict[str, object]] = []
    diagnostic_rows: List[Dict[str, object]] = []

    print("=" * 72)
    print("CLEAN FOUR-METHOD FUNCTIONAL-REGRESSION SIMULATION")
    print("=" * 72)
    print(
        f"Setups=3, replications/setup={replications}, n={sample_size}, "
        f"J={basis_size}, T={grid_size}, m_max={maximum_components}, folds={folds}"
    )
    print("Methods: " + ", ".join(methods))
    print("Arnoldi construction: fixed CGS2")
    print("Reported FPCR tuning: response-space GCV")

    for model_index, (model, seed_sequence) in enumerate(zip(models, seed_sequences)):
        rng = np.random.default_rng(seed_sequence)
        print(f"\n{model.name}")
        for replication in range(replications):
            if replication == 0 or (replication + 1) % max(1, replications // 10) == 0:
                print(f"  replication {replication + 1}/{replications}")

            X, y = core.generate_sample(rng, sample_size, basis, model, noise_sd)
            K, r = core.empirical_moments(X, y)

            fpcr_path = core.fpcr_spectral_path(
                r,
                K,
                min(maximum_components, sample_size - 1, grid_size - 1),
            )
            fpcr = core.select_fpcr_response_gcv(y, X, fpcr_path)
            cg_pilot = core.select_cg_variance_pilot(r, K, fpcr_path)
            cg_path = core.cg_fpls_path(r, K, maximum_components)
            cg = core.select_cg_fpls_code(
                cg_path,
                X,
                y,
                r,
                K,
                cg_pilot.beta,
                tau=tau,
                delta=delta,
            )

            fold_indices = core.make_folds(sample_size, folds, rng)
            fpls = core.cross_validate_fpls(
                y,
                X,
                K,
                r,
                maximum_components,
                fold_indices,
                rank_tolerance=rank_tolerance,
            )

            estimates = (
                cg.beta,
                fpls.beta_raw,
                fpls.beta_arnoldi,
                fpcr.beta,
            )
            dimensions = (
                cg.selected_components,
                fpls.m_raw,
                fpls.m_arnoldi,
                fpcr.selected_components,
            )

            X_test, y_test = core.generate_sample(
                rng,
                sample_size,
                basis,
                model,
                noise_sd,
            )
            raw_index = methods.index("Raw FPLS")
            arnoldi_index = methods.index("Arnoldi FPLS")
            raw_selected = fpls.m_raw - 1
            arnoldi_selected = fpls.m_arnoldi - 1
            raw_condition = max(
                float(fpls.raw_full_path.basis_condition[raw_selected]),
                float(fpls.raw_full_path.design_condition[raw_selected]),
            )
            arnoldi_defect = float(
                fpls.arnoldi_full_path.orthogonality_defect[arnoldi_selected]
            )

            for method_index, (method, beta_hat, dimension) in enumerate(
                zip(methods, estimates, dimensions)
            ):
                with np.errstate(over="ignore", invalid="ignore"):
                    ise_value = float(np.mean((beta_hat - model.beta) ** 2))
                    prediction = X_test @ beta_hat / grid_size
                    mspe_value = float(np.mean((y_test - prediction) ** 2))
                finite = bool(np.isfinite(ise_value) and np.isfinite(mspe_value))
                if method_index == raw_index:
                    unstable = (not finite) or raw_condition >= core.RAW_CONDITION_THRESHOLD
                    diagnostic = raw_condition
                elif method_index == arnoldi_index:
                    unstable = (not finite) or arnoldi_defect >= core.ARNOLDI_DEFECT_THRESHOLD
                    diagnostic = arnoldi_defect
                else:
                    unstable = not finite
                    diagnostic = math.nan

                selected_components[model_index, replication, method_index] = dimension
                ise[model_index, replication, method_index] = ise_value
                mspe[model_index, replication, method_index] = mspe_value
                method_instability[model_index, replication, method_index] = unstable
                replication_rows.append(
                    {
                        "model": model.name,
                        "replication": replication + 1,
                        "method": method,
                        "selected_components": dimension,
                        "ise": ise_value,
                        "mspe": mspe_value,
                        "finite_result": finite,
                        "numerical_instability": unstable,
                        "instability_diagnostic": diagnostic,
                    }
                )

            diagnostic_rows.append(
                {
                    "model": model.name,
                    "replication": replication + 1,
                    "raw_selected_components": fpls.m_raw,
                    "arnoldi_selected_components": fpls.m_arnoldi,
                    "raw_selected_condition": raw_condition,
                    "arnoldi_selected_orthogonality_defect": arnoldi_defect,
                    "cg_selected_components": cg.selected_components,
                    "cg_threshold_reached": cg.threshold_reached,
                    "cg_sigma2": cg.sigma2,
                    "cg_threshold": cg.threshold,
                    "cg_internal_pilot_components": cg_pilot.selected_components,
                    "fpcr_selected_components": fpcr.selected_components,
                }
            )

    summary_rows: List[Dict[str, object]] = []
    for model_index, model in enumerate(models):
        for method_index, method in enumerate(methods):
            ise_values = ise[model_index, :, method_index]
            mspe_values = mspe[model_index, :, method_index]
            components = selected_components[model_index, :, method_index]
            finite = np.isfinite(ise_values) & np.isfinite(mspe_values)
            instability = method_instability[model_index, :, method_index]
            extreme_ise = _extreme_tail_mask(ise_values, extreme_tail_multiple)
            extreme_mspe = _extreme_tail_mask(mspe_values, extreme_tail_multiple)
            summary_rows.append(
                {
                    "model": model.name,
                    "method": method,
                    "replications": replications,
                    "finite_replications": int(np.count_nonzero(finite)),
                    "failure_rate": float(1.0 - np.mean(finite)),
                    "mean_ise": _safe_mean(ise_values),
                    "se_ise": _standard_error(ise_values),
                    "median_ise": _safe_median(ise_values),
                    "p90_ise": _safe_quantile(ise_values, 0.90),
                    "p99_ise": _safe_quantile(ise_values, 0.99),
                    "max_ise": _safe_max(ise_values),
                    "mean_mspe": _safe_mean(mspe_values),
                    "se_mspe": _standard_error(mspe_values),
                    "median_mspe": _safe_median(mspe_values),
                    "p90_mspe": _safe_quantile(mspe_values, 0.90),
                    "p99_mspe": _safe_quantile(mspe_values, 0.99),
                    "max_mspe": _safe_max(mspe_values),
                    "mean_selected_components": float(np.mean(components)),
                    "median_selected_components": float(np.median(components)),
                    "numerical_instability_count": int(np.count_nonzero(instability)),
                    "numerical_instability_rate": float(np.mean(instability)),
                    "extreme_ise_count": int(np.count_nonzero(extreme_ise)),
                    "extreme_mspe_count": int(np.count_nonzero(extreme_mspe)),
                }
            )

    configuration: Dict[str, object] = {
        "study": "clean four-method functional-regression simulation",
        "replications_per_model": replications,
        "n": sample_size,
        "basis_size_J": basis_size,
        "grid_size_T": grid_size,
        "maximum_components": maximum_components,
        "cross_validation_folds": folds,
        "noise_sd": noise_sd,
        "seed": seed,
        "methods": list(methods),
        "models": list(core.MODEL_NAMES),
        "basis": "fixed Babii cosine convention",
        "raw_fpls": "unscaled Krylov powers and unregularized normal equations",
        "arnoldi_fpls": "CGS2 Arnoldi basis and response-space QR",
        "arnoldi_rank_tolerance": rank_tolerance,
        "fpcr_selection": "response-space GCV",
        "cg_method": "released-code stopping rule with m >= 1",
        "cg_tau": tau,
        "cg_delta": delta,
        "cg_variance_pilot": "internal moment-space FPCR plug-in estimate",
        "extreme_tail_multiple": extreme_tail_multiple,
        "raw_condition_threshold": core.RAW_CONDITION_THRESHOLD,
        "arnoldi_defect_threshold": core.ARNOLDI_DEFECT_THRESHOLD,
    }

    arrays = {
        "method_names": np.asarray(methods, dtype=np.str_),
        "model_names": np.asarray(core.MODEL_NAMES, dtype=np.str_),
        "selected_components": selected_components,
        "ise": ise,
        "mspe": mspe,
        "method_instability": method_instability,
    }
    _write_csv_atomic(output_dir / "replication_results.csv", replication_rows)
    _write_csv_atomic(output_dir / "numerical_diagnostics.csv", diagnostic_rows)
    _write_csv_atomic(output_dir / "summary.csv", summary_rows)
    _write_json_atomic(output_dir / "configuration.json", configuration)
    _write_npz_atomic(output_dir / "raw_results.npz", arrays)

    print("\nSummary")
    for row in summary_rows:
        print(
            f"  {row['model']} | {row['method']:<16} | "
            f"mean ISE={row['mean_ise']:.6g} | "
            f"median ISE={row['median_ise']:.6g} | "
            f"p99 ISE={row['p99_ise']:.6g} | "
            f"mean MSPE={row['mean_mspe']:.6g} | "
            f"instability={row['numerical_instability_rate']:.2%} | "
            f"mean m={row['mean_selected_components']:.2f}"
        )
    print(f"\nResults written to {output_dir.resolve()}")
    return {
        "configuration": configuration,
        "summary": summary_rows,
        "ise": ise,
        "mspe": mspe,
        "selected_components": selected_components,
        "method_instability": method_instability,
    }


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run exactly four estimators over the three simulation setups."
    )
    parser.add_argument("--replications", type=int, default=5000)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--basis-size", type=int, default=100)
    parser.add_argument("--grid-size", type=int, default=200)
    parser.add_argument("--m-max", type=int, default=70)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--noise-sd", type=float, default=1.0)
    parser.add_argument("--tau", type=float, default=1.01)
    parser.add_argument("--delta", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--rank-tolerance", type=float, default=1e-10)
    parser.add_argument("--extreme-tail-multiple", type=float, default=100.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("four_method_results"),
    )
    parser.add_argument("--self-test-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_arguments()
    core.run_self_tests()
    if args.self_test_only:
        return
    run_simulation(
        replications=args.replications,
        sample_size=args.n,
        basis_size=args.basis_size,
        grid_size=args.grid_size,
        maximum_components=args.m_max,
        folds=args.folds,
        noise_sd=args.noise_sd,
        tau=args.tau,
        delta=args.delta,
        seed=args.seed,
        rank_tolerance=args.rank_tolerance,
        extreme_tail_multiple=args.extreme_tail_multiple,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
