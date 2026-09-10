#!/usr/bin/env python3
"""Create the frozen tablet-NIR tables and thesis-ready scientific figures."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import tablet_nir_core as study_core


METHOD_ORDER = list(study_core.EXPECTED_METHODS)
METHOD_COLOURS = {
    "CG-FPLS-code": "#356A9A",
    "Raw FPLS": "#B55252",
    "Arnoldi FPLS": "#3D8460",
    "FPCR": "#8065A8",
}


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _save_figure(figure: plt.Figure, directory: Path, stem: str) -> None:
    figure.savefig(directory / f"{stem}.pdf", bbox_inches="tight")
    figure.savefig(directory / f"{stem}.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def _configure_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8.5,
            "figure.titlesize": 12,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _spectra_overview(
    archive_path: Path,
    config: Mapping[str, Any],
    figure_dir: Path,
) -> None:
    data = study_core.load_regression_data(
        archive_path,
        config["data"],
        instrument=int(config["data"]["primary_instrument"]),
        splits=("calibrate", "validate", "test"),
    )
    development = np.isin(data.source_split, ("calibrate", "validate"))
    benchmark = data.source_split == "test"
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))

    for mask, label, colour in (
        (development, "Development (n=195)", "#356A9A"),
        (benchmark, "Designated benchmark (n=460)", "#B55252"),
    ):
        values = data.X[mask]
        median = np.median(values, axis=0)
        lower, upper = np.quantile(values, (0.10, 0.90), axis=0)
        axes[0].plot(data.wavelengths, median, color=colour, linewidth=1.6, label=label)
        axes[0].fill_between(
            data.wavelengths, lower, upper, color=colour, alpha=0.16, linewidth=0
        )
    axes[0].axvline(1798.0, color="#555555", linestyle="--", linewidth=0.9)
    axes[0].set(
        xlabel="Wavelength (nm)",
        ylabel="Recorded spectral intensity",
        title="Instrument-1 spectra: median and 10–90% band",
    )
    axes[0].legend(frameon=False)

    bins = np.linspace(float(np.min(data.y)), float(np.max(data.y)), 25)
    for split, colour, label in (
        ("calibrate", "#356A9A", "Calibration (n=155)"),
        ("validate", "#D69B3E", "Validation (n=40)"),
        ("test", "#B55252", "Benchmark (n=460)"),
    ):
        axes[1].hist(
            data.y[data.source_split == split],
            bins=bins,
            density=True,
            histtype="step",
            linewidth=1.6,
            color=colour,
            label=label,
        )
    axes[1].set(
        xlabel="Assay",
        ylabel="Density",
        title="Response distributions by supplied split",
    )
    axes[1].legend(frameon=False)
    figure.tight_layout()
    _save_figure(figure, figure_dir, "spectra_assay_overview")


def _observed_vs_predicted(
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    primary_variant: str,
    figure_dir: Path,
) -> None:
    primary = predictions[predictions["variant"] == primary_variant]
    metric_frame = metrics[metrics["variant"] == primary_variant].set_index("method")
    observed_min = float(primary["observed_assay"].min())
    observed_max = float(primary["observed_assay"].max())
    predicted_min = float(primary["prediction"].min())
    predicted_max = float(primary["prediction"].max())
    lower = min(observed_min, predicted_min) - 2.0
    upper = max(observed_max, predicted_max) + 2.0
    figure, axes = plt.subplots(2, 2, figsize=(8.5, 8.0), sharex=True, sharey=True)
    for axis, method in zip(axes.flat, METHOD_ORDER, strict=True):
        subset = primary[primary["method"] == method]
        axis.scatter(
            subset["observed_assay"],
            subset["prediction"],
            s=15,
            alpha=0.55,
            color=METHOD_COLOURS[method],
            edgecolors="none",
        )
        axis.plot([lower, upper], [lower, upper], color="#333333", linewidth=1.0)
        row = metric_frame.loc[method]
        axis.text(
            0.04,
            0.95,
            f"RMSE = {row['rmse']:.2f}\n$R^2$ = {row['r2']:.3f}",
            transform=axis.transAxes,
            ha="left",
            va="top",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75},
        )
        axis.set_title(method)
        axis.set_xlim(lower, upper)
        axis.set_ylim(lower, upper)
    figure.supxlabel("Observed assay")
    figure.supylabel("Predicted assay")
    figure.suptitle("Designated benchmark: observed versus predicted assay")
    figure.tight_layout()
    _save_figure(figure, figure_dir, "benchmark_observed_vs_predicted")


def _rmse_intervals(
    bootstrap: pd.DataFrame,
    primary_variant: str,
    figure_dir: Path,
) -> None:
    frame = bootstrap[
        (bootstrap["variant"] == primary_variant)
        & (bootstrap["scheme"] == "iid")
        & (bootstrap["result_type"] == "method_metric")
        & (bootstrap["metric"] == "rmse")
    ].copy()
    frame["order"] = frame["method_left"].map({name: i for i, name in enumerate(METHOD_ORDER)})
    frame = frame.sort_values("order")
    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    y_positions = np.arange(len(frame))
    for position, (_, row) in zip(y_positions, frame.iterrows(), strict=True):
        method = str(row["method_left"])
        axis.errorbar(
            row["estimate"],
            position,
            xerr=np.array(
                [[row["estimate"] - row["ci_lower"]], [row["ci_upper"] - row["estimate"]]]
            ),
            fmt="o",
            markersize=6,
            capsize=3,
            color=METHOD_COLOURS[method],
            linewidth=1.4,
        )
    axis.set_yticks(y_positions, frame["method_left"])
    axis.invert_yaxis()
    axis.set_xlabel("Benchmark RMSE (paired iid-bootstrap 95% interval)")
    axis.set_title("Primary full-spectrum instrument-1 analysis")
    figure.tight_layout()
    _save_figure(figure, figure_dir, "benchmark_rmse_intervals")


def _selected_components(
    diagnostics: pd.DataFrame,
    primary_variant: str,
    figure_dir: Path,
) -> None:
    frame = diagnostics[
        (diagnostics["analysis"] == "development_repeated_nested_cv")
        & (diagnostics["variant"] == primary_variant)
        & (diagnostics["status"] == "ok")
    ]
    values = [
        frame.loc[frame["method"] == method, "selected_components"].dropna().to_numpy()
        for method in METHOD_ORDER
    ]
    figure, axis = plt.subplots(figsize=(7.8, 4.5))
    box = axis.boxplot(
        values,
        tick_labels=METHOD_ORDER,
        patch_artist=True,
        widths=0.55,
        showfliers=False,
        medianprops={"color": "#222222", "linewidth": 1.3},
    )
    for patch, method in zip(box["boxes"], METHOD_ORDER, strict=True):
        patch.set_facecolor(METHOD_COLOURS[method])
        patch.set_alpha(0.35)
        patch.set_edgecolor(METHOD_COLOURS[method])
    for position, (method, method_values) in enumerate(
        zip(METHOD_ORDER, values, strict=True), start=1
    ):
        jitter = np.linspace(-0.14, 0.14, len(method_values)) if len(method_values) else []
        axis.scatter(
            position + jitter,
            method_values,
            s=17,
            color=METHOD_COLOURS[method],
            alpha=0.75,
            zorder=3,
        )
    axis.set_ylabel("Selected component / stopping index")
    axis.set_title("Development nested-CV selection stability (25 outer fits)")
    axis.tick_params(axis="x", rotation=12)
    figure.tight_layout()
    _save_figure(figure, figure_dir, "development_selected_components")


def _residual_panels(
    predictions: pd.DataFrame,
    primary_variant: str,
    figure_dir: Path,
) -> None:
    primary = predictions[predictions["variant"] == primary_variant]
    residual_column = "residual_prediction_minus_observation"
    limit = float(np.nanmax(np.abs(primary[residual_column]))) * 1.05
    figure, axes = plt.subplots(2, 2, figsize=(8.5, 7.0), sharex=True, sharey=True)
    for axis, method in zip(axes.flat, METHOD_ORDER, strict=True):
        subset = primary[primary["method"] == method]
        axis.scatter(
            subset["observed_assay"],
            subset[residual_column],
            s=15,
            alpha=0.55,
            color=METHOD_COLOURS[method],
            edgecolors="none",
        )
        axis.axhline(0.0, color="#333333", linewidth=1.0)
        axis.set_title(method)
        axis.set_ylim(-limit, limit)
    figure.supxlabel("Observed assay")
    figure.supylabel("Prediction minus observation")
    figure.suptitle("Designated-benchmark residual diagnostics")
    figure.tight_layout()
    _save_figure(figure, figure_dir, "benchmark_residuals")


def _numerical_curves(
    curves: pd.DataFrame,
    primary_variant: str,
    figure_dir: Path,
) -> None:
    frame = curves[
        (curves["analysis"] == "designated_benchmark")
        & (curves["variant"] == primary_variant)
    ]
    raw = frame[frame["method"] == "Raw FPLS"]
    arnoldi = frame[frame["method"] == "Arnoldi FPLS"]
    figure, axes = plt.subplots(1, 2, figsize=(10.0, 3.9))
    axes[0].plot(
        raw["m"], raw["basis_condition_curve"], label="Raw basis", color="#B55252"
    )
    axes[0].plot(
        raw["m"], raw["design_condition_curve"], label="Raw response design", color="#D69B3E"
    )
    axes[0].axhline(
        1.0 / np.sqrt(np.finfo(float).eps),
        color="#333333",
        linestyle="--",
        linewidth=1.0,
        label=r"$1/\sqrt{\epsilon}$",
    )
    axes[0].set_yscale("log")
    axes[0].set(xlabel="Requested component count", ylabel="Condition number", title="Raw Krylov conditioning")
    axes[0].legend(frameon=False)

    axes[1].plot(
        arnoldi["m"],
        arnoldi["orthogonality_defect_curve"],
        color="#3D8460",
        label=r"$\|Q_m^\top Q_m-I\|_2$",
    )
    axes[1].axhline(
        np.sqrt(np.finfo(float).eps),
        color="#333333",
        linestyle="--",
        linewidth=1.0,
        label=r"$\sqrt{\epsilon}$",
    )
    axes[1].set_yscale("log")
    axes[1].set(xlabel="Requested component count", ylabel="Orthogonality defect", title="Arnoldi basis orthogonality")
    axes[1].legend(frameon=False)
    figure.tight_layout()
    _save_figure(figure, figure_dir, "raw_arnoldi_numerical_diagnostics")


def _sensitivity_figure(
    metrics: pd.DataFrame,
    config: Mapping[str, Any],
    figure_dir: Path,
) -> None:
    variant_order = [variant["name"] for variant in config["variants"]]
    labels = {
        variant_order[0]: "Instrument 1\nfull (primary)",
        variant_order[1]: "Instrument 1\n600–1798 nm",
        variant_order[2]: "Instrument 1\nSG smoothing",
        variant_order[3]: "Instrument 2\nfull",
    }
    figure, axis = plt.subplots(figsize=(9.0, 4.6))
    x = np.arange(len(variant_order), dtype=float)
    offsets = np.linspace(-0.24, 0.24, len(METHOD_ORDER))
    for offset, method in zip(offsets, METHOD_ORDER, strict=True):
        subset = metrics[metrics["method"] == method].set_index("variant")
        values = [float(subset.loc[variant, "rmse"]) for variant in variant_order]
        axis.plot(
            x + offset,
            values,
            marker="o",
            linestyle="none",
            markersize=6,
            label=method,
            color=METHOD_COLOURS[method],
        )
    axis.set_xticks(x, [labels[value] for value in variant_order])
    axis.set_ylabel("Designated-benchmark RMSE")
    axis.set_title("Prespecified preprocessing and instrument sensitivities")
    axis.legend(frameon=False, ncol=2)
    figure.tight_layout()
    _save_figure(figure, figure_dir, "benchmark_sensitivity_rmse")


def _write_tables(
    result_root: Path,
    config: Mapping[str, Any],
    metrics: pd.DataFrame,
    bootstrap: pd.DataFrame,
    development_metrics: pd.DataFrame,
    diagnostics: pd.DataFrame,
) -> dict[str, Any]:
    table_dir = result_root / "tables"
    table_dir.mkdir()
    primary = next(v["name"] for v in config["variants"] if v["role"] == "primary")
    iid_metrics = bootstrap[
        (bootstrap["variant"] == primary)
        & (bootstrap["scheme"] == "iid")
        & (bootstrap["result_type"] == "method_metric")
    ][["method_left", "metric", "estimate", "ci_lower", "ci_upper", "confidence"]]
    iid_metrics = iid_metrics.rename(columns={"method_left": "method"})
    _atomic_csv(table_dir / "primary_benchmark_metrics.csv", iid_metrics)

    contrasts = bootstrap[
        (bootstrap["variant"] == primary)
        & (bootstrap["result_type"] == "method_contrast")
        & (bootstrap["contrast_scope"].isin(("focal", "secondary")))
    ].copy()
    _atomic_csv(table_dir / "declared_benchmark_contrasts.csv", contrasts)

    repeated = development_metrics[
        development_metrics["analysis"] == "development_repeated_nested_cv"
    ].copy()
    _atomic_csv(table_dir / "development_repeat_metrics.csv", repeated)

    selected = diagnostics[
        diagnostics["analysis"] == "development_repeated_nested_cv"
    ].copy()
    stability_rows: list[dict[str, Any]] = []
    for method in METHOD_ORDER:
        subset = selected[selected["method"] == method]
        values = subset["selected_components"].dropna().to_numpy(dtype=float)
        stability_rows.append(
            {
                "method": method,
                "fits": int(len(subset)),
                "failures": int((subset["status"] != "ok").sum()),
                "selected_min": float(np.min(values)) if len(values) else math.nan,
                "selected_q25": float(np.quantile(values, 0.25)) if len(values) else math.nan,
                "selected_median": float(np.median(values)) if len(values) else math.nan,
                "selected_q75": float(np.quantile(values, 0.75)) if len(values) else math.nan,
                "selected_max": float(np.max(values)) if len(values) else math.nan,
                "selected_at_cap_rate": float(subset.get("selected_at_cap", pd.Series(False, index=subset.index)).fillna(False).mean()),
                "instability_rate": float(subset.get("instability_flag", pd.Series(False, index=subset.index)).fillna(False).mean()),
            }
        )
    stability = pd.DataFrame(stability_rows)
    _atomic_csv(table_dir / "development_selection_stability.csv", stability)

    original = development_metrics[
        development_metrics["analysis"] == "original_split_sensitivity"
    ].copy()
    _atomic_csv(table_dir / "original_split_sensitivity.csv", original)
    _atomic_csv(table_dir / "benchmark_sensitivity_metrics.csv", metrics)

    primary_rmse = metrics[
        (metrics["variant"] == primary) & (metrics["method"].isin(METHOD_ORDER))
    ].sort_values("rmse")
    best = primary_rmse.iloc[0]
    focal = contrasts[
        (contrasts["scheme"] == "iid")
        & (contrasts["contrast_scope"] == "focal")
        & (contrasts["metric"] == "rmse")
    ].iloc[0]
    summary = {
        "primary_variant": primary,
        "benchmark_tablets": int(config["data"]["expected_counts"]["test"]),
        "best_point_rmse_method": str(best["method"]),
        "best_point_rmse": float(best["rmse"]),
        "focal_contrast": str(focal["contrast_label"]),
        "focal_rmse_difference": float(focal["estimate"]),
        "focal_iid_ci_lower": float(focal["ci_lower"]),
        "focal_iid_ci_upper": float(focal["ci_upper"]),
    }
    _atomic_json(table_dir / "headline_results.json", summary)
    return summary


def make_figures(
    result_root: Path,
    archive_path: Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    result_root = result_root.resolve()
    figure_dir = result_root / "figures"
    if figure_dir.exists() or (result_root / "tables").exists():
        raise FileExistsError("figures/tables already exist and will not be overwritten")
    figure_dir.mkdir()
    _configure_style()

    predictions = pd.read_csv(result_root / "benchmark" / "predictions.csv")
    metrics = pd.read_csv(result_root / "benchmark" / "metrics.csv")
    bootstrap = pd.read_csv(result_root / "benchmark" / "bootstrap_summary.csv")
    diagnostics = pd.read_csv(result_root / "development" / "fit_diagnostics.csv")
    development_metrics = pd.read_csv(result_root / "development" / "repeat_metrics.csv")
    curves = pd.read_csv(result_root / "benchmark" / "tuning_and_numerical_curves.csv", low_memory=False)
    primary = next(v["name"] for v in config["variants"] if v["role"] == "primary")

    _spectra_overview(archive_path, config, figure_dir)
    _observed_vs_predicted(predictions, metrics, primary, figure_dir)
    _rmse_intervals(bootstrap, primary, figure_dir)
    _selected_components(diagnostics, primary, figure_dir)
    _residual_panels(predictions, primary, figure_dir)
    _numerical_curves(curves, primary, figure_dir)
    _sensitivity_figure(metrics, config, figure_dir)
    summary = _write_tables(
        result_root,
        config,
        metrics,
        bootstrap,
        development_metrics,
        diagnostics,
    )
    _atomic_json(
        figure_dir / "_SUCCESS.json",
        {
            "status": "passed",
            "figure_stems": [
                "spectra_assay_overview",
                "benchmark_observed_vs_predicted",
                "benchmark_rmse_intervals",
                "development_selected_components",
                "benchmark_residuals",
                "raw_arnoldi_numerical_diagnostics",
                "benchmark_sensitivity_rmse",
            ],
        },
    )
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config_path = args.config or args.results / "provenance" / "configuration.json"
    with config_path.open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    summary = make_figures(args.results, args.archive, config)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
