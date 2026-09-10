#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHODS = ["CG-FPLS-code", "Raw FPLS", "Arnoldi FPLS", "FPCR"]
COLOURS = {
    "CG-FPLS-code": "#356A9A",
    "Raw FPLS": "#B55252",
    "Arnoldi FPLS": "#3D8460",
    "FPCR": "#8065A8",
}
DISPLAY = {
    "CG-FPLS-code": "CG-FPLS",
    "Raw FPLS": "Raw FPLS",
    "Arnoldi FPLS": "Arnoldi FPLS",
    "FPCR": "FPCR",
}


def configure_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11.5,
            "axes.titlesize": 12.2,
            "axes.labelsize": 11.5,
            "legend.fontsize": 9.6,
            "figure.titlesize": 13.0,
            "xtick.labelsize": 10.2,
            "ytick.labelsize": 10.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save(figure: plt.Figure, output: Path, preview: Path | None) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight")
    if preview is not None:
        preview.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(preview, dpi=220, bbox_inches="tight")
    plt.close(figure)


def load_study_core(source_dir: Path):
    sys.path.insert(0, str(source_dir))
    source = source_dir / "tablet_nir_core.py"
    spec = importlib.util.spec_from_file_location("tablet_nir_core", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["tablet_nir_core"] = module
    spec.loader.exec_module(module)
    return module


def spectra_overview(
    study_core,
    data_archive: Path,
    config: dict,
    output_dir: Path,
    preview_dir: Path | None,
) -> None:
    data = study_core.load_regression_data(
        data_archive,
        config["data"],
        instrument=1,
        splits=("calibrate", "validate", "test"),
    )
    development = np.isin(data.source_split, ("calibrate", "validate"))
    benchmark = data.source_split == "test"
    figure, axes = plt.subplots(2, 1, figsize=(6.5, 7.2))
    for mask, label, colour in (
        (development, "Development (n=195)", "#356A9A"),
        (benchmark, "Designated benchmark (n=460)", "#B55252"),
    ):
        values = data.X[mask]
        median = np.median(values, axis=0)
        lower, upper = np.quantile(values, (0.10, 0.90), axis=0)
        axes[0].plot(data.wavelengths, median, color=colour, linewidth=1.7, label=label)
        axes[0].fill_between(
            data.wavelengths, lower, upper, color=colour, alpha=0.16, linewidth=0
        )
    axes[0].axvline(1798.0, color="#555555", linestyle="--", linewidth=1.0)
    axes[0].annotate(
        "1798 nm sensitivity cutoff",
        xy=(1798.0, 0.96),
        xycoords=("data", "axes fraction"),
        xytext=(-5, 0),
        textcoords="offset points",
        ha="right",
        va="top",
        fontsize=9.2,
        color="#444444",
    )
    axes[0].set(
        xlabel="Wavelength (nm)",
        ylabel="Recorded spectral intensity",
        title="Instrument-1 spectra: median and 10th-90th percentile band",
    )
    axes[0].legend(frameon=False, loc="lower left")

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
            linewidth=1.7,
            color=colour,
            label=label,
        )
    axes[1].set(xlabel="Assay", ylabel="Density", title="Response distributions by supplied split")
    axes[1].legend(frameon=False)
    figure.tight_layout(h_pad=2.1)
    save(
        figure,
        output_dir / "tablet_spectra_assay_overview.pdf",
        None if preview_dir is None else preview_dir / "tablet_spectra_assay_overview.png",
    )


def observed_predicted(
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    primary: str,
    output_dir: Path,
    preview_dir: Path | None,
) -> None:
    frame = predictions[predictions["variant"] == primary]
    metric_frame = metrics[metrics["variant"] == primary].set_index("method")
    lower = min(frame["observed_assay"].min(), frame["prediction"].min()) - 2
    upper = max(frame["observed_assay"].max(), frame["prediction"].max()) + 2
    figure, axes = plt.subplots(2, 2, figsize=(6.8, 6.4), sharex=True, sharey=True)
    for axis, method in zip(axes.flat, METHODS, strict=True):
        subset = frame[frame["method"] == method]
        axis.scatter(
            subset["observed_assay"],
            subset["prediction"],
            s=14,
            alpha=0.55,
            color=COLOURS[method],
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
            fontsize=9.4,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78},
        )
        axis.set_title(DISPLAY[method])
        axis.set_xlim(lower, upper)
        axis.set_ylim(lower, upper)
    figure.supxlabel("Observed assay")
    figure.supylabel("Predicted assay")
    figure.suptitle("Designated benchmark: observed versus predicted assay")
    figure.tight_layout()
    save(
        figure,
        output_dir / "tablet_benchmark_observed_vs_predicted.pdf",
        None if preview_dir is None else preview_dir / "tablet_benchmark_observed_vs_predicted.png",
    )


def rmse_intervals(
    bootstrap: pd.DataFrame,
    primary: str,
    output_dir: Path,
    preview_dir: Path | None,
) -> None:
    frame = bootstrap[
        (bootstrap["variant"] == primary)
        & (bootstrap["scheme"] == "iid")
        & (bootstrap["result_type"] == "method_metric")
        & (bootstrap["metric"] == "rmse")
    ].copy()
    frame["order"] = frame["method_left"].map({method: i for i, method in enumerate(METHODS)})
    frame = frame.sort_values("order")
    figure, axis = plt.subplots(figsize=(6.4, 3.8))
    for position, (_, row) in enumerate(frame.iterrows()):
        method = row["method_left"]
        axis.errorbar(
            row["estimate"],
            position,
            xerr=np.array([[row["estimate"] - row["ci_lower"]], [row["ci_upper"] - row["estimate"]]]),
            fmt="o",
            markersize=6.5,
            capsize=3.5,
            color=COLOURS[method],
            linewidth=1.5,
        )
    axis.set_yticks(range(len(frame)), [DISPLAY[x] for x in frame["method_left"]])
    axis.invert_yaxis()
    axis.set_xlabel("Benchmark RMSE (marginal iid-bootstrap 95% interval)")
    axis.set_title("Primary full-spectrum instrument-1 analysis")
    figure.tight_layout()
    save(
        figure,
        output_dir / "tablet_benchmark_rmse_intervals.pdf",
        None if preview_dir is None else preview_dir / "tablet_benchmark_rmse_intervals.png",
    )


def selected_components(
    diagnostics: pd.DataFrame,
    primary: str,
    output_dir: Path,
    preview_dir: Path | None,
) -> None:
    frame = diagnostics[
        (diagnostics["analysis"] == "development_repeated_nested_cv")
        & (diagnostics["variant"] == primary)
        & (diagnostics["status"] == "ok")
    ]
    values = [
        frame.loc[frame["method"] == method, "selected_components"].dropna().to_numpy()
        for method in METHODS
    ]
    figure, axis = plt.subplots(figsize=(6.5, 4.4))
    box = axis.boxplot(
        values,
        tick_labels=[DISPLAY[x] for x in METHODS],
        patch_artist=True,
        widths=0.55,
        showfliers=False,
        medianprops={"color": "#222222", "linewidth": 1.4},
    )
    for patch, method in zip(box["boxes"], METHODS, strict=True):
        patch.set_facecolor(COLOURS[method])
        patch.set_alpha(0.35)
        patch.set_edgecolor(COLOURS[method])
    for position, (method, method_values) in enumerate(zip(METHODS, values, strict=True), start=1):
        jitter = np.linspace(-0.14, 0.14, len(method_values))
        axis.scatter(position + jitter, method_values, s=18, color=COLOURS[method], alpha=0.75, zorder=3)
    axis.set_ylabel("Selected component / stopping index")
    axis.set_title("Selection and stopping stability across 25 development outer fits")
    axis.tick_params(axis="x", rotation=10)
    figure.tight_layout()
    save(
        figure,
        output_dir / "tablet_development_selected_components.pdf",
        None if preview_dir is None else preview_dir / "tablet_development_selected_components.png",
    )


def residuals(
    predictions: pd.DataFrame,
    primary: str,
    output_dir: Path,
    preview_dir: Path | None,
) -> None:
    frame = predictions[predictions["variant"] == primary]
    column = "residual_prediction_minus_observation"
    limit = float(np.max(np.abs(frame[column]))) * 1.05
    figure, axes = plt.subplots(2, 2, figsize=(6.8, 6.1), sharex=True, sharey=True)
    for axis, method in zip(axes.flat, METHODS, strict=True):
        subset = frame[frame["method"] == method]
        axis.scatter(
            subset["observed_assay"],
            subset[column],
            s=14,
            alpha=0.55,
            color=COLOURS[method],
            edgecolors="none",
        )
        axis.axhline(0, color="#333333", linewidth=1.0)
        axis.set_title(DISPLAY[method])
        axis.set_ylim(-limit, limit)
    figure.supxlabel("Observed assay")
    figure.supylabel("Prediction minus observation")
    figure.suptitle("Designated-benchmark residual diagnostics")
    figure.tight_layout()
    save(
        figure,
        output_dir / "tablet_benchmark_residuals.pdf",
        None if preview_dir is None else preview_dir / "tablet_benchmark_residuals.png",
    )


def numerical_diagnostics(
    curves: pd.DataFrame,
    primary: str,
    output_dir: Path,
    preview_dir: Path | None,
) -> None:
    frame = curves[(curves["analysis"] == "designated_benchmark") & (curves["variant"] == primary)]
    raw = frame[frame["method"] == "Raw FPLS"]
    arnoldi = frame[frame["method"] == "Arnoldi FPLS"]
    threshold = 1 / np.sqrt(np.finfo(float).eps)
    figure, axes = plt.subplots(2, 1, figsize=(6.5, 7.0), sharex=True)
    axes[0].plot(raw["m"], raw["basis_condition_curve"], label="Raw basis", color="#B55252", linewidth=1.7)
    axes[0].plot(raw["m"], raw["design_condition_curve"], label="Raw response design", color="#D69B3E", linewidth=1.7)
    axes[0].axhline(threshold, color="#333333", linestyle="--", linewidth=1.0, label=r"$1/\sqrt{\epsilon_{\rm mach}}$")
    axes[0].axvline(55, color="#777777", linestyle=":", linewidth=1.0)
    axes[0].text(55, 1.8, "selected $m=55$", rotation=90, va="bottom", ha="right", fontsize=9.2)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Condition number")
    axes[0].set_title("Raw Krylov conditioning")
    axes[0].legend(frameon=False, ncol=3, loc="upper left")

    defect = arnoldi["orthogonality_defect_curve"].to_numpy(float)
    plotted = np.where(defect > 0, defect, np.nan)
    axes[1].plot(arnoldi["m"], plotted, color="#3D8460", linewidth=1.7, label=r"$\|Q_m^\top Q_m-I\|_2$")
    axes[1].axhline(np.sqrt(np.finfo(float).eps), color="#333333", linestyle="--", linewidth=1.0, label=r"$\sqrt{\epsilon_{\rm mach}}$")
    axes[1].axvline(5, color="#777777", linestyle=":", linewidth=1.0)
    axes[1].text(5, 2.2e-15, "selected $m=5$", rotation=90, va="bottom", ha="right", fontsize=9.2)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Requested component count")
    axes[1].set_ylabel("Orthogonality defect")
    axes[1].set_title("Arnoldi basis orthogonality")
    axes[1].legend(frameon=False, loc="upper left")
    figure.tight_layout(h_pad=2.0)
    save(
        figure,
        output_dir / "tablet_raw_arnoldi_numerical_diagnostics.pdf",
        None if preview_dir is None else preview_dir / "tablet_raw_arnoldi_numerical_diagnostics.png",
    )


def sensitivity(
    metrics: pd.DataFrame,
    config: dict,
    output_dir: Path,
    preview_dir: Path | None,
) -> None:
    variants = [entry["name"] for entry in config["variants"]]
    labels = {
        variants[0]: "Instrument 1\nfull (primary)",
        variants[1]: "Instrument 1\n600-1798 nm",
        variants[2]: "Instrument 1\nSG smoothing",
        variants[3]: "Instrument 2\nfull",
    }
    figure, axis = plt.subplots(figsize=(6.8, 4.8))
    x = np.arange(len(variants), dtype=float)
    offsets = np.linspace(-0.22, 0.22, len(METHODS))
    for offset, method in zip(offsets, METHODS, strict=True):
        subset = metrics[metrics["method"] == method].set_index("variant")
        axis.plot(
            x + offset,
            [float(subset.loc[variant, "rmse"]) for variant in variants],
            marker="o",
            linestyle="none",
            markersize=6.5,
            label=DISPLAY[method],
            color=COLOURS[method],
        )
    axis.set_xticks(x, [labels[variant] for variant in variants])
    axis.set_ylabel("Designated-benchmark RMSE")
    axis.set_title("Prespecified preprocessing and instrument sensitivities")
    axis.legend(frameon=False, ncol=2)
    figure.tight_layout()
    save(
        figure,
        output_dir / "tablet_benchmark_sensitivity_rmse.pdf",
        None if preview_dir is None else preview_dir / "tablet_benchmark_sensitivity_rmse.png",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--data-archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preview-dir", type=Path)
    args = parser.parse_args()

    results = args.results.resolve()
    config = json.loads((results / "provenance/configuration.json").read_text(encoding="utf-8"))
    primary = next(entry["name"] for entry in config["variants"] if entry["role"] == "primary")
    study_core = load_study_core(results / "provenance/source")
    predictions = pd.read_csv(results / "benchmark/predictions.csv")
    metrics = pd.read_csv(results / "benchmark/metrics.csv")
    bootstrap = pd.read_csv(results / "benchmark/bootstrap_summary.csv")
    diagnostics = pd.read_csv(results / "development/fit_diagnostics.csv")
    curves = pd.read_csv(results / "benchmark/tuning_and_numerical_curves.csv", low_memory=False)

    configure_style()
    spectra_overview(study_core, args.data_archive, config, args.output_dir, args.preview_dir)
    observed_predicted(predictions, metrics, primary, args.output_dir, args.preview_dir)
    rmse_intervals(bootstrap, primary, args.output_dir, args.preview_dir)
    selected_components(diagnostics, primary, args.output_dir, args.preview_dir)
    residuals(predictions, primary, args.output_dir, args.preview_dir)
    numerical_diagnostics(curves, primary, args.output_dir, args.preview_dir)
    sensitivity(metrics, config, args.output_dir, args.preview_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
