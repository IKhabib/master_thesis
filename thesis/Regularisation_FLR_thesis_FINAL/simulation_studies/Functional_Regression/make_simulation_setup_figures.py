

from __future__ import annotations

import argparse
import csv
import io
import os
import re
import tempfile
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

import four_method_core as core


MODEL_COLORS = {
    "Model 1": "#0072B2",
    "Model 2": "#D55E00",
    "Model 3": "#009E73",
}
MODEL_SUBTITLES = {
    "Model 1": "Baseline",
    "Model 2": "Modified slope",
    "Model 3": "Modified predictor spectrum",
}
MODEL_ROLES = {
    "Model 1": "Baseline slope and predictor spectrum",
    "Model 2": "Model 1 spectrum; slower-decaying slope coefficients",
    "Model 3": "Model 1 slope; inflated first five eigenvalues",
}
GAUSSIAN_90_QUANTILE = 1.6448536269514722


def _apply_style() -> None:
    """Use a restrained, color-blind-friendly publication style."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlepad": 7.0,
            "axes.grid": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "legend.frameon": False,
            "lines.linewidth": 1.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _pdf_bytes_are_complete(data: bytes) -> bool:
    """Perform a lightweight structural check before committing a PDF."""
    if len(data) < 100 or not data.startswith(b"%PDF-"):
        return False
    if b"%%EOF" not in data[-2048:]:
        return False
    matches = list(re.finditer(rb"startxref\s+(\d+)\s+%%EOF", data[-4096:]))
    if not matches:
        return False
    xref_offset = int(matches[-1].group(1))
    if not 0 <= xref_offset < len(data):
        return False
    prefix = data[xref_offset : xref_offset + 64]
    return bool(prefix.startswith(b"xref") or re.match(rb"\d+\s+\d+\s+obj", prefix))


def _write_pdf_bytes(path: Path, data: bytes) -> None:
    """Atomically write a complete PDF to ``path``."""
    if not _pdf_bytes_are_complete(data):
        raise OSError(f"incomplete PDF generated for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
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
        if not _pdf_bytes_are_complete(temporary.read_bytes()):
            raise OSError(f"incomplete PDF write for {path}")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _png_bytes_are_complete(data: bytes) -> bool:
    """Check the fixed PNG signature and terminal IEND chunk."""
    return (
        len(data) >= 100
        and data.startswith(b"\x89PNG\r\n\x1a\n")
        and data.endswith(b"\x00\x00\x00\x00IEND\xaeB`\x82")
    )


def _write_png_bytes(path: Path, data: bytes) -> None:
    """Atomically write a structurally complete PNG."""
    if not _png_bytes_are_complete(data):
        raise OSError(f"incomplete PNG generated for {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.stem}.",
        suffix=".png",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        if not _png_bytes_are_complete(temporary.read_bytes()):
            raise OSError(f"incomplete PNG write for {path}")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _save_figure(fig: plt.Figure, pdf_path: Path, also_png: bool) -> None:
    """Save one figure as a checked vector PDF and optional 300-dpi PNG."""
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        buffer = io.BytesIO()
        fig.savefig(buffer, format="pdf", bbox_inches="tight")
        _write_pdf_bytes(pdf_path, buffer.getvalue())
        if also_png:
            png_buffer = io.BytesIO()
            fig.savefig(
                png_buffer,
                format="png",
                dpi=300,
                bbox_inches="tight",
            )
            _write_png_bytes(pdf_path.with_suffix(".png"), png_buffer.getvalue())
    finally:
        plt.close(fig)


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    """Write a rectangular CSV table."""
    if not rows:
        raise ValueError("setup summary cannot be empty")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _recover_basis_coefficients(
    basis: np.ndarray,
    models: Sequence[core.ModelSpec],
) -> Dict[str, np.ndarray]:
    """Recover coefficients from the slopes built by the frozen DGP code."""
    design = basis.T
    rank = int(np.linalg.matrix_rank(design))
    if rank != design.shape[1]:
        raise ValueError(
            "the requested grid does not identify all basis coefficients; "
            "increase --grid-size"
        )
    recovered: Dict[str, np.ndarray] = {}
    for model in models:
        coefficients, _, _, _ = np.linalg.lstsq(design, model.beta, rcond=None)
        reconstruction = design @ coefficients
        scale = max(float(np.linalg.norm(model.beta)), np.finfo(float).tiny)
        relative_error = float(np.linalg.norm(reconstruction - model.beta) / scale)
        if relative_error > 1e-10:
            raise RuntimeError(
                f"could not recover {model.name} coefficients accurately: "
                f"relative error {relative_error:.3e}"
            )
        recovered[model.name] = coefficients
    return recovered


def _signal_contributions(
    basis: np.ndarray,
    model: core.ModelSpec,
) -> np.ndarray:
    """Return exact discrete contributions to Var(X @ beta / T)."""
    grid_size = basis.shape[1]
    beta_coordinates = basis @ model.beta / grid_size
    contributions = model.eigenvalues * beta_coordinates**2
    if not np.all(np.isfinite(contributions)) or float(np.sum(contributions)) <= 0.0:
        raise RuntimeError(f"invalid predictive signal for {model.name}")
    return contributions


def _components_for_share(contributions: np.ndarray, share: float) -> int:
    cumulative = np.cumsum(contributions) / np.sum(contributions)
    return int(np.searchsorted(cumulative, share, side="left") + 1)


def _validate_controlled_comparisons(
    models: Sequence[core.ModelSpec],
    coefficients: Mapping[str, np.ndarray],
) -> None:
    """Guard the two one-factor comparisons on which the figures rely."""
    if tuple(model.name for model in models) != ("Model 1", "Model 2", "Model 3"):
        raise ValueError("the core DGP did not return the expected three models")
    by_name = {model.name: model for model in models}
    if not np.allclose(
        by_name["Model 1"].eigenvalues,
        by_name["Model 2"].eigenvalues,
        rtol=0.0,
        atol=0.0,
    ):
        raise RuntimeError("Models 1 and 2 must have the same predictor spectrum")
    if not np.allclose(
        coefficients["Model 1"], coefficients["Model 3"], rtol=1e-11, atol=1e-12
    ):
        raise RuntimeError("Models 1 and 3 must have the same slope coefficients")
    if np.allclose(
        coefficients["Model 1"], coefficients["Model 2"], rtol=1e-11, atol=1e-12
    ):
        raise RuntimeError("Model 2 must differ from Model 1 in its slope")
    if np.allclose(
        by_name["Model 1"].eigenvalues,
        by_name["Model 3"].eigenvalues,
        rtol=1e-11,
        atol=1e-12,
    ):
        raise RuntimeError("Model 3 must differ from Model 1 in its spectrum")


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.06,
        label,
        transform=ax.transAxes,
        fontsize=10.5,
        fontweight="bold",
        va="top",
    )


def plot_true_slope_structure(
    grid: np.ndarray,
    models: Sequence[core.ModelSpec],
    coefficients: Mapping[str, np.ndarray],
    coefficient_limit: int,
    output_dir: Path,
    also_png: bool,
) -> None:
    """Plot each true slope and its leading generating coefficients."""
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(10.8, 6.2),
        sharex="row",
        sharey="row",
        gridspec_kw={"height_ratios": (1.15, 1.0)},
    )
    beta_min = min(float(np.min(model.beta)) for model in models)
    beta_max = max(float(np.max(model.beta)) for model in models)
    beta_padding = 0.06 * max(beta_max - beta_min, 1.0)
    indices = np.arange(1, coefficient_limit + 1)

    for column, model in enumerate(models):
        color = MODEL_COLORS[model.name]
        top = axes[0, column]
        bottom = axes[1, column]
        top.plot(grid, model.beta, color=color, linewidth=2.0)
        top.axhline(0.0, color="#777777", linewidth=0.7, zorder=0)
        top.set_ylim(beta_min - beta_padding, beta_max + beta_padding)
        top.set_title(f"{model.name}\n{MODEL_SUBTITLES[model.name]}", fontweight="bold")
        top.set_xlabel(r"Location $s$")
        if column == 0:
            top.set_ylabel(r"True slope $\beta(s)$")

        values = np.abs(coefficients[model.name][:coefficient_limit])
        bottom.semilogy(
            indices,
            values,
            color=color,
            marker="o",
            markersize=3.3,
            linewidth=1.5,
        )
        bottom.set_xlim(0.5, coefficient_limit + 0.5)
        bottom.set_xlabel(r"Component $j$")
        bottom.grid(axis="y", which="both", color="#E5E5E5", linewidth=0.6)
        if column == 0:
            bottom.set_ylabel(r"Coefficient magnitude $|b_j|$")

    _panel_label(axes[0, 0], "A")
    _panel_label(axes[1, 0], "B")
    fig.suptitle(
        "True slope structure across the three simulation models",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )
    fig.text(
        0.5,
        0.012,
        "Models 1 and 3 share the same slope; Model 2 replaces the first five coefficients by 4.",
        ha="center",
        fontsize=8.7,
    )
    fig.tight_layout(rect=(0.02, 0.04, 0.995, 0.96), h_pad=1.35, w_pad=1.15)
    _save_figure(fig, output_dir / "true_slope_structure.pdf", also_png)


def plot_covariance_spectrum(
    models: Sequence[core.ModelSpec],
    spectrum_limit: int,
    output_dir: Path,
    also_png: bool,
) -> None:
    """Compare eigenvalue decay and cumulative predictor variance."""
    model_1, _, model_3 = models
    indices = np.arange(1, len(model_1.eigenvalues) + 1)
    visible = indices[:spectrum_limit]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.25))

    axes[0].semilogy(
        visible,
        model_1.eigenvalues[:spectrum_limit],
        color=MODEL_COLORS["Model 1"],
        marker="o",
        markersize=3.4,
        label="Models 1 and 2",
    )
    axes[0].semilogy(
        visible,
        model_3.eigenvalues[:spectrum_limit],
        color=MODEL_COLORS["Model 3"],
        marker="s",
        markersize=3.2,
        label="Model 3",
    )
    axes[0].axvline(5.5, color="#999999", linestyle=":", linewidth=1.0)
    axes[0].set_xlabel(r"Component $j$")
    axes[0].set_ylabel(r"Eigenvalue $\lambda_j$ (log scale)")
    axes[0].set_title("Eigenvalue decay", fontweight="bold")
    axes[0].grid(axis="y", which="both", color="#E5E5E5", linewidth=0.6)
    axes[0].legend(loc="upper right")
    _panel_label(axes[0], "A")

    for model, label, marker in (
        (model_1, "Models 1 and 2", "o"),
        (model_3, "Model 3", "s"),
    ):
        cumulative = np.cumsum(model.eigenvalues) / np.sum(model.eigenvalues)
        axes[1].plot(
            indices,
            cumulative,
            color=MODEL_COLORS[model.name],
            marker=marker,
            markevery=max(1, len(indices) // 10),
            markersize=3.5,
            label=label,
        )
    axes[1].axvline(5.5, color="#999999", linestyle=":", linewidth=1.0)
    axes[1].set_xlim(1, len(indices))
    axes[1].set_ylim(0.0, 1.02)
    axes[1].set_xlabel(r"Number of components $m$")
    axes[1].set_ylabel("Cumulative predictor variance")
    axes[1].yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    axes[1].set_title("Variance explained", fontweight="bold")
    axes[1].grid(axis="y", color="#E5E5E5", linewidth=0.6)
    axes[1].legend(loc="lower right")
    _panel_label(axes[1], "B")

    fig.suptitle(
        "Predictor covariance structure",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )
    fig.text(
        0.5,
        0.012,
        "Model 3 sets the first five eigenvalues to 2; Models 1 and 2 use the same power-law spectrum.",
        ha="center",
        fontsize=8.7,
    )
    fig.tight_layout(rect=(0.025, 0.055, 0.995, 0.94), w_pad=2.2)
    _save_figure(fig, output_dir / "covariance_spectrum.pdf", also_png)


def plot_predictive_signal_allocation(
    basis: np.ndarray,
    models: Sequence[core.ModelSpec],
    signal_limit: int,
    output_dir: Path,
    also_png: bool,
) -> None:
    """Plot how basis components allocate predictive signal variance."""
    indices = np.arange(1, basis.shape[0] + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.45))
    count_text: List[str] = []

    for model, marker in zip(models, ("o", "s", "^")):
        contribution = _signal_contributions(basis, model)
        color = MODEL_COLORS[model.name]
        axes[0].semilogy(
            indices[:signal_limit],
            contribution[:signal_limit],
            color=color,
            marker=marker,
            markersize=3.4,
            label=model.name,
        )
        cumulative = np.cumsum(contribution) / np.sum(contribution)
        axes[1].plot(
            indices[:signal_limit],
            cumulative[:signal_limit],
            color=color,
            marker=marker,
            markevery=max(1, len(indices) // 10),
            markersize=3.4,
            label=model.name,
        )
        m90 = _components_for_share(contribution, 0.90)
        m95 = _components_for_share(contribution, 0.95)
        m99 = _components_for_share(contribution, 0.99)
        count_text.append(f"{model.name}: {m90} / {m95} / {m99}")
        axes[1].scatter(
            [m99],
            [cumulative[m99 - 1]],
            s=26,
            facecolor="white",
            edgecolor=color,
            linewidth=1.2,
            zorder=4,
        )

    axes[0].set_xlim(0.5, signal_limit + 0.5)
    axes[0].set_xlabel(r"Component $j$")
    axes[0].set_ylabel(r"$\lambda_j\langle\beta,\phi_j\rangle_T^2$ (log scale)")
    axes[0].set_title("Componentwise signal contribution", fontweight="bold")
    axes[0].grid(axis="y", which="both", color="#E5E5E5", linewidth=0.6)
    axes[0].legend(loc="upper right")
    _panel_label(axes[0], "A")

    for share, style in ((0.90, "--"), (0.95, "-."), (0.99, ":")):
        axes[1].axhline(share, color="#777777", linestyle=style, linewidth=0.9)
        axes[1].text(
            signal_limit + 0.2,
            share,
            f"{share:.0%}",
            va="center",
            fontsize=8,
            color="#555555",
            clip_on=False,
        )
    axes[1].set_xlim(0.5, signal_limit + 0.5)
    axes[1].set_ylim(0.0, 1.025)
    axes[1].set_xlabel(r"Number of components $m$")
    axes[1].set_ylabel("Cumulative predictive signal")
    axes[1].yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    axes[1].set_title("Cumulative signal captured", fontweight="bold")
    axes[1].grid(axis="y", color="#E5E5E5", linewidth=0.6)
    axes[1].legend(loc="lower right")
    axes[1].text(
        0.04,
        0.08,
        "Components for 90% / 95% / 99%\n" + "\n".join(count_text),
        transform=axes[1].transAxes,
        fontsize=8.1,
        va="bottom",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#CCCCCC"},
    )
    _panel_label(axes[1], "B")

    fig.suptitle(
        "Allocation of predictive signal across basis components",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )
    fig.text(
        0.5,
        0.012,
        "Open markers identify the first component count exceeding 99% of the exact discrete signal variance.",
        ha="center",
        fontsize=8.7,
    )
    fig.tight_layout(rect=(0.025, 0.055, 0.975, 0.94), w_pad=2.25)
    _save_figure(fig, output_dir / "predictive_signal_allocation.pdf", also_png)


def plot_functional_predictor_examples(
    grid: np.ndarray,
    basis: np.ndarray,
    models: Sequence[core.ModelSpec],
    seed: int,
    trajectory_count: int,
    output_dir: Path,
    also_png: bool,
) -> None:
    """Plot matched sampled curves under the two distinct covariance designs."""
    model_1, _, model_3 = models
    rng = np.random.default_rng(seed)
    scores = rng.normal(size=(trajectory_count, basis.shape[0]))
    cases: Tuple[Tuple[core.ModelSpec, str], ...] = (
        (model_1, "Models 1 and 2"),
        (model_3, "Model 3"),
    )
    prepared: List[Tuple[core.ModelSpec, str, np.ndarray, np.ndarray]] = []
    maximum = 0.0
    for model, label in cases:
        curves = (scores * np.sqrt(model.eigenvalues)) @ basis
        pointwise_sd = np.sqrt(np.sum(model.eigenvalues[:, None] * basis**2, axis=0))
        band = GAUSSIAN_90_QUANTILE * pointwise_sd
        maximum = max(maximum, float(np.max(np.abs(curves))), float(np.max(band)))
        prepared.append((model, label, curves, band))

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.35), sharex=True, sharey=True)
    y_limit = 1.06 * maximum
    for index, (ax, item) in enumerate(zip(axes, prepared)):
        model, label, curves, band = item
        color = MODEL_COLORS[model.name]
        ax.fill_between(
            grid,
            -band,
            band,
            color=color,
            alpha=0.16,
            linewidth=0.0,
            label="Exact pointwise 90% band",
        )
        for curve_index, curve in enumerate(curves):
            ax.plot(
                grid,
                curve,
                color=color,
                alpha=0.34,
                linewidth=0.8,
                label="Sample curves" if curve_index == 0 else None,
            )
        ax.axhline(0.0, color="#666666", linewidth=0.7)
        ax.set_ylim(-y_limit, y_limit)
        ax.set_xlabel(r"Location $s$")
        ax.set_title(label, fontweight="bold")
        ax.legend(loc="upper right")
        _panel_label(ax, chr(ord("A") + index))
    axes[0].set_ylabel(r"Functional predictor $X(s)$")

    fig.suptitle(
        "Representative functional predictors",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )
    fig.text(
        0.5,
        0.012,
        "The same Gaussian score draws are used in both panels; shaded bands are exact under the DGP.",
        ha="center",
        fontsize=8.7,
    )
    fig.tight_layout(rect=(0.025, 0.055, 0.995, 0.94), w_pad=1.6)
    _save_figure(fig, output_dir / "functional_predictor_examples.pdf", also_png)


def make_summary_rows(
    basis: np.ndarray,
    models: Sequence[core.ModelSpec],
    coefficients: Mapping[str, np.ndarray],
    noise_sd: float,
) -> List[Dict[str, object]]:
    """Collect exact discrete setup quantities in a machine-readable table."""
    grid_size = basis.shape[1]
    rows: List[Dict[str, object]] = []
    for model in models:
        contribution = _signal_contributions(basis, model)
        signal_variance = float(np.sum(contribution))
        noise_variance = float(noise_sd**2)
        predictor_variance = float(
            np.sum(model.eigenvalues[:, None] * basis**2) / grid_size
        )
        first_five_share = float(np.sum(contribution[:5]) / signal_variance)
        rows.append(
            {
                "model": model.name,
                "controlled_role": MODEL_ROLES[model.name],
                "basis_size_J": basis.shape[0],
                "grid_size_T": grid_size,
                "noise_sd": noise_sd,
                "slope_norm_squared_discrete": float(model.beta @ model.beta / grid_size),
                "integrated_predictor_variance_discrete": predictor_variance,
                "signal_variance_discrete": signal_variance,
                "response_variance_discrete": signal_variance + noise_variance,
                "signal_to_noise_ratio": signal_variance / noise_variance,
                "first_five_signal_share": first_five_share,
                "components_for_90pct_signal": _components_for_share(contribution, 0.90),
                "components_for_95pct_signal": _components_for_share(contribution, 0.95),
                "components_for_99pct_signal": _components_for_share(contribution, 0.99),
                "first_slope_coefficient": float(coefficients[model.name][0]),
                "fifth_slope_coefficient": float(coefficients[model.name][4]),
                "first_eigenvalue": float(model.eigenvalues[0]),
                "fifth_eigenvalue": float(model.eigenvalues[4]),
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create thesis figures describing the three frozen simulation DGPs."
    )
    parser.add_argument("--basis-size", type=int, default=100, metavar="J")
    parser.add_argument("--grid-size", type=int, default=200, metavar="T")
    parser.add_argument("--noise-sd", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--trajectory-count",
        type=int,
        default=10,
        help="matched sample curves per covariance design",
    )
    parser.add_argument(
        "--coefficient-limit",
        type=int,
        default=20,
        help="number of slope coefficients displayed",
    )
    parser.add_argument(
        "--spectrum-limit",
        type=int,
        default=30,
        help="number of eigenvalues displayed in the decay panel",
    )
    parser.add_argument(
        "--signal-limit",
        type=int,
        default=20,
        help="number of componentwise signal contributions displayed",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("simulation_setup_figures"),
    )
    parser.add_argument(
        "--png",
        action="store_true",
        help="also create 300-dpi PNG copies of all figures",
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.basis_size < 5:
        raise ValueError("--basis-size must be at least 5")
    if args.grid_size < args.basis_size:
        raise ValueError("--grid-size must be at least --basis-size")
    if args.grid_size < 10:
        raise ValueError("--grid-size must be at least 10")
    if not np.isfinite(args.noise_sd) or args.noise_sd <= 0.0:
        raise ValueError("--noise-sd must be positive and finite")
    if not 1 <= args.trajectory_count <= 100:
        raise ValueError("--trajectory-count must be between 1 and 100")
    for option in ("coefficient_limit", "spectrum_limit", "signal_limit"):
        value = int(getattr(args, option))
        if not 1 <= value <= args.basis_size:
            raise ValueError(
                f"--{option.replace('_', '-')} must be between 1 and --basis-size"
            )


def main() -> None:
    args = parse_args()
    _validate_args(args)
    _apply_style()

    grid = np.linspace(0.0, 1.0, args.grid_size)
    basis = core.create_cosine_basis(grid, args.basis_size)
    models = core.make_model_specs(args.basis_size, basis)
    coefficients = _recover_basis_coefficients(basis, models)
    _validate_controlled_comparisons(models, coefficients)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print("Creating figures for the three frozen simulation setups")
    print(f"  basis: J={args.basis_size}, T={args.grid_size}, fixed Babii convention")
    print(f"  output: {output_dir.resolve()}")

    plot_true_slope_structure(
        grid,
        models,
        coefficients,
        args.coefficient_limit,
        output_dir,
        args.png,
    )
    plot_covariance_spectrum(models, args.spectrum_limit, output_dir, args.png)
    plot_predictive_signal_allocation(
        basis,
        models,
        args.signal_limit,
        output_dir,
        args.png,
    )
    plot_functional_predictor_examples(
        grid,
        basis,
        models,
        args.seed,
        args.trajectory_count,
        output_dir,
        args.png,
    )
    summary_rows = make_summary_rows(basis, models, coefficients, args.noise_sd)
    _write_csv(output_dir / "setup_summary.csv", summary_rows)

    for row in summary_rows:
        print(
            f"  {row['model']}: signal variance={row['signal_variance_discrete']:.6g}, "
            f"components for 99%={row['components_for_99pct_signal']}"
        )
    suffix = "PDF and PNG" if args.png else "PDF"
    print(f"Created four {suffix} figures and setup_summary.csv")


if __name__ == "__main__":
    main()
