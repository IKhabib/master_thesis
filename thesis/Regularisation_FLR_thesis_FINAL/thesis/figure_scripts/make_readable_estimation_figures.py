
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator, ScalarFormatter, LogLocator, NullFormatter
import numpy as np
from scipy.stats import norm

WIDTH = 6.14
METHODS = ("CG-FPLS-code", "Raw FPLS", "Arnoldi FPLS", "FPCR")
LABELS = ("CG", "Raw", "Arnoldi", "FPCR")
COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")
BIAS, VARIANCE, NOISE = "#D55E00", "#0072B2", "#777777"


def style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.titlesize": 9, "axes.labelsize": 9,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5, "axes.linewidth": .6,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "pdf.fonttype": 42, "savefig.dpi": 250,
    })


def save(fig: plt.Figure, path: Path) -> None:
    # Keep the declared physical canvas: bbox_inches='tight' would change
    # its final placement scale and make the font-size guarantee unreliable.
    fig.savefig(path)
    plt.close(fig)


def log_axis(ax: plt.Axes) -> None:
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=5))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(axis="y", which="major", alpha=.2, linewidth=.5)
    ax.set_xticks(range(4), LABELS)


def boxplots(raw: dict, output: Path) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(WIDTH, 6.1))
    fig.subplots_adjust(left=.115, right=.98, top=.91, bottom=.08,
                        hspace=.55, wspace=.35)
    for model in range(3):
        for col, metric in enumerate(("ise", "mspe")):
            ax = axes[model, col]
            data = [raw[metric][model, :, method] for method in range(4)]
            artists = ax.boxplot(
                data, positions=np.arange(4), widths=.58,
                showfliers=True, whis=1.5, patch_artist=True,
                flierprops={"marker": ".", "markersize": 1.7,
                            "markerfacecolor": "#555555",
                            "markeredgecolor": "#555555", "alpha": .35},
                medianprops={"color": "#111111", "linewidth": .8},
                whiskerprops={"linewidth": .6}, capprops={"linewidth": .6})
            for patch, color in zip(artists["boxes"], COLORS):
                patch.set_facecolor(color)
                patch.set_alpha(.72)
            flagged = raw["method_instability"][model, :, 1].astype(bool)
            ax.scatter(np.ones(flagged.sum()), raw[metric][model, flagged, 1],
                       marker="D", s=13, facecolors="none", edgecolors="#B2182B",
                       linewidths=.6, zorder=4)
            log_axis(ax)
            ax.set_title(f"Model {model + 1}", loc="left", pad=3)
            ax.set_ylabel("Slope ISE" if col == 0 else "Test MSPE")
    fig.legend(handles=[Line2D([], [], marker="D", linestyle="none",
                              markerfacecolor="none", color="#B2182B",
                              markersize=4, label="Raw instability flag")],
               loc="upper center", bbox_to_anchor=(.5, .995), frameon=False)
    save(fig, output / "babii_style_error_boxplots_with_tails.pdf")


def decomposition(rows: list[dict], output: Path, paired: bool) -> None:
    subset = "raw numerically stable subset" if paired else "all replications"
    # Use the exact frozen subset label rather than infer membership anew.
    if paired:
        candidates = {row["subset"] for row in rows if row["subset"] != "all replications"}
        if len(candidates) != 1:
            raise ValueError(f"Unexpected paired subsets: {candidates}")
        subset = candidates.pop()
    lookup = {(r["model"], r["method"]): r for r in rows if r["subset"] == subset}
    fig, axes = plt.subplots(3, 2, figsize=(WIDTH, 6.1))
    fig.subplots_adjust(left=.115, right=.98, top=.895, bottom=.08,
                        hspace=.57, wspace=.35)
    x = np.arange(4)
    for model in range(3):
        model_rows = [lookup[(f"Model {model + 1}", m)] for m in METHODS]
        title = f"Model {model + 1}"
        if paired:
            title += f"  (N = {int(model_rows[0]['decomposition_replications']):,})"
        for col in range(2):
            ax = axes[model, col]
            prefix = "slope" if col == 0 else "prediction"
            bias = np.array([float(r[f"{prefix}_squared_bias"]) for r in model_rows])
            variance = np.array([float(r[f"{prefix}_variance"]) for r in model_rows])
            width = .23
            ax.bar(x - width / 2 if col == 0 else x - width,
                   bias, width, color=BIAS)
            ax.bar(x + width / 2 if col == 0 else x,
                   variance, width, color=VARIANCE)
            if col == 1:
                ax.bar(x + width, [float(r["irreducible_noise_variance"]) for r in model_rows],
                       width, color=NOISE)
            total = []
            for row in model_rows:
                available = row["decomposition_available"].lower() == "true"
                key = ("slope_mse" if col == 0 else "expected_test_mspe") if available else (
                    "archived_mean_ise" if col == 0 else "archived_mean_test_mspe")
                total.append(float(row[key]))
            ax.scatter(x, total, marker="D", s=18, color="#111111", zorder=4)
            log_axis(ax)
            ax.set_title(title, loc="left", pad=3)
            ax.set_ylabel("Slope risk" if col == 0 else "Prediction risk")
    fig.legend(handles=[Patch(color=BIAS, label="Squared bias"),
                        Patch(color=VARIANCE, label="Variance"),
                        Patch(color=NOISE, label="Noise"),
                        Line2D([], [], color="#111111", marker="D", linestyle="none",
                               markersize=4, label="Total")],
               ncol=4, loc="upper center", bbox_to_anchor=(.5, .995),
               columnspacing=1.15, handlelength=1.3, frameon=False)
    filename = "bias_variance_decomposition_raw_stable_subset.pdf" if paired else "bias_variance_decomposition.pdf"
    save(fig, output / filename)


def robust_limits(samples: list[np.ndarray]) -> tuple[float, float]:
    combined = np.concatenate([s[np.isfinite(s)] for s in samples])
    lo, hi = np.quantile(combined, (.005, .995))
    pad = .04 * (hi - lo) if hi > lo else max(abs(lo) * .05, 1e-6)
    return float(lo - pad), float(hi + pad)


def compact_ticks(ax: plt.Axes) -> None:
    ax.xaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=2))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=2, min_n_ticks=2))
    for axis in (ax.xaxis, ax.yaxis):
        formatter = ScalarFormatter(useOffset=False)
        formatter.set_powerlimits((-3, 3))
        axis.set_major_formatter(formatter)
    ax.grid(axis="y", alpha=.17, linewidth=.5)


def histograms(cache: dict, output: Path, pointwise: bool) -> None:
    samples = cache["point_samples" if pointwise else "coefficient_samples"]
    truths = cache["true_points" if pointwise else "true_coefficients"]
    row_count = 5 if pointwise else 3
    filename = "beta_point_histograms.pdf" if pointwise else "basis_coefficient_histograms.pdf"
    with PdfPages(output / filename) as pdf:
        for model in range(3):
            height = 6.35 if pointwise else 4.7
            fig, axes = plt.subplots(row_count, 4, figsize=(WIDTH, height), squeeze=False)
            fig.subplots_adjust(left=.12, right=.965, top=.89, bottom=.085,
                                wspace=.43, hspace=.59 if pointwise else .54)
            for row in range(row_count):
                row_samples = [samples[model, :, method, row] for method in range(4)]
                limits = robust_limits(row_samples)
                for method, values in enumerate(row_samples):
                    ax = axes[row, method]
                    finite = values[np.isfinite(values)]
                    inside = finite[(finite >= limits[0]) & (finite <= limits[1])]
                    outside = finite.size - inside.size
                    ax.hist(inside, bins=36, density=True, color=COLORS[method],
                            edgecolor="white", linewidth=.18, alpha=.78)
                    ax.axvline(truths[model, row], color="#B2182B", linewidth=1.05)
                    mean = finite.mean()
                    if limits[0] <= mean <= limits[1]:
                        ax.axvline(mean, color="#1B7837", linestyle="--", linewidth=1.05)
                    else:
                        ax.text(.02, .96, "Mean outside", transform=ax.transAxes,
                                fontsize=8.5, va="top", color="#1B7837")
                    # Preserve numerical-tail disclosure while removing tiny
                    # bias/variance annotations already present in the archive.
                    if outside:
                        ax.text(.98, .95, f"out: {outside}", ha="right", va="top",
                                transform=ax.transAxes, fontsize=8.5,
                                bbox={"facecolor": "white", "edgecolor": "none", "alpha": .9, "pad": .6})
                    ax.set_xlim(limits)
                    compact_ticks(ax)
                    if row == 0:
                        ax.set_title(LABELS[method], pad=5)
                    if method == 0:
                        label = rf"$s={cache['beta_points'][row]:.2f}$" if pointwise else rf"$\widehat b_{row + 1}$"
                        ax.set_ylabel(label + "\nDensity", labelpad=4)
            fig.legend(handles=[Line2D([], [], color="#B2182B", label="True value"),
                                Line2D([], [], color="#1B7837", linestyle="--", label="Monte Carlo mean")],
                       loc="upper center", bbox_to_anchor=(.5, 1), ncol=2, frameon=False)
            fig.supxlabel("Estimated slope value" if pointwise else "Estimated coefficient value",
                          fontsize=9, y=.005)
            pdf.savefig(fig)
            plt.close(fig)


def qq(cache: dict, output: Path) -> None:
    with PdfPages(output / "coefficient_normal_qq.pdf") as pdf:
        for model in range(3):
            fig, axes = plt.subplots(3, 4, figsize=(WIDTH, 4.8))
            fig.subplots_adjust(left=.12, right=.965, top=.94, bottom=.12,
                                wspace=.43, hspace=.51)
            for row in range(3):
                for method in range(4):
                    ax = axes[row, method]
                    sample = cache["coefficient_samples"][model, :, method, row]
                    sample = np.sort(sample[np.isfinite(sample)])
                    observed = (sample - sample.mean()) / sample.std(ddof=0)
                    theoretical = norm.ppf((np.arange(1, sample.size + 1) - .5) / sample.size)
                    ax.scatter(theoretical, observed, s=1.5, alpha=.38,
                               color=COLORS[method], linewidths=0, rasterized=True)
                    low = min(theoretical[0], observed[0])
                    high = max(theoretical[-1], observed[-1])
                    ax.plot([low, high], [low, high], linestyle="--", linewidth=.65, color="#444444")
                    ax.set_xlim(low, high)
                    ax.set_ylim(low, high)
                    compact_ticks(ax)
                    if row == 0:
                        ax.set_title(LABELS[method], pad=5)
                    if method == 0:
                        ax.set_ylabel(rf"$\widehat b_{row + 1}$" + "\nObserved", labelpad=4)
            fig.supxlabel("Standard normal quantiles", y=.01, fontsize=9)
            pdf.savefig(fig)
            plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True,
                        help="Frozen four_method_study_FINAL_R5000 root")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.results / "simulation_results/raw_results.npz", allow_pickle=False) as src:
        raw = {name: src[name] for name in src.files}
    with np.load(args.results / "thesis_figures/selected_beta_diagnostics_cache.npz", allow_pickle=False) as src:
        cache = {name: src[name] for name in src.files}
    with (args.results / "thesis_figures/bias_variance_decomposition.csv").open(newline="") as src:
        rows = list(csv.DictReader(src))
    if tuple(raw["method_names"].astype(str)) != METHODS:
        raise ValueError("Unexpected method ordering")
    # Missing cached Raw-FPLS samples must correspond precisely to the frozen
    # instability rule. This keeps the original conditional scope explicit.
    for name in ("coefficient_samples", "point_samples"):
        missing = ~np.isfinite(cache[name]).all(axis=-1)
        expected = np.zeros_like(raw["method_instability"], dtype=bool)
        expected[:, :, 1] = raw["method_instability"][:, :, 1]
        if not np.array_equal(missing, expected):
            raise ValueError(f"Unexpected cache omission pattern: {name}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    style()
    boxplots(raw, args.output_dir)
    decomposition(rows, args.output_dir, paired=False)
    decomposition(rows, args.output_dir, paired=True)
    histograms(cache, args.output_dir, pointwise=False)
    histograms(cache, args.output_dir, pointwise=True)
    qq(cache, args.output_dir)
    print("Rendered six PDF assets (12 pages total) from frozen inputs; no models rerun.")


if __name__ == "__main__":
    main()
