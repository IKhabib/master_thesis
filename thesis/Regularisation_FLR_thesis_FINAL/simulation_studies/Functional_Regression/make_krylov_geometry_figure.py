

from __future__ import annotations

import argparse
import csv
import io
import math
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

import four_method_core as core


RAW_COLOR = "#D55E00"
ARNOLDI_COLOR = "#009E73"
NEUTRAL_COLOR = "#5F6368"
LIGHT_NEUTRAL = "#D7DADF"
@dataclass(frozen=True)
class Geometry:
    """Dependence matrices and scalar diagnostics for one simulation DGP."""

    model_name: str
    raw_dependence: np.ndarray
    arnoldi_dependence: np.ndarray
    raw_condition: float
    arnoldi_defect: float
    raw_edge_count: int
    arnoldi_edge_count: int


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _pdf_bytes_are_complete(data: bytes) -> bool:
    if len(data) < 100 or not data.startswith(b"%PDF-"):
        return False
    if b"%%EOF" not in data[-2048:]:
        return False
    matches = list(re.finditer(rb"startxref\s+(\d+)\s+%%EOF", data[-4096:]))
    if not matches:
        return False
    offset = int(matches[-1].group(1))
    if not 0 <= offset < len(data):
        return False
    prefix = data[offset : offset + 64]
    return bool(prefix.startswith(b"xref") or re.match(rb"\d+\s+\d+\s+obj", prefix))


def _write_pdf(path: Path, data: bytes) -> None:
    """Atomically write a structurally complete PDF."""
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


def _write_png(path: Path, data: bytes) -> None:
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


def _save_figure(fig: plt.Figure, output_dir: Path, png_dpi: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "krylov_geometry.pdf"
    png_path = output_dir / "krylov_geometry.png"
    try:
        buffer = io.BytesIO()
        fig.savefig(buffer, format="pdf", bbox_inches="tight")
        _write_pdf(pdf_path, buffer.getvalue())
        png_buffer = io.BytesIO()
        fig.savefig(
            png_buffer,
            format="png",
            dpi=png_dpi,
            bbox_inches="tight",
        )
        _write_png(png_path, png_buffer.getvalue())
    finally:
        plt.close(fig)


def _write_summary(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError("geometry summary cannot be empty")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _population_moments(
    basis: np.ndarray,
    model: core.ModelSpec,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return population K and r under the frozen discrete DGP."""
    grid_size = basis.shape[1]
    operator = basis.T @ (model.eigenvalues[:, None] * basis) / grid_size
    operator = 0.5 * (operator + operator.T)
    moment = operator @ model.beta
    return operator, moment


def _normalized_columns(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=0)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 0.0):
        raise RuntimeError("Krylov basis contains a zero or nonfinite direction")
    return matrix / norms


def compute_geometry(
    basis: np.ndarray,
    models: Sequence[core.ModelSpec],
    components: int,
    edge_threshold: float,
    rank_tolerance: float,
) -> List[Geometry]:
    """Compute comparable raw and Arnoldi geometry for the selected DGPs."""
    geometries: List[Geometry] = []
    upper = np.triu_indices(components, k=1)
    for model in models:
        operator, moment = _population_moments(basis, model)
        raw = core.raw_krylov_matrix(operator, moment, components)
        raw_normalized = _normalized_columns(raw)
        raw_dependence = np.abs(raw_normalized.T @ raw_normalized)

        arnoldi = core.arnoldi_krylov_basis(
            operator,
            moment,
            components,
            rank_tolerance=rank_tolerance,
        )
        if arnoldi.shape[1] != components:
            raise RuntimeError(
                f"{model.name} produced only {arnoldi.shape[1]} Arnoldi "
                f"directions; reduce --components or --rank-tolerance"
            )
        arnoldi_dependence = np.abs(arnoldi.T @ arnoldi)

        singular_values = np.linalg.svd(raw_normalized, compute_uv=False)
        raw_condition = float(singular_values[0] / singular_values[-1])
        arnoldi_defect = float(
            np.linalg.norm(
                arnoldi.T @ arnoldi - np.eye(components),
                ord=2,
            )
        )
        geometries.append(
            Geometry(
                model_name=model.name,
                raw_dependence=raw_dependence,
                arnoldi_dependence=arnoldi_dependence,
                raw_condition=raw_condition,
                arnoldi_defect=arnoldi_defect,
                raw_edge_count=int(
                    np.count_nonzero(raw_dependence[upper] >= edge_threshold)
                ),
                arnoldi_edge_count=int(
                    np.count_nonzero(arnoldi_dependence[upper] >= edge_threshold)
                ),
            )
        )
    return geometries


def _hex_rgb(color: str) -> np.ndarray:
    color = color.lstrip("#")
    return np.array([int(color[index : index + 2], 16) for index in (0, 2, 4)]) / 255.0


def _blend_with_white(color: str, strength: float) -> np.ndarray:
    strength = float(np.clip(strength, 0.0, 1.0))
    return (1.0 - strength) * np.ones(3) + strength * _hex_rgb(color)


def _composite_dependence_image(geometry: Geometry) -> np.ndarray:
    """Encode raw below and Arnoldi above the diagonal on a shared scale."""
    components = geometry.raw_dependence.shape[0]
    image = np.ones((components, components, 3), dtype=float)
    for row in range(components):
        for column in range(components):
            if row > column:
                value = geometry.raw_dependence[row, column]
                image[row, column] = _blend_with_white(RAW_COLOR, value**0.75)
            elif row < column:
                value = geometry.arnoldi_dependence[row, column]
                image[row, column] = _blend_with_white(ARNOLDI_COLOR, value**0.75)
            else:
                image[row, column] = np.array([0.94, 0.94, 0.94])
    return image


def _plot_dependence_matrix(
    ax: plt.Axes,
    geometry: Geometry,
    components: int,
) -> None:
    ax.imshow(
        _composite_dependence_image(geometry),
        interpolation="nearest",
        origin="upper",
        vmin=0.0,
        vmax=1.0,
    )
    indices = np.arange(components)
    labels = [str(index) for index in range(1, components + 1)]
    ax.set_xticks(indices)
    ax.set_yticks(indices)
    ax.set_xticklabels(labels, fontsize=7.3)
    ax.set_yticklabels(labels, fontsize=7.3)
    ax.set_xlabel("Krylov component", fontsize=8.5, labelpad=4)
    ax.set_ylabel("Krylov component", fontsize=8.5, labelpad=4)
    ax.set_xticks(np.arange(-0.5, components, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, components, 1), minor=True)
    ax.grid(which="minor", color="#FFFFFF", linewidth=0.45)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.7)
        spine.set_color("#777777")
    ax.text(
        0.97,
        0.97,
        "Arnoldi",
        transform=ax.transAxes,
        ha="right",
        va="top",
        color=ARNOLDI_COLOR,
        fontsize=7.8,
        fontweight="bold",
    )
    ax.text(
        0.04,
        0.07,
        "Raw",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        color="white",
        fontsize=7.8,
        fontweight="bold",
    )


def _network_positions(components: int) -> np.ndarray:
    angles = np.pi / 2.0 - 2.0 * np.pi * np.arange(components) / components
    return np.column_stack((np.cos(angles), np.sin(angles)))


def _plot_dependence_network(
    ax: plt.Axes,
    dependence: np.ndarray,
    color: str,
    edge_threshold: float,
    edge_count: int,
    diagnostic: str,
) -> None:
    components = dependence.shape[0]
    positions = _network_positions(components)
    guide = Circle(
        (0.0, 0.0),
        1.0,
        facecolor="none",
        edgecolor=LIGHT_NEUTRAL,
        linewidth=0.8,
        zorder=0,
    )
    ax.add_patch(guide)

    for first in range(components):
        for second in range(first + 1, components):
            value = float(dependence[first, second])
            if value < edge_threshold:
                continue
            relative = (value - edge_threshold) / max(1.0 - edge_threshold, 1e-12)
            alpha = 0.14 + 0.40 * relative
            linewidth = 0.35 + 0.95 * value
            ax.plot(
                positions[[first, second], 0],
                positions[[first, second], 1],
                color=color,
                alpha=alpha,
                linewidth=linewidth,
                solid_capstyle="round",
                zorder=1,
            )

    for index, (x_coordinate, y_coordinate) in enumerate(positions, start=1):
        node = Circle(
            (float(x_coordinate), float(y_coordinate)),
            0.082,
            facecolor="white",
            edgecolor=color,
            linewidth=1.35,
            zorder=3,
        )
        ax.add_patch(node)
        ax.text(
            x_coordinate,
            y_coordinate,
            str(index),
            ha="center",
            va="center",
            fontsize=7.4,
            color="#222222",
            zorder=4,
        )

    possible = components * (components - 1) // 2
    ax.text(
        0.5,
        -0.025,
        f"{edge_count}/{possible} edges shown",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8.0,
        color=NEUTRAL_COLOR,
    )
    ax.text(
        0.5,
        -0.115,
        diagnostic,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8.1,
        color="#222222",
    )
    ax.set_xlim(-1.20, 1.20)
    ax.set_ylim(-1.22, 1.20)
    ax.set_aspect("equal")
    ax.axis("off")


def _scientific_math(value: float) -> str:
    if value == 0.0:
        return "0"
    exponent = int(math.floor(math.log10(abs(value))))
    mantissa = value / (10.0**exponent)
    return rf"{mantissa:.1f}\times 10^{{{exponent}}}"


def make_figure(
    geometries: Sequence[Geometry],
    components: int,
    edge_threshold: float,
    basis_size: int,
    grid_size: int,
) -> plt.Figure:
    """Create the baseline-model matrix-and-network figure."""
    if len(geometries) != 1 or geometries[0].model_name != "Model 1":
        raise ValueError("the thesis geometry figure must contain baseline Model 1")

    fig = plt.figure(figsize=(12.2, 4.9))
    grid = fig.add_gridspec(
        1,
        3,
        left=0.065,
        right=0.975,
        bottom=0.205,
        top=0.765,
        wspace=0.25,
        width_ratios=(0.92, 1.05, 1.05),
    )

    geometry = geometries[0]
    matrix_axis = fig.add_subplot(grid[0, 0])
    raw_axis = fig.add_subplot(grid[0, 1])
    arnoldi_axis = fig.add_subplot(grid[0, 2])
    _plot_dependence_matrix(matrix_axis, geometry, components)
    _plot_dependence_network(
        raw_axis,
        geometry.raw_dependence,
        RAW_COLOR,
        edge_threshold,
        geometry.raw_edge_count,
        rf"$\kappa_2(\widetilde{{H}}_{{{components}}})="
        + _scientific_math(geometry.raw_condition)
        + "$",
    )
    _plot_dependence_network(
        arnoldi_axis,
        geometry.arnoldi_dependence,
        ARNOLDI_COLOR,
        edge_threshold,
        geometry.arnoldi_edge_count,
        rf"$\|Q^\mathsf{{T}}Q-I\|_2="
        + _scientific_math(geometry.arnoldi_defect)
        + "$",
    )
    matrix_axis.set_title(
        "Pairwise dependence matrix",
        fontsize=11.2,
        fontweight="bold",
        pad=12,
    )
    raw_axis.set_title(
        "Raw FPLS Krylov basis",
        fontsize=11.2,
        fontweight="bold",
        pad=12,
    )
    arnoldi_axis.set_title(
        "Arnoldi FPLS (CGS2)",
        fontsize=11.2,
        fontweight="bold",
        pad=12,
    )

    fig.suptitle(
        "Krylov geometry in Model 1 (baseline)",
        fontsize=16,
        fontweight="bold",
        y=0.975,
    )
    fig.text(
        0.5,
        0.885,
        "The same population Krylov spaces are expressed in nearly collinear versus orthonormal coordinates.",
        ha="center",
        va="center",
        fontsize=10.2,
        color=NEUTRAL_COLOR,
    )

    fig.text(
        0.5,
        0.025,
        (
            rf"An edge is shown when the absolute normalized inner product is at least {edge_threshold:.2f}. "
            rf"Population geometry: $J={basis_size}$, $T={grid_size}$, $m={components}$."
        ),
        ha="center",
        va="center",
        fontsize=8.5,
        color=NEUTRAL_COLOR,
    )
    return fig


def summary_rows(
    geometries: Sequence[Geometry],
    components: int,
    edge_threshold: float,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    upper = np.triu_indices(components, k=1)
    possible = components * (components - 1) // 2
    for geometry in geometries:
        raw_off_diagonal = geometry.raw_dependence[upper]
        arnoldi_off_diagonal = geometry.arnoldi_dependence[upper]
        rows.append(
            {
                "model": geometry.model_name,
                "components": components,
                "edge_threshold": edge_threshold,
                "possible_edges": possible,
                "raw_edges_shown": geometry.raw_edge_count,
                "arnoldi_edges_shown": geometry.arnoldi_edge_count,
                "raw_normalized_basis_condition": geometry.raw_condition,
                "raw_min_abs_offdiagonal_inner_product": float(
                    np.min(raw_off_diagonal)
                ),
                "raw_median_abs_offdiagonal_inner_product": float(
                    np.median(raw_off_diagonal)
                ),
                "raw_max_abs_offdiagonal_inner_product": float(
                    np.max(raw_off_diagonal)
                ),
                "arnoldi_orthogonality_defect_2norm": geometry.arnoldi_defect,
                "arnoldi_max_abs_offdiagonal_inner_product": float(
                    np.max(arnoldi_off_diagonal)
                ),
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create the raw-versus-Arnoldi Krylov geometry figure."
    )
    parser.add_argument("--basis-size", type=int, default=100, metavar="J")
    parser.add_argument("--grid-size", type=int, default=200, metavar="T")
    parser.add_argument("--components", type=int, default=12, metavar="M")
    parser.add_argument("--rank-tolerance", type=float, default=1e-10)
    parser.add_argument("--edge-threshold", type=float, default=0.20)
    parser.add_argument("--png-dpi", type=int, default=300)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("krylov_geometry_figure"),
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.basis_size < 5:
        raise ValueError("--basis-size must be at least 5")
    if args.grid_size < args.basis_size:
        raise ValueError("--grid-size must be at least --basis-size")
    if not 3 <= args.components <= min(20, args.basis_size, args.grid_size):
        raise ValueError("--components must be between 3 and min(20, J, T)")
    if not 0.0 < args.edge_threshold < 1.0:
        raise ValueError("--edge-threshold must lie strictly between 0 and 1")
    if not np.isfinite(args.rank_tolerance) or args.rank_tolerance <= 0.0:
        raise ValueError("--rank-tolerance must be positive and finite")
    if not 100 <= args.png_dpi <= 600:
        raise ValueError("--png-dpi must be between 100 and 600")


def main() -> None:
    args = parse_args()
    _validate_args(args)
    _apply_style()
    grid = np.linspace(0.0, 1.0, args.grid_size)
    basis = core.create_cosine_basis(grid, args.basis_size)
    models = core.make_model_specs(args.basis_size, basis)[:1]
    geometries = compute_geometry(
        basis,
        models,
        args.components,
        args.edge_threshold,
        args.rank_tolerance,
    )
    figure = make_figure(
        geometries,
        args.components,
        args.edge_threshold,
        args.basis_size,
        args.grid_size,
    )
    output_dir = Path(args.output_dir)
    _save_figure(figure, output_dir, args.png_dpi)
    rows = summary_rows(geometries, args.components, args.edge_threshold)
    _write_summary(output_dir / "krylov_geometry_summary.csv", rows)

    print(f"Created Krylov geometry outputs in {output_dir.resolve()}")
    for row in rows:
        print(
            f"  {row['model']}: raw edges={row['raw_edges_shown']}/"
            f"{row['possible_edges']}, Arnoldi edges={row['arnoldi_edges_shown']}/"
            f"{row['possible_edges']}, raw condition="
            f"{row['raw_normalized_basis_condition']:.3e}"
        )


if __name__ == "__main__":
    main()
