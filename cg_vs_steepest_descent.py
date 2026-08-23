#!/usr/bin/env python3
"""Standalone comparison of Conjugate Gradient and steepest descent.

The comparison concerns symmetric positive-definite (SPD) linear systems

    A x = b,

or, equivalently, minimization of the strictly convex quadratic

    f(x) = 0.5 x' A x - b' x.

Both methods start from the same point and stop at the same relative-residual
tolerance.  Steepest descent uses its optimal exact line-search step, so this
is a fair comparison rather than a comparison against a poorly tuned learning
rate.  One matrix-vector product is charged for each iteration of either
method.

Outputs
-------
representative_convergence.pdf
    Relative residual and relative objective gap versus matrix-vector work.
quadratic_trajectory_2d.pdf
    Contours of a two-dimensional quadratic with both optimization paths.
quadratic_surface_3d.pdf
    Three-dimensional quadratic surface with the same two optimization paths.
condition_number_sweep.pdf
    Iteration counts and convergence rates over matched SPD problems.
condition_number_trials.csv
    One row per condition number, trial, and method.
condition_number_summary.csv
    Aggregated iteration, residual, and convergence results.
representative_history.csv
    Complete convergence history for the representative problem.
trajectory_2d.csv
    Coordinates and objective heights of both two-dimensional paths.
configuration.json
    Exact settings needed to reproduce the experiment.

Add ``--png`` to create high-resolution PNG copies of the four figures.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CG_COLOR = "#0072B2"
SD_COLOR = "#D55E00"
NEUTRAL_COLOR = "#5F6368"
METHODS = ("Conjugate Gradient", "Steepest Descent")


@dataclass(frozen=True)
class QuadraticProblem:
    """An SPD quadratic represented by either a dense or diagonal matrix."""

    b: np.ndarray
    x_star: np.ndarray
    condition_number: float
    matrix: Optional[np.ndarray] = None
    diagonal: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        b = np.asarray(self.b, dtype=float).reshape(-1)
        x_star = np.asarray(self.x_star, dtype=float).reshape(-1)
        if b.shape != x_star.shape or len(b) < 1:
            raise ValueError("b and x_star must be nonempty vectors of equal length")
        if (self.matrix is None) == (self.diagonal is None):
            raise ValueError("provide exactly one of matrix or diagonal")
        if self.matrix is not None:
            matrix = np.asarray(self.matrix, dtype=float)
            if matrix.shape != (len(b), len(b)):
                raise ValueError("matrix has incompatible dimensions")
            if not np.allclose(matrix, matrix.T, rtol=1e-12, atol=1e-12):
                raise ValueError("matrix must be symmetric")
        if self.diagonal is not None:
            diagonal = np.asarray(self.diagonal, dtype=float).reshape(-1)
            if diagonal.shape != b.shape or np.any(diagonal <= 0.0):
                raise ValueError("diagonal must be positive and match b")
        if not np.isfinite(self.condition_number) or self.condition_number < 1.0:
            raise ValueError("condition_number must be finite and at least one")

    @property
    def dimension(self) -> int:
        return len(self.b)

    def matvec(self, vector: np.ndarray) -> np.ndarray:
        vector = np.asarray(vector, dtype=float)
        if self.matrix is not None:
            return self.matrix @ vector
        if self.diagonal is None:
            raise AssertionError("unreachable matrix representation")
        return self.diagonal * vector

    def dense_matrix(self) -> np.ndarray:
        if self.matrix is not None:
            return self.matrix.copy()
        if self.diagonal is None:
            raise AssertionError("unreachable matrix representation")
        return np.diag(self.diagonal)


@dataclass
class SolverResult:
    """Final state and optional convergence history for one solver."""

    method: str
    x: np.ndarray
    iterations: int
    matvecs: int
    converged: bool
    breakdown: bool
    relative_residual: float
    iteration_history: np.ndarray
    residual_history: np.ndarray
    objective_gap_history: np.ndarray
    a_error_history: np.ndarray
    path: np.ndarray


def _random_orthogonal(dimension: int, rng: np.random.Generator) -> np.ndarray:
    candidate = rng.normal(size=(dimension, dimension))
    q_matrix, r_matrix = np.linalg.qr(candidate)
    signs = np.sign(np.diag(r_matrix))
    signs[signs == 0.0] = 1.0
    return q_matrix * signs


def make_spd_problem(
    dimension: int,
    condition_number: float,
    rng: np.random.Generator,
    *,
    rotate: bool,
) -> QuadraticProblem:
    """Create an SPD problem with a controlled geometric spectrum."""
    if dimension < 2 or condition_number < 1.0:
        raise ValueError("dimension must be at least two and kappa at least one")
    eigenvalues = np.geomspace(1.0, condition_number, dimension)
    rng.shuffle(eigenvalues)
    x_star = rng.normal(size=dimension)
    if rotate:
        eigenvectors = _random_orthogonal(dimension, rng)
        matrix = (eigenvectors * eigenvalues) @ eigenvectors.T
        matrix = 0.5 * (matrix + matrix.T)
        b = matrix @ x_star
        return QuadraticProblem(
            b=b,
            x_star=x_star,
            condition_number=float(condition_number),
            matrix=matrix,
        )
    b = eigenvalues * x_star
    return QuadraticProblem(
        b=b,
        x_star=x_star,
        condition_number=float(condition_number),
        diagonal=eigenvalues,
    )


def make_two_dimensional_problem(condition_number: float) -> QuadraticProblem:
    """Return a rotated 2D quadratic that reveals steepest-descent zig-zagging."""
    angle = np.deg2rad(32.0)
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ]
    )
    matrix = rotation @ np.diag([1.0, condition_number]) @ rotation.T
    x_star = np.array([1.0, -0.75])
    return QuadraticProblem(
        b=matrix @ x_star,
        x_star=x_star,
        condition_number=float(condition_number),
        matrix=matrix,
    )


def _metric_values(
    problem: QuadraticProblem,
    x: np.ndarray,
    initial_residual_norm: float,
    initial_gap: float,
    initial_a_error: float,
) -> Tuple[float, float, float]:
    residual = problem.b - problem.matvec(x)
    error = x - problem.x_star
    a_error_squared = max(float(error @ problem.matvec(error)), 0.0)
    gap = 0.5 * a_error_squared
    relative_residual = float(np.linalg.norm(residual) / initial_residual_norm)
    relative_gap = gap / initial_gap if initial_gap > 0.0 else 0.0
    relative_a_error = (
        math.sqrt(a_error_squared) / initial_a_error
        if initial_a_error > 0.0
        else 0.0
    )
    return relative_residual, relative_gap, relative_a_error


def _initial_metrics(
    problem: QuadraticProblem,
    x0: np.ndarray,
) -> Tuple[np.ndarray, float, float, float]:
    residual = problem.b - problem.matvec(x0)
    residual_norm = float(np.linalg.norm(residual))
    error = x0 - problem.x_star
    a_error_squared = max(float(error @ problem.matvec(error)), 0.0)
    return residual, residual_norm, 0.5 * a_error_squared, math.sqrt(a_error_squared)


def steepest_descent(
    problem: QuadraticProblem,
    *,
    x0: Optional[np.ndarray] = None,
    tolerance: float = 1e-8,
    max_iterations: int = 20000,
    track_history: bool = True,
) -> SolverResult:
    """Steepest descent with the optimal exact line-search step."""
    if tolerance <= 0.0 or max_iterations < 1:
        raise ValueError("tolerance and max_iterations must be positive")
    x = (
        np.zeros(problem.dimension)
        if x0 is None
        else np.asarray(x0, dtype=float).reshape(-1).copy()
    )
    if x.shape != (problem.dimension,):
        raise ValueError("x0 has incompatible dimensions")
    residual, initial_norm, initial_gap, initial_a_error = _initial_metrics(problem, x)
    if initial_norm == 0.0:
        return _stationary_result("Steepest Descent", x)

    iterations: List[int] = [0]
    residuals: List[float] = [1.0]
    gaps: List[float] = [1.0]
    a_errors: List[float] = [1.0]
    points: List[np.ndarray] = [x.copy()] if track_history else []
    converged = False
    breakdown = False
    completed = 0

    for iteration in range(1, max_iterations + 1):
        A_residual = problem.matvec(residual)
        denominator = float(residual @ A_residual)
        numerator = float(residual @ residual)
        if denominator <= 0.0 or not np.isfinite(denominator):
            breakdown = True
            break
        step = numerator / denominator
        x += step * residual
        residual -= step * A_residual
        completed = iteration
        relative_residual = float(np.linalg.norm(residual) / initial_norm)

        if track_history:
            metric_residual, gap, a_error = _metric_values(
                problem,
                x,
                initial_norm,
                initial_gap,
                initial_a_error,
            )
            iterations.append(iteration)
            residuals.append(metric_residual)
            gaps.append(gap)
            a_errors.append(a_error)
            points.append(x.copy())
        if relative_residual <= tolerance:
            converged = True
            break

    final_relative = float(np.linalg.norm(problem.b - problem.matvec(x)) / initial_norm)
    return SolverResult(
        method="Steepest Descent",
        x=x,
        iterations=completed,
        matvecs=completed + 1,
        converged=converged,
        breakdown=breakdown,
        relative_residual=final_relative,
        iteration_history=np.asarray(iterations, dtype=int),
        residual_history=np.asarray(residuals, dtype=float),
        objective_gap_history=np.asarray(gaps, dtype=float),
        a_error_history=np.asarray(a_errors, dtype=float),
        path=np.asarray(points, dtype=float),
    )


def conjugate_gradient(
    problem: QuadraticProblem,
    *,
    x0: Optional[np.ndarray] = None,
    tolerance: float = 1e-8,
    max_iterations: int = 20000,
    track_history: bool = True,
) -> SolverResult:
    """Unpreconditioned Conjugate Gradient for an SPD system."""
    if tolerance <= 0.0 or max_iterations < 1:
        raise ValueError("tolerance and max_iterations must be positive")
    x = (
        np.zeros(problem.dimension)
        if x0 is None
        else np.asarray(x0, dtype=float).reshape(-1).copy()
    )
    if x.shape != (problem.dimension,):
        raise ValueError("x0 has incompatible dimensions")
    residual, initial_norm, initial_gap, initial_a_error = _initial_metrics(problem, x)
    if initial_norm == 0.0:
        return _stationary_result("Conjugate Gradient", x)
    direction = residual.copy()
    residual_squared = float(residual @ residual)

    iterations: List[int] = [0]
    residuals: List[float] = [1.0]
    gaps: List[float] = [1.0]
    a_errors: List[float] = [1.0]
    points: List[np.ndarray] = [x.copy()] if track_history else []
    converged = False
    breakdown = False
    completed = 0

    for iteration in range(1, max_iterations + 1):
        A_direction = problem.matvec(direction)
        denominator = float(direction @ A_direction)
        if denominator <= 0.0 or not np.isfinite(denominator):
            breakdown = True
            break
        step = residual_squared / denominator
        x += step * direction
        residual -= step * A_direction
        new_residual_squared = float(residual @ residual)
        completed = iteration
        relative_residual = math.sqrt(max(new_residual_squared, 0.0)) / initial_norm

        if track_history:
            metric_residual, gap, a_error = _metric_values(
                problem,
                x,
                initial_norm,
                initial_gap,
                initial_a_error,
            )
            iterations.append(iteration)
            residuals.append(metric_residual)
            gaps.append(gap)
            a_errors.append(a_error)
            points.append(x.copy())
        if relative_residual <= tolerance:
            converged = True
            break
        if residual_squared <= 0.0 or not np.isfinite(new_residual_squared):
            breakdown = True
            break
        coefficient = new_residual_squared / residual_squared
        direction = residual + coefficient * direction
        residual_squared = new_residual_squared

    final_relative = float(np.linalg.norm(problem.b - problem.matvec(x)) / initial_norm)
    return SolverResult(
        method="Conjugate Gradient",
        x=x,
        iterations=completed,
        matvecs=completed + 1,
        converged=converged,
        breakdown=breakdown,
        relative_residual=final_relative,
        iteration_history=np.asarray(iterations, dtype=int),
        residual_history=np.asarray(residuals, dtype=float),
        objective_gap_history=np.asarray(gaps, dtype=float),
        a_error_history=np.asarray(a_errors, dtype=float),
        path=np.asarray(points, dtype=float),
    )


def _stationary_result(method: str, x: np.ndarray) -> SolverResult:
    return SolverResult(
        method=method,
        x=x.copy(),
        iterations=0,
        matvecs=1,
        converged=True,
        breakdown=False,
        relative_residual=0.0,
        iteration_history=np.asarray([0], dtype=int),
        residual_history=np.asarray([0.0]),
        objective_gap_history=np.asarray([0.0]),
        a_error_history=np.asarray([0.0]),
        path=np.asarray([x.copy()]),
    )


def run_condition_sweep(
    condition_numbers: Sequence[float],
    *,
    dimension: int,
    trials: int,
    tolerance: float,
    max_iterations: int,
    seed: int,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    """Run matched diagonal-spectrum problems over condition numbers."""
    seed_sequences = np.random.SeedSequence(seed).spawn(
        len(condition_numbers) * trials
    )
    trial_rows: List[Dict[str, object]] = []
    seed_index = 0
    for condition_number in condition_numbers:
        for trial in range(1, trials + 1):
            rng = np.random.default_rng(seed_sequences[seed_index])
            seed_index += 1
            problem = make_spd_problem(
                dimension,
                condition_number,
                rng,
                rotate=False,
            )
            results = (
                conjugate_gradient(
                    problem,
                    tolerance=tolerance,
                    max_iterations=max_iterations,
                    track_history=False,
                ),
                steepest_descent(
                    problem,
                    tolerance=tolerance,
                    max_iterations=max_iterations,
                    track_history=False,
                ),
            )
            for result in results:
                trial_rows.append(
                    {
                        "condition_number": condition_number,
                        "dimension": dimension,
                        "trial": trial,
                        "method": result.method,
                        "converged": result.converged,
                        "breakdown": result.breakdown,
                        "iterations": result.iterations,
                        "matvecs": result.matvecs,
                        "budgeted_iterations": (
                            result.iterations
                            if result.converged
                            else max_iterations + 1
                        ),
                        "final_relative_residual": result.relative_residual,
                    }
                )

    summary_rows: List[Dict[str, object]] = []
    for condition_number in condition_numbers:
        for method in METHODS:
            rows = [
                row
                for row in trial_rows
                if row["condition_number"] == condition_number
                and row["method"] == method
            ]
            converged_iterations = np.asarray(
                [row["iterations"] for row in rows if row["converged"]],
                dtype=float,
            )
            budgeted = np.asarray(
                [row["budgeted_iterations"] for row in rows],
                dtype=float,
            )
            residuals = np.asarray(
                [row["final_relative_residual"] for row in rows],
                dtype=float,
            )
            summary_rows.append(
                {
                    "condition_number": condition_number,
                    "dimension": dimension,
                    "method": method,
                    "trials": trials,
                    "converged_count": int(sum(bool(row["converged"]) for row in rows)),
                    "convergence_rate": float(
                        np.mean([bool(row["converged"]) for row in rows])
                    ),
                    "median_iterations_converged": (
                        float(np.median(converged_iterations))
                        if len(converged_iterations)
                        else math.nan
                    ),
                    "median_budgeted_iterations": float(np.median(budgeted)),
                    "q25_budgeted_iterations": float(np.quantile(budgeted, 0.25)),
                    "q75_budgeted_iterations": float(np.quantile(budgeted, 0.75)),
                    "median_final_relative_residual": float(np.median(residuals)),
                }
            )
    return trial_rows, summary_rows


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "axes.grid": True,
            "grid.alpha": 0.18,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _pdf_complete(data: bytes) -> bool:
    if len(data) < 100 or not data.startswith(b"%PDF-"):
        return False
    if b"%%EOF" not in data[-2048:]:
        return False
    matches = list(re.finditer(rb"startxref\s+(\d+)\s+%%EOF", data[-4096:]))
    if not matches:
        return False
    offset = int(matches[-1].group(1))
    return 0 <= offset < len(data)


def _png_complete(data: bytes) -> bool:
    return (
        len(data) >= 100
        and data.startswith(b"\x89PNG\r\n\x1a\n")
        and data.endswith(b"\x00\x00\x00\x00IEND\xaeB`\x82")
    )


def _write_bytes_atomic(path: Path, data: bytes, validator) -> None:
    if not validator(data):
        raise OSError(f"generated file is incomplete: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.stem}.",
        suffix=path.suffix,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        if not validator(temporary.read_bytes()):
            raise OSError(f"file write is incomplete: {path}")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _save_figure(
    fig: plt.Figure,
    output_dir: Path,
    stem: str,
    also_png: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        pdf_buffer = io.BytesIO()
        fig.savefig(pdf_buffer, format="pdf", bbox_inches="tight")
        _write_bytes_atomic(
            output_dir / f"{stem}.pdf",
            pdf_buffer.getvalue(),
            _pdf_complete,
        )
        if also_png:
            png_buffer = io.BytesIO()
            fig.savefig(
                png_buffer,
                format="png",
                dpi=300,
                bbox_inches="tight",
            )
            _write_bytes_atomic(
                output_dir / f"{stem}.png",
                png_buffer.getvalue(),
                _png_complete,
            )
    finally:
        plt.close(fig)


def plot_representative_convergence(
    cg_result: SolverResult,
    sd_result: SolverResult,
    condition_number: float,
) -> plt.Figure:
    """Plot convergence against comparable matrix-vector work."""
    floor = 1e-20
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.5))
    for result, color in ((cg_result, CG_COLOR), (sd_result, SD_COLOR)):
        work = result.iteration_history + 1
        axes[0].semilogy(
            work,
            np.maximum(result.residual_history, floor),
            color=color,
            linewidth=2.2,
            label=result.method,
        )
        axes[1].semilogy(
            work,
            np.maximum(result.objective_gap_history, floor),
            color=color,
            linewidth=2.2,
            label=result.method,
        )
    axes[0].set_title("Linear-system residual")
    axes[0].set_xlabel("Matrix–vector products")
    axes[0].set_ylabel(r"Relative residual $\|b-Ax_k\|_2/\|b-Ax_0\|_2$")
    axes[1].set_title("Quadratic objective gap")
    axes[1].set_xlabel("Matrix–vector products")
    axes[1].set_ylabel(r"Relative gap $(f(x_k)-f_*)/(f(x_0)-f_*)$")
    axes[0].legend(loc="best")
    fig.suptitle(
        rf"Representative SPD problem ($\kappa_2(A)={condition_number:g}$)",
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.01,
        "Same initial point and stopping tolerance; steepest descent uses exact line search.",
        ha="center",
        color=NEUTRAL_COLOR,
        fontsize=9.2,
    )
    fig.tight_layout(rect=(0.0, 0.045, 1.0, 0.92))
    return fig


def plot_condition_sweep(
    summary_rows: Sequence[Mapping[str, object]],
    max_iterations: int,
) -> plt.Figure:
    """Plot iteration demand and convergence rate versus conditioning."""
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.5))
    for method, color in (("Conjugate Gradient", CG_COLOR), ("Steepest Descent", SD_COLOR)):
        rows = sorted(
            (row for row in summary_rows if row["method"] == method),
            key=lambda row: float(row["condition_number"]),
        )
        condition = np.asarray([row["condition_number"] for row in rows], dtype=float)
        median_iterations = np.asarray(
            [row["median_budgeted_iterations"] for row in rows],
            dtype=float,
        )
        rate = 100.0 * np.asarray(
            [row["convergence_rate"] for row in rows],
            dtype=float,
        )
        axes[0].plot(
            condition,
            median_iterations,
            marker="o",
            linewidth=2.2,
            color=color,
            label=method,
        )
        axes[1].plot(
            condition,
            rate,
            marker="o",
            linewidth=2.2,
            color=color,
            label=method,
        )
    axes[0].axhline(
        max_iterations + 1,
        color=NEUTRAL_COLOR,
        linestyle=":",
        linewidth=1.0,
        label="Budget exhausted",
    )
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel(r"Condition number $\kappa_2(A)$")
    axes[0].set_ylabel("Median iterations (non-convergence = budget + 1)")
    axes[0].set_title("Work needed to meet tolerance")
    axes[1].set_xscale("log")
    axes[1].set_ylim(-3.0, 103.0)
    axes[1].set_xlabel(r"Condition number $\kappa_2(A)$")
    axes[1].set_ylabel("Trials converged (%)")
    axes[1].set_title("Reliability within a shared budget")
    axes[0].legend(loc="best")
    axes[1].legend(loc="best")
    fig.suptitle(
        "Conditioning affects steepest descent much more severely",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    return fig


def plot_two_dimensional_trajectory(
    problem: QuadraticProblem,
    cg_result: SolverResult,
    sd_result: SolverResult,
) -> plt.Figure:
    """Plot both paths over contours of the same two-dimensional objective."""
    if problem.dimension != 2:
        raise ValueError("trajectory plot requires dimension two")
    matrix = problem.dense_matrix()
    all_points = np.vstack((cg_result.path, sd_result.path, problem.x_star[None, :]))
    lower = np.min(all_points, axis=0)
    upper = np.max(all_points, axis=0)
    padding = 0.18 * np.maximum(upper - lower, 1.0)
    x_values = np.linspace(lower[0] - padding[0], upper[0] + padding[0], 350)
    y_values = np.linspace(lower[1] - padding[1], upper[1] + padding[1], 350)
    xx, yy = np.meshgrid(x_values, y_values)
    dx = xx - problem.x_star[0]
    dy = yy - problem.x_star[1]
    gap = 0.5 * (
        matrix[0, 0] * dx**2
        + 2.0 * matrix[0, 1] * dx * dy
        + matrix[1, 1] * dy**2
    )
    positive = gap[gap > 0.0]
    levels = np.geomspace(
        max(float(np.quantile(positive, 0.01)), 1e-6),
        float(np.max(positive)),
        18,
    )

    fig, ax = plt.subplots(figsize=(7.1, 6.2))
    ax.contour(xx, yy, gap, levels=levels, colors="#B8BDC5", linewidths=0.7)
    for result, color in ((sd_result, SD_COLOR), (cg_result, CG_COLOR)):
        path = result.path
        marker_stride = max(1, len(path) // 24)
        ax.plot(
            path[:, 0],
            path[:, 1],
            color=color,
            linewidth=1.8,
            marker="o",
            markersize=3.3,
            markevery=marker_stride,
            label=f"{result.method} ({result.iterations} iterations)",
            zorder=3,
        )
    ax.scatter(
        [problem.x_star[0]],
        [problem.x_star[1]],
        marker="*",
        s=180,
        color="#009E73",
        edgecolor="white",
        linewidth=0.8,
        label="Exact minimizer",
        zorder=5,
    )
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_title(
        rf"Optimization paths on one SPD quadratic ($\kappa_2(A)={problem.condition_number:g}$)"
    )
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def _objective_gap_at_points(
    problem: QuadraticProblem,
    points: np.ndarray,
) -> np.ndarray:
    """Evaluate f(x)-f(x*) for one point or an array of row-wise points."""
    point_array = np.atleast_2d(np.asarray(points, dtype=float))
    if point_array.shape[1] != problem.dimension:
        raise ValueError("points have incompatible dimensions")
    errors = point_array - problem.x_star[None, :]
    matrix = problem.dense_matrix()
    return 0.5 * np.einsum("ij,jk,ik->i", errors, matrix, errors)


def plot_three_dimensional_surface(
    problem: QuadraticProblem,
    cg_result: SolverResult,
    sd_result: SolverResult,
) -> plt.Figure:
    """Lift the same optimization paths onto the quadratic objective surface."""
    if problem.dimension != 2:
        raise ValueError("surface plot requires dimension two")

    cg_gap = _objective_gap_at_points(problem, cg_result.path)
    sd_gap = _objective_gap_at_points(problem, sd_result.path)
    path_ceiling = max(float(np.max(cg_gap)), float(np.max(sd_gap)), 1e-12)
    z_ceiling = 1.08 * path_ceiling
    path_offset = 0.012 * z_ceiling

    eigenvalues, eigenvectors = np.linalg.eigh(problem.dense_matrix())
    radii = np.linspace(0.0, 1.0, 100)
    angles = np.linspace(0.0, 2.0 * np.pi, 220)
    radial_grid, angle_grid = np.meshgrid(radii, angles, indexing="ij")
    axis_scales = np.sqrt(2.0 * z_ceiling / eigenvalues)
    eigen_coordinates = np.column_stack(
        (
            axis_scales[0] * radial_grid.ravel() * np.cos(angle_grid.ravel()),
            axis_scales[1] * radial_grid.ravel() * np.sin(angle_grid.ravel()),
        )
    )
    surface_points = problem.x_star[None, :] + eigen_coordinates @ eigenvectors.T
    surface_x = surface_points[:, 0].reshape(radial_grid.shape)
    surface_y = surface_points[:, 1].reshape(radial_grid.shape)
    surface_z = z_ceiling * radial_grid**2

    lower = np.min(surface_points, axis=0)
    upper = np.max(surface_points, axis=0)
    x_values = np.linspace(lower[0], upper[0], 210)
    y_values = np.linspace(lower[1], upper[1], 210)
    xx, yy = np.meshgrid(x_values, y_values)
    grid_points = np.column_stack((xx.ravel(), yy.ravel()))
    zz = _objective_gap_at_points(problem, grid_points).reshape(xx.shape)

    fig = plt.figure(figsize=(9.2, 7.0))
    ax = fig.add_subplot(111, projection="3d", computed_zorder=False)
    ax.plot_surface(
        surface_x,
        surface_y,
        surface_z,
        cmap="Blues",
        alpha=0.38,
        linewidth=0.0,
        antialiased=True,
        rcount=100,
        ccount=110,
        zorder=1,
    )
    contour_levels = np.geomspace(max(z_ceiling * 0.015, 1e-6), z_ceiling, 10)
    ax.contour(
        xx,
        yy,
        zz,
        levels=contour_levels,
        zdir="z",
        offset=0.0,
        colors="#A7ADB5",
        linewidths=0.65,
        alpha=0.8,
        zorder=2,
    )

    for result, gap_values, color in (
        (sd_result, sd_gap, SD_COLOR),
        (cg_result, cg_gap, CG_COLOR),
    ):
        marker_stride = max(1, len(result.path) // 24)
        ax.plot(
            result.path[:, 0],
            result.path[:, 1],
            gap_values + path_offset,
            color=color,
            linewidth=2.4,
            marker="o",
            markersize=3.5,
            markevery=marker_stride,
            label=f"{result.method} ({result.iterations} iterations)",
            zorder=10,
        )

    ax.scatter(
        [problem.x_star[0]],
        [problem.x_star[1]],
        [0.055 * z_ceiling],
        marker="*",
        s=190,
        color="#009E73",
        edgecolor="white",
        linewidth=0.8,
        label="Exact minimizer",
        depthshade=False,
        zorder=12,
    )
    ax.set_xlabel(r"$x_1$", labelpad=8)
    ax.set_ylabel(r"$x_2$", labelpad=8)
    ax.set_zlabel("")
    ax.set_zlim(0.0, z_ceiling)
    ax.view_init(elev=30.0, azim=-55.0)
    ax.set_box_aspect((1.2, 1.0, 0.76))
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_alpha(0.0)
        axis._axinfo["grid"]["color"] = (0.72, 0.74, 0.77, 0.42)
    fig.suptitle(
        rf"Optimization paths on the quadratic surface ($\kappa_2(A)={problem.condition_number:g}$)",
        y=0.975,
        fontsize=14,
        fontweight="bold",
    )
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(0.045, 0.90),
    )
    fig.text(
        0.955,
        0.47,
        r"Objective gap $f(x)-f(x_*)$",
        rotation=90,
        va="center",
        ha="center",
    )
    fig.subplots_adjust(left=0.01, right=0.91, bottom=0.03, top=0.89)
    return fig


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    fieldnames: Optional[Sequence[str]] = None,
) -> None:
    if not rows and not fieldnames:
        raise ValueError(f"cannot infer CSV fields for {path}")
    fields = list(fieldnames) if fieldnames else list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _history_rows(results: Sequence[SolverResult]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for result in results:
        for index in range(len(result.iteration_history)):
            rows.append(
                {
                    "method": result.method,
                    "iteration": int(result.iteration_history[index]),
                    "matvecs": int(result.iteration_history[index] + 1),
                    "relative_residual": float(result.residual_history[index]),
                    "relative_objective_gap": float(
                        result.objective_gap_history[index]
                    ),
                    "relative_A_norm_error": float(result.a_error_history[index]),
                }
            )
    return rows


def _trajectory_rows(
    problem: QuadraticProblem,
    results: Sequence[SolverResult],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for result in results:
        gaps = _objective_gap_at_points(problem, result.path)
        for iteration, (point, gap) in enumerate(zip(result.path, gaps)):
            rows.append(
                {
                    "method": result.method,
                    "iteration": iteration,
                    "x1": float(point[0]),
                    "x2": float(point[1]),
                    "objective_gap": float(gap),
                }
            )
    return rows


def run_experiment(
    *,
    dimension: int,
    condition_numbers: Sequence[float],
    trials: int,
    tolerance: float,
    max_iterations: int,
    representative_condition: float,
    trajectory_condition: float,
    seed: int,
    output_dir: Path,
    also_png: bool,
) -> Dict[str, object]:
    """Run the standalone benchmark and create all outputs."""
    if dimension < 2 or trials < 1 or max_iterations < 1:
        raise ValueError("dimension, trials, and max_iterations are too small")
    if tolerance <= 0.0 or not np.isfinite(tolerance):
        raise ValueError("tolerance must be positive and finite")
    conditions = tuple(float(value) for value in condition_numbers)
    if not conditions or any(value < 1.0 or not np.isfinite(value) for value in conditions):
        raise ValueError("condition numbers must be finite and at least one")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _apply_style()

    representative_rng = np.random.default_rng(seed + 1)
    representative = make_spd_problem(
        dimension,
        representative_condition,
        representative_rng,
        rotate=True,
    )
    representative_cg = conjugate_gradient(
        representative,
        tolerance=tolerance,
        max_iterations=max_iterations,
        track_history=True,
    )
    representative_sd = steepest_descent(
        representative,
        tolerance=tolerance,
        max_iterations=max_iterations,
        track_history=True,
    )

    trajectory_problem = make_two_dimensional_problem(trajectory_condition)
    trajectory_eigenvalues, trajectory_eigenvectors = np.linalg.eigh(
        trajectory_problem.dense_matrix()
    )
    # Include both eigendirections, with the stiff-direction component scaled
    # to make the classic steepest-descent zig-zag visible without changing
    # the shared starting point used by the two methods.
    trajectory_error = (
        2.0 * trajectory_eigenvectors[:, 0]
        + 2.0
        / math.sqrt(float(trajectory_eigenvalues[-1]))
        * trajectory_eigenvectors[:, -1]
    )
    trajectory_start = trajectory_problem.x_star + trajectory_error
    trajectory_cg = conjugate_gradient(
        trajectory_problem,
        x0=trajectory_start,
        tolerance=min(tolerance, 1e-10),
        max_iterations=max_iterations,
        track_history=True,
    )
    trajectory_sd = steepest_descent(
        trajectory_problem,
        x0=trajectory_start,
        tolerance=min(tolerance, 1e-10),
        max_iterations=max_iterations,
        track_history=True,
    )

    trial_rows, summary_rows = run_condition_sweep(
        conditions,
        dimension=dimension,
        trials=trials,
        tolerance=tolerance,
        max_iterations=max_iterations,
        seed=seed + 2,
    )

    _save_figure(
        plot_representative_convergence(
            representative_cg,
            representative_sd,
            representative_condition,
        ),
        output_dir,
        "representative_convergence",
        also_png,
    )
    _save_figure(
        plot_two_dimensional_trajectory(
            trajectory_problem,
            trajectory_cg,
            trajectory_sd,
        ),
        output_dir,
        "quadratic_trajectory_2d",
        also_png,
    )
    _save_figure(
        plot_three_dimensional_surface(
            trajectory_problem,
            trajectory_cg,
            trajectory_sd,
        ),
        output_dir,
        "quadratic_surface_3d",
        also_png,
    )
    _save_figure(
        plot_condition_sweep(summary_rows, max_iterations),
        output_dir,
        "condition_number_sweep",
        also_png,
    )

    _write_csv(output_dir / "condition_number_trials.csv", trial_rows)
    _write_csv(output_dir / "condition_number_summary.csv", summary_rows)
    _write_csv(
        output_dir / "representative_history.csv",
        _history_rows((representative_cg, representative_sd)),
    )
    _write_csv(
        output_dir / "trajectory_2d.csv",
        _trajectory_rows(
            trajectory_problem,
            (trajectory_cg, trajectory_sd),
        ),
    )
    configuration: Dict[str, object] = {
        "problem_class": "symmetric positive-definite quadratic",
        "methods": list(METHODS),
        "steepest_descent_step": "exact line search",
        "initial_point": "zero except in the two-dimensional illustration",
        "dimension": dimension,
        "condition_numbers": list(conditions),
        "trials_per_condition": trials,
        "relative_residual_tolerance": tolerance,
        "maximum_iterations": max_iterations,
        "representative_condition_number": representative_condition,
        "trajectory_condition_number": trajectory_condition,
        "seed": seed,
        "work_measure": "matrix-vector products; one per iteration plus initial residual",
    }
    with (output_dir / "configuration.json").open("w", encoding="utf-8") as handle:
        json.dump(configuration, handle, indent=2)
        handle.write("\n")

    print("=" * 78)
    print("CONJUGATE GRADIENT VERSUS STEEPEST DESCENT")
    print("=" * 78)
    print(
        f"Representative problem: dimension={dimension}, "
        f"condition number={representative_condition:g}"
    )
    for result in (representative_cg, representative_sd):
        print(
            f"  {result.method:<20} | iterations={result.iterations:>6} | "
            f"converged={str(result.converged):<5} | "
            f"relative residual={result.relative_residual:.3e}"
        )
    print("\nCondition-number sweep")
    for condition_number in conditions:
        rows = [
            row
            for row in summary_rows
            if row["condition_number"] == condition_number
        ]
        cg_row = next(row for row in rows if row["method"] == "Conjugate Gradient")
        sd_row = next(row for row in rows if row["method"] == "Steepest Descent")
        ratio = float(sd_row["median_budgeted_iterations"]) / max(
            float(cg_row["median_budgeted_iterations"]),
            1.0,
        )
        print(
            f"  kappa={condition_number:>8g} | "
            f"CG median={cg_row['median_budgeted_iterations']:>8.1f}, "
            f"SD median={sd_row['median_budgeted_iterations']:>8.1f}, "
            f"SD/CG={ratio:>7.1f}x, "
            f"convergence={100*float(cg_row['convergence_rate']):.0f}%/"
            f"{100*float(sd_row['convergence_rate']):.0f}%"
        )
    print(f"\nOutputs written to {output_dir.resolve()}")
    return {
        "configuration": configuration,
        "trial_rows": trial_rows,
        "summary_rows": summary_rows,
        "representative_cg": representative_cg,
        "representative_sd": representative_sd,
    }


def run_self_tests() -> None:
    """Validate the algorithms and the fairness-critical invariants."""
    identity_problem = QuadraticProblem(
        b=np.array([1.0, -2.0, 3.0]),
        x_star=np.array([1.0, -2.0, 3.0]),
        condition_number=1.0,
        diagonal=np.ones(3),
    )
    identity_cg = conjugate_gradient(identity_problem, tolerance=1e-12)
    identity_sd = steepest_descent(identity_problem, tolerance=1e-12)
    if identity_cg.iterations != 1 or identity_sd.iterations != 1:
        raise AssertionError("both solvers must finish the identity problem in one step")

    problem_2d = make_two_dimensional_problem(25.0)
    start = np.array([-1.0, 1.0])
    cg = conjugate_gradient(problem_2d, x0=start, tolerance=1e-10)
    sd = steepest_descent(
        problem_2d,
        x0=start,
        tolerance=1e-10,
        max_iterations=5000,
    )
    if not cg.converged or cg.iterations > 2:
        raise AssertionError("CG must solve a 2D SPD system in at most two exact steps")
    if not sd.converged:
        raise AssertionError("steepest descent failed its self-test")
    if not np.allclose(cg.x, problem_2d.x_star, rtol=1e-8, atol=1e-8):
        raise AssertionError("CG solution is inaccurate")
    if not np.allclose(sd.x, problem_2d.x_star, rtol=1e-7, atol=1e-7):
        raise AssertionError("steepest-descent solution is inaccurate")
    for result in (cg, sd):
        differences = np.diff(result.objective_gap_history)
        if np.any(differences > 1e-10):
            raise AssertionError(f"{result.method} objective is not nonincreasing")
    print(
        "Self-tests passed: SPD validation, shared initialization, exact-line-search "
        "steepest descent, CG recurrence, convergence, and objective decrease."
    )


def _parse_condition_numbers(values: Sequence[str]) -> Tuple[float, ...]:
    parsed = tuple(float(value) for value in values)
    if not parsed:
        raise argparse.ArgumentTypeError("at least one condition number is required")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare Conjugate Gradient with exact-line-search steepest descent "
            "on reproducible SPD quadratic problems."
        )
    )
    parser.add_argument("--dimension", type=int, default=60)
    parser.add_argument(
        "--condition-numbers",
        nargs="+",
        default=(10.0, 100.0, 1000.0, 10000.0),
        metavar="KAPPA",
    )
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    parser.add_argument("--max-iterations", type=int, default=20000)
    parser.add_argument("--representative-condition", type=float, default=1000.0)
    parser.add_argument("--trajectory-condition", type=float, default=50.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("cg_vs_steepest_descent_results"),
    )
    parser.add_argument("--png", action="store_true")
    parser.add_argument("--self-test-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_self_tests()
    if args.self_test_only:
        return
    condition_numbers = _parse_condition_numbers(args.condition_numbers)
    run_experiment(
        dimension=args.dimension,
        condition_numbers=condition_numbers,
        trials=args.trials,
        tolerance=args.tolerance,
        max_iterations=args.max_iterations,
        representative_condition=args.representative_condition,
        trajectory_condition=args.trajectory_condition,
        seed=args.seed,
        output_dir=args.output_dir,
        also_png=args.png,
    )


if __name__ == "__main__":
    main()
