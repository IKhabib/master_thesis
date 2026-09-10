#!/usr/bin/env python3


from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


BLUE = "#2878B5"
ORANGE = "#D95F02"
GREEN = "#2A9D8F"
INK = "#202020"
GRID = "#D9D9D9"


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11.5,
            "axes.titlesize": 13.0,
            "axes.labelsize": 11.5,
            "legend.fontsize": 10.0,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "mathtext.fontset": "cm",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _finish_axis(ax: plt.Axes) -> None:
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-1.22, 1.22)
    ax.set_ylim(-1.22, 1.22)
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$", rotation=0, labelpad=9)
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    ax.axhline(0, color="#9A9A9A", linewidth=0.65, zorder=0)
    ax.axvline(0, color="#9A9A9A", linewidth=0.65, zorder=0)
    ax.grid(color=GRID, linewidth=0.55, alpha=0.55)


def _unit_balls(ax: plt.Axes) -> None:
    theta = np.linspace(0.0, 2.0 * np.pi, 900)

    diamond = np.array([[1, 0], [0, 1], [-1, 0], [0, -1], [1, 0]])
    square = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1], [1, 1]])

    ax.plot(
        diamond[:, 0],
        diamond[:, 1],
        color=BLUE,
        linewidth=2.5,
        label=r"$B_1^2$ (diamond)",
    )
    ax.plot(
        np.cos(theta),
        np.sin(theta),
        color=ORANGE,
        linewidth=2.5,
        label=r"$B_2^2$ (circle)",
    )
    ax.plot(
        square[:, 0],
        square[:, 1],
        color=GREEN,
        linewidth=2.5,
        label=r"$B_\infty^2$ (square)",
    )

    h = np.array([4.0 / 5.0, 3.0 / 5.0])
    ax.annotate(
        "",
        xy=h,
        xytext=(0, 0),
        arrowprops={"arrowstyle": "-|>", "color": INK, "lw": 2.0},
        zorder=5,
    )
    ax.scatter(*h, s=31, color=INK, zorder=6)
    ax.text(0.31, 0.39, r"$h=(4/5,3/5)$", fontsize=10.5)

    ax.text(
        -1.17,
        -1.17,
        r"$\|h\|_1=7/5$" "\n" r"$\|h\|_2=1$" "\n" r"$\|h\|_\infty=4/5$",
        ha="left",
        va="bottom",
        fontsize=10.2,
        bbox={"boxstyle": "round,pad=0.28", "fc": "white", "ec": "#AAAAAA", "alpha": 0.94},
    )
    ax.text(
        0.5,
        0.985,
        r"$B_1^2\subset B_2^2\subset B_\infty^2$",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=10.4,
        bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": "none", "alpha": 0.88},
    )
    ax.legend(loc="lower right", bbox_to_anchor=(0.99, 0.04), frameon=True, framealpha=0.94, borderpad=0.45)
    ax.set_title(r"(a) Unit balls and norm-induced distance", pad=9)
    _finish_axis(ax)


def _duality(ax: plt.Axes) -> None:
    theta = np.linspace(0.0, 2.0 * np.pi, 900)
    circle_x = np.cos(theta)
    circle_y = np.sin(theta)
    ax.fill(circle_x, circle_y, color=ORANGE, alpha=0.10, zorder=1)
    ax.plot(circle_x, circle_y, color=ORANGE, linewidth=2.6)

    u = np.array([4.0 / 5.0, 3.0 / 5.0])
    xx = np.linspace(-0.2, 1.22, 300)
    yy = (1.0 - u[0] * xx) / u[1]
    visible = (yy >= -1.22) & (yy <= 1.22)
    ax.plot(
        xx[visible],
        yy[visible],
        color=BLUE,
        linewidth=2.0,
        linestyle="--",
    )
    ax.annotate(
        "",
        xy=u,
        xytext=(0, 0),
        arrowprops={"arrowstyle": "-|>", "color": INK, "lw": 2.0},
        zorder=5,
    )
    ax.scatter(*u, s=36, color=INK, zorder=6)
    ax.text(
        u[0] - 0.04,
        u[1] - 0.17,
        r"$u$: point and norming vector",
        ha="right",
        va="top",
        fontsize=10.0,
    )
    ax.text(-0.78, 0.70, r"$B_2^2=(B_2^2)^\circ$", color=ORANGE, fontsize=10.7)

    ax.text(
        -1.17,
        -1.17,
        r"$(B_1^2)^\circ=B_\infty^2$" "\n"
        r"$(B_2^2)^\circ=B_2^2$" "\n"
        r"$(B_\infty^2)^\circ=B_1^2$",
        ha="left",
        va="bottom",
        fontsize=10.2,
        bbox={"boxstyle": "round,pad=0.28", "fc": "white", "ec": "#AAAAAA", "alpha": 0.94},
    )
    ax.text(
        0.5,
        0.985,
        r"$p^{-1}+q^{-1}=1,\qquad"
        r"\|y\|_q=\sup_{\|x\|_p\leq1}|y^\mathsf{T}x|$",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=10.0,
        bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": "none", "alpha": 0.88},
    )
    ax.set_title(r"(b) Duality and $p=2$ self-duality", pad=9)
    _finish_axis(ax)


def make_figure(output: Path) -> None:
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.95))
    _unit_balls(axes[0])
    _duality(axes[1])

    fig.subplots_adjust(left=0.075, right=0.985, top=0.89, bottom=0.255, wspace=0.27)
    bridge = (
        r"Orthonormal coefficients:  "
        r"$f=\sum_{j\geq1}f_j e_j\ \longleftrightarrow\ (f_j)\in\ell^2$"
        r"$\qquad\|f-g\|_{L^2}^{,2}=\sum_{j\geq1}|f_j-g_j|^2$"
        r"$\qquad\Lambda_\beta(f)=\langle f,\beta\rangle_{L^2}$"
    )
    fig.text(
        0.5,
        0.105,
        bridge,
        ha="center",
        va="center",
        fontsize=12.0,
        color=INK,
        bbox={"boxstyle": "round,pad=0.55", "fc": "#F4F5F6", "ec": "#A8ADB2", "lw": 0.9},
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format="pdf", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figures/lp_duality_geometry.pdf"),
        help="Destination vector-PDF path.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    make_figure(args.output)
    print(f"Wrote {args.output.resolve()}")
