"""Run the entire refactored four-method study with one command.

The flow is fixed and sequential:

1. validate the numerical core and run the Monte Carlo simulation;
2. create the three simulation-setup figures;
3. create the single baseline Krylov-geometry illustration; and
4. replay the frozen archive and create all thesis result figures.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


def _run_stage(label: str, command: List[str], script_directory: Path) -> None:
    print("\n" + "=" * 72, flush=True)
    print(label, flush=True)
    print("=" * 72, flush=True)
    subprocess.run(command, cwd=script_directory, check=True)


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the complete four-method simulation and figure pipeline."
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
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("four_method_study_FINAL_R5000"),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_arguments()
    if not 1 <= args.jobs <= 3:
        raise ValueError("jobs must be between one and three")
    if args.basis_size < 5 or args.grid_size < args.basis_size:
        raise ValueError("require basis-size >= 5 and grid-size >= basis-size")

    script_directory = Path(__file__).resolve().parent
    output_root = args.output_root.expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(
            f"output root already exists: {output_root}\n"
            "Choose a new --output-root so results from different runs cannot mix."
        )
    output_root.mkdir(parents=True, exist_ok=False)

    simulation_dir = output_root / "simulation_results"
    setup_dir = output_root / "simulation_setup_figures"
    geometry_dir = output_root / "krylov_geometry"
    thesis_dir = output_root / "thesis_figures"
    python = sys.executable

    _run_stage(
        "STAGE 1 OF 4: FOUR-METHOD SIMULATION",
        [
            python,
            str(script_directory / "run_four_method_simulation.py"),
            "--replications",
            str(args.replications),
            "--n",
            str(args.n),
            "--basis-size",
            str(args.basis_size),
            "--grid-size",
            str(args.grid_size),
            "--m-max",
            str(args.m_max),
            "--folds",
            str(args.folds),
            "--noise-sd",
            str(args.noise_sd),
            "--tau",
            str(args.tau),
            "--delta",
            str(args.delta),
            "--seed",
            str(args.seed),
            "--rank-tolerance",
            str(args.rank_tolerance),
            "--extreme-tail-multiple",
            str(args.extreme_tail_multiple),
            "--output-dir",
            str(simulation_dir),
        ],
        script_directory,
    )

    display_limit = min(20, args.basis_size)
    spectrum_limit = min(30, args.basis_size)
    _run_stage(
        "STAGE 2 OF 4: SIMULATION-SETUP FIGURES",
        [
            python,
            str(script_directory / "make_simulation_setup_figures.py"),
            "--basis-size",
            str(args.basis_size),
            "--grid-size",
            str(args.grid_size),
            "--noise-sd",
            str(args.noise_sd),
            "--seed",
            str(args.seed),
            "--coefficient-limit",
            str(display_limit),
            "--spectrum-limit",
            str(spectrum_limit),
            "--signal-limit",
            str(display_limit),
            "--output-dir",
            str(setup_dir),
            "--png",
        ],
        script_directory,
    )

    geometry_components = min(12, args.basis_size, args.grid_size)
    _run_stage(
        "STAGE 3 OF 4: BASELINE KRYLOV-GEOMETRY FIGURE",
        [
            python,
            str(script_directory / "make_krylov_geometry_figure.py"),
            "--basis-size",
            str(args.basis_size),
            "--grid-size",
            str(args.grid_size),
            "--components",
            str(geometry_components),
            "--rank-tolerance",
            str(args.rank_tolerance),
            "--edge-threshold",
            "0.20",
            "--output-dir",
            str(geometry_dir),
        ],
        script_directory,
    )

    _run_stage(
        "STAGE 4 OF 4: THESIS RESULT FIGURES",
        [
            python,
            str(script_directory / "make_thesis_simulation_figures.py"),
            "--results-dir",
            str(simulation_dir),
            "--output-dir",
            str(thesis_dir),
            "--jobs",
            str(args.jobs),
        ],
        script_directory,
    )

    print("\n" + "=" * 72)
    print("COMPLETE FOUR-METHOD STUDY FINISHED SUCCESSFULLY")
    print("=" * 72)
    print(f"All outputs: {output_root}")
    print("Methods: CG-FPLS-code, Raw FPLS, Arnoldi FPLS, FPCR")
    print("Simulation setups: Model 1, Model 2, Model 3")


if __name__ == "__main__":
    main()
