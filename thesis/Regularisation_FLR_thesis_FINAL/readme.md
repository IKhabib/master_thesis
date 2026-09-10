# Regularisation in Functional Linear Regression

**Master's thesis — Khabibullo Ibadullaev**  
Supervisor: Prof. Dr. Dominik Liebl

This package contains the complete thesis, LaTeX sources, Python programs,
tablet dataset, and saved study results.

## Start here

- **Read the thesis:** open `thesis/main.pdf`.
- **Run commands:** open Terminal in the extracted `Regularisation_FLR_thesis_FINAL`
  folder, which contains the three folders below. All commands assume this location.
- **Requirements:** macOS/Linux shell; Python 3.12 for the analyses; a LaTeX
  installation with pdfLaTeX, Biber, and latexmk for the standard PDF build.

| Folder | Contents |
| --- | --- |
| `thesis/` | PDF, LaTeX sources, bibliography, figures, and build helper |
| `simulation_studies/Functional_Regression/` | Simulation and inference programs and reference results |
| `empirical_application/` | Empirical programs, supplied dataset, and reference results |

## 1. Compile the thesis

```bash
latexmk -cd -pdf thesis/main.tex
```

The result is `thesis/main.pdf`. From inside `thesis/`, the equivalent command is
`latexmk -pdf main.tex` — the input is the `.tex` file.

Alternatively, use the included helper, which clears stale auxiliary files and
selects an available bibliography backend:

```bash
python3 thesis/build.py
```

All required figures are included. Compiling the thesis does not run Python analyses.

## 2. Set up Python

Create the environment and install dependencies once:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r empirical_application/code/requirements-spectroscopy.txt
```

Use one numerical thread for the studies:

```bash
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
```

Run the following commands in this same terminal. In a new terminal, activate
`.venv` again and repeat the four `export` commands.

## 3. Run the complete simulation study

```bash
python -B simulation_studies/Functional_Regression/run_complete_four_method_study.py \
  --replications 5000 \
  --output-root rerun_results/simulation
```

Runs all four methods across three models, with 5,000 simulated datasets per model,
and generates numerical checks, summaries, diagnostics, and study figures.

## 4. Run the complete inference study

```bash
python -B simulation_studies/Functional_Regression/babii_inference_study.py \
  --replications 5000 \
  --sample-size 100 \
  --power-sample-sizes 100 200 \
  --late-components 70 \
  --calibrations oracle \
  --power-calibration empirical \
  --power-direct \
  --confidence-calibration oracle \
  --confidence-components 10 \
  --confidence-basis-dimension 5 \
  --confidence-method analytic \
  --cg-engine stable \
  --seed 2025 \
  --output-dir rerun_results/inference
```

Runs stopping diagnostics, null calibration, power analysis, and confidence sets,
including summaries and figures. These options select the study settings used
in the thesis; the remaining defaults also match those settings.

## 5. Run the complete empirical study

```bash
python -B empirical_application/code/run_tablet_nir_study.py \
  --stage full \
  --archive empirical_application/dataset/nir_shootout_2002.mat_.zip \
  --output rerun_results/empirical
```

Runs data checks, development evaluation, benchmark prediction, bootstrap
intervals, sensitivity analyses, and figure/table generation.

## Outputs and reproducibility

- Each entry-point script calls its supporting Python files automatically.
- New outputs are written to `rerun_results/simulation`, `rerun_results/inference`,
  and `rerun_results/empirical`. Use fresh output names for each rerun; simulation
  and empirical runs refuse existing directories, while inference may overwrite files.
- The complete simulation and inference studies are computationally intensive. The supplied
  PDF and reference results can be inspected immediately.
- The pinned dependencies match the empirical reference environment. Software
  and platform differences can affect rerun results, especially Raw FPLS.
  The thesis reports the supplied frozen reference results.

