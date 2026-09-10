# Final corrected thesis and computational project

The complete project contains the corrected thesis, its compiled PDF, original
Python programs, frozen simulation and empirical results, the supplied tablet
dataset, and presentation scripts for the final figures.

## Layout

```text
Regularisation_FLR_thesis_FINAL/
  thesis/                         LaTeX sources, figures, main.pdf, build.py
  empirical_application/
    code/                         Exact frozen empirical source snapshots
    dataset/                      Original supplied tablet ZIP
    frozen_results/
      tablet_nir_study_FINAL/      Predictions, paths, resampling, provenance
  simulation_studies/
    Functional_Regression/        Original simulation programs
      four_method_study_FINAL_R5000/
      babii_inference_results_R5000/
      cg_vs_steepest_descent_results/
  FINAL_CORRECTIONS.md
  FILE_MANIFEST_SHA256.txt
```

## Generate the thesis PDF

For Overleaf, upload the separate thesis-source ZIP, or the entire contents of
`thesis/`, and select `main.tex` with pdfLaTeX. Recompile from scratch if reusing
an older Overleaf project. Every required figure is already supplied; Python
and the numerical archives are not needed to compile the document.

For a local installation, run from `thesis/`:

```bash
python build.py
```

The helper selects a supported bibliography backend and removes stale build
files before compiling. See `thesis/README_BUILD.md` for manual commands.
Generated bibliography and auxiliary files are deliberately excluded from
the source ZIPs, preventing the earlier BibTeX/BibLaTeX `main.bbl` conflict.

Complete `thesis/author_details.tex` with the author's personal and university
information before submission. An abstract is included. Add a declaration
only using the wording and requirements applicable to your institution.

## Regenerate the final figures

From `thesis/`, install NumPy, SciPy, pandas and Matplotlib in your Python
environment, then run:

```bash
python -B figure_scripts/make_readable_estimation_figures.py \
  --results ../simulation_studies/Functional_Regression/four_method_study_FINAL_R5000 \
  --output-dir figures
python -B figure_scripts/make_readable_supporting_figures.py \
  --code-root ../simulation_studies/Functional_Regression
python -B figure_scripts/make_tablet_thesis_figures.py \
  --results ../empirical_application/frozen_results/tablet_nir_study_FINAL \
  --data-archive ../empirical_application/dataset/nir_shootout_2002.mat_.zip \
  --output-dir figures
```

These final presentation scripts use frozen results and deterministic design
illustrations. They do not rerun the Monte Carlo studies or fit the empirical
models. Original programs and original archived figures remain available
separately in `simulation_studies/Functional_Regression/`.

## Inspect or replay the numerical studies

See `thesis/REPRODUCIBILITY.md` for simulation commands and design details.
The simulation archives retain their original contents; their provenance is
less extensive than that of the empirical archive. Use a new output directory
for any replay and retain the frozen archives as the reference.

For Chapter 8 and Appendix D, run from `empirical_application/code/` using
Python 3.12:

```bash
python -m pip install -r requirements-spectroscopy.txt
python run_tablet_nir_study.py \
  --stage full \
  --archive ../dataset/nir_shootout_2002.mat_.zip \
  --output ../tablet_nir_study_FINAL_replay
```

The replay output directory must not already exist. The empirical code folder
contains the exact nine source/configuration files from the frozen run.

## Integrity

From the complete project root, run `sha256sum -c FILE_MANIFEST_SHA256.txt`.
The original empirical archive also has its own unchanged manifest: from
`empirical_application/frozen_results/tablet_nir_study_FINAL/`, run
`sha256sum -c checksums.sha256`.

The supplied source ZIP has SHA-256
`04c03a4314ab8e8f63e0ecc788b657b5d1b5450791ce465ad41ad90cb5d52125`.
Its contained MAT file has SHA-256
`129a32ec9e194e568cbd96a200c88aabeb347156081b5dfbb0120372fdd6c22a`.

The tablet archive is included as supplied by the author for this project.
Its original provenance and the absence of an explicit licence in the supplied
archive are recorded in the thesis and empirical protocol.
