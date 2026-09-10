# Reproducing the four-method simulation study

The thesis text focuses on the statistical design. This file records the
software-facing details needed to reproduce and audit the numerical study.
The complete final package places the simulation scripts and their original
result directories together in `../simulation_studies/Functional_Regression/`
relative to this thesis directory. Empirical code and results are in
`../empirical_application/`. The separate thesis-source ZIP includes everything
needed for LaTeX compilation; numerical archives are in the complete-project ZIP.

## Final figure presentation

The final thesis figures use larger labels and revised panel arrangements.
Use `make_readable_estimation_figures.py`,
`make_readable_supporting_figures.py`, and `make_tablet_thesis_figures.py` in
`figure_scripts/` to reproduce this final presentation. Their exact commands
are in the complete project's `PACKAGE_README.md`. These scripts read frozen
outputs or deterministic design calculations without rerunning the studies.
The commands below document the original numerical workflows and their
original figure outputs.

## Reproducing the $\ell^p$ geometry figure

The explanatory figure in Chapter 2 is independent of the simulation and
inference archives. From the thesis directory, run:

```bash
python figure_scripts/make_lp_duality_figure.py
```

The script requires NumPy and Matplotlib and writes the vector graphic
`figures/lp_duality_geometry.pdf`. Its construction is deterministic: it
plots the two-dimensional $p=1,2,\infty$ unit balls, the fixed displacement
$h=(4/5,3/5)$, the Euclidean supporting line at the same unit vector, and the
Parseval--Riesz coefficient bridge used in Chapter 2.

## Required Python files

Keep these six files in one directory:

1. `four_method_core.py`
2. `run_four_method_simulation.py`
3. `make_simulation_setup_figures.py`
4. `make_krylov_geometry_figure.py`
5. `make_thesis_simulation_figures.py`
6. `run_complete_four_method_study.py`

## Complete run

From that directory, run:

```bash
python run_complete_four_method_study.py --output-root four_method_study_REPLAY_R5000
```

The default study uses 5,000 replications for each of three models, training
and test sample sizes of 100, 100 generating basis coefficients, 200 grid
points, a maximum of 70 components, five folds, error standard deviation 1,
CG constants `tau=1.01` and `delta=0.1`, seed 2026, Arnoldi rank tolerance
`1e-10`, and extreme-tail multiple 100. The default output root is
`four_method_study_FINAL_R5000`; the command above uses a fresh replay directory
to preserve the supplied frozen results. The program refuses to overwrite an
existing root.

The orchestration program runs four stages in order: numerical self-tests and
simulation, setup figures, the Model 1 Krylov-geometry figure, and final thesis
figures. The archive contains the configuration, raw arrays,
replication-level losses and selected dimensions, numerical diagnostics, and
summary tables.

## Replay validation

The final plotting stage regenerates selected estimates from the recorded seed,
model, and component counts. It compares replayed ISE and MSPE with the frozen
archive before drawing distributional or bias--variance figures. The scaled
validation tolerance is `5e-9` for CG--FPLS, Arnoldi FPLS, and FPCR, and
`5e-3` for the deliberately platform-sensitive raw normal-equation
implementation. These are numerical audit tolerances, not statistical
significance levels. Numerically flagged raw observations remain in every
primary loss, complexity, and tail summary.

## Environment record

For a permanent computational archive, record:

- Python, NumPy, SciPy, and Matplotlib versions;
- operating system and hardware architecture;
- BLAS/LAPACK implementation; and
- the exact command and configuration file used.

Small cross-platform differences are most plausible for raw FPLS because its
ill-conditioned normal equations are intentionally retained as part of the
method being diagnosed.

## Reproducing the CG--FPLS inference study

The inference workflow is separate from the six-file estimation comparison.
It requires the standalone program `babii_inference_study.py`. The following
command reproduces the frozen thesis configuration with a fresh output name;
the original run used `babii_inference_results_R5000`. Ensure that the replay
directory does not already exist:

```bash
python babii_inference_study.py \
  --replications 5000 \
  --sample-size 100 \
  --power-sample-sizes 100 200 \
  --T 200 \
  --J 100 \
  --noise-sd 1 \
  --late-components 70 \
  --stopping-components 1 2 3 5 10 20 40 70 \
  --alpha-levels 0.01 0.05 0.10 \
  --power-alpha 0.05 \
  --delta-step 0.05 \
  --oracle-draws 50000 \
  --calibrations oracle \
  --power-calibration empirical \
  --power-direct \
  --confidence-calibration oracle \
  --confidence-components 10 \
  --confidence-basis-dimension 5 \
  --confidence-method analytic \
  --cg-engine stable \
  --basis-convention babii \
  --seed 2025 \
  --output-dir babii_inference_results_REPLAY_R5000
```

The resulting archive contains `configuration.json`, five CSV summaries, the
replication-level `inference_raw_results.npz` file, and five vector-PDF figures.
Null size and confidence sets use known-design (oracle) weighted chi-square
critical values. Power uses empirical finite-sample null quantiles and the
direct-moment statistic selected by `--power-direct`; it does not run 70 CG
iterations for every alternative replication. The confidence illustration
uses a ten-component fitted moment and the exact analytic envelope of an
unrestricted five-dimensional coefficient ellipsoid, not the optional
paper-grid routine.

All random streams derive deterministically from seed 2025, but they are not
all independent. The confidence experiment intentionally regenerates the
null-size samples; it evaluates those samples at `confidence_components=10`,
whereas the reported size uses `late_components=70`. The empirical power
critical value for `n=100` reuses the direct null statistics from the size
branch, and the same oracle draws supply every branch requiring oracle
calibration. Alternative power samples and stopping-diagnostic samples use
separate derived streams. These reuses are part of the frozen design rather
than accidental dependence.

The inference study's estimation-oriented stopping comparator caps its
variance pilot at 20 components and uses the stabilised moment-residual
engine. This differs from the 70-component pilot and short recurrence in the
four-method estimation study. It uses the same threshold formula with
the program constants
`tau=1.01` and `delta=0.1`, which are not command-line arguments. The stable
Krylov engine uses an internal least-squares relative tolerance of `1e-12` and
a Krylov-vector tolerance of `1e-13`; the maximum-rank reference uses relative
tolerance `1e-12`. These constants should be retained with the program because
the archived JSON records the selected engine but not these internal values.
The recorded `calibration_draws=2000` and `confidence_grid_points=20` fields are
inactive in this configuration: the former applies only to data-dependent
plug-in or multiplier calibration, and the latter only to the unselected
paper-grid confidence routine.

The archived oracle reference uses the continuous-design eigenvalues. At the
fixed evaluation grid, the exact direct-moment limiting weights instead come
from the eigenvalues of the grid covariance, as derived in Chapter 6. The
original reference draws and numerical results are retained; this correction
qualifies their interpretation rather than replacing their values.

The oracle calibration is a simulation benchmark based on population inputs
known from the generating design. It is neither feasible for an ordinary
observed dataset nor an exact finite-sample calibration. The program also
contains plug-in and multiplier options, but those options were not selected
in the frozen thesis run and no result should be attributed to them.

Chapter 7 inserts `stopping_diagnostics.pdf`, `null_calibration.pdf`,
`power_curves.pdf`, and `confidence_sets.pdf` in the main argument. The fifth
output, `calibration_comparison.pdf`, is retained in Appendix C because its
5% display duplicates the exact rejection rates reported in the main table.

## Reproducing the pharmaceutical-tablet spectroscopy study

The empirical protocol and complete run are frozen under
`../empirical_application/`. The source archive is Eigenvector Research's
`nir_shootout_2002.mat_.zip`, accessed on 2026-09-08.  Before running, verify:

```text
ZIP SHA-256  04c03a4314ab8e8f63e0ecc788b657b5d1b5450791ce465ad41ad90cb5d52125
MAT SHA-256  129a32ec9e194e568cbd96a200c88aabeb347156081b5dfbb0120372fdd6c22a
```

From `../empirical_application/code/`, install the pinned empirical
requirements and run:

```bash
python -m pip install -r requirements-spectroscopy.txt
python run_tablet_nir_study.py \
  --stage full \
  --archive ../dataset/nir_shootout_2002.mat_.zip \
  --output ../tablet_nir_study_FINAL_replay
```

The frozen invocation, including its original absolute paths, is stored in
`../empirical_application/frozen_results/tablet_nir_study_FINAL/provenance/run_invocation.json`
relative to the thesis directory. The primary
configuration is `spectroscopy_config.json`; the human-readable freeze is
`SPECTROSCOPY_PROTOCOL.md`.  The run uses master seed `20260908`, separate
deterministic streams for fold and bootstrap operations, and a one-thread
numerical environment.

The development stage must pass before benchmark fitting.  Its success record
includes `benchmark_variables_loaded=false`.  The completed archive contains:

- exact outer, inner, final-tuning, iid-bootstrap, and moving-block indices;
- observation-level development and benchmark predictions;
- fold, repeat, and benchmark metrics;
- complete tuning and numerical paths;
- selected coefficients and intercepts in original-input coordinates;
- fit status, runtime, rank, conditioning, orthogonality, discrepancy, and
  stopping diagnostics;
- resolved configuration, source snapshots, package and BLAS/LAPACK versions,
  platform, Git state, and run invocation; and
- deterministic tables, figures, a manifest, and whole-archive checksums.

Verify the completed frozen archive from its root with:

```bash
sha256sum -c checksums.sha256
```

All listed hashes pass in the thesis archive.  Independent replay reconstructs
benchmark metrics from stored predictions to within `1.8e-15`, development
metrics to within `1.6e-14`, and predictions from stored coefficients and
intercepts to within `6.3e-13`.  Every one of the 140 stored fits succeeds and
all predictions are finite.  Numerical flags are nevertheless retained:
finite prediction does not imply that the raw Krylov representation is well
conditioned.

The Chapter 8 figures are thesis-scale renderings of the frozen numeric
outputs. Appendix D reports the complete development, bootstrap, numerical,
and sensitivity record. The author-supplied source ZIP containing the MAT file
is included separately in `../empirical_application/dataset/`; it is not part
of the unchanged derived-results archive.
