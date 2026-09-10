# Frozen protocol: 2002 pharmaceutical-tablet NIR study

Protocol version: 1.0.0  
Freeze date: 2026-09-08  
Status: analysis-protocol freeze completed before this workflow fits any of the
four thesis procedures to the 460-tablet benchmark. This is not described as a
formal preregistration because benchmark responses and preliminary ordinary
PCR/PLS performance were inspected during dataset selection.

This protocol governs the empirical comparison of `CG-FPLS-code`, `Raw FPLS`,
`Arnoldi FPLS`, and `FPCR`. It is deliberately separate from the frozen
simulation and inference studies. A preliminary dataset-selection exercise in
an earlier conversation inspected benchmark performance from ordinary PCR and
PLS. The test set is therefore a **designated benchmark**, not a prospectively
blind holdout. Structural quality checks (dimensions, response ranges,
finiteness, duplicates, row order, and spectral spikes) have also inspected the
whole archive, but no result from one of the four thesis procedures has been
used to choose this protocol.

## Data and estimand

- Source archive: Eigenvector Research, *2002 Pharmaceutical Tablet NIR
  Shootout*, `nir_shootout_2002.mat_.zip`.
- Source URL:
  <https://eigenvector.com/wp-content/uploads/2019/06/nir_shootout_2002.mat_.zip>
- Archive access date recorded for this study: 2026-09-08.
- ZIP SHA-256:
  `04c03a4314ab8e8f63e0ecc788b657b5d1b5450791ce465ad41ad90cb5d52125`.
- MAT-member SHA-256:
  `129a32ec9e194e568cbd96a200c88aabeb347156081b5dfbb0120372fdd6c22a`.
- The archive contains 155 calibration, 40 validation, and 460 test tablets,
  hence 655 complete rows for each instrument. The catalogue's count of 654 is
  retained as a metadata discrepancy; no row will be deleted.
- There are no global tablet identifiers. Immutable study identifiers are
  created from archive location and one-based row order, for example
  `calibrate:001`. Instrument 1 and instrument 2 rows are treated as paired
  measurements, never as independent tablets.
- The primary predictor is the instrument-1 spectrum at all 650 equally spaced
  wavelengths, 600--1898 nm in 2-nm increments. Wavelength units come from the
  external dataset description; they are not embedded in the MAT object.
- The sole primary response is assay, column 3 of the response matrices.
- Calibration and validation are concatenated, in archive order, into a
  195-tablet development set. The 460 supplied test tablets form the designated
  benchmark.
- The empirical inner product is the equal-weight average
  `T^{-1} sum_j f_j g_j`. This is the existing estimator convention and is
  appropriate for the uniform grid after normalising its domain. No test-set
  information is used to estimate quadrature weights.

## Primary modelling rule

The primary spectra are otherwise unprocessed. Within every training sample,
each wavelength and the assay response are mean-centred. The fitted training
means define the intercept and are applied unchanged to the corresponding
validation or benchmark observations. There is no variance scaling, wavelength
selection, smoothing, derivative, SNV, MSC, or baseline correction in the
primary analysis.

All four procedures reuse the frozen algebra in `four_method_core.py`:

1. Raw FPLS uses the unscaled raw Krylov sequence and response-space normal
   equations. Its component count minimises five-fold prediction MSE.
2. Arnoldi FPLS uses CGS2 orthogonalisation and response-space QR. Its component
   count independently minimises prediction MSE on exactly the same five folds
   as Raw FPLS. Copied path entries beyond the effective Arnoldi dimension are
   ineligible.
3. FPCR uses spectral cutoff and its existing response-space GCV rule.
4. CG-FPLS-code uses the released-code conjugate-residual recurrence and its
   existing discrepancy rule, with `tau=1.01`, `delta=0.10`, and its internal
   moment-GCV FPCR variance pilot.

The maximum requested path length is 70. Requested paths are capped by the
processed-grid limit and the relevant centred training-sample limit, including
the smallest inner-training sample for Raw and Arnoldi. Method-specific
numerical rank or recurrence breakdown further determines the eligible or
effective path. Invalid or nonfinite Raw-FPLS candidates are excluded and reported;
they are never repaired using a pseudoinverse, ridge term, or unreported
fallback. A complete method failure remains a reported failure rather than a
dropped fold. Component counts are not interpreted as equal effective degrees
of freedom across methods.

This is a comparison of complete procedures, including their native stopping
rules, rather than an attempt to isolate the regulariser while holding all
tuning rules equal. The prespecified focal contrast is
`RMSE(Raw FPLS) - RMSE(Arnoldi FPLS)`. Contrasts of CG-FPLS-code and FPCR with
Arnoldi FPLS are secondary; the other pairwise contrasts are exploratory.

## Development-only evaluation

- Five repeats of five-fold outer cross-validation are generated on the 195
  development tablets using deterministic, source-split-specific,
  assay-balanced folds. Within calibration and validation separately, rows are
  sorted by assay, divided into groups of five, and randomly assigned one fold
  label within each group. Every outer fold consequently contains 31
  calibration and 8 validation tablets.
- Within each outer training sample, Raw and Arnoldi use a newly generated
  assay-balanced five-fold inner partition. The same inner indices are used for
  both procedures. FPCR and CG retain their native training-sample rules.
- One complete out-of-fold metric is computed per repeat. The 25 folds and the
  975 repeated tablet predictions are not treated as independent observations.
  Fold distributions describe tuning and numerical stability only.
- A prespecified order-dependence sensitivity uses five source-specific
  contiguous outer folds: 31 consecutive calibration rows and 8 consecutive
  validation rows per fold. Leakage-safe inner tuning is performed on each
  remaining training sample.
- An original-split sensitivity trains and tunes on the 155 calibration
  tablets and evaluates once on the untouched 40 validation tablets. Because
  validation assay values occupy a very narrow range, RMSE and MAE receive more
  weight than validation-set R-squared.
- Master seed: 20260908. All fold indices, rather than only seeds, are archived.

## Designated benchmark evaluation

After development checks and code review, each procedure is refitted once on
all 195 development tablets. Raw and Arnoldi use one frozen, common,
assay-balanced five-fold partition for component selection. No benchmark
prediction or outcome is used for tuning, preprocessing, method repair, or
choice among analysis variants.

The primary endpoint is benchmark RMSE. Secondary metrics are MAE, signed
prediction bias (`prediction - observation`), and

`R^2 = 1 - SSE / sum_i (y_i - mean(y_benchmark))^2`.

R-squared is contextual: on a common benchmark it ranks methods exactly as
RMSE does, and it is not comparable across the narrow validation set and the
broader benchmark.

Conditional benchmark-tablet uncertainty is quantified by 10,000 paired
nonparametric bootstrap resamples. Every draw uses the same tablet indices for
all four methods. Percentile 95% intervals are reported for method metrics and
prespecified contrasts. Models are not refitted within this bootstrap. A
circular moving-block bootstrap with block length 10 is the principal
order-dependence sensitivity; lengths 5 and 20 assess that choice. The exact
resampling matrices are archived. These intervals do not include uncertainty
from development-sample selection or tuning.

## Prespecified spectral and instrument sensitivities

Only the centred full-spectrum instrument-1 analysis is primary. The following
are labelled sensitivities regardless of their performance:

1. Instrument 1, wavelengths 600--1798 nm inclusive, then fold-local centring.
   This removes the visibly spike-prone high-wavelength tail.
2. Instrument 1, fixed row-wise Savitzky--Golay smoothing (15-point window,
   cubic polynomial, derivative order 0, interpolation boundary mode), then
   fold-local centring.
3. Instrument 2, all 650 wavelengths with the primary centring rule. It uses
   the same tablets and is a paired robustness analysis, not a replication.

Each sensitivity is trained only on the development observations and is
evaluated on the corresponding benchmark rows. It cannot replace the primary
result. The primary paired bootstrap is not redefined after seeing any
sensitivity result.

## Outputs and interpretation limits

The final archive must contain resolved configuration and checksums; exact
outer, inner, and bootstrap indices; observation-level predictions; fold and
aggregate losses; full tuning curves; selected and effective dimensions;
Raw-FPLS basis/design conditioning; Arnoldi orthogonality and breakdown
diagnostics; CG stopping diagnostics; runtimes; coefficients represented as
original-input linear weights plus intercepts; software, BLAS/LAPACK, platform,
and Git metadata; figures; and an artifact checksum manifest.

The frozen main displays are: (i) spectra and assay-distribution overview,
(ii) a four-panel observed-versus-predicted benchmark plot with a one-to-one
line, (iii) benchmark RMSE and paired-bootstrap intervals, and (iv) selected
component distributions in development cross-validation. Appendix displays
show residuals, Raw/Arnoldi numerical curves, and sensitivity results. Frozen
tables report primary benchmark metrics, the three declared contrasts,
development-repeat metrics and selection stability, the original-split check,
and all prespecified sensitivities.

Claims concern prediction and numerical stability in this dataset. The true
slope is unknown, so slope ISE is not evaluated. The oracle simulation-inference
machinery is not transferred to these data. The archive contains no explicit
open-data licence; the raw MAT file will not be repackaged with derived results.
The source, Hopkins (2003), access date, and checksums will be cited.

Any deviation after the freeze must be recorded as a versioned amendment made
before the affected result is computed. Post-result analyses must be labelled
exploratory and cannot silently alter the primary analysis.
