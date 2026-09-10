# Reproducing the tablet-NIR empirical study

The empirical code is separate from the frozen simulation and inference
archives. It expects the original Eigenvector ZIP and validates both the ZIP
and contained MAT file against the hashes in `spectroscopy_config.json`. The
raw data are never copied into a result archive.

## Environment and tests

The recorded reference environment is Python 3.12 with the exact package
versions in `requirements-spectroscopy.txt`.

```bash
python run_tablet_nir_study.py --self-test-only
TABLET_NIR_ARCHIVE=/path/to/nir_shootout_2002.mat_.zip \
  python -m unittest -v test_tablet_nir_core.py
```

The attached-data test is skipped when `TABLET_NIR_ARCHIVE` is unset. It checks
the supplied archive's hashes, variables, dimensions, axes, finiteness, paired
instrument structure, and absence of duplicate spectra.

## Stages

Every command requires a new output path and refuses to overwrite an existing
directory.

Audit only, without loading benchmark variables:

```bash
python run_tablet_nir_study.py \
  --stage audit \
  --archive /path/to/nir_shootout_2002.mat_.zip \
  --output tablet_nir_audit
```

Complete development-only evaluation, again without loading benchmark
variables:

```bash
python run_tablet_nir_study.py \
  --stage development \
  --archive /path/to/nir_shootout_2002.mat_.zip \
  --output tablet_nir_development
```

Frozen full study, including the single designated-benchmark evaluation and
all figures/tables:

```bash
python run_tablet_nir_study.py \
  --stage full \
  --archive /path/to/nir_shootout_2002.mat_.zip \
  --output tablet_nir_study_FINAL
```

The full stage writes every bootstrap index before loading the benchmark
variables. It then repeats the development analysis, writes the final common
inner folds, fits each frozen procedure on the 195 development tablets, and
evaluates the corresponding 460 benchmark rows. The full-spectrum instrument-1
analysis remains primary regardless of sensitivity results.

## Integrity verification

From inside the completed result directory:

```bash
sha256sum -c checksums.sha256
```

The archive includes code, protocol, configuration, Git/worktree, software,
BLAS/LAPACK, exact fold/bootstrap, prediction, metric, coefficient,
preprocessing, runtime, tuning, and numerical-diagnostic provenance. A stage is
complete only when its `_SUCCESS.json` marker exists and `_FAILED.json` does
not.
