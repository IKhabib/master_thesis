"""Unit and integration tests for the empirical tablet-NIR wrapper."""

from __future__ import annotations

import json
import os
import unittest
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter

import four_method_core as method_core
import tablet_nir_core as study_core


ROOT = Path(__file__).resolve().parent


def synthetic_data(seed: int = 1928) -> study_core.RegressionData:
    rng = np.random.default_rng(seed)
    n, p = 50, 18
    wavelengths = 600.0 + 2.0 * np.arange(p)
    X = rng.normal(size=(n, p))
    beta = rng.normal(size=p)
    y = 11.0 + X @ beta / p + rng.normal(scale=0.03, size=n)
    source = np.array(["calibrate"] * 40 + ["validate"] * 10)
    rows = np.concatenate((np.arange(1, 41), np.arange(1, 11)))
    return study_core.RegressionData(
        X=X,
        y=y,
        wavelengths=wavelengths,
        tablet_ids=np.array(
            [
                f"{split}:{row:03d}"
                for split, row in zip(source, rows, strict=True)
            ]
        ),
        source_split=source,
        source_row=rows,
        instrument=1,
        response_name="assay",
    )


class FoldTests(unittest.TestCase):
    def test_source_balanced_fold_coverage_and_counts(self) -> None:
        data = synthetic_data()
        folds = study_core.make_source_balanced_folds(
            data.y, data.source_split, 5, [20260908, 101, 1]
        )
        study_core.validate_folds(folds, len(data.y))
        for fold in folds:
            self.assertEqual(np.count_nonzero(data.source_split[fold] == "calibrate"), 8)
            self.assertEqual(np.count_nonzero(data.source_split[fold] == "validate"), 2)
        duplicate = study_core.make_source_balanced_folds(
            data.y, data.source_split, 5, [20260908, 101, 1]
        )
        for left, right in zip(folds, duplicate, strict=True):
            np.testing.assert_array_equal(left, right)

    def test_source_contiguous_folds(self) -> None:
        data = synthetic_data()
        folds = study_core.make_source_contiguous_folds(data.source_split, 5)
        study_core.validate_folds(folds, len(data.y))
        for fold in folds:
            cal = data.source_row[fold][data.source_split[fold] == "calibrate"]
            val = data.source_row[fold][data.source_split[fold] == "validate"]
            self.assertTrue(np.all(np.diff(cal) == 1))
            self.assertTrue(np.all(np.diff(val) == 1))


class PreprocessingTests(unittest.TestCase):
    def test_fold_local_centring_and_no_validation_leakage(self) -> None:
        data = synthetic_data()
        spec = study_core.PreprocessSpec("identity", "primary", 1, "identity")
        train = np.arange(40)
        first = study_core.fit_preprocessor(
            data.X[train], data.y[train], data.wavelengths, spec
        )
        changed = data.X.copy()
        changed[40:] += 5000.0
        second = study_core.fit_preprocessor(
            changed[train], data.y[train], data.wavelengths, spec
        )
        np.testing.assert_array_equal(first.x_mean_processed, second.x_mean_processed)
        self.assertEqual(first.y_mean, second.y_mean)
        np.testing.assert_allclose(
            first.transform_X(data.X[train]).mean(axis=0), 0.0, atol=1e-14
        )
        self.assertAlmostEqual(float(first.transform_y(data.y[train]).mean()), 0.0, 14)

    def test_trim_and_savgol_are_exact(self) -> None:
        rng = np.random.default_rng(32)
        wavelengths = 600.0 + 2.0 * np.arange(650)
        X = rng.normal(size=(4, 650))
        trim = study_core.PreprocessSpec(
            "trim", "sensitivity", 1, "identity", 1798.0
        )
        operator, retained = study_core.build_spectral_operator(wavelengths, trim)
        self.assertEqual(operator.shape, (650, 600))
        self.assertEqual(retained[-1], 1798.0)
        np.testing.assert_array_equal(X @ operator, X[:, :600])

        smooth = study_core.PreprocessSpec(
            "smooth", "sensitivity", 1, "savgol", None, 15, 3, 0, 2.0, "interp"
        )
        operator, retained = study_core.build_spectral_operator(wavelengths, smooth)
        expected = savgol_filter(
            X, 15, 3, deriv=0, delta=2.0, axis=1, mode="interp"
        )
        np.testing.assert_allclose(X @ operator, expected, rtol=1e-12, atol=1e-12)
        np.testing.assert_array_equal(retained, wavelengths)


class ProcedureTests(unittest.TestCase):
    def test_all_methods_and_original_scale_prediction_identity(self) -> None:
        data = synthetic_data()
        spec = study_core.PreprocessSpec("identity", "primary", 1, "identity")
        folds = study_core.make_source_balanced_folds(
            data.y, data.source_split, 5, [77, 3]
        )
        config = {
            "maximum_components": 8,
            "rank_tolerance": 1e-10,
            "cg_tau": 1.01,
            "cg_delta": 0.1,
            "cg_path_tolerance": 1e-12,
        }
        fits = study_core.fit_all_procedures(data, spec, folds, config)
        self.assertEqual(tuple(fit.method for fit in fits), method_core.METHODS)
        for fit in fits:
            self.assertEqual(fit.status, "ok", fit.error_message)
            direct = fit.intercept + data.X @ fit.linear_weights_original
            np.testing.assert_allclose(direct, fit.predict(data.X), rtol=1e-11, atol=1e-11)

    def test_arnoldi_padded_path_is_ineligible(self) -> None:
        path = method_core.FPLSPath(
            beta=np.zeros((4, 5)),
            fitted=np.zeros((8, 5)),
            valid=np.ones(5, dtype=bool),
            basis=np.zeros((4, 2)),
            basis_condition=np.ones(5),
            design_condition=np.ones(5),
            orthogonality_defect=np.zeros(5),
            effective_dimension=2,
        )
        np.testing.assert_array_equal(
            study_core._usable_arnoldi(path),
            np.array([True, True, False, False, False]),
        )

    def test_intercept_shift_equivariance(self) -> None:
        data = synthetic_data()
        shifted = study_core.replace(data, y=data.y + 37.0)
        spec = study_core.PreprocessSpec("identity", "primary", 1, "identity")
        folds = study_core.make_source_balanced_folds(
            data.y, data.source_split, 5, [91, 4]
        )
        config = {
            "maximum_components": 6,
            "rank_tolerance": 1e-10,
            "cg_tau": 1.01,
            "cg_delta": 0.1,
            "cg_path_tolerance": 1e-12,
        }
        original = study_core.fit_all_procedures(data, spec, folds, config)
        translated = study_core.fit_all_procedures(shifted, spec, folds, config)
        for left, right in zip(original, translated, strict=True):
            np.testing.assert_allclose(
                right.predict(data.X), left.predict(data.X) + 37.0, atol=2e-9
            )


class BootstrapTests(unittest.TestCase):
    def test_iid_and_circular_indices_are_deterministic(self) -> None:
        first = study_core.make_bootstrap_indices(37, 50, [1, 2, 3])
        second = study_core.make_bootstrap_indices(37, 50, [1, 2, 3])
        np.testing.assert_array_equal(first, second)
        circular = study_core.make_bootstrap_indices(
            37, 50, [1, 2, 4], block_length=7
        )
        self.assertEqual(circular.shape, (50, 37))
        signed = circular.astype(np.int64)
        within_block = (signed[:, 1:7] - signed[:, :6]) % 37
        np.testing.assert_array_equal(within_block, np.ones_like(within_block))

    def test_pairing_preserves_constant_prediction_difference(self) -> None:
        y = np.linspace(0.0, 1.0, 30)
        predictions = {"left": y + 1.0, "right": y - 1.0}
        indices = study_core.make_bootstrap_indices(30, 100, [4, 5, 6])
        draws = study_core.bootstrap_metric_draws(y, predictions, indices)
        np.testing.assert_allclose(draws["left|rmse"], draws["right|rmse"])
        np.testing.assert_allclose(draws["left|bias"], 1.0)
        np.testing.assert_allclose(draws["right|bias"], -1.0)

    def test_any_nonfinite_prediction_invalidates_method_bootstrap(self) -> None:
        y = np.linspace(0.0, 1.0, 20)
        valid = y.copy()
        invalid = y.copy()
        invalid[0] = np.nan
        indices = study_core.make_bootstrap_indices(20, 1000, [8, 9, 10])
        draws = study_core.bootstrap_metric_draws(
            y, {"valid": valid, "invalid": invalid}, indices
        )
        for metric in ("rmse", "mae", "bias", "r2"):
            self.assertTrue(np.all(np.isfinite(draws[f"valid|{metric}"])))
            self.assertTrue(np.all(np.isnan(draws[f"invalid|{metric}"])))
        point = {
            "valid": study_core.prediction_metrics(y, valid),
            "invalid": study_core.prediction_metrics(y, invalid),
        }
        rows = study_core.bootstrap_summary_rows(
            draws,
            point,
            confidence=0.95,
            scheme="iid",
            declared_contrasts=[("invalid - valid", "invalid", "valid")],
        )
        invalid_rows = [
            row
            for row in rows
            if row["method_left"] == "invalid"
        ]
        self.assertTrue(invalid_rows)
        self.assertTrue(all(row["finite_replicates"] == 0 for row in invalid_rows))
        self.assertTrue(all(np.isnan(row["ci_lower"]) for row in invalid_rows))


class AttachedArchiveTests(unittest.TestCase):
    @unittest.skipUnless(
        os.environ.get("TABLET_NIR_ARCHIVE"),
        "set TABLET_NIR_ARCHIVE to run attached-data validation",
    )
    def test_attached_archive_identity_and_structure(self) -> None:
        with (ROOT / "spectroscopy_config.json").open("r", encoding="utf-8") as stream:
            config = json.load(stream)
        archive = Path(os.environ["TABLET_NIR_ARCHIVE"])
        audit = study_core.audit_archive(
            archive, config["data"], include_benchmark=True
        )
        self.assertEqual(audit["status"], "passed")
        self.assertEqual(audit["observations"], 655)
        self.assertEqual(audit["wavelength_count"], 650)
        self.assertEqual(audit["duplicate_spectra"]["instrument_1"], 0)
        self.assertEqual(audit["duplicate_spectra"]["instrument_2"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
