"""Leakage-safe empirical wrappers for the pharmaceutical-tablet NIR study.

This module deliberately leaves :mod:`four_method_core` unchanged.  It adds
data validation, fold-local preprocessing, empirical tuning, prediction
metrics, and paired bootstrap utilities around the frozen four-method core.
"""

from __future__ import annotations

import hashlib
import io
import itertools
import math
import time
import zipfile
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.io import loadmat
from scipy.signal import savgol_filter

import four_method_core as method_core


SPLITS = ("calibrate", "validate", "test")
EXPECTED_METHODS = method_core.METHODS
NUMERICAL_EXCEPTIONS = (
    np.linalg.LinAlgError,
    FloatingPointError,
    OverflowError,
    ArithmeticError,
)


@dataclass(frozen=True)
class RegressionData:
    """One instrument, one response, and one or more source splits."""

    X: np.ndarray
    y: np.ndarray
    wavelengths: np.ndarray
    tablet_ids: np.ndarray
    source_split: np.ndarray
    source_row: np.ndarray
    instrument: int
    response_name: str

    def subset(self, indices: Sequence[int] | np.ndarray) -> "RegressionData":
        idx = np.asarray(indices, dtype=int)
        return replace(
            self,
            X=self.X[idx],
            y=self.y[idx],
            tablet_ids=self.tablet_ids[idx],
            source_split=self.source_split[idx],
            source_row=self.source_row[idx],
        )


@dataclass(frozen=True)
class PreprocessSpec:
    """A fixed spectral transformation followed by learned mean centring."""

    name: str
    role: str
    instrument: int
    transform: str
    wavelength_max_nm: float | None = None
    savgol_window_length: int | None = None
    savgol_polynomial_order: int | None = None
    savgol_derivative_order: int = 0
    savgol_delta_nm: float = 1.0
    savgol_mode: str = "interp"

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "PreprocessSpec":
        return cls(
            name=str(values["name"]),
            role=str(values["role"]),
            instrument=int(values["instrument"]),
            transform=str(values["transform"]),
            wavelength_max_nm=(
                None
                if values.get("wavelength_max_nm") is None
                else float(values["wavelength_max_nm"])
            ),
            savgol_window_length=(
                None
                if values.get("savgol_window_length") is None
                else int(values["savgol_window_length"])
            ),
            savgol_polynomial_order=(
                None
                if values.get("savgol_polynomial_order") is None
                else int(values["savgol_polynomial_order"])
            ),
            savgol_derivative_order=int(
                values.get("savgol_derivative_order", 0)
            ),
            savgol_delta_nm=float(values.get("savgol_delta_nm", 1.0)),
            savgol_mode=str(values.get("savgol_mode", "interp")),
        )


@dataclass(frozen=True)
class FittedPreprocessor:
    """Training-only centring parameters and a deterministic linear operator."""

    spec: PreprocessSpec
    wavelengths_original: np.ndarray
    wavelengths_processed: np.ndarray
    operator: np.ndarray
    x_mean_processed: np.ndarray
    y_mean: float

    @property
    def processed_grid_size(self) -> int:
        return int(self.operator.shape[1])

    def transform_X(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.operator.shape[0]:
            raise ValueError("X has the wrong original wavelength dimension")
        transformed = apply_fixed_transform(X, self.wavelengths_original, self.spec)
        return transformed - self.x_mean_processed

    def transform_y(self, y: np.ndarray) -> np.ndarray:
        return np.asarray(y, dtype=float).reshape(-1) - self.y_mean


@dataclass
class ProcedureFit:
    """One fitted procedure with all information required for prediction."""

    method: str
    beta_processed: np.ndarray | None
    selected_components: int | None
    preprocessor: FittedPreprocessor
    criterion_name: str
    criterion_values: np.ndarray
    eligible: np.ndarray
    diagnostics: dict[str, Any]
    runtime_seconds: float
    status: str = "ok"
    error_type: str | None = None
    error_message: str | None = None

    @property
    def linear_weights_original(self) -> np.ndarray:
        """Weights w in intercept + X_original @ w."""
        if self.beta_processed is None:
            return np.full(self.preprocessor.operator.shape[0], np.nan)
        return (
            self.preprocessor.operator @ self.beta_processed
            / self.preprocessor.processed_grid_size
        )

    @property
    def intercept(self) -> float:
        weights = self.linear_weights_original
        if not np.all(np.isfinite(weights)):
            return math.nan
        transformed_mean_contribution = float(
            self.preprocessor.x_mean_processed @ self.beta_processed
            / self.preprocessor.processed_grid_size
        )
        return self.preprocessor.y_mean - transformed_mean_contribution

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.preprocessor.operator.shape[0]:
            raise ValueError("X has the wrong original wavelength dimension")
        if self.beta_processed is None or self.status != "ok":
            return np.full(X.shape[0], np.nan)
        centred = self.preprocessor.transform_X(X)
        return self.preprocessor.y_mean + (
            centred @ self.beta_processed
            / self.preprocessor.processed_grid_size
        )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _read_mat_bytes(
    archive_path: str | Path,
    data_config: Mapping[str, Any],
) -> tuple[bytes, dict[str, Any]]:
    path = Path(archive_path)
    if not path.is_file():
        raise FileNotFoundError(path)

    provenance: dict[str, Any] = {"input_path": str(path.resolve())}
    if zipfile.is_zipfile(path):
        zip_hash = sha256_file(path)
        if zip_hash != data_config["zip_sha256"]:
            raise ValueError(
                f"ZIP checksum mismatch: {zip_hash} != "
                f"{data_config['zip_sha256']}"
            )
        member = str(data_config["mat_member"])
        with zipfile.ZipFile(path) as archive:
            if member not in archive.namelist():
                raise ValueError(f"MAT member {member!r} is absent from archive")
            content = archive.read(member)
        provenance.update(
            {
                "container": "zip",
                "zip_sha256": zip_hash,
                "mat_member": member,
            }
        )
    else:
        content = path.read_bytes()
        provenance.update({"container": "mat", "mat_member": path.name})

    mat_hash = sha256_bytes(content)
    if mat_hash != data_config["mat_sha256"]:
        raise ValueError(
            f"MAT checksum mismatch: {mat_hash} != {data_config['mat_sha256']}"
        )
    provenance["mat_sha256"] = mat_hash
    provenance["mat_size_bytes"] = len(content)
    return content, provenance


def _extract_wavelengths(dataset_object: Any) -> np.ndarray:
    try:
        axis = np.asarray(dataset_object.axisscale[1, 0], dtype=float).reshape(-1)
    except (AttributeError, IndexError, TypeError, ValueError) as exc:
        raise ValueError("cannot extract spectral wavelength axis") from exc
    if axis.size < 2 or not np.all(np.isfinite(axis)):
        raise ValueError("invalid spectral wavelength axis")
    return axis


def _extract_response_labels(dataset_object: Any) -> tuple[str, ...]:
    try:
        values = np.asarray(dataset_object.label[1, 0]).reshape(-1)
    except (AttributeError, IndexError, TypeError) as exc:
        raise ValueError("cannot extract response labels") from exc
    return tuple(str(value).strip().lower() for value in values)


def _validate_expected_axis(
    wavelengths: np.ndarray,
    data_config: Mapping[str, Any],
) -> None:
    expected = np.arange(
        float(data_config["expected_wavelength_start_nm"]),
        float(data_config["expected_wavelength_stop_nm"])
        + 0.5 * float(data_config["expected_wavelength_step_nm"]),
        float(data_config["expected_wavelength_step_nm"]),
    )
    if len(expected) != int(data_config["expected_wavelength_count"]):
        raise ValueError("inconsistent expected wavelength configuration")
    if wavelengths.shape != expected.shape or not np.array_equal(
        wavelengths, expected
    ):
        raise ValueError("wavelength axis does not match the frozen protocol")


def load_regression_data(
    archive_path: str | Path,
    data_config: Mapping[str, Any],
    *,
    instrument: int,
    splits: Sequence[str],
    response_column: int | None = None,
) -> RegressionData:
    """Load only requested MAT variables and construct stable tablet IDs."""
    if instrument not in (1, 2):
        raise ValueError("instrument must be 1 or 2")
    splits = tuple(str(split) for split in splits)
    if not splits or any(split not in SPLITS for split in splits):
        raise ValueError("unknown or empty source split")
    if len(set(splits)) != len(splits):
        raise ValueError("source splits may not repeat")
    if response_column is None:
        response_column = int(data_config["response_column_zero_based"])

    content, _ = _read_mat_bytes(archive_path, data_config)
    variables = [name for split in splits for name in (f"{split}_{instrument}", f"{split}_Y")]
    loaded = loadmat(
        io.BytesIO(content),
        variable_names=variables,
        squeeze_me=True,
        struct_as_record=False,
    )

    X_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    id_parts: list[np.ndarray] = []
    split_parts: list[np.ndarray] = []
    row_parts: list[np.ndarray] = []
    common_axis: np.ndarray | None = None
    response_name = str(data_config["primary_response"]).lower()

    for split in splits:
        predictor_name = f"{split}_{instrument}"
        response_object_name = f"{split}_Y"
        if predictor_name not in loaded or response_object_name not in loaded:
            raise ValueError(f"missing expected MAT variables for {split}")
        predictor_object = loaded[predictor_name]
        response_object = loaded[response_object_name]
        X = np.asarray(predictor_object.data, dtype=float)
        responses = np.asarray(response_object.data, dtype=float)
        if X.ndim != 2 or responses.ndim != 2:
            raise ValueError(f"{split} data are not matrices")

        expected_n = int(data_config["expected_counts"][split])
        expected_p = int(data_config["expected_wavelength_count"])
        if X.shape != (expected_n, expected_p):
            raise ValueError(
                f"{predictor_name} has shape {X.shape}, expected "
                f"{(expected_n, expected_p)}"
            )
        if responses.shape[0] != expected_n or response_column >= responses.shape[1]:
            raise ValueError(f"{response_object_name} has an invalid shape")
        if not np.all(np.isfinite(X)) or not np.all(np.isfinite(responses)):
            raise ValueError(f"{split} contains missing or nonfinite values")

        labels = _extract_response_labels(response_object)
        if response_column >= len(labels) or labels[response_column] != response_name:
            raise ValueError(
                f"response column {response_column} is {labels}, not {response_name}"
            )
        wavelengths = _extract_wavelengths(predictor_object)
        _validate_expected_axis(wavelengths, data_config)
        if common_axis is None:
            common_axis = wavelengths.copy()
        elif not np.array_equal(common_axis, wavelengths):
            raise ValueError("wavelength axes differ across requested splits")

        rows = np.arange(1, expected_n + 1, dtype=int)
        X_parts.append(X)
        y_parts.append(responses[:, response_column])
        id_parts.append(np.array([f"{split}:{row:03d}" for row in rows]))
        split_parts.append(np.full(expected_n, split))
        row_parts.append(rows)

    assert common_axis is not None
    return RegressionData(
        X=np.vstack(X_parts),
        y=np.concatenate(y_parts),
        wavelengths=common_axis,
        tablet_ids=np.concatenate(id_parts),
        source_split=np.concatenate(split_parts),
        source_row=np.concatenate(row_parts),
        instrument=instrument,
        response_name=response_name,
    )


def audit_archive(
    archive_path: str | Path,
    data_config: Mapping[str, Any],
    *,
    include_benchmark: bool,
) -> dict[str, Any]:
    """Return deterministic structural checks without fitting any model."""
    content, provenance = _read_mat_bytes(archive_path, data_config)
    del content
    splits = SPLITS if include_benchmark else SPLITS[:2]
    by_instrument = {
        instrument: load_regression_data(
            archive_path,
            data_config,
            instrument=instrument,
            splits=splits,
        )
        for instrument in (1, 2)
    }
    first, second = by_instrument[1], by_instrument[2]
    if not np.array_equal(first.tablet_ids, second.tablet_ids):
        raise ValueError("tablet IDs do not align across instruments")
    if not np.array_equal(first.y, second.y):
        raise ValueError("responses do not align across instruments")

    split_summary: list[dict[str, Any]] = []
    for split in splits:
        mask = first.source_split == split
        values = first.y[mask]
        split_summary.append(
            {
                "split": split,
                "observations": int(mask.sum()),
                "assay_min": float(np.min(values)),
                "assay_max": float(np.max(values)),
                "assay_mean": float(np.mean(values)),
                "assay_sd_ddof1": float(np.std(values, ddof=1)),
            }
        )

    duplicate_counts = {
        f"instrument_{instrument}": int(
            len(data.X) - len(np.unique(data.X, axis=0))
        )
        for instrument, data in by_instrument.items()
    }
    return {
        "status": "passed",
        "benchmark_variables_loaded": bool(include_benchmark),
        "provenance": provenance,
        "observations": int(len(first.y)),
        "wavelength_count": int(len(first.wavelengths)),
        "wavelength_start_nm": float(first.wavelengths[0]),
        "wavelength_stop_nm": float(first.wavelengths[-1]),
        "wavelength_step_nm": float(np.diff(first.wavelengths)[0]),
        "all_finite": True,
        "duplicate_spectra": duplicate_counts,
        "split_summary": split_summary,
        "catalogue_count_discrepancy": {
            "catalogue": 654,
            "archive": int(sum(data_config["expected_counts"][s] for s in SPLITS)),
            "action": "retain all complete unique archive rows",
        },
    }


@lru_cache(maxsize=16)
def _cached_spectral_operator(
    wavelengths_tuple: tuple[float, ...],
    transform: str,
    wavelength_max_nm: float | None,
    window_length: int | None,
    polynomial_order: int | None,
    derivative_order: int,
    delta_nm: float,
    mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    wavelengths = np.asarray(wavelengths_tuple, dtype=float)
    grid_size = len(wavelengths)
    if transform == "identity":
        operator = np.eye(grid_size)
    elif transform == "savgol":
        window = window_length
        order = polynomial_order
        if window is None or order is None:
            raise ValueError("Savitzky-Golay settings are incomplete")
        if window % 2 != 1 or window <= order or window > grid_size:
            raise ValueError("invalid Savitzky-Golay window/order")
        operator = savgol_filter(
            np.eye(grid_size),
            window_length=window,
            polyorder=order,
            deriv=derivative_order,
            delta=delta_nm,
            axis=1,
            mode=mode,
        )
    else:
        raise ValueError(f"unknown transform {transform!r}")

    mask = np.ones(grid_size, dtype=bool)
    if wavelength_max_nm is not None:
        mask &= wavelengths <= wavelength_max_nm
    if np.count_nonzero(mask) < 2:
        raise ValueError("spectral transformation retains fewer than two points")
    return operator[:, mask], wavelengths[mask]


def build_spectral_operator(
    wavelengths: np.ndarray,
    spec: PreprocessSpec,
) -> tuple[np.ndarray, np.ndarray]:
    """Return B such that X_processed = X_original @ B."""
    wavelengths = np.asarray(wavelengths, dtype=float).reshape(-1)
    if len(wavelengths) < 2 or np.any(np.diff(wavelengths) <= 0.0):
        raise ValueError("wavelengths must be strictly increasing")
    return _cached_spectral_operator(
        tuple(float(value) for value in wavelengths),
        spec.transform,
        spec.wavelength_max_nm,
        spec.savgol_window_length,
        spec.savgol_polynomial_order,
        spec.savgol_derivative_order,
        spec.savgol_delta_nm,
        spec.savgol_mode,
    )


def apply_fixed_transform(
    X: np.ndarray,
    wavelengths: np.ndarray,
    spec: PreprocessSpec,
) -> np.ndarray:
    """Apply the fixed row-wise transform without dense identity products."""
    X = np.asarray(X, dtype=float)
    wavelengths = np.asarray(wavelengths, dtype=float).reshape(-1)
    if X.ndim != 2 or X.shape[1] != len(wavelengths):
        raise ValueError("X and wavelength axis disagree")
    if spec.transform == "identity":
        transformed = X
    elif spec.transform == "savgol":
        if spec.savgol_window_length is None or spec.savgol_polynomial_order is None:
            raise ValueError("Savitzky-Golay settings are incomplete")
        transformed = savgol_filter(
            X,
            window_length=spec.savgol_window_length,
            polyorder=spec.savgol_polynomial_order,
            deriv=spec.savgol_derivative_order,
            delta=spec.savgol_delta_nm,
            axis=1,
            mode=spec.savgol_mode,
        )
    else:
        raise ValueError(f"unknown transform {spec.transform!r}")
    if spec.wavelength_max_nm is not None:
        transformed = transformed[:, wavelengths <= spec.wavelength_max_nm]
    return np.asarray(transformed, dtype=float)


def fit_preprocessor(
    X_train: np.ndarray,
    y_train: np.ndarray,
    wavelengths: np.ndarray,
    spec: PreprocessSpec,
) -> FittedPreprocessor:
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float).reshape(-1)
    if X_train.ndim != 2 or X_train.shape[0] != len(y_train):
        raise ValueError("training X and y have incompatible shapes")
    if X_train.shape[1] != len(wavelengths) or len(y_train) < 2:
        raise ValueError("training data have invalid dimensions")
    if not np.all(np.isfinite(X_train)) or not np.all(np.isfinite(y_train)):
        raise ValueError("training data must be finite")
    operator, processed_wavelengths = build_spectral_operator(wavelengths, spec)
    transformed = apply_fixed_transform(X_train, wavelengths, spec)
    return FittedPreprocessor(
        spec=spec,
        wavelengths_original=np.asarray(wavelengths, dtype=float).copy(),
        wavelengths_processed=processed_wavelengths,
        operator=operator,
        x_mean_processed=np.mean(transformed, axis=0),
        y_mean=float(np.mean(y_train)),
    )


def _rng_from_parts(parts: Sequence[int]) -> np.random.Generator:
    seed_sequence = np.random.SeedSequence([int(x) for x in parts])
    return np.random.Generator(np.random.PCG64(seed_sequence))


def _balanced_one_group(
    indices: np.ndarray,
    y: np.ndarray,
    fold_count: int,
    rng: np.random.Generator,
) -> list[list[int]]:
    ordered = indices[np.argsort(y[indices], kind="stable")]
    allocations: list[list[int]] = [[] for _ in range(fold_count)]
    for start in range(0, len(ordered), fold_count):
        chunk = ordered[start : start + fold_count].copy()
        rng.shuffle(chunk)
        labels = rng.permutation(fold_count)[: len(chunk)]
        for index, label in zip(chunk, labels, strict=True):
            allocations[int(label)].append(int(index))
    return allocations


def make_source_balanced_folds(
    y: np.ndarray,
    source_split: np.ndarray,
    fold_count: int,
    seed_parts: Sequence[int],
) -> list[np.ndarray]:
    """Balance continuous y separately within each named source split."""
    y = np.asarray(y, dtype=float).reshape(-1)
    source_split = np.asarray(source_split).reshape(-1)
    if len(y) != len(source_split) or not 2 <= fold_count <= len(y):
        raise ValueError("invalid response, source labels, or fold count")
    rng = _rng_from_parts(seed_parts)
    folds: list[list[int]] = [[] for _ in range(fold_count)]
    for source in dict.fromkeys(str(value) for value in source_split):
        indices = np.flatnonzero(source_split == source)
        allocated = _balanced_one_group(indices, y, fold_count, rng)
        for fold, members in enumerate(allocated):
            folds[fold].extend(members)
    result = [np.asarray(sorted(members), dtype=int) for members in folds]
    validate_folds(result, len(y))
    return result


def make_source_contiguous_folds(
    source_split: np.ndarray,
    fold_count: int,
) -> list[np.ndarray]:
    source_split = np.asarray(source_split).reshape(-1)
    if not 2 <= fold_count <= len(source_split):
        raise ValueError("invalid fold count")
    folds: list[list[int]] = [[] for _ in range(fold_count)]
    for source in dict.fromkeys(str(value) for value in source_split):
        parts = np.array_split(np.flatnonzero(source_split == source), fold_count)
        for fold, part in enumerate(parts):
            folds[fold].extend(int(value) for value in part)
    result = [np.asarray(sorted(members), dtype=int) for members in folds]
    validate_folds(result, len(source_split))
    return result


def validate_folds(folds: Sequence[np.ndarray], sample_size: int) -> None:
    if len(folds) < 2:
        raise ValueError("at least two folds are required")
    seen = np.zeros(sample_size, dtype=int)
    for fold in folds:
        indices = np.asarray(fold, dtype=int)
        if len(indices) == 0:
            raise ValueError("folds may not be empty")
        if np.any(indices < 0) or np.any(indices >= sample_size):
            raise ValueError("fold index lies outside the sample")
        seen[indices] += 1
    if not np.all(seen == 1):
        raise ValueError("folds must partition every observation exactly once")


def training_indices(sample_size: int, validation_indices: np.ndarray) -> np.ndarray:
    mask = np.ones(sample_size, dtype=bool)
    mask[np.asarray(validation_indices, dtype=int)] = False
    return np.flatnonzero(mask)


def _usable_arnoldi(path: method_core.FPLSPath) -> np.ndarray:
    mask = np.asarray(path.valid, dtype=bool).copy()
    mask[np.arange(len(mask)) >= int(path.effective_dimension)] = False
    return mask


def _raw_log10_column_norms(path: method_core.FPLSPath) -> np.ndarray:
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        norms = np.linalg.norm(path.basis, axis=0)
        values = np.log10(norms)
    values[~np.isfinite(values)] = np.nan
    return values


def _failed_fit(
    method: str,
    preprocessor: FittedPreprocessor,
    criterion_name: str,
    criterion_values: np.ndarray,
    eligible: np.ndarray,
    diagnostics: dict[str, Any],
    runtime_seconds: float,
    exc: BaseException | None,
    message: str | None = None,
) -> ProcedureFit:
    return ProcedureFit(
        method=method,
        beta_processed=None,
        selected_components=None,
        preprocessor=preprocessor,
        criterion_name=criterion_name,
        criterion_values=criterion_values,
        eligible=eligible,
        diagnostics=diagnostics,
        runtime_seconds=runtime_seconds,
        status="failed",
        error_type=(type(exc).__name__ if exc is not None else "NoEligibleCandidate"),
        error_message=(str(exc) if exc is not None else message),
    )


def _successful_fit(
    *,
    method: str,
    beta: np.ndarray,
    selected: int,
    preprocessor: FittedPreprocessor,
    criterion_name: str,
    criterion_values: np.ndarray,
    eligible: np.ndarray,
    diagnostics: dict[str, Any],
    runtime_seconds: float,
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> ProcedureFit:
    fit = ProcedureFit(
        method=method,
        beta_processed=np.asarray(beta, dtype=float),
        selected_components=int(selected),
        preprocessor=preprocessor,
        criterion_name=criterion_name,
        criterion_values=np.asarray(criterion_values, dtype=float),
        eligible=np.asarray(eligible, dtype=bool),
        diagnostics=diagnostics,
        runtime_seconds=float(runtime_seconds),
    )
    prediction = fit.predict(X_train)
    errors = prediction - np.asarray(y_train, dtype=float)
    fit.diagnostics.update(
        {
            "training_rmse": float(np.sqrt(np.mean(errors**2))),
            "training_bias": float(np.mean(errors)),
            "coefficient_norm": float(np.linalg.norm(beta)),
            "coefficient_max_abs": float(np.max(np.abs(beta))),
            "prediction_finite": bool(np.all(np.isfinite(prediction))),
        }
    )
    if not fit.diagnostics["prediction_finite"]:
        fit.status = "failed"
        fit.error_type = "NonfinitePrediction"
        fit.error_message = "training prediction is nonfinite"
        fit.beta_processed = None
        fit.selected_components = None
    return fit


def _fit_fpls_pair(
    data: RegressionData,
    spec: PreprocessSpec,
    preprocessor: FittedPreprocessor,
    X_centered: np.ndarray,
    y_centered: np.ndarray,
    K: np.ndarray,
    r: np.ndarray,
    inner_folds: Sequence[np.ndarray],
    maximum_components: int,
    rank_tolerance: float,
) -> tuple[ProcedureFit, ProcedureFit]:
    validate_folds(inner_folds, len(data.y))
    smallest_inner_train = min(
        len(data.y) - len(np.asarray(fold)) for fold in inner_folds
    )
    cap = min(
        int(maximum_components),
        preprocessor.processed_grid_size - 1,
        len(data.y) - 1,
        smallest_inner_train - 1,
    )
    if cap < 1:
        raise ValueError("training sample is too small for a positive path")

    raw_runtime = 0.0
    arnoldi_runtime = 0.0
    raw_sse = np.zeros(cap)
    arnoldi_sse = np.zeros(cap)
    raw_valid_all = np.ones(cap, dtype=bool)
    arnoldi_valid_all = np.ones(cap, dtype=bool)
    raw_inner_errors: list[str] = []
    arnoldi_inner_errors: list[str] = []
    inner_effective_dimensions: list[int] = []

    for validation in inner_folds:
        validation = np.asarray(validation, dtype=int)
        train = training_indices(len(data.y), validation)
        local_preprocessor = fit_preprocessor(
            data.X[train], data.y[train], data.wavelengths, spec
        )
        X_train = local_preprocessor.transform_X(data.X[train])
        y_train = local_preprocessor.transform_y(data.y[train])
        X_validation = local_preprocessor.transform_X(data.X[validation])
        y_validation = data.y[validation]
        K_train, r_train = method_core.empirical_moments(X_train, y_train)

        start = time.perf_counter()
        try:
            raw_path = method_core.raw_fpls_path(
                y_train, X_train, K_train, r_train, cap
            )
        except NUMERICAL_EXCEPTIONS as exc:
            raw_path = None
            raw_valid_all[:] = False
            raw_inner_errors.append(f"{type(exc).__name__}: {exc}")
        raw_runtime += time.perf_counter() - start
        if raw_path is not None:
            raw_valid = np.asarray(raw_path.valid, dtype=bool)
            raw_valid_all &= raw_valid
            if np.any(raw_valid):
                prediction = local_preprocessor.y_mean + (
                    X_validation @ raw_path.beta[:, raw_valid]
                    / local_preprocessor.processed_grid_size
                )
                with np.errstate(over="ignore", invalid="ignore"):
                    errors = np.sum(
                        (y_validation[:, None] - prediction) ** 2, axis=0
                    )
                positions = np.flatnonzero(raw_valid)
                finite = np.isfinite(errors)
                raw_sse[positions[finite]] += errors[finite]
                raw_valid_all[positions[~finite]] = False

        start = time.perf_counter()
        try:
            arnoldi_path = method_core.arnoldi_fpls_path(
                y_train,
                X_train,
                K_train,
                r_train,
                cap,
                rank_tolerance=rank_tolerance,
            )
        except NUMERICAL_EXCEPTIONS as exc:
            arnoldi_path = None
            arnoldi_valid_all[:] = False
            arnoldi_inner_errors.append(f"{type(exc).__name__}: {exc}")
        arnoldi_runtime += time.perf_counter() - start
        if arnoldi_path is not None:
            arnoldi_valid = _usable_arnoldi(arnoldi_path)
            inner_effective_dimensions.append(int(arnoldi_path.effective_dimension))
            arnoldi_valid_all &= arnoldi_valid
            if np.any(arnoldi_valid):
                prediction = local_preprocessor.y_mean + (
                    X_validation @ arnoldi_path.beta[:, arnoldi_valid]
                    / local_preprocessor.processed_grid_size
                )
                errors = np.sum(
                    (y_validation[:, None] - prediction) ** 2, axis=0
                )
                positions = np.flatnonzero(arnoldi_valid)
                finite = np.isfinite(errors)
                arnoldi_sse[positions[finite]] += errors[finite]
                arnoldi_valid_all[positions[~finite]] = False

    start = time.perf_counter()
    raw_full_error: BaseException | None = None
    try:
        raw_full = method_core.raw_fpls_path(
            y_centered, X_centered, K, r, cap
        )
    except NUMERICAL_EXCEPTIONS as exc:
        raw_full = None
        raw_full_error = exc
    raw_runtime += time.perf_counter() - start

    start = time.perf_counter()
    arnoldi_full_error: BaseException | None = None
    try:
        arnoldi_full = method_core.arnoldi_fpls_path(
            y_centered,
            X_centered,
            K,
            r,
            cap,
            rank_tolerance=rank_tolerance,
        )
    except NUMERICAL_EXCEPTIONS as exc:
        arnoldi_full = None
        arnoldi_full_error = exc
    arnoldi_runtime += time.perf_counter() - start

    raw_cv = np.full(cap, np.inf)
    raw_diagnostics: dict[str, Any] = {
        "path_cap": cap,
        "processed_grid_size": preprocessor.processed_grid_size,
        "inner_fold_failures": len(raw_inner_errors),
        "inner_error_messages": " | ".join(raw_inner_errors),
    }
    if raw_full is None:
        raw_fit = _failed_fit(
            "Raw FPLS",
            preprocessor,
            "inner_cv_mse",
            raw_cv,
            np.zeros(cap, dtype=bool),
            raw_diagnostics,
            raw_runtime,
            raw_full_error,
        )
    else:
        raw_eligible = raw_valid_all & np.asarray(raw_full.valid, dtype=bool)
        raw_cv[raw_eligible] = raw_sse[raw_eligible] / len(data.y)
        raw_diagnostics.update(
            {
                "effective_dimension": int(raw_full.effective_dimension),
                "valid_curve": np.asarray(raw_full.valid, dtype=bool),
                "basis_condition_curve": raw_full.basis_condition,
                "design_condition_curve": raw_full.design_condition,
                "krylov_log10_column_norm_curve": _raw_log10_column_norms(raw_full),
            }
        )
        if not np.any(raw_eligible):
            raw_fit = _failed_fit(
                "Raw FPLS",
                preprocessor,
                "inner_cv_mse",
                raw_cv,
                raw_eligible,
                raw_diagnostics,
                raw_runtime,
                None,
                "Raw FPLS has no candidate valid in every inner fold and full fit",
            )
        else:
            raw_selected = int(np.argmin(raw_cv)) + 1
            raw_diagnostics.update(
                {
                    "selected_basis_condition": float(
                        raw_full.basis_condition[raw_selected - 1]
                    ),
                    "selected_design_condition": float(
                        raw_full.design_condition[raw_selected - 1]
                    ),
                    "instability_flag": bool(
                        max(
                            raw_full.basis_condition[raw_selected - 1],
                            raw_full.design_condition[raw_selected - 1],
                        )
                        > method_core.RAW_CONDITION_THRESHOLD
                    ),
                    "selected_at_cap": bool(raw_selected == cap),
                }
            )
            raw_fit = _successful_fit(
                method="Raw FPLS",
                beta=raw_full.beta[:, raw_selected - 1],
                selected=raw_selected,
                preprocessor=preprocessor,
                criterion_name="inner_cv_mse",
                criterion_values=raw_cv,
                eligible=raw_eligible,
                diagnostics=raw_diagnostics,
                runtime_seconds=raw_runtime,
                X_train=data.X,
                y_train=data.y,
            )

    arnoldi_cv = np.full(cap, np.inf)
    arnoldi_diagnostics: dict[str, Any] = {
        "path_cap": cap,
        "processed_grid_size": preprocessor.processed_grid_size,
        "inner_fold_failures": len(arnoldi_inner_errors),
        "inner_error_messages": " | ".join(arnoldi_inner_errors),
        "inner_effective_dimension_min": (
            min(inner_effective_dimensions) if inner_effective_dimensions else 0
        ),
        "inner_effective_dimension_max": (
            max(inner_effective_dimensions) if inner_effective_dimensions else 0
        ),
    }
    if arnoldi_full is None:
        arnoldi_fit = _failed_fit(
            "Arnoldi FPLS",
            preprocessor,
            "inner_cv_mse",
            arnoldi_cv,
            np.zeros(cap, dtype=bool),
            arnoldi_diagnostics,
            arnoldi_runtime,
            arnoldi_full_error,
        )
    else:
        arnoldi_full_valid = _usable_arnoldi(arnoldi_full)
        arnoldi_eligible = arnoldi_valid_all & arnoldi_full_valid
        arnoldi_cv[arnoldi_eligible] = (
            arnoldi_sse[arnoldi_eligible] / len(data.y)
        )
        arnoldi_diagnostics.update(
            {
                "effective_dimension": int(arnoldi_full.effective_dimension),
                "rank_breakdown": bool(arnoldi_full.effective_dimension < cap),
                "valid_curve": arnoldi_full_valid,
                "basis_condition_curve": arnoldi_full.basis_condition,
                "design_condition_curve": arnoldi_full.design_condition,
                "orthogonality_defect_curve": arnoldi_full.orthogonality_defect,
            }
        )
        if not np.any(arnoldi_eligible):
            arnoldi_fit = _failed_fit(
                "Arnoldi FPLS",
                preprocessor,
                "inner_cv_mse",
                arnoldi_cv,
                arnoldi_eligible,
                arnoldi_diagnostics,
                arnoldi_runtime,
                None,
                "Arnoldi FPLS has no effective candidate valid in every fold",
            )
        else:
            arnoldi_selected = int(np.argmin(arnoldi_cv)) + 1
            arnoldi_diagnostics.update(
                {
                    "selected_basis_condition": float(
                        arnoldi_full.basis_condition[arnoldi_selected - 1]
                    ),
                    "selected_design_condition": float(
                        arnoldi_full.design_condition[arnoldi_selected - 1]
                    ),
                    "selected_orthogonality_defect": float(
                        arnoldi_full.orthogonality_defect[arnoldi_selected - 1]
                    ),
                    "instability_flag": bool(
                        arnoldi_full.orthogonality_defect[arnoldi_selected - 1]
                        > method_core.ARNOLDI_DEFECT_THRESHOLD
                    ),
                    "selected_at_cap": bool(arnoldi_selected == cap),
                }
            )
            arnoldi_fit = _successful_fit(
                method="Arnoldi FPLS",
                beta=arnoldi_full.beta[:, arnoldi_selected - 1],
                selected=arnoldi_selected,
                preprocessor=preprocessor,
                criterion_name="inner_cv_mse",
                criterion_values=arnoldi_cv,
                eligible=arnoldi_eligible,
                diagnostics=arnoldi_diagnostics,
                runtime_seconds=arnoldi_runtime,
                X_train=data.X,
                y_train=data.y,
            )

    return raw_fit, arnoldi_fit


def _spectrum_diagnostics(X_centered: np.ndarray, cap: int) -> dict[str, Any]:
    sample_size, grid_size = X_centered.shape
    gram = X_centered @ X_centered.T / (sample_size * grid_size)
    eigenvalues = np.linalg.eigvalsh(0.5 * (gram + gram.T))[::-1]
    threshold = (
        np.finfo(float).eps
        * grid_size
        * max(float(np.max(np.abs(eigenvalues))), np.finfo(float).tiny)
    )
    positive = eigenvalues[eigenvalues > threshold]
    return {
        "numerical_positive_rank": int(len(positive)),
        "eigenvalue_threshold": float(threshold),
        "leading_eigenvalue_curve": eigenvalues[:cap],
    }


def _fit_fpcr_and_cg(
    data: RegressionData,
    preprocessor: FittedPreprocessor,
    X_centered: np.ndarray,
    y_centered: np.ndarray,
    K: np.ndarray,
    r: np.ndarray,
    maximum_components: int,
    cg_tau: float,
    cg_delta: float,
    cg_tolerance: float,
) -> tuple[ProcedureFit, ProcedureFit]:
    cap = min(
        int(maximum_components),
        preprocessor.processed_grid_size - 1,
        len(data.y) - 1,
    )
    if cap < 1:
        raise ValueError("training sample is too small for a positive path")
    common_diagnostics = _spectrum_diagnostics(X_centered, cap)

    start = time.perf_counter()
    fpcr_path_error: BaseException | None = None
    try:
        fpcr_path = method_core.fpcr_spectral_path(r, K, cap)
    except NUMERICAL_EXCEPTIONS as exc:
        fpcr_path = None
        fpcr_path_error = exc
    fpcr_path_elapsed = time.perf_counter() - start

    if fpcr_path is None:
        elapsed = fpcr_path_elapsed
        empty = np.full(cap, np.inf)
        fpcr_fit = _failed_fit(
            "FPCR",
            preprocessor,
            "response_gcv",
            empty,
            np.zeros(cap, dtype=bool),
            dict(common_diagnostics),
            elapsed,
            fpcr_path_error,
        )
        cg_fit = _failed_fit(
            "CG-FPLS-code",
            preprocessor,
            "discrepancy_moment",
            empty,
            np.zeros(cap, dtype=bool),
            dict(common_diagnostics),
            elapsed,
            fpcr_path_error,
            "CG variance pilot unavailable because FPCR path failed",
        )
        return fpcr_fit, cg_fit

    try:
        selection_start = time.perf_counter()
        fpcr_selection = method_core.select_fpcr_response_gcv(
            y_centered, X_centered, fpcr_path
        )
        fpcr_elapsed = fpcr_path_elapsed + time.perf_counter() - selection_start
        fpcr_eligible = np.ones(fpcr_path.shape[1], dtype=bool)
        fpcr_diagnostics = dict(common_diagnostics)
        eigenvalues = np.asarray(common_diagnostics["leading_eigenvalue_curve"])
        selected_eigenvalue = eigenvalues[fpcr_selection.selected_components - 1]
        fpcr_diagnostics.update(
            {
                "selected_eigenvalue": float(selected_eigenvalue),
                "selected_spectral_condition": float(
                    eigenvalues[0] / selected_eigenvalue
                ),
                "selected_at_cap": bool(
                    fpcr_selection.selected_components == fpcr_path.shape[1]
                ),
                "path_cap": int(fpcr_path.shape[1]),
                "processed_grid_size": preprocessor.processed_grid_size,
                "shared_fpcr_path_seconds": float(fpcr_path_elapsed),
            }
        )
        fpcr_fit = _successful_fit(
            method="FPCR",
            beta=fpcr_selection.beta,
            selected=fpcr_selection.selected_components,
            preprocessor=preprocessor,
            criterion_name="response_gcv",
            criterion_values=fpcr_selection.criterion_values,
            eligible=fpcr_eligible,
            diagnostics=fpcr_diagnostics,
            runtime_seconds=fpcr_elapsed,
            X_train=data.X,
            y_train=data.y,
        )
    except NUMERICAL_EXCEPTIONS as exc:
        fpcr_elapsed = fpcr_path_elapsed + time.perf_counter() - selection_start
        fpcr_fit = _failed_fit(
            "FPCR",
            preprocessor,
            "response_gcv",
            np.full(fpcr_path.shape[1], np.inf),
            np.zeros(fpcr_path.shape[1], dtype=bool),
            dict(common_diagnostics),
            fpcr_elapsed,
            exc,
        )

    start = time.perf_counter()
    try:
        pilot = method_core.select_cg_variance_pilot(r, K, fpcr_path)
        cg_path = method_core.cg_fpls_path(
            r, K, cap, tolerance=cg_tolerance
        )
        cg_selection = method_core.select_cg_fpls_code(
            cg_path,
            X_centered,
            y_centered,
            r,
            K,
            pilot.beta,
            tau=cg_tau,
            delta=cg_delta,
        )
        update_norms = np.empty(cg_path.shape[1])
        update_norms[0] = np.linalg.norm(cg_path[:, 0])
        if cg_path.shape[1] > 1:
            update_norms[1:] = np.linalg.norm(np.diff(cg_path, axis=1), axis=0)
        changed = np.flatnonzero(update_norms > 0.0)
        effective_dimension = int(changed[-1] + 1) if len(changed) else 0
        cg_diagnostics = dict(common_diagnostics)
        cg_diagnostics.update(
            {
                "pilot_components": int(pilot.selected_components),
                "pilot_gcv_curve": pilot.criterion_values,
                "threshold": float(cg_selection.threshold),
                "threshold_reached": bool(cg_selection.threshold_reached),
                "sigma2": float(cg_selection.sigma2),
                "moment_residual_m0": float(cg_selection.moment_path[0]),
                "moment_residual_curve": cg_selection.moment_path[1:],
                "update_norm_curve": update_norms,
                "effective_dimension": effective_dimension,
                "selected_beyond_effective_dimension": bool(
                    cg_selection.selected_components > effective_dimension
                ),
                "selected_at_cap": bool(
                    cg_selection.selected_components == cg_path.shape[1]
                ),
                "path_cap": int(cg_path.shape[1]),
                "processed_grid_size": preprocessor.processed_grid_size,
                "shared_fpcr_path_seconds": float(fpcr_path_elapsed),
            }
        )
        cg_fit = _successful_fit(
            method="CG-FPLS-code",
            beta=cg_selection.beta,
            selected=cg_selection.selected_components,
            preprocessor=preprocessor,
            criterion_name="discrepancy_moment",
            criterion_values=cg_selection.moment_path[1:],
            eligible=np.ones(cg_path.shape[1], dtype=bool),
            diagnostics=cg_diagnostics,
            runtime_seconds=fpcr_path_elapsed + time.perf_counter() - start,
            X_train=data.X,
            y_train=data.y,
        )
    except NUMERICAL_EXCEPTIONS as exc:
        cg_fit = _failed_fit(
            "CG-FPLS-code",
            preprocessor,
            "discrepancy_moment",
            np.full(cap, np.inf),
            np.zeros(cap, dtype=bool),
            dict(common_diagnostics),
            fpcr_path_elapsed + time.perf_counter() - start,
            exc,
        )
    return fpcr_fit, cg_fit


def fit_all_procedures(
    data: RegressionData,
    spec: PreprocessSpec,
    inner_folds: Sequence[np.ndarray],
    estimation_config: Mapping[str, Any],
) -> list[ProcedureFit]:
    """Fit all four frozen procedures with leakage-safe empirical handling."""
    if data.instrument != spec.instrument:
        raise ValueError("data instrument and preprocessing variant disagree")
    validate_folds(inner_folds, len(data.y))
    preprocessor = fit_preprocessor(data.X, data.y, data.wavelengths, spec)
    X_centered = preprocessor.transform_X(data.X)
    y_centered = preprocessor.transform_y(data.y)
    K, r = method_core.empirical_moments(X_centered, y_centered)

    raw_fit, arnoldi_fit = _fit_fpls_pair(
        data,
        spec,
        preprocessor,
        X_centered,
        y_centered,
        K,
        r,
        inner_folds,
        int(estimation_config["maximum_components"]),
        float(estimation_config["rank_tolerance"]),
    )
    fpcr_fit, cg_fit = _fit_fpcr_and_cg(
        data,
        preprocessor,
        X_centered,
        y_centered,
        K,
        r,
        int(estimation_config["maximum_components"]),
        float(estimation_config["cg_tau"]),
        float(estimation_config["cg_delta"]),
        float(estimation_config["cg_path_tolerance"]),
    )
    by_name = {
        fit.method: fit
        for fit in (raw_fit, arnoldi_fit, fpcr_fit, cg_fit)
    }
    if set(by_name) != set(EXPECTED_METHODS):
        raise AssertionError("unexpected fitted-method set")
    return [by_name[name] for name in EXPECTED_METHODS]


def prediction_metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=float).reshape(-1)
    prediction = np.asarray(prediction, dtype=float).reshape(-1)
    if y.shape != prediction.shape or len(y) < 2:
        raise ValueError("observations and predictions have incompatible shapes")
    if not np.all(np.isfinite(y)) or not np.all(np.isfinite(prediction)):
        return {"rmse": math.nan, "mae": math.nan, "bias": math.nan, "r2": math.nan}
    residual = prediction - y
    denominator = float(np.sum((y - np.mean(y)) ** 2))
    r2 = math.nan if denominator <= 0.0 else 1.0 - float(np.sum(residual**2)) / denominator
    return {
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "bias": float(np.mean(residual)),
        "r2": float(r2),
    }


def make_bootstrap_indices(
    sample_size: int,
    replicates: int,
    seed_parts: Sequence[int],
    *,
    block_length: int | None = None,
) -> np.ndarray:
    if sample_size < 2 or replicates < 1:
        raise ValueError("invalid bootstrap dimensions")
    rng = _rng_from_parts(seed_parts)
    dtype = np.uint16 if sample_size <= np.iinfo(np.uint16).max else np.uint32
    if block_length is None:
        return rng.integers(
            0, sample_size, size=(replicates, sample_size), dtype=dtype
        )
    if not 1 <= block_length <= sample_size:
        raise ValueError("invalid circular block length")
    block_count = math.ceil(sample_size / block_length)
    starts = rng.integers(
        0, sample_size, size=(replicates, block_count), dtype=np.int64
    )
    offsets = np.arange(block_length, dtype=np.int64)
    indices = (starts[:, :, None] + offsets[None, None, :]) % sample_size
    return indices.reshape(replicates, -1)[:, :sample_size].astype(dtype)


def bootstrap_metric_draws(
    y: np.ndarray,
    predictions: Mapping[str, np.ndarray],
    indices: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute paired bootstrap draws without refitting any model."""
    y = np.asarray(y, dtype=float).reshape(-1)
    indices = np.asarray(indices)
    if indices.ndim != 2 or indices.shape[1] != len(y):
        raise ValueError("bootstrap index matrix has an incompatible shape")
    y_draw = y[indices]
    y_mean = np.mean(y_draw, axis=1)
    sst = np.sum((y_draw - y_mean[:, None]) ** 2, axis=1)
    output: dict[str, np.ndarray] = {}
    for method, values in predictions.items():
        prediction = np.asarray(values, dtype=float).reshape(-1)
        if prediction.shape != y.shape:
            raise ValueError(f"prediction shape mismatch for {method}")
        if not np.all(np.isfinite(prediction)):
            invalid = np.full(indices.shape[0], np.nan)
            for metric in ("rmse", "mae", "bias", "r2"):
                output[f"{method}|{metric}"] = invalid.copy()
            continue
        residual = prediction[indices] - y_draw
        mse = np.mean(residual**2, axis=1)
        output[f"{method}|rmse"] = np.sqrt(mse)
        output[f"{method}|mae"] = np.mean(np.abs(residual), axis=1)
        output[f"{method}|bias"] = np.mean(residual, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            output[f"{method}|r2"] = 1.0 - np.sum(residual**2, axis=1) / sst
    return output


def bootstrap_summary_rows(
    draws: Mapping[str, np.ndarray],
    point_metrics: Mapping[str, Mapping[str, float]],
    *,
    confidence: float,
    scheme: str,
    declared_contrasts: Sequence[tuple[str, str, str]],
) -> list[dict[str, Any]]:
    if not 0.0 < confidence < 1.0:
        raise ValueError("bootstrap confidence must lie in (0, 1)")
    lower_q = 50.0 * (1.0 - confidence)
    upper_q = 100.0 - lower_q
    rows: list[dict[str, Any]] = []

    def interval(values: np.ndarray) -> tuple[float, float, int]:
        finite = values[np.isfinite(values)]
        if len(finite) == 0:
            return math.nan, math.nan, 0
        return (
            float(np.percentile(finite, lower_q)),
            float(np.percentile(finite, upper_q)),
            int(len(finite)),
        )

    for method, metrics in point_metrics.items():
        for metric, point in metrics.items():
            values = np.asarray(draws[f"{method}|{metric}"], dtype=float)
            ci_lower, ci_upper, finite_replicates = interval(values)
            rows.append(
                {
                    "scheme": scheme,
                    "result_type": "method_metric",
                    "method_left": method,
                    "method_right": "",
                    "contrast_label": "",
                    "metric": metric,
                    "estimate": float(point),
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                    "confidence": float(confidence),
                    "replicates": int(len(values)),
                    "finite_replicates": finite_replicates,
                }
            )
    for label, left, right in declared_contrasts:
        for metric in ("rmse", "mae", "bias", "r2"):
            difference = np.asarray(draws[f"{left}|{metric}"]) - np.asarray(
                draws[f"{right}|{metric}"]
            )
            point = point_metrics[left][metric] - point_metrics[right][metric]
            ci_lower, ci_upper, finite_replicates = interval(difference)
            rows.append(
                {
                    "scheme": scheme,
                    "result_type": "method_contrast",
                    "method_left": left,
                    "method_right": right,
                    "contrast_label": label,
                    "metric": metric,
                    "estimate": float(point),
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                    "confidence": float(confidence),
                    "replicates": int(len(difference)),
                    "finite_replicates": finite_replicates,
                }
            )
    return rows


def all_pairwise_contrasts(methods: Iterable[str]) -> list[tuple[str, str, str]]:
    return [
        (f"{left} - {right}", left, right)
        for left, right in itertools.combinations(methods, 2)
    ]


def fit_summary_record(fit: ProcedureFit) -> dict[str, Any]:
    scalar_diagnostics: dict[str, Any] = {}
    for key, value in fit.diagnostics.items():
        if isinstance(value, (str, bytes)):
            scalar_diagnostics[key] = str(value)
        elif np.isscalar(value):
            scalar_diagnostics[key] = (
                value.item() if isinstance(value, np.generic) else value
            )
    return {
        "method": fit.method,
        "status": fit.status,
        "error_type": fit.error_type or "",
        "error_message": fit.error_message or "",
        "selected_components": fit.selected_components,
        "criterion_name": fit.criterion_name,
        "runtime_seconds": float(fit.runtime_seconds),
        "intercept": float(fit.intercept),
        **scalar_diagnostics,
    }


def fit_curve_records(fit: ProcedureFit) -> list[dict[str, Any]]:
    arrays: dict[str, np.ndarray] = {
        "criterion": np.asarray(fit.criterion_values),
        "eligible": np.asarray(fit.eligible, dtype=bool),
    }
    for key, value in fit.diagnostics.items():
        array = (
            np.asarray(value)
            if key.endswith("_curve")
            and isinstance(value, (np.ndarray, list, tuple))
            else None
        )
        if array is not None and array.ndim == 1 and len(array) > 0:
            arrays[key] = array
    maximum = max((len(values) for values in arrays.values()), default=0)
    records: list[dict[str, Any]] = []
    for position in range(maximum):
        row: dict[str, Any] = {"method": fit.method, "m": position + 1}
        for name, values in arrays.items():
            if position >= len(values):
                row[name] = math.nan
            else:
                value = values[position]
                if isinstance(value, (np.bool_, bool)):
                    row[name] = bool(value)
                elif isinstance(value, (str, bytes)):
                    row[name] = str(value)
                else:
                    row[name] = float(value)
        records.append(row)
    return records


def coefficient_records(fit: ProcedureFit, wavelengths: np.ndarray) -> list[dict[str, Any]]:
    weights = fit.linear_weights_original
    if len(weights) != len(wavelengths):
        raise ValueError("coefficient and wavelength axes disagree")
    processed_mean = np.full(len(wavelengths), np.nan)
    retained = np.zeros(len(wavelengths), dtype=bool)
    for processed_position, wavelength in enumerate(
        fit.preprocessor.wavelengths_processed
    ):
        matches = np.flatnonzero(np.isclose(wavelengths, wavelength, rtol=0.0, atol=1e-12))
        if len(matches) != 1:
            raise ValueError("processed wavelength does not map uniquely to original axis")
        original_position = int(matches[0])
        retained[original_position] = True
        processed_mean[original_position] = fit.preprocessor.x_mean_processed[
            processed_position
        ]
    return [
        {
            "method": fit.method,
            "wavelength_nm": float(wavelength),
            "linear_weight": float(weight),
            "functional_beta_common_1_over_650": float(len(wavelengths) * weight),
            "intercept": float(fit.intercept),
            "training_y_mean": float(fit.preprocessor.y_mean),
            "retained_processed_wavelength": bool(retained[position]),
            "training_x_mean_after_fixed_transform": float(processed_mean[position]),
            "processed_grid_size": int(fit.preprocessor.processed_grid_size),
            "status": fit.status,
        }
        for position, (wavelength, weight) in enumerate(
            zip(wavelengths, weights, strict=True)
        )
    ]


def run_self_tests() -> None:
    """Fast deterministic checks that never open the pharmaceutical data."""
    rng = np.random.default_rng(74291)
    n, p = 48, 16
    wavelengths = np.arange(p, dtype=float) * 2.0 + 600.0
    X = rng.normal(size=(n, p))
    beta = rng.normal(size=p)
    y = 7.5 + X @ beta / p + rng.normal(scale=0.05, size=n)
    source = np.array(["calibrate"] * 38 + ["validate"] * 10)
    rows = np.concatenate((np.arange(1, 39), np.arange(1, 11)))
    data = RegressionData(
        X=X,
        y=y,
        wavelengths=wavelengths,
        tablet_ids=np.array(
            [f"{split}:{row:03d}" for split, row in zip(source, rows, strict=True)]
        ),
        source_split=source,
        source_row=rows,
        instrument=1,
        response_name="assay",
    )
    spec = PreprocessSpec("test", "primary", 1, "identity")
    folds = make_source_balanced_folds(y, source, 4, [123, 1])
    validate_folds(folds, n)
    folds_again = make_source_balanced_folds(y, source, 4, [123, 1])
    if any(not np.array_equal(a, b) for a, b in zip(folds, folds_again, strict=True)):
        raise AssertionError("fold generation is not deterministic")

    prep = fit_preprocessor(X[:36], y[:36], wavelengths, spec)
    if not np.allclose(np.mean(prep.transform_X(X[:36]), axis=0), 0.0, atol=1e-14):
        raise AssertionError("training spectra are not centred")
    if not np.isclose(np.mean(prep.transform_y(y[:36])), 0.0, atol=1e-14):
        raise AssertionError("training response is not centred")
    shifted_validation = X.copy()
    shifted_validation[36:] += 1000.0
    prep_shifted = fit_preprocessor(
        shifted_validation[:36], y[:36], wavelengths, spec
    )
    if not np.array_equal(prep.x_mean_processed, prep_shifted.x_mean_processed):
        raise AssertionError("validation values leaked into preprocessing")

    test_estimation = {
        "maximum_components": 8,
        "rank_tolerance": 1e-10,
        "cg_tau": 1.01,
        "cg_delta": 0.1,
        "cg_path_tolerance": 1e-12,
    }
    fits = fit_all_procedures(data, spec, folds, test_estimation)
    if tuple(fit.method for fit in fits) != EXPECTED_METHODS:
        raise AssertionError("method order changed")
    if any(fit.status != "ok" for fit in fits):
        failures = [(fit.method, fit.error_message) for fit in fits if fit.status != "ok"]
        raise AssertionError(f"synthetic integration fit failed: {failures}")
    for fit in fits:
        direct = fit.intercept + X @ fit.linear_weights_original
        if not np.allclose(direct, fit.predict(X), rtol=1e-11, atol=1e-11):
            raise AssertionError(f"original-scale prediction identity failed for {fit.method}")

    smooth_spec = PreprocessSpec(
        "smooth", "sensitivity", 1, "savgol", None, 7, 3, 0, 2.0, "interp"
    )
    smooth_operator, _ = build_spectral_operator(wavelengths, smooth_spec)
    if not np.allclose(
        X @ smooth_operator,
        savgol_filter(X, 7, 3, deriv=0, delta=2.0, axis=1, mode="interp"),
        rtol=1e-12,
        atol=1e-12,
    ):
        raise AssertionError("Savitzky-Golay operator orientation is wrong")

    indices = make_bootstrap_indices(n, 25, [222, 1])
    if indices.shape != (25, n) or np.any(indices >= n):
        raise AssertionError("iid bootstrap indices are invalid")
    blocked = make_bootstrap_indices(n, 25, [222, 2], block_length=5)
    if blocked.shape != (25, n) or np.any(blocked >= n):
        raise AssertionError("block-bootstrap indices are invalid")
    predictions = {fit.method: fit.predict(X) for fit in fits}
    draws = bootstrap_metric_draws(y, predictions, indices)
    if len(draws) != 4 * len(fits) or any(len(value) != 25 for value in draws.values()):
        raise AssertionError("bootstrap metric archive has an invalid shape")
