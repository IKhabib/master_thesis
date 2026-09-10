

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
from scipy.linalg import LinAlgError, eigh, solve_triangular


METHODS = (
    "CG-FPLS-code",
    "Raw FPLS",
    "Arnoldi FPLS",
    "FPCR",
)
MODEL_NAMES = ("Model 1", "Model 2", "Model 3")
CONDITION_CAP = 1.0 / np.finfo(float).eps
RAW_CONDITION_THRESHOLD = 1.0 / np.sqrt(np.finfo(float).eps)
ARNOLDI_DEFECT_THRESHOLD = np.sqrt(np.finfo(float).eps)


@dataclass(frozen=True)
class ModelSpec:
    """One of the three intended simulation setups."""

    name: str
    eigenvalues: np.ndarray
    beta: np.ndarray


@dataclass
class FPLSPath:
    """A nested FPLS path and its numerical diagnostics."""

    beta: np.ndarray
    fitted: np.ndarray
    valid: np.ndarray
    basis: np.ndarray
    basis_condition: np.ndarray
    design_condition: np.ndarray
    orthogonality_defect: np.ndarray
    effective_dimension: int


@dataclass
class FPLSCVResult:
    """Cross-validated Raw and Arnoldi FPLS fits."""

    beta_raw: np.ndarray
    beta_arnoldi: np.ndarray
    m_raw: int
    m_arnoldi: int
    cv_raw: np.ndarray
    cv_arnoldi: np.ndarray
    raw_full_path: FPLSPath
    arnoldi_full_path: FPLSPath
    raw_valid_all_folds: np.ndarray


@dataclass(frozen=True)
class FPCRSelection:
    """An FPCR estimate and its component-selection criterion."""

    beta: np.ndarray
    selected_components: int
    criterion_values: np.ndarray


@dataclass(frozen=True)
class CGSelection:
    """The released-code CG-FPLS stopping result."""

    beta: np.ndarray
    selected_components: int
    threshold_reached: bool
    sigma2: float
    threshold: float
    moment_path: np.ndarray


def _validate_xy(y: np.ndarray, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=float).reshape(-1)
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[0] != len(y):
        raise ValueError("X must be two-dimensional with len(y) rows")
    if X.shape[0] < 2 or X.shape[1] < 2:
        raise ValueError("X must contain at least two rows and two columns")
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
        raise ValueError("X and y must be finite")
    return y, X


def _validate_moments(
    K: np.ndarray,
    r: np.ndarray,
    grid_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    K = np.asarray(K, dtype=float)
    r = np.asarray(r, dtype=float).reshape(-1)
    if K.shape != (grid_size, grid_size) or r.shape != (grid_size,):
        raise ValueError("K and r have incompatible dimensions")
    if not np.all(np.isfinite(K)) or not np.all(np.isfinite(r)):
        raise ValueError("K and r must be finite")
    return 0.5 * (K + K.T), r


def create_cosine_basis(grid: np.ndarray, basis_size: int) -> np.ndarray:
    """Return the fixed Babii cosine basis as a J-by-T array."""
    grid = np.asarray(grid, dtype=float).reshape(-1)
    if basis_size < 1 or len(grid) < 2 or np.any(np.diff(grid) <= 0.0):
        raise ValueError("basis_size must be positive and grid increasing")
    frequencies = np.arange(1, basis_size + 1)
    basis = np.sqrt(2.0) * np.cos(np.pi * np.outer(grid, frequencies))
    basis[:, 0] = 1.0
    return basis.T


def make_model_specs(basis_size: int, basis: np.ndarray) -> List[ModelSpec]:
    """Construct exactly the three intended controlled simulation setups."""
    basis = np.asarray(basis, dtype=float)
    if basis_size < 5 or basis.shape[0] != basis_size:
        raise ValueError("basis_size must be at least five and match the basis")
    index = np.arange(1, basis_size + 1, dtype=float)

    baseline_coefficients = 4.0 / index**2.7
    modified_coefficients = baseline_coefficients.copy()
    modified_coefficients[:5] = 4.0

    baseline_eigenvalues = 2.0 / index**1.1
    modified_eigenvalues = baseline_eigenvalues.copy()
    modified_eigenvalues[:5] = 2.0

    models = [
        ModelSpec(
            "Model 1",
            baseline_eigenvalues,
            basis.T @ baseline_coefficients,
        ),
        ModelSpec(
            "Model 2",
            baseline_eigenvalues,
            basis.T @ modified_coefficients,
        ),
        ModelSpec(
            "Model 3",
            modified_eigenvalues,
            basis.T @ baseline_coefficients,
        ),
    ]
    if tuple(model.name for model in models) != MODEL_NAMES:
        raise AssertionError("unexpected simulation-model order")
    return models


def generate_sample(
    rng: np.random.Generator,
    sample_size: int,
    basis: np.ndarray,
    model: ModelSpec,
    noise_sd: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate one functional-predictor matrix and response vector."""
    if sample_size < 2 or noise_sd < 0.0:
        raise ValueError("sample_size and noise_sd are invalid")
    basis_size, grid_size = basis.shape
    if model.eigenvalues.shape != (basis_size,) or model.beta.shape != (
        grid_size,
    ):
        raise ValueError("model and basis dimensions disagree")
    scores = rng.normal(size=(sample_size, basis_size))
    X = (scores * np.sqrt(model.eigenvalues)) @ basis
    y = X @ model.beta / grid_size + rng.normal(0.0, noise_sd, sample_size)
    return X, y


def empirical_moments(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return K = X'X/(nT) and r = X'y/n."""
    y, X = _validate_xy(y, X)
    sample_size, grid_size = X.shape
    return X.T @ X / (sample_size * grid_size), X.T @ y / sample_size


def _capped_condition(columns: np.ndarray) -> float:
    if columns.size == 0:
        return math.nan
    try:
        singular_values = np.linalg.svd(columns, compute_uv=False)
    except np.linalg.LinAlgError:
        return CONDITION_CAP
    if len(singular_values) == 0:
        return math.nan
    largest = float(singular_values[0])
    smallest = float(singular_values[-1])
    if largest <= 0.0 or smallest <= 0.0 or not np.isfinite(smallest):
        return CONDITION_CAP
    return min(largest / smallest, CONDITION_CAP)


def _normalized_condition(columns: np.ndarray) -> float:
    norms = np.linalg.norm(columns, axis=0)
    if np.any(norms <= 0.0) or not np.all(np.isfinite(norms)):
        return CONDITION_CAP
    return _capped_condition(columns / norms)


def raw_krylov_matrix(K: np.ndarray, r: np.ndarray, components: int) -> np.ndarray:
    """Form [r, Kr, ..., K^(m-1)r] without stabilization."""
    r = np.asarray(r, dtype=float).reshape(-1)
    K = np.asarray(K, dtype=float)
    if components < 1 or K.shape != (len(r), len(r)):
        raise ValueError("K, r, and components have incompatible dimensions")
    matrix = np.zeros((len(r), components))
    matrix[:, 0] = r
    for component in range(1, components):
        matrix[:, component] = K @ matrix[:, component - 1]
    return matrix


def raw_fpls_path(
    y: np.ndarray,
    X: np.ndarray,
    K: np.ndarray,
    r: np.ndarray,
    maximum_components: int,
) -> FPLSPath:
    """Compute the deliberately unstabilized Raw FPLS path."""
    y, X = _validate_xy(y, X)
    _, grid_size = X.shape
    K, r = _validate_moments(K, r, grid_size)
    if maximum_components < 1:
        raise ValueError("maximum_components must be positive")

    raw_basis = raw_krylov_matrix(K, r, maximum_components)
    design = X @ raw_basis / grid_size
    gram = design.T @ design
    rhs = design.T @ y

    beta_path = np.full((grid_size, maximum_components), np.nan)
    fitted_path = np.full((len(y), maximum_components), np.nan)
    valid = np.zeros(maximum_components, dtype=bool)
    basis_condition = np.full(maximum_components, CONDITION_CAP)
    design_condition = np.full(maximum_components, CONDITION_CAP)

    for component in range(maximum_components):
        count = component + 1
        basis_prefix = raw_basis[:, :count]
        design_prefix = design[:, :count]
        basis_condition[component] = _normalized_condition(basis_prefix)
        design_condition[component] = _normalized_condition(design_prefix)
        try:
            coefficients = np.linalg.solve(gram[:count, :count], rhs[:count])
        except np.linalg.LinAlgError:
            continue
        with np.errstate(over="ignore", invalid="ignore"):
            beta = basis_prefix @ coefficients
            fitted = design_prefix @ coefficients
        if np.all(np.isfinite(beta)) and np.all(np.isfinite(fitted)):
            beta_path[:, component] = beta
            fitted_path[:, component] = fitted
            valid[component] = True

    return FPLSPath(
        beta=beta_path,
        fitted=fitted_path,
        valid=valid,
        basis=raw_basis,
        basis_condition=basis_condition,
        design_condition=design_condition,
        orthogonality_defect=np.full(maximum_components, np.nan),
        effective_dimension=int(np.count_nonzero(valid)),
    )


def arnoldi_krylov_basis(
    K: np.ndarray,
    r: np.ndarray,
    maximum_components: int,
    *,
    rank_tolerance: float = 1e-10,
) -> np.ndarray:
    """Construct the CGS2 Arnoldi basis for the FPLS Krylov spaces."""
    r = np.asarray(r, dtype=float).reshape(-1)
    K = np.asarray(K, dtype=float)
    if K.shape != (len(r), len(r)):
        raise ValueError("K and r have incompatible dimensions")
    if maximum_components < 1 or rank_tolerance <= 0.0:
        raise ValueError("component count and rank tolerance must be positive")

    norm_r = float(np.linalg.norm(r))
    if norm_r == 0.0:
        return np.zeros((len(r), 0))
    basis = np.zeros((len(r), maximum_components))
    basis[:, 0] = r / norm_r
    dimension = 1
    norm_K = max(float(np.linalg.norm(K, ord=np.inf)), np.finfo(float).tiny)

    while dimension < maximum_components:
        candidate = K @ basis[:, dimension - 1]
        current = basis[:, :dimension]
        candidate -= current @ (current.T @ candidate)
        candidate -= current @ (current.T @ candidate)
        candidate_norm = float(np.linalg.norm(candidate))
        if (
            not np.isfinite(candidate_norm)
            or candidate_norm <= rank_tolerance * norm_K
        ):
            break
        basis[:, dimension] = candidate / candidate_norm
        dimension += 1
    return basis[:, :dimension]


def arnoldi_fpls_path(
    y: np.ndarray,
    X: np.ndarray,
    K: np.ndarray,
    r: np.ndarray,
    maximum_components: int,
    *,
    rank_tolerance: float = 1e-10,
) -> FPLSPath:
    """Fit FPLS in the CGS2 Arnoldi basis using response-space QR."""
    y, X = _validate_xy(y, X)
    _, grid_size = X.shape
    K, r = _validate_moments(K, r, grid_size)
    basis = arnoldi_krylov_basis(
        K,
        r,
        maximum_components,
        rank_tolerance=rank_tolerance,
    )

    beta_path = np.zeros((grid_size, maximum_components))
    fitted_path = np.zeros((len(y), maximum_components))
    valid = np.zeros(maximum_components, dtype=bool)
    basis_condition = np.full(maximum_components, np.nan)
    design_condition = np.full(maximum_components, np.nan)
    orthogonality_defect = np.full(maximum_components, np.nan)
    dimension = basis.shape[1]
    if dimension == 0:
        return FPLSPath(
            beta_path,
            fitted_path,
            valid,
            basis,
            basis_condition,
            design_condition,
            orthogonality_defect,
            0,
        )

    design = X @ basis / grid_size
    q_design, r_design = np.linalg.qr(design, mode="reduced")
    projected_y = q_design.T @ y
    diagonal_reference = max(abs(float(r_design[0, 0])), np.finfo(float).tiny)
    fitted_dimension = 0

    for component in range(dimension):
        count = component + 1
        basis_prefix = basis[:, :count]
        design_prefix = design[:, :count]
        basis_condition[component] = _capped_condition(basis_prefix)
        design_condition[component] = _capped_condition(design_prefix)
        orthogonality_defect[component] = float(
            np.linalg.norm(basis_prefix.T @ basis_prefix - np.eye(count), ord=2)
        )
        if abs(float(r_design[component, component])) <= (
            rank_tolerance * diagonal_reference
        ):
            break
        coefficients = solve_triangular(
            r_design[:count, :count],
            projected_y[:count],
            lower=False,
            check_finite=False,
        )
        beta = basis_prefix @ coefficients
        fitted = X @ beta / grid_size
        if not np.all(np.isfinite(beta)) or not np.all(np.isfinite(fitted)):
            break
        beta_path[:, component] = beta
        fitted_path[:, component] = fitted
        valid[component] = True
        fitted_dimension = count

    if 0 < fitted_dimension < maximum_components:
        beta_path[:, fitted_dimension:] = beta_path[:, [fitted_dimension - 1]]
        fitted_path[:, fitted_dimension:] = fitted_path[:, [fitted_dimension - 1]]
        valid[fitted_dimension:] = True
        basis_condition[fitted_dimension:] = basis_condition[fitted_dimension - 1]
        design_condition[fitted_dimension:] = design_condition[fitted_dimension - 1]
        orthogonality_defect[fitted_dimension:] = orthogonality_defect[
            fitted_dimension - 1
        ]

    return FPLSPath(
        beta=beta_path,
        fitted=fitted_path,
        valid=valid,
        basis=basis,
        basis_condition=basis_condition,
        design_condition=design_condition,
        orthogonality_defect=orthogonality_defect,
        effective_dimension=fitted_dimension,
    )


def make_folds(
    sample_size: int,
    fold_count: int,
    rng: np.random.Generator,
) -> List[np.ndarray]:
    """Create one random partition used by both FPLS implementations."""
    if not 2 <= fold_count <= sample_size:
        raise ValueError("fold_count must be between two and sample_size")
    return [
        np.asarray(part, dtype=int)
        for part in np.array_split(rng.permutation(sample_size), fold_count)
    ]


def cross_validate_fpls(
    y: np.ndarray,
    X: np.ndarray,
    K: np.ndarray,
    r: np.ndarray,
    maximum_components: int,
    folds: Sequence[np.ndarray],
    *,
    rank_tolerance: float = 1e-10,
) -> FPLSCVResult:
    """Select Raw and Arnoldi FPLS dimensions on identical local folds."""
    y, X = _validate_xy(y, X)
    sample_size, grid_size = X.shape
    K, r = _validate_moments(K, r, grid_size)
    if len(folds) < 2:
        raise ValueError("at least two folds are required")

    raw_sse = np.zeros(maximum_components)
    arnoldi_sse = np.zeros(maximum_components)
    raw_valid_all = np.ones(maximum_components, dtype=bool)
    arnoldi_valid_all = np.ones(maximum_components, dtype=bool)
    observed = np.zeros(sample_size, dtype=bool)

    for validation_indices in folds:
        validation_indices = np.asarray(validation_indices, dtype=int)
        if len(validation_indices) == 0:
            raise ValueError("folds must be nonempty")
        if np.any(validation_indices < 0) or np.any(validation_indices >= sample_size):
            raise ValueError("fold index outside sample")
        if np.any(observed[validation_indices]):
            raise ValueError("folds overlap")
        observed[validation_indices] = True

        training_mask = np.ones(sample_size, dtype=bool)
        training_mask[validation_indices] = False
        X_train = X[training_mask]
        y_train = y[training_mask]
        X_validation = X[validation_indices]
        y_validation = y[validation_indices]
        K_train, r_train = empirical_moments(X_train, y_train)

        raw = raw_fpls_path(
            y_train,
            X_train,
            K_train,
            r_train,
            maximum_components,
        )
        arnoldi = arnoldi_fpls_path(
            y_train,
            X_train,
            K_train,
            r_train,
            maximum_components,
            rank_tolerance=rank_tolerance,
        )
        raw_valid_all &= raw.valid
        arnoldi_valid_all &= arnoldi.valid

        if np.any(raw.valid):
            predictions = X_validation @ raw.beta[:, raw.valid] / grid_size
            with np.errstate(over="ignore", invalid="ignore"):
                errors = np.sum((y_validation[:, None] - predictions) ** 2, axis=0)
            indices = np.flatnonzero(raw.valid)
            finite = np.isfinite(errors)
            raw_sse[indices[finite]] += errors[finite]
            raw_valid_all[indices[~finite]] = False

        if np.any(arnoldi.valid):
            predictions = X_validation @ arnoldi.beta[:, arnoldi.valid] / grid_size
            errors = np.sum((y_validation[:, None] - predictions) ** 2, axis=0)
            indices = np.flatnonzero(arnoldi.valid)
            finite = np.isfinite(errors)
            arnoldi_sse[indices[finite]] += errors[finite]
            arnoldi_valid_all[indices[~finite]] = False

    if not np.all(observed):
        raise ValueError("folds do not cover every observation exactly once")

    raw_full = raw_fpls_path(y, X, K, r, maximum_components)
    arnoldi_full = arnoldi_fpls_path(
        y,
        X,
        K,
        r,
        maximum_components,
        rank_tolerance=rank_tolerance,
    )
    raw_candidates = raw_valid_all & raw_full.valid
    arnoldi_candidates = arnoldi_valid_all & arnoldi_full.valid
    if not np.any(raw_candidates):
        raise LinAlgError("Raw FPLS has no component valid in every fold")
    if not np.any(arnoldi_candidates):
        raise LinAlgError("Arnoldi FPLS has no valid component")

    cv_raw = np.full(maximum_components, np.inf)
    cv_arnoldi = np.full(maximum_components, np.inf)
    cv_raw[raw_candidates] = raw_sse[raw_candidates] / sample_size
    cv_arnoldi[arnoldi_candidates] = arnoldi_sse[arnoldi_candidates] / sample_size
    m_raw = int(np.argmin(cv_raw)) + 1
    m_arnoldi = int(np.argmin(cv_arnoldi)) + 1
    return FPLSCVResult(
        beta_raw=raw_full.beta[:, m_raw - 1],
        beta_arnoldi=arnoldi_full.beta[:, m_arnoldi - 1],
        m_raw=m_raw,
        m_arnoldi=m_arnoldi,
        cv_raw=cv_raw,
        cv_arnoldi=cv_arnoldi,
        raw_full_path=raw_full,
        arnoldi_full_path=arnoldi_full,
        raw_valid_all_folds=raw_valid_all,
    )


def fpcr_spectral_path(
    r: np.ndarray,
    K: np.ndarray,
    maximum_components: int,
) -> np.ndarray:
    """Return spectral-cutoff FPCR estimates for m=1,...,m_max."""
    r = np.asarray(r, dtype=float).reshape(-1)
    K = np.asarray(K, dtype=float)
    grid_size = len(r)
    if K.shape != (grid_size, grid_size) or maximum_components < 1:
        raise ValueError("K, r, and maximum_components are incompatible")
    K = 0.5 * (K + K.T)
    try:
        eigenvalues, eigenvectors = eigh(K, check_finite=False)
        order = np.argsort(eigenvalues)[::-1]
    except LinAlgError:
        eigenvectors, eigenvalues, _ = np.linalg.svd(K, full_matrices=False)
        order = np.arange(len(eigenvalues))
    threshold = np.finfo(float).eps * grid_size * max(
        float(np.max(np.abs(eigenvalues))),
        np.finfo(float).tiny,
    )
    positive = order[eigenvalues[order] > threshold]
    maximum = min(maximum_components, len(positive), grid_size - 1)
    if maximum == 0:
        raise LinAlgError("K has no numerically positive eigenvalues")
    path = np.zeros((grid_size, maximum))
    for component in range(maximum):
        indices = positive[: component + 1]
        values = eigenvalues[indices]
        vectors = eigenvectors[:, indices]
        path[:, component] = vectors @ ((vectors.T @ r) / values)
    return path


def select_fpcr_response_gcv(
    y: np.ndarray,
    X: np.ndarray,
    beta_path: np.ndarray,
) -> FPCRSelection:
    """Select the reported FPCR estimator by response-space GCV."""
    y, X = _validate_xy(y, X)
    sample_size, grid_size = X.shape
    beta_path = np.asarray(beta_path, dtype=float)
    if beta_path.ndim != 2 or beta_path.shape[0] != grid_size:
        raise ValueError("beta_path has incompatible dimensions")
    values = np.full(beta_path.shape[1], np.inf)
    for component in range(beta_path.shape[1]):
        count = component + 1
        residual = y - X @ beta_path[:, component] / grid_size
        values[component] = np.mean(residual**2) / (1.0 - count / sample_size) ** 2
    selected = int(np.argmin(values)) + 1
    return FPCRSelection(beta_path[:, selected - 1], selected, values)


def select_cg_variance_pilot(
    r: np.ndarray,
    K: np.ndarray,
    beta_path: np.ndarray,
) -> FPCRSelection:
    """Select the internal moment-space FPCR pilot used only by CG-FPLS."""
    r = np.asarray(r, dtype=float).reshape(-1)
    K = np.asarray(K, dtype=float)
    beta_path = np.asarray(beta_path, dtype=float)
    grid_size = len(r)
    if K.shape != (grid_size, grid_size) or beta_path.shape[0] != grid_size:
        raise ValueError("K, r, and beta_path are incompatible")
    values = np.full(beta_path.shape[1], np.inf)
    for component in range(beta_path.shape[1]):
        count = component + 1
        residual = r - K @ beta_path[:, component]
        values[component] = np.mean(residual**2) / (1.0 - count / grid_size) ** 2
    selected = int(np.argmin(values)) + 1
    return FPCRSelection(beta_path[:, selected - 1], selected, values)


def cg_fpls_path(
    r: np.ndarray,
    K: np.ndarray,
    maximum_components: int,
    tolerance: float = 1e-12,
) -> np.ndarray:
    """Return the CG-FPLS iterates for K beta = r."""
    r = np.asarray(r, dtype=float).reshape(-1)
    K = np.asarray(K, dtype=float)
    if K.shape != (len(r), len(r)) or maximum_components < 1:
        raise ValueError("K, r, and maximum_components are incompatible")
    K = 0.5 * (K + K.T)
    path = np.zeros((len(r), maximum_components))
    beta_previous = np.zeros(len(r))
    residual = r.copy()
    direction = r.copy()
    norm_K = max(float(np.linalg.norm(K, ord=np.inf)), np.finfo(float).tiny)

    for component in range(maximum_components):
        K_residual = K @ residual
        K_direction = K @ direction
        numerator = float(residual @ K_residual)
        denominator = float(K_direction @ K_direction)
        scale = norm_K * max(float(np.linalg.norm(direction)), np.finfo(float).tiny)
        if (
            not np.isfinite(numerator)
            or not np.isfinite(denominator)
            or numerator <= 0.0
            or denominator <= 0.0
            or np.linalg.norm(K_direction) <= tolerance * scale
        ):
            path[:, component:] = beta_previous[:, None]
            break
        alpha = numerator / denominator
        beta_current = beta_previous + alpha * direction
        residual_new = residual - alpha * K_direction
        numerator_new = float(residual_new @ (K @ residual_new))
        gamma = numerator_new / numerator
        if not (
            np.all(np.isfinite(beta_current))
            and np.all(np.isfinite(residual_new))
            and np.isfinite(gamma)
        ):
            raise FloatingPointError("CG-FPLS produced a nonfinite iterate")
        path[:, component] = beta_current
        beta_previous = beta_current
        residual = residual_new
        direction = residual_new + gamma * direction
    return path


def _apply_cg_stopping_rule(
    path: np.ndarray,
    r: np.ndarray,
    K: np.ndarray,
    *,
    sigma2: float,
    X_norm: float,
    sample_size: int,
    tau: float,
    delta: float,
) -> Tuple[np.ndarray, int, bool, np.ndarray, float]:
    residuals = np.column_stack((r, r[:, None] - K @ path))
    moments = np.sqrt(np.mean(residuals**2, axis=0))
    threshold = tau * np.sqrt(2.0 * sigma2 * X_norm / (delta * sample_size))
    reached = np.flatnonzero(moments[1:] <= threshold)
    if len(reached):
        selected = int(reached[0] + 1)
        threshold_reached = True
    else:
        selected = path.shape[1]
        threshold_reached = False
    return (
        path[:, selected - 1],
        selected,
        threshold_reached,
        moments,
        float(threshold),
    )


def select_cg_fpls_code(
    path: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    r: np.ndarray,
    K: np.ndarray,
    pilot_beta: np.ndarray,
    *,
    tau: float,
    delta: float,
) -> CGSelection:
    """Apply the single released-code CG-FPLS stopping rule for m >= 1."""
    y, X = _validate_xy(y, X)
    sample_size, grid_size = X.shape
    K, r = _validate_moments(K, r, grid_size)
    path = np.asarray(path, dtype=float)
    pilot_beta = np.asarray(pilot_beta, dtype=float).reshape(-1)
    if path.ndim != 2 or path.shape[0] != grid_size or path.shape[1] < 1:
        raise ValueError("path must be a nonempty T-by-m array")
    if pilot_beta.shape != (grid_size,) or not np.all(np.isfinite(pilot_beta)):
        raise ValueError("pilot_beta must be a finite vector of length T")
    if not 0.0 < delta < 1.0 or tau <= 1.0:
        raise ValueError("tau must exceed one and delta must lie in (0,1)")

    sigma2 = max(float(np.mean((y - X @ pilot_beta / grid_size) ** 2)), 0.0)
    X_norm = float(np.mean(np.sum(X**2, axis=1) / grid_size))
    beta, selected, reached, moments, threshold = _apply_cg_stopping_rule(
        path,
        r,
        K,
        sigma2=sigma2,
        X_norm=X_norm,
        sample_size=sample_size,
        tau=tau,
        delta=delta,
    )
    return CGSelection(beta, selected, reached, sigma2, threshold, moments)


def relative_difference(left: np.ndarray, right: np.ndarray) -> float:
    denominator = max(
        float(np.linalg.norm(left)),
        float(np.linalg.norm(right)),
        np.finfo(float).tiny,
    )
    return float(np.linalg.norm(left - right) / denominator)


def run_self_tests() -> None:
    """Check the fixed four-method implementation before a simulation."""
    rng = np.random.default_rng(78123)
    sample_size, grid_size = 80, 18
    X = rng.normal(size=(sample_size, grid_size))
    beta_true = rng.normal(size=grid_size)
    y = X @ beta_true / grid_size + 0.1 * rng.normal(size=sample_size)
    K, r = empirical_moments(X, y)

    raw = raw_fpls_path(y, X, K, r, 4)
    arnoldi = arnoldi_fpls_path(
        y,
        X,
        K,
        r,
        4,
        rank_tolerance=1e-12,
    )
    if not np.all(raw.valid) or not np.all(arnoldi.valid):
        raise AssertionError("FPLS path failed on a well-conditioned test")
    for component in range(4):
        if relative_difference(
            raw.fitted[:, component], arnoldi.fitted[:, component]
        ) > 1e-7:
            raise AssertionError("Raw and Arnoldi FPLS disagree at fixed m")
    defect = np.linalg.norm(
        arnoldi.basis.T @ arnoldi.basis - np.eye(arnoldi.basis.shape[1]),
        ord=2,
    )
    if defect > 1e-12:
        raise AssertionError("Arnoldi basis is not orthonormal")

    folds = make_folds(sample_size, 5, np.random.default_rng(9182))
    selected_fpls = cross_validate_fpls(
        y,
        X,
        K,
        r,
        4,
        folds,
        rank_tolerance=1e-12,
    )
    if not (
        1 <= selected_fpls.m_raw <= 4
        and 1 <= selected_fpls.m_arnoldi <= 4
    ):
        raise AssertionError("FPLS cross-validation returned an invalid fit")

    fpcr_path = fpcr_spectral_path(r, K, 8)
    fpcr = select_fpcr_response_gcv(y, X, fpcr_path)
    pilot = select_cg_variance_pilot(r, K, fpcr_path)
    response_residual = y - X @ fpcr.beta / grid_size
    expected_gcv = np.mean(response_residual**2) / (
        1.0 - fpcr.selected_components / sample_size
    ) ** 2
    if not np.isclose(
        expected_gcv,
        fpcr.criterion_values[fpcr.selected_components - 1],
        rtol=1e-12,
        atol=1e-14,
    ):
        raise AssertionError("FPCR response-GCV scaling is incorrect")

    cg_path = cg_fpls_path(r, K, 8)
    cg = select_cg_fpls_code(
        cg_path,
        X,
        y,
        r,
        K,
        pilot.beta,
        tau=1.01,
        delta=0.1,
    )
    if not (
        1 <= cg.selected_components <= 8
        and cg.threshold > 0.0
        and np.all(np.isfinite(cg.beta))
    ):
        raise AssertionError("CG-FPLS-code returned an invalid fit")

    grid = np.linspace(0.0, 1.0, 30)
    basis = create_cosine_basis(grid, 10)
    models = make_model_specs(10, basis)
    if not np.array_equal(models[0].eigenvalues, models[1].eigenvalues):
        raise AssertionError("Model 2 must preserve Model 1's spectrum")
    if not np.array_equal(models[0].beta, models[2].beta):
        raise AssertionError("Model 3 must preserve Model 1's slope")
    if len(METHODS) != 4 or tuple(model.name for model in models) != MODEL_NAMES:
        raise AssertionError("unexpected methods or simulation setups")

    print(
        "Self-tests passed: exactly four estimators, exactly three setups, "
        "Raw/Arnoldi fixed-m equivalence, CGS2 orthogonality, fold-local CV, "
        "response-GCV FPCR, and the CG-FPLS-code stopping rule."
    )
