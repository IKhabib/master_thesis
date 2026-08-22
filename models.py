"""Raw versus reorthogonalized APLS for scalar-on-function regression.

This standalone program compares four estimators under the three simulation
designs used in ``model_8_corrected.py``:

1. FPCR (called PCA in the earlier script);
2. Babii et al.'s discrepancy-stopped conjugate-gradient FPLS;
3. raw APLS, implemented from the nonorthogonal powers
       H_m = [r, K r, ..., K^(m-1) r]
   and the unregularized normal equations; and
4. reorthogonalized APLS, which represents the same Krylov spaces with an
   orthonormal Arnoldi basis and solves response least squares by QR.

The raw routine deliberately applies no column scaling, ridge penalty,
pseudoinverse, truncated SVD, or Gram--Schmidt step.  It is therefore a
faithful finite-precision baseline for the explicit APLS construction in
Section 4.1 of Delaigle and Hall (2012), adapted to the known-zero-mean
simulation convention used here.  A failed raw solve is recorded rather than
silently repaired.

The reorthogonalized routine defaults to CGS2: classical Gram--Schmidt with
one reorthogonalization pass.  This is the accurate name for the vectorized
two-projection implementation used in the earlier project code.  The
published-style one-pass MGS benchmark and the other variants are available through
``--orthogonalization mgs1``, ``cgs1``, or ``mgs2``.  Thus the marginal effect
of a second pass can be studied rather than attributed to APLS itself.

Discretization
--------------
    prediction = X @ beta / T
    r          = X.T @ y / n
    K          = X.T @ X / (n * T)

Examples
--------
Quick checked run::

    python apls_raw_vs_reorthogonalized.py --replications 5

Larger comparison::

    python apls_raw_vs_reorthogonalized.py \
        --replications 500 --m-max 70 \
        --output-dir apls_raw_vs_reorth_results

References
----------
Babii, A., Carrasco, M. and Tsafack, I. (2025). Functional Partial
Least-Squares: Adaptive Estimation and Inference.
https://ababii.github.io/papers/functional_pls.pdf

Delaigle, A. and Hall, P. (2012). Methodology and theory for partial least
squares applied to functional data. Annals of Statistics 40, 322--352.
https://doi.org/10.1214/11-AOS958

Giraud, L., Langou, J. and Rozloznik, M. (2005). The loss of orthogonality in
the Gram--Schmidt orthogonalization process. Computers & Mathematics with
Applications 50, 1069--1075.
https://doi.org/10.1016/j.camwa.2005.08.009
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import LinAlgError, eigh, solve_triangular


METHOD_COLORS = {
    "CG-FPLS": "#0072B2",
    "APLS-raw": "#D55E00",
    "APLS-CGS1": "#009E73",
    "APLS-CGS2": "#009E73",
    "APLS-MGS1": "#009E73",
    "APLS-MGS2": "#009E73",
    "FPCR": "#CC79A7",
}
STABLE_COLOR = "#009E73"
MODEL_COLORS = {
    "Model 1": "#0072B2",
    "Model 2": "#D55E00",
    "Model 3": "#009E73",
}
CONDITION_CAP = 1.0 / np.finfo(float).eps


@dataclass(frozen=True)
class ModelSpec:
    """One simulation design."""

    name: str
    eigenvalues: np.ndarray
    beta: np.ndarray


@dataclass
class APLSPath:
    """A nested APLS regression path and its numerical diagnostics."""

    beta: np.ndarray
    fitted: np.ndarray
    valid: np.ndarray
    basis: np.ndarray
    basis_condition: np.ndarray
    design_condition: np.ndarray
    orthogonality_defect: np.ndarray
    effective_dimension: int


@dataclass
class APLSCVResult:
    """Cross-validated raw and reorthogonalized APLS fits."""

    beta_raw: np.ndarray
    beta_reorth: np.ndarray
    m_raw: int
    m_reorth: int
    cv_raw: np.ndarray
    cv_reorth: np.ndarray
    raw_full_path: APLSPath
    reorth_full_path: APLSPath
    raw_valid_all_folds: np.ndarray


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


def _validate_matrices(K: np.ndarray, r: np.ndarray, T: int) -> Tuple[np.ndarray, np.ndarray]:
    K = np.asarray(K, dtype=float)
    r = np.asarray(r, dtype=float).reshape(-1)
    if K.shape != (T, T) or r.shape != (T,):
        raise ValueError("K and r have incompatible dimensions")
    if not np.all(np.isfinite(K)) or not np.all(np.isfinite(r)):
        raise ValueError("K and r must be finite")
    return 0.5 * (K + K.T), r


def create_cosine_basis(
    grid: np.ndarray, J: int, convention: str = "babii"
) -> np.ndarray:
    """Return the cosine functions used by the simulation as a J-by-T array."""
    grid = np.asarray(grid, dtype=float).reshape(-1)
    if J < 1 or len(grid) < 2 or np.any(np.diff(grid) <= 0):
        raise ValueError("J must be positive and grid must be strictly increasing")
    if convention == "babii":
        frequencies = np.arange(1, J + 1)
    elif convention == "standard":
        frequencies = np.arange(J)
    else:
        raise ValueError("convention must be 'babii' or 'standard'")
    basis = np.sqrt(2.0) * np.cos(np.pi * np.outer(grid, frequencies))
    basis[:, 0] = 1.0
    return basis.T


def make_model_specs(J: int, basis: np.ndarray) -> List[ModelSpec]:
    """Construct the three DGPs used in the existing estimation study."""
    if J < 5 or basis.shape[0] != J:
        raise ValueError("J must be at least five and match basis.shape[0]")
    index = np.arange(1, J + 1, dtype=float)
    b1 = 4.0 / index ** 2.7
    b2 = b1.copy()
    b2[:5] = 4.0
    lambda1 = 2.0 / index ** 1.1
    lambda3 = lambda1.copy()
    lambda3[:5] = 2.0
    return [
        ModelSpec("Model 1", lambda1, basis.T @ b1),
        ModelSpec("Model 2", lambda1, basis.T @ b2),
        ModelSpec("Model 3", lambda3, basis.T @ b1),
    ]


def _capped_condition(columns: np.ndarray) -> float:
    """Return a finite diagnostic condition number capped at 1/eps."""
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


def _normalized_prefix_condition(columns: np.ndarray) -> float:
    """Condition after unit-norm column scaling, used only as a diagnostic."""
    norms = np.linalg.norm(columns, axis=0)
    if np.any(norms <= 0.0) or not np.all(np.isfinite(norms)):
        return CONDITION_CAP
    return _capped_condition(columns / norms)


def _raw_krylov_matrix(K: np.ndarray, r: np.ndarray, m_max: int) -> np.ndarray:
    """Form [r, Kr, ..., K^(m-1)r] without any stabilization."""
    T = len(r)
    H = np.zeros((T, m_max))
    H[:, 0] = r
    for component in range(1, m_max):
        H[:, component] = K @ H[:, component - 1]
    return H


def raw_apls_path(
    y: np.ndarray,
    X: np.ndarray,
    K: np.ndarray,
    r: np.ndarray,
    m_max: int,
) -> APLSPath:
    """Compute the deliberately unstabilized APLS path.

    For every prefix m, this routine solves

        (Z_m' Z_m) a_m = Z_m' y,   beta_m = H_m a_m,

    where H_m contains the raw covariance powers and Z_m = X H_m / T.
    There is intentionally no numerical rescue.  Invalid prefixes are filled
    with NaN and identified by ``valid``.
    """
    y, X = _validate_xy(y, X)
    n, T = X.shape
    del n
    if m_max < 1:
        raise ValueError("m_max must be positive")
    K, r = _validate_matrices(K, r, T)
    H = _raw_krylov_matrix(K, r, m_max)
    Z = X @ H / T
    gram = Z.T @ Z
    rhs = Z.T @ y

    beta_path = np.full((T, m_max), np.nan)
    fitted_path = np.full((len(y), m_max), np.nan)
    valid = np.zeros(m_max, dtype=bool)
    basis_condition = np.full(m_max, CONDITION_CAP)
    design_condition = np.full(m_max, CONDITION_CAP)
    orthogonality_defect = np.full(m_max, np.nan)

    for component in range(m_max):
        prefix = component + 1
        H_prefix = H[:, :prefix]
        Z_prefix = Z[:, :prefix]
        basis_condition[component] = _normalized_prefix_condition(H_prefix)
        design_condition[component] = _normalized_prefix_condition(Z_prefix)
        try:
            coefficients = np.linalg.solve(
                gram[:prefix, :prefix], rhs[:prefix]
            )
        except np.linalg.LinAlgError:
            continue
        with np.errstate(over="ignore", invalid="ignore"):
            beta_current = H_prefix @ coefficients
            fitted_current = Z_prefix @ coefficients
        if np.all(np.isfinite(beta_current)) and np.all(np.isfinite(fitted_current)):
            beta_path[:, component] = beta_current
            fitted_path[:, component] = fitted_current
            valid[component] = True

    return APLSPath(
        beta=beta_path,
        fitted=fitted_path,
        valid=valid,
        basis=H,
        basis_condition=basis_condition,
        design_condition=design_condition,
        orthogonality_defect=orthogonality_defect,
        effective_dimension=int(np.count_nonzero(valid)),
    )


def _orthogonalize_candidate(
    candidate: np.ndarray,
    basis: np.ndarray,
    method: str,
) -> np.ndarray:
    """Orthogonalize by one- or two-pass CGS/MGS."""
    if method not in {"cgs1", "cgs2", "mgs1", "mgs2"}:
        raise ValueError("method must be cgs1, cgs2, mgs1, or mgs2")
    passes = int(method[-1])
    family = method[:3]
    if family == "cgs":
        for _ in range(passes):
            candidate = candidate - basis @ (basis.T @ candidate)
    else:
        for _ in range(passes):
            for column in range(basis.shape[1]):
                q = basis[:, column]
                candidate = candidate - float(q @ candidate) * q
    return candidate


def reorthogonalized_krylov_basis(
    K: np.ndarray,
    r: np.ndarray,
    m_max: int,
    *,
    method: str = "cgs2",
    rank_tolerance: float = 1e-10,
) -> np.ndarray:
    """Construct an orthonormal Arnoldi basis of the APLS Krylov spaces."""
    T = len(r)
    if m_max < 1 or rank_tolerance <= 0:
        raise ValueError("m_max and rank_tolerance must be positive")
    norm_r = float(np.linalg.norm(r))
    if norm_r == 0.0:
        return np.zeros((T, 0))
    Q = np.zeros((T, m_max))
    Q[:, 0] = r / norm_r
    dimension = 1
    norm_K = max(float(np.linalg.norm(K, ord=np.inf)), np.finfo(float).tiny)

    while dimension < m_max:
        candidate = K @ Q[:, dimension - 1]
        candidate = _orthogonalize_candidate(
            candidate, Q[:, :dimension], method
        )
        candidate_norm = float(np.linalg.norm(candidate))
        if not np.isfinite(candidate_norm) or candidate_norm <= rank_tolerance * norm_K:
            break
        Q[:, dimension] = candidate / candidate_norm
        dimension += 1
    return Q[:, :dimension]


def reorthogonalized_apls_path(
    y: np.ndarray,
    X: np.ndarray,
    K: np.ndarray,
    r: np.ndarray,
    m_max: int,
    *,
    method: str = "cgs2",
    rank_tolerance: float = 1e-10,
) -> APLSPath:
    """Fit APLS over reorthogonalized Krylov bases using response QR."""
    y, X = _validate_xy(y, X)
    _, T = X.shape
    if m_max < 1:
        raise ValueError("m_max must be positive")
    K, r = _validate_matrices(K, r, T)
    Q = reorthogonalized_krylov_basis(
        K, r, m_max, method=method, rank_tolerance=rank_tolerance
    )
    beta_path = np.zeros((T, m_max))
    fitted_path = np.zeros((len(y), m_max))
    valid = np.zeros(m_max, dtype=bool)
    basis_condition = np.full(m_max, np.nan)
    design_condition = np.full(m_max, np.nan)
    orthogonality_defect = np.full(m_max, np.nan)

    dimension = Q.shape[1]
    if dimension == 0:
        return APLSPath(
            beta_path,
            fitted_path,
            valid,
            Q,
            basis_condition,
            design_condition,
            orthogonality_defect,
            0,
        )

    design = X @ Q / T
    q_design, r_design = np.linalg.qr(design, mode="reduced")
    projected_y = q_design.T @ y
    diagonal_reference = max(abs(float(r_design[0, 0])), np.finfo(float).tiny)
    fitted_dimension = 0

    for component in range(dimension):
        prefix = component + 1
        Q_prefix = Q[:, :prefix]
        design_prefix = design[:, :prefix]
        basis_condition[component] = _capped_condition(Q_prefix)
        design_condition[component] = _capped_condition(design_prefix)
        orthogonality_defect[component] = float(
            np.linalg.norm(Q_prefix.T @ Q_prefix - np.eye(prefix), ord=2)
        )
        if abs(float(r_design[component, component])) <= (
            rank_tolerance * diagonal_reference
        ):
            break
        coefficients = solve_triangular(
            r_design[:prefix, :prefix],
            projected_y[:prefix],
            lower=False,
            check_finite=False,
        )
        beta_current = Q_prefix @ coefficients
        fitted_current = X @ beta_current / T
        if not (
            np.all(np.isfinite(beta_current))
            and np.all(np.isfinite(fitted_current))
        ):
            break
        beta_path[:, component] = beta_current
        fitted_path[:, component] = fitted_current
        valid[component] = True
        fitted_dimension = prefix

    if fitted_dimension > 0 and fitted_dimension < m_max:
        beta_path[:, fitted_dimension:] = beta_path[:, [fitted_dimension - 1]]
        fitted_path[:, fitted_dimension:] = fitted_path[:, [fitted_dimension - 1]]
        valid[fitted_dimension:] = True
        basis_condition[fitted_dimension:] = basis_condition[fitted_dimension - 1]
        design_condition[fitted_dimension:] = design_condition[fitted_dimension - 1]
        orthogonality_defect[fitted_dimension:] = orthogonality_defect[
            fitted_dimension - 1
        ]

    return APLSPath(
        beta=beta_path,
        fitted=fitted_path,
        valid=valid,
        basis=Q,
        basis_condition=basis_condition,
        design_condition=design_condition,
        orthogonality_defect=orthogonality_defect,
        effective_dimension=fitted_dimension,
    )


def apls_cv_compare(
    y: np.ndarray,
    X: np.ndarray,
    K: np.ndarray,
    r: np.ndarray,
    m_max: int,
    folds: Sequence[np.ndarray],
    *,
    orthogonalization: str = "cgs2",
    rank_tolerance: float = 1e-10,
) -> APLSCVResult:
    """Cross-validate raw and reorthogonalized APLS on identical folds."""
    y, X = _validate_xy(y, X)
    n, T = X.shape
    K, r = _validate_matrices(K, r, T)
    if m_max < 1:
        raise ValueError("m_max must be positive")
    if len(folds) < 2:
        raise ValueError("at least two folds are required")

    cv_sse_raw = np.zeros(m_max)
    cv_sse_reorth = np.zeros(m_max)
    raw_valid_all = np.ones(m_max, dtype=bool)
    reorth_valid_all = np.ones(m_max, dtype=bool)
    observed = np.zeros(n, dtype=bool)

    for validation_indices in folds:
        validation_indices = np.asarray(validation_indices, dtype=int)
        if len(validation_indices) == 0:
            raise ValueError("folds must be nonempty")
        if np.any(validation_indices < 0) or np.any(validation_indices >= n):
            raise ValueError("fold index outside the sample")
        if np.any(observed[validation_indices]):
            raise ValueError("folds overlap")
        observed[validation_indices] = True
        training_mask = np.ones(n, dtype=bool)
        training_mask[validation_indices] = False
        X_train = X[training_mask]
        y_train = y[training_mask]
        X_validation = X[validation_indices]
        y_validation = y[validation_indices]
        n_train = len(y_train)
        K_train = X_train.T @ X_train / (n_train * T)
        r_train = X_train.T @ y_train / n_train

        raw_path = raw_apls_path(
            y_train, X_train, K_train, r_train, m_max
        )
        reorth_path = reorthogonalized_apls_path(
            y_train,
            X_train,
            K_train,
            r_train,
            m_max,
            method=orthogonalization,
            rank_tolerance=rank_tolerance,
        )
        raw_valid_all &= raw_path.valid
        reorth_valid_all &= reorth_path.valid

        valid_raw = raw_path.valid
        if np.any(valid_raw):
            raw_predictions = X_validation @ raw_path.beta[:, valid_raw] / T
            with np.errstate(over="ignore", invalid="ignore"):
                raw_errors = np.sum(
                    (y_validation[:, None] - raw_predictions) ** 2, axis=0
                )
            finite_raw_errors = np.isfinite(raw_errors)
            raw_indices = np.flatnonzero(valid_raw)
            cv_sse_raw[raw_indices[finite_raw_errors]] += raw_errors[
                finite_raw_errors
            ]
            raw_valid_all[raw_indices[~finite_raw_errors]] = False

        valid_reorth = reorth_path.valid
        if np.any(valid_reorth):
            reorth_predictions = (
                X_validation @ reorth_path.beta[:, valid_reorth] / T
            )
            reorth_errors = np.sum(
                (y_validation[:, None] - reorth_predictions) ** 2, axis=0
            )
            finite_reorth_errors = np.isfinite(reorth_errors)
            reorth_indices = np.flatnonzero(valid_reorth)
            cv_sse_reorth[reorth_indices[finite_reorth_errors]] += (
                reorth_errors[finite_reorth_errors]
            )
            reorth_valid_all[reorth_indices[~finite_reorth_errors]] = False

    if not np.all(observed):
        raise ValueError("folds do not cover every observation exactly once")

    raw_full = raw_apls_path(y, X, K, r, m_max)
    reorth_full = reorthogonalized_apls_path(
        y,
        X,
        K,
        r,
        m_max,
        method=orthogonalization,
        rank_tolerance=rank_tolerance,
    )
    raw_candidates = raw_valid_all & raw_full.valid
    reorth_candidates = reorth_valid_all & reorth_full.valid
    if not np.any(raw_candidates):
        raise LinAlgError("raw APLS has no valid component in every fold")
    if not np.any(reorth_candidates):
        raise LinAlgError("reorthogonalized APLS has no valid component")

    cv_raw = np.full(m_max, np.inf)
    cv_reorth = np.full(m_max, np.inf)
    cv_raw[raw_candidates] = cv_sse_raw[raw_candidates] / n
    cv_reorth[reorth_candidates] = cv_sse_reorth[reorth_candidates] / n
    m_raw = int(np.argmin(cv_raw)) + 1
    m_reorth = int(np.argmin(cv_reorth)) + 1

    return APLSCVResult(
        beta_raw=raw_full.beta[:, m_raw - 1],
        beta_reorth=reorth_full.beta[:, m_reorth - 1],
        m_raw=m_raw,
        m_reorth=m_reorth,
        cv_raw=cv_raw,
        cv_reorth=cv_reorth,
        raw_full_path=raw_full,
        reorth_full_path=reorth_full,
        raw_valid_all_folds=raw_valid_all,
    )


def cg_fpls_path(
    r: np.ndarray,
    K: np.ndarray,
    m_max: int,
    tolerance: float = 1e-12,
) -> np.ndarray:
    """Conjugate-gradient FPLS iterates for K beta = r."""
    r = np.asarray(r, dtype=float).reshape(-1)
    K = np.asarray(K, dtype=float)
    if K.shape != (len(r), len(r)) or m_max < 1:
        raise ValueError("K, r, and m_max have incompatible dimensions")
    K = 0.5 * (K + K.T)
    T = len(r)
    path = np.zeros((T, m_max))
    beta_previous = np.zeros(T)
    residual = r.copy()
    direction = r.copy()
    norm_K = max(float(np.linalg.norm(K, ord=np.inf)), np.finfo(float).tiny)

    for component in range(m_max):
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
            raise FloatingPointError("CG-FPLS produced a non-finite iterate")
        path[:, component] = beta_current
        beta_previous = beta_current
        residual = residual_new
        direction = residual_new + gamma * direction
    return path


def select_cg_fpls(
    path: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    r: np.ndarray,
    K: np.ndarray,
    beta_fpcr: np.ndarray,
    *,
    tau: float,
    delta: float,
) -> Tuple[np.ndarray, int, bool]:
    """Apply Babii et al.'s discrepancy stopping rule."""
    y, X = _validate_xy(y, X)
    n, T = X.shape
    if not 0.0 < delta < 1.0 or tau <= 0.0:
        raise ValueError("tau must be positive and delta must lie in (0,1)")
    sigma2 = max(float(np.mean((y - X @ beta_fpcr / T) ** 2)), 0.0)
    X_norm = float(np.mean(np.sum(X ** 2, axis=1) / T))
    residuals = K @ path - r[:, None]
    moments = np.sqrt(np.mean(residuals ** 2, axis=0))
    threshold = tau * np.sqrt(2.0 * sigma2 * X_norm / (delta * n))
    reached = np.flatnonzero(moments <= threshold)
    if len(reached):
        m_selected = int(reached[0]) + 1
    else:
        m_selected = path.shape[1]
    return path[:, m_selected - 1], m_selected, bool(len(reached))


def fpcr_gcv(r: np.ndarray, K: np.ndarray, m_max: int) -> Tuple[np.ndarray, int]:
    """Functional principal-component regression with moment-space GCV."""
    r = np.asarray(r, dtype=float).reshape(-1)
    K = np.asarray(K, dtype=float)
    T = len(r)
    if K.shape != (T, T) or m_max < 1:
        raise ValueError("K, r, and m_max have incompatible dimensions")
    K = 0.5 * (K + K.T)
    try:
        eigenvalues, eigenvectors = eigh(K, check_finite=False)
    except LinAlgError:
        eigenvectors, eigenvalues, _ = np.linalg.svd(K, full_matrices=False)
    threshold = np.finfo(float).eps * T * max(float(np.max(eigenvalues)), 1.0)
    positive = np.flatnonzero(eigenvalues > threshold)[::-1]
    maximum = min(m_max, len(positive), T - 1)
    if maximum == 0:
        raise LinAlgError("K has no numerically positive eigenvalues")
    beta_path = np.zeros((T, maximum))
    gcv = np.full(maximum, np.inf)
    for component in range(maximum):
        count = component + 1
        indices = positive[:count]
        values = eigenvalues[indices]
        vectors = eigenvectors[:, indices]
        beta_path[:, component] = vectors @ ((vectors.T @ r) / values)
        residual = r - K @ beta_path[:, component]
        gcv[component] = np.mean(residual ** 2) / (1.0 - count / T) ** 2
    selected = int(np.argmin(gcv)) + 1
    return beta_path[:, selected - 1], selected


def _relative_difference(left: np.ndarray, right: np.ndarray) -> float:
    denominator = max(
        float(np.linalg.norm(left)),
        float(np.linalg.norm(right)),
        np.finfo(float).tiny,
    )
    return float(np.linalg.norm(left - right) / denominator)


def _safe_mean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    return float(np.mean(array)) if len(array) else math.nan


def _safe_median(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    return float(np.median(array)) if len(array) else math.nan


def _standard_error(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if len(array) < 2:
        return math.nan
    return float(np.std(array, ddof=1) / np.sqrt(len(array)))


def _write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    """Write a rectangular sequence of dictionaries."""
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _is_complete_pdf(path: Path) -> bool:
    """Check the PDF signature and end-of-file marker."""
    try:
        if path.stat().st_size < 100:
            return False
        with path.open("rb") as handle:
            header = handle.read(5)
            handle.seek(max(0, path.stat().st_size - 2048))
            trailer = handle.read()
    except OSError:
        return False
    return header == b"%PDF-" and b"%%EOF" in trailer


def _save_pdf(fig: plt.Figure, path: Path) -> None:
    """Atomically save and validate a Matplotlib PDF, retrying transient writes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(f".{path.stem}.staging.pdf")
    last_error: Optional[BaseException] = None
    for _ in range(3):
        try:
            fig.savefig(staging, bbox_inches="tight")
            if _is_complete_pdf(staging):
                os.replace(staging, path)
                return
            last_error = OSError(f"incomplete PDF write: {staging}")
        except (OSError, ValueError) as error:
            last_error = error
        staging.unlink(missing_ok=True)
    raise OSError(f"could not create a complete PDF at {path}") from last_error


def _make_folds(
    n: int, k_folds: int, rng: np.random.Generator
) -> List[np.ndarray]:
    if not 2 <= k_folds <= n:
        raise ValueError("k_folds must be between two and n")
    return [np.asarray(part, dtype=int) for part in np.array_split(rng.permutation(n), k_folds)]


def run_simulation(
    *,
    replications: int,
    n: int,
    J: int,
    T: int,
    m_max: int,
    k_folds: int,
    noise_sd: float,
    tau: float,
    delta: float,
    seed: int,
    basis_convention: str,
    orthogonalization: str,
    rank_tolerance: float,
    output_dir: Path,
    make_plots: bool,
) -> Dict[str, object]:
    """Run all three models, write results, and return in-memory outputs."""
    if replications < 1 or n < 2 or J < 5 or T < 2 or m_max < 1:
        raise ValueError("replications, n, J, T, and m_max are too small")
    if noise_sd < 0.0:
        raise ValueError("noise_sd must be non-negative")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    grid = np.linspace(0.0, 1.0, T)
    basis = create_cosine_basis(grid, J, convention=basis_convention)
    models = make_model_specs(J, basis)
    stable_label = f"APLS-{orthogonalization.upper()}"
    methods = ("CG-FPLS", "APLS-raw", stable_label, "FPCR")
    seed_sequences = np.random.SeedSequence(seed).spawn(len(models))

    replication_rows: List[Dict[str, object]] = []
    diagnostic_rows: List[Dict[str, object]] = []
    path_rows: List[Dict[str, object]] = []
    raw_conditions = np.full((len(models), replications, m_max), np.nan)
    reorth_conditions = np.full((len(models), replications, m_max), np.nan)
    reorth_defects = np.full((len(models), replications, m_max), np.nan)
    beta_path_discrepancy = np.full(
        (len(models), replications, m_max), np.nan
    )
    fitted_path_discrepancy = np.full(
        (len(models), replications, m_max), np.nan
    )
    selected_components = np.zeros(
        (len(models), replications, len(methods)), dtype=int
    )
    ise = np.full((len(models), replications, len(methods)), np.nan)
    mspe = np.full((len(models), replications, len(methods)), np.nan)
    raw_valid = np.zeros((len(models), replications, m_max), dtype=bool)
    representative: Dict[str, Dict[str, np.ndarray]] = {}

    print("=" * 72)
    print("RAW VERSUS ORTHOGONALIZED APLS")
    print("=" * 72)
    print(
        f"Models=3, replications/model={replications}, n={n}, T={T}, "
        f"J={J}, m_max={m_max}, folds={k_folds}"
    )
    print(f"Orthogonalization: {orthogonalization.upper()}")

    for model_index, (model, seed_sequence) in enumerate(
        zip(models, seed_sequences)
    ):
        rng = np.random.default_rng(seed_sequence)
        sqrt_eigenvalues = np.sqrt(model.eigenvalues)
        print(f"\n{model.name}")

        for replication in range(replications):
            if replication == 0 or (replication + 1) % max(1, replications // 10) == 0:
                print(f"  replication {replication + 1}/{replications}")

            scores = rng.normal(size=(n, J))
            X = (scores * sqrt_eigenvalues) @ basis
            y = X @ model.beta / T + rng.normal(0.0, noise_sd, n)
            K = X.T @ X / (n * T)
            r = X.T @ y / n

            beta_fpcr, m_fpcr = fpcr_gcv(r, K, m_max)
            cg_path = cg_fpls_path(r, K, m_max)
            beta_cg, m_cg, cg_threshold_reached = select_cg_fpls(
                cg_path,
                X,
                y,
                r,
                K,
                beta_fpcr,
                tau=tau,
                delta=delta,
            )
            folds = _make_folds(n, k_folds, rng)
            apls = apls_cv_compare(
                y,
                X,
                K,
                r,
                m_max,
                folds,
                orthogonalization=orthogonalization,
                rank_tolerance=rank_tolerance,
            )

            beta_by_method = {
                "CG-FPLS": beta_cg,
                "APLS-raw": apls.beta_raw,
                stable_label: apls.beta_reorth,
                "FPCR": beta_fpcr,
            }
            m_by_method = {
                "CG-FPLS": m_cg,
                "APLS-raw": apls.m_raw,
                stable_label: apls.m_reorth,
                "FPCR": m_fpcr,
            }

            test_scores = rng.normal(size=(n, J))
            X_test = (test_scores * sqrt_eigenvalues) @ basis
            y_test = X_test @ model.beta / T + rng.normal(0.0, noise_sd, n)

            for method_index, method in enumerate(methods):
                beta_hat = beta_by_method[method]
                with np.errstate(over="ignore", invalid="ignore"):
                    ise_value = float(np.mean((beta_hat - model.beta) ** 2))
                    prediction = X_test @ beta_hat / T
                    mspe_value = float(np.mean((y_test - prediction) ** 2))
                ise[model_index, replication, method_index] = ise_value
                mspe[model_index, replication, method_index] = mspe_value
                selected_components[model_index, replication, method_index] = (
                    m_by_method[method]
                )
                replication_rows.append(
                    {
                        "model": model.name,
                        "replication": replication + 1,
                        "method": method,
                        "selected_components": m_by_method[method],
                        "ise": ise_value,
                        "mspe": mspe_value,
                        "finite_result": bool(
                            np.isfinite(ise_value) and np.isfinite(mspe_value)
                        ),
                    }
                )

            raw_conditions[model_index, replication] = (
                apls.raw_full_path.basis_condition
            )
            reorth_conditions[model_index, replication] = (
                apls.reorth_full_path.basis_condition
            )
            reorth_defects[model_index, replication] = (
                apls.reorth_full_path.orthogonality_defect
            )
            raw_valid[model_index, replication] = apls.raw_full_path.valid

            for component in range(m_max):
                prefix = component + 1
                same_m_valid = bool(
                    apls.raw_full_path.valid[component]
                    and apls.reorth_full_path.valid[component]
                )
                if same_m_valid:
                    beta_gap = _relative_difference(
                        apls.raw_full_path.beta[:, component],
                        apls.reorth_full_path.beta[:, component],
                    )
                    fitted_gap = _relative_difference(
                        apls.raw_full_path.fitted[:, component],
                        apls.reorth_full_path.fitted[:, component],
                    )
                else:
                    beta_gap = math.nan
                    fitted_gap = math.nan
                beta_path_discrepancy[
                    model_index, replication, component
                ] = beta_gap
                fitted_path_discrepancy[
                    model_index, replication, component
                ] = fitted_gap
                path_rows.append(
                    {
                        "model": model.name,
                        "replication": replication + 1,
                        "components": prefix,
                        "raw_valid": bool(apls.raw_full_path.valid[component]),
                        "raw_normalized_basis_condition": float(
                            apls.raw_full_path.basis_condition[component]
                        ),
                        "raw_normalized_design_condition": float(
                            apls.raw_full_path.design_condition[component]
                        ),
                        "reorth_basis_condition": float(
                            apls.reorth_full_path.basis_condition[component]
                        ),
                        "reorth_orthogonality_defect": float(
                            apls.reorth_full_path.orthogonality_defect[component]
                        ),
                        "relative_beta_difference_same_m": beta_gap,
                        "relative_fitted_difference_same_m": fitted_gap,
                    }
                )

            common_component = min(apls.m_raw, apls.m_reorth)
            raw_common_valid = bool(
                apls.raw_full_path.valid[common_component - 1]
                and apls.reorth_full_path.valid[common_component - 1]
            )
            if raw_common_valid:
                beta_difference_common = _relative_difference(
                    apls.raw_full_path.beta[:, common_component - 1],
                    apls.reorth_full_path.beta[:, common_component - 1],
                )
                fitted_difference_common = _relative_difference(
                    apls.raw_full_path.fitted[:, common_component - 1],
                    apls.reorth_full_path.fitted[:, common_component - 1],
                )
            else:
                beta_difference_common = math.nan
                fitted_difference_common = math.nan

            diagnostic_rows.append(
                {
                    "model": model.name,
                    "replication": replication + 1,
                    "raw_selected_m": apls.m_raw,
                    "reorth_selected_m": apls.m_reorth,
                    "raw_cv_min": float(apls.cv_raw[apls.m_raw - 1]),
                    "reorth_cv_min": float(apls.cv_reorth[apls.m_reorth - 1]),
                    "raw_valid_prefixes_full": int(
                        np.count_nonzero(apls.raw_full_path.valid)
                    ),
                    "raw_valid_prefixes_all_folds": int(
                        np.count_nonzero(apls.raw_valid_all_folds)
                    ),
                    "reorth_effective_dimension": (
                        apls.reorth_full_path.effective_dimension
                    ),
                    "raw_condition_at_selected_m": float(
                        apls.raw_full_path.basis_condition[apls.m_raw - 1]
                    ),
                    "reorth_orthogonality_defect_at_selected_m": float(
                        apls.reorth_full_path.orthogonality_defect[
                            apls.m_reorth - 1
                        ]
                    ),
                    "comparison_component": common_component,
                    "relative_beta_difference_same_m": beta_difference_common,
                    "relative_fitted_difference_same_m": fitted_difference_common,
                    "cg_threshold_reached": cg_threshold_reached,
                }
            )

            if replication == 0:
                representative[model.name] = {
                    "cv_raw": apls.cv_raw.copy(),
                    "cv_reorth": apls.cv_reorth.copy(),
                    "raw_condition": apls.raw_full_path.basis_condition.copy(),
                    "reorth_condition": apls.reorth_full_path.basis_condition.copy(),
                    "raw_valid": apls.raw_full_path.valid.copy(),
                }

    summary_rows: List[Dict[str, object]] = []
    for model_index, model in enumerate(models):
        for method_index, method in enumerate(methods):
            ise_values = ise[model_index, :, method_index]
            mspe_values = mspe[model_index, :, method_index]
            finite = np.isfinite(ise_values) & np.isfinite(mspe_values)
            components = selected_components[model_index, :, method_index]
            summary_rows.append(
                {
                    "model": model.name,
                    "method": method,
                    "replications": replications,
                    "finite_replications": int(np.count_nonzero(finite)),
                    "failure_rate": float(1.0 - np.mean(finite)),
                    "mean_ise": _safe_mean(ise_values),
                    "se_ise": _standard_error(ise_values),
                    "median_ise": _safe_median(ise_values),
                    "mean_mspe": _safe_mean(mspe_values),
                    "se_mspe": _standard_error(mspe_values),
                    "median_mspe": _safe_median(mspe_values),
                    "mean_selected_components": float(np.mean(components)),
                    "median_selected_components": float(np.median(components)),
                }
            )

    configuration = {
        "replications_per_model": replications,
        "n": n,
        "basis_size_J": J,
        "grid_size_T": T,
        "maximum_components": m_max,
        "cross_validation_folds": k_folds,
        "noise_sd": noise_sd,
        "cg_tau": tau,
        "cg_delta": delta,
        "seed": seed,
        "basis_convention": basis_convention,
        "reorthogonalization": orthogonalization,
        "rank_tolerance": rank_tolerance,
        "raw_apls": {
            "basis": "[r, Kr, ..., K^(m-1)r]",
            "solver": "unregularized normal equations via numpy.linalg.solve",
            "column_scaling": False,
            "ridge": False,
            "pseudoinverse_fallback": False,
            "orthogonalization": False,
        },
        "methods": list(methods),
    }

    _write_csv(output_dir / "replication_results.csv", replication_rows)
    _write_csv(output_dir / "summary.csv", summary_rows)
    _write_csv(output_dir / "apls_diagnostics.csv", diagnostic_rows)
    _write_csv(output_dir / "apls_path_diagnostics.csv", path_rows)
    with (output_dir / "configuration.json").open("w", encoding="utf-8") as handle:
        json.dump(configuration, handle, indent=2)
        handle.write("\n")
    np.savez_compressed(
        output_dir / "raw_results.npz",
        ise=ise,
        mspe=mspe,
        selected_components=selected_components,
        raw_conditions=raw_conditions,
        reorth_conditions=reorth_conditions,
        reorthogonality_defects=reorth_defects,
        beta_path_discrepancy=beta_path_discrepancy,
        fitted_path_discrepancy=fitted_path_discrepancy,
        raw_valid=raw_valid,
        method_names=np.asarray(methods),
        model_names=np.asarray([model.name for model in models]),
    )

    if make_plots:
        plot_performance(ise, mspe, models, methods, output_dir)
        plot_conditioning(
            raw_conditions, reorth_conditions, models, output_dir
        )
        plot_same_m_discrepancy(
            beta_path_discrepancy,
            fitted_path_discrepancy,
            models,
            output_dir,
        )
        plot_component_selection(
            selected_components, models, methods, output_dir
        )
        plot_representative_cv(representative, stable_label, output_dir)

    print("\nSummary")
    for row in summary_rows:
        print(
            f"  {row['model']} | {row['method']:<12} | "
            f"mean ISE={row['mean_ise']:.6g} | "
            f"mean MSPE={row['mean_mspe']:.6g} | "
            f"mean m={row['mean_selected_components']:.2f}"
        )
    print(f"\nResults written to {output_dir.resolve()}")

    return {
        "configuration": configuration,
        "summary": summary_rows,
        "replications": replication_rows,
        "diagnostics": diagnostic_rows,
        "ise": ise,
        "mspe": mspe,
        "selected_components": selected_components,
        "raw_conditions": raw_conditions,
        "reorth_conditions": reorth_conditions,
        "reorth_defects": reorth_defects,
        "beta_path_discrepancy": beta_path_discrepancy,
        "fitted_path_discrepancy": fitted_path_discrepancy,
        "raw_valid": raw_valid,
    }


def _finite_positive(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values) & (values > 0.0)]


def plot_performance(
    ise: np.ndarray,
    mspe: np.ndarray,
    models: Sequence[ModelSpec],
    methods: Sequence[str],
    output_dir: Path,
) -> None:
    """Boxplots of estimation and prediction error on logarithmic axes."""
    fig, axes = plt.subplots(2, len(models), figsize=(15, 8), squeeze=False)
    for model_index, model in enumerate(models):
        for row_index, (metric, label) in enumerate(
            ((ise, "Integrated squared error"), (mspe, "Test MSPE"))
        ):
            ax = axes[row_index, model_index]
            data = [
                _finite_positive(metric[model_index, :, method_index])
                for method_index in range(len(methods))
            ]
            ax.boxplot(data, showfliers=False, patch_artist=True)
            for patch, method in zip(ax.patches, methods):
                patch.set_facecolor(METHOD_COLORS[method])
                patch.set_alpha(0.65)
            ax.set_yscale("log")
            ax.set_xticks(range(1, len(methods) + 1))
            ax.set_xticklabels(methods, rotation=30, ha="right")
            ax.grid(True, which="both", axis="y", alpha=0.25)
            if row_index == 0:
                ax.set_title(model.name, fontweight="bold")
            if model_index == 0:
                ax.set_ylabel(label)
    fig.suptitle("Estimator performance: raw and orthogonalized APLS", y=0.995)
    fig.tight_layout()
    _save_pdf(fig, output_dir / "performance_comparison.pdf")
    plt.close(fig)


def plot_conditioning(
    raw_conditions: np.ndarray,
    reorth_conditions: np.ndarray,
    models: Sequence[ModelSpec],
    output_dir: Path,
) -> None:
    """Median numerical conditioning of the two APLS bases."""
    m_max = raw_conditions.shape[2]
    components = np.arange(1, m_max + 1)
    fig, axes = plt.subplots(1, len(models), figsize=(15, 4.6), squeeze=False)
    for model_index, model in enumerate(models):
        ax = axes[0, model_index]
        raw = np.minimum(raw_conditions[model_index], CONDITION_CAP)
        reorth = np.minimum(reorth_conditions[model_index], CONDITION_CAP)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            raw_median = np.nanmedian(raw, axis=0)
            reorth_median = np.nanmedian(reorth, axis=0)
        ax.plot(
            components,
            raw_median,
            color=METHOD_COLORS["APLS-raw"],
            linewidth=2.0,
            label="Raw normalized powers",
        )
        ax.plot(
            components,
            reorth_median,
            color=STABLE_COLOR,
            linewidth=2.0,
            label="Orthogonalized basis",
        )
        ax.axhline(
            CONDITION_CAP,
            color="#666666",
            linestyle=":",
            linewidth=1.0,
            label="1 / machine epsilon" if model_index == 0 else None,
        )
        ax.set_yscale("log")
        ax.set_xlabel("Component count m")
        ax.set_title(model.name, fontweight="bold")
        ax.grid(True, which="both", alpha=0.25)
        if model_index == 0:
            ax.set_ylabel("Median basis condition number")
            ax.legend(fontsize=8)
    fig.suptitle("Numerical conditioning of APLS coordinates", y=0.99)
    fig.tight_layout()
    _save_pdf(fig, output_dir / "apls_conditioning.pdf")
    plt.close(fig)


def plot_component_selection(
    selected_components: np.ndarray,
    models: Sequence[ModelSpec],
    methods: Sequence[str],
    output_dir: Path,
) -> None:
    """Selected-component distributions for raw and reorthogonalized APLS."""
    fig, axes = plt.subplots(1, len(models), figsize=(15, 4.5), squeeze=False)
    raw_index = methods.index("APLS-raw")
    stable_label = methods[2]
    reorth_index = methods.index(stable_label)
    for model_index, model in enumerate(models):
        ax = axes[0, model_index]
        raw = selected_components[model_index, :, raw_index]
        reorth = selected_components[model_index, :, reorth_index]
        component_values = np.arange(
            min(int(np.min(raw)), int(np.min(reorth))),
            max(int(np.max(raw)), int(np.max(reorth))) + 1,
        )
        raw_counts = np.asarray(
            [np.count_nonzero(raw == value) for value in component_values]
        )
        reorth_counts = np.asarray(
            [np.count_nonzero(reorth == value) for value in component_values]
        )
        ax.bar(
            component_values - 0.2,
            raw_counts,
            width=0.4,
            color=METHOD_COLORS["APLS-raw"],
            label="APLS-raw",
        )
        ax.bar(
            component_values + 0.2,
            reorth_counts,
            width=0.4,
            color=STABLE_COLOR,
            label=stable_label,
        )
        ax.set_xlabel("Cross-validated component count")
        ax.set_title(model.name, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.25)
        if model_index == 0:
            ax.set_ylabel("Replications")
            ax.legend(fontsize=9)
    fig.suptitle("APLS component selection", y=0.99)
    fig.tight_layout()
    _save_pdf(fig, output_dir / "apls_component_selection.pdf")
    plt.close(fig)


def plot_same_m_discrepancy(
    beta_discrepancy: np.ndarray,
    fitted_discrepancy: np.ndarray,
    models: Sequence[ModelSpec],
    output_dir: Path,
) -> None:
    """Median finite-precision difference at identical component counts."""
    m_max = beta_discrepancy.shape[2]
    components = np.arange(1, m_max + 1)
    fig, axes = plt.subplots(1, len(models), figsize=(15, 4.6), squeeze=False)
    for model_index, model in enumerate(models):
        ax = axes[0, model_index]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            beta_median = np.nanmedian(beta_discrepancy[model_index], axis=0)
            fitted_median = np.nanmedian(
                fitted_discrepancy[model_index], axis=0
            )
        beta_median = np.maximum(beta_median, np.finfo(float).tiny)
        fitted_median = np.maximum(fitted_median, np.finfo(float).tiny)
        ax.plot(
            components,
            beta_median,
            color="#6A3D9A",
            linewidth=2.0,
            label="Slope difference",
        )
        ax.plot(
            components,
            fitted_median,
            color="#1F78B4",
            linewidth=2.0,
            label="Fitted-value difference",
        )
        ax.set_yscale("log")
        ax.set_xlabel("Common component count m")
        ax.set_title(model.name, fontweight="bold")
        ax.grid(True, which="both", alpha=0.25)
        if model_index == 0:
            ax.set_ylabel("Median relative difference")
            ax.legend(fontsize=9)
    fig.suptitle(
        "Finite-precision divergence: raw versus orthogonalized APLS",
        y=0.99,
    )
    fig.tight_layout()
    _save_pdf(fig, output_dir / "apls_same_m_discrepancy.pdf")
    plt.close(fig)


def plot_representative_cv(
    representative: Dict[str, Dict[str, np.ndarray]],
    stable_label: str,
    output_dir: Path,
) -> None:
    """First-replication cross-validation paths for each DGP."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), squeeze=False)
    for model_index, model_name in enumerate(("Model 1", "Model 2", "Model 3")):
        ax = axes[0, model_index]
        data = representative[model_name]
        components = np.arange(1, len(data["cv_raw"]) + 1)
        raw = np.asarray(data["cv_raw"], dtype=float)
        reorth = np.asarray(data["cv_reorth"], dtype=float)
        ax.plot(
            components,
            np.where(np.isfinite(raw), raw, np.nan),
            color=METHOD_COLORS["APLS-raw"],
            linewidth=2.0,
            label="APLS-raw",
        )
        ax.plot(
            components,
            np.where(np.isfinite(reorth), reorth, np.nan),
            color=STABLE_COLOR,
            linewidth=2.0,
            label=stable_label,
        )
        ax.set_yscale("log")
        ax.set_xlabel("Component count m")
        ax.set_title(model_name, fontweight="bold")
        ax.grid(True, which="both", alpha=0.25)
        if model_index == 0:
            ax.set_ylabel("Five-fold CV error")
            ax.legend(fontsize=9)
    fig.suptitle("Representative APLS cross-validation paths", y=0.99)
    fig.tight_layout()
    _save_pdf(fig, output_dir / "apls_cv_paths.pdf")
    plt.close(fig)


def run_self_tests() -> None:
    """Run deterministic checks of scaling, equivalence, and orthogonality."""
    rng = np.random.default_rng(78123)
    n, T = 80, 18
    X = rng.normal(size=(n, T))
    beta_true = rng.normal(size=T)
    y = X @ beta_true / T + 0.1 * rng.normal(size=n)
    K = X.T @ X / (n * T)
    r = X.T @ y / n

    raw = raw_apls_path(y, X, K, r, 4)
    reorth_cgs2 = reorthogonalized_apls_path(
        y, X, K, r, 4, method="cgs2", rank_tolerance=1e-12
    )
    reorth_mgs2 = reorthogonalized_apls_path(
        y, X, K, r, 4, method="mgs2", rank_tolerance=1e-12
    )
    published_mgs1 = reorthogonalized_apls_path(
        y, X, K, r, 4, method="mgs1", rank_tolerance=1e-12
    )
    if not np.all(raw.valid):
        raise AssertionError("raw APLS unexpectedly failed on the well-conditioned test")
    if _relative_difference(raw.beta[:, 0], reorth_cgs2.beta[:, 0]) > 1e-11:
        raise AssertionError("raw and reorthogonalized first components disagree")
    for component in range(4):
        if _relative_difference(
            raw.fitted[:, component], reorth_cgs2.fitted[:, component]
        ) > 1e-7:
            raise AssertionError(
                f"raw and CGS2 fitted values disagree at m={component + 1}"
            )
        if _relative_difference(
            reorth_cgs2.fitted[:, component],
            reorth_mgs2.fitted[:, component],
        ) > 1e-11:
            raise AssertionError(
                f"CGS2 and MGS2 fitted values disagree at m={component + 1}"
            )
        if _relative_difference(
            reorth_cgs2.fitted[:, component],
            published_mgs1.fitted[:, component],
        ) > 1e-10:
            raise AssertionError(
                f"CGS2 and MGS1 fitted values disagree at m={component + 1}"
            )
    defect_cgs2 = np.linalg.norm(
        reorth_cgs2.basis.T @ reorth_cgs2.basis
        - np.eye(reorth_cgs2.basis.shape[1]),
        ord=2,
    )
    defect_mgs2 = np.linalg.norm(
        reorth_mgs2.basis.T @ reorth_mgs2.basis
        - np.eye(reorth_mgs2.basis.shape[1]),
        ord=2,
    )
    if defect_cgs2 > 1e-12 or defect_mgs2 > 1e-12:
        raise AssertionError("reorthogonalized bases are not orthonormal")

    folds = _make_folds(n, 5, np.random.default_rng(9182))
    compared = apls_cv_compare(
        y,
        X,
        K,
        r,
        4,
        folds,
        orthogonalization="cgs2",
        rank_tolerance=1e-12,
    )
    if not (
        1 <= compared.m_raw <= 4
        and 1 <= compared.m_reorth <= 4
        and np.all(np.isfinite(compared.beta_raw))
        and np.all(np.isfinite(compared.beta_reorth))
    ):
        raise AssertionError("cross-validation returned an invalid fit")
    print(
        "Self-tests passed: raw formula, same-m equivalence, MGS1/CGS2/MGS2 "
        "agreement, orthogonality, and fold-local CV."
    )


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replications", type=int, default=5)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--basis-size", dest="J", type=int, default=100)
    parser.add_argument("--grid-size", dest="T", type=int, default=200)
    parser.add_argument("--m-max", type=int, default=70)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--noise-sd", type=float, default=1.0)
    parser.add_argument("--tau", type=float, default=1.01)
    parser.add_argument("--delta", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument(
        "--basis-convention",
        choices=("babii", "standard"),
        default="babii",
    )
    parser.add_argument(
        "--orthogonalization",
        choices=("cgs1", "cgs2", "mgs1", "mgs2"),
        default="cgs2",
        help=(
            "CGS2 matches the earlier vectorized two-pass implementation; "
            "MGS1 supplies the published-style one-pass benchmark"
        ),
    )
    parser.add_argument("--rank-tolerance", type=float, default=1e-10)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("apls_raw_vs_reorth_results"),
    )
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument(
        "--self-test-only",
        action="store_true",
        help="run deterministic checks and exit without a simulation",
    )
    return parser.parse_args()


def main() -> None:
    arguments = _parse_arguments()
    run_self_tests()
    if arguments.self_test_only:
        return
    run_simulation(
        replications=arguments.replications,
        n=arguments.n,
        J=arguments.J,
        T=arguments.T,
        m_max=arguments.m_max,
        k_folds=arguments.folds,
        noise_sd=arguments.noise_sd,
        tau=arguments.tau,
        delta=arguments.delta,
        seed=arguments.seed,
        basis_convention=arguments.basis_convention,
        orthogonalization=arguments.orthogonalization,
        rank_tolerance=arguments.rank_tolerance,
        output_dir=arguments.output_dir,
        make_plots=not arguments.no_plots,
    )


if __name__ == "__main__":
    main()
