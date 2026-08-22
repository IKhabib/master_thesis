"""Corrected raw-versus-orthogonalized APLS simulation study.

This standalone program compares six reported methods under the three
simulation designs used in ``model_8_corrected.py``:

1. FPCR selected by explicitly labelled response- or moment-space GCV;
2. ``CG-FPLS-code``, which reproduces Babii et al.'s released simulation
   notebook: one moment-GCV FPCR plug-in variance estimate and m >= 1;
3. ``CG-FPLS-supplement``, which follows the written supplementary algorithm:
   iterative residual-variance estimation and the admissible m=0 estimate;
4. ``CG-FPLS-oracle``, an explicitly infeasible simulation diagnostic using
   the known data-generating noise variance and the theoretical m >= 0 rule;
5. raw APLS, implemented from the nonorthogonal powers
       H_m = [r, K r, ..., K^(m-1) r]
   and the unregularized normal equations; and
6. orthogonalized APLS, labelled APLS--Arnoldi--CGS2 (or the selected
   Gram--Schmidt variant), which represents the same Krylov spaces with an
   orthonormal Arnoldi basis and solves response least squares by QR.

All three CG-FPLS variants use exactly the same conjugate-gradient path.  They
differ only in noise calibration and in whether m=0 is an admissible stopping
point.  The oracle variant must never be presented as a feasible estimator;
it is included solely to diagnose performance lost through variance
estimation and threshold calibration.

The raw routine deliberately applies no column scaling, ridge penalty,
pseudoinverse, truncated SVD, or Gram--Schmidt step.  It is therefore a
faithful finite-precision baseline for the explicit APLS construction in
Section 4.1 of Delaigle and Hall (2012), adapted to the known-zero-mean
simulation convention used here.  A failed raw solve is recorded rather than
silently repaired.

The orthogonalized routine defaults to CGS2: classical Gram--Schmidt with
one reorthogonalization pass.  This is the accurate name for the vectorized
two-projection implementation used in the earlier project code.  The
published-style one-pass MGS benchmark and the other variants are available through
``--orthogonalization mgs1``, ``cgs1``, or ``mgs2``.  Thus the marginal effect
of a second pass can be studied rather than attributed to APLS itself.

The default FPCR tuning criterion is conventional response-space GCV.  The
moment-space rule used by the earlier notebook remains reproducible through
``--fpcr-gcv moment`` and is labelled ``FPCR-moment-GCV`` in every output.
Performance summaries report means, standard errors, medians, 90th and 99th
percentiles, maxima, transparent extreme-tail rates, and numerical-instability
rates.  All boxplot outliers are shown, and raw-APLS extreme cases are also
listed and plotted separately.  ``summary.csv`` contains every method, while
``summary_feasible.csv`` excludes the oracle diagnostic.  The main performance
figure likewise excludes the oracle, and a separately labelled figure includes
it for diagnostic comparison.

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
        --fpcr-gcv response \
        --output-dir apls_cg_variants_results

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
    "CG-FPLS-code": "#0072B2",
    "CG-FPLS-supplement": "#56B4E9",
    "CG-FPLS-oracle": "#6A3D9A",
    "APLS-raw": "#D55E00",
    "APLS-CGS1": "#009E73",
    "APLS-CGS2": "#009E73",
    "APLS-MGS1": "#009E73",
    "APLS-MGS2": "#009E73",
    "APLS-Arnoldi-CGS1": "#009E73",
    "APLS-Arnoldi-CGS2": "#009E73",
    "APLS-Arnoldi-MGS1": "#009E73",
    "APLS-Arnoldi-MGS2": "#009E73",
    "FPCR": "#CC79A7",
    "FPCR-response-GCV": "#CC79A7",
    "FPCR-moment-GCV": "#CC79A7",
}
CG_CODE_LABEL = "CG-FPLS-code"
CG_SUPPLEMENT_LABEL = "CG-FPLS-supplement"
CG_ORACLE_LABEL = "CG-FPLS-oracle"
CG_VARIANT_LABELS = (
    CG_CODE_LABEL,
    CG_SUPPLEMENT_LABEL,
    CG_ORACLE_LABEL,
)
STABLE_COLOR = "#009E73"
MODEL_COLORS = {
    "Model 1": "#0072B2",
    "Model 2": "#D55E00",
    "Model 3": "#009E73",
}
CONDITION_CAP = 1.0 / np.finfo(float).eps
RAW_NORMAL_EQUATIONS_CONDITION_THRESHOLD = 1.0 / np.sqrt(
    np.finfo(float).eps
)
ORTHOGONALITY_DEFECT_THRESHOLD = np.sqrt(np.finfo(float).eps)


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


@dataclass
class FPCRResult:
    """An FPCR fit and its explicitly identified GCV path."""

    beta: np.ndarray
    selected_components: int
    gcv: np.ndarray
    criterion: str


@dataclass
class CGFPLSSelection:
    """One Babii CG-FPLS stopping-rule selection and its diagnostics."""

    beta: np.ndarray
    selected_components: int
    threshold_reached: bool
    converged: bool
    variance_updates: int
    sigma2_initial: float
    sigma2_final: float
    sigma2_history: np.ndarray
    selected_components_history: np.ndarray
    threshold_history: np.ndarray
    moment_path: np.ndarray
    variant: str = "supplement"
    minimum_components: int = 0
    sigma2_source: str = "iterated residual variance"
    variance_iteration_used: bool = True
    feasible_estimator: bool = True


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


def _select_cg_for_variance(
    path: np.ndarray,
    r: np.ndarray,
    K: np.ndarray,
    *,
    sigma2: float,
    X_norm: float,
    n: int,
    tau: float,
    delta: float,
    minimum_components: int = 0,
) -> Tuple[np.ndarray, int, bool, np.ndarray, float]:
    """Apply the discrepancy rule for one fixed residual-variance value.

    The returned moment path always includes m=0 as its first entry.
    ``minimum_components=0`` implements the theoretical/supplementary rule,
    while ``minimum_components=1`` reproduces the released simulation code.
    """
    path = np.asarray(path, dtype=float)
    r = np.asarray(r, dtype=float).reshape(-1)
    K = np.asarray(K, dtype=float)
    if path.ndim != 2 or path.shape[0] != len(r):
        raise ValueError("path must have one row per element of r")
    if K.shape != (len(r), len(r)):
        raise ValueError("K and r have incompatible dimensions")
    if sigma2 < 0.0 or not np.isfinite(sigma2):
        raise ValueError("sigma2 must be finite and non-negative")
    if X_norm < 0.0 or not np.isfinite(X_norm) or n < 2:
        raise ValueError("X_norm and n are invalid")
    if not 0 <= minimum_components <= path.shape[1]:
        raise ValueError("minimum_components is outside the available CG path")

    residuals = np.column_stack((r, r[:, None] - K @ path))
    moments = np.sqrt(np.mean(residuals ** 2, axis=0))
    threshold = tau * np.sqrt(2.0 * sigma2 * X_norm / (delta * n))
    reached = np.flatnonzero(
        moments[minimum_components:] <= threshold
    )
    if len(reached):
        selected = int(reached[0] + minimum_components)
        threshold_reached = True
    else:
        selected = path.shape[1]
        threshold_reached = False
    beta = np.zeros(len(r)) if selected == 0 else path[:, selected - 1]
    return beta, selected, threshold_reached, moments, float(threshold)


def _fixed_variance_cg_selection(
    path: np.ndarray,
    X: np.ndarray,
    r: np.ndarray,
    K: np.ndarray,
    *,
    sigma2: float,
    tau: float,
    delta: float,
    minimum_components: int,
    variant: str,
    sigma2_source: str,
    feasible_estimator: bool,
) -> CGFPLSSelection:
    """Select one CG iterate using a fixed discrepancy variance."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[0] < 2 or X.shape[1] < 2:
        raise ValueError("X must be a nonempty two-dimensional design")
    if not np.all(np.isfinite(X)):
        raise ValueError("X must be finite")
    n, T = X.shape
    K, r = _validate_matrices(K, r, T)
    path = np.asarray(path, dtype=float)
    if path.ndim != 2 or path.shape[0] != T or path.shape[1] < 1:
        raise ValueError("path must be a nonempty T-by-m array")
    if not np.all(np.isfinite(path)):
        raise ValueError("path must be finite")
    if not 0.0 < delta < 1.0 or tau <= 1.0:
        raise ValueError("tau must exceed one and delta must lie in (0,1)")
    if sigma2 < 0.0 or not np.isfinite(sigma2):
        raise ValueError("sigma2 must be finite and non-negative")

    X_norm = float(np.mean(np.sum(X ** 2, axis=1) / T))
    beta, selected, reached, moments, threshold = _select_cg_for_variance(
        path,
        r,
        K,
        sigma2=sigma2,
        X_norm=X_norm,
        n=n,
        tau=tau,
        delta=delta,
        minimum_components=minimum_components,
    )
    return CGFPLSSelection(
        beta=beta,
        selected_components=selected,
        threshold_reached=reached,
        converged=True,
        variance_updates=0,
        sigma2_initial=float(sigma2),
        sigma2_final=float(sigma2),
        sigma2_history=np.asarray([sigma2], dtype=float),
        selected_components_history=np.asarray([selected], dtype=int),
        threshold_history=np.asarray([threshold], dtype=float),
        moment_path=np.asarray(moments, dtype=float),
        variant=variant,
        minimum_components=minimum_components,
        sigma2_source=sigma2_source,
        variance_iteration_used=False,
        feasible_estimator=feasible_estimator,
    )


def select_cg_fpls_code(
    path: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    r: np.ndarray,
    K: np.ndarray,
    beta_pilot: np.ndarray,
    *,
    tau: float,
    delta: float,
) -> CGFPLSSelection:
    """Reproduce the CG-FPLS selector in the authors' released notebook.

    The notebook estimates sigma squared once from a moment-GCV FPCR pilot and
    searches only the positive CG iterates m=1,...,m_max.  The caller supplies
    that pilot explicitly so the variance source remains auditable.
    """
    y, X = _validate_xy(y, X)
    T = X.shape[1]
    beta_pilot = np.asarray(beta_pilot, dtype=float).reshape(-1)
    if beta_pilot.shape != (T,) or not np.all(np.isfinite(beta_pilot)):
        raise ValueError("beta_pilot must be a finite vector of length T")
    sigma2 = max(float(np.mean((y - X @ beta_pilot / T) ** 2)), 0.0)
    return _fixed_variance_cg_selection(
        path,
        X,
        r,
        K,
        sigma2=sigma2,
        tau=tau,
        delta=delta,
        minimum_components=1,
        variant="code",
        sigma2_source="one moment-GCV FPCR plug-in estimate",
        feasible_estimator=True,
    )


def select_cg_fpls_oracle(
    path: np.ndarray,
    X: np.ndarray,
    r: np.ndarray,
    K: np.ndarray,
    *,
    sigma2_true: float,
    tau: float,
    delta: float,
) -> CGFPLSSelection:
    """Diagnostic CG-FPLS selector using the known simulation noise variance."""
    return _fixed_variance_cg_selection(
        path,
        X,
        r,
        K,
        sigma2=sigma2_true,
        tau=tau,
        delta=delta,
        minimum_components=0,
        variant="oracle",
        sigma2_source="known data-generating noise variance",
        feasible_estimator=False,
    )


def select_cg_fpls_supplement(
    path: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    r: np.ndarray,
    K: np.ndarray,
    beta_pilot: np.ndarray,
    *,
    tau: float,
    delta: float,
    variance_tolerance: float = 0.01,
    variance_kmax: int = 10,
) -> CGFPLSSelection:
    """Apply Babii et al.'s full iterative discrepancy-stopping procedure.

    Starting from a regularized pilot, the routine alternates between the
    discrepancy-stopped CG-FPLS fit and the residual variance estimate.  The
    loop follows Supplement S.5 literally for k=0,...,kmax, so at most
    ``variance_kmax + 1`` variance updates are made.
    """
    y, X = _validate_xy(y, X)
    n, T = X.shape
    K, r = _validate_matrices(K, r, T)
    path = np.asarray(path, dtype=float)
    beta_pilot = np.asarray(beta_pilot, dtype=float).reshape(-1)
    if path.ndim != 2 or path.shape[0] != T or path.shape[1] < 1:
        raise ValueError("path must be a nonempty T-by-m array")
    if beta_pilot.shape != (T,) or not np.all(np.isfinite(beta_pilot)):
        raise ValueError("beta_pilot must be a finite vector of length T")
    if not 0.0 < delta < 1.0 or tau <= 1.0:
        raise ValueError("tau must exceed one and delta must lie in (0,1)")
    if variance_tolerance < 0.0 or variance_kmax < 0:
        raise ValueError("variance_tolerance and variance_kmax must be non-negative")

    sigma2 = max(float(np.mean((y - X @ beta_pilot / T) ** 2)), 0.0)
    sigma2_initial = sigma2
    X_norm = float(np.mean(np.sum(X ** 2, axis=1) / T))
    sigma2_history = [sigma2]
    selected_history: List[int] = []
    threshold_history: List[float] = []
    converged = False
    beta_selected = np.zeros(T)
    selected = 0
    threshold_reached = False
    moment_path = np.full(path.shape[1] + 1, np.nan)

    for _ in range(variance_kmax + 1):
        (
            beta_selected,
            selected,
            threshold_reached,
            moment_path,
            threshold,
        ) = _select_cg_for_variance(
            path,
            r,
            K,
            sigma2=sigma2,
            X_norm=X_norm,
            n=n,
            tau=tau,
            delta=delta,
        )
        sigma2_new = max(
            float(np.mean((y - X @ beta_selected / T) ** 2)), 0.0
        )
        selected_history.append(selected)
        threshold_history.append(threshold)
        sigma2_history.append(sigma2_new)
        if abs(sigma2_new - sigma2) <= variance_tolerance:
            converged = True
            sigma2 = sigma2_new
            break
        sigma2 = sigma2_new

    return CGFPLSSelection(
        beta=beta_selected,
        selected_components=selected,
        threshold_reached=threshold_reached,
        converged=converged,
        variance_updates=len(selected_history),
        sigma2_initial=sigma2_initial,
        sigma2_final=sigma2,
        sigma2_history=np.asarray(sigma2_history, dtype=float),
        selected_components_history=np.asarray(selected_history, dtype=int),
        threshold_history=np.asarray(threshold_history, dtype=float),
        moment_path=np.asarray(moment_path, dtype=float),
        variant="supplement",
        minimum_components=0,
        sigma2_source="iterated residual variance from moment-GCV FPCR pilot",
        variance_iteration_used=True,
        feasible_estimator=True,
    )


def select_cg_fpls(
    path: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    r: np.ndarray,
    K: np.ndarray,
    beta_pilot: np.ndarray,
    *,
    tau: float,
    delta: float,
    variance_tolerance: float = 0.01,
    variance_kmax: int = 10,
) -> CGFPLSSelection:
    """Backward-compatible alias for ``select_cg_fpls_supplement``."""
    return select_cg_fpls_supplement(
        path,
        X,
        y,
        r,
        K,
        beta_pilot,
        tau=tau,
        delta=delta,
        variance_tolerance=variance_tolerance,
        variance_kmax=variance_kmax,
    )


def _fpcr_spectral_path(
    r: np.ndarray, K: np.ndarray, m_max: int
) -> np.ndarray:
    """Return spectral-cutoff FPCR estimates for m=1,...,m_max."""
    r = np.asarray(r, dtype=float).reshape(-1)
    K = np.asarray(K, dtype=float)
    T = len(r)
    if K.shape != (T, T) or m_max < 1:
        raise ValueError("K, r, and m_max have incompatible dimensions")
    K = 0.5 * (K + K.T)
    try:
        eigenvalues, eigenvectors = eigh(K, check_finite=False)
        order = np.argsort(eigenvalues)[::-1]
    except LinAlgError:
        eigenvectors, eigenvalues, _ = np.linalg.svd(K, full_matrices=False)
        order = np.arange(len(eigenvalues))
    threshold = np.finfo(float).eps * T * max(
        float(np.max(np.abs(eigenvalues))), np.finfo(float).tiny
    )
    positive = order[eigenvalues[order] > threshold]
    maximum = min(m_max, len(positive), T - 1)
    if maximum == 0:
        raise LinAlgError("K has no numerically positive eigenvalues")
    beta_path = np.zeros((T, maximum))
    for component in range(maximum):
        count = component + 1
        indices = positive[:count]
        values = eigenvalues[indices]
        vectors = eigenvectors[:, indices]
        beta_path[:, component] = vectors @ ((vectors.T @ r) / values)
    return beta_path


def _select_fpcr_gcv_from_path(
    y: np.ndarray,
    X: np.ndarray,
    r: np.ndarray,
    K: np.ndarray,
    beta_path: np.ndarray,
    *,
    criterion: str,
) -> FPCRResult:
    """Evaluate one GCV criterion on an already computed FPCR path."""
    y, X = _validate_xy(y, X)
    n, T = X.shape
    K, r = _validate_matrices(K, r, T)
    beta_path = np.asarray(beta_path, dtype=float)
    if (
        beta_path.ndim != 2
        or beta_path.shape[0] != T
        or beta_path.shape[1] < 1
        or not np.all(np.isfinite(beta_path))
    ):
        raise ValueError("beta_path must be a finite nonempty T-by-m array")
    if criterion not in {"response", "moment"}:
        raise ValueError("criterion must be 'response' or 'moment'")

    gcv = np.full(beta_path.shape[1], np.inf)
    for component in range(beta_path.shape[1]):
        count = component + 1
        if criterion == "response":
            residual = y - X @ beta_path[:, component] / T
            gcv[component] = np.mean(residual ** 2) / (1.0 - count / n) ** 2
        else:
            residual = r - K @ beta_path[:, component]
            gcv[component] = np.mean(residual ** 2) / (1.0 - count / T) ** 2
    selected = int(np.argmin(gcv)) + 1
    return FPCRResult(
        beta=beta_path[:, selected - 1],
        selected_components=selected,
        gcv=gcv,
        criterion=criterion,
    )


def select_fpcr_gcv(
    y: np.ndarray,
    X: np.ndarray,
    r: np.ndarray,
    K: np.ndarray,
    m_max: int,
    *,
    criterion: str = "response",
) -> FPCRResult:
    """Select FPCR by response- or moment-space GCV.

    ``response`` uses RSS from y - X beta/T and the sample-size penalty
    (1-m/n)^2.  ``moment`` reproduces the earlier notebook's criterion based
    on r - K beta and the grid-size penalty (1-m/T)^2.
    """
    y, X = _validate_xy(y, X)
    n, T = X.shape
    K, r = _validate_matrices(K, r, T)
    beta_path = _fpcr_spectral_path(r, K, min(m_max, n - 1, T - 1))
    return _select_fpcr_gcv_from_path(
        y,
        X,
        r,
        K,
        beta_path,
        criterion=criterion,
    )


def fpcr_gcv(r: np.ndarray, K: np.ndarray, m_max: int) -> Tuple[np.ndarray, int]:
    """Backward-compatible wrapper for the earlier moment-space GCV rule."""
    r = np.asarray(r, dtype=float).reshape(-1)
    K = np.asarray(K, dtype=float)
    T = len(r)
    beta_path = _fpcr_spectral_path(r, K, min(m_max, T - 1))
    gcv = np.full(beta_path.shape[1], np.inf)
    for component in range(beta_path.shape[1]):
        count = component + 1
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


def _safe_quantile(values: Iterable[float], probability: float) -> float:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must lie in [0,1]")
    return float(np.quantile(array, probability)) if len(array) else math.nan


def _safe_max(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    return float(np.max(array)) if len(array) else math.nan


def _extreme_tail_mask(values: np.ndarray, multiple: float) -> np.ndarray:
    """Flag finite values strictly exceeding multiple times their median."""
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    if multiple <= 1.0:
        raise ValueError("the extreme-tail multiple must exceed one")
    median = _safe_median(values)
    if not np.isfinite(median) or median <= 0.0:
        return np.zeros(values.shape, dtype=bool)
    return finite & (values > multiple * median)


def _standard_error(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if len(array) < 2:
        return math.nan
    return float(np.std(array, ddof=1) / np.sqrt(len(array)))


def _write_csv(
    path: Path,
    rows: Sequence[Dict[str, object]],
    *,
    fieldnames: Optional[Sequence[str]] = None,
) -> None:
    """Write a rectangular sequence of dictionaries."""
    if not rows and not fieldnames:
        raise ValueError(f"cannot infer fields for empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(fieldnames) if fieldnames else list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
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
    """Save and validate a Matplotlib PDF, retrying transient writes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    last_error: Optional[BaseException] = None
    for _ in range(3):
        try:
            with path.open("wb") as handle:
                fig.savefig(handle, format="pdf", bbox_inches="tight")
                handle.flush()
                os.fsync(handle.fileno())
            if _is_complete_pdf(path):
                return
            last_error = OSError(f"incomplete PDF write: {path}")
        except (OSError, ValueError) as error:
            last_error = error
        path.unlink(missing_ok=True)
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
    cg_variance_tolerance: float,
    cg_variance_kmax: int,
    seed: int,
    basis_convention: str,
    fpcr_gcv_criterion: str,
    orthogonalization: str,
    rank_tolerance: float,
    extreme_tail_multiple: float,
    output_dir: Path,
    make_plots: bool,
) -> Dict[str, object]:
    """Run all three models, write results, and return in-memory outputs."""
    if replications < 1 or n < 2 or J < 5 or T < 2 or m_max < 1:
        raise ValueError("replications, n, J, T, and m_max are too small")
    if noise_sd < 0.0:
        raise ValueError("noise_sd must be non-negative")
    if fpcr_gcv_criterion not in {"response", "moment"}:
        raise ValueError("fpcr_gcv_criterion must be response or moment")
    if cg_variance_tolerance < 0.0 or cg_variance_kmax < 0:
        raise ValueError("CG variance iteration controls must be non-negative")
    if extreme_tail_multiple <= 1.0:
        raise ValueError("extreme_tail_multiple must exceed one")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    grid = np.linspace(0.0, 1.0, T)
    basis = create_cosine_basis(grid, J, convention=basis_convention)
    models = make_model_specs(J, basis)
    stable_label = f"APLS-Arnoldi-{orthogonalization.upper()}"
    fpcr_label = f"FPCR-{fpcr_gcv_criterion}-GCV"
    methods = (*CG_VARIANT_LABELS, "APLS-raw", stable_label, fpcr_label)
    method_metadata: Dict[str, Dict[str, object]] = {
        CG_CODE_LABEL: {
            "variant": "code",
            "comparison_role": "main reproduction benchmark",
            "feasible_estimator": True,
            "minimum_components": 1,
            "sigma2_source": "one moment-GCV FPCR plug-in estimate",
        },
        CG_SUPPLEMENT_LABEL: {
            "variant": "supplement",
            "comparison_role": "written-method sensitivity analysis",
            "feasible_estimator": True,
            "minimum_components": 0,
            "sigma2_source": (
                "iterated residual variance from moment-GCV FPCR pilot"
            ),
        },
        CG_ORACLE_LABEL: {
            "variant": "oracle",
            "comparison_role": "infeasible simulation diagnostic only",
            "feasible_estimator": False,
            "minimum_components": 0,
            "sigma2_source": "known data-generating noise variance",
        },
        "APLS-raw": {
            "variant": "raw Krylov powers",
            "comparison_role": "numerical baseline",
            "feasible_estimator": True,
        },
        stable_label: {
            "variant": orthogonalization.lower(),
            "comparison_role": "stabilized APLS comparator",
            "feasible_estimator": True,
        },
        fpcr_label: {
            "variant": fpcr_gcv_criterion,
            "comparison_role": "FPCR benchmark",
            "feasible_estimator": True,
        },
    }
    seed_sequences = np.random.SeedSequence(seed).spawn(len(models))

    replication_rows: List[Dict[str, object]] = []
    diagnostic_rows: List[Dict[str, object]] = []
    path_rows: List[Dict[str, object]] = []
    cg_diagnostic_rows: List[Dict[str, object]] = []
    fpcr_diagnostic_rows: List[Dict[str, object]] = []
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
    method_instability = np.zeros(
        (len(models), replications, len(methods)), dtype=bool
    )
    raw_selected_design_condition = np.full(
        (len(models), replications), np.nan
    )
    raw_selected_basis_condition = np.full(
        (len(models), replications), np.nan
    )
    stable_selected_orthogonality_defect = np.full(
        (len(models), replications), np.nan
    )
    cg_initial_sigma2 = np.full(
        (len(models), replications, len(CG_VARIANT_LABELS)), np.nan
    )
    cg_final_sigma2 = np.full_like(cg_initial_sigma2, np.nan)
    cg_variance_updates = np.zeros(
        (len(models), replications, len(CG_VARIANT_LABELS)), dtype=int
    )
    cg_variance_converged = np.zeros(
        (len(models), replications, len(CG_VARIANT_LABELS)), dtype=bool
    )
    cg_threshold_reached = np.zeros(
        (len(models), replications, len(CG_VARIANT_LABELS)), dtype=bool
    )
    cg_pilot_selected_components = np.zeros(
        (len(models), replications), dtype=int
    )
    representative: Dict[str, Dict[str, np.ndarray]] = {}

    print("=" * 72)
    print("APLS AND THREE CG-FPLS STOPPING-RULE VARIANTS")
    print("=" * 72)
    print(
        f"Models=3, replications/model={replications}, n={n}, T={T}, "
        f"J={J}, m_max={m_max}, folds={k_folds}"
    )
    print(f"Orthogonalization: {orthogonalization.upper()}")
    print(f"FPCR tuning: {fpcr_gcv_criterion}-space GCV")
    print("CG-FPLS-code: moment-GCV pilot, one sigma2 estimate, m >= 1")
    print(
        "CG-FPLS-supplement: moment-GCV pilot, iterative sigma2, "
        f"xi={cg_variance_tolerance:g}, kmax={cg_variance_kmax}, m >= 0"
    )
    print(
        "CG-FPLS-oracle: known simulation sigma2="
        f"{noise_sd ** 2:g}, m >= 0 (diagnostic only)"
    )

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

            fpcr_path = _fpcr_spectral_path(
                r, K, min(m_max, n - 1, T - 1)
            )
            fpcr = _select_fpcr_gcv_from_path(
                y,
                X,
                r,
                K,
                fpcr_path,
                criterion=fpcr_gcv_criterion,
            )
            if fpcr_gcv_criterion == "moment":
                fpcr_moment_pilot = fpcr
            else:
                fpcr_moment_pilot = _select_fpcr_gcv_from_path(
                    y,
                    X,
                    r,
                    K,
                    fpcr_path,
                    criterion="moment",
                )
            cg_path = cg_fpls_path(r, K, m_max)
            cg_code = select_cg_fpls_code(
                cg_path,
                X,
                y,
                r,
                K,
                fpcr_moment_pilot.beta,
                tau=tau,
                delta=delta,
            )
            cg_supplement = select_cg_fpls_supplement(
                cg_path,
                X,
                y,
                r,
                K,
                fpcr_moment_pilot.beta,
                tau=tau,
                delta=delta,
                variance_tolerance=cg_variance_tolerance,
                variance_kmax=cg_variance_kmax,
            )
            cg_oracle = select_cg_fpls_oracle(
                cg_path,
                X,
                r,
                K,
                sigma2_true=noise_sd ** 2,
                tau=tau,
                delta=delta,
            )
            cg_selections = {
                CG_CODE_LABEL: cg_code,
                CG_SUPPLEMENT_LABEL: cg_supplement,
                CG_ORACLE_LABEL: cg_oracle,
            }
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
                **{
                    method: selection.beta
                    for method, selection in cg_selections.items()
                },
                "APLS-raw": apls.beta_raw,
                stable_label: apls.beta_reorth,
                fpcr_label: fpcr.beta,
            }
            m_by_method = {
                **{
                    method: selection.selected_components
                    for method, selection in cg_selections.items()
                },
                "APLS-raw": apls.m_raw,
                stable_label: apls.m_reorth,
                fpcr_label: fpcr.selected_components,
            }

            cg_pilot_selected_components[model_index, replication] = (
                fpcr_moment_pilot.selected_components
            )
            for cg_index, method in enumerate(CG_VARIANT_LABELS):
                selection = cg_selections[method]
                cg_initial_sigma2[
                    model_index, replication, cg_index
                ] = selection.sigma2_initial
                cg_final_sigma2[
                    model_index, replication, cg_index
                ] = selection.sigma2_final
                cg_variance_updates[
                    model_index, replication, cg_index
                ] = selection.variance_updates
                cg_variance_converged[
                    model_index, replication, cg_index
                ] = selection.converged
                cg_threshold_reached[
                    model_index, replication, cg_index
                ] = selection.threshold_reached
                cg_diagnostic_rows.append(
                    {
                        "model": model.name,
                        "replication": replication + 1,
                        "method": method,
                        "variant": selection.variant,
                        "comparison_role": method_metadata[method][
                            "comparison_role"
                        ],
                        "feasible_estimator": selection.feasible_estimator,
                        "minimum_components": selection.minimum_components,
                        "sigma2_source": selection.sigma2_source,
                        "pilot_gcv_criterion": (
                            "moment" if method != CG_ORACLE_LABEL else ""
                        ),
                        "pilot_selected_components": (
                            fpcr_moment_pilot.selected_components
                            if method != CG_ORACLE_LABEL
                            else ""
                        ),
                        "selected_components": selection.selected_components,
                        "m_zero_selected": selection.selected_components == 0,
                        "threshold_reached": selection.threshold_reached,
                        "variance_iteration_used": (
                            selection.variance_iteration_used
                        ),
                        "variance_converged": (
                            selection.converged
                            if selection.variance_iteration_used
                            else ""
                        ),
                        "variance_updates": selection.variance_updates,
                        "sigma2_initial": selection.sigma2_initial,
                        "sigma2_final": selection.sigma2_final,
                        "sigma2_history": json.dumps(
                            selection.sigma2_history.tolist()
                        ),
                        "selected_components_history": json.dumps(
                            selection.selected_components_history.tolist()
                        ),
                        "threshold_history": json.dumps(
                            selection.threshold_history.tolist()
                        ),
                    }
                )
            fpcr_diagnostic_rows.append(
                {
                    "model": model.name,
                    "replication": replication + 1,
                    "method": fpcr_label,
                    "gcv_criterion": fpcr.criterion,
                    "selected_components": fpcr.selected_components,
                    "selected_gcv": float(
                        fpcr.gcv[fpcr.selected_components - 1]
                    ),
                    "evaluated_components": len(fpcr.gcv),
                }
            )

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
                finite_result = bool(
                    np.isfinite(ise_value) and np.isfinite(mspe_value)
                )
                if method == "APLS-raw":
                    selected_index = apls.m_raw - 1
                    design_diagnostic = float(
                        apls.raw_full_path.design_condition[selected_index]
                    )
                    basis_diagnostic = float(
                        apls.raw_full_path.basis_condition[selected_index]
                    )
                    diagnostic_value = max(
                        design_diagnostic, basis_diagnostic
                    )
                    unstable = (
                        not finite_result
                        or diagnostic_value
                        >= RAW_NORMAL_EQUATIONS_CONDITION_THRESHOLD
                    )
                elif method == stable_label:
                    selected_index = apls.m_reorth - 1
                    diagnostic_value = float(
                        apls.reorth_full_path.orthogonality_defect[
                            selected_index
                        ]
                    )
                    unstable = (
                        not finite_result
                        or diagnostic_value >= ORTHOGONALITY_DEFECT_THRESHOLD
                    )
                else:
                    diagnostic_value = math.nan
                    unstable = not finite_result
                method_instability[
                    model_index, replication, method_index
                ] = unstable
                replication_rows.append(
                    {
                        "model": model.name,
                        "replication": replication + 1,
                        "method": method,
                        "comparison_role": method_metadata[method][
                            "comparison_role"
                        ],
                        "feasible_estimator": method_metadata[method][
                            "feasible_estimator"
                        ],
                        "selected_components": m_by_method[method],
                        "ise": ise_value,
                        "mspe": mspe_value,
                        "finite_result": finite_result,
                        "numerical_instability": unstable,
                        "instability_diagnostic": diagnostic_value,
                    }
                )

            raw_selected_design_condition[model_index, replication] = float(
                apls.raw_full_path.design_condition[apls.m_raw - 1]
            )
            raw_selected_basis_condition[model_index, replication] = float(
                apls.raw_full_path.basis_condition[apls.m_raw - 1]
            )
            stable_selected_orthogonality_defect[
                model_index, replication
            ] = float(
                apls.reorth_full_path.orthogonality_defect[apls.m_reorth - 1]
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
                    "raw_design_condition_at_selected_m": float(
                        apls.raw_full_path.design_condition[apls.m_raw - 1]
                    ),
                    "reorth_orthogonality_defect_at_selected_m": float(
                        apls.reorth_full_path.orthogonality_defect[
                            apls.m_reorth - 1
                        ]
                    ),
                    "comparison_component": common_component,
                    "relative_beta_difference_same_m": beta_difference_common,
                    "relative_fitted_difference_same_m": fitted_difference_common,
                    "cg_code_selected_m": cg_code.selected_components,
                    "cg_code_threshold_reached": (
                        cg_code.threshold_reached
                    ),
                    "cg_supplement_selected_m": (
                        cg_supplement.selected_components
                    ),
                    "cg_supplement_threshold_reached": (
                        cg_supplement.threshold_reached
                    ),
                    "cg_supplement_variance_converged": (
                        cg_supplement.converged
                    ),
                    "cg_supplement_variance_updates": (
                        cg_supplement.variance_updates
                    ),
                    "cg_oracle_selected_m": cg_oracle.selected_components,
                    "cg_oracle_threshold_reached": (
                        cg_oracle.threshold_reached
                    ),
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
    raw_instability_rows: List[Dict[str, object]] = []
    raw_outlier_rows: List[Dict[str, object]] = []
    cg_stopping_rows: List[Dict[str, object]] = []
    raw_method_index = methods.index("APLS-raw")
    stable_method_index = methods.index(stable_label)

    for model_index, model in enumerate(models):
        for method_index, method in enumerate(methods):
            ise_values = ise[model_index, :, method_index]
            mspe_values = mspe[model_index, :, method_index]
            finite = np.isfinite(ise_values) & np.isfinite(mspe_values)
            components = selected_components[model_index, :, method_index]
            extreme_ise = _extreme_tail_mask(
                ise_values, extreme_tail_multiple
            )
            extreme_mspe = _extreme_tail_mask(
                mspe_values, extreme_tail_multiple
            )
            instability = method_instability[model_index, :, method_index]
            if method == "APLS-raw":
                instability_rule = (
                    "nonfinite result, or selected normalized Krylov-basis or "
                    "response-design condition >= 1/sqrt(machine epsilon)"
                )
            elif method == stable_label:
                instability_rule = (
                    "nonfinite result or selected Arnoldi-basis orthogonality "
                    "defect >= sqrt(machine epsilon)"
                )
            else:
                instability_rule = "nonfinite result"
            summary_rows.append(
                {
                    "model": model.name,
                    "method": method,
                    "variant": method_metadata[method]["variant"],
                    "comparison_role": method_metadata[method][
                        "comparison_role"
                    ],
                    "feasible_estimator": method_metadata[method][
                        "feasible_estimator"
                    ],
                    "replications": replications,
                    "finite_replications": int(np.count_nonzero(finite)),
                    "failure_rate": float(1.0 - np.mean(finite)),
                    "mean_ise": _safe_mean(ise_values),
                    "se_ise": _standard_error(ise_values),
                    "median_ise": _safe_median(ise_values),
                    "p90_ise": _safe_quantile(ise_values, 0.90),
                    "p99_ise": _safe_quantile(ise_values, 0.99),
                    "max_ise": _safe_max(ise_values),
                    "mean_mspe": _safe_mean(mspe_values),
                    "se_mspe": _standard_error(mspe_values),
                    "median_mspe": _safe_median(mspe_values),
                    "p90_mspe": _safe_quantile(mspe_values, 0.90),
                    "p99_mspe": _safe_quantile(mspe_values, 0.99),
                    "max_mspe": _safe_max(mspe_values),
                    "mean_selected_components": float(np.mean(components)),
                    "median_selected_components": float(np.median(components)),
                    "p90_selected_components": float(
                        np.quantile(components, 0.90)
                    ),
                    "p99_selected_components": float(
                        np.quantile(components, 0.99)
                    ),
                    "numerical_instability_count": int(
                        np.count_nonzero(instability)
                    ),
                    "numerical_instability_rate": float(
                        np.mean(instability)
                    ),
                    "numerical_instability_rule": instability_rule,
                    "extreme_tail_multiple": extreme_tail_multiple,
                    "extreme_ise_count": int(np.count_nonzero(extreme_ise)),
                    "extreme_ise_rate": float(np.mean(extreme_ise)),
                    "extreme_mspe_count": int(np.count_nonzero(extreme_mspe)),
                    "extreme_mspe_rate": float(np.mean(extreme_mspe)),
                }
            )

        raw_ise = ise[model_index, :, raw_method_index]
        raw_mspe = mspe[model_index, :, raw_method_index]
        raw_components = selected_components[
            model_index, :, raw_method_index
        ]
        stable_components = selected_components[
            model_index, :, stable_method_index
        ]
        raw_instability = method_instability[
            model_index, :, raw_method_index
        ]
        extreme_raw_ise = _extreme_tail_mask(
            raw_ise, extreme_tail_multiple
        )
        extreme_raw_mspe = _extreme_tail_mask(
            raw_mspe, extreme_tail_multiple
        )
        raw_ise_median = _safe_median(raw_ise)
        raw_mspe_median = _safe_median(raw_mspe)
        outlier_union = raw_instability | extreme_raw_ise | extreme_raw_mspe
        for replication_index in np.flatnonzero(outlier_union):
            raw_outlier_rows.append(
                {
                    "model": model.name,
                    "replication": int(replication_index + 1),
                    "raw_selected_components": int(
                        raw_components[replication_index]
                    ),
                    "stable_selected_components": int(
                        stable_components[replication_index]
                    ),
                    "raw_selected_design_condition": float(
                        raw_selected_design_condition[
                            model_index, replication_index
                        ]
                    ),
                    "raw_selected_basis_condition": float(
                        raw_selected_basis_condition[
                            model_index, replication_index
                        ]
                    ),
                    "condition_threshold": (
                        RAW_NORMAL_EQUATIONS_CONDITION_THRESHOLD
                    ),
                    "numerical_instability": bool(
                        raw_instability[replication_index]
                    ),
                    "ise": float(raw_ise[replication_index]),
                    "ise_median": raw_ise_median,
                    "ise_multiple_of_median": (
                        float(raw_ise[replication_index] / raw_ise_median)
                        if raw_ise_median > 0.0
                        else math.nan
                    ),
                    "extreme_ise": bool(extreme_raw_ise[replication_index]),
                    "mspe": float(raw_mspe[replication_index]),
                    "mspe_median": raw_mspe_median,
                    "mspe_multiple_of_median": (
                        float(raw_mspe[replication_index] / raw_mspe_median)
                        if raw_mspe_median > 0.0
                        else math.nan
                    ),
                    "extreme_mspe": bool(
                        extreme_raw_mspe[replication_index]
                    ),
                }
            )

        raw_ise_finite = np.where(np.isfinite(raw_ise), raw_ise, -np.inf)
        raw_mspe_finite = np.where(np.isfinite(raw_mspe), raw_mspe, -np.inf)
        max_ise_index = int(np.argmax(raw_ise_finite))
        max_mspe_index = int(np.argmax(raw_mspe_finite))
        raw_instability_rows.append(
            {
                "model": model.name,
                "replications": replications,
                "condition_threshold": (
                    RAW_NORMAL_EQUATIONS_CONDITION_THRESHOLD
                ),
                "numerical_instability_count": int(
                    np.count_nonzero(raw_instability)
                ),
                "numerical_instability_rate": float(
                    np.mean(raw_instability)
                ),
                "raw_stable_component_disagreement_count": int(
                    np.count_nonzero(raw_components != stable_components)
                ),
                "raw_stable_component_disagreement_rate": float(
                    np.mean(raw_components != stable_components)
                ),
                "extreme_tail_multiple": extreme_tail_multiple,
                "extreme_ise_count": int(
                    np.count_nonzero(extreme_raw_ise)
                ),
                "extreme_ise_rate": float(np.mean(extreme_raw_ise)),
                "extreme_mspe_count": int(
                    np.count_nonzero(extreme_raw_mspe)
                ),
                "extreme_mspe_rate": float(np.mean(extreme_raw_mspe)),
                "max_ise": float(raw_ise[max_ise_index]),
                "max_ise_replication": max_ise_index + 1,
                "max_ise_selected_components": int(
                    raw_components[max_ise_index]
                ),
                "max_mspe": float(raw_mspe[max_mspe_index]),
                "max_mspe_replication": max_mspe_index + 1,
                "max_mspe_selected_components": int(
                    raw_components[max_mspe_index]
                ),
            }
        )

        for cg_index, method in enumerate(CG_VARIANT_LABELS):
            method_index = methods.index(method)
            cg_components = selected_components[
                model_index, :, method_index
            ]
            variance_iteration_used = method == CG_SUPPLEMENT_LABEL
            cg_stopping_rows.append(
                {
                    "model": model.name,
                    "method": method,
                    "variant": method_metadata[method]["variant"],
                    "comparison_role": method_metadata[method][
                        "comparison_role"
                    ],
                    "feasible_estimator": method_metadata[method][
                        "feasible_estimator"
                    ],
                    "replications": replications,
                    "minimum_components": method_metadata[method][
                        "minimum_components"
                    ],
                    "sigma2_source": method_metadata[method][
                        "sigma2_source"
                    ],
                    "variance_iteration_used": variance_iteration_used,
                    "variance_converged_count": (
                        int(
                            np.count_nonzero(
                                cg_variance_converged[
                                    model_index, :, cg_index
                                ]
                            )
                        )
                        if variance_iteration_used
                        else ""
                    ),
                    "variance_converged_rate": (
                        float(
                            np.mean(
                                cg_variance_converged[
                                    model_index, :, cg_index
                                ]
                            )
                        )
                        if variance_iteration_used
                        else ""
                    ),
                    "mean_variance_updates": float(
                        np.mean(
                            cg_variance_updates[
                                model_index, :, cg_index
                            ]
                        )
                    ),
                    "threshold_reached_count": int(
                        np.count_nonzero(
                            cg_threshold_reached[
                                model_index, :, cg_index
                            ]
                        )
                    ),
                    "threshold_reached_rate": float(
                        np.mean(
                            cg_threshold_reached[
                                model_index, :, cg_index
                            ]
                        )
                    ),
                    "m_zero_count": int(
                        np.count_nonzero(cg_components == 0)
                    ),
                    "m_zero_rate": float(np.mean(cg_components == 0)),
                    "m_max_count": int(
                        np.count_nonzero(cg_components == m_max)
                    ),
                    "m_max_rate": float(np.mean(cg_components == m_max)),
                    "mean_selected_components": float(
                        np.mean(cg_components)
                    ),
                    "median_selected_components": float(
                        np.median(cg_components)
                    ),
                    "mean_sigma2_initial": _safe_mean(
                        cg_initial_sigma2[model_index, :, cg_index]
                    ),
                    "mean_sigma2_final": _safe_mean(
                        cg_final_sigma2[model_index, :, cg_index]
                    ),
                    "median_sigma2_final": _safe_median(
                        cg_final_sigma2[model_index, :, cg_index]
                    ),
                    "true_sigma2": noise_sd ** 2,
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
        "true_noise_variance_for_oracle": noise_sd ** 2,
        "cg_tau": tau,
        "cg_delta": delta,
        "cg_common_path": (
            "identical conjugate-gradient iterates for all three variants"
        ),
        "cg_pilot_gcv_criterion": "moment",
        "cg_supplement_variance_tolerance_xi": cg_variance_tolerance,
        "cg_supplement_variance_kmax": cg_variance_kmax,
        "cg_supplement_variance_loop": (
            "k = 0,...,kmax (at most kmax+1 updates)"
        ),
        "cg_variants": {
            method: method_metadata[method]
            for method in CG_VARIANT_LABELS
        },
        "seed": seed,
        "basis_convention": basis_convention,
        "fpcr_gcv_criterion": fpcr_gcv_criterion,
        "fpcr_label": fpcr_label,
        "reorthogonalization": orthogonalization,
        "orthogonalized_apls_label": stable_label,
        "rank_tolerance": rank_tolerance,
        "extreme_tail_multiple": extreme_tail_multiple,
        "numerical_instability": {
            "raw_apls_condition_threshold": (
                RAW_NORMAL_EQUATIONS_CONDITION_THRESHOLD
            ),
            "raw_apls_rule": (
                "selected normalized Krylov-basis or response-design "
                "condition >= 1/sqrt(machine epsilon), or nonfinite result"
            ),
            "orthogonalized_apls_defect_threshold": (
                ORTHOGONALITY_DEFECT_THRESHOLD
            ),
            "orthogonalized_apls_rule": (
                "selected Arnoldi-basis orthogonality defect >= "
                "sqrt(machine epsilon), or nonfinite result"
            ),
            "extreme_tail_rule": (
                "metric strictly exceeds extreme_tail_multiple times its "
                "within-model/method median"
            ),
        },
        "raw_apls": {
            "basis": "[r, Kr, ..., K^(m-1)r]",
            "solver": "unregularized normal equations via numpy.linalg.solve",
            "column_scaling": False,
            "ridge": False,
            "pseudoinverse_fallback": False,
            "orthogonalization": False,
        },
        "methods": list(methods),
        "method_metadata": method_metadata,
    }

    _write_csv(output_dir / "replication_results.csv", replication_rows)
    _write_csv(output_dir / "summary.csv", summary_rows)
    _write_csv(
        output_dir / "summary_feasible.csv",
        [row for row in summary_rows if row["feasible_estimator"]],
    )
    _write_csv(output_dir / "apls_diagnostics.csv", diagnostic_rows)
    _write_csv(output_dir / "apls_path_diagnostics.csv", path_rows)
    _write_csv(output_dir / "cg_fpls_diagnostics.csv", cg_diagnostic_rows)
    _write_csv(output_dir / "cg_fpls_stopping_summary.csv", cg_stopping_rows)
    _write_csv(output_dir / "fpcr_diagnostics.csv", fpcr_diagnostic_rows)
    _write_csv(
        output_dir / "raw_apls_instability_summary.csv",
        raw_instability_rows,
    )
    raw_outlier_fields = [
        "model",
        "replication",
        "raw_selected_components",
        "stable_selected_components",
        "raw_selected_design_condition",
        "raw_selected_basis_condition",
        "condition_threshold",
        "numerical_instability",
        "ise",
        "ise_median",
        "ise_multiple_of_median",
        "extreme_ise",
        "mspe",
        "mspe_median",
        "mspe_multiple_of_median",
        "extreme_mspe",
    ]
    _write_csv(
        output_dir / "raw_apls_outliers.csv",
        raw_outlier_rows,
        fieldnames=raw_outlier_fields,
    )
    with (output_dir / "configuration.json").open("w", encoding="utf-8") as handle:
        json.dump(configuration, handle, indent=2)
        handle.write("\n")
    cg_supplement_index = CG_VARIANT_LABELS.index(CG_SUPPLEMENT_LABEL)
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
        method_instability=method_instability,
        raw_selected_design_condition=raw_selected_design_condition,
        raw_selected_basis_condition=raw_selected_basis_condition,
        stable_selected_orthogonality_defect=(
            stable_selected_orthogonality_defect
        ),
        # Backward-compatible unqualified arrays refer to the supplement rule.
        cg_initial_sigma2=(
            cg_initial_sigma2[:, :, cg_supplement_index]
        ),
        cg_final_sigma2=cg_final_sigma2[:, :, cg_supplement_index],
        cg_variance_updates=(
            cg_variance_updates[:, :, cg_supplement_index]
        ),
        cg_variance_converged=(
            cg_variance_converged[:, :, cg_supplement_index]
        ),
        cg_threshold_reached=(
            cg_threshold_reached[:, :, cg_supplement_index]
        ),
        cg_variant_initial_sigma2=cg_initial_sigma2,
        cg_variant_final_sigma2=cg_final_sigma2,
        cg_variant_variance_updates=cg_variance_updates,
        cg_variant_variance_converged=cg_variance_converged,
        cg_variant_threshold_reached=cg_threshold_reached,
        cg_pilot_selected_components=cg_pilot_selected_components,
        cg_variant_names=np.asarray(CG_VARIANT_LABELS),
        method_names=np.asarray(methods),
        model_names=np.asarray([model.name for model in models]),
    )

    if make_plots:
        feasible_method_indices = [
            index
            for index, method in enumerate(methods)
            if bool(method_metadata[method]["feasible_estimator"])
        ]
        plot_performance(
            ise,
            mspe,
            selected_components,
            models,
            methods,
            extreme_tail_multiple,
            output_dir,
            method_indices=feasible_method_indices,
            filename="performance_comparison.pdf",
            title=(
                "Feasible estimator performance "
                "(all boxplot outliers shown; raw extremes marked)"
            ),
        )
        plot_performance(
            ise,
            mspe,
            selected_components,
            models,
            methods,
            extreme_tail_multiple,
            output_dir,
            filename="performance_comparison_with_oracle.pdf",
            title=(
                "Estimator performance including infeasible CG oracle "
                "(diagnostic only)"
            ),
        )
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
        plot_cg_fpls_variants(
            selected_components,
            cg_final_sigma2,
            models,
            methods,
            noise_sd ** 2,
            output_dir,
        )
        plot_representative_cv(representative, stable_label, output_dir)
        plot_raw_apls_outliers(
            ise,
            mspe,
            selected_components,
            method_instability,
            np.maximum(
                raw_selected_design_condition,
                raw_selected_basis_condition,
            ),
            models,
            methods,
            extreme_tail_multiple,
            output_dir,
        )

    print("\nSummary")
    for row in summary_rows:
        diagnostic_marker = (
            " [diagnostic only]"
            if not row["feasible_estimator"]
            else ""
        )
        print(
            f"  {row['model']} | {row['method']:<26} | "
            f"mean ISE={row['mean_ise']:.6g} | "
            f"median ISE={row['median_ise']:.6g} | "
            f"p99 ISE={row['p99_ise']:.6g} | "
            f"mean MSPE={row['mean_mspe']:.6g} | "
            f"instability={row['numerical_instability_rate']:.2%} | "
            f"mean m={row['mean_selected_components']:.2f}"
            f"{diagnostic_marker}"
        )
    print(f"\nResults written to {output_dir.resolve()}")

    return {
        "configuration": configuration,
        "summary": summary_rows,
        "replications": replication_rows,
        "diagnostics": diagnostic_rows,
        "cg_diagnostics": cg_diagnostic_rows,
        "cg_stopping_summary": cg_stopping_rows,
        "fpcr_diagnostics": fpcr_diagnostic_rows,
        "raw_instability_summary": raw_instability_rows,
        "raw_outliers": raw_outlier_rows,
        "ise": ise,
        "mspe": mspe,
        "selected_components": selected_components,
        "raw_conditions": raw_conditions,
        "reorth_conditions": reorth_conditions,
        "reorth_defects": reorth_defects,
        "beta_path_discrepancy": beta_path_discrepancy,
        "fitted_path_discrepancy": fitted_path_discrepancy,
        "raw_valid": raw_valid,
        "method_instability": method_instability,
        "raw_selected_design_condition": raw_selected_design_condition,
        "raw_selected_basis_condition": raw_selected_basis_condition,
        "stable_selected_orthogonality_defect": (
            stable_selected_orthogonality_defect
        ),
        "cg_initial_sigma2": (
            cg_initial_sigma2[:, :, cg_supplement_index]
        ),
        "cg_final_sigma2": (
            cg_final_sigma2[:, :, cg_supplement_index]
        ),
        "cg_variance_updates": (
            cg_variance_updates[:, :, cg_supplement_index]
        ),
        "cg_variance_converged": (
            cg_variance_converged[:, :, cg_supplement_index]
        ),
        "cg_threshold_reached": (
            cg_threshold_reached[:, :, cg_supplement_index]
        ),
        "cg_variant_initial_sigma2": cg_initial_sigma2,
        "cg_variant_final_sigma2": cg_final_sigma2,
        "cg_variant_variance_updates": cg_variance_updates,
        "cg_variant_variance_converged": cg_variance_converged,
        "cg_variant_threshold_reached": cg_threshold_reached,
        "cg_pilot_selected_components": cg_pilot_selected_components,
        "cg_variant_names": CG_VARIANT_LABELS,
    }


def _finite_positive(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values) & (values > 0.0)]


def plot_performance(
    ise: np.ndarray,
    mspe: np.ndarray,
    selected_components: np.ndarray,
    models: Sequence[ModelSpec],
    methods: Sequence[str],
    extreme_tail_multiple: float,
    output_dir: Path,
    *,
    method_indices: Optional[Sequence[int]] = None,
    filename: str = "performance_comparison.pdf",
    title: str = (
        "Estimator performance "
        "(all boxplot outliers shown; raw extremes marked)"
    ),
) -> None:
    """Boxplots that retain and explicitly annotate raw-APLS tail cases."""
    if method_indices is None:
        display_indices = list(range(len(methods)))
    else:
        display_indices = [int(index) for index in method_indices]
    if not display_indices or any(
        index < 0 or index >= len(methods) for index in display_indices
    ):
        raise ValueError("method_indices contains no valid display methods")
    display_methods = [methods[index] for index in display_indices]
    raw_index = methods.index("APLS-raw")
    if raw_index not in display_indices:
        raise ValueError("performance display must include APLS-raw")
    raw_display_index = display_indices.index(raw_index)
    fig, axes = plt.subplots(2, len(models), figsize=(18, 9.5), squeeze=False)
    for model_index, model in enumerate(models):
        for row_index, (metric, label) in enumerate(
            ((ise, "Integrated squared error"), (mspe, "Test MSPE"))
        ):
            ax = axes[row_index, model_index]
            data = [
                _finite_positive(metric[model_index, :, method_index])
                for method_index in display_indices
            ]
            artists = ax.boxplot(
                data,
                showfliers=True,
                patch_artist=True,
                flierprops={
                    "marker": ".",
                    "markersize": 2.5,
                    "markerfacecolor": "#666666",
                    "markeredgecolor": "#666666",
                    "alpha": 0.35,
                },
            )
            for patch, method in zip(artists["boxes"], display_methods):
                patch.set_facecolor(METHOD_COLORS.get(method, "#999999"))
                patch.set_alpha(0.65)

            raw_values = metric[model_index, :, raw_index]
            extreme = _extreme_tail_mask(
                raw_values, extreme_tail_multiple
            )
            extreme_indices = np.flatnonzero(extreme)
            if len(extreme_indices):
                ax.scatter(
                    np.full(len(extreme_indices), raw_display_index + 1),
                    raw_values[extreme_indices],
                    marker="D",
                    s=24,
                    facecolors="none",
                    edgecolors="#B2182B",
                    linewidths=0.9,
                    zorder=4,
                )
            finite_for_max = np.where(
                np.isfinite(raw_values), raw_values, -np.inf
            )
            maximum_index = int(np.argmax(finite_for_max))
            annotation = (
                f"Raw > {extreme_tail_multiple:g}x median: "
                f"{len(extreme_indices)}\n"
                f"Raw max: {raw_values[maximum_index]:.3g} "
                f"(rep {maximum_index + 1}, "
                f"m={selected_components[model_index, maximum_index, raw_index]})"
            )
            ax.text(
                0.03,
                0.97,
                annotation,
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=7.5,
                bbox={
                    "boxstyle": "round,pad=0.25",
                    "facecolor": "white",
                    "edgecolor": "#BBBBBB",
                    "alpha": 0.88,
                },
            )
            ax.set_yscale("log")
            ax.set_xticks(range(1, len(display_methods) + 1))
            ax.set_xticklabels(
                display_methods, rotation=32, ha="right", fontsize=8
            )
            ax.grid(True, which="both", axis="y", alpha=0.25)
            if row_index == 0:
                ax.set_title(model.name, fontweight="bold")
            if model_index == 0:
                ax.set_ylabel(label)
    fig.suptitle(title, y=0.995)
    fig.tight_layout()
    _save_pdf(fig, output_dir / filename)
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
            RAW_NORMAL_EQUATIONS_CONDITION_THRESHOLD,
            color="#B2182B",
            linestyle="--",
            linewidth=1.1,
            label=(
                "Instability flag: 1 / sqrt(machine epsilon)"
                if model_index == 0
                else None
            ),
        )
        ax.axhline(
            CONDITION_CAP,
            color="#666666",
            linestyle=":",
            linewidth=1.0,
            label=(
                "Diagnostic cap: 1 / machine epsilon"
                if model_index == 0
                else None
            ),
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
    stable_label = next(
        method for method in methods if method.startswith("APLS-Arnoldi-")
    )
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


def plot_cg_fpls_variants(
    selected_components: np.ndarray,
    cg_final_sigma2: np.ndarray,
    models: Sequence[ModelSpec],
    methods: Sequence[str],
    true_sigma2: float,
    output_dir: Path,
) -> None:
    """Compare stopping and variance calibration across CG-FPLS variants."""
    if cg_final_sigma2.shape != (
        len(models),
        selected_components.shape[1],
        len(CG_VARIANT_LABELS),
    ):
        raise ValueError("cg_final_sigma2 has an incompatible shape")
    short_labels = ("Code", "Supplement", "Oracle")
    colors = [METHOD_COLORS[label] for label in CG_VARIANT_LABELS]
    method_indices = [methods.index(label) for label in CG_VARIANT_LABELS]
    fig, axes = plt.subplots(2, len(models), figsize=(16, 8.5), squeeze=False)

    for model_index, model in enumerate(models):
        component_data = [
            selected_components[model_index, :, method_index]
            for method_index in method_indices
        ]
        component_artists = axes[0, model_index].boxplot(
            component_data,
            showfliers=True,
            patch_artist=True,
            flierprops={
                "marker": ".",
                "markersize": 2.5,
                "markerfacecolor": "#666666",
                "markeredgecolor": "#666666",
                "alpha": 0.35,
            },
        )
        for patch, color in zip(component_artists["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.68)
        axes[0, model_index].set_xticks(range(1, 4))
        axes[0, model_index].set_xticklabels(short_labels, rotation=20)
        axes[0, model_index].set_title(model.name, fontweight="bold")
        axes[0, model_index].set_xlabel("Stopping-rule variant")
        axes[0, model_index].grid(True, axis="y", alpha=0.25)
        if model_index == 0:
            axes[0, model_index].set_ylabel("Selected CG components m")

        sigma_data = [
            cg_final_sigma2[model_index, :, variant_index]
            for variant_index in range(len(CG_VARIANT_LABELS))
        ]
        sigma_artists = axes[1, model_index].boxplot(
            sigma_data,
            showfliers=True,
            patch_artist=True,
            flierprops={
                "marker": ".",
                "markersize": 2.5,
                "markerfacecolor": "#666666",
                "markeredgecolor": "#666666",
                "alpha": 0.35,
            },
        )
        for patch, color in zip(sigma_artists["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.68)
        axes[1, model_index].axhline(
            true_sigma2,
            color="#222222",
            linestyle="--",
            linewidth=1.1,
            label="True noise variance" if model_index == 0 else None,
        )
        axes[1, model_index].set_xticks(range(1, 4))
        axes[1, model_index].set_xticklabels(short_labels, rotation=20)
        axes[1, model_index].set_xlabel("Variance-calibration variant")
        axes[1, model_index].grid(True, axis="y", alpha=0.25)
        if model_index == 0:
            axes[1, model_index].set_ylabel("Variance used for stopping")
            axes[1, model_index].legend(fontsize=8)

    fig.suptitle(
        "CG-FPLS stopping variants (oracle is an infeasible diagnostic)",
        y=0.995,
    )
    fig.tight_layout()
    _save_pdf(fig, output_dir / "cg_fpls_variant_diagnostics.pdf")
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


def plot_raw_apls_outliers(
    ise: np.ndarray,
    mspe: np.ndarray,
    selected_components: np.ndarray,
    method_instability: np.ndarray,
    raw_selected_condition: np.ndarray,
    models: Sequence[ModelSpec],
    methods: Sequence[str],
    extreme_tail_multiple: float,
    output_dir: Path,
) -> None:
    """Show every raw-APLS replication and identify numerical tail cases."""
    raw_index = methods.index("APLS-raw")
    replications = ise.shape[1]
    replication_axis = np.arange(1, replications + 1)
    fig, axes = plt.subplots(2, len(models), figsize=(16, 8.5), squeeze=False)

    for model_index, model in enumerate(models):
        instability = method_instability[model_index, :, raw_index]
        for row_index, (metric, label) in enumerate(
            ((ise, "Raw APLS ISE"), (mspe, "Raw APLS test MSPE"))
        ):
            ax = axes[row_index, model_index]
            values = np.asarray(metric[model_index, :, raw_index], dtype=float)
            positive = np.isfinite(values) & (values > 0.0)
            extreme = _extreme_tail_mask(values, extreme_tail_multiple)
            ordinary = positive & ~instability & ~extreme
            ax.scatter(
                replication_axis[ordinary],
                values[ordinary],
                s=7,
                color="#777777",
                alpha=0.38,
                linewidths=0,
                label="Other replication" if model_index == 0 and row_index == 0 else None,
            )
            ax.scatter(
                replication_axis[positive & instability],
                values[positive & instability],
                s=24,
                marker="^",
                color="#B2182B",
                alpha=0.8,
                linewidths=0,
                label="Numerically unstable" if model_index == 0 and row_index == 0 else None,
            )
            ax.scatter(
                replication_axis[positive & extreme],
                values[positive & extreme],
                s=30,
                marker="D",
                facecolors="none",
                edgecolors="#E66101",
                linewidths=1.0,
                label=(
                    f"> {extreme_tail_multiple:g}x median"
                    if model_index == 0 and row_index == 0
                    else None
                ),
            )
            median = _safe_median(values)
            if np.isfinite(median) and median > 0.0:
                ax.axhline(
                    median,
                    color="#2166AC",
                    linewidth=1.1,
                    linestyle="--",
                    label="Median" if model_index == 0 and row_index == 0 else None,
                )
                if np.any(extreme):
                    ax.axhline(
                        extreme_tail_multiple * median,
                        color="#E66101",
                        linewidth=1.0,
                        linestyle=":",
                    )
            maximum_values = np.where(positive, values, -np.inf)
            maximum_index = int(np.argmax(maximum_values))
            annotation = (
                f"max={values[maximum_index]:.3g}\n"
                f"rep={maximum_index + 1}, "
                f"m={selected_components[model_index, maximum_index, raw_index]}\n"
                f"max cond={raw_selected_condition[model_index, maximum_index]:.2e}"
            )
            ax.text(
                0.98,
                0.97,
                annotation,
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=7.5,
                bbox={
                    "boxstyle": "round,pad=0.25",
                    "facecolor": "white",
                    "edgecolor": "#BBBBBB",
                    "alpha": 0.88,
                },
            )
            ax.set_yscale("log")
            ax.set_xlabel("Replication")
            ax.grid(True, which="both", axis="y", alpha=0.22)
            if row_index == 0:
                ax.set_title(model.name, fontweight="bold")
            if model_index == 0:
                ax.set_ylabel(label)
    axes[0, 0].legend(fontsize=7.5, loc="lower right")
    fig.suptitle(
        "Raw APLS tail diagnostics: no outliers suppressed",
        y=0.995,
    )
    fig.tight_layout()
    _save_pdf(fig, output_dir / "raw_apls_outliers.pdf")
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

    fpcr_response = select_fpcr_gcv(
        y, X, r, K, 8, criterion="response"
    )
    fpcr_moment = select_fpcr_gcv(
        y, X, r, K, 8, criterion="moment"
    )
    if not (
        1 <= fpcr_response.selected_components <= 8
        and 1 <= fpcr_moment.selected_components <= 8
        and np.all(np.isfinite(fpcr_response.gcv))
        and np.all(np.isfinite(fpcr_moment.gcv))
    ):
        raise AssertionError("FPCR GCV selection returned an invalid result")
    response_residual = y - X @ fpcr_response.beta / T
    expected_response_gcv = np.mean(response_residual ** 2) / (
        1.0 - fpcr_response.selected_components / n
    ) ** 2
    if not np.isclose(
        expected_response_gcv,
        fpcr_response.gcv[fpcr_response.selected_components - 1],
        rtol=1e-12,
        atol=1e-14,
    ):
        raise AssertionError("response-space FPCR GCV scaling is incorrect")

    cg_path = cg_fpls_path(r, K, 8)
    cg_code = select_cg_fpls_code(
        cg_path,
        X,
        y,
        r,
        K,
        fpcr_moment.beta,
        tau=1.01,
        delta=0.1,
    )
    cg_supplement = select_cg_fpls_supplement(
        cg_path,
        X,
        y,
        r,
        K,
        fpcr_moment.beta,
        tau=1.01,
        delta=0.1,
        variance_tolerance=0.01,
        variance_kmax=10,
    )
    cg_oracle = select_cg_fpls_oracle(
        cg_path,
        X,
        r,
        K,
        sigma2_true=0.01,
        tau=1.01,
        delta=0.1,
    )
    expected_code_sigma2 = float(
        np.mean((y - X @ fpcr_moment.beta / T) ** 2)
    )
    if not (
        1 <= cg_code.selected_components <= 8
        and cg_code.minimum_components == 1
        and cg_code.variance_updates == 0
        and np.isclose(cg_code.sigma2_final, expected_code_sigma2)
        and 0 <= cg_supplement.selected_components <= 8
        and cg_supplement.variance_updates >= 1
        and len(cg_supplement.sigma2_history)
        == cg_supplement.variance_updates + 1
        and np.isclose(
            cg_supplement.moment_path[0], np.sqrt(np.mean(r ** 2))
        )
        and 0 <= cg_oracle.selected_components <= 8
        and np.isclose(cg_oracle.sigma2_final, 0.01)
        and not cg_oracle.feasible_estimator
    ):
        raise AssertionError("CG-FPLS variant diagnostics are inconsistent")
    _, selected_zero, reached_zero, _, _ = _select_cg_for_variance(
        cg_path,
        r,
        K,
        sigma2=1e20,
        X_norm=float(np.mean(np.sum(X ** 2, axis=1) / T)),
        n=n,
        tau=1.01,
        delta=0.1,
    )
    if selected_zero != 0 or not reached_zero:
        raise AssertionError("CG-FPLS discrepancy search does not admit m=0")
    _, selected_one, reached_one, _, _ = _select_cg_for_variance(
        cg_path,
        r,
        K,
        sigma2=1e20,
        X_norm=float(np.mean(np.sum(X ** 2, axis=1) / T)),
        n=n,
        tau=1.01,
        delta=0.1,
        minimum_components=1,
    )
    if selected_one != 1 or not reached_one:
        raise AssertionError("released-code CG selector does not enforce m >= 1")
    cg_alias = select_cg_fpls(
        cg_path,
        X,
        y,
        r,
        K,
        fpcr_moment.beta,
        tau=1.01,
        delta=0.1,
        variance_tolerance=0.01,
        variance_kmax=10,
    )
    if (
        cg_alias.selected_components != cg_supplement.selected_components
        or not np.allclose(cg_alias.beta, cg_supplement.beta)
    ):
        raise AssertionError("CG-FPLS backward-compatible alias changed")
    print(
        "Self-tests passed: raw formula, same-m equivalence, MGS1/CGS2/MGS2 "
        "agreement, orthogonality, fold-local CV, both FPCR GCV criteria, "
        "shared CG path, code/supplement/oracle stopping variants, and the "
        "m=0 versus m>=1 distinction."
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
    parser.add_argument(
        "--cg-variance-tolerance",
        type=float,
        default=0.01,
        help=(
            "supplement-variant variance-iteration tolerance xi "
            "(default: 0.01)"
        ),
    )
    parser.add_argument(
        "--cg-variance-kmax",
        type=int,
        default=10,
        help=(
            "supplement-variant variance-iteration kmax (default: 10)"
        ),
    )
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument(
        "--basis-convention",
        choices=("babii", "standard"),
        default="babii",
    )
    parser.add_argument(
        "--fpcr-gcv",
        choices=("response", "moment"),
        default="response",
        help=(
            "response is conventional regression GCV; moment reproduces "
            "the earlier notebook and is labelled explicitly"
        ),
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
        "--extreme-tail-multiple",
        type=float,
        default=100.0,
        help=(
            "flag metrics above this multiple of their model/method median "
            "(default: 100)"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("apls_cg_variants_results"),
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
        cg_variance_tolerance=arguments.cg_variance_tolerance,
        cg_variance_kmax=arguments.cg_variance_kmax,
        seed=arguments.seed,
        basis_convention=arguments.basis_convention,
        fpcr_gcv_criterion=arguments.fpcr_gcv,
        orthogonalization=arguments.orthogonalization,
        rank_tolerance=arguments.rank_tolerance,
        extreme_tail_multiple=arguments.extreme_tail_multiple,
        output_dir=arguments.output_dir,
        make_plots=not arguments.no_plots,
    )


if __name__ == "__main__":
    main()
