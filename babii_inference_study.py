r"""Babii et al. CG-FPLS inference study for scalar-on-function regression.

This standalone program implements the inference branch of

    Babii, Carrasco, and Tsafack,
    "Functional Partial Least-Squares: Adaptive Estimation and Inference."

It deliberately remains separate from ``model_8_corrected.py``, which compares
early-stopped CG-FPLS, standard FPLS/APLS, and FPCR for estimation and
prediction.  The present program studies the hypothesis

    H0: beta = b  versus  H1: beta != b

with the late-stopped statistic

    T_n(b) = n || K_hat (beta_hat_m - b) ||_L2^2.

The discretisation is exactly the one used in the supplied Julia notebook:

    y = X @ beta / T + epsilon,
    r = X.T @ y / n,
    K = X.T @ X / (n * T),
    ||f||_L2^2 = mean(f ** 2).

The paper baseline uses m=70 and oracle weighted-chi-square calibration.  This
implementation also provides:

* the direct-moment identity S_n(b) = n ||r - Kb||^2;
* feasible plug-in and centered multiplier calibrations;
* early-versus-late stopping diagnostics;
* analytic finite-dimensional confidence ellipsoids and envelopes;
* an optional, memory-batched reproduction of the paper's coefficient grid;
* CSV, NPZ, JSON, and vector-PDF outputs.

Examples
--------
Quick end-to-end run::

    python babii_inference_study.py --replications 20

More stable appendix run::

    python babii_inference_study.py \
        --replications 500 \
        --calibrations oracle plugin multiplier \
        --output-dir babii_inference_output

Paper-scale baseline (computationally intensive)::

    python babii_inference_study.py \
        --size-replications 5000 \
        --power-replications 5000 \
        --confidence-replications 5000 \
        --stopping-replications 5000 \
        --calibrations oracle \
        --delta-step 0.05
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.linalg import eigh, solve_triangular


plt.rcParams.update(
    {
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "semibold",
        "figure.dpi": 120,
        "font.size": 10,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


MODEL_COLORS = {
    "Model 1": "#2F6B9A",
    "Model 2": "#D9772B",
    "Model 3": "#3E8E6B",
}
CALIBRATION_COLORS = {
    "oracle": "#2F6B9A",
    "plugin": "#D9772B",
    "multiplier": "#3E8E6B",
}


@dataclass(frozen=True)
class ModelSpec:
    """One of the three Monte Carlo designs."""

    name: str
    eigenvalues: np.ndarray
    beta: np.ndarray
    coefficients: np.ndarray


@dataclass(frozen=True)
class Design:
    """Shared grid, basis, and model definitions."""

    s: np.ndarray
    basis: np.ndarray
    models: Tuple[ModelSpec, ...]
    noise_sd: float
    basis_convention: str

    @property
    def T(self) -> int:
        return int(self.s.size)

    @property
    def J(self) -> int:
        return int(self.basis.shape[0])


@dataclass
class KrylovPath:
    """Nested minimum-moment-residual CG-FPLS path."""

    beta: np.ndarray
    fitted_moment: np.ndarray
    residual_norm: np.ndarray
    effective_dimension: int
    numerical_rank: int
    orthogonality_error: float
    breakdown: bool


@dataclass
class ConfidenceEnvelope:
    """Finite-dimensional test-inversion result."""

    lower: np.ndarray
    upper: np.ndarray
    center_curve: np.ndarray
    center_coefficients: np.ndarray
    radius_squared: float
    minimum_statistic: float
    rank: int
    empty: bool
    unbounded: bool


def _safe_key(value: str) -> str:
    return value.lower().replace(" ", "_").replace("-", "_")


def create_cosine_basis(
    s: np.ndarray, J: int, convention: str = "babii"
) -> np.ndarray:
    """Return a J-by-T cosine basis.

    ``convention='babii'`` reproduces the supplied notebook exactly: the first
    vector is constant and the remaining vectors are
    ``sqrt(2) cos(j*pi*s)`` for ``j=2,...,J``.  Thus the notebook convention
    omits ``cos(pi*s)``.  ``convention='standard'`` uses frequencies
    ``0,...,J-1``.
    """

    s = np.asarray(s, dtype=float).reshape(-1)
    if s.size < 2 or not np.all(np.isfinite(s)) or np.any(np.diff(s) <= 0):
        raise ValueError("s must be a finite, strictly increasing grid")
    if J < 1:
        raise ValueError("J must be positive")
    if convention == "babii":
        frequencies = np.arange(1, J + 1, dtype=float)
    elif convention == "standard":
        frequencies = np.arange(J, dtype=float)
    else:
        raise ValueError("convention must be 'babii' or 'standard'")
    basis_t_by_j = np.sqrt(2.0) * np.cos(np.pi * np.outer(s, frequencies))
    basis_t_by_j[:, 0] = 1.0
    return basis_t_by_j.T


def build_design(
    T: int = 200,
    J: int = 100,
    noise_sd: float = 1.0,
    basis_convention: str = "babii",
) -> Design:
    """Construct the three models from the paper and Julia notebook."""

    if T < 2 or J < 1 or noise_sd < 0:
        raise ValueError("T and J must be positive and noise_sd non-negative")
    s = np.linspace(0.0, 1.0, T)
    basis = create_cosine_basis(s, J, basis_convention)
    j = np.arange(1, J + 1, dtype=float)
    b1 = 4.0 / j**2.7
    beta1 = basis.T @ b1
    lambda1 = 2.0 / j**1.1
    b2 = b1.copy()
    b2[: min(5, J)] = 4.0
    beta2 = basis.T @ b2
    lambda3 = lambda1.copy()
    lambda3[: min(5, J)] = 2.0
    models = (
        ModelSpec("Model 1", lambda1, beta1, b1),
        ModelSpec("Model 2", lambda1.copy(), beta2, b2),
        ModelSpec("Model 3", lambda3, beta1.copy(), b1.copy()),
    )
    return Design(s, basis, models, float(noise_sd), basis_convention)


def simulate_sample(
    model: ModelSpec,
    design: Design,
    n: int,
    rng: np.random.Generator,
    beta_data: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate one sample using the paper's functional linear model."""

    if n < 2:
        raise ValueError("n must be at least two")
    beta = model.beta if beta_data is None else np.asarray(beta_data, dtype=float)
    if beta.shape != (design.T,):
        raise ValueError("beta_data has the wrong shape")
    scores = rng.normal(size=(n, design.J))
    X = (scores * np.sqrt(model.eigenvalues)) @ design.basis
    epsilon = rng.normal(0.0, design.noise_sd, size=n)
    y = X @ beta / design.T + epsilon
    return X, y, epsilon


def empirical_moments(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return r_hat and K_hat under the exact 1/T scaling."""

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    if X.ndim != 2 or X.shape[0] != y.size:
        raise ValueError("X and y have incompatible shapes")
    n, T = X.shape
    r = X.T @ y / n
    K = X.T @ X / (n * T)
    K = 0.5 * (K + K.T)
    return r, K


def l2_norm(vector: np.ndarray) -> float:
    """Discrete L2 norm with one Riemann-sum factor 1/T."""

    vector = np.asarray(vector, dtype=float).reshape(-1)
    return float(np.sqrt(np.mean(vector**2)))


def test_statistic(
    fitted_moment: np.ndarray,
    K: np.ndarray,
    null_beta: np.ndarray,
    n: int,
) -> float:
    """Compute T_n(b)=n||K beta_hat_m-Kb||^2."""

    difference = np.asarray(fitted_moment) - np.asarray(K) @ np.asarray(null_beta)
    return float(n * np.mean(difference**2))


def direct_moment_statistic(
    r: np.ndarray, K: np.ndarray, null_beta: np.ndarray, n: int
) -> float:
    """Compute S_n(b)=n||r-Kb||^2, the late-stopping reference."""

    difference = np.asarray(r) - np.asarray(K) @ np.asarray(null_beta)
    return float(n * np.mean(difference**2))


def direct_moment_statistic_from_data(
    X: np.ndarray, y: np.ndarray, null_beta: np.ndarray
) -> float:
    """Compute S_n(b) directly from null score vectors, without forming K."""

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    null_beta = np.asarray(null_beta, dtype=float).reshape(-1)
    n, T = X.shape
    if y.size != n or null_beta.size != T:
        raise ValueError("X, y, and null_beta have incompatible shapes")
    null_residuals = y - X @ null_beta / T
    moment = X.T @ null_residuals / n
    return float(n * np.mean(moment**2))


def simulate_direct_moment_statistic(
    model: ModelSpec,
    design: Design,
    n: int,
    rng: np.random.Generator,
    beta_data: np.ndarray,
    null_beta: np.ndarray,
    basis_gram: Optional[np.ndarray] = None,
) -> float:
    """Fast exact S_n(b) simulation in the J-dimensional score coordinates."""

    beta_data = np.asarray(beta_data, dtype=float).reshape(-1)
    null_beta = np.asarray(null_beta, dtype=float).reshape(-1)
    if beta_data.size != design.T or null_beta.size != design.T:
        raise ValueError("beta_data or null_beta has the wrong length")
    score_coefficients = rng.normal(size=(n, design.J)) * np.sqrt(model.eigenvalues)
    data_projection = design.basis @ beta_data / design.T
    null_projection = design.basis @ null_beta / design.T
    y = score_coefficients @ data_projection + rng.normal(
        0.0, design.noise_sd, size=n
    )
    null_residuals = y - score_coefficients @ null_projection
    coefficient_moment = score_coefficients.T @ null_residuals / n
    if basis_gram is None:
        basis_gram = design.basis @ design.basis.T / design.T
    return float(n * coefficient_moment @ basis_gram @ coefficient_moment)


def stable_cg_fpls_path(
    r: np.ndarray,
    K: np.ndarray,
    m_max: int,
    rcond: float = 1e-12,
    krylov_tolerance: float = 1e-13,
) -> KrylovPath:
    """Reorthogonalized minimum-residual realization of Babii's CG-FPLS.

    In exact arithmetic the paper's recurrence and this routine solve the same
    nested problems

        min ||r-K beta||  over span(r, Kr, ..., K^(m-1)r).

    Full two-pass reorthogonalization prevents the severe loss of Krylov-basis
    independence that otherwise occurs when m is deliberately large for
    inference.
    """

    r = np.asarray(r, dtype=float).reshape(-1)
    K = np.asarray(K, dtype=float)
    T = r.size
    if K.shape != (T, T) or m_max < 1:
        raise ValueError("K has the wrong shape or m_max is not positive")
    if rcond <= 0 or krylov_tolerance <= 0:
        raise ValueError("rcond and krylov_tolerance must be positive")
    K = 0.5 * (K + K.T)
    beta_path = np.zeros((T, m_max))
    fitted_path = np.zeros((T, m_max))
    residual_norm = np.full(m_max, l2_norm(r))
    norm_r = float(np.linalg.norm(r))
    if norm_r <= np.finfo(float).tiny:
        return KrylovPath(beta_path, fitted_path, residual_norm, 0, 0, 0.0, False)

    Q = np.zeros((T, m_max))
    Q[:, 0] = r / norm_r
    dimension = 1
    norm_K = max(float(np.linalg.norm(K, ord=np.inf)), np.finfo(float).tiny)
    while dimension < m_max:
        candidate = K @ Q[:, dimension - 1]
        for _ in range(2):
            candidate -= Q[:, :dimension] @ (Q[:, :dimension].T @ candidate)
        candidate_norm = float(np.linalg.norm(candidate))
        if candidate_norm <= krylov_tolerance * norm_K:
            break
        Q[:, dimension] = candidate / candidate_norm
        dimension += 1

    Q = Q[:, :dimension]
    orthogonality_error = float(
        np.linalg.norm(Q.T @ Q - np.eye(dimension), ord=np.inf)
    )
    KQ = K @ Q
    Q_moment, R_moment = np.linalg.qr(KQ, mode="reduced")
    projected_r = Q_moment.T @ r
    diagonal = np.abs(np.diag(R_moment))
    reference = max(float(diagonal[0]), np.finfo(float).tiny)
    active = 0
    for component in range(dimension):
        d = component + 1
        if diagonal[component] <= rcond * reference:
            break
        coefficients = solve_triangular(
            R_moment[:d, :d],
            projected_r[:d],
            lower=False,
            check_finite=False,
        )
        beta = Q[:, :d] @ coefficients
        fitted = KQ[:, :d] @ coefficients
        if not np.all(np.isfinite(beta)) or not np.all(np.isfinite(fitted)):
            break
        beta_path[:, component] = beta
        fitted_path[:, component] = fitted
        residual_norm[component] = l2_norm(r - fitted)
        active = d

    breakdown = active < m_max
    if active == 0:
        return KrylovPath(
            beta_path,
            fitted_path,
            residual_norm,
            dimension,
            0,
            orthogonality_error,
            True,
        )
    beta_path[:, active:] = beta_path[:, [active - 1]]
    fitted_path[:, active:] = fitted_path[:, [active - 1]]
    residual_norm[active:] = residual_norm[active - 1]
    # Small roundoff wiggles are not scientifically meaningful.  The exact
    # nested minimum-residual path is monotone.
    residual_norm = np.minimum.accumulate(residual_norm)
    return KrylovPath(
        beta_path,
        fitted_path,
        residual_norm,
        dimension,
        active,
        orthogonality_error,
        breakdown,
    )


def julia_cg_fpls_path(
    r: np.ndarray,
    K: np.ndarray,
    m_max: int,
    breakdown_tolerance: float = 1e-14,
) -> KrylovPath:
    """Literal, guarded translation of the recurrence in simulations.ipynb."""

    r = np.asarray(r, dtype=float).reshape(-1)
    K = np.asarray(K, dtype=float)
    T = r.size
    if K.shape != (T, T) or m_max < 1:
        raise ValueError("K has the wrong shape or m_max is not positive")
    K = 0.5 * (K + K.T)
    beta_path = np.zeros((T, m_max))
    fitted_path = np.zeros((T, m_max))
    residual_norm = np.full(m_max, l2_norm(r))
    beta = np.zeros(T)
    residual = r.copy()
    direction = r.copy()
    active = 0
    breakdown = False
    for component in range(m_max):
        K_direction = K @ direction
        K_residual = K @ residual
        numerator = float(residual @ K_residual)
        denominator = float(K_direction @ K_direction)
        scale = max(
            abs(numerator),
            np.linalg.norm(residual) * np.linalg.norm(K_residual),
            np.finfo(float).tiny,
        )
        if (
            not np.isfinite(numerator)
            or not np.isfinite(denominator)
            or numerator <= breakdown_tolerance * scale
            or denominator <= np.finfo(float).tiny
        ):
            breakdown = True
            break
        alpha = numerator / denominator
        beta_new = beta + alpha * direction
        residual_new = residual - alpha * K_direction
        K_residual_new = K @ residual_new
        gamma = float(residual_new @ K_residual_new) / numerator
        if not (
            np.all(np.isfinite(beta_new))
            and np.all(np.isfinite(residual_new))
            and np.isfinite(gamma)
        ):
            breakdown = True
            break
        beta = beta_new
        residual = residual_new
        direction = residual + gamma * direction
        beta_path[:, component] = beta
        fitted_path[:, component] = r - residual
        residual_norm[component] = l2_norm(residual)
        active = component + 1
    if active > 0:
        beta_path[:, active:] = beta_path[:, [active - 1]]
        fitted_path[:, active:] = fitted_path[:, [active - 1]]
        residual_norm[active:] = residual_norm[active - 1]
    return KrylovPath(
        beta_path,
        fitted_path,
        residual_norm,
        active,
        active,
        float("nan"),
        breakdown,
    )


def cg_fpls_path(
    r: np.ndarray,
    K: np.ndarray,
    m_max: int,
    engine: str = "stable",
) -> KrylovPath:
    """Dispatch to the stabilized or literal Julia CG realization."""

    if engine == "stable":
        return stable_cg_fpls_path(r, K, m_max)
    if engine == "julia":
        return julia_cg_fpls_path(r, K, m_max)
    raise ValueError("engine must be 'stable' or 'julia'")


def maximum_rank_fitted_moment(
    r: np.ndarray, K: np.ndarray, rcond: float = 1e-12
) -> Tuple[np.ndarray, int]:
    """Project r onto the numerical range of K."""

    eigenvalues, eigenvectors = eigh(0.5 * (K + K.T), check_finite=False)
    largest = max(float(np.max(eigenvalues)), np.finfo(float).tiny)
    keep = eigenvalues > rcond * largest
    if not np.any(keep):
        return np.zeros_like(r), 0
    vectors = eigenvectors[:, keep]
    return vectors @ (vectors.T @ r), int(np.sum(keep))


def fpcr_gcv_pilot(
    X: np.ndarray,
    y: np.ndarray,
    r: np.ndarray,
    K: np.ndarray,
    m_max: int = 20,
) -> Tuple[np.ndarray, int, np.ndarray]:
    """Moment-GCV FPCR pilot used only to estimate regression residuals."""

    n, T = X.shape
    eigenvalues, eigenvectors = eigh(0.5 * (K + K.T), check_finite=False)
    largest = max(float(np.max(eigenvalues)), np.finfo(float).tiny)
    positive = np.flatnonzero(eigenvalues > np.finfo(float).eps * T * largest)[::-1]
    maximum = min(m_max, positive.size, n - 1, T - 1)
    if maximum < 1:
        beta = np.zeros(T)
        return beta, 0, y.copy()
    scores = np.full(maximum, np.inf)
    beta_path = np.zeros((T, maximum))
    for number in range(1, maximum + 1):
        indices = positive[:number]
        vectors = eigenvectors[:, indices]
        values = eigenvalues[indices]
        beta = vectors @ ((vectors.T @ r) / values)
        beta_path[:, number - 1] = beta
        moment_residual = r - K @ beta
        scores[number - 1] = np.mean(moment_residual**2) / (1.0 - number / T) ** 2
    selected = int(np.argmin(scores))
    beta = beta_path[:, selected]
    residuals = y - X @ beta / T
    return beta, selected + 1, residuals


def early_stopping_choice(
    path: KrylovPath,
    X: np.ndarray,
    y: np.ndarray,
    r: np.ndarray,
    K: np.ndarray,
    tau: float = 1.01,
    delta: float = 0.1,
    pilot_components: int = 20,
) -> Tuple[int, float, float, int, np.ndarray]:
    """Apply Babii et al.'s adaptive estimation-oriented discrepancy rule."""

    n, T = X.shape
    _, pilot_m, pilot_residuals = fpcr_gcv_pilot(
        X, y, r, K, m_max=pilot_components
    )
    sigma_squared = max(float(np.mean(pilot_residuals**2)), 0.0)
    x_norm = float(np.mean(np.sum(X**2, axis=1) / T))
    threshold = tau * math.sqrt(2.0 * sigma_squared * x_norm / (delta * n))
    candidates = np.flatnonzero(path.residual_norm <= threshold)
    selected = int(candidates[0] + 1) if candidates.size else path.residual_norm.size
    return selected, threshold, sigma_squared, pilot_m, pilot_residuals


def weighted_chi_square_samples(
    weights: np.ndarray,
    draws: int,
    rng: np.random.Generator,
    chunk_size: int = 5000,
) -> np.ndarray:
    """Simulate sum_j weights_j Z_j^2 without a large permanent matrix."""

    weights = np.asarray(weights, dtype=float).reshape(-1)
    weights = weights[np.isfinite(weights) & (weights > 0)]
    if draws < 1 or weights.size == 0:
        raise ValueError("draws and the number of positive weights must be positive")
    result = np.empty(draws)
    for start in range(0, draws, chunk_size):
        stop = min(draws, start + chunk_size)
        result[start:stop] = rng.chisquare(
            1.0, size=(stop - start, weights.size)
        ) @ weights
    return result


def calibration_quantiles(samples: np.ndarray, alphas: Sequence[float]) -> np.ndarray:
    alphas_array = np.asarray(alphas, dtype=float)
    if np.any((alphas_array <= 0) | (alphas_array >= 1)):
        raise ValueError("all significance levels must lie in (0,1)")
    return np.quantile(np.asarray(samples, dtype=float), 1.0 - alphas_array)


def plugin_variance_weights(X: np.ndarray, residuals: np.ndarray) -> np.ndarray:
    """Eigenvalues of V_hat=n^-1 sum residual_i^2 X_i tensor X_i."""

    n, T = X.shape
    residuals = np.asarray(residuals, dtype=float).reshape(-1)
    if residuals.size != n:
        raise ValueError("residuals and X have incompatible shapes")
    weighted_design = residuals[:, None] * X
    singular_values = np.linalg.svd(weighted_design, compute_uv=False)
    weights = singular_values**2 / (n * T)
    cutoff = np.finfo(float).eps * max(n, T) * max(float(weights[0]), 1.0)
    return weights[weights > cutoff]


def multiplier_limit_samples(
    X: np.ndarray,
    residuals: np.ndarray,
    draws: int,
    rng: np.random.Generator,
    chunk_size: int = 1000,
) -> np.ndarray:
    """Centered Gaussian-multiplier approximation to the null moment law."""

    n, T = X.shape
    scores = np.asarray(residuals, dtype=float).reshape(-1, 1) * X
    if scores.shape != (n, T):
        raise ValueError("residuals and X have incompatible shapes")
    scores -= np.mean(scores, axis=0, keepdims=True)
    result = np.empty(draws)
    scale = math.sqrt(n)
    for start in range(0, draws, chunk_size):
        stop = min(draws, start + chunk_size)
        multipliers = rng.normal(size=(stop - start, n))
        bootstrap_moments = multipliers @ scores / scale
        result[start:stop] = np.mean(bootstrap_moments**2, axis=1)
    return result


def _calibration_samples_for_dataset(
    method: str,
    X: np.ndarray,
    pilot_residuals: np.ndarray,
    draws: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if method == "plugin":
        weights = plugin_variance_weights(X, pilot_residuals)
        return weighted_chi_square_samples(weights, draws, rng)
    if method == "multiplier":
        return multiplier_limit_samples(X, pilot_residuals, draws, rng)
    raise ValueError("dataset calibration must be 'plugin' or 'multiplier'")


def _single_test_statistic(
    X: np.ndarray,
    y: np.ndarray,
    null_beta: np.ndarray,
    late_components: int,
    cg_engine: str,
    use_direct: bool = False,
) -> Tuple[float, float, float, KrylovPath, np.ndarray, np.ndarray]:
    r, K = empirical_moments(X, y)
    direct = direct_moment_statistic(r, K, null_beta, X.shape[0])
    if use_direct:
        T = r.size
        empty = np.zeros((T, 1))
        path = KrylovPath(empty, r[:, None], np.zeros(1), 0, 0, 0.0, False)
        return direct, direct, 0.0, path, r, K
    path = cg_fpls_path(r, K, late_components, engine=cg_engine)
    fitted = path.fitted_moment[:, late_components - 1]
    statistic = test_statistic(fitted, K, null_beta, X.shape[0])
    scaled_residual = math.sqrt(X.shape[0]) * l2_norm(r - fitted)
    return statistic, direct, scaled_residual, path, r, K


def _oracle_calibrations(
    design: Design,
    draws: int,
    seed: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[float, float]]]:
    samples: Dict[str, np.ndarray] = {}
    quantiles: Dict[str, Dict[float, float]] = {}
    # Store a dense grid of commonly used levels; requested levels can also be
    # computed directly from the retained samples.
    common_alphas = (0.01, 0.05, 0.10)
    for model_index, model in enumerate(design.models):
        rng = np.random.default_rng(np.random.SeedSequence([seed, 11, model_index]))
        values = weighted_chi_square_samples(
            design.noise_sd**2 * model.eigenvalues, draws, rng
        )
        samples[model.name] = values
        qs = calibration_quantiles(values, common_alphas)
        quantiles[model.name] = dict(zip(common_alphas, map(float, qs)))
    return samples, quantiles


def run_null_size_study(
    design: Design,
    replications: int,
    n: int,
    late_components: int,
    alphas: Sequence[float],
    calibrations: Sequence[str],
    calibration_draws: int,
    oracle_samples: Mapping[str, np.ndarray],
    seed: int,
    cg_engine: str,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], Dict[str, np.ndarray]]:
    """Null law, empirical size, and calibration comparison."""

    if replications < 1:
        raise ValueError("replications must be positive")
    alphas = tuple(float(value) for value in alphas)
    calibrations = tuple(dict.fromkeys(calibrations))
    unknown = set(calibrations) - {"oracle", "plugin", "multiplier"}
    if unknown:
        raise ValueError(f"unknown calibrations: {sorted(unknown)}")
    size_rows: List[Dict[str, object]] = []
    distribution_rows: List[Dict[str, object]] = []
    raw: Dict[str, np.ndarray] = {}

    for model_index, model in enumerate(design.models):
        print(f"  Null/size: {model.name} ({replications} replications)")
        data_rng = np.random.default_rng(
            np.random.SeedSequence([seed, 101, model_index, n])
        )
        calibration_rngs = {
            method: np.random.default_rng(
                np.random.SeedSequence([seed, 102, model_index, n, index])
            )
            for index, method in enumerate(calibrations)
            if method != "oracle"
        }
        statistics = np.empty(replications)
        direct_statistics = np.empty(replications)
        scaled_residuals = np.empty(replications)
        effective_dimensions = np.empty(replications, dtype=int)
        numerical_ranks = np.empty(replications, dtype=int)
        critical_values = {
            method: np.empty((replications, len(alphas))) for method in calibrations
        }
        oracle_q = calibration_quantiles(oracle_samples[model.name], alphas)

        for replication in range(replications):
            X, y, _ = simulate_sample(model, design, n, data_rng)
            statistic, direct, scaled, path, r, K = _single_test_statistic(
                X, y, model.beta, late_components, cg_engine
            )
            statistics[replication] = statistic
            direct_statistics[replication] = direct
            scaled_residuals[replication] = scaled
            effective_dimensions[replication] = path.effective_dimension
            numerical_ranks[replication] = path.numerical_rank
            if "oracle" in calibrations:
                critical_values["oracle"][replication] = oracle_q
            if any(method in calibrations for method in ("plugin", "multiplier")):
                # For the simple null H0: beta=b, these are observable null
                # residuals.  They estimate epsilon under H0 directly and are
                # preferable to response-fitted pilot residuals for test-size
                # calibration.
                null_residuals = y - X @ model.beta / X.shape[1]
                for method in ("plugin", "multiplier"):
                    if method not in calibrations:
                        continue
                    samples = _calibration_samples_for_dataset(
                        method,
                        X,
                        null_residuals,
                        calibration_draws,
                        calibration_rngs[method],
                    )
                    critical_values[method][replication] = calibration_quantiles(
                        samples, alphas
                    )

        key = _safe_key(model.name)
        raw[f"null_{key}_late_statistic"] = statistics
        raw[f"null_{key}_direct_statistic"] = direct_statistics
        raw[f"null_{key}_sqrt_n_residual"] = scaled_residuals
        raw[f"null_{key}_effective_dimension"] = effective_dimensions
        raw[f"null_{key}_numerical_rank"] = numerical_ranks
        raw[f"oracle_{key}_limit_draws"] = np.asarray(oracle_samples[model.name])
        for method, values in critical_values.items():
            raw[f"null_{key}_{method}_critical_values"] = values

        for method in calibrations:
            for alpha_index, alpha in enumerate(alphas):
                critical = critical_values[method][:, alpha_index]
                rejection = statistics > critical
                rate = float(np.mean(rejection))
                size_rows.append(
                    {
                        "model": model.name,
                        "n": n,
                        "calibration": method,
                        "residual_source": "known-DGP weights"
                        if method == "oracle"
                        else "null residuals",
                        "alpha": alpha,
                        "replications": replications,
                        "rejection_rate": rate,
                        "mcse": math.sqrt(rate * (1.0 - rate) / replications),
                        "mean_critical_value": float(np.mean(critical)),
                        "median_critical_value": float(np.median(critical)),
                        "mean_T_late": float(np.mean(statistics)),
                        "variance_T_late": float(np.var(statistics, ddof=1))
                        if replications > 1
                        else 0.0,
                        "mean_T_direct": float(np.mean(direct_statistics)),
                        "max_abs_T_difference": float(
                            np.max(np.abs(statistics - direct_statistics))
                        ),
                        "median_sqrt_n_residual": float(np.median(scaled_residuals)),
                        "late_components": late_components,
                    }
                )

        limit = np.asarray(oracle_samples[model.name])
        empirical_quantiles = np.quantile(statistics, [0.50, 0.90, 0.95, 0.99])
        limit_quantiles = np.quantile(limit, [0.50, 0.90, 0.95, 0.99])
        distribution_rows.append(
            {
                "model": model.name,
                "n": n,
                "replications": replications,
                "empirical_mean": float(np.mean(statistics)),
                "limit_mean": float(np.mean(limit)),
                "empirical_variance": float(np.var(statistics, ddof=1))
                if replications > 1
                else 0.0,
                "limit_variance": float(np.var(limit, ddof=1)),
                **{
                    f"empirical_q{int(probability * 100)}": float(value)
                    for probability, value in zip(
                        (0.50, 0.90, 0.95, 0.99), empirical_quantiles
                    )
                },
                **{
                    f"limit_q{int(probability * 100)}": float(value)
                    for probability, value in zip(
                        (0.50, 0.90, 0.95, 0.99), limit_quantiles
                    )
                },
            }
        )
    return size_rows, distribution_rows, raw


def _simulate_null_statistics(
    model: ModelSpec,
    design: Design,
    n: int,
    replications: int,
    late_components: int,
    seed_sequence: np.random.SeedSequence,
    cg_engine: str,
    use_direct: bool,
) -> np.ndarray:
    rng = np.random.default_rng(seed_sequence)
    values = np.empty(replications)
    basis_gram = design.basis @ design.basis.T / design.T if use_direct else None
    for replication in range(replications):
        if use_direct:
            values[replication] = simulate_direct_moment_statistic(
                model,
                design,
                n,
                rng,
                beta_data=model.beta,
                null_beta=model.beta,
                basis_gram=basis_gram,
            )
        else:
            X, y, _ = simulate_sample(model, design, n, rng)
            values[replication] = _single_test_statistic(
                X,
                y,
                model.beta,
                late_components,
                cg_engine,
            )[0]
    return values


def run_power_study(
    design: Design,
    replications: int,
    sample_sizes: Sequence[int],
    deltas: Sequence[float],
    late_components: int,
    alpha: float,
    calibration: str,
    oracle_samples: Mapping[str, np.ndarray],
    seed: int,
    cg_engine: str,
    use_direct: bool,
    reusable_null: Optional[Mapping[Tuple[str, int], np.ndarray]] = None,
) -> Tuple[List[Dict[str, object]], Dict[str, np.ndarray]]:
    """Power under beta_delta(s)=beta(s)+delta*s."""

    if calibration not in {"empirical", "oracle"}:
        raise ValueError("power calibration must be 'empirical' or 'oracle'")
    deltas = np.asarray(deltas, dtype=float)
    rows: List[Dict[str, object]] = []
    raw: Dict[str, np.ndarray] = {}
    reusable_null = {} if reusable_null is None else reusable_null

    for model_index, model in enumerate(design.models):
        key = _safe_key(model.name)
        for n in sample_sizes:
            print(
                f"  Power: {model.name}, n={n}, {deltas.size} delta values "
                f"({replications} replications each)"
            )
            if calibration == "oracle":
                critical = float(
                    np.quantile(oracle_samples[model.name], 1.0 - alpha)
                )
                null_statistics = np.empty(0)
            else:
                candidate = reusable_null.get((model.name, int(n)))
                if candidate is not None and candidate.size >= replications:
                    null_statistics = np.asarray(candidate[:replications])
                else:
                    null_statistics = _simulate_null_statistics(
                        model,
                        design,
                        int(n),
                        replications,
                        late_components,
                        np.random.SeedSequence([seed, 201, model_index, int(n)]),
                        cg_engine,
                        use_direct,
                    )
                critical = float(np.quantile(null_statistics, 1.0 - alpha))
            raw[f"power_{key}_n{n}_null_statistic"] = null_statistics
            statistics_by_delta = np.empty((deltas.size, replications))
            data_rng = np.random.default_rng(
                np.random.SeedSequence([seed, 202, model_index, int(n)])
            )
            basis_gram = (
                design.basis @ design.basis.T / design.T if use_direct else None
            )
            for delta_index, delta_value in enumerate(deltas):
                beta_data = model.beta + float(delta_value) * design.s
                for replication in range(replications):
                    if use_direct:
                        statistic = simulate_direct_moment_statistic(
                            model,
                            design,
                            int(n),
                            data_rng,
                            beta_data=beta_data,
                            null_beta=model.beta,
                            basis_gram=basis_gram,
                        )
                    else:
                        X, y, _ = simulate_sample(
                            model, design, int(n), data_rng, beta_data=beta_data
                        )
                        statistic = _single_test_statistic(
                            X,
                            y,
                            model.beta,
                            late_components,
                            cg_engine,
                        )[0]
                    statistics_by_delta[delta_index, replication] = statistic
                rejection = statistics_by_delta[delta_index] > critical
                power = float(np.mean(rejection))
                rows.append(
                    {
                        "model": model.name,
                        "n": int(n),
                        "delta": float(delta_value),
                        "replications": replications,
                        "alpha": alpha,
                        "calibration": calibration,
                        "test_engine": "direct-moment" if use_direct else "late-cg",
                        "critical_value": critical,
                        "power": power,
                        "mcse": math.sqrt(power * (1.0 - power) / replications),
                    }
                )
            raw[f"power_{key}_n{n}_deltas"] = deltas
            raw[f"power_{key}_n{n}_statistics"] = statistics_by_delta
    return rows, raw


def confidence_ellipsoid_envelope(
    fitted_moment: np.ndarray,
    K: np.ndarray,
    basis_functions: np.ndarray,
    critical_value: float,
    n: int,
    rcond: float = 1e-12,
) -> ConfidenceEnvelope:
    """Analytic envelope of a finite-dimensional inverted-test ellipsoid.

    Candidate functions have the form ``basis_functions @ coefficients``.
    The accepted coefficients satisfy

        n/T ||K beta_hat - K H c||_2^2 <= critical_value.
    """

    fitted_moment = np.asarray(fitted_moment, dtype=float).reshape(-1)
    K = np.asarray(K, dtype=float)
    H = np.asarray(basis_functions, dtype=float)
    T = fitted_moment.size
    if K.shape != (T, T) or H.ndim != 2 or H.shape[0] != T:
        raise ValueError("fitted_moment, K, and basis_functions are incompatible")
    if critical_value <= 0 or n < 2:
        raise ValueError("critical_value and n must be positive")
    p = H.shape[1]
    A = K @ H
    center, _, rank, _ = np.linalg.lstsq(A, fitted_moment, rcond=rcond)
    residual = fitted_moment - A @ center
    residual_squared = float(residual @ residual)
    minimum_statistic = float(n * residual_squared / T)
    radius_squared = float(critical_value * T / n - residual_squared)
    empty = radius_squared < -1e-10 * max(critical_value * T / n, 1.0)
    radius_squared = max(radius_squared, 0.0)
    center_curve = H @ center
    nan_curve = np.full(T, np.nan)
    if empty:
        return ConfidenceEnvelope(
            nan_curve.copy(),
            nan_curve.copy(),
            center_curve,
            center,
            radius_squared,
            minimum_statistic,
            int(rank),
            True,
            False,
        )
    if int(rank) < p:
        return ConfidenceEnvelope(
            np.full(T, -np.inf),
            np.full(T, np.inf),
            center_curve,
            center,
            radius_squared,
            minimum_statistic,
            int(rank),
            False,
            True,
        )
    gram = A.T @ A
    gram_inverse = np.linalg.inv(gram)
    directional_variance = np.einsum("ij,jk,ik->i", H, gram_inverse, H)
    directional_variance = np.maximum(directional_variance, 0.0)
    half_width = np.sqrt(radius_squared * directional_variance)
    return ConfidenceEnvelope(
        center_curve - half_width,
        center_curve + half_width,
        center_curve,
        center,
        radius_squared,
        minimum_statistic,
        int(rank),
        False,
        False,
    )


def paper_grid_confidence_envelope(
    fitted_moment: np.ndarray,
    K: np.ndarray,
    basis_functions: np.ndarray,
    critical_value: float,
    n: int,
    grid_points: int = 20,
    coefficient_min: float = 0.0,
    coefficient_max: float = 4.5,
    batch_size: int = 10000,
) -> ConfidenceEnvelope:
    """Memory-batched reproduction of the notebook's coefficient-grid search."""

    H = np.asarray(basis_functions, dtype=float)
    T, p = H.shape
    if p > 8:
        raise ValueError("paper-grid mode is intended only for a small basis dimension")
    if grid_points < 2 or batch_size < 1:
        raise ValueError("grid_points and batch_size must be positive")
    values = np.linspace(coefficient_min, coefficient_max, grid_points)
    total = int(grid_points**p)
    A = np.asarray(K, dtype=float) @ H
    fitted_moment = np.asarray(fitted_moment, dtype=float).reshape(-1)
    lower = np.full(T, np.inf)
    upper = np.full(T, -np.inf)
    accepted = 0
    best_statistic = np.inf
    best_coefficients = np.zeros(p)
    for start in range(0, total, batch_size):
        stop = min(total, start + batch_size)
        flat = np.arange(start, stop, dtype=np.int64)
        digits = np.empty((stop - start, p), dtype=np.int64)
        work = flat.copy()
        for column in range(p - 1, -1, -1):
            digits[:, column] = work % grid_points
            work //= grid_points
        coefficients = values[digits].T
        differences = fitted_moment[:, None] - A @ coefficients
        statistics = n * np.mean(differences**2, axis=0)
        local_best = int(np.argmin(statistics))
        if statistics[local_best] < best_statistic:
            best_statistic = float(statistics[local_best])
            best_coefficients = coefficients[:, local_best].copy()
        keep = statistics <= critical_value
        if np.any(keep):
            curves = H @ coefficients[:, keep]
            lower = np.minimum(lower, np.min(curves, axis=1))
            upper = np.maximum(upper, np.max(curves, axis=1))
            accepted += int(np.sum(keep))
    empty = accepted == 0
    if empty:
        lower[:] = np.nan
        upper[:] = np.nan
    center_curve = H @ best_coefficients
    return ConfidenceEnvelope(
        lower,
        upper,
        center_curve,
        best_coefficients,
        float("nan"),
        best_statistic,
        p,
        empty,
        False,
    )


def run_confidence_study(
    design: Design,
    replications: int,
    n: int,
    late_components: int,
    alpha: float,
    calibration: str,
    calibration_draws: int,
    oracle_samples: Mapping[str, np.ndarray],
    basis_dimension: int,
    method: str,
    grid_points: int,
    seed: int,
    cg_engine: str,
) -> Tuple[List[Dict[str, object]], Dict[str, np.ndarray]]:
    """Invert the test and summarize global and truncated-set coverage."""

    if calibration not in {"oracle", "plugin", "multiplier"}:
        raise ValueError("unknown confidence calibration")
    if method not in {"analytic", "paper-grid"}:
        raise ValueError("confidence method must be analytic or paper-grid")
    if not 1 <= basis_dimension <= design.J:
        raise ValueError("basis_dimension is out of range")
    rows: List[Dict[str, object]] = []
    raw: Dict[str, np.ndarray] = {}
    H = design.basis[:basis_dimension].T

    for model_index, model in enumerate(design.models):
        print(f"  Confidence set: {model.name} ({replications} replications)")
        # Use the same Monte Carlo stream as the null-size experiment.  When
        # replication counts and calibration match, global test-inversion
        # coverage is then visibly equal to one minus empirical size.
        data_rng = np.random.default_rng(
            np.random.SeedSequence([seed, 101, model_index, n])
        )
        calibration_rng = np.random.default_rng(
            np.random.SeedSequence([seed, 302, model_index, n])
        )
        oracle_critical = float(
            np.quantile(oracle_samples[model.name], 1.0 - alpha)
        )
        lowers = np.full((replications, design.T), np.nan)
        uppers = np.full((replications, design.T), np.nan)
        centers = np.full((replications, design.T), np.nan)
        critical_values = np.empty(replications)
        global_coverage = np.zeros(replications, dtype=bool)
        projected_coverage = np.zeros(replications, dtype=bool)
        empty = np.zeros(replications, dtype=bool)
        unbounded = np.zeros(replications, dtype=bool)
        ranks = np.zeros(replications, dtype=int)
        minimum_statistics = np.empty(replications)
        mean_widths = np.full(replications, np.nan)
        projected_beta = H @ model.coefficients[:basis_dimension]

        for replication in range(replications):
            X, y, _ = simulate_sample(model, design, n, data_rng)
            statistic, _, _, path, r, K = _single_test_statistic(
                X, y, model.beta, late_components, cg_engine
            )
            fitted = path.fitted_moment[:, late_components - 1]
            if calibration == "oracle":
                critical = oracle_critical
            else:
                _, _, pilot_residuals = fpcr_gcv_pilot(
                    X, y, r, K, m_max=min(20, late_components)
                )
                samples = _calibration_samples_for_dataset(
                    calibration,
                    X,
                    pilot_residuals,
                    calibration_draws,
                    calibration_rng,
                )
                critical = float(np.quantile(samples, 1.0 - alpha))
            critical_values[replication] = critical
            global_coverage[replication] = statistic <= critical
            projected_statistic = test_statistic(fitted, K, projected_beta, n)
            projected_coverage[replication] = projected_statistic <= critical
            if method == "analytic":
                envelope = confidence_ellipsoid_envelope(
                    fitted, K, H, critical, n
                )
            else:
                envelope = paper_grid_confidence_envelope(
                    fitted,
                    K,
                    H,
                    critical,
                    n,
                    grid_points=grid_points,
                )
            lowers[replication] = envelope.lower
            uppers[replication] = envelope.upper
            centers[replication] = envelope.center_curve
            empty[replication] = envelope.empty
            unbounded[replication] = envelope.unbounded
            ranks[replication] = envelope.rank
            minimum_statistics[replication] = envelope.minimum_statistic
            if not envelope.empty and not envelope.unbounded:
                mean_widths[replication] = float(np.mean(envelope.upper - envelope.lower))

        finite_rows = np.isfinite(mean_widths)
        if np.any(finite_rows):
            median_lower = np.nanmedian(lowers[finite_rows], axis=0)
            median_upper = np.nanmedian(uppers[finite_rows], axis=0)
            target = float(np.nanmedian(mean_widths[finite_rows]))
            candidates = np.flatnonzero(finite_rows)
            representative = int(
                candidates[np.argmin(np.abs(mean_widths[candidates] - target))]
            )
        else:
            median_lower = np.full(design.T, np.nan)
            median_upper = np.full(design.T, np.nan)
            representative = 0
        key = _safe_key(model.name)
        raw[f"confidence_{key}_lower"] = lowers
        raw[f"confidence_{key}_upper"] = uppers
        raw[f"confidence_{key}_center"] = centers
        raw[f"confidence_{key}_critical_value"] = critical_values
        raw[f"confidence_{key}_global_coverage"] = global_coverage
        raw[f"confidence_{key}_projected_coverage"] = projected_coverage
        raw[f"confidence_{key}_empty"] = empty
        raw[f"confidence_{key}_unbounded"] = unbounded
        raw[f"confidence_{key}_rank"] = ranks
        raw[f"confidence_{key}_mean_width"] = mean_widths
        raw[f"confidence_{key}_median_lower"] = median_lower
        raw[f"confidence_{key}_median_upper"] = median_upper
        raw[f"confidence_{key}_representative_lower"] = lowers[representative]
        raw[f"confidence_{key}_representative_upper"] = uppers[representative]
        global_rate = float(np.mean(global_coverage))
        projected_rate = float(np.mean(projected_coverage))
        rows.append(
            {
                "model": model.name,
                "n": n,
                "replications": replications,
                "confidence_level": 1.0 - alpha,
                "calibration": calibration,
                "method": method,
                "basis_dimension": basis_dimension,
                "global_coverage": global_rate,
                "global_coverage_mcse": math.sqrt(
                    global_rate * (1.0 - global_rate) / replications
                ),
                "projected_coverage": projected_rate,
                "projected_coverage_mcse": math.sqrt(
                    projected_rate * (1.0 - projected_rate) / replications
                ),
                "mean_set_width": float(np.nanmean(mean_widths))
                if np.any(finite_rows)
                else float("nan"),
                "median_set_width": float(np.nanmedian(mean_widths))
                if np.any(finite_rows)
                else float("nan"),
                "empty_frequency": float(np.mean(empty)),
                "unbounded_frequency": float(np.mean(unbounded)),
                "median_rank": float(np.median(ranks)),
                "median_minimum_statistic": float(np.median(minimum_statistics)),
            }
        )
    return rows, raw


def run_stopping_study(
    design: Design,
    replications: int,
    n: int,
    fixed_components: Sequence[int],
    alpha: float,
    oracle_samples: Mapping[str, np.ndarray],
    seed: int,
    cg_engine: str,
    tau: float = 1.01,
    delta: float = 0.1,
) -> Tuple[List[Dict[str, object]], Dict[str, np.ndarray]]:
    """Compare adaptive early stopping, fixed m, and maximum-rank fitting."""

    fixed = np.unique(np.asarray(fixed_components, dtype=int))
    fixed = fixed[fixed > 0]
    if fixed.size == 0:
        raise ValueError("at least one positive fixed component count is required")
    maximum_component = int(np.max(fixed))
    labels = [f"m={value}" for value in fixed] + ["adaptive", "max-rank"]
    rows: List[Dict[str, object]] = []
    raw: Dict[str, np.ndarray] = {}

    for model_index, model in enumerate(design.models):
        print(f"  Stopping diagnostics: {model.name} ({replications} replications)")
        rng = np.random.default_rng(
            np.random.SeedSequence([seed, 401, model_index, n])
        )
        critical = float(np.quantile(oracle_samples[model.name], 1.0 - alpha))
        number_methods = len(labels)
        statistics = np.empty((replications, number_methods))
        residuals = np.empty((replications, number_methods))
        discrepancies = np.empty((replications, number_methods))
        selected_components = np.empty((replications, number_methods), dtype=int)
        thresholds = np.empty(replications)
        pilot_sigmas = np.empty(replications)
        numerical_ranks = np.empty(replications, dtype=int)

        for replication in range(replications):
            X, y, _ = simulate_sample(model, design, n, rng)
            r, K = empirical_moments(X, y)
            path = cg_fpls_path(r, K, maximum_component, engine=cg_engine)
            direct = direct_moment_statistic(r, K, model.beta, n)
            for method_index, component in enumerate(fixed):
                fitted = path.fitted_moment[:, component - 1]
                statistics[replication, method_index] = test_statistic(
                    fitted, K, model.beta, n
                )
                residuals[replication, method_index] = math.sqrt(n) * l2_norm(
                    r - fitted
                )
                discrepancies[replication, method_index] = abs(
                    statistics[replication, method_index] - direct
                )
                selected_components[replication, method_index] = component
            early_m, threshold, sigma_squared, _, _ = early_stopping_choice(
                path,
                X,
                y,
                r,
                K,
                tau=tau,
                delta=delta,
                pilot_components=min(20, maximum_component),
            )
            early_index = fixed.size
            early_fitted = path.fitted_moment[:, early_m - 1]
            statistics[replication, early_index] = test_statistic(
                early_fitted, K, model.beta, n
            )
            residuals[replication, early_index] = math.sqrt(n) * l2_norm(
                r - early_fitted
            )
            discrepancies[replication, early_index] = abs(
                statistics[replication, early_index] - direct
            )
            selected_components[replication, early_index] = early_m
            maximum_fitted, rank = maximum_rank_fitted_moment(r, K)
            maximum_index = fixed.size + 1
            statistics[replication, maximum_index] = test_statistic(
                maximum_fitted, K, model.beta, n
            )
            residuals[replication, maximum_index] = math.sqrt(n) * l2_norm(
                r - maximum_fitted
            )
            discrepancies[replication, maximum_index] = abs(
                statistics[replication, maximum_index] - direct
            )
            selected_components[replication, maximum_index] = rank
            thresholds[replication] = math.sqrt(n) * threshold
            pilot_sigmas[replication] = sigma_squared
            numerical_ranks[replication] = rank

        key = _safe_key(model.name)
        raw[f"stopping_{key}_statistics"] = statistics
        raw[f"stopping_{key}_sqrt_n_residual"] = residuals
        raw[f"stopping_{key}_abs_stat_difference"] = discrepancies
        raw[f"stopping_{key}_selected_components"] = selected_components
        raw[f"stopping_{key}_sqrt_n_threshold"] = thresholds
        raw[f"stopping_{key}_pilot_sigma_squared"] = pilot_sigmas
        raw[f"stopping_{key}_numerical_rank"] = numerical_ranks
        for method_index, label in enumerate(labels):
            rejection = statistics[:, method_index] > critical
            rate = float(np.mean(rejection))
            rows.append(
                {
                    "model": model.name,
                    "n": n,
                    "stopping_rule": label,
                    "component_value": int(fixed[method_index])
                    if method_index < fixed.size
                    else float(np.median(selected_components[:, method_index])),
                    "median_selected_components": float(
                        np.median(selected_components[:, method_index])
                    ),
                    "replications": replications,
                    "alpha": alpha,
                    "critical_value": critical,
                    "rejection_rate": rate,
                    "mcse": math.sqrt(rate * (1.0 - rate) / replications),
                    "mean_sqrt_n_residual": float(
                        np.mean(residuals[:, method_index])
                    ),
                    "median_sqrt_n_residual": float(
                        np.median(residuals[:, method_index])
                    ),
                    "mean_abs_stat_difference": float(
                        np.mean(discrepancies[:, method_index])
                    ),
                    "median_abs_stat_difference": float(
                        np.median(discrepancies[:, method_index])
                    ),
                    "median_sqrt_n_threshold": float(np.median(thresholds)),
                }
            )
    return rows, raw


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _save_figure(figure: plt.Figure, path: Path) -> None:
    figure.savefig(
        path,
        bbox_inches="tight",
        metadata={"Creator": "babii_inference_study.py"},
    )
    plt.close(figure)


def plot_null_calibration(
    design: Design, raw: Mapping[str, np.ndarray], output_path: Path
) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(12.5, 7.2), constrained_layout=True)
    probabilities = np.linspace(0.01, 0.99, 99)
    for column, model in enumerate(design.models):
        key = _safe_key(model.name)
        empirical = raw[f"null_{key}_late_statistic"]
        limit = raw[f"oracle_{key}_limit_draws"]
        upper = float(np.quantile(np.concatenate([empirical, limit]), 0.995))
        bins = np.linspace(0.0, upper, 42)
        axes[0, column].hist(
            empirical,
            bins=bins,
            density=True,
            alpha=0.55,
            color="#2F6B9A",
            label=r"Finite-sample $T_n$",
        )
        axes[0, column].hist(
            limit,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=1.7,
            color="#D9772B",
            label=r"Weighted-$\chi^2$ limit",
        )
        axes[0, column].set_title(model.name)
        axes[0, column].set_xlabel("Statistic")
        if column == 0:
            axes[0, column].set_ylabel("Density")
            axes[0, column].legend(loc="upper right", fontsize=8)
        empirical_q = np.quantile(empirical, probabilities)
        limit_q = np.quantile(limit, probabilities)
        axes[1, column].scatter(
            empirical_q, limit_q, s=12, color=MODEL_COLORS[model.name], alpha=0.8
        )
        lower = min(float(empirical_q[0]), float(limit_q[0]))
        upper_q = max(float(empirical_q[-1]), float(limit_q[-1]))
        axes[1, column].plot([lower, upper_q], [lower, upper_q], "--", color="0.35")
        axes[1, column].set_xlabel(r"Finite-sample $T_n$ quantiles")
        if column == 0:
            axes[1, column].set_ylabel("Limit quantiles")
        axes[1, column].set_aspect("equal", adjustable="box")
    figure.suptitle("Null calibration of the late-stopped CG-FPLS test", fontsize=14)
    _save_figure(figure, output_path)


def plot_calibration_comparison(
    design: Design,
    size_rows: Sequence[Mapping[str, object]],
    alpha: float,
    output_path: Path,
) -> None:
    selected = [row for row in size_rows if abs(float(row["alpha"]) - alpha) < 1e-12]
    calibrations = [
        method
        for method in ("oracle", "plugin", "multiplier")
        if any(row["calibration"] == method for row in selected)
    ]
    figure, axis = plt.subplots(figsize=(8.6, 4.8), constrained_layout=True)
    x = np.arange(len(design.models), dtype=float)
    offsets = np.linspace(-0.22, 0.22, max(len(calibrations), 1))
    for offset, method in zip(offsets, calibrations):
        values = []
        errors = []
        for model in design.models:
            row = next(
                item
                for item in selected
                if item["model"] == model.name and item["calibration"] == method
            )
            values.append(float(row["rejection_rate"]))
            errors.append(1.96 * float(row["mcse"]))
        axis.errorbar(
            x + offset,
            values,
            yerr=errors,
            marker="o",
            capsize=3,
            linewidth=1.5,
            color=CALIBRATION_COLORS[method],
            label=method.capitalize(),
        )
    axis.axhline(alpha, color="0.25", linestyle="--", linewidth=1.2, label="Nominal")
    axis.set_xticks(x, [model.name for model in design.models])
    axis.set_ylabel("Empirical rejection probability")
    axis.set_ylim(bottom=0.0)
    axis.set_title(f"Critical-value calibration comparison at alpha={alpha:.2f}")
    axis.legend(ncol=max(1, len(calibrations) + 1), loc="upper center")
    _save_figure(figure, output_path)


def plot_power_curves(
    design: Design,
    rows: Sequence[Mapping[str, object]],
    alpha: float,
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(12.8, 4.1), sharey=True, constrained_layout=True)
    sample_sizes = sorted({int(row["n"]) for row in rows})
    styles = ["-", "--", "-.", ":"]
    for axis, model in zip(axes, design.models):
        for style, n in zip(styles, sample_sizes):
            subset = sorted(
                (
                    row
                    for row in rows
                    if row["model"] == model.name and int(row["n"]) == n
                ),
                key=lambda item: float(item["delta"]),
            )
            axis.plot(
                [float(row["delta"]) for row in subset],
                [float(row["power"]) for row in subset],
                style,
                linewidth=2,
                label=f"n={n}",
            )
        axis.axhline(alpha, color="0.35", linestyle=":", linewidth=1.2, label="Nominal")
        axis.set_title(model.name)
        axis.set_xlabel(r"Alternative scale $\delta$")
        axis.set_ylim(0.0, 1.02)
        axis.legend(loc="upper center", fontsize=8)
    axes[0].set_ylabel("Empirical rejection probability")
    figure.suptitle(r"Power against $\beta_\delta(s)=\beta(s)+\delta s$", fontsize=14)
    _save_figure(figure, output_path)


def plot_stopping_diagnostics(
    design: Design,
    rows: Sequence[Mapping[str, object]],
    alpha: float,
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(3, 3, figsize=(12.8, 9.2), constrained_layout=True)
    metrics = (
        ("median_sqrt_n_residual", r"Median $\sqrt{n}\|q_m\|$", True),
        ("median_abs_stat_difference", r"Median $|T_{n,m}-S_n|$", True),
        ("rejection_rate", "Empirical rejection probability", False),
    )
    for column, model in enumerate(design.models):
        subset = [row for row in rows if row["model"] == model.name]
        fixed = sorted(
            (row for row in subset if str(row["stopping_rule"]).startswith("m=")),
            key=lambda item: float(item["component_value"]),
        )
        special = [row for row in subset if row not in fixed]
        for row_index, (metric, ylabel, log_scale) in enumerate(metrics):
            axis = axes[row_index, column]
            axis.plot(
                [float(row["component_value"]) for row in fixed],
                [float(row[metric]) for row in fixed],
                marker="o",
                color=MODEL_COLORS[model.name],
                linewidth=1.8,
                label="Fixed m",
            )
            markers = {"adaptive": "*", "max-rank": "D"}
            for row in special:
                label = str(row["stopping_rule"])
                axis.scatter(
                    float(row["component_value"]),
                    float(row[metric]),
                    marker=markers[label],
                    s=80 if label == "adaptive" else 45,
                    label=label.capitalize(),
                    zorder=4,
                )
            if row_index == 2:
                axis.axhline(alpha, color="0.35", linestyle="--", linewidth=1.1)
                axis.set_ylim(bottom=0.0)
            elif log_scale:
                positive = [float(row[metric]) for row in subset if float(row[metric]) > 0]
                if positive:
                    axis.set_yscale("log")
                if row_index == 0:
                    axis.axhline(
                        float(subset[0]["median_sqrt_n_threshold"]),
                        color="0.4",
                        linestyle=":",
                        linewidth=1.2,
                        label="Early threshold",
                    )
            axis.set_xscale("log")
            axis.set_xlabel("Components / numerical rank")
            if column == 0:
                axis.set_ylabel(ylabel)
            if row_index == 0:
                axis.set_title(model.name)
                axis.legend(fontsize=7, loc="best")
    figure.suptitle("Why inference uses late rather than adaptive early stopping", fontsize=14)
    _save_figure(figure, output_path)


def plot_confidence_sets(
    design: Design, raw: Mapping[str, np.ndarray], output_path: Path
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(12.8, 4.2), constrained_layout=True)
    for axis, model in zip(axes, design.models):
        key = _safe_key(model.name)
        lower = raw[f"confidence_{key}_median_lower"]
        upper = raw[f"confidence_{key}_median_upper"]
        representative_lower = raw[f"confidence_{key}_representative_lower"]
        representative_upper = raw[f"confidence_{key}_representative_upper"]
        axis.fill_between(
            design.s,
            lower,
            upper,
            color="#A9ADB4",
            alpha=0.45,
            label="Median 95% envelope",
        )
        axis.plot(
            design.s,
            representative_lower,
            color="#7B8794",
            linewidth=0.9,
            linestyle="--",
            alpha=0.8,
        )
        axis.plot(
            design.s,
            representative_upper,
            color="#7B8794",
            linewidth=0.9,
            linestyle="--",
            alpha=0.8,
            label="Representative envelope",
        )
        axis.plot(design.s, model.beta, color="black", linewidth=2.0, label="True beta")
        axis.set_title(model.name)
        axis.set_xlabel("s")
        if axis is axes[0]:
            axis.set_ylabel(r"$\beta(s)$")
            axis.legend(fontsize=8, loc="best")
    figure.suptitle("Finite-dimensional confidence sets obtained by test inversion", fontsize=14)
    _save_figure(figure, output_path)


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output-dir", type=Path, default=Path("babii_inference_output"))
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument(
        "--replications",
        type=int,
        default=100,
        help="common replication count unless an experiment-specific value is supplied",
    )
    parser.add_argument("--size-replications", type=int)
    parser.add_argument("--power-replications", type=int)
    parser.add_argument("--confidence-replications", type=int)
    parser.add_argument("--stopping-replications", type=int)
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--power-sample-sizes", type=int, nargs="+", default=[100, 200])
    parser.add_argument("--T", type=int, default=200)
    parser.add_argument("--J", type=int, default=100)
    parser.add_argument("--noise-sd", type=float, default=1.0)
    parser.add_argument("--late-components", type=int, default=70)
    parser.add_argument(
        "--stopping-components",
        type=int,
        nargs="+",
        default=[1, 2, 3, 5, 10, 20, 40, 70],
    )
    parser.add_argument("--alpha-levels", type=float, nargs="+", default=[0.01, 0.05, 0.10])
    parser.add_argument("--power-alpha", type=float, default=0.05)
    parser.add_argument("--delta-step", type=float, default=0.05)
    parser.add_argument("--oracle-draws", type=int, default=50000)
    parser.add_argument("--calibration-draws", type=int, default=2000)
    parser.add_argument(
        "--calibrations",
        nargs="+",
        choices=("oracle", "plugin", "multiplier"),
        default=["oracle", "plugin", "multiplier"],
    )
    parser.add_argument(
        "--power-calibration", choices=("empirical", "oracle"), default="empirical"
    )
    parser.add_argument(
        "--power-direct",
        action="store_true",
        help="use the direct-moment late-stopping limit for the expensive power loop",
    )
    parser.add_argument(
        "--confidence-calibration",
        choices=("oracle", "plugin", "multiplier"),
        default="oracle",
    )
    parser.add_argument(
        "--confidence-components",
        type=int,
        help=(
            "CG components for confidence-set inversion; defaults to --late-components. "
            "Use 10 to reproduce the hard-coded choice in the supplied notebook."
        ),
    )
    parser.add_argument("--confidence-basis-dimension", type=int, default=5)
    parser.add_argument(
        "--confidence-method", choices=("analytic", "paper-grid"), default="analytic"
    )
    parser.add_argument("--confidence-grid-points", type=int, default=20)
    parser.add_argument(
        "--cg-engine", choices=("stable", "julia"), default="stable"
    )
    parser.add_argument(
        "--basis-convention", choices=("babii", "standard"), default="babii"
    )
    parser.add_argument("--skip-size", action="store_true")
    parser.add_argument("--skip-power", action="store_true")
    parser.add_argument("--skip-confidence", action="store_true")
    parser.add_argument("--skip-stopping", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def _resolved_replications(arguments: argparse.Namespace, name: str) -> int:
    value = getattr(arguments, f"{name}_replications")
    return int(arguments.replications if value is None else value)


def _validate_arguments(arguments: argparse.Namespace) -> None:
    counts = [
        arguments.replications,
        *(
            value
            for value in (
                arguments.size_replications,
                arguments.power_replications,
                arguments.confidence_replications,
                arguments.stopping_replications,
            )
            if value is not None
        ),
    ]
    if any(value < 1 for value in counts):
        raise ValueError("all replication counts must be positive")
    if arguments.sample_size < 2 or any(value < 2 for value in arguments.power_sample_sizes):
        raise ValueError("sample sizes must be at least two")
    if arguments.late_components < 1:
        raise ValueError("late-components must be positive")
    if arguments.confidence_components is not None and arguments.confidence_components < 1:
        raise ValueError("confidence-components must be positive")
    if arguments.delta_step <= 0 or arguments.delta_step > 2:
        raise ValueError("delta-step must lie in (0,2]")
    if arguments.oracle_draws < 100 or arguments.calibration_draws < 100:
        raise ValueError("calibration draw counts must be at least 100")
    if any(not 0 < value < 1 for value in arguments.alpha_levels):
        raise ValueError("alpha-levels must lie in (0,1)")
    if not 0 < arguments.power_alpha < 1:
        raise ValueError("power-alpha must lie in (0,1)")
    if arguments.confidence_method == "paper-grid" and _resolved_replications(
        arguments, "confidence"
    ) > 10:
        print(
            "Warning: paper-grid confidence inversion is extremely expensive; "
            "analytic mode is recommended for more than ten replications.",
            file=sys.stderr,
        )


def run_study(arguments: argparse.Namespace) -> Dict[str, object]:
    """Run the requested experiments and write all reproducibility outputs."""

    _validate_arguments(arguments)
    output_dir = Path(arguments.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    design = build_design(
        T=arguments.T,
        J=arguments.J,
        noise_sd=arguments.noise_sd,
        basis_convention=arguments.basis_convention,
    )
    size_replications = _resolved_replications(arguments, "size")
    power_replications = _resolved_replications(arguments, "power")
    confidence_replications = _resolved_replications(arguments, "confidence")
    stopping_replications = _resolved_replications(arguments, "stopping")
    confidence_components = (
        arguments.late_components
        if arguments.confidence_components is None
        else int(arguments.confidence_components)
    )
    size_alphas = sorted(
        set(float(value) for value in arguments.alpha_levels)
        | {float(arguments.power_alpha)}
    )
    deltas = np.arange(
        -1.0, 1.0 + 0.5 * arguments.delta_step, arguments.delta_step, dtype=float
    )
    deltas[np.abs(deltas) < 1e-12] = 0.0

    print("Precomputing oracle weighted-chi-square calibrations...")
    oracle_samples, _ = _oracle_calibrations(
        design, arguments.oracle_draws, arguments.seed
    )
    size_rows: List[Dict[str, object]] = []
    distribution_rows: List[Dict[str, object]] = []
    power_rows: List[Dict[str, object]] = []
    confidence_rows: List[Dict[str, object]] = []
    stopping_rows: List[Dict[str, object]] = []
    raw: Dict[str, np.ndarray] = {}

    if not arguments.skip_size:
        print("Running null calibration and empirical-size study...")
        size_rows, distribution_rows, null_raw = run_null_size_study(
            design,
            size_replications,
            arguments.sample_size,
            arguments.late_components,
            size_alphas,
            arguments.calibrations,
            arguments.calibration_draws,
            oracle_samples,
            arguments.seed,
            arguments.cg_engine,
        )
        raw.update(null_raw)

    if not arguments.skip_power:
        print("Running power study...")
        reusable_null: Dict[Tuple[str, int], np.ndarray] = {}
        if not arguments.skip_size:
            for model in design.models:
                key = _safe_key(model.name)
                reusable_null[(model.name, arguments.sample_size)] = raw[
                    f"null_{key}_{'direct' if arguments.power_direct else 'late'}_statistic"
                ]
        power_rows, power_raw = run_power_study(
            design,
            power_replications,
            arguments.power_sample_sizes,
            deltas,
            arguments.late_components,
            arguments.power_alpha,
            arguments.power_calibration,
            oracle_samples,
            arguments.seed,
            arguments.cg_engine,
            arguments.power_direct,
            reusable_null,
        )
        raw.update(power_raw)

    if not arguments.skip_confidence:
        print("Running confidence-set study...")
        confidence_rows, confidence_raw = run_confidence_study(
            design,
            confidence_replications,
            arguments.sample_size,
            confidence_components,
            arguments.power_alpha,
            arguments.confidence_calibration,
            arguments.calibration_draws,
            oracle_samples,
            arguments.confidence_basis_dimension,
            arguments.confidence_method,
            arguments.confidence_grid_points,
            arguments.seed,
            arguments.cg_engine,
        )
        raw.update(confidence_raw)

    if not arguments.skip_stopping:
        print("Running stopping diagnostics...")
        stopping_components = sorted(
            set(arguments.stopping_components + [arguments.late_components])
        )
        stopping_rows, stopping_raw = run_stopping_study(
            design,
            stopping_replications,
            arguments.sample_size,
            stopping_components,
            arguments.power_alpha,
            oracle_samples,
            arguments.seed,
            arguments.cg_engine,
        )
        raw.update(stopping_raw)

    print("Writing tables and reproducibility data...")
    _write_csv(
        output_dir / "size_summary.csv",
        size_rows,
        [
            "model",
            "n",
            "calibration",
            "residual_source",
            "alpha",
            "replications",
            "rejection_rate",
            "mcse",
            "mean_critical_value",
            "median_critical_value",
            "mean_T_late",
            "variance_T_late",
            "mean_T_direct",
            "max_abs_T_difference",
            "median_sqrt_n_residual",
            "late_components",
        ],
    )
    _write_csv(
        output_dir / "null_distribution_summary.csv",
        distribution_rows,
        [
            "model",
            "n",
            "replications",
            "empirical_mean",
            "limit_mean",
            "empirical_variance",
            "limit_variance",
            "empirical_q50",
            "limit_q50",
            "empirical_q90",
            "limit_q90",
            "empirical_q95",
            "limit_q95",
            "empirical_q99",
            "limit_q99",
        ],
    )
    _write_csv(
        output_dir / "power_summary.csv",
        power_rows,
        [
            "model",
            "n",
            "delta",
            "replications",
            "alpha",
            "calibration",
            "test_engine",
            "critical_value",
            "power",
            "mcse",
        ],
    )
    _write_csv(
        output_dir / "stopping_summary.csv",
        stopping_rows,
        [
            "model",
            "n",
            "stopping_rule",
            "component_value",
            "median_selected_components",
            "replications",
            "alpha",
            "critical_value",
            "rejection_rate",
            "mcse",
            "mean_sqrt_n_residual",
            "median_sqrt_n_residual",
            "mean_abs_stat_difference",
            "median_abs_stat_difference",
            "median_sqrt_n_threshold",
        ],
    )
    _write_csv(
        output_dir / "confidence_set_summary.csv",
        confidence_rows,
        [
            "model",
            "n",
            "replications",
            "confidence_level",
            "calibration",
            "method",
            "basis_dimension",
            "global_coverage",
            "global_coverage_mcse",
            "projected_coverage",
            "projected_coverage_mcse",
            "mean_set_width",
            "median_set_width",
            "empty_frequency",
            "unbounded_frequency",
            "median_rank",
            "median_minimum_statistic",
        ],
    )
    np.savez_compressed(output_dir / "inference_raw_results.npz", **raw)

    configuration = {
        "program": "babii_inference_study.py",
        "seed": arguments.seed,
        "T": design.T,
        "J": design.J,
        "noise_sd": design.noise_sd,
        "basis_convention": design.basis_convention,
        "sample_size": arguments.sample_size,
        "power_sample_sizes": list(map(int, arguments.power_sample_sizes)),
        "late_components": arguments.late_components,
        "cg_engine": arguments.cg_engine,
        "size_replications": size_replications,
        "power_replications": power_replications,
        "confidence_replications": confidence_replications,
        "stopping_replications": stopping_replications,
        "alpha_levels": size_alphas,
        "power_alpha": arguments.power_alpha,
        "delta_step": arguments.delta_step,
        "oracle_draws": arguments.oracle_draws,
        "calibration_draws": arguments.calibration_draws,
        "calibrations": list(arguments.calibrations),
        "power_calibration": arguments.power_calibration,
        "power_direct": bool(arguments.power_direct),
        "confidence_calibration": arguments.confidence_calibration,
        "confidence_components": confidence_components,
        "confidence_method": arguments.confidence_method,
        "confidence_basis_dimension": arguments.confidence_basis_dimension,
        "confidence_grid_points": arguments.confidence_grid_points,
        "stopping_components": list(map(int, arguments.stopping_components)),
        "skips": {
            "size": bool(arguments.skip_size),
            "power": bool(arguments.skip_power),
            "confidence": bool(arguments.skip_confidence),
            "stopping": bool(arguments.skip_stopping),
        },
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__,
        },
    }
    with (output_dir / "configuration.json").open("w", encoding="utf-8") as stream:
        json.dump(configuration, stream, indent=2, sort_keys=True)
        stream.write("\n")

    if not arguments.no_plots:
        print("Creating vector-PDF figures...")
        if not arguments.skip_size:
            plot_null_calibration(design, raw, output_dir / "null_calibration.pdf")
            plot_calibration_comparison(
                design,
                size_rows,
                arguments.power_alpha,
                output_dir / "calibration_comparison.pdf",
            )
        if not arguments.skip_power:
            plot_power_curves(
                design, power_rows, arguments.power_alpha, output_dir / "power_curves.pdf"
            )
        if not arguments.skip_stopping:
            plot_stopping_diagnostics(
                design,
                stopping_rows,
                arguments.power_alpha,
                output_dir / "stopping_diagnostics.pdf",
            )
        if not arguments.skip_confidence:
            plot_confidence_sets(design, raw, output_dir / "confidence_sets.pdf")

    print(f"Completed. Results written to {output_dir.resolve()}")
    if size_rows:
        oracle_five = [
            row
            for row in size_rows
            if row["calibration"] == "oracle"
            and abs(float(row["alpha"]) - arguments.power_alpha) < 1e-12
        ]
        for row in oracle_five:
            print(
                f"  {row['model']}: oracle size={float(row['rejection_rate']):.3f}, "
                f"max |T_n-S_n|={float(row['max_abs_T_difference']):.3e}"
            )
    return {
        "design": design,
        "size_summary": size_rows,
        "null_distribution_summary": distribution_rows,
        "power_summary": power_rows,
        "confidence_set_summary": confidence_rows,
        "stopping_summary": stopping_rows,
        "raw": raw,
    }


def main() -> None:
    arguments = _parse_arguments()
    run_study(arguments)


if __name__ == "__main__":
    main()
