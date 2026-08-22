"""Monte Carlo comparison for scalar-on-function linear regression.

The discretisation used throughout is

    y = X @ beta / T + epsilon,
    r = X.T @ y / n,
    K = X.T @ X / (n * T).

This is a corrected, focused implementation of the accompanying Julia
experiment. It compares Babii et al.'s early-stopped conjugate-gradient FPLS,
cross-validated APLS, and functional principal-component regression (FPCR).
APLS cross-validation uses the same functional inner product in training and
validation, and every reported integrated squared error contains exactly one
factor ``1 / T``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import warnings

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.linalg import LinAlgError, eigh, solve_triangular

sns.set_style("whitegrid")

# ============================================================================
# COLORS
# ============================================================================

COLORS = {
    'PLS': '#1f77b4',
    'APLS': '#ff7f0e',
    'PCA': '#d62728'
}

# This order is used for every returned MSE/MSPE matrix and every summary.
METHOD_NAMES = ("PLS", "APLS", "PCA")

BIAS_COLOR = '#d62728'  # Red for Bias²
VAR_COLOR = '#1f77b4'  # Blue for Variance
NOISE_COLOR = '#7f7f7f'  # Grey for irreducible response noise


# ============================================================================
# PART 1: VALIDATION AND BASIS FUNCTIONS
# ============================================================================

def _validate_grid(s: np.ndarray) -> np.ndarray:
    s = np.asarray(s, dtype=float)
    if s.ndim != 1 or len(s) < 2:
        raise ValueError("s must be a one-dimensional grid with at least two points")
    if not np.all(np.isfinite(s)) or np.any(np.diff(s) <= 0):
        raise ValueError("s must be finite and strictly increasing")
    return s


def _validate_xy(y: np.ndarray, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=float).reshape(-1)
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[0] != len(y):
        raise ValueError("X must be two-dimensional with len(y) rows")
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
        raise ValueError("X and y must contain only finite values")
    return y, X


def create_cosine_basis(
    s: np.ndarray, J: int, convention: str = "babii"
) -> np.ndarray:
    """Return a ``J x T`` cosine basis.

    ``convention='babii'`` reproduces the paper and supplied notebook exactly:
    ``v_1=1`` and ``v_j=sqrt(2)cos(j*pi*s)`` for ``j>=2``; this convention has
    no ``cos(pi*s)`` term.  ``convention='standard'`` uses the usual sequence
    ``1, sqrt(2) cos(pi*s), ..., sqrt(2) cos((J-1)pi*s)``.
    """
    s = _validate_grid(s)
    if J < 1:
        raise ValueError("J must be positive")
    if convention == "babii":
        frequencies = np.arange(1, J + 1)
        basis = np.sqrt(2.0) * np.cos(np.pi * np.outer(s, frequencies))
        basis[:, 0] = 1.0
    elif convention == "standard":
        frequencies = np.arange(J)
        basis = np.sqrt(2.0) * np.cos(np.pi * np.outer(s, frequencies))
        basis[:, 0] = 1.0
    else:
        raise ValueError("convention must be 'babii' or 'standard'")
    return basis.T


# ============================================================================
# PART 2: ESTIMATION METHODS
# ============================================================================

def pls(r: np.ndarray, K: np.ndarray, m: int, tolerance: float = 1e-12) -> np.ndarray:
    """PLS/conjugate-gradient iterates for the sample moment equation K beta=r."""
    r = np.asarray(r, dtype=float).reshape(-1)
    K = np.asarray(K, dtype=float)
    if K.shape != (len(r), len(r)):
        raise ValueError("K must be square with dimension len(r)")
    if m < 1:
        raise ValueError("m must be positive")
    K = 0.5 * (K + K.T)
    T = len(r)
    beta_path = np.zeros((T, m))
    beta_previous = np.zeros(T)
    residual = r.copy()
    direction = r.copy()
    # The infinity norm is a cheap upper bound for the spectral norm and is
    # sufficient for numerical-breakdown checks inside a large simulation.
    norm_K = np.linalg.norm(K, ord=np.inf)

    for component in range(m):
        K_residual = K @ residual
        K_direction = K @ direction
        numerator = float(residual @ K_residual)
        direction_scale = max(norm_K * np.linalg.norm(direction), np.finfo(float).tiny)
        if (
            not np.isfinite(numerator)
            or numerator <= tolerance * np.linalg.norm(residual) * np.linalg.norm(K_residual)
            or np.linalg.norm(K_direction) <= tolerance * direction_scale
        ):
            beta_path[:, component:] = beta_previous[:, None]
            warnings.warn(
                f"PLS reached numerical breakdown after {component} components; "
                "the last finite iterate is reused.",
                RuntimeWarning,
                stacklevel=2,
            )
            break

        denominator = float(K_direction @ K_direction)
        alpha = numerator / denominator
        beta_current = beta_previous + alpha * direction
        residual_new = residual - alpha * K_direction
        K_residual_new = K @ residual_new
        gamma = float(residual_new @ K_residual_new) / numerator

        if not (
            np.all(np.isfinite(beta_current))
            and np.all(np.isfinite(residual_new))
            and np.isfinite(gamma)
        ):
            raise FloatingPointError("PLS produced a non-finite iterate")

        beta_path[:, component] = beta_current
        beta_previous = beta_current
        residual = residual_new
        direction = residual_new + gamma * direction
    return beta_path


def pls_early_stop(beta_hat_pls: np.ndarray, X: np.ndarray, y: np.ndarray,
                   r: np.ndarray, K: np.ndarray, tau: float, delta: float,
                   n: int, beta_hat_pca: np.ndarray) -> Tuple[np.ndarray, int, np.ndarray, float]:
    """Select the first PLS iterate satisfying Babii et al.'s discrepancy rule."""
    y, X = _validate_xy(y, X)
    if n != X.shape[0]:
        raise ValueError("n must equal X.shape[0]")
    if tau <= 0 or not 0 < delta < 1:
        raise ValueError("tau must be positive and delta must lie in (0, 1)")
    T, m = beta_hat_pls.shape
    X_norm = np.mean(np.sum(X ** 2, axis=1) / T)
    sigma2 = max(float(np.mean((y - X @ beta_hat_pca / T) ** 2)), 0.0)
    moment = np.zeros(m)
    for j in range(m):
        residual = K @ beta_hat_pls[:, j] - r
        moment[j] = np.sqrt(np.sum(residual ** 2) / T)
    threshold = tau * np.sqrt(2 * sigma2 * X_norm / (delta * n))
    indices = np.where(moment <= threshold)[0]
    if len(indices) > 0:
        m_hat = indices[0] + 1
        beta_opt = beta_hat_pls[:, m_hat - 1]
    else:
        warnings.warn("No component satisfies the stopping criterion. Using last component.")
        m_hat = m
        beta_opt = beta_hat_pls[:, -1]
    return beta_opt, m_hat, moment, threshold


def _orthonormal_krylov_basis(
    K: np.ndarray, r: np.ndarray, m_max: int, tolerance: float
) -> np.ndarray:
    """Build an orthonormal basis of span(r, Kr, ..., K^(m-1)r)."""
    T = len(r)
    norm_r = np.linalg.norm(r)
    if norm_r == 0:
        return np.zeros((T, 0))
    Q = np.zeros((T, m_max))
    Q[:, 0] = r / norm_r
    dimension = 1
    norm_K = max(np.linalg.norm(K, ord=np.inf), np.finfo(float).tiny)
    while dimension < m_max:
        candidate = K @ Q[:, dimension - 1]
        # Two-pass modified Gram-Schmidt is inexpensive here and materially
        # improves the stability of high-order APLS components.
        for _ in range(2):
            candidate -= Q[:, :dimension] @ (Q[:, :dimension].T @ candidate)
        candidate_norm = np.linalg.norm(candidate)
        if candidate_norm <= tolerance * norm_K:
            break
        Q[:, dimension] = candidate / candidate_norm
        dimension += 1
    return Q[:, :dimension]


def _fit_krylov_path(
    y: np.ndarray,
    X: np.ndarray,
    K: np.ndarray,
    r: np.ndarray,
    m_max: int,
    rcond: float,
) -> np.ndarray:
    """Fit nested least-squares models over an orthonormal Krylov basis."""
    T = X.shape[1]
    krylov = _orthonormal_krylov_basis(K, r, m_max, rcond)
    beta_path = np.zeros((T, m_max))
    if krylov.shape[1] == 0:
        return beta_path

    design = X @ krylov / T
    Q_design, R_design = np.linalg.qr(design, mode="reduced")
    projected_y = Q_design.T @ y
    numerical_rank = 0
    reference = max(abs(R_design[0, 0]), np.finfo(float).tiny)
    for component in range(krylov.shape[1]):
        if abs(R_design[component, component]) <= rcond * reference:
            break
        numerical_rank = component + 1
        coefficients = solve_triangular(
            R_design[:numerical_rank, :numerical_rank],
            projected_y[:numerical_rank],
            lower=False,
            check_finite=False,
        )
        beta_path[:, component] = krylov[:, :numerical_rank] @ coefficients

    if numerical_rank == 0:
        return beta_path
    beta_path[:, numerical_rank:] = beta_path[:, [numerical_rank - 1]]
    return beta_path


def apls_cv_stable(y: np.ndarray, X: np.ndarray, K: np.ndarray, r: np.ndarray,
                   m_max: int, k_folds: int = 5, random_state: Optional[int] = None,
                   reg_param: Optional[float] = None,
                   rcond: float = 1e-10,
                   rng: Optional[np.random.Generator] = None,
                   ) -> Tuple[np.ndarray, int, np.ndarray, np.ndarray]:
    """Five-fold CV for alternative PLS over the Krylov subspaces.

    There is deliberately no pointwise standardisation: it changes the
    functional inner product and hence the estimator.  Every fold uses
    ``K_tr = X_tr.T @ X_tr / (n_tr*T)`` and validation predictions use
    ``X_val @ beta / T``.  After selecting ``m``, the model is refitted on the
    complete sample.
    """
    y, X = _validate_xy(y, X)
    n, T = X.shape
    if m_max < 1:
        raise ValueError("m_max must be positive")
    if not 2 <= k_folds <= n:
        raise ValueError("k_folds must be between 2 and n")
    if rcond <= 0:
        raise ValueError("rcond must be positive")
    if reg_param not in (None, 0):
        warnings.warn(
            "reg_param is ignored; stable orthonormalisation replaces the "
            "ridge perturbation without changing the APLS subspace.",
            DeprecationWarning,
            stacklevel=2,
        )
    K = 0.5 * (np.asarray(K, dtype=float) + np.asarray(K, dtype=float).T)
    r = np.asarray(r, dtype=float).reshape(-1)
    if K.shape != (T, T) or r.shape != (T,):
        raise ValueError("K and r have incompatible dimensions")

    if rng is None:
        rng = np.random.default_rng(random_state)
    folds = np.array_split(rng.permutation(n), k_folds)
    cv_sse = np.zeros(m_max)

    for validation_indices in folds:
        training_mask = np.ones(n, dtype=bool)
        training_mask[validation_indices] = False
        X_train, y_train = X[training_mask], y[training_mask]
        X_validation, y_validation = X[validation_indices], y[validation_indices]
        n_train = len(y_train)
        r_train = X_train.T @ y_train / n_train
        K_train = X_train.T @ X_train / (n_train * T)
        beta_path = _fit_krylov_path(
            y_train, X_train, K_train, r_train, m_max, rcond
        )
        predictions = X_validation @ beta_path / T
        cv_sse += np.sum((y_validation[:, None] - predictions) ** 2, axis=0)

    cv_errors = cv_sse / n
    if not np.all(np.isfinite(cv_errors)):
        raise FloatingPointError("APLS cross-validation produced non-finite errors")
    m_opt = int(np.argmin(cv_errors)) + 1
    beta_path_full = _fit_krylov_path(y, X, K, r, m_max, rcond)
    return beta_path_full[:, m_opt - 1], m_opt, cv_errors, beta_path_full


def pca_gcv(r: np.ndarray, K: np.ndarray, m: int) -> Tuple[np.ndarray, int, np.ndarray]:
    """Functional principal-component regression with moment-space GCV."""
    r = np.asarray(r, dtype=float).reshape(-1)
    K = np.asarray(K, dtype=float)
    T = len(r)
    if K.shape != (T, T) or m < 1:
        raise ValueError("K has the wrong shape or m is not positive")
    K = 0.5 * (K + K.T)
    try:
        eigenvalues, eigenvectors = eigh(K, check_finite=False)
    except LinAlgError:
        eigenvectors, eigenvalues, _ = np.linalg.svd(K, full_matrices=False)
    threshold = np.finfo(float).eps * T * max(float(np.max(eigenvalues)), 1.0)
    positive = np.flatnonzero(eigenvalues > threshold)[::-1]
    maximum_components = min(m, len(positive), T - 1)
    if maximum_components == 0:
        raise LinAlgError("K has no numerically positive eigenvalues")

    beta_path = np.zeros((T, m))
    gcv = np.full(m, np.inf)
    for number in range(1, maximum_components + 1):
        indices = positive[:number]
        values = eigenvalues[indices]
        vectors = eigenvectors[:, indices]
        beta_path[:, number - 1] = vectors @ ((vectors.T @ r) / values)
        residual = r - K @ beta_path[:, number - 1]
        gcv[number - 1] = (
            np.mean(residual ** 2) / (1.0 - number / T) ** 2
        )
    if maximum_components < m:
        beta_path[:, maximum_components:] = beta_path[:, [maximum_components - 1]]
    m_opt = int(np.argmin(gcv)) + 1
    return beta_path[:, m_opt - 1], m_opt, gcv


# ============================================================================
# PART 3: BIAS-VARIANCE FUNCTIONS (from previous code)
# ============================================================================

def compute_bias_variance_beta(beta_estimates: np.ndarray, beta_true: np.ndarray) -> Dict:
    """Pointwise and integrated Monte Carlo decomposition for the slope."""
    beta_estimates = np.asarray(beta_estimates, dtype=float)
    beta_true = np.asarray(beta_true, dtype=float).reshape(-1)
    T, M = beta_estimates.shape
    if beta_true.shape != (T,) or M < 1:
        raise ValueError("beta_estimates and beta_true have incompatible shapes")
    beta_mean = np.mean(beta_estimates, axis=1)
    bias2 = (beta_mean - beta_true) ** 2
    # ddof=0 gives the exact finite-M identity
    # mean[(beta_hat-beta)^2] = bias^2 + variance.
    variance = np.var(beta_estimates, axis=1, ddof=0)
    mse = bias2 + variance
    return {
        'bias2': bias2,
        'variance': variance,
        'mse': mse,
        'beta_mean': beta_mean,
        'integrated_bias2': np.mean(bias2),
        'integrated_variance': np.mean(variance),
        'integrated_mse': np.mean(mse)
    }


def compute_bias_variance_prediction(
    beta_estimates: np.ndarray,
    X: np.ndarray,
    beta_true: np.ndarray,
    T: int,
    noise_variance: float = 1.0,
    return_predictions: bool = False,
) -> Dict:
    """Prediction decomposition on an independent, fixed evaluation design.

    Bias is measured against the noise-free regression function
    ``X @ beta_true / T``.  ``noise_variance`` is then added exactly once when
    reporting response prediction MSE.
    """
    beta_estimates = np.asarray(beta_estimates, dtype=float)
    X = np.asarray(X, dtype=float)
    beta_true = np.asarray(beta_true, dtype=float).reshape(-1)
    if X.ndim != 2 or X.shape[1] != T or beta_true.shape != (T,):
        raise ValueError("X, beta_true and T have incompatible dimensions")
    if beta_estimates.shape[0] != T or beta_estimates.shape[1] < 1:
        raise ValueError("beta_estimates must have shape (T, M), with M >= 1")
    if noise_variance < 0 or not np.isfinite(noise_variance):
        raise ValueError("noise_variance must be finite and non-negative")
    n = X.shape[0]
    y_pred = X @ beta_estimates / T
    signal_true = X @ beta_true / T
    y_mean = np.mean(y_pred, axis=1)
    bias2 = (y_mean - signal_true) ** 2
    variance = np.var(y_pred, axis=1, ddof=0)
    noise = np.full(n, noise_variance)
    signal_mse = bias2 + variance
    mse = signal_mse + noise
    result = {
        'bias2': bias2,
        'variance': variance,
        'noise': noise,
        'mse': mse,
        'y_mean': y_mean,
        'signal_true': signal_true,
        'signal_mse': signal_mse,
        'integrated_bias2': np.mean(bias2),
        'integrated_variance': np.mean(variance),
        'integrated_signal_mse': np.mean(signal_mse),
        'integrated_mse': np.mean(mse),
        'integrated_noise': float(noise_variance),
    }
    if return_predictions:
        result['y_pred'] = y_pred
    return result


# ============================================================================
# PART 4: COEFFICIENT DISTRIBUTION FUNCTIONS (NEW)
# ============================================================================

def _save_figure(
    fig: plt.Figure, save_dir: Union[str, Path], filename: str
) -> Path:
    directory = Path(save_dir)
    directory.mkdir(parents=True, exist_ok=True)
    output_path = directory / filename
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path.name}")
    return output_path


def project_to_cosine_basis(beta_estimates: np.ndarray, v: np.ndarray, J: int) -> np.ndarray:
    """
    Project beta estimates onto the cosine basis.

    Parameters:
    beta_estimates : ndarray of shape (T, M)
        M estimates of beta from M simulations
    v : ndarray of shape (J_total, T)
        Full cosine basis
    J : int
        Number of basis functions to use (first J)

    Returns:
    coeffs : ndarray of shape (J, M)
        Coefficients in the cosine basis for each simulation
    """
    T, M = beta_estimates.shape
    v_subset = v[:J, :]  # Use first J basis functions

    coeffs = np.zeros((J, M))
    for k in range(M):
        # Least squares projection: beta ≈ v_subset.T @ c
        # Solve for c: v_subset.T @ c = beta
        c = np.linalg.lstsq(v_subset.T, beta_estimates[:, k], rcond=None)[0]
        coeffs[:, k] = c

    return coeffs


def plot_coefficient_distributions(beta_estimates_dict: Dict, v: np.ndarray,
                                   beta_true: np.ndarray, method_names: List[str],
                                   model_name: str, J: int = 10,
                                   n_coeffs: int = 3, save_dir: str = "."):
    """
    Plot histograms of the first n_coeffs coefficients in the cosine basis.
    """
    # Project true beta to cosine basis
    true_coeffs = np.linalg.lstsq(v[:J, :].T, beta_true, rcond=None)[0]

    fig, axes = plt.subplots(n_coeffs, len(method_names),
                             figsize=(5 * len(method_names), 4 * n_coeffs))

    if n_coeffs == 1:
        axes = axes[np.newaxis, :]
    if len(method_names) == 1:
        axes = axes[:, np.newaxis]

    for i, method in enumerate(method_names):
        if method not in beta_estimates_dict:
            continue

        # Project to cosine basis
        coeffs = project_to_cosine_basis(beta_estimates_dict[method], v, J)

        for j in range(n_coeffs):
            ax = axes[j, i]

            # Histogram of coefficients
            ax.hist(coeffs[j, :], bins=30, density=True, alpha=0.6,
                    color=COLORS[method], edgecolor='black', linewidth=0.5)

            # True coefficient (vertical line)
            ax.axvline(true_coeffs[j], color='red', linestyle='-', linewidth=2.5,
                       label=f'True = {true_coeffs[j]:.3f}')

            # Mean of estimates
            mean_coeff = np.mean(coeffs[j, :])
            ax.axvline(mean_coeff, color='green', linestyle='--', linewidth=2,
                       label=f'Mean = {mean_coeff:.3f}')

            # Add statistics
            bias = mean_coeff - true_coeffs[j]
            var = np.var(coeffs[j, :], ddof=1) if coeffs.shape[1] > 1 else 0.0

            ax.set_xlabel(f'Coefficient {j + 1}', fontsize=10)
            if j == 0:
                ax.set_ylabel('Density', fontsize=10)
            ax.set_title(f'{method}\nBias={bias:.3f}, Var={var:.3f}',
                         fontsize=10, fontweight='bold')
            ax.legend(fontsize=8, loc='best')
            ax.grid(True, alpha=0.2)

    plt.suptitle(f'{model_name}: Distributions of First {n_coeffs} Cosine Basis Coefficients',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    _save_figure(
        fig,
        save_dir,
        f"{model_name.lower().replace(' ', '_')}_coeff_distributions.pdf",
    )
    plt.close(fig)

    return fig


def plot_beta_at_points(beta_estimates_dict: Dict, s: np.ndarray,
                        beta_true: np.ndarray, method_names: List[str],
                        model_name: str, n_points: int = 5, save_dir: str = "."):
    """
    Plot histograms of β(s) at selected grid points.
    """
    # Select grid points
    indices = np.linspace(0, len(s) - 1, n_points, dtype=int)

    fig, axes = plt.subplots(n_points, len(method_names),
                             figsize=(5 * len(method_names), 4 * n_points))

    if n_points == 1:
        axes = axes[np.newaxis, :]
    if len(method_names) == 1:
        axes = axes[:, np.newaxis]

    for i, method in enumerate(method_names):
        if method not in beta_estimates_dict:
            continue

        beta_ests = beta_estimates_dict[method]

        for j, idx in enumerate(indices):
            ax = axes[j, i]

            # Histogram of β(s_idx)
            beta_vals = beta_ests[idx, :]
            ax.hist(beta_vals, bins=30, density=True, alpha=0.6,
                    color=COLORS[method], edgecolor='black', linewidth=0.5)

            # True value
            true_val = beta_true[idx]
            ax.axvline(true_val, color='red', linestyle='-', linewidth=2.5,
                       label=f'True = {true_val:.3f}')

            # Mean
            mean_val = np.mean(beta_vals)
            ax.axvline(mean_val, color='green', linestyle='--', linewidth=2,
                       label=f'Mean = {mean_val:.3f}')

            # Statistics
            bias = mean_val - true_val
            var = np.var(beta_vals, ddof=1) if beta_vals.size > 1 else 0.0

            ax.set_xlabel(f'β({s[idx]:.2f})', fontsize=10)
            if j == 0:
                ax.set_ylabel('Density', fontsize=10)
            ax.set_title(f'{method}\ns={s[idx]:.2f}, Bias={bias:.3f}, Var={var:.3f}',
                         fontsize=9, fontweight='bold')
            ax.legend(fontsize=7, loc='best')
            ax.grid(True, alpha=0.2)

    plt.suptitle(f'{model_name}: Distributions of β(s) at Selected Grid Points',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    _save_figure(
        fig,
        save_dir,
        f"{model_name.lower().replace(' ', '_')}_beta_point_distributions.pdf",
    )
    plt.close(fig)

    return fig


def plot_coefficient_comparison_across_models(all_results: Dict, v: np.ndarray,
                                              method_names: List[str],
                                              model_names: List[str],
                                              J: int = 10, n_coeffs: int = 3,
                                              save_dir: str = "."):
    """
    Compare coefficient distributions across all models for each method.
    """
    fig, axes = plt.subplots(n_coeffs, len(method_names),
                             figsize=(5 * len(method_names), 4 * n_coeffs))

    if n_coeffs == 1:
        axes = axes[np.newaxis, :]
    if len(method_names) == 1:
        axes = axes[:, np.newaxis]

    colors_models = {'Model 1': '#1f77b4', 'Model 2': '#ff7f0e', 'Model 3': '#2ca02c'}

    for i, method in enumerate(method_names):
        for j in range(n_coeffs):
            ax = axes[j, i]

            # Collect coefficients for this method and coefficient index
            for model_name in model_names:
                if model_name not in all_results:
                    continue
                beta_ests = all_results[model_name]['beta_estimates'][method]
                coeffs = project_to_cosine_basis(beta_ests, v, J)

                # KDE for smooth histogram
                try:
                    from scipy.stats import gaussian_kde
                    kde = gaussian_kde(coeffs[j, :])
                    x_range = np.linspace(np.percentile(coeffs[j, :], 1),
                                          np.percentile(coeffs[j, :], 99), 100)
                    ax.plot(x_range, kde(x_range),
                            color=colors_models[model_name],
                            linewidth=2, label=model_name, alpha=0.8)
                except (LinAlgError, ValueError):
                    ax.hist(coeffs[j, :], bins=20, density=True, alpha=0.3,
                            color=colors_models[model_name], label=model_name)

            ax.set_xlabel(f'Coefficient {j + 1}', fontsize=10)
            if j == 0:
                ax.set_ylabel('Density', fontsize=10)
            ax.set_title(f'{method}', fontsize=11, fontweight='bold')
            ax.legend(fontsize=8, loc='best')
            ax.grid(True, alpha=0.2)

    plt.suptitle(f'Coefficient Distributions Across Models (First {n_coeffs} Coefficients)',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    _save_figure(fig, save_dir, "coeff_distributions_across_models.pdf")
    plt.close(fig)

    return fig


def plot_beta_distribution_across_models(all_results: Dict, s: np.ndarray,
                                         method_names: List[str],
                                         model_names: List[str],
                                         n_points: int = 5,
                                         save_dir: str = "."):
    """
    Compare β(s) distributions across all models for each method.
    """
    indices = np.linspace(0, len(s) - 1, n_points, dtype=int)

    fig, axes = plt.subplots(n_points, len(method_names),
                             figsize=(5 * len(method_names), 4 * n_points))

    if n_points == 1:
        axes = axes[np.newaxis, :]
    if len(method_names) == 1:
        axes = axes[:, np.newaxis]

    colors_models = {'Model 1': '#1f77b4', 'Model 2': '#ff7f0e', 'Model 3': '#2ca02c'}

    for i, method in enumerate(method_names):
        for j, idx in enumerate(indices):
            ax = axes[j, i]

            for model_name in model_names:
                if model_name not in all_results:
                    continue
                beta_ests = all_results[model_name]['beta_estimates'][method]
                beta_vals = beta_ests[idx, :]

                try:
                    from scipy.stats import gaussian_kde
                    kde = gaussian_kde(beta_vals)
                    x_range = np.linspace(np.percentile(beta_vals, 1),
                                          np.percentile(beta_vals, 99), 100)
                    ax.plot(x_range, kde(x_range),
                            color=colors_models[model_name],
                            linewidth=2, label=model_name, alpha=0.8)
                except (LinAlgError, ValueError):
                    ax.hist(beta_vals, bins=20, density=True, alpha=0.3,
                            color=colors_models[model_name], label=model_name)

            ax.set_xlabel(f'β({s[idx]:.2f})', fontsize=10)
            if j == 0:
                ax.set_ylabel('Density', fontsize=10)
            ax.set_title(f'{method}\ns={s[idx]:.2f}', fontsize=10, fontweight='bold')
            ax.legend(fontsize=8, loc='best')
            ax.grid(True, alpha=0.2)

    plt.suptitle(f'β(s) Distributions Across Models at Selected Points',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    _save_figure(fig, save_dir, "beta_distributions_across_models.pdf")
    plt.close(fig)

    return fig


# ============================================================================
# PART 5: COMPOSITE BAR GRAPHS (from previous code)
# ============================================================================

def plot_composite_mse_bars(results_beta: Dict, results_pred: Dict,
                            method_names: List[str], model_name: str,
                            save_dir: str = "."):
    """Composite bar graphs for MSE decomposition."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # β̂
    ax = axes[0]
    methods = []
    bias2_vals = []
    var_vals = []
    mse_vals = []
    for method in method_names:
        if method in results_beta:
            methods.append(method)
            bias2_vals.append(results_beta[method]['integrated_bias2'])
            var_vals.append(results_beta[method]['integrated_variance'])
            mse_vals.append(results_beta[method]['integrated_mse'])

    x_pos = np.arange(len(methods))
    width = 0.6

    bars1 = ax.bar(x_pos, bias2_vals, width, color=BIAS_COLOR, alpha=0.8,
                   label='Bias²', edgecolor='black', linewidth=0.8)
    bars2 = ax.bar(x_pos, var_vals, width, color=VAR_COLOR, alpha=0.8,
                   label='Variance', bottom=bias2_vals,
                   edgecolor='black', linewidth=0.8)

    for idx, (b, v) in enumerate(zip(bias2_vals, var_vals)):
        total = b + v
        ax.text(idx, total + 0.001, f'{total:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        if b > 0.001:
            ax.text(idx, b / 2, f'{b:.3f}', ha='center', va='center', fontsize=9, color='white', fontweight='bold')
        if v > 0.001:
            ax.text(idx, b + v / 2, f'{v:.3f}', ha='center', va='center', fontsize=9, color='white', fontweight='bold')

    ax.set_xlabel('Method', fontsize=14)
    ax.set_ylabel('MSE Decomposition', fontsize=14)
    ax.set_title(f'{model_name}: MSE for β̂ (Coefficient Estimation)', fontsize=15, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods, fontsize=12)
    ax.legend(fontsize=12, loc='upper right')
    ax.grid(True, alpha=0.2, axis='y')
    ax.set_ylim(0, max(mse_vals) * 1.25 if mse_vals else 1)

    # ŷ
    ax = axes[1]
    methods = []
    bias2_vals = []
    var_vals = []
    noise_vals = []
    mse_vals = []
    for method in method_names:
        if method in results_pred:
            methods.append(method)
            bias2_vals.append(results_pred[method]['integrated_bias2'])
            var_vals.append(results_pred[method]['integrated_variance'])
            noise_vals.append(results_pred[method]['integrated_noise'])
            mse_vals.append(results_pred[method]['integrated_mse'])

    x_pos = np.arange(len(methods))

    bars1 = ax.bar(x_pos, bias2_vals, width, color=BIAS_COLOR, alpha=0.8,
                   label='Bias²', edgecolor='black', linewidth=0.8)
    bars2 = ax.bar(x_pos, var_vals, width, color=VAR_COLOR, alpha=0.8,
                   label='Variance', bottom=bias2_vals,
                   edgecolor='black', linewidth=0.8)
    ax.bar(
        x_pos,
        noise_vals,
        width,
        color=NOISE_COLOR,
        alpha=0.8,
        label='Noise',
        bottom=np.asarray(bias2_vals) + np.asarray(var_vals),
        edgecolor='black',
        linewidth=0.8,
    )

    for idx, (b, v, noise) in enumerate(zip(bias2_vals, var_vals, noise_vals)):
        total = b + v + noise
        ax.text(idx, total + 0.001, f'{total:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        if b > 0.001:
            ax.text(idx, b / 2, f'{b:.3f}', ha='center', va='center', fontsize=9, color='white', fontweight='bold')
        if v > 0.001:
            ax.text(idx, b + v / 2, f'{v:.3f}', ha='center', va='center', fontsize=9, color='white', fontweight='bold')

    ax.set_xlabel('Method', fontsize=14)
    ax.set_ylabel('MSE Decomposition', fontsize=14)
    ax.set_title(f'{model_name}: MSE for ŷ (Prediction)', fontsize=15, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods, fontsize=12)
    ax.legend(fontsize=12, loc='upper right')
    ax.grid(True, alpha=0.2, axis='y')
    ax.set_ylim(0, max(mse_vals) * 1.25 if mse_vals else 1)

    plt.suptitle(f'{model_name}: MSE Decomposition (Bias² + Variance + Noise)',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    _save_figure(
        fig,
        save_dir,
        f"{model_name.lower().replace(' ', '_')}_composite_mse.pdf",
    )
    plt.close(fig)
    return fig


def plot_composite_mse_across_models(all_results: Dict, method_names: List[str],
                                     save_dir: str = "."):
    """Composite bar graphs across all models."""
    model_names = list(all_results.keys())

    # β̂
    fig1, axes1 = plt.subplots(1, 3, figsize=(18, 6))
    for idx, model_name in enumerate(model_names):
        ax = axes1[idx]
        results_beta = all_results[model_name]['results_beta']
        methods = []
        bias2_vals = []
        var_vals = []
        mse_vals = []
        for method in method_names:
            if method in results_beta:
                methods.append(method)
                bias2_vals.append(results_beta[method]['integrated_bias2'])
                var_vals.append(results_beta[method]['integrated_variance'])
                mse_vals.append(results_beta[method]['integrated_mse'])
        x_pos = np.arange(len(methods))
        width = 0.6
        ax.bar(x_pos, bias2_vals, width, color=BIAS_COLOR, alpha=0.8, label='Bias²', edgecolor='black', linewidth=0.8)
        ax.bar(x_pos, var_vals, width, color=VAR_COLOR, alpha=0.8, label='Variance', bottom=bias2_vals,
               edgecolor='black', linewidth=0.8)
        for i, (b, v) in enumerate(zip(bias2_vals, var_vals)):
            total = b + v
            ax.text(i, total + 0.001, f'{total:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax.set_xlabel('Method', fontsize=12)
        ax.set_ylabel('MSE' if idx == 0 else '', fontsize=12)
        ax.set_title(model_name, fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=10)
        ax.grid(True, alpha=0.2, axis='y')
        if idx == 0:
            ax.legend(fontsize=11)
        ax.set_ylim(0, max(mse_vals) * 1.2 if mse_vals else 1)
    plt.suptitle('MSE for β̂ (Coefficient Estimation) Across All Models', fontsize=16, fontweight='bold')
    plt.tight_layout()
    _save_figure(fig1, save_dir, "composite_mse_beta_across_models.pdf")
    plt.close(fig1)

    # ŷ
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))
    for idx, model_name in enumerate(model_names):
        ax = axes2[idx]
        results_pred = all_results[model_name]['results_pred']
        methods = []
        bias2_vals = []
        var_vals = []
        noise_vals = []
        mse_vals = []
        for method in method_names:
            if method in results_pred:
                methods.append(method)
                bias2_vals.append(results_pred[method]['integrated_bias2'])
                var_vals.append(results_pred[method]['integrated_variance'])
                noise_vals.append(results_pred[method]['integrated_noise'])
                mse_vals.append(results_pred[method]['integrated_mse'])
        x_pos = np.arange(len(methods))
        width = 0.6
        ax.bar(x_pos, bias2_vals, width, color=BIAS_COLOR, alpha=0.8, label='Bias²', edgecolor='black', linewidth=0.8)
        ax.bar(x_pos, var_vals, width, color=VAR_COLOR, alpha=0.8, label='Variance', bottom=bias2_vals,
               edgecolor='black', linewidth=0.8)
        ax.bar(
            x_pos,
            noise_vals,
            width,
            color=NOISE_COLOR,
            alpha=0.8,
            label='Noise',
            bottom=np.asarray(bias2_vals) + np.asarray(var_vals),
            edgecolor='black',
            linewidth=0.8,
        )
        for i, (b, v, noise) in enumerate(zip(bias2_vals, var_vals, noise_vals)):
            total = b + v + noise
            ax.text(i, total + 0.001, f'{total:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax.set_xlabel('Method', fontsize=12)
        ax.set_ylabel('MSE' if idx == 0 else '', fontsize=12)
        ax.set_title(model_name, fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=10)
        ax.grid(True, alpha=0.2, axis='y')
        if idx == 0:
            ax.legend(fontsize=11)
        ax.set_ylim(0, max(mse_vals) * 1.2 if mse_vals else 1)
    plt.suptitle('MSE for ŷ (Prediction) Across All Models', fontsize=16, fontweight='bold')
    plt.tight_layout()
    _save_figure(fig2, save_dir, "composite_mse_prediction_across_models.pdf")
    plt.close(fig2)
    return fig1, fig2


def plot_composite_mse_summary(all_results: Dict, method_names: List[str],
                               save_dir: str = "."):
    """Summary composite bar graph."""
    model_names = list(all_results.keys())
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    width = 0.15
    x_pos = np.arange(len(model_names))

    # β̂
    ax = axes[0]
    for m_idx, method in enumerate(method_names):
        bias_vals = []
        var_vals = []
        for model_name in model_names:
            if method in all_results[model_name]['results_beta']:
                bias_vals.append(all_results[model_name]['results_beta'][method]['integrated_bias2'])
                var_vals.append(all_results[model_name]['results_beta'][method]['integrated_variance'])
            else:
                bias_vals.append(0)
                var_vals.append(0)
        offset = (m_idx - len(method_names) / 2 + 0.5) * width
        bottom = np.zeros(len(model_names))
        ax.bar(x_pos + offset, bias_vals, width, color=BIAS_COLOR, alpha=0.7, edgecolor='black', linewidth=0.5,
               bottom=bottom)
        bottom += np.array(bias_vals)
        ax.bar(x_pos + offset, var_vals, width, color=COLORS[method], alpha=0.7, label=method, edgecolor='black',
               linewidth=0.5, bottom=bottom)
    ax.set_xlabel('Data Model', fontsize=13)
    ax.set_ylabel('MSE', fontsize=13)
    ax.set_title('β̂ (Coefficient Estimation)', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(model_names, fontsize=12)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.2, axis='y')

    # ŷ
    ax = axes[1]
    for m_idx, method in enumerate(method_names):
        bias_vals = []
        var_vals = []
        noise_vals = []
        for model_name in model_names:
            if method in all_results[model_name]['results_pred']:
                bias_vals.append(all_results[model_name]['results_pred'][method]['integrated_bias2'])
                var_vals.append(all_results[model_name]['results_pred'][method]['integrated_variance'])
                noise_vals.append(all_results[model_name]['results_pred'][method]['integrated_noise'])
            else:
                bias_vals.append(0)
                var_vals.append(0)
                noise_vals.append(0)
        offset = (m_idx - len(method_names) / 2 + 0.5) * width
        bottom = np.zeros(len(model_names))
        ax.bar(x_pos + offset, bias_vals, width, color=BIAS_COLOR, alpha=0.7, edgecolor='black', linewidth=0.5,
               bottom=bottom)
        bottom += np.array(bias_vals)
        ax.bar(x_pos + offset, var_vals, width, color=COLORS[method], alpha=0.7, label=method, edgecolor='black',
               linewidth=0.5, bottom=bottom)
        bottom += np.array(var_vals)
        ax.bar(
            x_pos + offset,
            noise_vals,
            width,
            color=NOISE_COLOR,
            alpha=0.55,
            edgecolor='black',
            linewidth=0.5,
            bottom=bottom,
        )
    ax.set_xlabel('Data Model', fontsize=13)
    ax.set_ylabel('MSE', fontsize=13)
    ax.set_title('ŷ (Prediction)', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(model_names, fontsize=12)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.2, axis='y')

    plt.suptitle('Composite MSE Summary Across All Models and Methods', fontsize=16, fontweight='bold')
    plt.tight_layout()
    _save_figure(fig, save_dir, "composite_mse_summary.pdf")
    plt.close(fig)
    return fig


def create_bias_variance_table(results_beta: Dict, results_pred: Dict,
                               method_names: List[str], model_name: str):
    """Print Bias-Variance tables."""
    print(f"\n{'=' * 80}")
    print(f"BIAS-VARIANCE DECOMPOSITION - {model_name}")
    print('=' * 80)
    print("\n--- COEFFICIENT ESTIMATION (β̂) ---")
    print(f"{'Method':<12} {'Bias²':<14} {'Variance':<14} {'MSE':<14}")
    print("-" * 54)
    for method in method_names:
        if method in results_beta:
            print(f"{method:<12} {results_beta[method]['integrated_bias2']:<14.6f} "
                  f"{results_beta[method]['integrated_variance']:<14.6f} "
                  f"{results_beta[method]['integrated_mse']:<14.6f}")
    print("\n--- PREDICTION (ŷ) ---")
    print(f"{'Method':<12} {'Bias²':<14} {'Variance':<14} {'Noise':<14} {'MSE':<14}")
    print("-" * 70)
    for method in method_names:
        if method in results_pred:
            print(f"{method:<12} {results_pred[method]['integrated_bias2']:<14.6f} "
                  f"{results_pred[method]['integrated_variance']:<14.6f} "
                  f"{results_pred[method]['integrated_noise']:<14.6f} "
                  f"{results_pred[method]['integrated_mse']:<14.6f}")
    print("\n" + "=" * 80)
    if results_beta:
        best_beta = min(results_beta.keys(), key=lambda x: results_beta[x]['integrated_mse'])
        print(f"Best for β̂: {best_beta} (MSE = {results_beta[best_beta]['integrated_mse']:.6f})")
    if results_pred:
        best_pred = min(results_pred.keys(), key=lambda x: results_pred[x]['integrated_mse'])
        print(f"Best for ŷ: {best_pred} (MSE = {results_pred[best_pred]['integrated_mse']:.6f})")


# ============================================================================
# PART 6: SIMULATION
# ============================================================================

def simulation_cv_with_beta(lam: np.ndarray, beta: np.ndarray, v: np.ndarray,
                            s: np.ndarray, M: int, n: int, J: int, m: int,
                            tau: float = 1.5, delta: float = 0.1,
                            verbose: bool = False,
                            noise_sd: float = 1.0,
                            n_evaluation: int = 1000,
                            rng: Optional[np.random.Generator] = None,
                            evaluation_rng: Optional[np.random.Generator] = None,
                            ) -> Tuple:
    """Run one Monte Carlo design and retain every slope estimate.

    The independent test response used for MSPE includes fresh noise.  The
    returned ``X_evaluation`` is a separate, fixed design used later for a
    valid conditional bias-variance decomposition against the noise-free
    regression function. Rows of the returned MSPE and ISE matrices follow
    ``METHOD_NAMES``: PLS, APLS, then PCA/FPCR.
    """
    s = _validate_grid(s)
    lam = np.asarray(lam, dtype=float).reshape(-1)
    beta = np.asarray(beta, dtype=float).reshape(-1)
    v = np.asarray(v, dtype=float)
    T = len(s)
    if M < 1 or n < 2 or J < 1 or m < 1 or n_evaluation < 1:
        raise ValueError("M, n, J, m and n_evaluation must be positive")
    if len(lam) != J or v.shape != (J, T) or beta.shape != (T,):
        raise ValueError("lam, beta, v, J and s have incompatible dimensions")
    if np.any(lam < 0) or noise_sd < 0:
        raise ValueError("eigenvalues and noise_sd must be non-negative")
    if rng is None:
        rng = np.random.default_rng()
    if evaluation_rng is None:
        evaluation_rng = np.random.default_rng()

    beta_hat_pls = np.zeros((T, M))
    beta_hat_apls = np.zeros((T, M))
    beta_hat_pca = np.zeros((T, M))
    mspe_pls = np.zeros(M)
    mspe_apls = np.zeros(M)
    mspe_pca = np.zeros(M)
    m_pls = np.zeros(M, dtype=int)
    m_apls = np.zeros(M, dtype=int)
    m_pca = np.zeros(M, dtype=int)
    pls_threshold_reached = np.zeros(M, dtype=bool)
    sqrt_lam = np.sqrt(lam)
    for k in range(M):
        if verbose and (k == 0 or (k + 1) % max(1, M // 10) == 0):
            print(f"Simulation {k + 1}/{M}")
        eps = rng.normal(0.0, noise_sd, n)
        u = rng.normal(size=(n, J))
        X = (sqrt_lam * u) @ v
        y = X @ beta / T + eps
        r = X.T @ y / n
        K = X.T @ X / (n * T)
        try:
            beta_hat_pca[:, k], m_pca[k], _ = pca_gcv(r, K, m)
            beta_pls_m = pls(r, K, m)
            beta_hat_pls[:, k], m_pls[k], moments, threshold = pls_early_stop(
                beta_pls_m, X, y, r, K, tau, delta, n, beta_hat_pca[:, k]
            )
            pls_threshold_reached[k] = bool(np.any(moments <= threshold))
            beta_hat_apls[:, k], m_apls[k], _, _ = apls_cv_stable(
                y, X, K, r, m, k_folds=5, rng=rng
            )
        except (ValueError, LinAlgError, FloatingPointError) as error:
            raise RuntimeError(
                f"estimator failure in Monte Carlo replication {k + 1}"
            ) from error

        u_test = rng.normal(size=(n, J))
        X_test = (sqrt_lam * u_test) @ v
        y_test = X_test @ beta / T + rng.normal(0.0, noise_sd, n)
        mspe_pls[k] = np.mean((y_test - X_test @ beta_hat_pls[:, k] / T) ** 2)
        mspe_apls[k] = np.mean((y_test - X_test @ beta_hat_apls[:, k] / T) ** 2)
        mspe_pca[k] = np.mean((y_test - X_test @ beta_hat_pca[:, k] / T) ** 2)

    # np.mean over the T grid points is already the Riemann-sum factor 1/T.
    mse_pls = np.mean((beta_hat_pls - beta[:, np.newaxis]) ** 2, axis=0)
    mse_apls = np.mean((beta_hat_apls - beta[:, np.newaxis]) ** 2, axis=0)
    mse_pca = np.mean((beta_hat_pca - beta[:, np.newaxis]) ** 2, axis=0)
    mspe_all = np.vstack([mspe_pls, mspe_apls, mspe_pca])
    mse_all = np.vstack([mse_pls, mse_apls, mse_pca])
    beta_estimates = {
        'PLS': beta_hat_pls,
        'APLS': beta_hat_apls,
        'PCA': beta_hat_pca,
    }
    u_evaluation = evaluation_rng.normal(size=(n_evaluation, J))
    X_evaluation = (sqrt_lam * u_evaluation) @ v
    diagnostics = {
        'm_pls': m_pls,
        'm_apls': m_apls,
        'm_pca': m_pca,
        'pls_threshold_reached': pls_threshold_reached,
    }
    return (
        mspe_all,
        mse_all,
        m_pls,
        m_apls,
        beta_estimates,
        X_evaluation,
        diagnostics,
    )


# ============================================================================
# PART 7: MAIN
# ============================================================================

def run_full_simulation_with_bias_variance(
    M: int = 100,
    test_mode: bool = True,
    output_dir: Union[str, Path] = ".",
    seed: int = 2025,
    basis_convention: str = "babii",
    make_plots: bool = True,
):
    """Run all three designs with the settings in the supplied notebook."""
    n, J, T, m = 100, 100, 200, 70
    s = np.linspace(0, 1, T)
    j = np.arange(1, J + 1)
    tau, delta = 1.01, 0.1
    method_names = list(METHOD_NAMES)
    v = create_cosine_basis(s, J, convention=basis_convention)
    b1 = 4 / j ** 2.7
    beta1 = v.T @ b1
    lambda1 = 2 / j ** 1.1
    b2 = b1.copy()
    b2[:5] = 4
    beta2 = v.T @ b2
    lambda3 = lambda1.copy()
    lambda3[:5] = 2
    models = [("Model 1", lambda1, beta1), ("Model 2", lambda1, beta2), ("Model 3", lambda3, beta1)]
    all_results = {}
    M_actual = min(M, 5) if test_mode else M
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    child_seeds = np.random.SeedSequence(seed).spawn(2 * len(models))
    print("=" * 60)
    print("RUNNING SIMULATIONS WITH BIAS-VARIANCE ANALYSIS")
    print("=" * 60)
    print(f"Test mode: {test_mode}, Simulations: {M_actual}")
    for model_index, (model_name, lam, beta) in enumerate(models):
        print(f"\nRunning {model_name}...")
        simulation_rng = np.random.default_rng(child_seeds[2 * model_index])
        evaluation_rng = np.random.default_rng(child_seeds[2 * model_index + 1])
        (
            mspe,
            mse,
            m_pls,
            m_apls,
            beta_estimates,
            X_evaluation,
            diagnostics,
        ) = simulation_cv_with_beta(
            lam, beta, v, s, M_actual, n, J, m,
            tau=tau,
            delta=delta,
            verbose=True,
            noise_sd=1.0,
            n_evaluation=1000,
            rng=simulation_rng,
            evaluation_rng=evaluation_rng,
        )
        print(f"  Avg m_PLS: {np.mean(m_pls):.1f}, Avg m_APLS: {np.mean(m_apls):.1f}")

        # Bias-Variance
        results_beta = {}
        for method in method_names:
            results_beta[method] = compute_bias_variance_beta(beta_estimates[method], beta)
        results_pred = {}
        for method in method_names:
            results_pred[method] = compute_bias_variance_prediction(
                beta_estimates[method],
                X_evaluation,
                beta,
                T,
                noise_variance=1.0,
            )

        if make_plots:
            print("  Generating composite MSE plots...")
            plot_composite_mse_bars(
                results_beta, results_pred, method_names, model_name,
                save_dir=output_dir,
            )

            print("  Generating coefficient distribution plots...")
            plot_coefficient_distributions(
                beta_estimates, v, beta, method_names, model_name,
                J=10, n_coeffs=3, save_dir=output_dir,
            )

            plot_beta_at_points(
                beta_estimates, s, beta, method_names, model_name,
                n_points=5, save_dir=output_dir,
            )

        create_bias_variance_table(results_beta, results_pred, method_names, model_name)

        all_results[model_name] = {
            'mspe': mspe, 'mse': mse,
            'method_names': tuple(method_names),
            'beta_estimates': beta_estimates,
            'results_beta': results_beta,
            'results_pred': results_pred,
            'diagnostics': diagnostics,
        }

    if make_plots:
        print("\nGenerating cross-model comparison plots...")
        model_names = list(all_results.keys())
        plot_coefficient_comparison_across_models(
            all_results, v, method_names, model_names,
            J=10, n_coeffs=3, save_dir=output_dir,
        )
        plot_beta_distribution_across_models(
            all_results, s, method_names, model_names,
            n_points=5, save_dir=output_dir,
        )
        plot_composite_mse_across_models(
            all_results, method_names, save_dir=output_dir
        )
        plot_composite_mse_summary(
            all_results, method_names, save_dir=output_dir
        )

    print("\n" + "=" * 60)
    print("ALL COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    return all_results


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--replications",
        type=int,
        default=5,
        help="Monte Carlo replications per model (use 5000 for the paper design)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="do not cap the run at five smoke-test replications",
    )
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--output-dir", type=Path, default=Path("simulation_output"))
    parser.add_argument(
        "--basis-convention",
        choices=("babii", "standard"),
        default="babii",
        help="'babii' reproduces the basis convention stated in the paper",
    )
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse_arguments()
    print("=" * 60)
    print("FUNCTIONAL DATA ANALYSIS WITH BIAS-VARIANCE DECOMPOSITION")
    print("AND COEFFICIENT DISTRIBUTIONS")
    print("=" * 60)

    results = run_full_simulation_with_bias_variance(
        M=arguments.replications,
        test_mode=not arguments.full,
        output_dir=arguments.output_dir,
        seed=arguments.seed,
        basis_convention=arguments.basis_convention,
        make_plots=not arguments.no_plots,
    )

    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    for model_name, data in results.items():
        mspe = data['mspe']
        mse = data['mse']
        print(f"\n{model_name}:")
        for method, mspe_mean, ise_mean in zip(
            data['method_names'], np.mean(mspe, axis=1), np.mean(mse, axis=1)
        ):
            print(
                f"  {method:<4}  MSPE mean: {mspe_mean:.6f}  "
                f"ISE mean: {ise_mean:.6f}"
            )

    print("\nALL COMPLETED SUCCESSFULLY!")
