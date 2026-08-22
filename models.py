"""
Functional Data Analysis with PLS and Related Methods
=====================================================
A comprehensive package for functional data analysis using:
- PLS (Partial Least Squares)
- APLS (Adaptive PLS)
- Splines with GCV
- PCA with GCV
- RKHS with GCV

Author: Based on Julia code translation
Date: 2025
"""
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import solve, lstsq, eigh, svd
from scipy.linalg import LinAlgError
from scipy.interpolate import splrep, splev
from itertools import product
import warnings
from typing import Tuple, Optional, Dict, List
from math import factorial
from datetime import datetime

# Set seaborn style
sns.set_style("whitegrid")

# ============================================================================
# PART 1: BASIS FUNCTIONS AND SPLINE UTILITIES
# ============================================================================

def create_cosine_basis(s: np.ndarray, J: int) -> np.ndarray:
    """
    Create cosine basis functions.

    Parameters:
    s : ndarray of shape (T,)
        Grid points
    J : int
        Number of basis functions

    Returns:
    v : ndarray of shape (J, T)
        Basis matrix (J basis functions, T grid points)
    """
    T = len(s)
    j = np.arange(1, J + 1)

    # Create basis: v[j, t] = sqrt(2) * cos(pi * s[t] * j)
    v = np.sqrt(2) * np.cos(np.pi * s[:, np.newaxis] * j[np.newaxis, :])
    v[:, 0] = 1  # First column is ones

    return v.T  # Shape: (J, T)

def natural_spline_basis(s: np.ndarray, d: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute natural spline basis functions and their derivatives.
    """
    T = len(s)

    # Create B-spline basis using scipy
    degree = d - 1

    # Create knots with boundary conditions
    t = np.concatenate([
        [s[0]] * (degree + 1),
        s[1:-1],
        [s[-1]] * (degree + 1)
    ])

    # Cox-de Boor recursion for B-spline basis
    def bspline_basis(x, t, degree, i):
        if degree == 0:
            return np.where((t[i] <= x) & (x < t[i+1]), 1.0, 0.0)
        else:
            denom1 = t[i+degree] - t[i]
            denom2 = t[i+degree+1] - t[i+1]

            term1 = np.zeros_like(x)
            term2 = np.zeros_like(x)

            if denom1 != 0:
                term1 = (x - t[i]) / denom1 * bspline_basis(x, t, degree-1, i)
            if denom2 != 0:
                term2 = (t[i+degree+1] - x) / denom2 * bspline_basis(x, t, degree-1, i+1)

            return term1 + term2

    # Evaluate basis functions
    B = np.zeros((T, T))
    dB = np.zeros((T, T))

    # Simplified approach: use sine/cosine basis for demonstration
    for j in range(T):
        B[:, j] = np.sin(j * np.pi * (s - s[0]) / (s[-1] - s[0])) / np.sqrt(T/2)
        dB[:, j] = np.cos(j * np.pi * (s - s[0]) / (s[-1] - s[0])) * \
                   (j * np.pi / (s[-1] - s[0])) / np.sqrt(T/2)

    return B, dB

# ============================================================================
# PART 2: PLS AND RELATED METHODS
# ============================================================================

def pls(r: np.ndarray, K: np.ndarray, m: int) -> np.ndarray:
    """
    PLS (Partial Least Squares) estimator using Conjugate Gradient method.
    """
    T = len(r)

    beta_hat = np.zeros((T, m + 1))
    e = r.copy()
    d = r.copy()
    Ke = K @ e

    for j in range(1, m + 1):
        Kd = K @ d
        alpha1 = e @ Ke
        alpha = alpha1 / (Kd @ Kd)

        beta_hat[:, j] = beta_hat[:, j - 1] + alpha * d
        e = e - alpha * Kd
        Ke = K @ e
        gamma = (e @ Ke) / alpha1
        d = e + gamma * d

    return beta_hat[:, 1:]

def pls_early_stop(beta_hat_pls: np.ndarray, X: np.ndarray, y: np.ndarray,
                   r: np.ndarray, K: np.ndarray, tau: float, delta: float,
                   n: int, beta_hat_pca: np.ndarray) -> Tuple[np.ndarray, int, np.ndarray, float]:
    """
    Early stopping for PLS based on moment criterion.
    """
    T, m = beta_hat_pls.shape

    X_norm = np.mean(np.sum(X**2, axis=1) / T)
    sigma2 = np.mean((y - X @ beta_hat_pca / T)**2)

    moment = np.zeros(m)
    for j in range(m):
        residual = K @ beta_hat_pls[:, j] - r
        moment[j] = np.sqrt(np.sum(residual**2) / T)

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

def apls_cv_stable(y: np.ndarray, X: np.ndarray, K: np.ndarray, r: np.ndarray,
                   m_max: int, k_folds: int = 5, random_state: Optional[int] = None,
                   reg_param: float = 1e-4) -> Tuple[np.ndarray, int, np.ndarray, np.ndarray]:
    """
    Stable APLS with Cross-Validation.
    """
    n, T = X.shape

    # Normalize data
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0) + 1e-10
    X_norm = (X - X_mean) / X_std

    beta_mat = np.zeros((T, m_max))
    CV_errors = np.zeros(m_max)

    if random_state is not None:
        np.random.seed(random_state)

    fold_indices = np.random.permutation(n)
    fold_size = int(np.ceil(n / k_folds))

    K_reg = K + reg_param * np.eye(T)

    for m_val in range(1, m_max + 1):
        fold_errors = np.zeros(k_folds)

        for k in range(k_folds):
            val_start = k * fold_size
            val_end = min((k + 1) * fold_size, n)
            val_idx = fold_indices[val_start:val_end]
            train_idx = np.setdiff1d(np.arange(n), val_idx)

            y_tr = y[train_idx]
            X_tr = X_norm[train_idx, :]
            y_val = y[val_idx]
            X_val = X_norm[val_idx, :]

            n_tr = len(train_idx)

            r_tr = X_tr.T @ y_tr / n_tr
            K_tr = X_tr.T @ X_tr / (T * n_tr) + reg_param * np.eye(T)

            # Build components
            Kr_tr = r_tr.reshape(-1, 1)
            Zm_tr = X_tr @ Kr_tr / T

            for j in range(2, m_val + 1):
                new_Kr = K_tr @ Kr_tr[:, -1] / n_tr
                new_Zm = X_tr @ new_Kr / n_tr
                Kr_tr = np.column_stack([Kr_tr, new_Kr.reshape(-1, 1)])
                Zm_tr = np.column_stack([Zm_tr, new_Zm.reshape(-1, 1)])

            # Solve with regularization
            ZtZ = Zm_tr.T @ Zm_tr
            reg_scale = np.trace(ZtZ) / ZtZ.shape[0]
            ZtZ_reg = ZtZ + reg_param * reg_scale * np.eye(ZtZ.shape[0])

            try:
                alpha = lstsq(ZtZ_reg, Zm_tr.T @ y_tr, rcond=1e-6)[0]
            except:
                alpha = np.linalg.pinv(ZtZ_reg, rcond=1e-6) @ (Zm_tr.T @ y_tr)

            beta_current = Kr_tr @ alpha
            beta_mat[:, m_val-1] = beta_current.flatten()

            y_pred = X_val @ beta_current
            fold_errors[k] = np.mean((y_val - y_pred.flatten())**2)

        CV_errors[m_val-1] = np.mean(fold_errors)

    if np.any(np.isnan(CV_errors)) or np.any(np.isinf(CV_errors)):
        warnings.warn("Numerical issues detected. Using first component.")
        m_opt = 1
    else:
        m_opt = np.argmin(CV_errors) + 1

    beta_opt = beta_mat[:, m_opt-1] / X_std

    return beta_opt, m_opt, CV_errors, beta_mat

# ============================================================================
# PART 3: SPLINE METHODS
# ============================================================================

def spline_gcv(y: np.ndarray, X: np.ndarray, s: np.ndarray,
               rho: float, d: int) -> float:
    """
    Compute Generalized Cross-Validation score for spline estimator.
    """
    n, T = X.shape

    B, dB = natural_spline_basis(s, d)

    BtB = B.T @ B
    BtB_inv = solve(BtB + 1e-10 * np.eye(T), np.eye(T))

    P = B @ BtB_inv @ B.T
    Dm = dB.T @ dB

    S = np.column_stack([s ** i for i in range(2 * d)])
    StS = S.T @ S
    StS_inv = solve(StS + 1e-50 * np.eye(StS.shape[0]), np.eye(StS.shape[0]))
    Am = (S @ StS_inv @ S.T) / T + (B @ BtB_inv @ Dm @ BtB_inv @ B.T) / T

    XtX = X.T @ X
    Xt = X.T

    M = XtX / (n * T**2) + rho * Am / T

    try:
        Z = solve(M + 1e-10 * np.eye(T), Xt, assume_a='pos')
        Hrho = X @ P @ Z / (n * T**2)
    except:
        M_pinv = np.linalg.pinv(M + 1e-10 * np.eye(T))
        Hrho = X @ P @ M_pinv @ Xt / (n * T**2)

    residuals = y - Hrho @ y
    gcv = np.sum(residuals**2) / n / (1 - np.trace(Hrho) / n)**2

    return gcv

def spline(y: np.ndarray, X: np.ndarray, s: np.ndarray,
           rho_grid: np.ndarray, d: int) -> np.ndarray:
    """
    Spline estimator for given regularization parameters.
    """
    n, T = X.shape

    B, dB = natural_spline_basis(s, d)

    BtB = B.T @ B
    BtB_inv = solve(BtB + 1e-10 * np.eye(T), np.eye(T))

    P = B @ BtB_inv @ B.T
    Dm = dB.T @ dB

    S = np.column_stack([s ** i for i in range(2 * d)])
    StS = S.T @ S
    StS_inv = solve(StS + 1e-50 * np.eye(StS.shape[0]), np.eye(StS.shape[0]))
    Am = (S @ StS_inv @ S.T) / T + (B @ BtB_inv @ Dm @ BtB_inv @ B.T) / T

    beta_hat = np.zeros((T, len(rho_grid)))
    XtX = X.T @ X
    XtY = X.T @ y

    for i, rho in enumerate(rho_grid):
        M = XtX / (n * T) + rho * Am
        try:
            M_inv = solve(M + 1e-10 * np.eye(T), np.eye(T))
            beta_hat[:, i] = P @ M_inv @ (XtY / n)
        except:
            M_pinv = np.linalg.pinv(M + 1e-10 * np.eye(T))
            beta_hat[:, i] = P @ M_pinv @ (XtY / n)

    return beta_hat

# ============================================================================
# PART 4: PCA AND RKHS METHODS
# ============================================================================

def pca_gcv(r: np.ndarray, K: np.ndarray, m: int) -> Tuple[np.ndarray, int, np.ndarray]:
    """
    PCA estimator with GCV-based component selection.
    """
    T = len(r)

    beta_hat = np.zeros((T, m))
    GCV = np.zeros(m)

    try:
        evals, evecs = eigh(K)
    except np.linalg.LinAlgError:
        warnings.warn("Matrix K is not positive definite, using SVD instead")
        U, s, Vt = np.linalg.svd(K)
        evals = s
        evecs = U

    for j in range(1, m + 1):
        lambda_hat = evals[-j:]
        v_hat = evecs[:, -j:]

        beta_hat[:, j-1] = v_hat @ ((v_hat.T @ r) / lambda_hat)

        residuals = r - K @ beta_hat[:, j-1]
        RSS = np.sum(residuals**2)
        GCV[j-1] = (RSS / T) / (1 - j / T)**2

    m_opt = np.argmin(GCV) + 1

    return beta_hat[:, m_opt-1], m_opt, GCV

def rkhs_gcv(y: np.ndarray, X: np.ndarray, lambda_grid: np.ndarray,
             s: np.ndarray) -> np.ndarray:
    """
    RKHS estimator with GCV.
    """
    n, T = X.shape

    beta_matrix = np.zeros((T, len(lambda_grid)))
    GCV = np.zeros(len(lambda_grid))

    # Compute kernel matrix
    Delta = s[:, np.newaxis] - s[np.newaxis, :]
    B2_s = s**2 - s + 1/6
    B2_t = s**2 - s + 1/6
    abs_Delta = np.abs(Delta)
    B4_st = abs_Delta**4 - 2*abs_Delta**3 + abs_Delta**2 - 1/30

    Km = (1/factorial(2)**2) * np.outer(B2_s, B2_t) - (1/factorial(4)) * B4_st

    Sigma = X @ Km @ X.T / T**2

    T_matrix = np.zeros((n, 2))
    T_matrix[:, 0] = np.sum(X, axis=1) / T
    T_matrix[:, 1] = X @ s / T

    XKm = X @ Km

    for j, lam in enumerate(lambda_grid):
        W = Sigma + n * lam * np.eye(n)

        try:
            W_inv_T = solve(W, T_matrix, assume_a='pos')
            W_inv_y = solve(W, y, assume_a='pos')

            d = solve(T_matrix.T @ W_inv_T, T_matrix.T @ W_inv_y, assume_a='pos')
            c = W_inv_y - W_inv_T @ d
            beta_vals = d[0] + d[1] * s + (c @ XKm / T).T

        except:
            W_pinv = np.linalg.pinv(W)
            W_inv_T = W_pinv @ T_matrix
            W_inv_y = W_pinv @ y
            d = np.linalg.lstsq(T_matrix.T @ W_inv_T, T_matrix.T @ W_inv_y, rcond=None)[0]
            c = W_inv_y - W_inv_T @ d
            beta_vals = d[0] + d[1] * s + (c @ XKm / T).T

        residuals = y - X @ beta_vals / T
        RSS = np.sum(residuals**2)

        try:
            W_inv_Sigma = solve(W, Sigma, assume_a='pos')
            trace_term = np.trace(W_inv_Sigma)
        except:
            W_inv_Sigma = np.linalg.pinv(W) @ Sigma
            trace_term = np.trace(W_inv_Sigma)

        GCV[j] = (RSS / n) / (1 - (2 + trace_term) / n)**2
        beta_matrix[:, j] = beta_vals.flatten()

    j_opt = np.argmin(GCV)
    return beta_matrix[:, j_opt]

# ============================================================================
# PART 5: SIMULATION FUNCTIONS
# ============================================================================

def simulation_cv(lam: np.ndarray, beta: np.ndarray, v: np.ndarray,
                  s: np.ndarray, M: int, n: int, J: int, m: int,
                  param_grid: List[Tuple], lam_grid: np.ndarray,
                  tau: float = 1.5, delta: float = 2.0,
                  d: int = 4, verbose: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulation study with cross-validation for tuning parameter selection.
    """
    T = len(s)

    # Initialize storage
    beta_hat_pls = np.zeros((T, M))
    beta_hat_apls = np.zeros((T, M))
    beta_hat_spl = np.zeros((T, M))
    beta_hat_rkhs = np.zeros((T, M))
    beta_hat_pca = np.zeros((T, M))

    mspe_pls = np.zeros(M)
    mspe_apls = np.zeros(M)
    mspe_spl = np.zeros(M)
    mspe_rkhs = np.zeros(M)
    mspe_pca = np.zeros(M)

    m_pls = np.zeros(M, dtype=int)
    m_apls = np.zeros(M, dtype=int)

    sqrt_lam = np.sqrt(lam)

    for k in range(M):
        if verbose and k % 10 == 0:
            print(f"Simulation {k+1}/{M}")

        # Generate data
        eps = np.random.normal(0, 1, n)
        u = np.random.normal(0, 1, (n, J))
        X = (sqrt_lam * u) @ v
        y = X @ beta / T + eps

        r = X.T @ y / n
        K = X.T @ X / (n * T)

        # PCA with GCV
        beta_hat_pca[:, k], m_pca, gcv_scores = pca_gcv(r, K, m)

        # PLS with early stopping
        beta_pls_m = pls(r, K, m)
        beta_hat_pls[:, k], m_pls[k], moment, threshold = pls_early_stop(
            beta_pls_m, X, y, r, K, tau, delta, n, beta_hat_pca[:, k]
        )

        # APLS with CV
        beta_hat_apls[:, k], m_apls[k], cv_errors, beta_mat = apls_cv_stable(
            y, X, K, r, m, k_folds=5, random_state=None, reg_param=1e-4
        )

        # Spline with GCV
        gcv_vals = []
        for rho, d_val in param_grid:
            gcv_val = spline_gcv(y, X, s, rho, d_val)
            gcv_vals.append(gcv_val)

        best_idx = np.argmin(gcv_vals)
        rho_opt, d_opt = param_grid[best_idx]
        beta_hat_spl[:, k] = spline(y, X, s, [rho_opt], d_opt).flatten()

        # RKHS with GCV
        beta_hat_rkhs[:, k] = rkhs_gcv(y, X, lam_grid, s)

        # Generate test data
        u_test = np.random.normal(0, 1, (n, J))
        X_test = (sqrt_lam * u_test) @ v
        y_test = X_test @ beta / T + np.random.normal(0, 1, n)

        # Compute MSPE for each method
        mspe_pls[k] = np.mean((y_test - X_test @ beta_hat_pls[:, k] / T)**2)
        mspe_apls[k] = np.mean((y_test - X_test @ beta_hat_apls[:, k] / T)**2)
        mspe_spl[k] = np.mean((y_test - X_test @ beta_hat_spl[:, k] / T)**2)
        mspe_pca[k] = np.mean((y_test - X_test @ beta_hat_pca[:, k] / T)**2)
        mspe_rkhs[k] = np.mean((y_test - X_test @ beta_hat_rkhs[:, k] / T)**2)

    # Compute MSE
    mse_pls = np.mean((beta_hat_pls - beta[:, np.newaxis])**2, axis=0) / T
    mse_apls = np.mean((beta_hat_apls - beta[:, np.newaxis])**2, axis=0) / T
    mse_spl = np.mean((beta_hat_spl - beta[:, np.newaxis])**2, axis=0) / T
    mse_pca = np.mean((beta_hat_pca - beta[:, np.newaxis])**2, axis=0) / T
    mse_rkhs = np.mean((beta_hat_rkhs - beta[:, np.newaxis])**2, axis=0) / T

    mspe_all = np.vstack([mspe_pls, mspe_apls, mspe_spl, mspe_pca, mspe_rkhs])
    mse_all = np.vstack([mse_pls, mse_apls, mse_spl, mse_pca, mse_rkhs])

    return mspe_all, mse_all, m_pls, m_apls

# ============================================================================
# PART 6: VISUALIZATION FUNCTIONS (FIXED)
# ============================================================================

def plot_results(mspe: np.ndarray, mse: np.ndarray, labels: List[str],
                 title_suffix: str = "", save_path: Optional[str] = None):
    """
    Create boxplots for MSPE and MSE results.
    """
    mspe_data = [mspe[i, :] for i in range(mspe.shape[0])]
    mse_data = [mse[i, :] for i in range(mse.shape[0])]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # MSPE
    bp1 = ax1.boxplot(mspe_data, patch_artist=True, showfliers=False)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("MSPE", fontsize=12)
    ax1.set_title(f"MSPE {title_suffix}", fontsize=14)
    ax1.grid(True, alpha=0.3)

    # MSE
    bp2 = ax2.boxplot(mse_data, patch_artist=True, showfliers=False)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("ISE", fontsize=12)
    ax2.set_title(f"MSE {title_suffix}", fontsize=14)
    ax2.grid(True, alpha=0.3)

    # Color boxes
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    for bp, color in [(bp1, colors), (bp2, colors)]:
        for patch, c in zip(bp['boxes'], color):
            patch.set_facecolor(c)
            patch.set_alpha(0.7)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig

# ============================================================================
# PART 7: MAIN SIMULATION SCRIPT
# ============================================================================

def run_full_simulation(M: int = 5000, test_mode: bool = True):
    """
    Run the complete simulation for Figure 1 with three models.

    Parameters:
    M : int
        Number of simulations
    test_mode : bool
        If True, use reduced M for testing
    """

    # Set random seed
    np.random.seed(2025)

    # Parameters
    n = 100
    J = 100
    T = 200
    m = 70

    # Grids
    s = np.linspace(0, 1, T)
    j = np.arange(1, J + 1)

    # Early stopping parameters
    tau = 1.01
    delta = 0.1

    # Tuning grids
    rho_grid = np.logspace(-12, -1, 10)
    d_grid = np.arange(1, 5)
    param_grid = [(rho, d) for d in d_grid for rho in rho_grid]
    lam_grid = np.logspace(-6, 0, 10)

    group_labels = ["PLS", "APLS", "Splines", "PCA", "RKHS"]

    # Create basis
    v = np.sqrt(2) * np.cos(np.pi * s[:, np.newaxis] * j[np.newaxis, :])
    v[:, 0] = 1
    v = v.T  # Shape: (J, T)

    # Model definitions
    b1 = 4 / j**2.7
    beta1 = v.T @ b1
    lambda1 = 2 / j**1.1

    b2 = b1.copy()
    b2[:5] = 4
    beta2 = v.T @ b2

    lambda3 = lambda1.copy()
    lambda3[:5] = 2

    models = [
        ("Model 1", lambda1, beta1),
        ("Model 2", lambda1, beta2),
        ("Model 3", lambda3, beta1)
    ]

    # Run simulations
    results = {}
    M_actual = min(M, 100) if test_mode else M

    print("="*60)
    print("RUNNING SIMULATIONS")
    print("="*60)
    print(f"Test mode: {test_mode}")
    print(f"Simulations: {M_actual}")
    print(f"Sample size: {n}")
    print(f"Grid points: {T}")
    print("="*60)

    for idx, (model_name, lam, beta) in enumerate(models, 1):
        print(f"\nRunning {model_name}...")

        start_time = datetime.now()

        mspe, mse, m_pls, m_apls = simulation_cv(
            lam, beta, v, s, M_actual, n, J, m,
            param_grid, lam_grid,
            tau=tau, delta=delta, d=4, verbose=True
        )

        elapsed = (datetime.now() - start_time).total_seconds()

        print(f"\n{model_name} completed in {elapsed:.2f}s")
        print(f"Avg MSPE: {np.mean(mspe, axis=1)}")
        print(f"Avg MSE: {np.mean(mse, axis=1)}")
        print(f"Avg m_PLS: {np.mean(m_pls):.1f}")
        print(f"Avg m_APLS: {np.mean(m_apls):.1f}")

        results[model_name] = {
            'mspe': mspe, 'mse': mse,
            'm_pls': m_pls, 'm_apls': m_apls
        }

        # Create plots
        fig = plot_results(mspe, mse, group_labels, f"({model_name})")
        plt.suptitle(f"{model_name}: Estimation and Prediction", fontsize=16)
#        plt.show()

        try:
            fig.savefig(f"model{idx}_results.pdf", dpi=300, bbox_inches='tight')
        except:
            pass

    return results


# ============================================================================
# PART 8: ASYMPTOTIC APPROXIMATION ANALYSIS
# ============================================================================

def simulation_inference(lam: np.ndarray, beta: np.ndarray, v: np.ndarray,
                         M: int, n: int, J: int, T: int, m: int,
                         verbose: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulation for inference (power and size).

    Parameters:
    lam : ndarray of shape (J,)
        Eigenvalues
    beta : ndarray of shape (T,)
        True coefficients
    v : ndarray of shape (J, T)
        Eigenvectors
    M : int
        Number of simulations
    n : int
        Sample size
    J : int
        Number of factors
    T : int
        Number of grid points
    m : int
        Number of PLS components
    verbose : bool
        If True, print progress

    Returns:
    Tn_H0 : ndarray of shape (M,)
        Test statistics under H0
    D : ndarray of shape (M,)
        Random effects (asymptotic distribution)
    """
    Tn_H0 = np.zeros(M)
    D = np.zeros(M)

    sqrt_lam = np.sqrt(lam)

    for k in range(M):
        if verbose and k % 10 == 0:
            print(f"Inference simulation {k + 1}/{M}")

        # Generate data under H0
        eps = np.random.normal(0, 1, n)
        u = np.random.normal(0, 1, (n, J))
        X = (sqrt_lam * u) @ v
        y = X @ beta / T + eps

        # Compute statistics
        r = X.T @ y / n
        K = X.T @ X / (T * n)

        # PLS estimate
        beta_pls = pls(r, K, m)[:, -1]

        # Test statistic under H0
        diff = K @ (beta_pls - beta)
        Tn_H0[k] = n * np.sum(diff ** 2) / T

        # Random effect for asymptotic distribution
        D[k] = lam @ (np.random.normal(0, 1, J) ** 2)

    return Tn_H0, D


def plot_asymptotic_analysis(Tn_H0: np.ndarray, D: np.ndarray,
                             model_name: str, save_dir: str = "."):
    """
    Create histogram comparison and Q-Q plot for asymptotic analysis.

    Parameters:
    Tn_H0 : ndarray of shape (M,)
        Test statistics under H0
    D : ndarray of shape (M,)
        Random effects (asymptotic distribution)
    model_name : str
        Name of the model for labeling
    save_dir : str
        Directory to save figures
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Remove NaN values if any
    Tn_clean = Tn_H0[~np.isnan(Tn_H0)]
    D_clean = D[~np.isnan(D)]

    # Plot 1: Histogram comparison
    bins = 50

    # Histogram of exact distribution
    ax1.hist(Tn_clean, bins=bins, density=True, alpha=0.5,
             color='blue', label='Exact Distribution of Tn')

    # Histogram of asymptotic distribution
    ax1.hist(D_clean, bins=bins, density=True, alpha=0.5,
             color='red', label='Asymptotic Distribution of T')

    ax1.set_xlabel('Value', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.set_title(f'{model_name}: Distribution Comparison', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(labelsize=12)

    # Plot 2: Q-Q plot
    # Compute quantiles
    n_quantiles = min(len(Tn_clean), len(D_clean), 1000)
    quantiles_exact = np.percentile(Tn_clean, np.linspace(0, 100, n_quantiles))
    quantiles_asym = np.percentile(D_clean, np.linspace(0, 100, n_quantiles))

    # Q-Q plot
    ax2.scatter(quantiles_exact, quantiles_asym, alpha=0.5, s=20, color='blue')

    # Add diagonal line
    min_val = min(quantiles_exact.min(), quantiles_asym.min())
    max_val = max(quantiles_exact.max(), quantiles_asym.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2,
             label='y = x')

    ax2.set_xlabel('Exact Tₙ Quantiles', fontsize=12)
    ax2.set_ylabel('Asymptotic T Quantiles', fontsize=12)
    ax2.set_title(f'{model_name}: Q-Q Plot', fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(labelsize=12)

    plt.tight_layout()

    # Save figures
    try:
        # Save combined figure
        fig.savefig(f"{save_dir}/asymptotic_analysis_{model_name.lower().replace(' ', '_')}.pdf",
                    dpi=300, bbox_inches='tight')

        # Also save individual figures for compatibility with original code
        fig_hist, ax_hist = plt.subplots(1, 1, figsize=(10, 6))
        ax_hist.hist(Tn_clean, bins=bins, density=True, alpha=0.5,
                     color='blue', label='Exact Distribution of Tn')
        ax_hist.hist(D_clean, bins=bins, density=True, alpha=0.5,
                     color='red', label='Asymptotic Distribution of T')
        ax_hist.set_xlabel('Value', fontsize=12)
        ax_hist.set_ylabel('Density', fontsize=12)
        ax_hist.set_title(f'{model_name}: Distribution Comparison', fontsize=14)
        ax_hist.legend(fontsize=11)
        ax_hist.grid(True, alpha=0.3)
        ax_hist.tick_params(labelsize=12)
        plt.tight_layout()
        fig_hist.savefig(f"{save_dir}/asymptotic_distribution_{model_name.lower().replace(' ', '_')}.pdf",
                         dpi=300, bbox_inches='tight')
        plt.close(fig_hist)

        fig_qq, ax_qq = plt.subplots(1, 1, figsize=(10, 6))
        ax_qq.scatter(quantiles_exact, quantiles_asym, alpha=0.5, s=20, color='blue')
        min_val = min(quantiles_exact.min(), quantiles_asym.min())
        max_val = max(quantiles_exact.max(), quantiles_asym.max())
        ax_qq.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2,
                   label='y = x')
        ax_qq.set_xlabel('Exact Tn Quantiles', fontsize=12)
        ax_qq.set_ylabel('Asymptotic T Quantiles', fontsize=12)
        ax_qq.set_title(f'{model_name}: Q-Q Plot', fontsize=14)
        ax_qq.legend(fontsize=11)
        ax_qq.grid(True, alpha=0.3)
        ax_qq.tick_params(labelsize=12)
        plt.tight_layout()
        fig_qq.savefig(f"{save_dir}/qqplot_{model_name.lower().replace(' ', '_')}.pdf",
                       dpi=300, bbox_inches='tight')
        plt.close(fig_qq)

        print(f"Saved figures for {model_name}")
    except Exception as e:
        print(f"Could not save figures for {model_name}: {e}")

#    plt.show()

    return fig


# ============================================================================
# PART 9: RUN ASYMPTOTIC ANALYSIS
# ============================================================================

def run_asymptotic_analysis(M: int = 1000, test_mode: bool = True):
    """
    Run the asymptotic approximation analysis for all three models.

    Parameters:
    M : int
        Number of simulations
    test_mode : bool
        If True, use reduced M for testing
    """
    print("\n" + "=" * 60)
    print("ASYMPTOTIC APPROXIMATION ANALYSIS")
    print("=" * 60)

    # Parameters
    n = 100
    J = 100
    T = 200
    m = 70

    # Grids
    s = np.linspace(0, 1, T)
    j = np.arange(1, J + 1)

    # Create basis
    v = np.sqrt(2) * np.cos(np.pi * s[:, np.newaxis] * j[np.newaxis, :])
    v[:, 0] = 1
    v = v.T  # Shape: (J, T)

    # Model definitions (same as before)
    b1 = 4 / j ** 2.7
    beta1 = v.T @ b1
    lambda1 = 2 / j ** 1.1

    b2 = b1.copy()
    b2[:5] = 4
    beta2 = v.T @ b2

    lambda3 = lambda1.copy()
    lambda3[:5] = 2

    models = [
        ("Model 1", lambda1, beta1),
        ("Model 2", lambda1, beta2),
        ("Model 3", lambda3, beta1)
    ]

    # Run simulations
    results = {}
    M_actual = min(M, 100) if test_mode else M

    print(f"Test mode: {test_mode}")
    print(f"Simulations: {M_actual}")
    print("=" * 60)

    for idx, (model_name, lam, beta) in enumerate(models, 1):
        print(f"\nRunning {model_name} for asymptotic analysis...")

        start_time = datetime.now()

        # Run inference simulation
        Tn_H0, D = simulation_inference(
            lam, beta, v, M_actual, n, J, T, m, verbose=True
        )

        elapsed = (datetime.now() - start_time).total_seconds()

        print(f"{model_name} completed in {elapsed:.2f}s")
        print(f"Mean Tn_H0: {np.mean(Tn_H0):.4f}")
        print(f"Mean D: {np.mean(D):.4f}")

        results[model_name] = {
            'Tn_H0': Tn_H0,
            'D': D
        }

        # Create plots
        fig = plot_asymptotic_analysis(Tn_H0, D, model_name)

    return results


# ============================================================================
# PART 11: POWER CURVE ANALYSIS
# ============================================================================

def power_curve(beta: np.ndarray, lam: np.ndarray, v: np.ndarray, s: np.ndarray,
                M: int = 1000, n1: int = 100, n2: int = 200,
                J: int = 100, T: int = 200, m: int = 70,
                n_sims: int = 100, verbose: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute power curve for the test.

    Parameters:
    beta : ndarray of shape (T,)
        True coefficients
    lam : ndarray of shape (J,)
        Eigenvalues
    v : ndarray of shape (J, T)
        Eigenvectors
    s : ndarray of shape (T,)
        Grid points
    M : int
        Number of simulations for critical value
    n1, n2 : int
        Sample sizes
    J : int
        Number of factors
    T : int
        Number of grid points
    m : int
        Number of PLS components
    n_sims : int
        Number of simulations for power (can be smaller than M)
    verbose : bool
        If True, print progress

    Returns:
    delta_vals : ndarray
        Values of delta
    power100 : ndarray
        Power for n=100
    power200 : ndarray
        Power for n=200
    """
    # Delta values
    delta_vals = np.arange(-1, 1.05, 0.05)

    # Pre-compute sqrt(lam)
    sqrt_lam = np.sqrt(lam)

    print(f"\nComputing power curve...")
    print(f"Delta range: [{delta_vals[0]:.2f}, {delta_vals[-1]:.2f}]")
    print(f"Number of delta values: {len(delta_vals)}")

    # ============================================
    # Critical value for n = 100
    # ============================================
    print(f"\nComputing critical value for n={n1}...")
    Tn_H0 = np.zeros(M)

    for k in range(M):
        if verbose and k % 100 == 0:
            print(f"  Critical value simulation {k + 1}/{M}")

        # Generate data under H0
        eps = np.random.normal(0, 1, n1)
        u = np.random.normal(0, 1, (n1, J))
        X = (sqrt_lam * u) @ v
        y0 = X @ beta / T + eps

        # Compute statistics
        r0 = X.T @ y0 / n1
        K = X.T @ X / (T * n1)
        beta_pls = pls(r0, K, m)[:, -1]

        # Test statistic
        diff = K @ (beta_pls - beta)
        Tn_H0[k] = n1 * np.sum(diff ** 2) / T

    # Remove NaN values
    Tn_H0_clean = Tn_H0[~np.isnan(Tn_H0)]
    if len(Tn_H0_clean) == 0:
        warnings.warn("All Tn_H0 values are NaN for n=100. Using default critical value.")
        crit_val_100 = 1.0
    else:
        crit_val_100 = np.quantile(Tn_H0_clean, 0.95)
    print(f"Critical value for n={n1}: {crit_val_100:.4f}")

    # ============================================
    # Power for n = 100
    # ============================================
    print(f"\nComputing power for n={n1}...")
    power100 = np.zeros(len(delta_vals))

    for i, delta in enumerate(delta_vals):
        if verbose:
            print(f"  delta = {delta:.2f}")

        Tn_H1 = np.zeros(n_sims)
        for k in range(n_sims):
            # Generate data under H1
            eps = np.random.normal(0, 1, n1)
            u = np.random.normal(0, 1, (n1, J))
            X = (sqrt_lam * u) @ v
            y1 = X @ (beta + delta * s) / T + eps

            # Compute statistics
            r1 = X.T @ y1 / n1
            K = X.T @ X / (T * n1)
            beta_pls = pls(r1, K, m)[:, -1]

            # Test statistic
            diff = K @ (beta_pls - beta)
            Tn_H1[k] = n1 * np.sum(diff ** 2) / T

        # Remove NaN values
        Tn_H1_clean = Tn_H1[~np.isnan(Tn_H1)]
        if len(Tn_H1_clean) > 0:
            power100[i] = np.mean(Tn_H1_clean > crit_val_100)
        else:
            power100[i] = np.nan

    # ============================================
    # Critical value for n = 200
    # ============================================
    print(f"\nComputing critical value for n={n2}...")
    Tn_H0 = np.zeros(M)

    for k in range(M):
        if verbose and k % 100 == 0:
            print(f"  Critical value simulation {k + 1}/{M}")

        eps = np.random.normal(0, 1, n2)
        u = np.random.normal(0, 1, (n2, J))
        X = (sqrt_lam * u) @ v
        y0 = X @ beta / T + eps

        r0 = X.T @ y0 / n2
        K = X.T @ X / (T * n2)
        beta_pls = pls(r0, K, m)[:, -1]

        diff = K @ (beta_pls - beta)
        Tn_H0[k] = n2 * np.sum(diff ** 2) / T

    Tn_H0_clean = Tn_H0[~np.isnan(Tn_H0)]
    if len(Tn_H0_clean) == 0:
        warnings.warn("All Tn_H0 values are NaN for n=200. Using default critical value.")
        crit_val_200 = 1.0
    else:
        crit_val_200 = np.quantile(Tn_H0_clean, 0.95)
    print(f"Critical value for n={n2}: {crit_val_200:.4f}")

    # ============================================
    # Power for n = 200
    # ============================================
    print(f"\nComputing power for n={n2}...")
    power200 = np.zeros(len(delta_vals))

    for i, delta in enumerate(delta_vals):
        if verbose:
            print(f"  delta = {delta:.2f}")

        Tn_H1 = np.zeros(n_sims)
        for k in range(n_sims):
            eps = np.random.normal(0, 1, n2)
            u = np.random.normal(0, 1, (n2, J))
            X = (sqrt_lam * u) @ v
            y1 = X @ (beta + delta * s) / T + eps

            r1 = X.T @ y1 / n2
            K = X.T @ X / (T * n2)
            beta_pls = pls(r1, K, m)[:, -1]

            diff = K @ (beta_pls - beta)
            Tn_H1[k] = n2 * np.sum(diff ** 2) / T

        Tn_H1_clean = Tn_H1[~np.isnan(Tn_H1)]
        if len(Tn_H1_clean) > 0:
            power200[i] = np.mean(Tn_H1_clean > crit_val_200)
        else:
            power200[i] = np.nan

    return delta_vals, power100, power200


def plot_power_curve(delta_vals: np.ndarray, power100: np.ndarray,
                     power200: np.ndarray, model_name: str,
                     save_dir: str = ".") -> plt.Figure:
    """
    Plot power curve.

    Parameters:
    delta_vals : ndarray
        Values of delta
    power100 : ndarray
        Power for n=100
    power200 : ndarray
        Power for n=200
    model_name : str
        Name of the model for labeling
    save_dir : str
        Directory to save figures

    Returns:
    fig : matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    # Plot power curves
    ax.plot(delta_vals, power100, 'b-', linewidth=2.5, label='n=100')
    ax.plot(delta_vals, power200, 'r--', linewidth=2.5, label='n=200')

    # Add horizontal line at 0.05 (nominal level)
    ax.axhline(y=0.05, color='k', linestyle=':', linewidth=2, label='Nominal 5%')

    # Add vertical line at delta=0
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    ax.set_xlabel('Scale Factor δ', fontsize=14)
    ax.set_ylabel('Empirical Rejection Probability', fontsize=14)
    ax.set_title(f'Power Curve - {model_name}', fontsize=16)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(delta_vals[0] - 0.1, delta_vals[-1] + 0.1)
    ax.legend(loc='best', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=12)

    plt.tight_layout()

    # Save figure
    try:
        filename = f"{save_dir}/power_curve_{model_name.lower().replace(' ', '_')}.pdf"
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved power curve for {model_name}")
    except Exception as e:
        print(f"Could not save power curve for {model_name}: {e}")

    return fig


# ============================================================================
# PART 12: RUN POWER CURVE ANALYSIS
# ============================================================================

def run_power_curve_analysis(M: int = 1000, test_mode: bool = True):
    """
    Run power curve analysis for all three models.

    Parameters:
    M : int
        Number of simulations for critical value
    test_mode : bool
        If True, use reduced parameters for testing
    """
    print("\n" + "=" * 60)
    print("POWER CURVE ANALYSIS (FIGURE 3)")
    print("=" * 60)

    # Parameters
    n1 = 100
    n2 = 200
    J = 100
    T = 200
    m = 70

    # Grids
    s = np.linspace(0, 1, T)
    j = np.arange(1, J + 1)

    # Create basis
    v = np.sqrt(2) * np.cos(np.pi * s[:, np.newaxis] * j[np.newaxis, :])
    v[:, 0] = 1
    v = v.T  # Shape: (J, T)

    # Model definitions
    b1 = 4 / j ** 2.7
    beta1 = v.T @ b1
    lambda1 = 2 / j ** 1.1

    b2 = b1.copy()
    b2[:5] = 4
    beta2 = v.T @ b2

    lambda3 = lambda1.copy()
    lambda3[:5] = 2

    models = [
        ("Model 1", beta1, lambda1),
        ("Model 2", beta2, lambda1),
        ("Model 3", beta1, lambda3)
    ]

    # Adjust parameters for test mode
    if test_mode:
        M_actual = min(M, 100)  # Fewer simulations for critical value
        n_sims = 50  # Fewer simulations for power
        print("TEST MODE: Using reduced parameters")
    else:
        M_actual = M
        n_sims = 200  # More simulations for accurate power

    print(f"Critical value simulations: {M_actual}")
    print(f"Power simulations per delta: {n_sims}")
    print("=" * 60)

    results = {}

    for idx, (model_name, beta, lam) in enumerate(models, 1):
        print(f"\n{'=' * 50}")
        print(f"Running {model_name}")
        print(f"{'=' * 50}")

        start_time = datetime.now()

        # Compute power curve
        delta_vals, power100, power200 = power_curve(
            beta, lam, v, s,
            M=M_actual,
            n1=n1, n2=n2,
            J=J, T=T, m=m,
            n_sims=n_sims,
            verbose=False
        )

        elapsed = (datetime.now() - start_time).total_seconds()

        print(f"\n{model_name} completed in {elapsed:.2f}s")
        print(f"Max power (n=100): {np.nanmax(power100):.4f}")
        print(f"Max power (n=200): {np.nanmax(power200):.4f}")

        # Store results
        results[model_name] = {
            'delta_vals': delta_vals,
            'power100': power100,
            'power200': power200
        }

        # Create plot
        fig = plot_power_curve(delta_vals, power100, power200, model_name)
#        plt.show()

    return results


# ============================================================================
# PART 13: UPDATED MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Suppress warnings (optional)
    import warnings

    warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

    print("=" * 60)
    print("FUNCTIONAL DATA ANALYSIS WITH PLS")
    print("=" * 60)

    # Figure 1: Estimation and Prediction Accuracy
    print("\n" + "=" * 60)
    print("FIGURE 1: ESTIMATION AND PREDICTION ACCURACY")
    print("=" * 60)

    results_fig1 = run_full_simulation(M=5000, test_mode=False)

    # Figure 2: Accuracy of Asymptotic Approximation
    print("\n" + "=" * 60)
    print("FIGURE 2: ACCURACY OF ASYMPTOTIC APPROXIMATION")
    print("=" * 60)

    results_fig2 = run_asymptotic_analysis(M=1000, test_mode=False)

    # Figure 3: Power Curves
    print("\n" + "=" * 60)
    print("FIGURE 3: POWER CURVES")
    print("=" * 60)

    results_fig3 = run_power_curve_analysis(M=1000, test_mode=False)

    # Print all summaries
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS - FIGURE 1")
    print("=" * 60)
    for model_name, data in results_fig1.items():
        mspe = data['mspe']
        mse = data['mse']
        print(f"\n{model_name}:")
        print(f"  MSPE mean: {np.mean(mspe, axis=1)}")
        print(f"  MSE mean:  {np.mean(mse, axis=1)}")
        print(f"  Avg m_PLS: {np.mean(data['m_pls']):.1f}")
        print(f"  Avg m_APLS: {np.mean(data['m_apls']):.1f}")

    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS - FIGURE 2")
    print("=" * 60)
    for model_name, data in results_fig2.items():
        Tn_H0 = data['Tn_H0']
        D = data['D']
        Tn_clean = Tn_H0[~np.isnan(Tn_H0)]
        D_clean = D[~np.isnan(D)]
        from scipy.stats import ks_2samp

        ks_stat, ks_pval = ks_2samp(Tn_clean, D_clean)
        print(f"\n{model_name}:")
        print(f"  Exact Mean: {np.mean(Tn_clean):.4f}, Std: {np.std(Tn_clean):.4f}")
        print(f"  Asymp Mean: {np.mean(D_clean):.4f}, Std: {np.std(D_clean):.4f}")
        print(f"  KS test: stat={ks_stat:.4f}, p={ks_pval:.4f}")

    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS - FIGURE 3")
    print("=" * 60)
    for model_name, data in results_fig3.items():
        power100 = data['power100']
        power200 = data['power200']
        print(f"\n{model_name}:")
        print(f"  Max power (n=100): {np.nanmax(power100):.4f}")
        print(f"  Max power (n=200): {np.nanmax(power200):.4f}")
        print(f"  Power at δ=0 (n=100): {power100[np.where(np.round(data['delta_vals'], 2) == 0.00)[0][0]]:.4f}")
        print(f"  Power at δ=0 (n=200): {power200[np.where(np.round(data['delta_vals'], 2) == 0.00)[0][0]]:.4f}")

    print("\n" + "=" * 60)
    print("ALL ANALYSES COMPLETED SUCCESSFULLY!")
    print("=" * 60)