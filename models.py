"""
Functional Data Analysis with PLS and Related Methods
=====================================================
A comprehensive package for functional data analysis using:
- PLS (Partial Least Squares)
- APLS (Adaptive PLS)
- Splines with GCV
- PCA with GCV
- RKHS with GCV

Includes Bias-Variance decomposition with composite bar graphs.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.linalg import solve, lstsq, eigh
from scipy.linalg import LinAlgError
from itertools import product
import warnings
from typing import Tuple, Optional, List, Dict
from math import factorial
from datetime import datetime

# Suppress warnings
warnings.filterwarnings('ignore')
sns.set_style("whitegrid")

# ============================================================================
# COLORS
# ============================================================================

COLORS = {
    'PLS': '#1f77b4',
    'APLS': '#ff7f0e',
    'Splines': '#2ca02c',
    'PCA': '#d62728',
    'RKHS': '#9467bd'
}

BIAS_COLOR = '#d62728'    # Red for Bias²
VAR_COLOR = '#1f77b4'     # Blue for Variance

# ============================================================================
# PART 1: BASIS FUNCTIONS AND SPLINE UTILITIES
# ============================================================================

def create_cosine_basis(s: np.ndarray, J: int) -> np.ndarray:
    T = len(s)
    j = np.arange(1, J + 1)
    v = np.sqrt(2) * np.cos(np.pi * s[:, np.newaxis] * j[np.newaxis, :])
    v[:, 0] = 1
    return v.T

def natural_spline_basis(s: np.ndarray, d: int) -> Tuple[np.ndarray, np.ndarray]:
    T = len(s)
    degree = d - 1
    t = np.concatenate([[s[0]] * (degree + 1), s[1:-1], [s[-1]] * (degree + 1)])

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

    B = np.zeros((T, T))
    dB = np.zeros((T, T))
    for j in range(T):
        B[:, j] = np.sin(j * np.pi * (s - s[0]) / (s[-1] - s[0])) / np.sqrt(T/2)
        dB[:, j] = np.cos(j * np.pi * (s - s[0]) / (s[-1] - s[0])) * \
                   (j * np.pi / (s[-1] - s[0])) / np.sqrt(T/2)
    return B, dB

# ============================================================================
# PART 2: ESTIMATION METHODS
# ============================================================================

def pls(r: np.ndarray, K: np.ndarray, m: int) -> np.ndarray:
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
    n, T = X.shape
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
            Kr_tr = r_tr.reshape(-1, 1)
            Zm_tr = X_tr @ Kr_tr / T
            for j in range(2, m_val + 1):
                new_Kr = K_tr @ Kr_tr[:, -1] / n_tr
                new_Zm = X_tr @ new_Kr / n_tr
                Kr_tr = np.column_stack([Kr_tr, new_Kr.reshape(-1, 1)])
                Zm_tr = np.column_stack([Zm_tr, new_Zm.reshape(-1, 1)])
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

def spline_gcv(y: np.ndarray, X: np.ndarray, s: np.ndarray,
               rho: float, d: int) -> float:
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

def pca_gcv(r: np.ndarray, K: np.ndarray, m: int) -> Tuple[np.ndarray, int, np.ndarray]:
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
    n, T = X.shape
    beta_matrix = np.zeros((T, len(lambda_grid)))
    GCV = np.zeros(len(lambda_grid))
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
# PART 3: BIAS-VARIANCE FUNCTIONS
# ============================================================================

def compute_bias_variance_beta(beta_estimates: np.ndarray, beta_true: np.ndarray) -> Dict:
    T, M = beta_estimates.shape
    beta_mean = np.mean(beta_estimates, axis=1)
    bias2 = (beta_mean - beta_true)**2
    variance = np.var(beta_estimates, axis=1, ddof=1)
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

def compute_bias_variance_prediction(beta_estimates: np.ndarray, X: np.ndarray,
                                      y_true: np.ndarray, T: int) -> Dict:
    n = len(y_true)
    M = beta_estimates.shape[1]
    y_pred = X @ beta_estimates / T
    y_mean = np.mean(y_pred, axis=1)
    bias2 = (y_mean - y_true)**2
    variance = np.var(y_pred, axis=1, ddof=1)
    noise = np.var(y_true - np.mean(y_true))
    mse = bias2 + variance + noise
    return {
        'bias2': bias2,
        'variance': variance,
        'noise': noise,
        'mse': mse,
        'y_mean': y_mean,
        'y_pred': y_pred,
        'integrated_bias2': np.mean(bias2),
        'integrated_variance': np.mean(variance),
        'integrated_mse': np.mean(mse),
        'integrated_noise': noise
    }

# ============================================================================
# PART 4: COMPOSITE BAR GRAPHS
# ============================================================================

def plot_composite_mse_bars(results_beta: Dict, results_pred: Dict,
                             method_names: List[str], model_name: str,
                             save_dir: str = "."):
    """Composite bar graphs for MSE decomposition."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # ===== β̂ =====
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
            ax.text(idx, b/2, f'{b:.3f}', ha='center', va='center', fontsize=9, color='white', fontweight='bold')
        if v > 0.001:
            ax.text(idx, b + v/2, f'{v:.3f}', ha='center', va='center', fontsize=9, color='white', fontweight='bold')

    ax.set_xlabel('Method', fontsize=14)
    ax.set_ylabel('MSE Decomposition', fontsize=14)
    ax.set_title(f'{model_name}: MSE for β̂ (Coefficient Estimation)', fontsize=15, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods, fontsize=12)
    ax.legend(fontsize=12, loc='upper right')
    ax.grid(True, alpha=0.2, axis='y')
    ax.set_ylim(0, max(mse_vals) * 1.25 if mse_vals else 1)

    # ===== ŷ =====
    ax = axes[1]
    methods = []
    bias2_vals = []
    var_vals = []
    mse_vals = []
    for method in method_names:
        if method in results_pred:
            methods.append(method)
            bias2_vals.append(results_pred[method]['integrated_bias2'])
            var_vals.append(results_pred[method]['integrated_variance'])
            mse_vals.append(results_pred[method]['integrated_mse'])

    x_pos = np.arange(len(methods))

    bars1 = ax.bar(x_pos, bias2_vals, width, color=BIAS_COLOR, alpha=0.8,
                   label='Bias²', edgecolor='black', linewidth=0.8)
    bars2 = ax.bar(x_pos, var_vals, width, color=VAR_COLOR, alpha=0.8,
                   label='Variance', bottom=bias2_vals,
                   edgecolor='black', linewidth=0.8)

    for idx, (b, v) in enumerate(zip(bias2_vals, var_vals)):
        total = b + v
        ax.text(idx, total + 0.001, f'{total:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        if b > 0.001:
            ax.text(idx, b/2, f'{b:.3f}', ha='center', va='center', fontsize=9, color='white', fontweight='bold')
        if v > 0.001:
            ax.text(idx, b + v/2, f'{v:.3f}', ha='center', va='center', fontsize=9, color='white', fontweight='bold')

    ax.set_xlabel('Method', fontsize=14)
    ax.set_ylabel('MSE Decomposition', fontsize=14)
    ax.set_title(f'{model_name}: MSE for ŷ (Prediction)', fontsize=15, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods, fontsize=12)
    ax.legend(fontsize=12, loc='upper right')
    ax.grid(True, alpha=0.2, axis='y')
    ax.set_ylim(0, max(mse_vals) * 1.25 if mse_vals else 1)

    plt.suptitle(f'{model_name}: Composite MSE Decomposition (Bias² + Variance)',
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    try:
        fig.savefig(f"{save_dir}/{model_name.lower().replace(' ', '_')}_composite_mse.pdf",
                    dpi=300, bbox_inches='tight')
    except:
        pass
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
        ax.bar(x_pos, var_vals, width, color=VAR_COLOR, alpha=0.8, label='Variance', bottom=bias2_vals, edgecolor='black', linewidth=0.8)
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
    try:
        fig1.savefig(f"{save_dir}/composite_mse_beta_across_models.pdf", dpi=300, bbox_inches='tight')
    except:
        pass
    plt.close(fig1)

    # ŷ
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))
    for idx, model_name in enumerate(model_names):
        ax = axes2[idx]
        results_pred = all_results[model_name]['results_pred']
        methods = []
        bias2_vals = []
        var_vals = []
        mse_vals = []
        for method in method_names:
            if method in results_pred:
                methods.append(method)
                bias2_vals.append(results_pred[method]['integrated_bias2'])
                var_vals.append(results_pred[method]['integrated_variance'])
                mse_vals.append(results_pred[method]['integrated_mse'])
        x_pos = np.arange(len(methods))
        width = 0.6
        ax.bar(x_pos, bias2_vals, width, color=BIAS_COLOR, alpha=0.8, label='Bias²', edgecolor='black', linewidth=0.8)
        ax.bar(x_pos, var_vals, width, color=VAR_COLOR, alpha=0.8, label='Variance', bottom=bias2_vals, edgecolor='black', linewidth=0.8)
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
    plt.suptitle('MSE for ŷ (Prediction) Across All Models', fontsize=16, fontweight='bold')
    plt.tight_layout()
    try:
        fig2.savefig(f"{save_dir}/composite_mse_prediction_across_models.pdf", dpi=300, bbox_inches='tight')
    except:
        pass
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
        offset = (m_idx - len(method_names)/2 + 0.5) * width
        bottom = np.zeros(len(model_names))
        ax.bar(x_pos + offset, bias_vals, width, color=BIAS_COLOR, alpha=0.7, edgecolor='black', linewidth=0.5, bottom=bottom)
        bottom += np.array(bias_vals)
        ax.bar(x_pos + offset, var_vals, width, color=COLORS[method], alpha=0.7, label=method, edgecolor='black', linewidth=0.5, bottom=bottom)
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
        for model_name in model_names:
            if method in all_results[model_name]['results_pred']:
                bias_vals.append(all_results[model_name]['results_pred'][method]['integrated_bias2'])
                var_vals.append(all_results[model_name]['results_pred'][method]['integrated_variance'])
            else:
                bias_vals.append(0)
                var_vals.append(0)
        offset = (m_idx - len(method_names)/2 + 0.5) * width
        bottom = np.zeros(len(model_names))
        ax.bar(x_pos + offset, bias_vals, width, color=BIAS_COLOR, alpha=0.7, edgecolor='black', linewidth=0.5, bottom=bottom)
        bottom += np.array(bias_vals)
        ax.bar(x_pos + offset, var_vals, width, color=COLORS[method], alpha=0.7, label=method, edgecolor='black', linewidth=0.5, bottom=bottom)
    ax.set_xlabel('Data Model', fontsize=13)
    ax.set_ylabel('MSE', fontsize=13)
    ax.set_title('ŷ (Prediction)', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(model_names, fontsize=12)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.2, axis='y')

    plt.suptitle('Composite MSE Summary Across All Models and Methods', fontsize=16, fontweight='bold')
    plt.tight_layout()
    try:
        fig.savefig(f"{save_dir}/composite_mse_summary.pdf", dpi=300, bbox_inches='tight')
    except:
        pass
    plt.close(fig)
    return fig

def create_bias_variance_table(results_beta: Dict, results_pred: Dict,
                                method_names: List[str], model_name: str):
    """Print Bias-Variance tables."""
    print(f"\n{'='*80}")
    print(f"BIAS-VARIANCE DECOMPOSITION - {model_name}")
    print('='*80)
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
    print("\n" + "="*80)
    if results_beta:
        best_beta = min(results_beta.keys(), key=lambda x: results_beta[x]['integrated_mse'])
        print(f"Best for β̂: {best_beta} (MSE = {results_beta[best_beta]['integrated_mse']:.6f})")
    if results_pred:
        best_pred = min(results_pred.keys(), key=lambda x: results_pred[x]['integrated_mse'])
        print(f"Best for ŷ: {best_pred} (MSE = {results_pred[best_pred]['integrated_mse']:.6f})")

# ============================================================================
# PART 5: SIMULATION
# ============================================================================

def simulation_cv_with_beta(lam: np.ndarray, beta: np.ndarray, v: np.ndarray,
                             s: np.ndarray, M: int, n: int, J: int, m: int,
                             param_grid: List[Tuple], lam_grid: np.ndarray,
                             tau: float = 1.5, delta: float = 2.0,
                             d: int = 4, verbose: bool = False) -> Tuple:
    T = len(s)
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
    X_data = None
    y_true = None
    for k in range(M):
        if verbose and k % 10 == 0:
            print(f"Simulation {k+1}/{M}")
        eps = np.random.normal(0, 1, n)
        u = np.random.normal(0, 1, (n, J))
        X = (sqrt_lam * u) @ v
        y = X @ beta / T + eps
        r = X.T @ y / n
        K = X.T @ X / (n * T)
        if k == 0:
            X_data = X.copy()
            y_true = y.copy()
        beta_hat_pca[:, k], m_pca, _ = pca_gcv(r, K, m)
        beta_pls_m = pls(r, K, m)
        beta_hat_pls[:, k], m_pls[k], _, _ = pls_early_stop(
            beta_pls_m, X, y, r, K, tau, delta, n, beta_hat_pca[:, k]
        )
        beta_hat_apls[:, k], m_apls[k], _, _ = apls_cv_stable(
            y, X, K, r, m, k_folds=5, random_state=None, reg_param=1e-4
        )
        gcv_vals = []
        for rho, d_val in param_grid:
            gcv_vals.append(spline_gcv(y, X, s, rho, d_val))
        best_idx = np.argmin(gcv_vals)
        rho_opt, d_opt = param_grid[best_idx]
        beta_hat_spl[:, k] = spline(y, X, s, [rho_opt], d_opt).flatten()
        beta_hat_rkhs[:, k] = rkhs_gcv(y, X, lam_grid, s)
        u_test = np.random.normal(0, 1, (n, J))
        X_test = (sqrt_lam * u_test) @ v
        y_test = X_test @ beta / T + np.random.normal(0, 1, n)
        mspe_pls[k] = np.mean((y_test - X_test @ beta_hat_pls[:, k] / T)**2)
        mspe_apls[k] = np.mean((y_test - X_test @ beta_hat_apls[:, k] / T)**2)
        mspe_spl[k] = np.mean((y_test - X_test @ beta_hat_spl[:, k] / T)**2)
        mspe_pca[k] = np.mean((y_test - X_test @ beta_hat_pca[:, k] / T)**2)
        mspe_rkhs[k] = np.mean((y_test - X_test @ beta_hat_rkhs[:, k] / T)**2)
    mse_pls = np.mean((beta_hat_pls - beta[:, np.newaxis])**2, axis=0) / T
    mse_apls = np.mean((beta_hat_apls - beta[:, np.newaxis])**2, axis=0) / T
    mse_spl = np.mean((beta_hat_spl - beta[:, np.newaxis])**2, axis=0) / T
    mse_pca = np.mean((beta_hat_pca - beta[:, np.newaxis])**2, axis=0) / T
    mse_rkhs = np.mean((beta_hat_rkhs - beta[:, np.newaxis])**2, axis=0) / T
    mspe_all = np.vstack([mspe_pls, mspe_apls, mspe_spl, mspe_pca, mspe_rkhs])
    mse_all = np.vstack([mse_pls, mse_apls, mse_spl, mse_pca, mse_rkhs])
    beta_estimates = {
        'PLS': beta_hat_pls, 'APLS': beta_hat_apls,
        'Splines': beta_hat_spl, 'PCA': beta_hat_pca, 'RKHS': beta_hat_rkhs
    }
    return mspe_all, mse_all, m_pls, m_apls, beta_estimates, X_data, y_true

# ============================================================================
# PART 6: MAIN
# ============================================================================

def run_full_simulation_with_bias_variance(M: int = 100, test_mode: bool = True):
    np.random.seed(2025)
    n, J, T, m = 100, 100, 200, 20
    s = np.linspace(0, 1, T)
    j = np.arange(1, J + 1)
    tau, delta = 1.01, 0.1
    rho_grid = np.logspace(-12, -1, 5)
    d_grid = np.arange(1, 3)
    param_grid = [(rho, d) for d in d_grid for rho in rho_grid]
    lam_grid = np.logspace(-6, 0, 5)
    method_names = ["PLS", "APLS", "Splines", "PCA", "RKHS"]
    v = np.sqrt(2) * np.cos(np.pi * s[:, np.newaxis] * j[np.newaxis, :])
    v[:, 0] = 1
    v = v.T
    b1 = 4 / j**2.7
    beta1 = v.T @ b1
    lambda1 = 2 / j**1.1
    b2 = b1.copy()
    b2[:5] = 4
    beta2 = v.T @ b2
    lambda3 = lambda1.copy()
    lambda3[:5] = 2
    models = [("Model 1", lambda1, beta1), ("Model 2", lambda1, beta2), ("Model 3", lambda3, beta1)]
    all_results = {}
    M_actual = min(M, 50) if test_mode else M
    print("="*60)
    print("RUNNING SIMULATIONS WITH BIAS-VARIANCE ANALYSIS")
    print("="*60)
    print(f"Test mode: {test_mode}, Simulations: {M_actual}")
    for model_name, lam, beta in models:
        print(f"\nRunning {model_name}...")
        mspe, mse, m_pls, m_apls, beta_estimates, X_data, y_true = simulation_cv_with_beta(
            lam, beta, v, s, M_actual, n, J, m, param_grid, lam_grid,
            tau=tau, delta=delta, d=4, verbose=True
        )
        print(f"  Avg m_PLS: {np.mean(m_pls):.1f}, Avg m_APLS: {np.mean(m_apls):.1f}")
        results_beta = {}
        for method in method_names:
            results_beta[method] = compute_bias_variance_beta(beta_estimates[method], beta)
        results_pred = {}
        for method in method_names:
            y_pred_all = X_data @ beta_estimates[method] / T
            bias2 = (np.mean(y_pred_all, axis=1) - y_true)**2
            variance = np.var(y_pred_all, axis=1, ddof=1)
            noise = np.var(y_true - np.mean(y_true))
            results_pred[method] = {
                'bias2': bias2, 'variance': variance, 'noise': noise,
                'mse': bias2 + variance + noise,
                'integrated_bias2': np.mean(bias2),
                'integrated_variance': np.mean(variance),
                'integrated_mse': np.mean(bias2 + variance + noise),
                'integrated_noise': noise
            }
        print("  Generating plots...")
        plot_composite_mse_bars(results_beta, results_pred, method_names, model_name, save_dir='.')
        create_bias_variance_table(results_beta, results_pred, method_names, model_name)
        all_results[model_name] = {'mspe': mspe, 'mse': mse, 'results_beta': results_beta, 'results_pred': results_pred}
    print("\nGenerating cross-model comparison plots...")
    plot_composite_mse_across_models(all_results, method_names, save_dir='.')
    plot_composite_mse_summary(all_results, method_names, save_dir='.')
    print("\n" + "="*60)
    print("ALL COMPLETED SUCCESSFULLY!")
    print("="*60)
    return all_results

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    print("="*60)
    print("FUNCTIONAL DATA ANALYSIS WITH BIAS-VARIANCE DECOMPOSITION")
    print("="*60)
    results = run_full_simulation_with_bias_variance(M=100, test_mode=True)
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    for model_name, data in results.items():
        mspe = data['mspe']
        mse = data['mse']
        print(f"\n{model_name}:")
        print(f"  MSPE mean: {np.mean(mspe, axis=1)}")
        print(f"  MSE mean:  {np.mean(mse, axis=1)}")
    print("\nALL COMPLETED SUCCESSFULLY!")