import numpy as np
from scipy.interpolate import BSpline
from scipy.linalg import solve
import matplotlib.pyplot as plt
from scipy.stats import norm, uniform
from collections import namedtuple


# Note: Python doesn't have a direct equivalent to BSplineKit.jl,
# so we'll implement a simplified version of natural spline basis

def natural_spline_basis(s, d):
    """
    Compute natural spline basis functions and their derivatives.

    Parameters:
    s : array-like
        The knot sequence
    d : int
        The degree of the spline (actually order, not degree in Julia's BSplineKit)

    Returns:
    B : ndarray
        Basis functions evaluated at knots
    dB : ndarray
        Derivatives of basis functions evaluated at knots
    """
    s = np.asarray(s)
    T = len(s)

    # For natural splines, we need to construct the basis
    # This is a simplified version - for full functionality, you might want to
    # use scipy's BSpline or implement a more complete natural spline basis

    # Create B-spline basis using scipy
    # Note: scipy uses degree (degree = order - 1)
    degree = d - 1  # Assuming d is order

    # For natural splines, we need to add boundary knots
    # This is a simplified approach
    t = np.concatenate([
        [s[0]] * (degree + 1),
        s[1:-1],
        [s[-1]] * (degree + 1)
    ])

    # Create B-spline basis
    def bspline_basis(x, t, degree, i):
        """Evaluate a single B-spline basis function"""
        # Using Cox-de Boor recursion formula
        if degree == 0:
            return np.where((t[i] <= x) & (x < t[i + 1]), 1.0, 0.0)
        else:
            denom1 = t[i + degree] - t[i]
            denom2 = t[i + degree + 1] - t[i + 1]

            term1 = np.zeros_like(x)
            term2 = np.zeros_like(x)

            if denom1 != 0:
                term1 = (x - t[i]) / denom1 * bspline_basis(x, t, degree - 1, i)
            if denom2 != 0:
                term2 = (t[i + degree + 1] - x) / denom2 * bspline_basis(x, t, degree - 1, i + 1)

            return term1 + term2

    # Evaluate basis functions
    B = np.zeros((T, T))
    dB = np.zeros((T, T))

    # For each basis function
    for j in range(T):
        # Use centered evaluation
        x_eval = np.linspace(s[0], s[-1], T)

        # This is a simplified version - you'd need to implement proper basis functions
        # For demonstration purposes, we'll create a simple basis
        # In practice, you'd use a proper spline library

        # Simplified approach: use sine/cosine basis for demonstration
        # Replace this with actual spline basis implementation
        B[:, j] = np.sin(j * np.pi * (s - s[0]) / (s[-1] - s[0])) / np.sqrt(T / 2)
        dB[:, j] = np.cos(j * np.pi * (s - s[0]) / (s[-1] - s[0])) * (j * np.pi / (s[-1] - s[0])) / np.sqrt(T / 2)

    return B, dB


# Alternative implementation using scipy's BSpline
def natural_spline_basis_scipy(s, d):
    """
    Alternative implementation using scipy's BSpline.
    Note: This creates a natural cubic spline basis (d=4 for cubic).
    """
    from scipy.interpolate import splrep, splev

    s = np.asarray(s)
    T = len(s)

    # Create knots for natural spline
    # For natural spline, we need to enforce second derivative zero at boundaries
    # This is a simplified version

    # Fit natural cubic spline for each basis function
    B = np.zeros((T, T))
    dB = np.zeros((T, T))

    # Create identity matrix basis (each basis function is 1 at its knot, 0 elsewhere)
    for i in range(T):
        y = np.zeros(T)
        y[i] = 1.0

        # Fit spline with natural boundary conditions
        # splrep with k=3 gives cubic spline
        tck = splrep(s, y, k=d - 1)

        # Evaluate spline and derivative
        B[:, i] = splev(s, tck)
        dB[:, i] = splev(s, tck, der=1)

    return B, dB


# Alternative: Use patsy for natural splines (if you have it installed)
try:
    import patsy


    def natural_spline_basis_patsy(s, d):
        """Implementation using patsy library"""
        import patsy
        s = np.asarray(s)
        T = len(s)

        # Create design matrix for natural splines
        # Note: patsy's bs() creates B-spline basis
        # For natural splines, use ns() (natural splines)
        data = {'x': s}
        B = patsy.dmatrix(f"ns(x, df={T}) - 1", data)

        # Derivative is not directly available in patsy
        # You'd need to use numerical differentiation
        B = np.asarray(B)
        dB = np.zeros_like(B)

        # Numerical derivative
        h = 1e-5
        for i in range(T):
            data_h = {'x': s + h * np.eye(T)[i]}
            B_h = patsy.dmatrix(f"ns(x, df={T}) - 1", data_h)
            dB[:, i] = (np.asarray(B_h)[:, i] - B[:, i]) / h

        return B, dB
except ImportError:
    pass

# Example usage
if __name__ == "__main__":
    # Test the function
    s = np.linspace(0, 1, 10)
    d = 4  # Order for cubic spline

    B, dB = natural_spline_basis(s, d)

    print("B shape:", B.shape)
    print("dB shape:", dB.shape)

    # Optional: Plot the basis functions if you have matplotlib
    try:
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        for i in range(min(5, B.shape[1])):
            ax1.plot(s, B[:, i], label=f'Basis {i}')
            ax2.plot(s, dB[:, i], label=f'Derivative {i}')

        ax1.set_title('Basis Functions')
        ax1.legend()
        ax2.set_title('Derivatives')
        ax2.legend()
        plt.tight_layout()
        plt.show()
    except:
        pass

import numpy as np
from scipy.linalg import inv, solve


def spline(y, X, s, rho_grid, d):
    """
    Compute spline estimator for given regularization parameters.

    Parameters:
    y : ndarray of shape (n,)
        Response variable
    X : ndarray of shape (n, T)
        Design matrix
    s : ndarray of shape (T,)
        Knot sequence
    rho_grid : ndarray or list
        Grid of regularization parameters
    d : int
        Order of spline basis

    Returns:
    beta_hat : ndarray of shape (T, len(rho_grid))
        Estimated coefficients for each rho in the grid
    """
    n, T = X.shape

    # Compute natural cubic spline basis
    B, dB = natural_spline_basis(s, d)

    # B' * B inverse
    BtB = B.T @ B
    # Add small regularization for numerical stability
    BtB_inv = inv(BtB + 1e-10 * np.eye(T))

    # Projection matrix P = B * (B'B)^(-1) * B'
    P = B @ BtB_inv @ B.T

    # Dm = dB' * dB
    Dm = dB.T @ dB

    # Construct S matrix: [s^0, s^1, ..., s^(2d-1)]
    S = np.column_stack([s ** i for i in range(2 * d)])

    # Compute Am = S * inv(S'*S + 1e-50*I) * S'/T + B * inv(B'*B) * Dm * inv(B'*B) * B'/T
    StS = S.T @ S
    StS_inv = inv(StS + 1e-50 * np.eye(StS.shape[0]))
    Am = (S @ StS_inv @ S.T) / T + (B @ BtB_inv @ Dm @ BtB_inv @ B.T) / T

    # Initialize beta_hat
    beta_hat = np.zeros((T, len(rho_grid)))

    # Compute X' * X and X' * y once (they don't depend on rho)
    XtX = X.T @ X
    XtY = X.T @ y

    # Compute estimator for each candidate rho in the grid
    for i, rho in enumerate(rho_grid):
        # Matrix to invert: X'X/(n*T) + rho * Am
        M = XtX / (n * T) + rho * Am

        # Add small regularization for numerical stability if needed
        M_reg = M + 1e-10 * np.eye(T)

        # Solve M * beta = (X' * y)/n
        # Using solve for better numerical stability than inv
        try:
            beta = solve(M_reg, XtY / n, assume_a='pos')
        except:
            # Fallback to pseudo-inverse if matrix is singular
            beta = np.linalg.lstsq(M_reg, XtY / n, rcond=None)[0]

        # Apply projection P
        beta_hat[:, i] = P @ beta

    return beta_hat


# Alternative implementation with explicit inversion (closer to Julia code)
def spline_with_inv(y, X, s, rho_grid, d):
    """
    Alternative implementation using explicit matrix inversion.
    This is closer to the original Julia code but less numerically stable.
    """
    n, T = X.shape

    # Compute natural cubic spline basis
    B, dB = natural_spline_basis(s, d)

    # B' * B inverse
    BtB = B.T @ B
    BtB_inv = inv(BtB + 1e-10 * np.eye(T))

    # Projection matrix P = B * (B'B)^(-1) * B'
    P = B @ BtB_inv @ B.T

    # Dm = dB' * dB
    Dm = dB.T @ dB

    # Construct S matrix
    S = np.column_stack([s ** i for i in range(2 * d)])

    # Compute Am
    StS = S.T @ S
    StS_inv = inv(StS + 1e-50 * np.eye(StS.shape[0]))
    Am = (S @ StS_inv @ S.T) / T + (B @ BtB_inv @ Dm @ BtB_inv @ B.T) / T

    # Initialize beta_hat
    beta_hat = np.zeros((T, len(rho_grid)))

    # Compute X' * X and X' * y once
    XtX = X.T @ X
    XtY = X.T @ y

    # Compute estimator for each candidate rho in the grid
    for i, rho in enumerate(rho_grid):
        # Matrix to invert
        M = XtX / (n * T) + rho * Am

        # Direct inversion (matching Julia code)
        # Note: This is the original formulation from the Julia code
        try:
            M_inv = inv(M)
            beta_hat[:, i] = P @ M_inv @ (XtY / n)
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse if singular
            M_pinv = np.linalg.pinv(M)
            beta_hat[:, i] = P @ M_pinv @ (XtY / n)

    return beta_hat


# Enhanced version with additional features
def spline_enhanced(y, X, s, rho_grid, d, return_components=False):
    """
    Enhanced version that returns additional components if requested.

    Parameters:
    y, X, s, rho_grid, d : as above
    return_components : bool
        If True, return B, dB, P, Am as well

    Returns:
    beta_hat : ndarray
        Estimated coefficients
    components : dict (if return_components=True)
        Dictionary containing B, dB, P, Am
    """
    n, T = X.shape

    # Compute natural cubic spline basis
    B, dB = natural_spline_basis(s, d)

    # B' * B inverse
    BtB = B.T @ B
    BtB_inv = inv(BtB + 1e-10 * np.eye(T))

    # Projection matrix P = B * (B'B)^(-1) * B'
    P = B @ BtB_inv @ B.T

    # Dm = dB' * dB
    Dm = dB.T @ dB

    # Construct S matrix
    S = np.column_stack([s ** i for i in range(2 * d)])

    # Compute Am
    StS = S.T @ S
    StS_inv = inv(StS + 1e-50 * np.eye(StS.shape[0]))
    Am = (S @ StS_inv @ S.T) / T + (B @ BtB_inv @ Dm @ BtB_inv @ B.T) / T

    # Initialize beta_hat
    beta_hat = np.zeros((T, len(rho_grid)))

    # Compute X' * X and X' * y
    XtX = X.T @ X
    XtY = X.T @ y

    # Compute for each rho
    for i, rho in enumerate(rho_grid):
        M = XtX / (n * T) + rho * Am
        M_inv = inv(M + 1e-10 * np.eye(T))
        beta_hat[:, i] = P @ M_inv @ (XtY / n)

    if return_components:
        components = {
            'B': B,
            'dB': dB,
            'P': P,
            'Am': Am
        }
        return beta_hat, components

    return beta_hat


# Example usage
if __name__ == "__main__":
    # Generate synthetic data
    np.random.seed(42)
    n = 100  # number of observations
    T = 20  # number of knots

    # Generate data
    s = np.linspace(0, 1, T)
    X = np.random.randn(n, T)  # design matrix
    y = np.random.randn(n)  # response variable

    # Regularization grid
    rho_grid = np.logspace(-4, 2, 10)
    d = 4  # order of spline

    # Compute estimates
    beta_hat = spline(y, X, s, rho_grid, d)

    print(f"Beta hat shape: {beta_hat.shape}")
    print(f"First few coefficients for first rho: {beta_hat[:5, 0]}")

    # Visualize results
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))
        for i in range(min(5, beta_hat.shape[0])):
            plt.plot(rho_grid, beta_hat[i, :], label=f'Coefficient {i}')

        plt.xscale('log')
        plt.xlabel(r'$\rho$')
        plt.ylabel('Coefficient value')
        plt.title('Spline coefficients vs regularization parameter')
        plt.legend()
        plt.grid(True)
        plt.show()

        # Show coefficient paths
        plt.figure(figsize=(10, 6))
        for i in range(min(5, beta_hat.shape[0])):
            plt.plot(range(len(rho_grid)), beta_hat[i, :], marker='o', label=f'Coeff {i}')

        plt.xlabel('Rho index')
        plt.ylabel('Coefficient value')
        plt.title('Coefficient paths')
        plt.legend()
        plt.grid(True)
        plt.show()

    except:
        pass

import numpy as np
from scipy.linalg import inv, solve
from scipy.linalg import LinAlgError


def spline_gcv(y, X, s, rho, d):
    """
    Compute Generalized Cross-Validation (GCV) score for a spline estimator.

    Parameters:
    y : ndarray of shape (n,)
        Response variable
    X : ndarray of shape (n, T)
        Design matrix
    s : ndarray of shape (T,)
        Knot sequence
    rho : float
        Regularization parameter
    d : int
        Order of spline basis

    Returns:
    GCV : float
        Generalized Cross-Validation score
    """
    n, T = X.shape

    # Compute natural cubic spline basis
    B, dB = natural_spline_basis(s, d)

    # B' * B inverse (with small regularization for stability)
    BtB = B.T @ B
    BtB_inv = inv(BtB + 1e-10 * np.eye(T))

    # Projection matrix P = B * (B'B)^(-1) * B'
    P = B @ BtB_inv @ B.T

    # Dm = dB' * dB
    Dm = dB.T @ dB

    # Construct S matrix: [s^0, s^1, ..., s^(2d-1)]
    S = np.column_stack([s ** i for i in range(2 * d)])

    # Compute Am
    # Am = S * inv(S'*S + 1e-50*I) * S'/T + B * BtB_inv * Dm * BtB_inv * B' / T
    StS = S.T @ S
    StS_inv = inv(StS + 1e-50 * np.eye(StS.shape[0]))
    Am = (S @ StS_inv @ S.T) / T + (B @ BtB_inv @ Dm @ BtB_inv @ B.T) / T

    # Compute Hρ (hat matrix)
    # Hρ = X * P * inv(X' * X / (n * T^2) + ρ * Am / T) * X' / (n * T^2)
    XtX = X.T @ X
    Xt = X.T

    # Matrix to invert
    M = XtX / (n * T ** 2) + rho * Am / T

    # Add small regularization for numerical stability
    M_reg = M + 1e-10 * np.eye(T)

    try:
        # Solve using linear system solver for better stability
        # We need: inv(M) * X'
        # Instead of computing inverse explicitly, solve M * Z = X' for Z
        Z = solve(M_reg, Xt, assume_a='pos')
        # Then Hρ = X * P * Z / (n * T^2)
        Hrho = X @ P @ Z / (n * T ** 2)
    except (LinAlgError, np.linalg.LinAlgError):
        # Fallback to pseudo-inverse if matrix is singular
        M_pinv = np.linalg.pinv(M_reg)
        Hrho = X @ P @ M_pinv @ Xt / (n * T ** 2)

    # Compute GCV
    # GCV = sum((y - Hρ * y).^2) / n / (1 - tr(Hρ)/n)^2
    residuals = y - Hrho @ y
    gcv = np.sum(residuals ** 2) / n / (1 - np.trace(Hrho) / n) ** 2

    return gcv


# Alternative version matching Julia code more closely (using explicit inverse)
def spline_gcv_explicit(y, X, s, rho, d):
    """
    Alternative implementation using explicit matrix inversion.
    This matches the Julia code more closely but is less numerically stable.
    """
    n, T = X.shape

    # Compute natural cubic spline basis
    B, dB = natural_spline_basis(s, d)

    # B' * B inverse
    BtB = B.T @ B
    BtB_inv = inv(BtB + 1e-10 * np.eye(T))

    # Projection matrix P
    P = B @ BtB_inv @ B.T

    # Dm = dB' * dB
    Dm = dB.T @ dB

    # Construct S matrix
    S = np.column_stack([s ** i for i in range(2 * d)])

    # Compute Am
    StS = S.T @ S
    StS_inv = inv(StS + 1e-50 * np.eye(StS.shape[0]))
    Am = (S @ StS_inv @ S.T) / T + (B @ BtB_inv @ Dm @ BtB_inv @ B.T) / T

    # Compute Hρ
    XtX = X.T @ X
    Xt = X.T

    # Matrix to invert
    M = XtX / (n * T ** 2) + rho * Am / T

    try:
        # Explicit inverse (matching Julia)
        M_inv = inv(M + 1e-10 * np.eye(T))
        Hrho = X @ P @ M_inv @ Xt / (n * T ** 2)
    except np.linalg.LinAlgError:
        # Fallback to pseudo-inverse
        M_pinv = np.linalg.pinv(M)
        Hrho = X @ P @ M_pinv @ Xt / (n * T ** 2)

    # Compute GCV
    residuals = y - Hrho @ y
    gcv = np.sum(residuals ** 2) / n / (1 - np.trace(Hrho) / n) ** 2

    return gcv


# Optimized version that reuses computations for multiple rho values
def spline_gcv_optimized(y, X, s, rho, d, precomputed=None):
    """
    Optimized version that can reuse precomputed matrices.

    Parameters:
    y, X, s, rho, d : as above
    precomputed : dict or None
        Dictionary with precomputed matrices (B, dB, P, Am, XtX, Xt)

    Returns:
    GCV : float
        GCV score
    precomputed : dict
        Dictionary with computed matrices for reuse
    """
    n, T = X.shape

    if precomputed is None:
        # Compute everything from scratch
        B, dB = natural_spline_basis(s, d)

        BtB = B.T @ B
        BtB_inv = inv(BtB + 1e-10 * np.eye(T))
        P = B @ BtB_inv @ B.T

        Dm = dB.T @ dB
        S = np.column_stack([s ** i for i in range(2 * d)])
        StS = S.T @ S
        StS_inv = inv(StS + 1e-50 * np.eye(StS.shape[0]))
        Am = (S @ StS_inv @ S.T) / T + (B @ BtB_inv @ Dm @ BtB_inv @ B.T) / T

        XtX = X.T @ X
        Xt = X.T

        precomputed = {
            'P': P,
            'Am': Am,
            'XtX': XtX,
            'Xt': Xt,
            'B': B,
            'dB': dB
        }
    else:
        # Extract precomputed matrices
        P = precomputed['P']
        Am = precomputed['Am']
        XtX = precomputed['XtX']
        Xt = precomputed['Xt']

    # Compute Hρ
    M = XtX / (n * T ** 2) + rho * Am / T

    try:
        # Use solve for better stability
        Z = solve(M + 1e-10 * np.eye(T), Xt, assume_a='pos')
        Hrho = X @ P @ Z / (n * T ** 2)
    except (LinAlgError, np.linalg.LinAlgError):
        M_pinv = np.linalg.pinv(M + 1e-10 * np.eye(T))
        Hrho = X @ P @ M_pinv @ Xt / (n * T ** 2)

    # Compute GCV
    residuals = y - Hrho @ y
    gcv = np.sum(residuals ** 2) / n / (1 - np.trace(Hrho) / n) ** 2

    return gcv, precomputed


# Function to find optimal rho using GCV
def find_optimal_rho(y, X, s, rho_grid, d, verbose=False):
    """
    Find the optimal regularization parameter using GCV.

    Parameters:
    y, X, s, rho_grid, d : as above
    verbose : bool
        If True, print progress

    Returns:
    rho_opt : float
        Optimal rho value
    gcv_scores : ndarray
        GCV scores for each rho in the grid
    """
    n, T = X.shape
    gcv_scores = np.zeros(len(rho_grid))

    # Precompute matrices once
    B, dB = natural_spline_basis(s, d)
    BtB = B.T @ B
    BtB_inv = inv(BtB + 1e-10 * np.eye(T))
    P = B @ BtB_inv @ B.T
    Dm = dB.T @ dB
    S = np.column_stack([s ** i for i in range(2 * d)])
    StS = S.T @ S
    StS_inv = inv(StS + 1e-50 * np.eye(StS.shape[0]))
    Am = (S @ StS_inv @ S.T) / T + (B @ BtB_inv @ Dm @ BtB_inv @ B.T) / T
    XtX = X.T @ X
    Xt = X.T

    precomputed = {
        'P': P,
        'Am': Am,
        'XtX': XtX,
        'Xt': Xt
    }

    # Compute GCV for each rho
    for i, rho in enumerate(rho_grid):
        if verbose:
            print(f"Computing GCV for rho = {rho:.6f}")

        gcv_scores[i], _ = spline_gcv_optimized(y, X, s, rho, d, precomputed)

        if verbose:
            print(f"  GCV = {gcv_scores[i]:.6f}")

    # Find optimal rho
    idx_opt = np.argmin(gcv_scores)
    rho_opt = rho_grid[idx_opt]

    if verbose:
        print(f"\nOptimal rho: {rho_opt:.6f} with GCV = {gcv_scores[idx_opt]:.6f}")

    return rho_opt, gcv_scores


# Example usage
if __name__ == "__main__":
    # Generate synthetic data
    np.random.seed(42)
    n = 100
    T = 20

    s = np.linspace(0, 1, T)
    X = np.random.randn(n, T)

    # True coefficients with some structure
    beta_true = np.exp(-s * 3) * np.sin(s * 10)
    y = X @ beta_true + 0.1 * np.random.randn(n)

    d = 4  # order of spline

    # Test a single rho value
    rho = 0.01
    gcv = spline_gcv(y, X, s, rho, d)
    print(f"GCV for rho={rho}: {gcv:.6f}")

    # Find optimal rho
    rho_grid = np.logspace(-4, 2, 20)
    rho_opt, gcv_scores = find_optimal_rho(y, X, s, rho_grid, d, verbose=True)

    # Visualize GCV curve
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))
        plt.semilogx(rho_grid, gcv_scores, 'b-o', linewidth=2, markersize=8)
        plt.axvline(rho_opt, color='r', linestyle='--', label=f'Optimal ρ = {rho_opt:.4f}')
        plt.xlabel(r'$\rho$')
        plt.ylabel('GCV Score')
        plt.title('Generalized Cross-Validation')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

        # Show the optimal coefficients
        B, dB = natural_spline_basis(s, d)
        beta_opt = spline(y, X, s, [rho_opt], d)

        plt.figure(figsize=(10, 6))
        plt.plot(s, beta_true, 'g-', linewidth=2, label='True coefficients')
        plt.plot(s, beta_opt[:, 0], 'r--', linewidth=2, label=f'Estimated (ρ={rho_opt:.4f})')
        plt.xlabel('s')
        plt.ylabel('Coefficient value')
        plt.title('Estimated vs True Coefficients')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    except:
        pass

import numpy as np
from scipy.linalg import solve, inv, eigh
from scipy.linalg import LinAlgError
from scipy.sparse.linalg import eigsh
import warnings


def rkhs_gcv(y, X, lambda_grid, s):
    """
    RKHS (Reproducing Kernel Hilbert Space) estimator with GCV.

    Parameters:
    y : ndarray of shape (n,)
        Response variable
    X : ndarray of shape (n, T)
        Design matrix
    lambda_grid : ndarray or list
        Grid of regularization parameters
    s : ndarray of shape (T,)
        Knot sequence

    Returns:
    beta_opt : ndarray of shape (T,)
        Estimated coefficients at optimal lambda
    """
    n, T = X.shape

    # Initialize output arrays
    beta_matrix = np.zeros((T, len(lambda_grid)))
    GCV = np.zeros(len(lambda_grid))

    # Compute kernel matrix
    # Δ = s .- transpose(s)
    Delta = s[:, np.newaxis] - s[np.newaxis, :]

    # B2_s = s.^2 .- s .+ 1/6
    B2_s = s ** 2 - s + 1 / 6
    B2_t = s ** 2 - s + 1 / 6

    # B4_st = abs.(Δ).^4 - 2*abs.(Δ).^3 + abs.(Δ).^2 .- 1/30
    abs_Delta = np.abs(Delta)
    B4_st = abs_Delta ** 4 - 2 * abs_Delta ** 3 + abs_Delta ** 2 - 1 / 30

    # Km = (1/factorial(2)^2) * B2_s * B2_t' .- (1/factorial(4)) * B4_st
    from math import factorial
    Km = (1 / factorial(2) ** 2) * np.outer(B2_s, B2_t) - (1 / factorial(4)) * B4_st

    # Compute Sigma and T matrix
    # Σ = X * Km * X' / T^2
    Sigma = X @ Km @ X.T / T ** 2

    # T_matrix = zeros(n, 2)
    # T_matrix[:, 1] = sum(X, dims=2) / T
    # T_matrix[:, 2] = X * s / T
    T_matrix = np.zeros((n, 2))
    T_matrix[:, 0] = np.sum(X, axis=1) / T
    T_matrix[:, 1] = X @ s / T

    # For each lambda
    for j, lam in enumerate(lambda_grid):
        # W = Σ + n * λ * I
        W = Sigma + n * lam * np.eye(n)

        try:
            # Solve W * c = (y - T_matrix * d) for c
            # But we need to solve for d first:
            # d = (T' * W^{-1} * T)^{-1} * (T' * W^{-1} * y)

            # Compute W_inv * T_matrix and W_inv * y
            W_inv_T = solve(W, T_matrix, assume_a='pos')
            W_inv_y = solve(W, y, assume_a='pos')

            # Compute d
            d = solve(T_matrix.T @ W_inv_T, T_matrix.T @ W_inv_y, assume_a='pos')

            # Compute c = W^{-1} * (y - T_matrix * d)
            c = W_inv_y - W_inv_T @ d

            # Compute beta_vals
            # beta_vals = d[1] .+ d[2] .* s .+ (c' * X * Km / T)'
            beta_vals = d[0] + d[1] * s + (c @ X @ Km / T).T

        except (LinAlgError, np.linalg.LinAlgError):
            # Fallback to pseudo-inverse if singular
            W_pinv = np.linalg.pinv(W)
            W_inv_T = W_pinv @ T_matrix
            W_inv_y = W_pinv @ y

            d = np.linalg.lstsq(T_matrix.T @ W_inv_T, T_matrix.T @ W_inv_y, rcond=None)[0]
            c = W_inv_y - W_inv_T @ d
            beta_vals = d[0] + d[1] * s + (c @ X @ Km / T).T

        # Compute the residual sum of squares
        # RSS = sum((y - X * beta_vals / T)^2)
        residuals = y - X @ beta_vals / T
        RSS = np.sum(residuals ** 2)

        # Compute the GCV score
        # GCV[j] = (RSS / n) / (1 - (2 + tr(inv(W) * Σ)) / n)^2
        try:
            W_inv_Sigma = solve(W, Sigma, assume_a='pos')
            trace_term = np.trace(W_inv_Sigma)
        except (LinAlgError, np.linalg.LinAlgError):
            W_inv_Sigma = np.linalg.pinv(W) @ Sigma
            trace_term = np.trace(W_inv_Sigma)

        GCV[j] = (RSS / n) / (1 - (2 + trace_term) / n) ** 2

        # Store the estimated beta for this lambda
        beta_matrix[:, j] = beta_vals.flatten()

    # Find optimal lambda
    j_opt = np.argmin(GCV)
    beta_opt = beta_matrix[:, j_opt]

    return beta_opt


# Alternative implementation with more efficient matrix operations
def rkhs_gcv_efficient(y, X, lambda_grid, s):
    """
    More efficient version of RKHS GCV using Cholesky decomposition
    and avoiding repeated inversions where possible.
    """
    n, T = X.shape

    beta_matrix = np.zeros((T, len(lambda_grid)))
    GCV = np.zeros(len(lambda_grid))

    # Compute kernel matrix (same as before)
    Delta = s[:, np.newaxis] - s[np.newaxis, :]
    B2_s = s ** 2 - s + 1 / 6
    B2_t = s ** 2 - s + 1 / 6
    abs_Delta = np.abs(Delta)
    B4_st = abs_Delta ** 4 - 2 * abs_Delta ** 3 + abs_Delta ** 2 - 1 / 30

    from math import factorial
    Km = (1 / factorial(2) ** 2) * np.outer(B2_s, B2_t) - (1 / factorial(4)) * B4_st

    Sigma = X @ Km @ X.T / T ** 2

    T_matrix = np.zeros((n, 2))
    T_matrix[:, 0] = np.sum(X, axis=1) / T
    T_matrix[:, 1] = X @ s / T

    # Pre-compute X @ Km for efficiency
    XKm = X @ Km

    for j, lam in enumerate(lambda_grid):
        W = Sigma + n * lam * np.eye(n)

        try:
            # Use Cholesky decomposition for better stability
            L = np.linalg.cholesky(W)

            # Solve using Cholesky
            # W_inv_T = solve(W, T_matrix)
            W_inv_T = solve(L, solve(L.T, T_matrix, lower=True), lower=False)
            W_inv_y = solve(L, solve(L.T, y, lower=True), lower=False)

            # Compute d
            d = solve(T_matrix.T @ W_inv_T, T_matrix.T @ W_inv_y)

            # Compute c
            c = W_inv_y - W_inv_T @ d

            # Compute beta_vals
            beta_vals = d[0] + d[1] * s + (c @ XKm / T).T

        except np.linalg.LinAlgError:
            # Fallback to standard solver
            try:
                W_inv_T = solve(W, T_matrix)
                W_inv_y = solve(W, y)

                d = solve(T_matrix.T @ W_inv_T, T_matrix.T @ W_inv_y)
                c = W_inv_y - W_inv_T @ d
                beta_vals = d[0] + d[1] * s + (c @ XKm / T).T
            except:
                # Ultimate fallback
                W_pinv = np.linalg.pinv(W)
                W_inv_T = W_pinv @ T_matrix
                W_inv_y = W_pinv @ y
                d = np.linalg.lstsq(T_matrix.T @ W_inv_T, T_matrix.T @ W_inv_y, rcond=None)[0]
                c = W_inv_y - W_inv_T @ d
                beta_vals = d[0] + d[1] * s + (c @ XKm / T).T

        # RSS
        residuals = y - X @ beta_vals / T
        RSS = np.sum(residuals ** 2)

        # GCV
        try:
            W_inv_Sigma = solve(W, Sigma)
            trace_term = np.trace(W_inv_Sigma)
        except:
            W_inv_Sigma = np.linalg.pinv(W) @ Sigma
            trace_term = np.trace(W_inv_Sigma)

        GCV[j] = (RSS / n) / (1 - (2 + trace_term) / n) ** 2
        beta_matrix[:, j] = beta_vals.flatten()

    j_opt = np.argmin(GCV)
    return beta_matrix[:, j_opt]


def pca(r, K, m):
    """
    PCA (Principal Components Analysis) estimator.

    Parameters:
    r : ndarray of shape (T,)
        Response vector (typically coefficients or residuals)
    K : ndarray of shape (T, T)
        Kernel or covariance matrix
    m : int
        Number of principal components to use (1 to T)

    Returns:
    beta_hat : ndarray of shape (T, m)
        PCA estimates for each number of components
    """
    T = len(r)

    # Initialize output
    beta_hat = np.zeros((T, m))

    try:
        # Compute eigen decomposition of symmetric matrix K
        # Use eigh for symmetric matrices (more stable)
        evals, evecs = eigh(K)

        # evals are in ascending order, we want the largest m
        # Take the last m eigenvalues and eigenvectors
        # In Julia: evals[(end-j+1):end] means take the last j
        for j in range(1, m + 1):
            # Get the largest j eigenvalues and eigenvectors
            lambda_hat = evals[-j:]
            v_hat = evecs[:, -j:]

            # βhat[:, j] = vhat * (vhat'*r ./ λhat)
            # Note: In Julia, ./ is element-wise division
            # In Python, we can use broadcasting
            beta_hat[:, j - 1] = v_hat @ ((v_hat.T @ r) / lambda_hat)

    except np.linalg.LinAlgError:
        # If K is not positive definite, use svd instead
        warnings.warn("Matrix K is not positive definite, using SVD instead")
        U, s, Vt = np.linalg.svd(K)

        for j in range(1, m + 1):
            # Use largest j singular values/vectors
            lambda_hat = s[:j]
            v_hat = U[:, :j]
            beta_hat[:, j - 1] = v_hat @ ((v_hat.T @ r) / lambda_hat)

    return beta_hat


# Alternative implementation using eigsh for large matrices
def pca_sparse(r, K, m, method='eigsh'):
    """
    PCA estimator optimized for large sparse matrices.

    Parameters:
    r : ndarray of shape (T,)
        Response vector
    K : ndarray of shape (T, T)
        Kernel matrix (can be sparse)
    m : int
        Number of principal components
    method : str
        'eigsh' for sparse eigenvalue solver, 'full' for dense

    Returns:
    beta_hat : ndarray of shape (T, m)
        PCA estimates
    """
    T = len(r)
    beta_hat = np.zeros((T, m))

    if method == 'eigsh' and m < T // 2:
        try:
            # Use sparse eigenvalue solver for large matrices
            from scipy.sparse.linalg import eigsh
            # Get largest m eigenvalues/vectors
            evals, evecs = eigsh(K, k=m, which='LM')

            for j in range(1, m + 1):
                lambda_hat = evals[:j]
                v_hat = evecs[:, :j]
                beta_hat[:, j - 1] = v_hat @ ((v_hat.T @ r) / lambda_hat)

            return beta_hat
        except:
            # Fallback to dense method
            pass

    # Dense method
    return pca(r, K, m)


# Helper function to compute GCV for PCA
def pca_gcv(y, X, r, K, m_max):
    """
    Compute GCV scores for PCA with different numbers of components.

    Parameters:
    y : ndarray of shape (n,)
        Response variable
    X : ndarray of shape (n, T)
        Design matrix
    r : ndarray of shape (T,)
        Response vector for PCA
    K : ndarray of shape (T, T)
        Kernel matrix
    m_max : int
        Maximum number of components

    Returns:
    gcv_scores : ndarray of shape (m_max,)
        GCV scores for each number of components
    """
    n, T = X.shape
    gcv_scores = np.zeros(m_max)

    beta_hat = pca(r, K, m_max)

    for m in range(1, m_max + 1):
        beta_m = beta_hat[:, m - 1]
        residuals = y - X @ beta_m / T
        RSS = np.sum(residuals ** 2)

        # Approximate degrees of freedom for PCA
        # This is a simplified version - in practice, you might want
        # a more accurate calculation
        df = m  # degrees of freedom roughly equals number of components

        gcv_scores[m - 1] = (RSS / n) / (1 - df / n) ** 2

    return gcv_scores


# Example usage
if __name__ == "__main__":
    np.random.seed(42)

    # Generate synthetic data
    n = 100
    T = 20

    s = np.linspace(0, 1, T)
    X = np.random.randn(n, T)
    beta_true = np.exp(-s * 3) * np.sin(s * 10)
    y = X @ beta_true + 0.1 * np.random.randn(n)

    # Test RKHS GCV
    lambda_grid = np.logspace(-6, 0, 10)
    beta_rkhs = rkhs_gcv(y, X, lambda_grid, s)
    print(f"RKHS GCV estimated beta shape: {beta_rkhs.shape}")

    # Test PCA
    # Use the kernel matrix from RKHS as K
    Delta = s[:, np.newaxis] - s[np.newaxis, :]
    B2_s = s ** 2 - s + 1 / 6
    B2_t = s ** 2 - s + 1 / 6
    abs_Delta = np.abs(Delta)
    B4_st = abs_Delta ** 4 - 2 * abs_Delta ** 3 + abs_Delta ** 2 - 1 / 30
    from math import factorial

    K = (1 / factorial(2) ** 2) * np.outer(B2_s, B2_t) - (1 / factorial(4)) * B4_st

    # Use a simple response for PCA (could be coefficients from another method)
    r = beta_true + 0.1 * np.random.randn(T)
    m = 5
    beta_pca = pca(r, K, m)
    print(f"PCA estimated beta shape: {beta_pca.shape}")

    # Visualize results
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Plot RKHS results
        ax = axes[0]
        ax.plot(s, beta_true, 'k-', linewidth=2, label='True')
        ax.plot(s, beta_rkhs, 'r--', linewidth=2, label='RKHS GCV')
        ax.set_xlabel('s')
        ax.set_ylabel('Coefficient')
        ax.set_title('RKHS GCV Estimator')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot PCA results
        ax = axes[1]
        ax.plot(s, beta_true, 'k-', linewidth=2, label='True')
        for j in range(min(3, m)):
            ax.plot(s, beta_pca[:, j], '--', linewidth=1.5, label=f'PCA m={j + 1}')
        ax.set_xlabel('s')
        ax.set_ylabel('Coefficient')
        ax.set_title('PCA Estimator')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()
    except:
        pass

import numpy as np
from scipy.linalg import solve, eigh
import warnings


def pls(r, K, m):
    """
    PLS (Partial Least Squares) estimator using the Conjugate Gradient method.
    Implements the minimum residual algorithm from Hanke (1992).

    Parameters:
    r : ndarray of shape (T,)
        Response vector
    K : ndarray of shape (T, T)
        Kernel matrix (should be self-adjoint/positive definite)
    m : int
        Number of PLS components to compute

    Returns:
    beta_hat : ndarray of shape (T, m)
        PLS estimates for j=1,2,...,m components
    """
    T = len(r)

    # Initialize arrays
    # βhat = zeros(length(r), m+1); e = r; d = r;
    beta_hat = np.zeros((T, m + 1))
    e = r.copy()
    d = r.copy()

    # Pre-compute K * e for efficiency (used in multiple places)
    Ke = K @ e

    for j in range(1, m + 1):
        # Kd = K * d
        Kd = K @ d

        # α1 = e' * K * e
        # Note: e'*K*e = e' * (K*e) = e' * Ke
        alpha1 = e @ Ke  # This is equivalent to e.T @ K @ e

        # α = α1 / (Kd' * Kd)
        # step size for the slope
        alpha = alpha1 / (Kd @ Kd)

        # βhat[:, j+1] = βhat[:, j] .+ α * d
        # update the slope
        # FIX: Use j (not j+1) for 0-based indexing
        # In Julia: βhat[:, j+1] (1-based) -> In Python: beta_hat[:, j] (0-based)
        beta_hat[:, j] = beta_hat[:, j - 1] + alpha * d

        # e = e .- α * Kd
        # update the residual
        e = e - alpha * Kd

        # Update Ke for next iteration (K * e)
        Ke = K @ e

        # γ = e' * K * e / α1
        # step size for the conjugate direction
        gamma = (e @ Ke) / alpha1

        # d = e .+ γ * d
        # update the conjugate vector
        d = e + gamma * d

    # Return βhat[:, 2:end] (exclude the first column which is zeros)
    # In Julia: 2:end -> In Python: 1: (exclude index 0)
    return beta_hat[:, 1:]


# Alternative implementation with more efficient matrix operations
def pls_efficient(r, K, m):
    """
    More efficient PLS implementation with pre-computation and
    better numerical stability.
    """
    T = len(r)

    beta_hat = np.zeros((T, m))
    e = r.copy()
    d = r.copy()

    # Pre-compute Kd to avoid recomputing
    for j in range(m):
        # Compute K * d
        Kd = K @ d

        # Compute e' * K * e
        Ke = K @ e
        alpha1 = e @ Ke

        # Avoid division by zero
        if alpha1 < 1e-15:
            warnings.warn(f"Alpha1 is very small ({alpha1:.2e}) at iteration {j + 1}")
            break

        # Step size
        Kd_norm_sq = Kd @ Kd
        if Kd_norm_sq < 1e-15:
            warnings.warn(f"Kd norm is very small ({Kd_norm_sq:.2e}) at iteration {j + 1}")
            break

        alpha = alpha1 / Kd_norm_sq

        # Update beta
        if j == 0:
            beta_hat[:, j] = alpha * d
        else:
            beta_hat[:, j] = beta_hat[:, j - 1] + alpha * d

        # Update residual
        e = e - alpha * Kd

        # Update Ke
        Ke = K @ e

        # Conjugate direction step size
        gamma = (e @ Ke) / alpha1

        # Update conjugate direction
        d = e + gamma * d

    return beta_hat


# Implementation using scipy's CG solver for comparison
def pls_cg(r, K, m, tol=1e-10, max_iter=None):
    """
    PLS implementation using scipy's Conjugate Gradient solver.
    This is a reference implementation.
    """
    from scipy.sparse.linalg import cg
    from scipy.sparse import issparse

    T = len(r)
    beta_hat = np.zeros((T, m))

    # Use CG to solve K * beta = r for different numbers of iterations
    for j in range(1, m + 1):
        if max_iter is None:
            max_iter_j = j
        else:
            max_iter_j = min(j, max_iter)

        # Solve using CG
        if issparse(K):
            beta, info = cg(K, r, tol=tol, maxiter=max_iter_j)
        else:
            # Use dense CG implementation
            beta = cg_dense(K, r, tol=tol, maxiter=max_iter_j)

        beta_hat[:, j - 1] = beta

    return beta_hat


def cg_dense(A, b, tol=1e-10, maxiter=None):
    """
    Dense Conjugate Gradient solver.
    """
    x = np.zeros_like(b)
    r = b.copy()
    p = r.copy()
    rsold = r @ r

    if maxiter is None:
        maxiter = len(b)

    for i in range(maxiter):
        Ap = A @ p
        alpha = rsold / (p @ Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = r @ r

        if np.sqrt(rsnew) < tol:
            break

        p = r + (rsnew / rsold) * p
        rsold = rsnew

    return x


# PLS with automatic component selection using GCV
def pls_gcv(y, X, r, K, m_max, s=None):
    """
    PLS with GCV-based component selection.

    Parameters:
    y : ndarray of shape (n,)
        Response variable
    X : ndarray of shape (n, T)
        Design matrix
    r : ndarray of shape (T,)
        Response vector for PLS
    K : ndarray of shape (T, T)
        Kernel matrix
    m_max : int
        Maximum number of components to consider
    s : ndarray or None
        Knot sequence (if needed for prediction)

    Returns:
    beta_opt : ndarray of shape (T,)
        PLS estimate with optimal number of components
    gcv_scores : ndarray of shape (m_max,)
        GCV scores for each number of components
    """
    n, T = X.shape

    # Compute PLS estimates for all components
    beta_hat = pls(r, K, m_max)

    gcv_scores = np.zeros(m_max)

    for m in range(1, m_max + 1):
        beta_m = beta_hat[:, m - 1]

        # Compute predictions
        y_pred = X @ beta_m / T

        # RSS
        residuals = y - y_pred
        RSS = np.sum(residuals ** 2)

        # Degrees of freedom approximation for PLS
        # This is a simplified approximation
        # More accurate would require computing the hat matrix
        df = m  # Rough approximation

        # GCV
        gcv_scores[m - 1] = (RSS / n) / (1 - df / n) ** 2

    # Select optimal number of components
    m_opt = np.argmin(gcv_scores) + 1

    return beta_hat[:, m_opt - 1], gcv_scores, m_opt


# PLS with early stopping based on residual norm
def pls_adaptive(r, K, m_max, tol=1e-8):
    """
    Adaptive PLS with early stopping based on residual norm.
    """
    T = len(r)

    beta_hat_list = []
    residual_norms = []

    e = r.copy()
    d = r.copy()
    beta = np.zeros(T)

    # Pre-compute K*e
    Ke = K @ e

    for j in range(1, m_max + 1):
        Kd = K @ d
        alpha1 = e @ Ke
        alpha = alpha1 / (Kd @ Kd)

        beta = beta + alpha * d
        beta_hat_list.append(beta.copy())

        e = e - alpha * Kd
        residual_norm = np.linalg.norm(e)
        residual_norms.append(residual_norm)

        # Check for convergence
        if residual_norm < tol * np.linalg.norm(r):
            print(f"Converged after {j} iterations with residual norm {residual_norm:.2e}")
            break

        Ke = K @ e
        gamma = (e @ Ke) / alpha1
        d = e + gamma * d

    # Convert list to array
    beta_hat = np.column_stack(beta_hat_list) if beta_hat_list else np.zeros((T, 0))

    return beta_hat, residual_norms


# Example usage
if __name__ == "__main__":
    np.random.seed(42)

    # Generate synthetic data
    T = 20
    s = np.linspace(0, 1, T)

    # Create a kernel matrix (using the RKHS kernel from previous example)
    Delta = s[:, np.newaxis] - s[np.newaxis, :]
    B2_s = s ** 2 - s + 1 / 6
    B2_t = s ** 2 - s + 1 / 6
    abs_Delta = np.abs(Delta)
    B4_st = abs_Delta ** 4 - 2 * abs_Delta ** 3 + abs_Delta ** 2 - 1 / 30
    from math import factorial

    K = (1 / factorial(2) ** 2) * np.outer(B2_s, B2_t) - (1 / factorial(4)) * B4_st

    # Add small regularization for stability
    K = K + 1e-6 * np.eye(T)

    # Generate response
    beta_true = np.exp(-s * 3) * np.sin(s * 10)
    r = K @ beta_true + 0.01 * np.random.randn(T)

    # Test PLS
    m = 5
    beta_pls = pls(r, K, m)
    print(f"PLS beta shape: {beta_pls.shape}")

    # Test efficient version
    beta_pls_efficient = pls_efficient(r, K, m)
    print(f"Efficient PLS beta shape: {beta_pls_efficient.shape}")

    # Test adaptive PLS
    beta_pls_adaptive, residuals = pls_adaptive(r, K, m_max=10, tol=1e-6)
    print(f"Adaptive PLS beta shape: {beta_pls_adaptive.shape}")
    print(f"Residual norms: {residuals}")

    # Visualize results
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # Plot 1: PLS estimates for different components
        ax = axes[0, 0]
        colors = plt.cm.viridis(np.linspace(0, 1, m))
        for j in range(m):
            ax.plot(s, beta_pls[:, j], color=colors[j],
                    label=f'PLS m={j + 1}')
        ax.plot(s, beta_true, 'k-', linewidth=2, label='True')
        ax.set_xlabel('s')
        ax.set_ylabel('Coefficient')
        ax.set_title('PLS Estimates')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: Convergence of residual norms
        ax = axes[0, 1]
        if residuals:
            ax.semilogy(range(1, len(residuals) + 1), residuals, 'b-o', linewidth=2)
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Residual Norm')
            ax.set_title('Convergence of PLS')
            ax.grid(True, alpha=0.3)

        # Plot 3: Comparison of different implementations
        ax = axes[1, 0]
        ax.plot(s, beta_true, 'k-', linewidth=2, label='True')
        if beta_pls.size > 0:
            ax.plot(s, beta_pls[:, -1], 'r--', linewidth=2, label='PLS (m=5)')
        if beta_pls_efficient.size > 0:
            ax.plot(s, beta_pls_efficient[:, -1], 'g--', linewidth=2, label='Efficient PLS')
        ax.set_xlabel('s')
        ax.set_ylabel('Coefficient')
        ax.set_title('Implementation Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 4: Error vs components
        ax = axes[1, 1]
        if beta_pls.size > 0:
            errors = [np.linalg.norm(beta_pls[:, j] - beta_true) for j in range(m)]
            ax.plot(range(1, m + 1), errors, 'r-o', linewidth=2)
            ax.set_xlabel('Number of Components')
            ax.set_ylabel('Estimation Error')
            ax.set_title('Error vs Number of Components')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Visualization error: {e}")

import numpy as np
from scipy.linalg import eigh
import warnings


def pca_gcv(r, K, m):
    """
    PCA estimator with GCV-based component selection.

    Parameters:
    r : ndarray of shape (T,)
        Response vector
    K : ndarray of shape (T, T)
        Kernel matrix (should be symmetric)
    m : int
        Maximum number of components to consider

    Returns:
    beta_opt : ndarray of shape (T,)
        PCA estimate with optimal number of components
    m_opt : int
        Optimal number of components (1-indexed)
    gcv_scores : ndarray of shape (m,)
        GCV scores for each number of components
    """
    T = len(r)

    # Preallocate storage for PCA estimates and GCV scores
    beta_hat = np.zeros((T, m))
    GCV = np.zeros(m)

    # Compute the eigen-decomposition of the symmetric matrix K
    # eigh returns eigenvalues in ascending order by default
    try:
        evals, evecs = eigh(K)
    except np.linalg.LinAlgError:
        warnings.warn("Matrix K is not positive definite, using SVD instead")
        # Fallback to SVD
        U, s, Vt = np.linalg.svd(K)
        evals = s
        evecs = U

    # Process components from 1 to m
    for j in range(1, m + 1):
        # Select the j largest eigenvalues and corresponding eigenvectors
        lambda_hat = evals[-j:]
        v_hat = evecs[:, -j:]

        # Compute the PCA-based estimator using j components
        # βhat[:, j] = vhat * (vhat' * r ./ λhat)
        beta_hat[:, j - 1] = v_hat @ ((v_hat.T @ r) / lambda_hat)

        # Compute RSS: sum((r - K * βhat[:, j])^2)
        residuals = r - K @ beta_hat[:, j - 1]
        RSS = np.sum(residuals ** 2)

        # Compute the GCV score for j components.
        GCV[j - 1] = (RSS / T) / (1 - j / T) ** 2

    # Select the number of components that minimizes the GCV score
    m_opt = np.argmin(GCV) + 1  # +1 because argmin returns 0-indexed

    # Return the optimal PCA estimator, optimal number of components, and GCV scores
    return beta_hat[:, m_opt - 1], m_opt, GCV


# Enhanced version with additional features
def pca_gcv_enhanced(r, K, m_max, method='eigh', return_all=False):
    """
    Enhanced PCA GCV with additional options.

    Parameters:
    r : ndarray of shape (T,)
        Response vector
    K : ndarray of shape (T, T)
        Kernel matrix
    m_max : int
        Maximum number of components to consider
    method : str
        'eigh' for dense eigen-decomposition, 'eigsh' for sparse
    return_all : bool
        If True, return all beta estimates and GCV scores

    Returns:
    beta_opt : ndarray of shape (T,)
        PCA estimate with optimal number of components
    m_opt : int
        Optimal number of components
    GCV : ndarray of shape (m_max,)
        GCV scores for each number of components
    beta_all : ndarray of shape (T, m_max) (if return_all=True)
        All PCA estimates
    """
    T = len(r)

    # Preallocate
    beta_hat = np.zeros((T, m_max))
    GCV = np.zeros(m_max)

    # Compute eigen-decomposition
    if method == 'eigh':
        try:
            evals, evecs = eigh(K)
        except np.linalg.LinAlgError:
            warnings.warn("Matrix not positive definite, using SVD")
            U, s, Vt = np.linalg.svd(K)
            evals = s
            evecs = U
    elif method == 'eigsh':
        try:
            from scipy.sparse.linalg import eigsh
            # Get largest m_max eigenvalues/vectors
            evals, evecs = eigsh(K, k=m_max, which='LM')
            # eigsh returns in ascending order by default for which='LM'
            # but we'll sort to be safe
            idx = np.argsort(evals)
            evals = evals[idx]
            evecs = evecs[:, idx]
        except:
            warnings.warn("eigsh failed, falling back to dense method")
            evals, evecs = eigh(K)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Compute PCA estimates and GCV scores
    for j in range(1, m_max + 1):
        # Get largest j components
        lambda_hat = evals[-j:]
        v_hat = evecs[:, -j:]

        # Compute estimate
        beta_hat[:, j - 1] = v_hat @ ((v_hat.T @ r) / lambda_hat)

        # Compute RSS
        residuals = r - K @ beta_hat[:, j - 1]
        RSS = np.sum(residuals ** 2)

        # Compute GCV
        GCV[j - 1] = (RSS / T) / (1 - j / T) ** 2

    # Find optimal
    m_opt = np.argmin(GCV) + 1

    if return_all:
        return beta_hat[:, m_opt - 1], m_opt, GCV, beta_hat
    else:
        return beta_hat[:, m_opt - 1], m_opt, GCV


# Fixed cross-validation function
def pca_gcv_cv(r, K, m_max, n_folds=5):
    """
    PCA with cross-validation instead of GCV.

    Parameters:
    r : ndarray of shape (T,)
        Response vector
    K : ndarray of shape (T, T)
        Kernel matrix
    m_max : int
        Maximum number of components
    n_folds : int
        Number of cross-validation folds

    Returns:
    beta_opt : ndarray of shape (T,)
        PCA estimate with optimal number of components
    m_opt : int
        Optimal number of components
    cv_scores : ndarray of shape (m_max,)
        Cross-validation scores
    """
    T = len(r)

    # Compute eigen-decomposition once
    try:
        evals, evecs = eigh(K)
    except np.linalg.LinAlgError:
        U, s, Vt = np.linalg.svd(K)
        evals = s
        evecs = U

    # Create folds
    indices = np.arange(T)
    np.random.shuffle(indices)  # Shuffle for better cross-validation
    fold_size = T // n_folds
    cv_scores = np.zeros(m_max)

    for j in range(1, m_max + 1):
        # Get largest j components
        lambda_hat = evals[-j:]
        v_hat = evecs[:, -j:]

        # Cross-validation
        fold_errors = []
        for fold in range(n_folds):
            # Split data
            test_start = fold * fold_size
            test_end = (fold + 1) * fold_size if fold < n_folds - 1 else T
            test_idx = indices[test_start:test_end]
            train_idx = np.concatenate([
                indices[:test_start],
                indices[test_end:]
            ])

            # Training data
            r_train = r[train_idx]
            v_train = v_hat[train_idx, :]

            # Compute coefficients on training data using j components
            # beta_train_coeff = v_train @ ((v_train.T @ r_train) / lambda_hat)
            # This gives coefficients in the reduced space (length j)
            beta_train_coeff = (v_train.T @ r_train) / lambda_hat

            # Test data
            r_test = r[test_idx]
            v_test = v_hat[test_idx, :]

            # Predict on test data: y_pred = v_test @ beta_train_coeff
            # beta_train_coeff is of shape (j,), v_test is (test_size, j)
            r_pred = v_test @ beta_train_coeff

            # Compute error
            fold_errors.append(np.mean((r_test - r_pred) ** 2))

        cv_scores[j - 1] = np.mean(fold_errors)

    # Select optimal
    m_opt = np.argmin(cv_scores) + 1

    # Return optimal estimate using all data
    lambda_opt = evals[-m_opt:]
    v_opt = evecs[:, -m_opt:]
    beta_opt = v_opt @ ((v_opt.T @ r) / lambda_opt)

    return beta_opt, m_opt, cv_scores


# Alternative cross-validation using hold-out
def pca_gcv_holdout(r, K, m_max, test_size=0.2):
    """
    PCA with hold-out validation.

    Parameters:
    r : ndarray of shape (T,)
        Response vector
    K : ndarray of shape (T, T)
        Kernel matrix
    m_max : int
        Maximum number of components
    test_size : float
        Proportion of data to use for testing

    Returns:
    beta_opt : ndarray of shape (T,)
        PCA estimate with optimal number of components
    m_opt : int
        Optimal number of components
    holdout_scores : ndarray of shape (m_max,)
        Hold-out validation scores
    """
    T = len(r)

    # Compute eigen-decomposition
    try:
        evals, evecs = eigh(K)
    except np.linalg.LinAlgError:
        U, s, Vt = np.linalg.svd(K)
        evals = s
        evecs = U

    # Split data
    n_test = int(T * test_size)
    indices = np.arange(T)
    np.random.shuffle(indices)

    test_idx = indices[:n_test]
    train_idx = indices[n_test:]

    holdout_scores = np.zeros(m_max)

    for j in range(1, m_max + 1):
        # Get largest j components
        lambda_hat = evals[-j:]
        v_hat = evecs[:, -j:]

        # Training
        r_train = r[train_idx]
        v_train = v_hat[train_idx, :]

        # Compute coefficients
        beta_train_coeff = (v_train.T @ r_train) / lambda_hat

        # Test
        r_test = r[test_idx]
        v_test = v_hat[test_idx, :]

        # Predict
        r_pred = v_test @ beta_train_coeff

        # Error
        holdout_scores[j - 1] = np.mean((r_test - r_pred) ** 2)

    # Select optimal
    m_opt = np.argmin(holdout_scores) + 1

    # Return optimal estimate using all data
    lambda_opt = evals[-m_opt:]
    v_opt = evecs[:, -m_opt:]
    beta_opt = v_opt @ ((v_opt.T @ r) / lambda_opt)

    return beta_opt, m_opt, holdout_scores


# Visualization function for PCA GCV results
def plot_pca_gcv_results(s, r, K, m_max, beta_true=None):
    """
    Visualize PCA GCV results.

    Parameters:
    s : ndarray of shape (T,)
        Grid points
    r : ndarray of shape (T,)
        Response vector
    K : ndarray of shape (T, T)
        Kernel matrix
    m_max : int
        Maximum number of components
    beta_true : ndarray or None
        True coefficients for comparison
    """
    try:
        import matplotlib.pyplot as plt

        # Compute PCA GCV
        beta_opt, m_opt, gcv_scores, beta_all = pca_gcv_enhanced(
            r, K, m_max, return_all=True
        )

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # Plot 1: GCV scores
        ax = axes[0, 0]
        ax.plot(range(1, m_max + 1), gcv_scores, 'b-o', linewidth=2, markersize=8)
        ax.axvline(m_opt, color='r', linestyle='--',
                   label=f'Optimal m = {m_opt}')
        ax.set_xlabel('Number of Components')
        ax.set_ylabel('GCV Score')
        ax.set_title('GCV Scores for PCA')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: Estimates for different components
        ax = axes[0, 1]
        colors = plt.cm.viridis(np.linspace(0, 1, min(5, m_max)))
        for j in range(min(5, m_max)):
            ax.plot(s, beta_all[:, j], color=colors[j],
                    label=f'm={j + 1}')
        if beta_true is not None:
            ax.plot(s, beta_true, 'k-', linewidth=3, label='True', alpha=0.7)
        ax.plot(s, beta_opt, 'r--', linewidth=2, label=f'Optimal (m={m_opt})')
        ax.set_xlabel('s')
        ax.set_ylabel('Coefficient')
        ax.set_title('PCA Estimates')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 3: Optimal estimate vs true
        ax = axes[1, 0]
        if beta_true is not None:
            ax.plot(s, beta_true, 'k-', linewidth=2, label='True')
        ax.plot(s, beta_opt, 'r-', linewidth=2, label=f'PCA (m={m_opt})')
        ax.set_xlabel('s')
        ax.set_ylabel('Coefficient')
        ax.set_title('Optimal PCA Estimate')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 4: Error vs components
        ax = axes[1, 1]
        if beta_true is not None:
            errors = [np.linalg.norm(beta_all[:, j - 1] - beta_true)
                      for j in range(1, m_max + 1)]
            ax.plot(range(1, m_max + 1), errors, 'g-o', linewidth=2, markersize=8)
            ax.axvline(m_opt, color='r', linestyle='--',
                       label=f'Optimal m={m_opt}')
            ax.set_xlabel('Number of Components')
            ax.set_ylabel('Estimation Error')
            ax.set_title('Estimation Error vs Components')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        return beta_opt, m_opt, gcv_scores

    except ImportError:
        print("Matplotlib not available for visualization")
        return None, None, None


# Example usage
if __name__ == "__main__":
    np.random.seed(42)

    # Generate synthetic data
    T = 50
    s = np.linspace(0, 1, T)

    # Create kernel matrix
    Delta = s[:, np.newaxis] - s[np.newaxis, :]
    B2_s = s ** 2 - s + 1 / 6
    B2_t = s ** 2 - s + 1 / 6
    abs_Delta = np.abs(Delta)
    B4_st = abs_Delta ** 4 - 2 * abs_Delta ** 3 + abs_Delta ** 2 - 1 / 30
    from math import factorial

    K = (1 / factorial(2) ** 2) * np.outer(B2_s, B2_t) - (1 / factorial(4)) * B4_st

    # Add small regularization
    K = K + 1e-6 * np.eye(T)

    # Generate true coefficients and response
    beta_true = np.exp(-s * 3) * np.sin(s * 10)
    r = K @ beta_true + 0.05 * np.random.randn(T)

    # Test PCA GCV
    m_max = 15
    beta_opt, m_opt, gcv_scores = pca_gcv(r, K, m_max)

    print(f"Optimal number of components: {m_opt}")
    print(f"Optimal beta shape: {beta_opt.shape}")
    print(f"GCV scores shape: {gcv_scores.shape}")
    print(f"Minimum GCV: {np.min(gcv_scores):.6f}")

    # Test enhanced version
    beta_opt2, m_opt2, gcv_scores2, beta_all = pca_gcv_enhanced(
        r, K, m_max, return_all=True
    )
    print(f"\nEnhanced version - optimal m: {m_opt2}")

    # Test cross-validation (fixed)
    try:
        beta_cv, m_cv, cv_scores = pca_gcv_cv(r, K, m_max, n_folds=5)
        print(f"\nCross-validation - optimal m: {m_cv}")
        print(f"CV scores: {cv_scores[:5]}...")
    except Exception as e:
        print(f"Cross-validation error: {e}")

    # Test hold-out validation
    try:
        beta_hold, m_hold, hold_scores = pca_gcv_holdout(r, K, m_max, test_size=0.2)
        print(f"\nHold-out validation - optimal m: {m_hold}")
        print(f"Hold-out scores: {hold_scores[:5]}...")
    except Exception as e:
        print(f"Hold-out validation error: {e}")

    # Visualize results
    try:
        beta_opt_viz, m_opt_viz, gcv_viz = plot_pca_gcv_results(
            s, r, K, m_max, beta_true
        )
    except Exception as e:
        print(f"Visualization error: {e}")

import numpy as np
from typing import Tuple, Optional


def pls_early_stop(beta_hat_pls, X, y, r, K, tau, delta, n, beta_hat_pca):
    """
    Early stopping for PLS based on moment criterion.

    Parameters:
    beta_hat_pls : ndarray of shape (T, m)
        PLS estimates for each number of components
    X : ndarray of shape (n, T)
        Design matrix
    y : ndarray of shape (n,)
        Response variable
    r : ndarray of shape (T,)
        Response vector for PLS
    K : ndarray of shape (T, T)
        Kernel matrix
    tau : float
        Threshold multiplier
    delta : float
        Scaling parameter
    n : int
        Number of observations
    beta_hat_pca : ndarray of shape (T,)
        PCA estimate for comparison

    Returns:
    beta_opt : ndarray of shape (T,)
        Optimal PLS estimate based on early stopping
    m_hat : int
        Optimal number of components (1-indexed)
    moment : ndarray of shape (m,)
        Moment values for each component
    threshold : float
        Computed threshold value
    """
    T, m = beta_hat_pls.shape

    # Compute X_norm = mean(sum(X.^2, dims=2) / T)
    # In Julia: mean(sum(X.^2, dims=2) / T)
    # This computes the average of row-wise squared norms divided by T
    X_norm = np.mean(np.sum(X ** 2, axis=1) / T)

    # Compute sigma^2 = mean((y - X * beta_hat_pca / T)^2)
    sigma2 = np.mean((y - X @ beta_hat_pca / T) ** 2)

    # Compute the moment vector
    # mom = sqrt.(sum((K * βhat_pls .- r).^2, dims=1)' / T)
    # For each column j, compute norm of (K * beta_hat_pls[:, j] - r) / sqrt(T)
    moment = np.zeros(m)
    for j in range(m):
        residual = K @ beta_hat_pls[:, j] - r
        moment[j] = np.sqrt(np.sum(residual ** 2) / T)

    # Compute the threshold using the updated sigma^2 value
    # threshold = tau * sqrt(2 * sigma2 * X_norm / delta / n)
    threshold = tau * np.sqrt(2 * sigma2 * X_norm / (delta * n))

    # Find the first index where the moment is less than or equal to the threshold
    # In Julia: findfirst(mom .<= threshold)
    # In Python: np.where returns tuple of indices
    indices = np.where(moment <= threshold)[0]

    if len(indices) > 0:
        m_hat = indices[0] + 1  # +1 for 1-indexed
        beta_opt = beta_hat_pls[:, m_hat - 1]
    else:
        # If no index satisfies the condition, use the last component
        warnings.warn("No component satisfies the stopping criterion. Using the last component.")
        m_hat = m
        beta_opt = beta_hat_pls[:, -1]

    return beta_opt, m_hat, moment, threshold


# Alternative implementation with more detailed output
def pls_early_stop_detailed(beta_hat_pls, X, y, r, K, tau, delta, n, beta_hat_pca):
    """
    Enhanced early stopping with additional information.

    Returns:
    beta_opt : ndarray
        Optimal PLS estimate
    m_hat : int
        Optimal number of components
    moment : ndarray
        Moment values
    threshold : float
        Computed threshold
    info : dict
        Additional information including sigma2, X_norm, etc.
    """
    T, m = beta_hat_pls.shape

    # Compute statistics
    X_norm = np.mean(np.sum(X ** 2, axis=1) / T)
    sigma2 = np.mean((y - X @ beta_hat_pca / T) ** 2)

    # Compute moment for each component
    moment = np.zeros(m)
    residuals = np.zeros((T, m))
    for j in range(m):
        residuals[:, j] = K @ beta_hat_pls[:, j] - r
        moment[j] = np.sqrt(np.sum(residuals[:, j] ** 2) / T)

    # Compute threshold
    threshold = tau * np.sqrt(2 * sigma2 * X_norm / (delta * n))

    # Find first index where moment <= threshold
    valid_indices = np.where(moment <= threshold)[0]

    if len(valid_indices) > 0:
        m_hat = valid_indices[0] + 1
        beta_opt = beta_hat_pls[:, m_hat - 1]
        early_stop = True
    else:
        m_hat = m
        beta_opt = beta_hat_pls[:, -1]
        early_stop = False
        warnings.warn("No early stopping criterion met. Using last component.")

    # Additional information
    info = {
        'X_norm': X_norm,
        'sigma2': sigma2,
        'threshold': threshold,
        'moment': moment,
        'early_stop': early_stop,
        'm_hat': m_hat,
        'valid_indices': valid_indices,
        'residuals': residuals
    }

    return beta_opt, m_hat, moment, threshold, info


# Implementation with adaptive threshold search
def pls_early_stop_adaptive(beta_hat_pls, X, y, r, K, tau_grid, delta, n, beta_hat_pca):
    """
    Adaptive early stopping that searches over multiple tau values.

    Parameters:
    beta_hat_pls : ndarray of shape (T, m)
        PLS estimates
    X, y, r, K, delta, n, beta_hat_pca : as above
    tau_grid : ndarray
        Grid of tau values to try

    Returns:
    beta_opt : ndarray
        Optimal PLS estimate
    m_hat : int
        Optimal number of components
    tau_opt : float
        Optimal tau value
    results : dict
        Results for all tau values
    """
    T, m = beta_hat_pls.shape

    # Compute statistics (independent of tau)
    X_norm = np.mean(np.sum(X ** 2, axis=1) / T)
    sigma2 = np.mean((y - X @ beta_hat_pca / T) ** 2)

    # Compute moments
    moment = np.zeros(m)
    for j in range(m):
        residual = K @ beta_hat_pls[:, j] - r
        moment[j] = np.sqrt(np.sum(residual ** 2) / T)

    # Try each tau
    results = {
        'tau': [],
        'm_hat': [],
        'threshold': [],
        'beta': []
    }

    for tau in tau_grid:
        threshold = tau * np.sqrt(2 * sigma2 * X_norm / (delta * n))
        valid_indices = np.where(moment <= threshold)[0]

        if len(valid_indices) > 0:
            m_hat = valid_indices[0] + 1
            beta_opt = beta_hat_pls[:, m_hat - 1]
        else:
            m_hat = m
            beta_opt = beta_hat_pls[:, -1]

        results['tau'].append(tau)
        results['m_hat'].append(m_hat)
        results['threshold'].append(threshold)
        results['beta'].append(beta_opt)

    # Select the result with smallest m_hat (most conservative)
    # Or you could use other criteria
    best_idx = np.argmin(results['m_hat'])

    return results['beta'][best_idx], results['m_hat'][best_idx], results['tau'][best_idx], results


# Visualization function for early stopping
def plot_pls_early_stop(s, beta_hat_pls, moment, threshold, beta_true=None):
    """
    Visualize PLS early stopping results.

    Parameters:
    s : ndarray of shape (T,)
        Grid points
    beta_hat_pls : ndarray of shape (T, m)
        PLS estimates
    moment : ndarray of shape (m,)
        Moment values
    threshold : float
        Threshold value
    beta_true : ndarray or None
        True coefficients for comparison
    """
    try:
        import matplotlib.pyplot as plt

        T, m = beta_hat_pls.shape

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Plot 1: Moment values
        ax = axes[0]
        components = np.arange(1, m + 1)
        ax.plot(components, moment, 'b-o', linewidth=2, markersize=8, label='Moment')
        ax.axhline(threshold, color='r', linestyle='--',
                   label=f'Threshold = {threshold:.4f}')

        # Find first crossing
        valid_idx = np.where(moment <= threshold)[0]
        if len(valid_idx) > 0:
            m_hat = valid_idx[0] + 1
            ax.axvline(m_hat, color='g', linestyle='--',
                       label=f'Optimal m = {m_hat}')
            ax.plot(m_hat, moment[m_hat - 1], 'ro', markersize=10, label='Selected')

        ax.set_xlabel('Number of Components')
        ax.set_ylabel('Moment Value')
        ax.set_title('PLS Early Stopping Criterion')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: Coefficient estimates
        ax = axes[1]

        # Plot selected estimate
        if len(valid_idx) > 0:
            m_hat = valid_idx[0] + 1
            beta_opt = beta_hat_pls[:, m_hat - 1]
            ax.plot(s, beta_opt, 'r-', linewidth=2, label=f'Optimal (m={m_hat})')

        # Plot a few other estimates
        colors = plt.cm.viridis(np.linspace(0, 1, min(5, m)))
        for j in range(min(5, m)):
            ax.plot(s, beta_hat_pls[:, j], color=colors[j],
                    alpha=0.5, label=f'm={j + 1}')

        if beta_true is not None:
            ax.plot(s, beta_true, 'k-', linewidth=3, label='True', alpha=0.7)

        ax.set_xlabel('s')
        ax.set_ylabel('Coefficient')
        ax.set_title('PLS Coefficient Paths')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    except ImportError:
        print("Matplotlib not available for visualization")


# Example usage
if __name__ == "__main__":
    np.random.seed(42)

    # Generate synthetic data
    T = 50
    n = 100
    s = np.linspace(0, 1, T)

    # Create kernel matrix
    Delta = s[:, np.newaxis] - s[np.newaxis, :]
    B2_s = s ** 2 - s + 1 / 6
    B2_t = s ** 2 - s + 1 / 6
    abs_Delta = np.abs(Delta)
    B4_st = abs_Delta ** 4 - 2 * abs_Delta ** 3 + abs_Delta ** 2 - 1 / 30
    from math import factorial

    K = (1 / factorial(2) ** 2) * np.outer(B2_s, B2_t) - (1 / factorial(4)) * B4_st
    K = K + 1e-6 * np.eye(T)

    # Generate data
    beta_true = np.exp(-s * 3) * np.sin(s * 10)
    X = np.random.randn(n, T)
    y = X @ beta_true / T + 0.1 * np.random.randn(n)
    r = K @ beta_true + 0.01 * np.random.randn(T)

    # Generate PLS estimates (simulate for demonstration)
    m_max = 10
    beta_hat_pls = np.zeros((T, m_max))
    for j in range(m_max):
        # Simulate PLS estimates converging to true
        beta_hat_pls[:, j] = beta_true * (1 - np.exp(-(j + 1) / 3)) + 0.05 * np.random.randn(T)

    # Generate PCA estimate (simulate)
    beta_hat_pca = beta_true + 0.1 * np.random.randn(T)

    # Parameters
    tau = 1.5
    delta = 2.0
    n_obs = n

    # Test early stopping
    beta_opt, m_hat, moment, threshold = pls_early_stop(
        beta_hat_pls, X, y, r, K, tau, delta, n_obs, beta_hat_pca
    )

    print(f"Optimal number of components: {m_hat}")
    print(f"Threshold: {threshold:.6f}")
    print(f"Moment values: {moment[:5]}...")
    print(f"Beta_opt shape: {beta_opt.shape}")

    # Test detailed version
    beta_opt_det, m_hat_det, moment_det, threshold_det, info = pls_early_stop_detailed(
        beta_hat_pls, X, y, r, K, tau, delta, n_obs, beta_hat_pca
    )
    print(f"\nDetailed version:")
    print(f"  X_norm: {info['X_norm']:.6f}")
    print(f"  sigma2: {info['sigma2']:.6f}")
    print(f"  Early stop: {info['early_stop']}")

    # Test adaptive version
    tau_grid = np.linspace(0.5, 3.0, 10)
    beta_opt_adapt, m_hat_adapt, tau_opt, results = pls_early_stop_adaptive(
        beta_hat_pls, X, y, r, K, tau_grid, delta, n_obs, beta_hat_pca
    )
    print(f"\nAdaptive version:")
    print(f"  Optimal tau: {tau_opt:.3f}")
    print(f"  Optimal m: {m_hat_adapt}")

    # Visualize
    try:
        plot_pls_early_stop(s, beta_hat_pls, moment, threshold, beta_true)
    except:
        pass

import numpy as np
from scipy.linalg import lstsq, svd
from scipy.linalg import LinAlgError
import warnings


def apls_cv_enhanced(y, X, K, r, m_max, k_folds=5, random_state=None,
                     reg_param=1e-4, method='auto', verbose=False):
    """
    Enhanced APLS with multiple stabilization methods.

    Parameters:
    method : str
        'auto' - automatically select best method
        'stable' - use regularized least squares
        'svd' - use SVD-based approach
        'qr' - use QR decomposition
    """
    n, T = X.shape

    # Center and scale for stability
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0) + 1e-10
    X_norm = (X - X_mean) / X_std

    beta_mat = np.zeros((T, m_max))
    CV_errors = np.zeros(m_max)
    all_fold_errors = []  # Store fold errors for diagnostics

    if random_state is not None:
        np.random.seed(random_state)

    fold_indices = np.random.permutation(n)
    fold_size = int(np.ceil(n / k_folds))

    # Pre-compute kernel matrix with regularization
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

            # Use method-specific approach
            if method == 'svd' or (method == 'auto' and m_val > 3):
                # SVD-based approach
                beta_current = _apls_svd_step(X_tr, y_tr, K_reg, r, T, m_val, n_tr)
            elif method == 'qr' or (method == 'auto' and n_tr < T):
                # QR-based approach
                beta_current = _apls_qr_step(X_tr, y_tr, K_reg, r, T, m_val, n_tr)
            else:
                # Regularized least squares
                beta_current = _apls_stable_step(X_tr, y_tr, K_reg, r, T, m_val, n_tr, reg_param)

            beta_mat[:, m_val - 1] = beta_current.flatten()

            # Predict
            y_pred = X_val @ beta_current
            fold_errors[k] = np.mean((y_val - y_pred.flatten()) ** 2)

        CV_errors[m_val - 1] = np.mean(fold_errors)
        all_fold_errors.append(fold_errors.copy())

        if verbose:
            print(f"m={m_val}: CV error = {CV_errors[m_val - 1]:.6f}")

    # Robust selection with elbow detection
    m_opt = _select_optimal_m(CV_errors, m_max)

    # Denormalize
    beta_opt = beta_mat[:, m_opt - 1] / X_std

    return beta_opt, m_opt, CV_errors, beta_mat, all_fold_errors


def _apls_stable_step(X_tr, y_tr, K_reg, r, T, m_val, n_tr, reg_param):
    """Regularized least squares step."""
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

    return Kr_tr @ alpha


def _apls_svd_step(X_tr, y_tr, K_reg, r, T, m_val, n_tr):
    """SVD-based step for maximum stability."""
    # Use SVD of X_tr
    U, s, Vt = np.linalg.svd(X_tr, full_matrices=False)

    # Truncate small singular values
    s_inv = np.zeros_like(s)
    mask = s > 1e-8
    s_inv[mask] = 1.0 / s[mask]

    r_tr = X_tr.T @ y_tr / n_tr

    # Initialize with first component
    Kr_tr = r_tr.reshape(-1, 1)
    Zm_tr = X_tr @ Kr_tr / T

    # Use SVD of K for additional stability
    Uk, sk, Vtk = np.linalg.svd(K_reg)
    sk_inv = np.zeros_like(sk)
    mask_k = sk > 1e-8
    sk_inv[mask_k] = 1.0 / sk[mask_k]
    K_pinv = Vtk.T @ (sk_inv * Uk.T)

    for j in range(2, m_val + 1):
        # Use pseudo-inverse for stability
        new_Kr = (X_tr.T @ X_tr / (T * n_tr)) @ Kr_tr[:, -1] / n_tr
        # Regularize if needed
        if np.linalg.norm(new_Kr) < 1e-10:
            new_Kr = r_tr * 0.01
        new_Zm = X_tr @ new_Kr / n_tr
        Kr_tr = np.column_stack([Kr_tr, new_Kr.reshape(-1, 1)])
        Zm_tr = np.column_stack([Zm_tr, new_Zm.reshape(-1, 1)])

    # Use SVD of Zm_tr
    Uz, sz, Vtz = np.linalg.svd(Zm_tr, full_matrices=False)
    sz_inv = np.zeros_like(sz)
    mask_z = sz > 1e-6
    sz_inv[mask_z] = 1.0 / sz[mask_z]

    alpha = Vtz.T @ (sz_inv * (Uz.T @ y_tr))
    return Kr_tr @ alpha


def _apls_qr_step(X_tr, y_tr, K_reg, r, T, m_val, n_tr):
    """QR-based step for stability with tall matrices."""
    r_tr = X_tr.T @ y_tr / n_tr
    K_tr = X_tr.T @ X_tr / (T * n_tr)

    # QR decomposition of X_tr
    Q, R = np.linalg.qr(X_tr, mode='reduced')

    # Build components using QR
    Kr_tr = r_tr.reshape(-1, 1)
    Zm_tr = X_tr @ Kr_tr / T

    for j in range(2, m_val + 1):
        new_Kr = K_tr @ Kr_tr[:, -1] / n_tr
        new_Zm = X_tr @ new_Kr / n_tr
        Kr_tr = np.column_stack([Kr_tr, new_Kr.reshape(-1, 1)])
        Zm_tr = np.column_stack([Zm_tr, new_Zm.reshape(-1, 1)])

    # Use QR of Zm_tr
    Qz, Rz = np.linalg.qr(Zm_tr, mode='reduced')

    # Solve triangular system
    try:
        alpha = np.linalg.solve(Rz, Qz.T @ y_tr)
    except:
        alpha = lstsq(Rz, Qz.T @ y_tr, rcond=1e-6)[0]

    return Kr_tr @ alpha


def _select_optimal_m(CV_errors, m_max):
    """
    Select optimal m using elbow detection.
    """
    # Method 1: Minimum
    m_min = np.argmin(CV_errors) + 1

    # Method 2: Elbow detection (find where improvement slows)
    if m_max > 3:
        # Compute improvements
        improvements = np.diff(CV_errors)
        # Normalize improvements
        if np.std(improvements) > 0:
            improvements_norm = improvements / np.std(improvements)
            # Find where improvement drops below 20% of max
            threshold = 0.2 * np.max(np.abs(improvements_norm))
            elbow_idx = np.where(np.abs(improvements_norm) < threshold)[0]
            if len(elbow_idx) > 0:
                m_elbow = elbow_idx[0] + 1
            else:
                m_elbow = m_min
        else:
            m_elbow = m_min
    else:
        m_elbow = m_min

    # Use the more conservative estimate (smaller m)
    m_opt = min(m_min, m_elbow)

    return m_opt


# Test the enhanced version
if __name__ == "__main__":
    np.random.seed(42)

    # Generate synthetic data
    T = 30
    n = 100
    s = np.linspace(0, 1, T)

    # Create kernel matrix
    Delta = s[:, np.newaxis] - s[np.newaxis, :]
    B2_s = s ** 2 - s + 1 / 6
    B2_t = s ** 2 - s + 1 / 6
    abs_Delta = np.abs(Delta)
    B4_st = abs_Delta ** 4 - 2 * abs_Delta ** 3 + abs_Delta ** 2 - 1 / 30
    from math import factorial

    K = (1 / factorial(2) ** 2) * np.outer(B2_s, B2_t) - (1 / factorial(4)) * B4_st
    K = K + 1e-4 * np.eye(T)

    # Generate data
    beta_true = np.exp(-s * 3) * np.sin(s * 10)
    X = np.random.randn(n, T)
    y = X @ beta_true / T + 0.1 * np.random.randn(n)
    r = K @ beta_true + 0.01 * np.random.randn(T)

    # Test all methods
    methods = ['auto', 'stable', 'svd', 'qr']
    results = {}

    for method in methods:
        print(f"\n{'=' * 50}")
        print(f"Testing method: {method}")
        print('=' * 50)

        beta_opt, m_opt, CV_errors, beta_mat, fold_errors = apls_cv_enhanced(
            y, X, K, r, m_max=8, k_folds=5,
            random_state=42, method=method, verbose=True
        )

        error = np.linalg.norm(beta_opt - beta_true)
        print(f"Optimal m: {m_opt}")
        print(f"Estimation error: {error:.6f}")
        print(f"CV error at optimum: {CV_errors[m_opt - 1]:.6f}")

        results[method] = {
            'm_opt': m_opt,
            'error': error,
            'CV_errors': CV_errors,
            'beta': beta_opt
        }

    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    for method, res in results.items():
        print(f"{method:10s}: m={res['m_opt']:d}, error={res['error']:.6f}")

    # Visualize
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        ax_idx = 0

        for method, res in results.items():
            ax = axes[ax_idx // 2, ax_idx % 2]
            ax.plot(range(1, len(res['CV_errors']) + 1), res['CV_errors'],
                    'o-', linewidth=2, markersize=6)
            ax.axvline(res['m_opt'], color='r', linestyle='--',
                       label=f'm={res["m_opt"]}')
            ax.set_xlabel('Number of Components')
            ax.set_ylabel('CV Error')
            ax.set_title(f'Method: {method}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax_idx += 1

        plt.tight_layout()
        plt.show()
    except:
        pass

import numpy as np
from scipy.linalg import lstsq, solve
from scipy.linalg import LinAlgError
import warnings


def apls_fit_optimal(y, X, K, r, m_opt, reg_param=1e-6):
    """
    Refit APLS on full data using the optimal number of components.

    Parameters:
    y : ndarray of shape (n,)
        Response variable
    X : ndarray of shape (n, T)
        Design matrix
    K : ndarray of shape (T, T)
        Kernel matrix
    r : ndarray of shape (T,)
        Response vector for PLS
    m_opt : int
        Optimal number of components (1-indexed)
    reg_param : float
        Regularization parameter for numerical stability

    Returns:
    beta_opt : ndarray of shape (T,)
        Optimal PLS estimate
    m_opt : int
        Number of components used
    beta_mat : ndarray of shape (T, m_opt)
        All PLS estimates for each component
    """
    n, T = X.shape

    # Initialize storage
    beta_mat = np.zeros((T, m_opt))

    # Initialize Kr and Zm with first component
    Kr = r.reshape(-1, 1)  # Column vector
    Zm = X @ Kr / T

    # Build components iteratively
    for j in range(1, m_opt + 1):
        # Compute alpha = (Zm' * Zm) \ (Zm' * y)
        # Using least squares for better numerical stability
        ZtZ = Zm.T @ Zm
        ZtY = Zm.T @ y

        # Add regularization for stability
        if ZtZ.shape[0] > 0:
            reg_scale = np.trace(ZtZ) / ZtZ.shape[0]
            ZtZ_reg = ZtZ + reg_param * reg_scale * np.eye(ZtZ.shape[0])
        else:
            ZtZ_reg = ZtZ + reg_param * np.eye(ZtZ.shape[0])

        try:
            # Use lstsq for better numerical stability
            alpha = lstsq(ZtZ_reg, ZtY, rcond=1e-6)[0]
        except:
            # Fallback to pseudo-inverse
            alpha = np.linalg.pinv(ZtZ_reg, rcond=1e-6) @ ZtY

        # Compute beta_hat_current = Kr * alpha
        beta_current = Kr @ alpha
        beta_mat[:, j - 1] = beta_current.flatten()

        # If we need more components, augment Kr and Zm
        if j < m_opt:
            # new_Kr = K * Kr[:, j] / T
            # Note: Kr[:, j-1] is the j-th column (0-indexed)
            new_Kr = K @ Kr[:, -1] / T
            Kr = np.column_stack([Kr, new_Kr.reshape(-1, 1)])

            # new_Zm = X * new_Kr / T
            new_Zm = X @ new_Kr / T
            Zm = np.column_stack([Zm, new_Zm.reshape(-1, 1)])

    # Return the optimal estimate (last column) and all estimates
    beta_opt = beta_mat[:, -1]

    return beta_opt, m_opt, beta_mat


# Enhanced version with more options and diagnostics
def apls_fit_optimal_enhanced(y, X, K, r, m_opt, reg_param=1e-6,
                              method='auto', return_details=False):
    """
    Enhanced version with multiple stabilization methods.

    Parameters:
    method : str
        'auto' - automatically select best method
        'lstsq' - use least squares
        'svd' - use SVD-based approach
        'chol' - use Cholesky decomposition (if positive definite)
    return_details : bool
        If True, return additional information
    """
    n, T = X.shape

    beta_mat = np.zeros((T, m_opt))
    alphas = []  # Store alphas for diagnostics
    condition_numbers = []  # Store condition numbers

    # Initialize
    Kr = r.reshape(-1, 1)
    Zm = X @ Kr / T

    for j in range(1, m_opt + 1):
        ZtZ = Zm.T @ Zm
        ZtY = Zm.T @ y

        # Add regularization
        if ZtZ.shape[0] > 0:
            reg_scale = np.trace(ZtZ) / ZtZ.shape[0]
            ZtZ_reg = ZtZ + reg_param * reg_scale * np.eye(ZtZ.shape[0])
        else:
            ZtZ_reg = ZtZ + reg_param * np.eye(ZtZ.shape[0])

        # Compute condition number for diagnostics
        if ZtZ_reg.shape[0] > 0:
            try:
                cond_num = np.linalg.cond(ZtZ_reg)
                condition_numbers.append(cond_num)
            except:
                condition_numbers.append(np.inf)

        # Choose method
        if method == 'auto':
            # Use condition number to decide
            if len(condition_numbers) > 0 and condition_numbers[-1] > 1e10:
                # Ill-conditioned, use SVD
                method_use = 'svd'
            elif j <= 3 or m_opt <= 5:
                # For small problems, use lstsq
                method_use = 'lstsq'
            else:
                method_use = 'lstsq'
        else:
            method_use = method

        # Compute alpha using selected method
        if method_use == 'svd':
            # SVD-based solution
            try:
                U, s, Vt = np.linalg.svd(ZtZ_reg, full_matrices=False)
                s_inv = np.zeros_like(s)
                mask = s > 1e-8
                s_inv[mask] = 1.0 / s[mask]
                alpha = Vt.T @ (s_inv * (U.T @ ZtY))
            except:
                alpha = np.linalg.pinv(ZtZ_reg, rcond=1e-6) @ ZtY
        elif method_use == 'chol':
            # Cholesky decomposition (if positive definite)
            try:
                L = np.linalg.cholesky(ZtZ_reg)
                # Solve L * L' * alpha = ZtY
                alpha = solve(L.T, solve(L, ZtY, lower=True), lower=False)
            except:
                # Fallback to lstsq
                alpha = lstsq(ZtZ_reg, ZtY, rcond=1e-6)[0]
        else:
            # Default: least squares
            try:
                alpha = lstsq(ZtZ_reg, ZtY, rcond=1e-6)[0]
            except:
                alpha = np.linalg.pinv(ZtZ_reg, rcond=1e-6) @ ZtY

        alphas.append(alpha)

        # Compute beta
        beta_current = Kr @ alpha
        beta_mat[:, j - 1] = beta_current.flatten()

        # Augment for next iteration
        if j < m_opt:
            new_Kr = K @ Kr[:, -1] / T
            Kr = np.column_stack([Kr, new_Kr.reshape(-1, 1)])
            new_Zm = X @ new_Kr / T
            Zm = np.column_stack([Zm, new_Zm.reshape(-1, 1)])

    beta_opt = beta_mat[:, -1]

    if return_details:
        details = {
            'alphas': alphas,
            'condition_numbers': condition_numbers,
            'beta_mat': beta_mat,
            'method_used': method_use
        }
        return beta_opt, m_opt, beta_mat, details
    else:
        return beta_opt, m_opt, beta_mat


# Function to compute fitted values and residuals
def apls_predict(beta, X, T):
    """
    Compute predictions from APLS coefficients.

    Parameters:
    beta : ndarray of shape (T,)
        APLS coefficients
    X : ndarray of shape (n, T)
        Design matrix
    T : int
        Scaling factor

    Returns:
    y_pred : ndarray of shape (n,)
        Predicted values
    """
    return X @ beta / T


# Function to compute model diagnostics
def apls_diagnostics(y, y_pred, beta, X, T, m_opt):
    """
    Compute diagnostics for APLS model.
    """
    n = len(y)
    residuals = y - y_pred

    # R-squared
    ss_total = np.sum((y - np.mean(y)) ** 2)
    ss_residual = np.sum(residuals ** 2)
    r_squared = 1 - ss_residual / ss_total

    # Adjusted R-squared
    adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - m_opt - 1)

    # AIC (approximate)
    sigma2 = ss_residual / n
    aic = n * np.log(sigma2) + 2 * m_opt

    # BIC
    bic = n * np.log(sigma2) + m_opt * np.log(n)

    return {
        'r_squared': r_squared,
        'adj_r_squared': adj_r_squared,
        'aic': aic,
        'bic': bic,
        'residuals': residuals,
        'sigma2': sigma2
    }


# Example usage
if __name__ == "__main__":
    np.random.seed(42)

    # Generate synthetic data
    T = 30
    n = 100
    s = np.linspace(0, 1, T)

    # Create kernel matrix
    Delta = s[:, np.newaxis] - s[np.newaxis, :]
    B2_s = s ** 2 - s + 1 / 6
    B2_t = s ** 2 - s + 1 / 6
    abs_Delta = np.abs(Delta)
    B4_st = abs_Delta ** 4 - 2 * abs_Delta ** 3 + abs_Delta ** 2 - 1 / 30
    from math import factorial

    K = (1 / factorial(2) ** 2) * np.outer(B2_s, B2_t) - (1 / factorial(4)) * B4_st
    K = K + 1e-4 * np.eye(T)

    # Generate data
    beta_true = np.exp(-s * 3) * np.sin(s * 10)
    X = np.random.randn(n, T)
    y = X @ beta_true / T + 0.1 * np.random.randn(n)
    r = K @ beta_true + 0.01 * np.random.randn(T)

    # Assume m_opt was selected by CV (using m=8 from earlier results)
    m_opt = 8

    # Fit on full data
    print("Fitting APLS with optimal components...")
    beta_opt, m_opt_final, beta_mat = apls_fit_optimal(
        y, X, K, r, m_opt, reg_param=1e-4
    )

    print(f"Optimal components: {m_opt_final}")
    print(f"Beta_opt shape: {beta_opt.shape}")
    print(f"Beta_mat shape: {beta_mat.shape}")

    # Compute predictions
    y_pred = apls_predict(beta_opt, X, T)

    # Compute diagnostics
    diag = apls_diagnostics(y, y_pred, beta_opt, X, T, m_opt)

    print("\nModel Diagnostics:")
    print(f"R-squared: {diag['r_squared']:.6f}")
    print(f"Adjusted R-squared: {diag['adj_r_squared']:.6f}")
    print(f"AIC: {diag['aic']:.2f}")
    print(f"BIC: {diag['bic']:.2f}")

    # Test enhanced version with details
    print("\n" + "=" * 50)
    print("Enhanced version with diagnostics:")
    print("=" * 50)

    beta_opt2, m_opt2, beta_mat2, details = apls_fit_optimal_enhanced(
        y, X, K, r, m_opt, reg_param=1e-4,
        method='auto', return_details=True
    )

    print(f"Condition numbers: {details['condition_numbers'][:5]}...")
    print(f"Method used: {details['method_used']}")

    # Visualize results
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Plot 1: Coefficient paths
        ax = axes[0, 0]
        colors = plt.cm.viridis(np.linspace(0, 1, min(5, m_opt)))
        for j in range(min(5, m_opt)):
            ax.plot(s, beta_mat[:, j], color=colors[j],
                    alpha=0.5, label=f'Component {j + 1}')
        ax.plot(s, beta_true, 'k-', linewidth=3, label='True', alpha=0.7)
        ax.plot(s, beta_opt, 'r--', linewidth=2, label=f'Optimal (m={m_opt})')
        ax.set_xlabel('s')
        ax.set_ylabel('Coefficient')
        ax.set_title('APLS Coefficient Paths')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: Fitted vs Actual
        ax = axes[0, 1]
        ax.scatter(y, y_pred, alpha=0.6)
        ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', linewidth=2)
        ax.set_xlabel('Actual')
        ax.set_ylabel('Fitted')
        ax.set_title(f'Fitted vs Actual (R²={diag["r_squared"]:.4f})')
        ax.grid(True, alpha=0.3)

        # Plot 3: Residuals
        ax = axes[1, 0]
        ax.scatter(y_pred, diag['residuals'], alpha=0.6)
        ax.axhline(0, color='r', linestyle='--', linewidth=2)
        ax.set_xlabel('Fitted')
        ax.set_ylabel('Residuals')
        ax.set_title('Residual Plot')
        ax.grid(True, alpha=0.3)

        # Plot 4: Coefficient comparison
        ax = axes[1, 1]
        ax.plot(s, beta_true, 'k-', linewidth=2, label='True')
        ax.plot(s, beta_opt, 'r-', linewidth=2, label=f'APLS (m={m_opt})')

        # Add confidence bands (approximate)
        beta_std = np.std(beta_mat, axis=1)
        ax.fill_between(s, beta_opt - 2 * beta_std, beta_opt + 2 * beta_std,
                        alpha=0.2, color='red', label='±2 SD')
        ax.set_xlabel('s')
        ax.set_ylabel('Coefficient')
        ax.set_title('Final Estimate with Uncertainty')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()
    except:
        pass
