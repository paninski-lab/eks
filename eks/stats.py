import numpy as np
from sklearn.decomposition import FactorAnalysis


def compute_mahalanobis(x, v, n_latent=3, likelihoods=None, likelihood_threshold=0.9):
    """
    Computes Mahalanobis distances and posterior predictive variance, including
    Factor Analysis to determine W and mu_x. Rows with low likelihoods are excluded
    from Factor Analysis fitting.

    Args:
        x: Observed data (Nx2C array).
        v: Variance data (Nx2C array).
        n_latent: Number of latent dimensions to extract (default: 3).
        likelihoods: Likelihoods for each row and view (NxC array, optional).
        likelihood_threshold: Minimum likelihood for a row to be used in FA fitting (default: 0.9).

    Returns:
        dict: Mahalanobis distances, posterior predictive variance, and reconstructed data.
    """
    # Filter rows based on likelihood threshold if likelihoods are provided
    if likelihoods is not None:
        valid_rows = np.min(likelihoods, axis=1) >= likelihood_threshold
    else:
        valid_rows = np.ones(x.shape[0], dtype=bool)

    # Perform Factor Analysis to estimate W and mu_x using valid rows
    fa = FactorAnalysis(n_components=n_latent)
    fa.fit(x[valid_rows])
    W = fa.components_.T  # W is (2C x n_latent)
    mu_x = fa.mean_       # Mean of the observations (2C array)

    # Posterior variance (B)
    B = np.zeros((v.shape[0], W.shape[1], W.shape[1]))
    for i in range(v.shape[0]):
        B[i] = np.linalg.inv(W.T @ np.diag(1.0 / v[i]) @ W)

    # Posterior mean (z_hat)
    z_hat = np.zeros((v.shape[0], W.shape[1]))
    for i in range(v.shape[0]):
        z_hat[i] = B[i] @ W.T @ np.diag(1.0 / v[i]) @ (x[i] - mu_x)

    # Posterior predictive mean (x_hat)
    xhat = np.dot(W, z_hat.T).T + mu_x

    # Compute residuals (diff) once
    diff = x - xhat

    # Posterior predictive variance per view (Q)
    num_views = x.shape[1] // 2
    Q = {view_idx: np.zeros((v.shape[0], 2, 2)) for view_idx in range(num_views)}
    for i in range(v.shape[0]):
        for view_idx in range(num_views):
            idxs = slice(2 * view_idx, 2 * (view_idx + 1))
            Q[view_idx][i] = np.diag(v[i, idxs]) + W[idxs] @ B[i] @ W[idxs].T

    # Mahalanobis distance (M)
    M = {view_idx: np.zeros((v.shape[0], 1)) for view_idx in range(num_views)}
    for i in range(v.shape[0]):
        for view_idx in range(num_views):
            idxs = slice(2 * view_idx, 2 * (view_idx + 1))
            M[view_idx][i] = diff[i, idxs].T @ np.linalg.inv(Q[view_idx][i]) @ diff[i, idxs]

    return {
        'mahalanobis': M,
        'posterior_variance': Q,
        'reconstructed': xhat
    }
