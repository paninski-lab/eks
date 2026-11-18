"""Statistical utilities: PCA fitting and Mahalanobis-distance-based variance inflation."""

import logging
import time

import numpy as np
from sklearn.decomposition import PCA, FactorAnalysis

from eks.marker_array import MarkerArray, mA_to_stacked_array

logger = logging.getLogger(__name__)


def compute_pca(
        valid_frames_mask: np.ndarray,
        emA_centered_preds: MarkerArray,
        emA_good_centered_preds: MarkerArray,
        n_components: int = 3,
        pca_object: PCA | None = None,
) -> tuple[list, list]:
    """
    Performs Principal Component Analysis (PCA) per keypoint using filtered + centered predictions.

    Args:
        valid_frames_mask (np.ndarray): Boolean mask indicating valid frames per keypoint.
            Shape: (n_frames, n_keypoints).
        emA_centered_preds (MarkerArray): Centered ensemble predictions for all frames.
            Shape: (1, n_cameras, n_frames, n_keypoints, 2).
        emA_good_centered_preds (MarkerArray): Centered predictions for variance-filtered frames.
            Shape: (1, n_cameras, n_filtered_frames, n_keypoints, 2).
        n_components (int, optional): Number of principal components to retain. Defaults to 3.
        pca_object: pre-computed PCA matrix for PCA computation

    Returns:
        tuple:
            ensemble_pca (list): List of trained PCA models, one per keypoint.
            good_pcs_list (list): List of PCA-transformed coordinates for variance-filtered frames.
    """
    n_models, n_cameras, n_frames, n_keypoints, _ = emA_centered_preds.shape
    assert n_models == 1, "MarkerArray should have n_models = 1 after ensembling."

    ensemble_pca = []
    good_pcs_list = []
    for k in range(n_keypoints):
        # Find valid frame indices for the current keypoint
        good_frame_indices = np.where(valid_frames_mask[:, k])[0]  # Shape: (n_filtered_frames,)

        emA_centered_preds_k = emA_centered_preds.slice("keypoints", k)
        emA_good_centered_preds_k = emA_good_centered_preds.slice("keypoints", k)

        # Reshape good_centered_preds_k and centered_preds_k for PCA
        reshaped_gsp_k = mA_to_stacked_array(emA_good_centered_preds_k, 0)
        reshaped_sp_k = mA_to_stacked_array(emA_centered_preds_k, 0)

        # Fit PCA per keypoint
        if pca_object is None:
            pca = PCA(n_components=n_components)
            ensemble_pca_k = pca.fit(reshaped_gsp_k)
        else:
            ensemble_pca_k = pca_object
        # Transform full dataset
        pcs = ensemble_pca_k.transform(reshaped_sp_k)
        good_pcs = pcs[good_frame_indices]

        # Store results
        ensemble_pca.append(ensemble_pca_k)
        good_pcs_list.append(good_pcs)  # Append instead of assigning

    return ensemble_pca, good_pcs_list


def _select_fa_rows(
    v: np.ndarray,
    v_quantile_threshold: float | None,
) -> np.ndarray:
    """Select the rows used to fit factor analysis.

    Rows are kept if the row's maximum ensemble variance is strictly below the
    `v_quantile_threshold` percentile of those maxima. Ensemble variance already reflects
    prediction confidence (for a single model it is the inverse of the likelihood), so this
    favors confident frames.

    Args:
        v: variance data (Nx2C array)
        v_quantile_threshold: percentile of the per-row max variance below which rows are kept,
            or None to keep all rows

    Returns:
        boolean mask of shape (N,)

    """
    valid_rows = np.ones(v.shape[0], dtype=bool)
    if v_quantile_threshold is not None:
        ev_max = v.max(axis=1)
        valid_rows = ev_max < np.percentile(ev_max, v_quantile_threshold)
    return valid_rows


def _fit_factor_analysis(
    x: np.ndarray,
    n_latent: int,
    max_iters: int = 500,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit factor analysis and return the loading matrix and mean.

    Args:
        x: observations used for fitting (nx2C array)
        n_latent: number of latent dimensions
        max_iters: maximum number of EM iterations; sklearn emits a ConvergenceWarning if the
            fit has not converged by then

    Returns:
        tuple:
            W: loading matrix (2C x n_latent)
            mu_x: mean of the observations (2C,)

    """
    fa = FactorAnalysis(n_components=n_latent, max_iter=max_iters)
    t_fa = time.perf_counter()
    fa.fit(x)
    logger.debug(
        f'[maha] factor analysis: {x.shape[0]} rows, {fa.n_iter_} iters, '
        f'{time.perf_counter() - t_fa:.2f}s'
    )
    return fa.components_.T, fa.mean_


def _posterior_latent(
    x: np.ndarray,
    v: np.ndarray,
    W: np.ndarray,
    mu_x: np.ndarray,
    epsilon: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the per-row posterior of the latents given heteroscedastic observation noise.

    Args:
        x: observations (Nx2C array)
        v: observation variances (Nx2C array)
        W: loading matrix (2C x n_latent)
        mu_x: mean (2C,)
        epsilon: added to variances to prevent singular matrices

    Returns:
        tuple:
            z_hat: posterior mean (N x n_latent)
            B: posterior covariance (N x n_latent x n_latent)

    """
    inv_v = 1.0 / (v + epsilon)  # (N, 2C)
    # precision_n = W^T diag(1 / (v_n + eps)) W, batched over frames
    precision = np.einsum('dl,nd,dm->nlm', W, inv_v, W)
    B = np.linalg.inv(precision)
    # z_hat_n = B_n W^T diag(1 / (v_n + eps)) (x_n - mu)
    z_hat = np.einsum('nlm,dm,nd->nl', B, W, inv_v * (x - mu_x))
    return z_hat, B


def _posterior_predictive_variance(
    v: np.ndarray,
    W: np.ndarray,
    B: np.ndarray,
) -> dict[int, np.ndarray]:
    """Compute the posterior predictive covariance of each view's 2D observation.

    Args:
        v: observation variances (Nx2C array)
        W: loading matrix (2C x n_latent)
        B: posterior latent covariance (N x n_latent x n_latent)

    Returns:
        dict mapping view index to covariances of shape (N, 2, 2)

    """
    num_views = v.shape[1] // 2
    W_views = W.reshape(num_views, 2, W.shape[1])  # (C, 2, n_latent)
    # latent contribution W_c B_n W_c^T for every frame and view: (N, C, 2, 2)
    latent_cov = np.einsum('cal,nlm,cbm->ncab', W_views, B, W_views)
    v_views = v.reshape(v.shape[0], num_views, 2)
    Q = {}
    for view_idx in range(num_views):
        Q_view = latent_cov[:, view_idx].copy()
        Q_view[:, 0, 0] += v_views[:, view_idx, 0]
        Q_view[:, 1, 1] += v_views[:, view_idx, 1]
        Q[view_idx] = Q_view
    return Q


def _mahalanobis_distance(
    diff: np.ndarray,
    Q: dict[int, np.ndarray],
) -> dict[int, np.ndarray]:
    """Compute the squared Mahalanobis distance of each view's residual.

    Args:
        diff: residuals x - x_hat (Nx2C array)
        Q: per-view residual covariances, each (N, 2, 2)

    Returns:
        dict mapping view index to distances of shape (N, 1)

    """
    num_views = diff.shape[1] // 2
    M = {}
    for view_idx in range(num_views):
        d = diff[:, 2 * view_idx:2 * (view_idx + 1)]  # (N, 2)
        solved = np.linalg.solve(Q[view_idx], d[:, :, None])[:, :, 0]  # Q^{-1} d
        M[view_idx] = np.sum(d * solved, axis=1, keepdims=True)
    return M


def compute_mahalanobis(
    x: np.ndarray,
    v: np.ndarray,
    n_latent: int = 3,
    v_quantile_threshold: float | None = 50.0,
    epsilon: float | None = 1e-6,
    loading_matrix: np.ndarray | None = None,
    mean: np.ndarray | None = None,
    max_iters_fa: int = 500,
) -> dict:
    """Compute Mahalanobis distance and posterior predictive variance for observations.

    This function assumes the observations x are generated with a linear latent variable model.
    Parameters for this model are learned using Factor Analysis.
    Observations with high ensemble variance (above v_quantile_threshold) are excluded from Factor
    Analysis fitting; reconstructions, posterior predictive variances, and Mahalanobis distances
    are then computed for all observations.

    Args:
        x: Observed data (Nx2C array).
        v: Variance data (Nx2C array).
        n_latent: Number of latent dimensions to extract.
        v_quantile_threshold: maximum variance (percentage) for a row to be used in FA fitting.
        epsilon: Specifies epsilon added to prevent singular matrices.
        loading_matrix: shape (2C x n_latent)
        mean: shape (2C,)
        max_iters_fa: maximum number of EM iterations for the factor analysis fit.

    Returns:
        dict: Mahalanobis distances, posterior predictive variance, and reconstructed data.

    """

    if loading_matrix is None or mean is None:
        valid_rows = _select_fa_rows(v, v_quantile_threshold)
        W, mu_x = _fit_factor_analysis(x[valid_rows], n_latent, max_iters_fa)
    else:
        W = loading_matrix
        mu_x = mean

    z_hat, B = _posterior_latent(x, v, W, mu_x, epsilon)

    # posterior predictive mean (x_hat) and residuals
    xhat = np.dot(W, z_hat.T).T + mu_x
    diff = x - xhat

    Q = _posterior_predictive_variance(v, W, B)
    M = _mahalanobis_distance(diff, Q)

    return {
        'mahalanobis': M,
        'posterior_variance': Q,
        'reconstructed': xhat,
    }
