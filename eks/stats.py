import numpy as np
from sklearn.decomposition import FactorAnalysis, PCA
from typeguard import typechecked
from eks.marker_array import MarkerArray
from typing import Optional


def compute_pca(
        ema_preds: MarkerArray,
        ema_vars: MarkerArray,
        quantile_keep_pca: float,
        n_components: int = 3
):
    """
    Perform PCA for each keypoint while filtering frames with high variance.

    Args:
        ema_preds: Ensemble MarkerArray containing predicted keypoint positions.
            Shape: (1, n_cameras, n_frames, n_keypoints, 2)
        ema_vars: Ensemble MarkerArray containing variance data.
            Shape: (1, n_cameras, n_frames, n_keypoints, 2)
        quantile_keep_pca: Threshold percentage for filtering low-variance frames.
        n_components: Number of principal components to keep.

    Returns:
        tuple:
            ensemble_pca (list): List of PCA models (1 per keypoint).
            ensemble_ex_var (np.ndarray): Explained variance ratios for each keypoint.
            good_ema_pcs (MarkerArray): PCA-transformed coordinates for good frames.
            ema_pcs (MarkerArray): PCA-transformed coordinates for all frames.
            means_camera (np.ndarray): Mean x and y coords for each camera (n_cameras, 2)
            scaled_ema (MarkerArray): Centered ensemble predictions.
    """

    n_models, n_cameras, n_frames, n_keypoints, _ = ema_preds.array.shape
    assert n_models == 1, "MarkerArray should have n_models = 1 after ensembling."

    preds_array = np.array(ema_preds.array[0])  # (n_cameras, n_frames, n_keypoints, 2)
    vars_array = np.array(ema_vars.array[0])

    # Initialize storage lists
    ensemble_pca = []
    ensemble_ex_var = []
    good_pcs_list = []
    pcs_list = []
    scaled_preds_array = np.zeros(
        (n_models, n_cameras, n_frames, n_keypoints, 2))

    for k in range(n_keypoints):
        # Filter by low ensemble variances
        hstacked_vars = np.hstack(vars_array[:, :, k])  # Shape: (n_frames, n_cameras * 2)
        max_vars = np.max(hstacked_vars, axis=1)  # Shape: (n_frames, n_cameras * 2)
        good_frames = np.where(max_vars <= np.percentile(max_vars, quantile_keep_pca))[0]
        good_preds = preds_array[:, good_frames, k]
        means_camera = np.mean(good_preds, axis=1)
        good_scaled_preds = good_preds - means_camera[:, None, :]

        # Fit PCA per keypoint
        pca = PCA(n_components=n_components)
        ensemble_pca_curr = pca.fit(good_scaled_preds.transpose(1, 0, 2).reshape(
            good_scaled_preds.shape[1], good_scaled_preds.shape[0] * good_scaled_preds.shape[2]))
        ensemble_ex_var_curr = pca.explained_variance_ratio_

        # Transform full dataset
        scaled_preds = preds_array[:, :, k] - means_camera[:, None, :]
        pcs = pca.transform(scaled_preds.transpose(1, 0, 2).reshape(
            scaled_preds.shape[1], scaled_preds.shape[0] * scaled_preds.shape[2]))

        good_pcs = pcs[good_frames]

        # Store results
        ensemble_pca.append(ensemble_pca_curr)
        ensemble_ex_var.append(ensemble_ex_var_curr)
        good_pcs_list.append(good_pcs)  # Append instead of assigning
        pcs_list.append(pcs)  # Append instead of assigning
        scaled_preds_array[..., k, :] = scaled_preds  # This stays as an array

    scaled_ema = MarkerArray(scaled_preds_array, data_fields=['x', 'y'])

    return (
        ensemble_pca,
        ensemble_ex_var,
        good_pcs_list,
        pcs_list,
        means_camera,
        scaled_ema
    )


@typechecked
def compute_mahalanobis(
    x: np.ndarray,
    v: np.ndarray,
    n_latent: int = 3,
    v_quantile_threshold: float | None = 50.0,
    likelihoods: np.ndarray | None = None,
    likelihood_threshold: float | None = 0.9,
    epsilon: float | None = 1e-6
) -> dict:
    """Compute Mahalanobis distance and posterior predictive variance for observations.

    This function assumes the observations x are generated with a linear latent variable model.
    Parameters for this model are learned using Factor Analysis.
    Observations with high ensemble variance (above v_quantile_threshold) or low likelihoods (below
    likelihood_threshold) are excluded from Factor Analysis fitting; reconstructions, posterior
    predictive variances, and Mahalanobis distances are then computed for all observations.

    Args:
        x: Observed data (Nx2C array).
        v: Variance data (Nx2C array).
        n_latent: Number of latent dimensions to extract.
        v_quantile_threshold: maximum variance (percentage) for a row to be used in FA fitting.
        likelihoods: Likelihoods for each row and view (NxC array).
        likelihood_threshold: Minimum likelihood for a row to be used in FA fitting.
        epsilon: Specifies epsilon added to prevent singular matrices.

    Returns:
        dict: Mahalanobis distances, posterior predictive variance, and reconstructed data.

    """

    # Filter rows based on likelihood threshold if likelihoods are provided
    if likelihoods is not None and likelihood_threshold is not None:
        valid_rows = np.min(likelihoods, axis=1) >= likelihood_threshold
    else:
        valid_rows = np.ones(x.shape[0], dtype=bool)

    # Filter rows based on ensemble variance (we want low variance predictions for FA)
    if v_quantile_threshold is not None:
        ev_max = v.max(axis=1)
        valid_rows_ev = ev_max < np.percentile(ev_max, v_quantile_threshold)
        valid_rows = valid_rows & valid_rows_ev

    # Perform Factor Analysis to estimate W and mu_x using valid rows
    fa = FactorAnalysis(n_components=n_latent)
    fa.fit(x[valid_rows])
    W = fa.components_.T  # W is (2C x n_latent)
    mu_x = fa.mean_       # Mean of the observations (2C array)

    # Posterior variance (B)
    B = np.zeros((v.shape[0], W.shape[1], W.shape[1]))
    for i in range(v.shape[0]):
        B[i] = np.linalg.inv(W.T @ np.diag(1.0 / (v[i] + epsilon)) @ W)

    # Posterior mean (z_hat)
    z_hat = np.zeros((v.shape[0], W.shape[1]))
    for i in range(v.shape[0]):
        z_hat[i] = B[i] @ W.T @ np.diag(1.0 / (v[i] + epsilon)) @ (x[i] - mu_x)

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

