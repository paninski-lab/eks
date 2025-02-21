import numpy as np
from sklearn.decomposition import FactorAnalysis, PCA
from typeguard import typechecked
from eks.marker_array import MarkerArray, mA_to_stacked_array, stacked_array_to_mA
from typing import Optional


def compute_pca(
        emA_preds: MarkerArray,
        emA_vars: MarkerArray,
        quantile_keep_pca: float,
        n_components: int = 3
):
    """
    Perform PCA for each keypoint while filtering frames with high variance.

    Args:
        emA_preds: Ensemble MarkerArray containing predicted keypoint positions.
            Shape: (1, n_cameras, n_frames, n_keypoints, 2)
        emA_vars: Ensemble MarkerArray containing variance data.
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

    n_models, n_cameras, n_frames, n_keypoints, _ = emA_preds.shape()
    assert n_models == 1, "MarkerArray should have n_models = 1 after ensembling."

    # Maximum variance for each keypoint in each frame, independent of camera
    max_vars_per_frame = np.max(emA_vars.array, axis=(0, 1, 4))  # Shape: (n_frames, n_keypoints)
    # Compute variance threshold for each keypoint
    thresholds = np.percentile(max_vars_per_frame, quantile_keep_pca, axis=0)

    valid_frames_mask = max_vars_per_frame <= thresholds
    good_preds = emA_preds.array[:, :, valid_frames_mask, :]

    # Compute valid frame mask per (frame, keypoint)
    valid_frames_mask = max_vars_per_frame <= thresholds  # Shape: (n_frames, n_keypoints)

    ensemble_pca = []
    ensemble_ex_var = []
    good_pcs_list = []
    pcs_list = []
    emA_scaled_preds_list = []
    emA_means_list = []
    for k in range(n_keypoints):
        # Find valid frame indices for the current keypoint
        good_frame_indices = np.where(valid_frames_mask[:, k])[0]  # Shape: (n_filtered_frames,)

        # Extract valid frames for this keypoint
        # Shape: (n_models, n_cameras, n_filtered_frames, n_fields)
        good_preds_k = emA_preds.array[:, :, good_frame_indices, k, :]
        # Shape: (n_models, n_cameras, n_filtered_frames, 1, n_fields)
        good_preds_k = np.expand_dims(good_preds_k, axis=3)

        # Scale predictions by subtracting means (over frames) from predictions
        means_k = np.mean(good_preds_k, axis=2)[:, :, None, :, :]
        scaled_preds_k = emA_preds.slice("keypoints", k).array - means_k
        good_scaled_preds_k = good_preds_k - means_k

        # Reshape good_scaled_preds_k and scaled_preds_k for PCA
        reshaped_gsp_k = mA_to_stacked_array(MarkerArray(good_scaled_preds_k, data_fields=["x", "y"]), 0)
        reshaped_sp_k = mA_to_stacked_array(MarkerArray(scaled_preds_k, data_fields=["x", "y"]), 0)

        # Fit PCA per keypoint
        pca = PCA(n_components=n_components)
        ensemble_pca_k = pca.fit(reshaped_gsp_k)
        ensemble_ex_var_k = pca.explained_variance_ratio_

        # Transform full dataset
        pcs = ensemble_pca_k.transform(reshaped_sp_k)

        good_pcs = pcs[good_frame_indices]
        # Store results
        ensemble_pca.append(ensemble_pca_k)
        ensemble_ex_var.append(ensemble_ex_var_k)
        good_pcs_list.append(good_pcs)  # Append instead of assigning
        pcs_list.append(pcs)  # Append instead of assigning
        emA_scaled_preds_list.append(MarkerArray(scaled_preds_k, data_fields=["x", "y"]))
        emA_means_list.append(MarkerArray(means_k, data_fields=["x", "y"]))

    # Concatenate all keypoint-wise filtered results along the keypoints axis
    emA_scaled_preds = MarkerArray.stack(emA_scaled_preds_list, "keypoints")
    emA_means = MarkerArray.stack(emA_means_list, "keypoints")

    return (
        ensemble_pca,
        ensemble_ex_var,
        good_pcs_list,
        pcs_list,
        emA_scaled_preds,
        emA_means
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

