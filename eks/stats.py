import numpy as np
from sklearn.decomposition import PCA, FactorAnalysis
from typeguard import typechecked

from eks.marker_array import MarkerArray, mA_to_stacked_array


def compute_pca(
        valid_frames_mask,
        emA_centered_preds: MarkerArray,
        emA_good_centered_preds: MarkerArray,
        n_components: int = 3,
        pca_object: PCA | None = None
):
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


@typechecked
def compute_mahalanobis(
    x: np.ndarray,
    v: np.ndarray,
    n_latent: int = 3,
    v_quantile_threshold: float | None = 50.0,
    likelihoods: np.ndarray | None = None,
    likelihood_threshold: float | None = 0.9,
    epsilon: float | None = 1e-6,
    loading_matrix: np.ndarray | None = None,
    mean: np.ndarray | None = None,
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
        loading_matrix: shape (2C x n_latent)
        mean: shape (2C,)

    Returns:
        dict: Mahalanobis distances, posterior predictive variance, and reconstructed data.

    """

    if loading_matrix is None or mean is None:
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
    else:
        W = loading_matrix
        mu_x = mean

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
