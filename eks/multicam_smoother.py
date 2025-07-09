import os

import jax.numpy as jnp
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from typeguard import typechecked

from eks.core import ensemble, optimize_smooth_param
from eks.marker_array import (
    MarkerArray,
    input_dfs_to_markerArray,
    mA_to_stacked_array,
    stacked_array_to_mA,
)
from eks.stats import compute_mahalanobis, compute_pca
from eks.utils import center_predictions, format_data, make_dlc_pandas_index


@typechecked
def fit_eks_mirrored_multicam(
    input_source: str | list,
    save_file: str,
    bodypart_list: list | None = None,
    smooth_param: float | list | None = None,
    s_frames: list | None = None,
    camera_names: list | None = None,
    quantile_keep_pca: float = 95.0,
    avg_mode: str = 'median',
    var_mode: str = 'confidence_weighted_var',
    verbose: bool = False,
    inflate_vars: bool = False,
    n_latent: int = 3
) -> tuple:
    """
    Fit the Ensemble Kalman Smoother for mirrored multi-camera data.

    Args:
        input_source: Directory path or list of CSV file paths with columns for all cameras.
        save_file: Directory to save output DataFrame.
        bodypart_list: List of body parts.
        smooth_param: Value in (0, Inf); smaller values lead to more smoothing.
        s_frames: Frames for automatic optimization if smooth_param is not provided.
        camera_names: List of camera names corresponding to the input data.
        quantile_keep_pca: Percentage of points kept for PCA (default: 95).
        avg_mode: Mode for averaging across ensemble ('median', 'mean').
        var_mode: mode for computing ensemble variance
            'var' | 'confidence_weighted_var'
        verbose: True to print out details
        inflate_vars: True to use Mahalanobis distance thresholding to inflate ensemble variance
        n_latent: number of dimensions to keep from PCA

    Returns:
            tuple:
                    camera_dfs (list): List of Output Dataframes
                    s_finals (list): List of optimized smoothing factors for each keypoint.
                    input_dfs (list): List of input DataFrames for plotting.
                    bodypart_list (list): List of body parts used.
    """
    # Load and format input files
    input_dfs_list, keypoint_names = format_data(input_source)
    if bodypart_list is None:
        seen = set()
        bodypart_list = []
        for name in keypoint_names:
            base = name.split("_")[0]
            if base not in seen:
                seen.add(base)
                bodypart_list.append(base)

    n_models = len(input_dfs_list)
    n_cameras = len(camera_names)

    # Initialize a nested list with shape (n_cameras, n_models)
    camera_model_dfs = [[None] * n_models for _ in range(n_cameras)]

    for model_idx, df in enumerate(input_dfs_list):
        for cam_idx, cam_name in enumerate(camera_names):
            # Extract columns belonging to this camera
            camera_columns = {
                col: col.replace(f"_{cam_name}", "")
                for col in df.columns if f"_{cam_name}_" in col
            }
            # Create DataFrame for this camera and rename columns
            camera_df = df[list(camera_columns.keys())].rename(columns=camera_columns)
            # Store in the structured list
            camera_model_dfs[cam_idx][model_idx] = camera_df
    marker_array = input_dfs_to_markerArray(camera_model_dfs, bodypart_list, camera_names)

    # Run the ensemble Kalman smoother for multi-camera data
    camera_dfs, smooth_params_final = ensemble_kalman_smoother_multicam(
        marker_array=marker_array,
        keypoint_names=bodypart_list,
        smooth_param=smooth_param,
        quantile_keep_pca=quantile_keep_pca,
        camera_names=camera_names,
        s_frames=s_frames,
        avg_mode=avg_mode,
        var_mode=var_mode,
        verbose=verbose,
        inflate_vars=inflate_vars,
        n_latent=n_latent
    )
    final_df = None
    for c, camera_df in enumerate(camera_dfs):
        suffix = f'{camera_names[c]}'
        new_columns = [(scorer, f'{kp}_{suffix}', attr) for scorer, kp, attr in camera_df.columns]
        camera_df.columns = pd.MultiIndex.from_tuples(new_columns, names=camera_df.columns.names)
        if final_df is None:
            final_df = camera_df
        else:
            final_df = pd.concat([final_df, camera_df], axis=1)
    # Save the output DataFrames to CSV file
    os.makedirs(os.path.dirname(save_file), exist_ok=True)
    final_df.to_csv(f"{save_file}")
    return final_df, smooth_params_final, input_dfs_list, bodypart_list


@typechecked
def fit_eks_multicam(
    input_source: str | list,
    save_dir: str,
    bodypart_list: list | None = None,
    smooth_param: float | list | None = None,
    s_frames: list | None = None,
    camera_names: list | None = None,
    quantile_keep_pca: float = 95.0,
    avg_mode: str = 'median',
    var_mode: str = 'confidence_weighted_var',
    inflate_vars: bool = False,
    verbose: bool = False,
    n_latent: int = 3
) -> tuple:
    """
    Fit the Ensemble Kalman Smoother for un-mirrored multi-camera data.

    Args:
        input_source: Directory path or list of CSV file paths with columns for all cameras.
        save_dir: Directory to save output DataFrame.
        bodypart_list: List of body parts.
        smooth_param: Value in (0, Inf); smaller values lead to more smoothing.
        s_frames: Frames for automatic optimization if smooth_param is not provided.
        camera_names: List of camera names corresponding to the input data.
        quantile_keep_pca: Percentage of points kept for PCA (default: 95).
        avg_mode: Mode for averaging across ensemble ('median', 'mean').
        var_mode: mode for computing ensemble variance
            'var' | 'confidence_weighted_var'
        inflate_vars: True to use Mahalanobis distance thresholding to inflate ensemble variance
        verbose: True to print out details
        n_latent: number of dimensions to keep from PCA

    Returns:
        tuple:
            camera_dfs (list): List of Output Dataframes
            s_finals (list): List of optimized smoothing factors for each keypoint.
            input_dfs (list): List of input DataFrames for plotting.
            bodypart_list (list): List of body parts used.

    """
    # Load and format input files
    # NOTE: input_dfs_list is a list of camera-specific lists of Dataframes
    input_dfs_list, keypoint_names = format_data(input_source, camera_names=camera_names)
    if bodypart_list is None:
        bodypart_list = keypoint_names

    marker_array = input_dfs_to_markerArray(input_dfs_list, bodypart_list, camera_names)

    # Run the ensemble Kalman smoother for multi-camera data
    camera_dfs, smooth_params_final = ensemble_kalman_smoother_multicam(
        marker_array=marker_array,
        keypoint_names=bodypart_list,
        smooth_param=smooth_param,
        quantile_keep_pca=quantile_keep_pca,
        camera_names=camera_names,
        s_frames=s_frames,
        avg_mode=avg_mode,
        var_mode=var_mode,
        verbose=verbose,
        inflate_vars=inflate_vars,
        n_latent=n_latent
    )
    # Save output DataFrames to CSVs (one per camera view)
    os.makedirs(save_dir, exist_ok=True)
    for c, camera in enumerate(camera_names):
        save_filename = f'multicam_{camera}_results.csv'
        camera_dfs[c].to_csv(os.path.join(save_dir, save_filename))
    return camera_dfs, smooth_params_final, input_dfs_list, bodypart_list


@typechecked
def ensemble_kalman_smoother_multicam(
    marker_array: MarkerArray,
    keypoint_names: list,
    smooth_param: float | list | None = None,
    quantile_keep_pca: float = 95.0,
    camera_names: list | None = None,
    s_frames: list | None = None,
    avg_mode: str = 'median',
    var_mode: str = 'confidence_weighted_var',
    inflate_vars: bool = False,
    inflate_vars_kwargs: dict = {},
    verbose: bool = False,
    pca_object: PCA | None = None,
    n_latent: int = 3
) -> tuple:
    """
    Use multi-view constraints to fit a 3D latent subspace for each body part.

    Args:
        marker_array: MarkerArray object containing marker data.
            Shape (n_models, n_cameras, n_frames, n_keypoints, 3 (for x, y, likelihood))
        keypoint_names: List of body parts to run smoothing on
        smooth_param: Value in (0, Inf); smaller values lead to more smoothing (default: None).
        quantile_keep_pca: Percentage of points kept for PCA (default: 95).
        camera_names: List of camera names corresponding to the input data (default: None).
        s_frames: Frames for auto-optimization if smooth_param is not provided (default: None).
        avg_mode: mode for averaging across ensemble
            'median' | 'mean'
        var_mode: mode for computing ensemble variance
            'var' | 'confidence_weighted_var'
        inflate_vars: True to use Mahalanobis distance thresholding to inflate ensemble variance
        inflate_vars_kwargs: kwargs for compute_mahalanobis function for variance inflation
        pca_object: pre-computed PCA matrix for PCA computation
        n_latent: number of dimensions to keep from PCA

        verbose: True to print out details

    Returns:
        tuple: Dataframes with smoothed predictions, final smoothing parameters.
    """

    n_models, n_cameras, n_frames, n_keypoints, _ = marker_array.shape

    # MarkerArray (1, n_cameras, n_frames, n_keypoints, 5 (x, y, var_x, var_y, likelihood))
    ensemble_marker_array = ensemble(marker_array, avg_mode=avg_mode, var_mode=var_mode)
    emA_unsmoothed_preds = ensemble_marker_array.slice_fields("x", "y")
    emA_vars = ensemble_marker_array.slice_fields("var_x", "var_y")
    emA_likes = ensemble_marker_array.slice_fields("likelihood")

    (
        valid_frames_mask,
        emA_centered_preds,
        emA_good_centered_preds,
        emA_means
    ) = center_predictions(ensemble_marker_array, quantile_keep_pca)

    (
        ensemble_pca,
        good_pcs_list,  # List-by-keypoint of (n_good_frames, n_pca_components)
    ) = compute_pca(
        valid_frames_mask,
        emA_centered_preds,
        emA_good_centered_preds,
        n_components=n_latent,
        pca_object=pca_object,
    )

    if inflate_vars:
        if inflate_vars_kwargs.get("mean", None) is not None:
            # set mean to zero since we are passing in centered predictions
            inflate_vars_kwargs["mean"] = np.zeros_like(inflate_vars_kwargs["mean"])
        emA_inflated_vars = mA_compute_maha(
            emA_centered_preds, emA_vars, emA_likes, n_latent,
            inflate_vars_kwargs=inflate_vars_kwargs,
        )
    else:
        emA_inflated_vars = emA_vars

    # Kalman Filter Section ------------------------------------------

    # Initialize Kalman filter parameters
    m0s, S0s, As, cov_mats, Cs = initialize_kalman_filter_pca(
        good_pcs_list=good_pcs_list,
        ensemble_pca=ensemble_pca,
        n_latent=n_latent,
    )

    # Collect observations and variances
    ys = np.stack([
        mA_to_stacked_array(emA_centered_preds, k)
        for k in range(n_keypoints)
    ])
    ensemble_vars = np.stack([
        mA_to_stacked_array(emA_inflated_vars, k)
        for k in range(n_keypoints)
    ])

    # Optimize smoothing
    s_finals, ms, Vs = optimize_smooth_param(
        cov_mats=cov_mats,
        ys=ys,
        m0s=m0s,
        S0s=S0s,
        Cs=Cs,
        As=As,
        ensemble_vars=np.swapaxes(ensemble_vars, 0, 1),
        s_frames=s_frames,
        smooth_param=smooth_param,
        verbose=verbose
    )
    # Reproject from latent space back to observed space
    camera_arrs = [[] for _ in camera_names]
    for k, keypoint in enumerate(keypoint_names):
        C = Cs[k]
        ms_k = ms[k]
        Vs_k = Vs[k]
        inflated_vars_k = ensemble_vars[k]
        y_m_smooth = np.dot(C, ms_k.T).T
        y_v_smooth = np.swapaxes(np.dot(C, np.dot(Vs_k, C.T)), 0, 1)
        # Final cleanup
        c_i = [[camera * 2, camera * 2 + 1] for camera in range(n_cameras)]
        for c, camera in enumerate(camera_names):
            data_arr = camera_arrs[c]
            x_i, y_i = c_i[c]
            data_arr.extend([
                y_m_smooth.T[x_i] + [
                    emA_means.slice("keypoints", k).slice("cameras", c).slice_fields(
                        "x").get_array(squeeze=True)],
                y_m_smooth.T[y_i] + [
                    emA_means.slice("keypoints", k).slice("cameras", c).slice_fields(
                        "y").get_array(squeeze=True)],
                emA_likes.slice("keypoints", k).slice("cameras", c).get_array(squeeze=True),
                emA_unsmoothed_preds.slice("keypoints", k).slice("cameras", c).slice_fields(
                    "x").get_array(squeeze=True),
                emA_unsmoothed_preds.slice("keypoints", k).slice("cameras", c).slice_fields(
                    "y").get_array(squeeze=True),
                emA_inflated_vars.slice("keypoints", k).slice("cameras", c).slice_fields(
                    "var_x").get_array(squeeze=True),
                emA_inflated_vars.slice("keypoints", k).slice("cameras", c).slice_fields(
                    "var_y").get_array(squeeze=True),
                y_v_smooth[:, x_i, x_i] + inflated_vars_k[:, x_i],
                y_v_smooth[:, y_i, y_i] + inflated_vars_k[:, y_i],
            ])

    labels = [
        'x', 'y', 'likelihood', 'x_ens_median', 'y_ens_median',
        'x_ens_var', 'y_ens_var', 'x_posterior_var', 'y_posterior_var'
    ]

    pdindex = make_dlc_pandas_index(keypoint_names, labels=labels)
    camera_dfs = []

    for c, camera in enumerate(camera_names):
        camera_arr = np.asarray(camera_arrs[c])
        camera_df = pd.DataFrame(camera_arr.T, columns=pdindex)
        camera_dfs.append(camera_df)
    return camera_dfs, s_finals


def initialize_kalman_filter_pca(
    good_pcs_list: list[np.ndarray],
    ensemble_pca: list[PCA],
    n_latent: int,
) -> tuple:
    """
    Initialize Kalman filter parameters for PCA-projected keypoints.

    Parameters:
        good_pcs_list: List of (n_good_frames, n_latent) arrays per keypoint.
        ensemble_pca: List of PCA objects (one per keypoint).
        n_latent: Number of latent dimensions (usually <= 3).

    Returns:
        tuple: (m0s, S0s, As, cov_mats, Cs, Rs) as arrays stacked over keypoints.
    """

    n_keypoints = len(good_pcs_list)

    m0s = np.zeros((n_keypoints, n_latent))
    S0s = np.array([
        np.diag([np.var(good_pcs_list[k][:, i]) for i in range(n_latent)])
        for k in range(n_keypoints)
    ])
    As = np.tile(np.eye(n_latent), (n_keypoints, 1, 1))
    Cs = np.stack([pca.components_.T for pca in ensemble_pca])
    Rs = np.tile(np.eye(n_latent), (n_keypoints, 1, 1))

    cov_mats = []
    for k in range(n_keypoints):
        pcs = good_pcs_list[k]
        d_t = pcs[1:] - pcs[:-1]
        cov = np.cov(d_t.T)
        cov_norm = cov / np.max(np.abs(cov)) if np.max(np.abs(cov)) > 0 else cov
        cov_mats.append(cov_norm)

    cov_mats = np.stack(cov_mats)

    return (
        jnp.array(m0s),
        jnp.array(S0s),
        jnp.array(As),
        jnp.array(cov_mats),
        jnp.array(Cs),
    )


def mA_compute_maha(centered_emA_preds, emA_vars, emA_likes, n_latent,
                    inflate_vars_kwargs={}, threshold=5, scalar=2):
    """
    Reshape marker arrays for Mahalanobis computation, compute Mahalanobis distances,
    and optionally inflate variances for all keypoints.

    Args:
        centered_emA_preds (numpy.ndarray): centered predicted marker positions.
        emA_vars (numpy.ndarray): Variance values associated with predictions.
        emA_likes (numpy.ndarray): Likelihood values associated with predictions.
        n_latent (int): Number of dimensions to extract from dimensional reduction.
        threshold (float, optional): Mahalanobis distance threshold for inflation.
        scalar (float, optional): Factor by which to inflate variance.

    Returns:
        list: A list of tuples, each being (maha_results, inflated_ens_vars) for a keypoint.
    """
    _, n_cameras, _, n_keypoints, _ = centered_emA_preds.shape

    emA_inflated_vars_list = []
    for k in range(n_keypoints):
        # Transform mA into array of Shape: (n_models, n_frames, n_cameras * n_fields)
        preds = mA_to_stacked_array(centered_emA_preds, k)
        vars = mA_to_stacked_array(emA_vars, k)
        likes = mA_to_stacked_array(emA_likes, k)

        # set some maha defaults
        if 'likelihood_threshold' not in inflate_vars_kwargs:
            inflate_vars_kwargs['likelihood_threshold'] = 0.9
        if 'v_quantile_threshold' not in inflate_vars_kwargs:
            inflate_vars_kwargs['v_quantile_threshold'] = 50.0
        inflated = True
        tmp_vars = vars

        while inflated:
            # Compute Mahalanobis distances
            if inflate_vars_kwargs.get("likelihoods", None) is None:
                maha_results = compute_mahalanobis(preds, tmp_vars,
                                                   n_latent=n_latent,
                                                   **inflate_vars_kwargs)
            else:
                maha_results = compute_mahalanobis(preds, tmp_vars,
                                                   n_latent=n_latent,
                                                   likelihoods=likes,
                                                   **inflate_vars_kwargs)
            # Inflate variances based on Mahalanobis distances
            inflated_ens_vars_k, inflated = inflate_variance(
                tmp_vars, maha_results['mahalanobis'], threshold, scalar)
            tmp_vars = inflated_ens_vars_k

        # Reshape array back into mA
        emA_inflated_vars_k = stacked_array_to_mA(
            inflated_ens_vars_k, n_cameras, data_fields=["var_x", "var_y"])

        # Store in list
        emA_inflated_vars_list.append(emA_inflated_vars_k)

    # Stack individual keypoint emAs together
    emA_inflated_vars = MarkerArray.stack(emA_inflated_vars_list, "keypoints")

    return emA_inflated_vars


@typechecked
def inflate_variance(
    v: np.ndarray,
    maha_dict: dict,
    threshold: float = 5.0,
    scalar: float = 2.0
) -> tuple:
    """Inflate ensemble variances for Mahalanobis distances exceeding a threshold.

    Args:
        v: Variance data (Nx2C array).
        maha_dict: Dictionary containing Mahalanobis distances for each view.
        threshold: Threshold above which to inflate variances (default: 5.0).
        scalar: Scalar to multiply variances by (default: 2.0).

    Returns:
        np.ndarray: Updated variance array with inflated values.
        bool: Whether any variances were updated.
    """
    assert len(maha_dict) >= 2, 'must have >=2 views to inflate variance'
    updated_v = v.copy()
    N, D = v.shape
    C = len(maha_dict)  # Number of views

    # Create an inflation mask for all views
    inflation_mask = np.zeros((N, C), dtype=bool)

    for view_idx, distances in maha_dict.items():
        inflation_mask[:, view_idx] = distances[:, 0] > threshold

    # Duplicate columns of inflation_mask to match the shape of v
    inflation_mask_full = np.repeat(inflation_mask, 2, axis=1)

    if C == 2:
        # For 2 views, ensure a full row is set to True if any individual entry is True
        inflation_mask_full |= inflation_mask_full.any(axis=1, keepdims=True)

    # Apply inflation where the mask is True
    updated_v[inflation_mask_full] *= scalar
    inflated = inflation_mask_full.any()

    return updated_v, inflated
