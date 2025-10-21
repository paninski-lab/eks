import os
import cv2

import jax.numpy as jnp
import jax
from jax import jacfwd, vmap, jit
import numpy as np
import pandas as pd
from aniposelib.cameras import CameraGroup
from sklearn.decomposition import PCA
from typeguard import typechecked
from typing import Dict, Callable, Tuple, List

from eks.core import ensemble, run_kalman_smoother
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
    n_latent: int = 3,
    calibration: str | None = None
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
        calibration: path to the .toml calibration file for nonlinear projection

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
    if calibration is not None:
        camgroup = CameraGroup.load(calibration)
    else:
        camgroup = None

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
        n_latent=n_latent,
        camgroup=camgroup
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
    n_latent: int = 3,
    camgroup: CameraGroup | None = None
) -> tuple:
    """
    Multi-view EKS with optional nonlinear camera projection (EKF) when calibration TOML exists.

    Linear path (default / calibration=None):
        - PCA builds C matrices; linear Kalman smoother uses (A,Q,C,R) w/ diag R from ens vars.

    Nonlinear path (calibration TOML provided):
        - latent becomes 3D per keypoint (overrides n_latent=3).
        - observation model is h_fn using calibrated projection.

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
        verbose: True to print out details
        pca_object: pre-computed PCA matrix for PCA computation
        n_latent: number of dimensions to keep from PCA
        camgroup: loaded calibration file for nonlinear projection

    Returns:
        tuple: Dataframes with smoothed predictions, final smoothing parameters.
    """

    M, V, T, K, _ = marker_array.shape  # n_models, n_cameras, n_timesteps, n_keypoints, (n_coords)

    # Ensemble + Centering ------------------------------------------------------------------------
    # MarkerArray (1, n_cameras, n_frames, n_keypoints, 5 (x, y, var_x, var_y, likelihood))
    ensemble_marker_array = ensemble(marker_array, avg_mode=avg_mode, var_mode=var_mode)
    emA_unsm = ensemble_marker_array.slice_fields("x", "y")
    emA_vars = ensemble_marker_array.slice_fields("var_x", "var_y")
    emA_likes = ensemble_marker_array.slice_fields("likelihood")

    valid_mask, emA_centered, emA_good_centered, emA_means = center_predictions(
        ensemble_marker_array, quantile_keep_pca)

    # Optional variance inflation -----------------------------------------------------------------
    if inflate_vars:
        print('inflating')
        if inflate_vars_kwargs.get("mean", None) is not None:
            # set mean to zero since we are passing in centered predictions
            inflate_vars_kwargs["mean"] = np.zeros_like(inflate_vars_kwargs["mean"])
        emA_inflated_vars = mA_compute_maha(
            emA_centered, emA_vars, emA_likes, n_latent,
            inflate_vars_kwargs=inflate_vars_kwargs,
        )
    else:
        emA_inflated_vars = emA_vars

    using_nonlinear = camgroup is not None
    if using_nonlinear:
        if verbose: print(
            "[EKS] Nonlinear path: triangulate + geometric init + calibrated projection")

        # camera order default from camgroup if needed
        if camera_names is None:
            camera_names = [getattr(cam, "name", f"cam{i}") for i, cam in
                            enumerate(camgroup.cameras)]

        # 1) triangulate (M,K,T,3) → average over models → ys_3d (K,T,3)
        tri_models = triangulate_3d_models(marker_array, camgroup)
        ys_3d = tri_models.mean(axis=0)  # (K,T,3)
        ensemble_vars_3d = tri_models.var(axis=0)  # (K, T, 3)

        # 2) init KF params for 3D latent from geometric helper
        m0s, S0s, As, Qs, Cs = initialize_kalman_filter_geometric(ys_3d)

        # 3) make multi-view h_fn (ℝ³ → ℝ^{2V})
        h_fn_combined, h_cams = make_projection_from_camgroup(camgroup)

        # 4) 2D observations and variances
        ys_list, Rs_list = [], []
        for k in range(K):
            y_list, R_list = [], []
            for c in range(V):
                xy = emA_unsm.slice("keypoints", k).slice("cameras", c).get_array(
                    squeeze=True)  # (T,2)
                y_list.append(xy)
                var_x = emA_inflated_vars.slice("keypoints", k).slice("cameras", c).slice_fields(
                    "var_x").get_array(squeeze=True)  # (T,)
                var_y = emA_inflated_vars.slice("keypoints", k).slice("cameras", c).slice_fields(
                    "var_y").get_array(squeeze=True)  # (T,)
                R_list.append(np.stack([var_x, var_y], axis=-1))  # (T,2)
            y = np.concatenate(y_list, axis=1)  # (T, 2C)
            R = np.concatenate(R_list, axis=1)  # (T, 2C)
            ys_list.append(y)
            Rs_list.append(R)

        ys = np.stack(ys_list, axis=0)  # (K, T, 2C)
        ensemble_vars = np.stack(Rs_list, 0)  # (K, T, 2C)

    else:
        if verbose: print("[EKS] Linear path: PCA subspace + linear emissions")

        # 1) PCA + C
        (ensemble_pca, good_pcs_list) = compute_pca(
            valid_mask, emA_centered, emA_good_centered,
            n_components=n_latent, pca_object=pca_object
        )
        # 2) init linear KF params
        m0s, S0s, As, Qs, Cs = initialize_kalman_filter_pca(
            good_pcs_list=good_pcs_list, ensemble_pca=ensemble_pca, n_latent=n_latent
        )
        # 3) observations & R
        ys = np.stack([mA_to_stacked_array(emA_centered, k) for k in range(K)])
        ensemble_vars = np.stack([mA_to_stacked_array(emA_inflated_vars, k) for k in range(K)])

        h_fn_combined = None

    # Smoother ------------------------------------------------------------------------------------
    s_finals, ms, Vs = run_kalman_smoother(
        ys=jnp.asarray(ys),  # (K, T, 2C)
        m0s=m0s, S0s=S0s, As=As, Qs=Qs, Cs=Cs,
        ensemble_vars=np.swapaxes(ensemble_vars, 0, 1),  # (T,K,2C)
        s_frames=s_frames, smooth_param=smooth_param,
        verbose=verbose,
        h_fn=h_fn_combined
    )

    # Reprojection & packaging --------------------------------------------------------------------
    camera_arrs = [[] for _ in camera_names]

    if using_nonlinear:
        for k in range(K):
            ms_k = ms[k]
            Vs_k = Vs[k]

            for c, _ in enumerate(camera_names):
                xy_proj = np.array(vmap(h_cams[c])(ms_k))  # (T, 2)

                try:
                    var_x, var_y = project_3d_covariance_to_2d(ms_k, Vs_k, h_cams[c],
                                                               ensemble_vars[k])
                except AttributeError:
                    var_x = np.full(ms_k.shape[0], np.nan)
                    var_y = np.full(ms_k.shape[0], np.nan)

                data_arr = camera_arrs[c]
                data_arr.extend([
                    xy_proj[:, 0],
                    xy_proj[:, 1],
                    emA_likes.slice("keypoints", k).slice("cameras", c).get_array(squeeze=True),
                    emA_unsm.slice("keypoints", k).slice("cameras", c).slice_fields(
                        "x").get_array(squeeze=True),
                    emA_unsm.slice("keypoints", k).slice("cameras", c).slice_fields(
                        "y").get_array(squeeze=True),
                    emA_vars.slice("keypoints", k).slice("cameras", c).slice_fields(
                        "var_x").get_array(squeeze=True),
                    emA_vars.slice("keypoints", k).slice("cameras", c).slice_fields(
                        "var_y").get_array(squeeze=True),
                    var_x,
                    var_y,
                ])
    else:
        for k in range(K):
            C_k = Cs[k]
            ms_k = ms[k]
            Vs_k = Vs[k]
            y_m_smooth = np.dot(C_k, ms_k.T).T
            y_v_smooth = np.swapaxes(np.dot(C_k, np.dot(Vs_k, C_k.T)), 0, 1)
            # Final cleanup
            c_i = [[c * 2, c * 2 + 1] for c in range(V)]
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
                    emA_unsm.slice("keypoints", k).slice("cameras", c).slice_fields(
                        "x").get_array(squeeze=True),
                    emA_unsm.slice("keypoints", k).slice("cameras", c).slice_fields(
                        "y").get_array(squeeze=True),
                    emA_inflated_vars.slice("keypoints", k).slice("cameras", c).slice_fields(
                        "var_x").get_array(squeeze=True),
                    emA_inflated_vars.slice("keypoints", k).slice("cameras", c).slice_fields(
                        "var_y").get_array(squeeze=True),
                    y_v_smooth[:, x_i, x_i] + ensemble_vars[k, :, x_i],
                    y_v_smooth[:, y_i, y_i] + ensemble_vars[k, :, y_i]
                ])

    labels = ['x', 'y', 'likelihood', 'x_ens_median', 'y_ens_median',
              'x_ens_var', 'y_ens_var', 'x_posterior_var', 'y_posterior_var']
    pdindex = make_dlc_pandas_index(keypoint_names, labels=labels)

    camera_dfs = []
    for c, cam_name in enumerate(camera_names):
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


def initialize_kalman_filter_geometric(ys: np.ndarray) -> Tuple[jnp.ndarray, ...]:
    """
    Initialize Kalman filter parameters for geometric (3D) keypoints.

    Args:
        ys: Array of shape (K, T, 3) — triangulated keypoints.

    Returns:
        Tuple of Kalman filter parameters:
            - m0s: (K, 3) initial means
            - S0s: (K, 3, 3) initial covariances
            - As: (K, 3, 3) transition matrices
            - Qs: (K, 3, 3) process noise covariances (from robust lag-1 diffs)
            - Cs: (K, 3, 3) observation matrices
    """
    K, T, D = ys.shape

    # Initial state means (can also use ys[:, 0, :] if preferred)
    m0s = np.array([ys[k, :10].mean(axis=0) for k in range(K)])
    S0s = np.array([
        np.diag([
            np.nanvar(ys[k, :, d]) + 1e-4  # avoid degenerate matrices
            for d in range(D)
        ])
        for k in range(K)
    ])  # (K, 3, 3)

    # Identity transition and observation matrices
    As = np.tile(np.eye(D), (K, 1, 1))
    Cs = np.tile(np.eye(D), (K, 1, 1))

    # process noise covariances from lag-1 differences
    Qs = []
    for k in range(K):
        x = ys[k]                       # (T, 3)
        dx = np.diff(x, axis=0)         # (T-1, 3) with A = I
        # Median absolute deviation per dim
        med = np.median(dx, axis=0)
        mad = np.median(np.abs(dx - med), axis=0) + 1e-12
        sigma = 1.4826 * mad            # MAD → std under Gaussian assumption
        var = np.maximum(sigma**2, 1e-8)
        Qs.append(np.diag(var))
    Qs = np.array(Qs)                   # (K, 3, 3)

    return (
        jnp.array(m0s),
        jnp.array(S0s),
        jnp.array(As),
        jnp.array(Qs),
        jnp.array(Cs),
    )


def mA_compute_maha(centered_emA_preds, emA_vars, emA_likes, n_latent,
                    inflate_vars_kwargs={}, threshold=5, scalar=10):
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
        print(f'inflating keypoint: {k}')
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


# ================================================================
# Calibration helpers (only used when `calibration` TOML is given)
# ================================================================

def rodrigues(rvec):
    """OpenCV-style Rodrigues: rvec (3,) -> R (3,3)."""
    theta = jnp.linalg.norm(rvec)

    def small_angle(_):
        rx, ry, rz = rvec
        K = jnp.array([[0.0, -rz,  ry],
                       [rz,  0.0, -rx],
                       [-ry, rx,  0.0]])
        return jnp.eye(3) + K

    def general(_):
        rx, ry, rz = rvec / theta
        K = jnp.array([[0.0, -rz,  ry],
                       [rz,  0.0, -rx],
                       [-ry, rx,  0.0]])
        s = jnp.sin(theta)
        c = jnp.cos(theta)
        return jnp.eye(3) + s*K + (1.0 - c) * (K @ K)

    return jax.lax.cond(theta < 1e-12, small_angle, general, operand=None)

def parse_dist(dist_coeffs):
    """
    OpenCV pinhole distortion ordering:
      [k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4, tx, ty]
    """
    dc = jnp.pad(jnp.asarray(dist_coeffs, dtype=jnp.float64), (0, max(0, 14 - len(dist_coeffs))))
    k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4, tx, ty = [dc[i] for i in range(14)]
    return dict(k1=k1, k2=k2, p1=p1, p2=p2, k3=k3, k4=k4, k5=k5, k6=k6, s1=s1, s2=s2, s3=s3, s4=s4)


def make_jax_projection_fn(rvec, tvec, K, dist_coeffs):
    """
    JAX-compatible replacement for cv2.projectPoints.
    rvec: (3,), tvec: (3,), K: (3,3) with optional skew K[0,1], dist_coeffs: OpenCV order.
    Returns: project(object_points: (...,3)) -> (...,2)
    """
    rvec = jnp.asarray(rvec, dtype=jnp.float64)
    tvec = jnp.asarray(tvec, dtype=jnp.float64)
    K    = jnp.asarray(K,    dtype=jnp.float64)
    fx, fy, cx, cy, skew = K[0,0], K[1,1], K[0,2], K[1,2], K[0,1]
    d = parse_dist(dist_coeffs)
    R = rodrigues(rvec)

    @jit
    def project(object_points):
        Xw = jnp.asarray(object_points, dtype=jnp.float64)
        # world -> camera
        Xc = Xw @ R.T + tvec  # (..., 3)
        X, Y, Z = Xc[..., 0], Xc[..., 1], Xc[..., 2]

        # normalized coords
        x = X / Z
        y = Y / Z

        r2  = x * x + y * y
        r4  = r2 * r2
        r6  = r4 * r2
        r8  = r4 * r4
        r10 = r8 * r2
        r12 = r6 * r6

        radial = 1.0 + d["k1"] * r2 + d["k2"] * r4 + d["k3"] * r6 + d["k4"] * r8 + d["k5"] * r10 \
                 + d["k6"] * r12
        x_tan = 2.0*d["p1"] * x * y + d["p2"] * (r2 + 2.0 * x * x)
        y_tan = d["p1"] * (r2 + 2.0 * y * y) + 2.0 * d["p2"] * x * y
        # thin-prism
        x_tp = d["s1"] * r2 + d["s2"] * r4
        y_tp = d["s3"] * r2 + d["s4"] * r4

        xd = x * radial + x_tan + x_tp
        yd = y * radial + y_tan + y_tp

        # intrinsics (allow nonzero skew)
        u = fx * xd + skew * yd + cx
        v = fy * yd + cy
        return jnp.stack([u, v], axis=-1)  # (..., 2)

    return project


def make_projection_from_camgroup(camgroup):
    """
    Build a combined multi-view projector h_fn: (B,3) -> (B, 2*C),
    and also return per-camera heads (3,) -> (2,) for variance projection.
    """
    h_cams = []
    for cam in camgroup.cameras:
        rot = np.array(cam.get_rotation())
        rvec = cv2.Rodrigues(rot)[0].ravel() if rot.shape == (3, 3) else rot.ravel()
        tvec = np.array(cam.get_translation()).ravel()
        K = np.array(cam.get_camera_matrix())
        dist = np.array(cam.get_distortions()).ravel()  # distortion coeffs: k1,k2,p1,p2,k3,...

        h_cams.append(
            make_jax_projection_fn(
                jnp.array(rvec),
                jnp.array(tvec),
                jnp.array(K),
                jnp.array(dist)
            )
        )

    def make_combined_h_fn(h_list):
        def h_fn(x):
            return jnp.concatenate([h(x) for h in h_list], axis=0)
        return h_fn

    h_fn_combined = make_combined_h_fn(h_cams)
    return h_fn_combined, h_cams


def triangulate_3d_models(marker_array, camgroup) -> np.ndarray:
    """Triangulate per-model, per-kpt, per-frame: (M,K,T,3)."""
    M, C, T, K, _ = marker_array.shape
    raw = marker_array.get_array()  # (M,C,T,K,3)
    tri = np.zeros((M, K, T, 3), dtype=float)
    for m in range(M):
        for k in range(K):
            for t in range(T):
                xy_views = [raw[m, c, t, k, :2] for c in range(C)]
                tri[m, k, t] = camgroup.triangulate(np.array(xy_views), fast=True)
    return tri


def project_3d_covariance_to_2d(ms_k, Vs_k, h_cam, inflated_vars_k):
    """
    Project 3D covariance matrices to 2D using the Jacobian of the projection function.

    Args:
        ms_k: (T, 3) - 3D state means for keypoint k
        Vs_k: (T, 3, 3) - 3D covariance matrices for keypoint k
        h_cam: JAX projection function for this camera
        inflated_vars_k: (T, 3) - ensemble variances for keypoint k

    Returns:
        var_x: (T,) - x-direction posterior variances
        var_y: (T,) - y-direction posterior variances
    """

    # Compute Jacobian of projection function at each 3D point
    def project_single_point(x_3d):
        return h_cam(x_3d)

    # Compute Jacobian for each time point
    jacobians = []
    for t in range(ms_k.shape[0]):
        jac = jax.jacfwd(project_single_point)(ms_k[t])
        jacobians.append(jac)

    jacobians = np.array(jacobians)  # (T, 2, 3)

    # Project 3D covariance to 2D: Cov_2D = J * Cov_3D * J^T
    cov2d_proj = np.zeros((ms_k.shape[0], 2, 2))
    for t in range(ms_k.shape[0]):
        J = jacobians[t]  # (2, 3)
        V_3d = Vs_k[t]  # (3, 3)
        cov2d_proj[t] = J @ V_3d @ J.T  # (2, 2)

    # Extract x and y variances and add ensemble variance
    var_x = cov2d_proj[:, 0, 0] + inflated_vars_k[:, 0]
    var_y = cov2d_proj[:, 1, 1] + inflated_vars_k[:, 1]

    return var_x, var_y