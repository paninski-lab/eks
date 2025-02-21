import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typeguard import typechecked

from eks.core import backward_pass, compute_initial_guesses, compute_nll, forward_pass, \
    jax_ensemble
from eks.stats import compute_mahalanobis, compute_pca
from eks.utils import crop_frames, format_data, make_dlc_pandas_index
from eks.marker_array import MarkerArray, input_dfs_to_markerArray, mA_to_stacked_array, stacked_array_to_mA


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
) -> tuple:
    """
    Fit the Ensemble Kalman Smoother for mirrored multi-camera data.

    Args:
        input_source: Directory path or list of CSV file paths with columns for all cameras.
        save_file: File to save output DataFrame.
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
        bodypart_list = keypoint_names

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

    markerArray = input_dfs_to_markerArray(camera_model_dfs, bodypart_list, camera_names)

    # Run the ensemble Kalman smoother for multi-camera data
    camera_dfs, smooth_params_final = ensemble_kalman_smoother_multicam(
        markerArray=markerArray,
        keypoint_names=bodypart_list,
        smooth_param=smooth_param,
        quantile_keep_pca=quantile_keep_pca,
        camera_names=camera_names,
        s_frames=s_frames,
        avg_mode=avg_mode,
        var_mode=var_mode,
        verbose=verbose,
        inflate_vars=inflate_vars
    )
    final_df = None
    for c, camera_df in enumerate(camera_dfs):
        suffix = f'{camera_names[c]}'
        new_columns = [(scorer, f'{kp}_{suffix}', attr) for scorer, kp, attr in camera_df.columns]
        camera_df.columns = pd.MultiIndex.from_tuples(new_columns, names=camera_df.columns.names)
        if final_df is None:
            final_df = camera_df
        else:
            pd.concat([final_df, camera_df], axis=1)

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

    markerArray = input_dfs_to_markerArray(input_dfs_list, bodypart_list, camera_names)

    # Run the ensemble Kalman smoother for multi-camera data
    camera_dfs, smooth_params_final = ensemble_kalman_smoother_multicam(
        markerArray=markerArray,
        keypoint_names=bodypart_list,
        smooth_param=smooth_param,
        quantile_keep_pca=quantile_keep_pca,
        camera_names=camera_names,
        s_frames=s_frames,
        avg_mode=avg_mode,
        var_mode=var_mode,
        verbose=verbose,
        inflate_vars=inflate_vars
    )

    # Save output DataFrames to CSVs (one per camera view)
    os.makedirs(save_dir, exist_ok=True)
    for c, camera in enumerate(camera_names):
        save_filename = f'multicam_{camera}_results_new.csv'
        camera_dfs[c].to_csv(os.path.join(save_dir, save_filename))

    # Save output DataFrames to CSVs (one per camera view)
    os.makedirs(save_dir, exist_ok=True)
    for c, camera in enumerate(camera_names):
        save_filename = f'multicam_{camera}_results.csv'
        camera_dfs[c].to_csv(os.path.join(save_dir, save_filename))
    return camera_dfs, smooth_params_final, input_dfs_list, bodypart_list


@typechecked
def ensemble_kalman_smoother_multicam(
        markerArray: MarkerArray,
        keypoint_names: list,
        smooth_param: float | list | None = None,
        quantile_keep_pca: float = 95.0,
        camera_names: list | None = None,
        s_frames: list | None = None,
        avg_mode: str = 'median',
        var_mode: str = 'confidence_weighted_var',
        inflate_vars: bool = False,
        verbose: bool = False,
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
        inflate_vars_kwargs: kwargs for compute_mahalanobis function when running variance inflation

        verbose: True to print out details

    Returns:
        tuple: Dataframes with smoothed predictions, final smoothing parameters.
    """

    n_models, n_cameras, n_frames, n_keypoints, _ = markerArray.shape()

    # MarkerArray (1, n_cameras, n_frames, n_keypoints, 5 (x, y, var_x, var_y, likelihood))
    ensemble_markerArray = jax_ensemble(markerArray, avg_mode=avg_mode, var_mode=var_mode)

    emA_preds = ensemble_markerArray.slice_fields("x", "y")
    emA_vars = ensemble_markerArray.slice_fields("var_x", "var_y")
    emA_likes = ensemble_markerArray.slice_fields("likelihood")

    (
        ensemble_pca,
        ensemble_ex_var,
        good_pcs_list,  # List-by-keypoint of (n_good_frames, n_pca_components)
        pcs_list,
        emA_scaled_preds,  # MarkerArray containing scaled predictions
        emA_means  # MarkerArray containing means across frames.
                   # Shape: (1, n_cameras, 1, n_keypoints, 2)
    ) = compute_pca(emA_preds, emA_vars, quantile_keep_pca, n_components=3)

    emA_inflated_vars = mA_compute_maha(emA_scaled_preds, emA_vars, emA_likes,
                                        inflate_vars=inflate_vars)
    # Kalman Filter Section ------------------------------------------

    # Collection array for marker output by camera view
    camera_arrs = [[] for _ in camera_names]
    for k, keypoint in enumerate(keypoint_names):

        # Initializations
        m0 = np.asarray([0.0, 0.0, 0.0])
        S0 = np.asarray([[np.var(good_pcs_list[k][:, 0]), 0.0, 0.0],
                         [0.0, np.var(good_pcs_list[k][:, 1]), 0.0],
                         [0.0, 0.0, np.var(good_pcs_list[k][:, 2])]])  # diagonal: var
        A = np.eye(3)
        d_t = good_pcs_list[k][1:] - good_pcs_list[k][:-1]
        C = ensemble_pca[k].components_.T
        R = np.eye(ensemble_pca[k].components_.shape[1])
        cov_matrix = np.cov(d_t.T)
        # emA to stacked array conversions
        inflated_vars_k = mA_to_stacked_array(emA_inflated_vars, k)
        scaled_preds_k = mA_to_stacked_array(emA_scaled_preds, k)
        # Smoothing parameter auto-tuning + final smooth
        smooth_param_final, ms, Vs, _, _ = multicam_optimize_smooth(
            cov_matrix, scaled_preds_k, m0, S0, C, A, R, inflated_vars_k, s_frames, smooth_param
        )

        if verbose:
            print(f"Smoothed {keypoint} at smooth_param={smooth_param_final}")

        y_m_smooth = np.dot(C, ms.T).T
        y_v_smooth = np.swapaxes(np.dot(C, np.dot(Vs, C.T)), 0, 1)

        # Final cleanup
        c_i = [[camera * 2, camera * 2 + 1] for camera in range(n_cameras)]
        for c, camera in enumerate(camera_names):
            data_arr = camera_arrs[c]
            x_i, y_i = c_i[c]

            data_arr.extend([
                y_m_smooth.T[x_i] + [emA_means.slice("keypoints", k).slice("cameras", c).slice_fields("x").get_array(squeeze=True)],
                y_m_smooth.T[y_i] + [emA_means.slice("keypoints", k).slice("cameras", c).slice_fields("y").get_array(squeeze=True)],
                emA_likes.slice("keypoints", k).slice("cameras", c).get_array(squeeze=True),
                emA_preds.slice("keypoints", k).slice("cameras", c).slice_fields("x").get_array(squeeze=True),
                emA_preds.slice("keypoints", k).slice("cameras", c).slice_fields("y").get_array(squeeze=True),
                emA_inflated_vars.slice("keypoints", k).slice("cameras", c).slice_fields("var_x").get_array(squeeze=True),
                emA_inflated_vars.slice("keypoints", k).slice("cameras", c).slice_fields("var_y").get_array(squeeze=True),
                y_v_smooth[:, x_i, x_i],
                y_v_smooth[:, y_i, y_i],
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

    return camera_dfs, smooth_param_final


def multicam_optimize_smooth(
    cov_matrix, y, m0, s0, C, A, R, ensemble_vars,
    s_frames=[(None, None)],
    smooth_param=None
):
    """
    Optimizes s using Nelder-Mead minimization, then smooths using s.
    Compatible with the singlecam and multicam examples.
    """
    # Optimize smooth_param
    if smooth_param is None:
        guess = compute_initial_guesses(ensemble_vars)

        # Update xatol during optimization
        def callback(xk):
            # Update xatol based on the current solution xk
            xatol = np.log(np.abs(xk)) * 0.01

            # Update the options dictionary with the new xatol value
            options['xatol'] = xatol

        # Initialize options with initial xatol
        options = {'xatol': np.log(guess)}

        # Unpack s_frames
        cropped_y = crop_frames(y, s_frames)

        # Minimize negative log likelihood
        sol = minimize(
            multicam_smooth_min,
            x0=guess,  # initial smooth param guess
            args=(cov_matrix, cropped_y, m0, s0, C, A, R, ensemble_vars),
            method='Nelder-Mead',
            options=options,
            callback=callback,  # Pass the callback function
            bounds=[(0, None)]
        )
        smooth_param = sol.x[0]
    # Final smooth with optimized s
    ms, Vs, nll, nll_values = multicam_smooth_final(
        smooth_param, cov_matrix, y, m0, s0, C, A, R, ensemble_vars)

    return smooth_param, ms, Vs, nll, nll_values


def multicam_smooth_final(smooth_param, cov_matrix, y, m0, S0, C, A, R, ensemble_vars):
    """
    Smooths once using the given smooth_param, used after optimizing smooth_param.
    Compatible with the singlecam and multicam example scripts.
    """
    # Adjust Q based on smooth_param and cov_matrix
    Q = smooth_param * cov_matrix
    # Run filtering and smoothing with the current smooth_param
    mf, Vf, S, innovs, innov_cov = forward_pass(y, m0, S0, C, R, A, Q, ensemble_vars)
    ms, Vs, CV = backward_pass(y, mf, Vf, S, A)
    # Compute the negative log-likelihood based on innovations and their covariance
    nll, nll_values = compute_nll(innovs, innov_cov)
    return ms, Vs, nll, nll_values


def multicam_smooth_min(smooth_param, cov_matrix, y, m0, S0, C, A, R, ensemble_vars):
    """
    Smooths once using the given smooth_param. Returns only the nll, which is the parameter to
    be minimized using the scipy.minimize() function
    """
    # Adjust Q based on smooth_param and cov_matrix
    Q = smooth_param * cov_matrix
    # Run filtering with the current smooth_param
    mf, Vf, S, innovs, innov_cov = forward_pass(y, m0, S0, C, R, A, Q, ensemble_vars)
    # Compute the negative log-likelihood based on innovations and their covariance
    nll, nll_values = compute_nll(innovs, innov_cov)
    return nll


def mA_compute_maha(scaled_emA_preds, emA_vars, emA_likes,
                    inflate_vars=False, threshold=5, scalar=2):
    """
    Reshape marker arrays for Mahalanobis computation, compute Mahalanobis distances,
    and optionally inflate variances for all keypoints.

    Args:
        scaled_emA_preds (numpy.ndarray): Scaled predicted marker positions.
        emA_vars (numpy.ndarray): Variance values associated with predictions.
        emA_likes (numpy.ndarray): Likelihood values associated with predictions.
        inflate_vars (bool, optional): Whether to inflate variances based on Mahalanobis distances.
        threshold (float, optional): Mahalanobis distance threshold for inflation.
        scalar (float, optional): Factor by which to inflate variance.

    Returns:
        list: A list of tuples, where each tuple contains (maha_results, inflated_ens_vars) for a keypoint.
    """
    _, n_cameras, _, n_keypoints, _ = scaled_emA_preds.shape()

    emA_inflated_vars_list = []
    for k in range(n_keypoints):
        # Transform mA into array of Shape: (n_models, n_frames, n_cameras * n_fields)
        preds = mA_to_stacked_array(scaled_emA_preds, k)
        vars = mA_to_stacked_array(emA_vars, k)
        likes = mA_to_stacked_array(emA_likes, k)

        if inflate_vars:
            inflated = True
            tmp_vars = vars
            while inflated:
                # Compute Mahalanobis distances
                maha_results = compute_mahalanobis(preds, tmp_vars, likelihoods=likes)
                # Inflate variances based on Mahalanobis distances
                inflated_ens_vars_k, inflated = inflate_variance(
                    tmp_vars, maha_results['mahalanobis'], threshold, scalar)
                tmp_vars = inflated_ens_vars_k
        else:
            inflated_ens_vars_k = vars
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

