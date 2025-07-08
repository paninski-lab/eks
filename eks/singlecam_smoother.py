import os

import jax.numpy as jnp
import numpy as np
import pandas as pd
from typeguard import typechecked

from eks.core import ensemble, optimize_smooth_param
from eks.marker_array import MarkerArray, input_dfs_to_markerArray
from eks.utils import center_predictions, format_data, make_dlc_pandas_index


@typechecked
def fit_eks_singlecam(
    input_source: str | list,
    save_file: str,
    bodypart_list: list | None = None,
    smooth_param: float | list | None = None,
    s_frames: list | None = None,
    blocks: list = [],
    avg_mode: str = 'median',
    var_mode: str = 'confidence_weighted_var',
    verbose: bool = False,
) -> tuple:
    """Fit the Ensemble Kalman Smoother for single-camera data.

    Args:
        input_source: directory path or list of CSV file paths. If a directory path, all files
            within this directory will be used.
        save_file: File to save output dataframe.
        bodypart_list: list of body parts to analyze.
        smooth_param: value in (0, Inf); smaller values lead to more smoothing
        s_frames: Frames for automatic optimization if smooth_param is not provided.
        blocks: keypoints to be blocked for correlated noise. Generates on smoothing param per
            block, as opposed to per keypoint.
            Specified by the form "x1, x2, x3; y1, y2" referring to keypoint indices (start at 0)
        avg_mode: mode for averaging across ensemble
            'median' | 'mean'
        var_mode: mode for computing ensemble variance
            'var' | 'confidence_weighted_var'
        verbose: Extra print statements if True

    Returns:
        tuple:
            df_smoothed (pd.DataFrame)
            s_finals (list): List of optimized smoothing factors for each keypoint.
            input_dfs (list): List of input DataFrames for plotting.
            bodypart_list (list): List of body parts used.

    """
    # Load and format input files using the unified format_data function
    input_dfs_list, keypoint_names = format_data(input_source)

    if bodypart_list is None:
        bodypart_list = keypoint_names
        print(f'Input data loaded for keypoints:\n{bodypart_list}')
    marker_array = input_dfs_to_markerArray([input_dfs_list], bodypart_list, [""])
    # Run the ensemble Kalman smoother
    df_smoothed, smooth_params_final = ensemble_kalman_smoother_singlecam(
        marker_array=marker_array,
        keypoint_names=bodypart_list,
        smooth_param=smooth_param,
        s_frames=s_frames,
        blocks=blocks,
        avg_mode=avg_mode,
        var_mode=var_mode,
        verbose=verbose,
    )

    # Save the output DataFrame to CSV
    os.makedirs(os.path.dirname(save_file), exist_ok=True)
    df_smoothed.to_csv(save_file)
    print("DataFrames successfully converted to CSV")

    return df_smoothed, smooth_params_final, input_dfs_list, bodypart_list


@typechecked
def ensemble_kalman_smoother_singlecam(
    marker_array: MarkerArray,
    keypoint_names: list,
    smooth_param: float | list | None = None,
    s_frames: list | None = None,
    blocks: list = [],
    avg_mode: str = 'median',
    var_mode: str = 'confidence_weighted_var',
    verbose: bool = False,
) -> tuple:
    """Perform Ensemble Kalman Smoothing for single-camera data.

    Args:
        marker_array: MarkerArray object containing marker data.
            Shape (n_models, n_cameras, n_frames, n_keypoints, 3 (for x, y, likelihood))
        keypoint_names: List of body parts to run smoothing on
        smooth_param: value in (0, Inf); smaller values lead to more smoothing
        s_frames: List of frames for automatic computation of smoothing parameter
        blocks: keypoints to be blocked for correlated noise. Generates on smoothing param per
            block, as opposed to per keypoint.
            Specified by the form "x1, x2, x3; y1, y2" referring to keypoint indices (start at 0)
        avg_mode: mode for averaging across ensemble
            'median' | 'mean'
        var_mode: mode for computing ensemble variance
            'var' | 'confidence_weighted_var'
        verbose: True to print out details

    Returns:
        tuple: Dataframes with smoothed predictions, final smoothing parameters.

    """
    n_models, n_cameras, n_frames, n_keypoints, n_data_fields = marker_array.shape

    # MarkerArray (1, 1, n_frames, n_keypoints, 5 (x, y, var_x, var_y, likelihood))
    ensemble_marker_array = ensemble(marker_array, avg_mode=avg_mode, var_mode=var_mode)
    emA_unsmoothed_preds = ensemble_marker_array.slice_fields("x", "y")
    emA_vars = ensemble_marker_array.slice_fields("var_x", "var_y")
    emA_likes = ensemble_marker_array.slice_fields("likelihood")

    # Save ensemble medians for output
    emA_medians = MarkerArray(
        marker_array=emA_unsmoothed_preds,
        data_fields=["x_median", "y_median"],
    )

    # Create new MarkerArray with centered predictions
    _, emA_centered_preds, _, emA_means = center_predictions(
        ensemble_marker_array, quantile_keep_pca=100,
    )
    # MarkerArray data_fields=["x", "y", "likelihood", "var_x", "var_y"]
    ensemble_marker_array = MarkerArray.stack_fields(
        emA_centered_preds,
        emA_likes,
        emA_vars,
    )

    # Prepare params for singlecam_optimize_smooth()
    ys = emA_centered_preds.get_array(squeeze=True).transpose(1, 0, 2)
    m0s, S0s, As, cov_mats, Cs = initialize_kalman_filter(emA_centered_preds)

    # Main smoothing function
    s_finals, ms, Vs = optimize_smooth_param(
        cov_mats, ys, m0s, S0s, Cs, As, emA_vars.get_array(squeeze=True),
        s_frames, smooth_param, blocks, verbose=verbose,
    )

    y_m_smooths = np.zeros((n_keypoints, n_frames, 2))
    y_v_smooths = np.zeros((n_keypoints, n_frames, 2, 2))

    # Make emAs for smoothed preds and posterior variances
    emA_smoothed_preds_list = []
    emA_postvars_list = []
    for k in range(n_keypoints):
        y_m_smooths[k] = np.dot(Cs[k], ms[k].T).T
        y_v_smooths[k] = np.swapaxes(np.dot(Cs[k], np.dot(Vs[k], Cs[k].T)), 0, 1)
        mean_x_obs = emA_means.slice("keypoints", k).slice_fields("x").get_array(squeeze=True)
        mean_y_obs = emA_means.slice("keypoints", k).slice_fields("y").get_array(squeeze=True)

        # Unscale (re-add means to) smoothed x and y
        smoothed_xs_k = y_m_smooths[k].T[0] + mean_x_obs
        smoothed_ys_k = y_m_smooths[k].T[1] + mean_y_obs

        # Reshape into MarkerArray format
        smoothed_xs_k = smoothed_xs_k[None, None, :, None, None]
        smoothed_ys_k = smoothed_ys_k[None, None, :, None, None]

        # Create smoothed preds emA for current keypoint
        emA_smoothed_xs_k = MarkerArray(smoothed_xs_k, data_fields=["x"])
        emA_smoothed_ys_k = MarkerArray(smoothed_ys_k, data_fields=["y"])
        emA_smoothed_preds_k = MarkerArray.stack_fields(emA_smoothed_xs_k, emA_smoothed_ys_k)
        emA_smoothed_preds_list.append(emA_smoothed_preds_k)

        # Create posterior variance emA for current keypoint
        postvar_xs_k = y_v_smooths[k][:, 0, 0]
        postvar_ys_k = y_v_smooths[k][:, 1, 1]
        postvar_xs_k = postvar_xs_k[None, None, :, None, None]
        postvar_ys_k = postvar_ys_k[None, None, :, None, None]
        emA_postvar_xs_k = MarkerArray(postvar_xs_k, data_fields=["postvar_x"])
        emA_postvar_ys_k = MarkerArray(postvar_ys_k, data_fields=["postvar_y"])
        emA_postvars_k = MarkerArray.stack_fields(emA_postvar_xs_k, emA_postvar_ys_k)
        emA_postvars_list.append(emA_postvars_k)

    emA_smoothed_preds = MarkerArray.stack(emA_smoothed_preds_list, "keypoints")
    emA_postvars = MarkerArray.stack(emA_postvars_list, "keypoints")

    # Create Final MarkerArray
    emA_final = MarkerArray.stack_fields(
        emA_smoothed_preds,  # x, y
        ensemble_marker_array.slice_fields("likelihood"),  # likelihood
        emA_medians,  # x_median, y_median
        ensemble_marker_array.slice_fields("var_x", "var_y"),  # var_x, var_y
        emA_postvars  # postvar_x, postvar_y
    )

    labels = [
        'x', 'y', 'likelihood', 'x_ens_median', 'y_ens_median',
        'x_ens_var', 'y_ens_var', 'x_posterior_var', 'y_posterior_var'
    ]

    final_array = emA_final.get_array(squeeze=True)

    # Put data into dataframe
    pdindex = make_dlc_pandas_index(keypoint_names, labels=labels)
    final_array = final_array.reshape(n_frames, n_keypoints * len(labels))
    markers_df = pd.DataFrame(final_array, columns=pdindex)

    return markers_df, s_finals


def initialize_kalman_filter(emA_centered_preds: MarkerArray) -> tuple:
    """
    Initialize the Kalman filter values.

    Parameters:
        centered_ensemble_preds (MarkerArray): centered ensemble predictions.

    Returns:
        tuple: Initial Kalman filter values and covariance matrices.
    """
    _, _, _, n_keypoints, _ = emA_centered_preds.shape

    # Shape: (n_frames, n_keypoints, 2 (for x, y))
    centered_preds = emA_centered_preds.slice_fields("x", "y").get_array(squeeze=True)

    m0s = np.zeros((n_keypoints, 2))  # Initial state means: (n_keypoints, 2)
    S0s = np.array([
        [[np.nanvar(centered_preds[:, k, 0]), 0.0],  # [var(x)  0 ]
         [0.0, np.nanvar(centered_preds[:, k, 1])]]  # [ 0  var(y)]
        for k in range(n_keypoints)
    ])  # Initial covariance matrices: (n_keypoints, 2, 2)

    # State-transition and measurement matrices
    As = np.tile(np.eye(2), (n_keypoints, 1, 1))  # (n_keypoints, 2, 2)
    Cs = np.tile(np.eye(2), (n_keypoints, 1, 1))  # (n_keypoints, 2, 2)

    # Compute covariance matrices
    cov_mats = []
    for i in range(n_keypoints):
        cov_mats.append([[1, 0], [0, 1]])
    cov_mats = np.array(cov_mats)

    return (
        jnp.array(m0s),
        jnp.array(S0s),
        jnp.array(As),
        jnp.array(cov_mats),
        jnp.array(Cs),
    )
