import os
import warnings
from functools import partial

import jax
import numpy as np
import optax
import pandas as pd
from jax import jit
from jax import numpy as jnp
from typeguard import typechecked

from eks.core import backward_pass, ensemble, forward_pass
from eks.marker_array import MarkerArray, input_dfs_to_markerArray
from eks.utils import crop_frames, format_data, make_dlc_pandas_index


@typechecked
def get_pupil_location(dlc: dict) -> np.ndarray:
    """get mean of both pupil diameters
    d1 = top - bottom, d2 = left - right
    and in addition assume it's a circle and
    estimate diameter from other pairs of points
    Author: Michael Schartner
    """
    s = 1
    t = np.vstack((dlc['pupil_top_r_x'], dlc['pupil_top_r_y'])).T / s
    b = np.vstack((dlc['pupil_bottom_r_x'], dlc['pupil_bottom_r_y'])).T / s
    le = np.vstack((dlc['pupil_left_r_x'], dlc['pupil_left_r_y'])).T / s
    r = np.vstack((dlc['pupil_right_r_x'], dlc['pupil_right_r_y'])).T / s
    center = np.zeros(t.shape)

    # ok if either top or bottom is nan in x-dir
    tmp_x1 = np.nanmedian(np.hstack([t[:, 0, None], b[:, 0, None]]), axis=1)
    # both left and right must be present in x-dir
    tmp_x2 = np.median(np.hstack([r[:, 0, None], le[:, 0, None]]), axis=1)
    center[:, 0] = np.nanmedian(np.hstack([tmp_x1[:, None], tmp_x2[:, None]]), axis=1)

    # both top and bottom must be present in ys-dir
    tmp_y1 = np.median(np.hstack([t[:, 1, None], b[:, 1, None]]), axis=1)
    # ok if either left or right is nan in ys-dir
    tmp_y2 = np.nanmedian(np.hstack([r[:, 1, None], le[:, 1, None]]), axis=1)
    center[:, 1] = np.nanmedian(np.hstack([tmp_y1[:, None], tmp_y2[:, None]]), axis=1)
    return center


@typechecked
def get_pupil_diameter(dlc: dict) -> np.ndarray:
    """
    from: https://int-brain-lab.github.io/iblenv/_modules/brainbox/behavior/dlc.html
    Estimates pupil diameter by taking median of different computations.

    The two most straightforward estimates: d1 = top - bottom, d2 = left - right
    In addition, assume the pupil is a circle and estimate diameter from other pairs of points

    :param dlc: dlc pqt table with pupil estimates, should be likelihood thresholded (e.g. at 0.9)
    :return: np.array, pupil diameter estimate for each time point, shape (n_frames,)
    """
    diameters = []
    # Get the x,ys coordinates of the four pupil points
    top, bottom, left, right = [
        np.vstack((dlc[f'pupil_{point}_r_x'], dlc[f'pupil_{point}_r_y']))
        for point in ['top', 'bottom', 'left', 'right']
    ]
    # First compute direct diameters
    diameters.append(np.linalg.norm(top - bottom, axis=0))
    diameters.append(np.linalg.norm(left - right, axis=0))

    # For non-crossing edges, estimate diameter via circle assumption
    for pair in [(top, left), (top, right), (bottom, left), (bottom, right)]:
        diameters.append(np.linalg.norm(pair[0] - pair[1], axis=0) * 2 ** 0.5)

    # Ignore all nan runtime warning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmedian(diameters, axis=0)


def add_mean_to_array(pred_arr, keys, mean_x, mean_y):
    pred_arr_copy = pred_arr.copy()
    processed_arr_dict = {}
    for i, key in enumerate(keys):
        if 'x' in key:
            processed_arr_dict[key] = pred_arr_copy[:, i] + mean_x
        else:
            processed_arr_dict[key] = pred_arr_copy[:, i] + mean_y
    return processed_arr_dict


@typechecked
def fit_eks_pupil(
    input_source: str | list,
    save_file: str,
    smooth_params: list | None = None,
    s_frames: list | None = None,
    avg_mode: str = 'median',
    var_mode: str = 'confidence_weighted_var',
    verbose: bool = False,
) -> tuple:
    """Fit the Ensemble Kalman Smoother for the ibl-pupil dataset.

    Args:
        input_source: directory path or list of CSV file paths. If a directory path, all files
            within this directory will be used.
        save_file: File to save output dataframe.
        smooth_params: [diameter param, center of mass param]
            each value should be in (0, 1); closer to 1 means more smoothing
        s_frames: Frames for automatic optimization if needed.
        avg_mode: mode for averaging across ensemble
            'median' | 'mean'
        var_mode: mode for computing ensemble variance
            'var' | 'confidence_weighted_var'
        verbose: Extra print statements if True

    Returns:
        tuple:
            df_smoothed (pd.DataFrame)
            smooth_params (list): Final smoothing parameters used.
            input_dfs_list (list): List of input DataFrames.
            keypoint_names (list): List of keypoint names.

    """
    # pupil smoother only works for a pre-specified set of points
    # NOTE: this order MUST be kept
    bodypart_list = ['pupil_top_r', 'pupil_bottom_r', 'pupil_right_r', 'pupil_left_r']

    # Load and format input files
    input_dfs_list, _ = format_data(input_source)
    print(f"Input data loaded for keypoints: {bodypart_list}")
    marker_array = input_dfs_to_markerArray([input_dfs_list], bodypart_list, [""])

    # Run the ensemble Kalman smoother
    df_smoothed, smooth_params_final = ensemble_kalman_smoother_ibl_pupil(
        marker_array=marker_array,
        keypoint_names=bodypart_list,
        smooth_params=smooth_params,
        s_frames=s_frames,
        avg_mode=avg_mode,
        var_mode=var_mode,
        verbose=verbose
    )

    # Save the output DataFrame to CSV
    os.makedirs(os.path.dirname(save_file), exist_ok=True)
    df_smoothed.to_csv(save_file)
    print("DataFrames successfully converted to CSV")

    return df_smoothed, smooth_params_final, input_dfs_list, bodypart_list


@typechecked
def ensemble_kalman_smoother_ibl_pupil(
    marker_array: MarkerArray,
    keypoint_names: list,
    smooth_params: list | None = None,
    s_frames: list | None = None,
    avg_mode: str = 'median',
    var_mode: str = 'confidence_weighted_var',
    verbose: bool = False
) -> tuple:
    """Perform Ensemble Kalman Smoothing on pupil data.

    Args:
        marker_array: MarkerArray object containing marker data.
            Shape (n_models, n_cameras, n_frames, n_keypoints, 3 (for x, y, likelihood))
        markers_list: pd.DataFrames
            each list element is a dataframe of predictions from one ensemble member
        keypoint_names: List of body parts to run smoothing on
        smooth_params: contains smoothing parameters for diameter and center of mass
        s_frames: frames for automatic optimization if s is not provided
        avg_mode: mode for averaging across ensemble
            'median' | 'mean'
        var_mode: mode for computing ensemble variance
            'var' | 'confidence_weighted_var'
        verbose: True to print out details

    Returns:
        tuple:
            smoothed markers dataframe
            final smooth params values
            final nll

    """
    n_models, n_cameras, n_frames, n_keypoints, n_data_fields = marker_array.shape
    keys = [f'{kp}_{coord}' for kp in keypoint_names for coord in ['x', 'y']]

    # Compute ensemble information
    # MarkerArray (1, 1, n_frames, n_keypoints, 5 (x, y, var_x, var_y, likelihood))
    ensemble_marker_array = ensemble(marker_array, avg_mode=avg_mode, var_mode=var_mode)
    emA_unsmoothed_preds = ensemble_marker_array.slice_fields("x", "y")
    emA_vars = ensemble_marker_array.slice_fields("var_x", "var_y")
    emA_likes = ensemble_marker_array.slice_fields("likelihood")

    # Extract stacked arrays (predicted coordinates, variances, likelihoods)
    ensemble_preds = emA_unsmoothed_preds.get_array(squeeze=True).reshape(n_frames, -1)
    ensemble_vars = emA_vars.get_array(squeeze=True).reshape(n_frames, -1)
    ensemble_likes = emA_likes.get_array(squeeze=True)

    # Compute center of mass + diameter
    pupil_diameters = get_pupil_diameter({key: ensemble_preds[:, i] for i, key in enumerate(keys)})
    pupil_locations = get_pupil_location({key: ensemble_preds[:, i] for i, key in enumerate(keys)})
    mean_x_obs = np.mean(pupil_locations[:, 0])
    mean_y_obs = np.mean(pupil_locations[:, 1])

    # Center predictions
    x_t_obs, y_t_obs = pupil_locations[:, 0] - mean_x_obs, pupil_locations[:, 1] - mean_y_obs

    # -------------------------------------------------------
    # Set values for Kalman filter (Specific to this dataset)
    # -------------------------------------------------------
    # initial state: mean
    m0 = np.asarray([np.mean(pupil_diameters), 0.0, 0.0])

    # diagonal: var
    S0 = np.asarray([
        [np.nanvar(pupil_diameters), 0.0, 0.0],
        [0.0, np.nanvar(x_t_obs), 0.0],
        [0.0, 0.0, np.nanvar(y_t_obs)]
    ])

    # Measurement function
    C = np.asarray([
        [0, 1, 0], [-.5, 0, 1],
        [0, 1, 0], [.5, 0, 1],
        [.5, 1, 0], [0, 0, 1],
        [-.5, 1, 0], [0, 0, 1]
    ])

    # placeholder diagonal matrix for ensemble variance
    R = np.eye(8)

    centered_ensemble_preds = ensemble_preds.copy()
    # subtract COM means from the ensemble predictions
    for i in range(ensemble_preds.shape[1]):
        if i % 2 == 0:
            centered_ensemble_preds[:, i] -= mean_x_obs
        else:
            centered_ensemble_preds[:, i] -= mean_y_obs
    y_obs = centered_ensemble_preds

    # -------------------------------------------------------
    # Perform filtering with SINGLE PAIR of diameter_s, com_s
    # -------------------------------------------------------
    s_finals, ms, Vs, nll = pupil_optimize_smooth(
        y_obs, m0, S0, C, R, ensemble_vars,
        np.var(pupil_diameters), np.var(x_t_obs), np.var(y_t_obs), s_frames, smooth_params,
        verbose=verbose)
    if verbose:
        print(f"diameter_s={s_finals[0]}, com_s={s_finals[1]}")
    # Smoothed posterior over ys
    y_m_smooth = np.dot(C, ms.T).T
    y_v_smooth = np.swapaxes(np.dot(C, np.dot(Vs, C.T)), 0, 1)

    # --------------------------------------
    # cleanup
    # --------------------------------------

    # collect data
    processed_arr_dict = add_mean_to_array(y_m_smooth, keys, mean_x_obs, mean_y_obs)
    key_pair_list = [
        ['pupil_top_r_x', 'pupil_top_r_y'],
        ['pupil_right_r_x', 'pupil_right_r_y'],
        ['pupil_bottom_r_x', 'pupil_bottom_r_y'],
        ['pupil_left_r_x', 'pupil_left_r_y'],
    ]
    ensemble_indices = [(0, 1), (4, 5), (2, 3), (6, 7)]
    data_arr = []
    for i, key_pair in enumerate(key_pair_list):
        # keep track of labels for each data entry
        labels = []
        # smoothed x vals
        data_arr.append(processed_arr_dict[key_pair[0]])
        labels.append('x')
        # smoothed y vals
        data_arr.append(processed_arr_dict[key_pair[1]])
        labels.append('y')
        # mean likelihood
        data_arr.append(ensemble_likes[:, i])
        labels.append('likelihood')
        # x vals ensemble median
        data_arr.append(ensemble_preds[:, ensemble_indices[i][0]])
        labels.append('x_ens_median')
        # y vals ensemble median
        data_arr.append(ensemble_preds[:, ensemble_indices[i][1]])
        labels.append('y_ens_median')
        # x vals ensemble variance
        data_arr.append(ensemble_vars[:, ensemble_indices[i][0]])
        labels.append('x_ens_var')
        # y vals ensemble variance
        data_arr.append(ensemble_vars[:, ensemble_indices[i][1]])
        labels.append('y_ens_var')
        # x vals posterior variance
        data_arr.append(y_v_smooth[:, i, i])
        labels.append('x_posterior_var')
        # y vals posterior variance
        data_arr.append(y_v_smooth[:, i + 1, i + 1])
        labels.append('y_posterior_var')

    data_arr = np.asarray(data_arr)

    # put data in dataframe
    pdindex = make_dlc_pandas_index(keypoint_names, labels=labels)
    markers_df = pd.DataFrame(data_arr.T, columns=pdindex)

    return markers_df, s_finals


def pupil_optimize_smooth(
        ys: np.ndarray,
        m0: np.ndarray,
        S0: np.ndarray,
        C: np.ndarray,
        R: np.ndarray,
        ensemble_vars: np.ndarray,
        diameters_var: np.ndarray,
        x_var: np.ndarray,
        y_var: np.ndarray,
        s_frames: list | None = [(1, 2000)],
        smooth_params: list | None = [None, None],
        maxiter: int = 1000,
        verbose: bool = False,
) -> tuple:
    """Optimize-and-smooth function for the pupil example script.

    Parameters:
        ys: Observations. Shape (keypoints, frames, coordinates).
        m0: Initial mean state.
        S0: Initial state covariance.
        C: Measurement function.
        R: Measurement noise covariance.
        ensemble_vars: Ensemble variances.
        diameters_var: Diameter variance
        x_var: x variance for COM
        y_var: y variance for COM
        s_frames: List of frames.
        smooth_params: Smoothing parameter tuple (diameter_s, com_s)
        verbose: Prints extra information for smoothing parameter iterations

    Returns:
        tuple: Final smoothing parameters, smoothed means, smoothed covariances,
               negative log-likelihoods, negative log-likelihood values.
    """

    @partial(jit)
    def nll_loss_sequential_scan(
            s_log, ys, m0, S0, C, R, ensemble_vars, diameters_var, x_var, y_var):
        s = jnp.exp(s_log)  # Ensure positivity
        return pupil_smooth(
            s, ys, m0, S0, C, R, ensemble_vars, diameters_var, x_var, y_var)

    loss_function = nll_loss_sequential_scan
    # Optimize smooth_param
    if smooth_params is None or smooth_params[0] is None or smooth_params[1] is None:
        # Crop to only contain s_frames for time axis
        y_cropped = crop_frames(ys, s_frames)
        ensemble_vars_cropped = crop_frames(ensemble_vars, s_frames)

        # Optimize negative log likelihood
        s_init = jnp.log(jnp.array([0.99, 0.98]))  # reasonable guess for s_finals
        optimizer = optax.adam(learning_rate=0.005)
        opt_state = optimizer.init(s_init)

        def step(s, opt_state):
            loss, grads = jax.value_and_grad(loss_function)(
                s, y_cropped, m0, S0, C, R, ensemble_vars_cropped, diameters_var, x_var, y_var
            )
            updates, opt_state = optimizer.update(grads, opt_state)
            s = optax.apply_updates(s, updates)
            return s, opt_state, loss

        prev_loss = jnp.inf
        for iteration in range(maxiter):
            s_init, opt_state, loss = step(s_init, opt_state)

            if verbose and iteration % 10 == 0 or iteration == maxiter - 1:
                print(f'Iteration {iteration}, Current loss: {loss}, Current s: {jnp.exp(s_init)}')

            tol = 1e-6 * jnp.abs(jnp.log(prev_loss))
            if jnp.linalg.norm(loss - prev_loss) < tol + 1e-6:
                break
            prev_loss = loss

        s_finals = jnp.exp(s_init)
        s_finals = [round(s_finals[0], 5), round(s_finals[1], 5)]
        print(f'Optimized to diameter_s={s_finals[0]}, com_s={s_finals[1]}')
    else:
        s_finals = smooth_params

    # Final smooth with optimized s
    ms, Vs, nll = pupil_smooth(
        s_finals, ys, m0, S0, C, R, ensemble_vars, diameters_var, x_var, y_var, return_full=True)

    return s_finals, ms, Vs, nll


def pupil_smooth(smooth_params, ys, m0, S0, C, R, ensemble_vars, diameters_var, x_var, y_var,
                     return_full=False):
    """
    Smooths once using the given smooth_param. Returns only the nll loss by default
    (if return_full is False).

    Parameters:
    smooth_params (float): Smoothing parameter.
    block (list): List of blocks.
    cov_mats (np.ndarray): Covariance matrices.
    ys (np.ndarray): Observations.
    m0s (np.ndarray): Initial mean state.
    S0s (np.ndarray): Initial state covariance.
    Cs (np.ndarray): Measurement function.
    As (np.ndarray): State-transition matrix.
    Rs (np.ndarray): Measurement noise covariance.

    Returns:
    float: Negative log-likelihood.
    """
    # Construct As
    diameter_s, com_s = smooth_params[0], smooth_params[1]
    A = jnp.array([
        [diameter_s, 0, 0],
        [0, com_s, 0],
        [0, 0, com_s]
    ])

    # Construct cov_matrix Q
    Q = jnp.array([
        [diameters_var * (1 - (A[0, 0] ** 2)), 0, 0],
        [0, x_var * (1 - A[1, 1] ** 2), 0],
        [0, 0, y_var * (1 - (A[2, 2] ** 2))]
    ])

    mf, Vf, nll = forward_pass(ys, m0, S0, A, Q, C, ensemble_vars)

    if return_full:
        ms, Vs = backward_pass(mf, Vf, A, Q)
        return ms, Vs, nll

    return nll