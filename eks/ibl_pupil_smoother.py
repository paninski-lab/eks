import os
import warnings
from typing import Optional, Union

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from eks.core import backward_pass, compute_nll, eks_zscore, ensemble, forward_pass
from eks.utils import crop_frames, make_dlc_pandas_index, format_data


def get_pupil_location(dlc):
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


def get_pupil_diameter(dlc):
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


def fit_eks_pupil(
    input_source: Union[str, list],
    save_file: str,
    smooth_params: Optional[list] = None,
    s_frames: Optional[list] = None,
    avg_mode: str = 'median',
    var_mode: str = 'confidence_weighted_var',
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

    Returns:
        tuple:
            df_smoothed (pd.DataFrame)
            smooth_params (list): Final smoothing parameters used.
            input_dfs_list (list): List of input DataFrames.
            keypoint_names (list): List of keypoint names.
            nll_values (list): List of NLL values.

    """

    # Load and format input files
    input_dfs_list, _, keypoint_names = format_data(input_source)

    print(f"Input data loaded for keypoints: {keypoint_names}")

    # Run the ensemble Kalman smoother
    df_smoothed, smooth_params_final, nll_values = ensemble_kalman_smoother_ibl_pupil(
        markers_list=input_dfs_list,
        smooth_params=smooth_params,
        s_frames=s_frames,
        avg_mode=avg_mode,
        var_mode=var_mode,
    )

    # Save the output DataFrame to CSV
    os.makedirs(os.path.dirname(save_file), exist_ok=True)
    df_smoothed.to_csv(save_file)
    print("DataFrames successfully converted to CSV")

    return df_smoothed, smooth_params_final, input_dfs_list, keypoint_names, nll_values


def ensemble_kalman_smoother_ibl_pupil(
    markers_list: list,
    smooth_params: Optional[list] = None,
    s_frames: Optional[list] = None,
    avg_mode: str = 'median',
    var_mode: str = 'confidence_weighted_var',
    zscore_threshold: float = 2,
) -> tuple:
    """Perform Ensemble Kalman Smoothing on pupil data.

    Args:
        markers_list: pd.DataFrames
            each list element is a dataframe of predictions from one ensemble member
        smooth_params: contains smoothing parameters for diameter and center of mass
        s_frames: frames for automatic optimization if s is not provided
        avg_mode
            'median' | 'mean'
        var_mode
            'confidence_weighted_var' | 'var'
        zscore_threshold: Minimum std threshold to reduce the effect of low ensemble std on a
            zscore metric (default 2).

    Returns:
        tuple:
            smoothed markers dataframe
            final smooth params values
            final nll

    """

    # pupil smoother only works for a pre-specified set of points
    # NOTE: this order MUST be kept
    keypoint_names = ['pupil_top_r', 'pupil_bottom_r', 'pupil_right_r', 'pupil_left_r']
    keys = [f'{kp}_{coord}' for kp in keypoint_names for coord in ['x', 'y']]

    # compute ensemble information
    ensemble_preds, ensemble_vars, ensemble_likes, _ = ensemble(
        markers_list, keys, avg_mode=avg_mode, var_mode=var_mode,
    )

    # compute center of mass + diameter
    pupil_diameters = get_pupil_diameter({key: ensemble_preds[:, i] for i, key in enumerate(keys)})
    pupil_locations = get_pupil_location({key: ensemble_preds[:, i] for i, key in enumerate(keys)})
    mean_x_obs = np.mean(pupil_locations[:, 0])
    mean_y_obs = np.mean(pupil_locations[:, 1])

    # make the mean zero
    x_t_obs, y_t_obs = pupil_locations[:, 0] - mean_x_obs, pupil_locations[:, 1] - mean_y_obs

    # --------------------------------------
    # Set values for kalman filter
    # --------------------------------------
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

    scaled_ensemble_preds = ensemble_preds.copy()
    # subtract COM means from the ensemble predictions
    for i in range(ensemble_preds.shape[1]):
        if i % 2 == 0:
            scaled_ensemble_preds[:, i] -= mean_x_obs
        else:
            scaled_ensemble_preds[:, i] -= mean_y_obs
    y_obs = scaled_ensemble_preds

    # --------------------------------------
    # perform filtering
    # --------------------------------------
    smooth_params, ms, Vs, nll, nll_values = pupil_optimize_smooth(
        y_obs, m0, S0, C, R, ensemble_vars,
        np.var(pupil_diameters), np.var(x_t_obs), np.var(y_t_obs), s_frames, smooth_params)
    diameter_s, com_s = smooth_params[0], smooth_params[1]
    print(f"NLL is {nll} for diameter_s={diameter_s}, com_s={com_s}")
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
        data_arr.append(ensemble_likes[:, ensemble_indices[i][0]])
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

        # compute zscore for EKS to see how it deviates from the ensemble
        # eks_predictions = \
        #     np.asarray([processed_arr_dict[key_pair[0]], processed_arr_dict[key_pair[1]]]).T
        # ensemble_preds_curr = ensemble_preds[:, ensemble_indices[i][0]: ensemble_indices[i][1] + 1]
        # ensemble_vars_curr = ensemble_vars[:, ensemble_indices[i][0]: ensemble_indices[i][1] + 1]
        # zscore, _ = eks_zscore(
        #     eks_predictions,
        #     ensemble_preds_curr,
        #     ensemble_vars_curr,
        #     min_ensemble_std=zscore_threshold,
        # )
        # data_arr.append(zscore)

    data_arr = np.asarray(data_arr)

    # put data in dataframe
    pdindex = make_dlc_pandas_index(keypoint_names, labels=labels)
    markers_df = pd.DataFrame(data_arr.T, columns=pdindex)

    return markers_df, smooth_params, nll_values


def pupil_optimize_smooth(
    y, m0, S0, C, R, ensemble_vars, diameters_var, x_var, y_var,
    s_frames: Optional[list] = [(1, 2000)],
    smooth_params: Optional[list] = [None, None],
):
    """Optimize-and-smooth function for the pupil example script."""
    # Optimize smooth_param
    if smooth_params is None or smooth_params[0] is None or smooth_params[1] is None:

        # Unpack s_frames
        y_shortened = crop_frames(y, s_frames)

        # Minimize negative log likelihood
        smooth_params = minimize(
            pupil_smooth_min,  # function to minimize
            x0=[1, 1],
            args=(y_shortened, m0, S0, C, R, ensemble_vars, diameters_var, x_var, y_var),
            method='Nelder-Mead',
            tol=0.002,
            bounds=[(0, 1), (0, 1)]  # bounds for each parameter in smooth_params
        )
        smooth_params = [round(smooth_params.x[0], 5), round(smooth_params.x[1], 5)]
        print(f'Optimal at diameter_s={smooth_params[0]}, com_s={smooth_params[1]}')

    # Final smooth with optimized s
    ms, Vs, nll, nll_values = pupil_smooth_final(
        y, smooth_params, m0, S0, C, R, ensemble_vars, diameters_var, x_var, y_var)

    return smooth_params, ms, Vs, nll, nll_values


def pupil_smooth_final(y, smooth_params, m0, S0, C, R, ensemble_vars, diameters_var, x_var, y_var):
    # Construct state transition matrix
    diameter_s = smooth_params[0]
    com_s = smooth_params[1]
    A = np.asarray([
        [diameter_s, 0, 0],
        [0, com_s, 0],
        [0, 0, com_s]
    ])
    # cov_matrix
    Q = np.asarray([
        [diameters_var * (1 - (A[0, 0] ** 2)), 0, 0],
        [0, x_var * (1 - A[1, 1] ** 2), 0],
        [0, 0, y_var * (1 - (A[2, 2] ** 2))]
    ])
    # Run filtering and smoothing with the current smooth_param
    mf, Vf, S, innovs, innov_cov = forward_pass(y, m0, S0, C, R, A, Q, ensemble_vars)
    ms, Vs, CV = backward_pass(y, mf, Vf, S, A)
    # Compute the negative log-likelihood based on innovations and their covariance
    nll, nll_values = compute_nll(innovs, innov_cov)
    return ms, Vs, nll, nll_values


def pupil_smooth_min(smooth_params, y, m0, S0, C, R, ensemble_vars, diameters_var, x_var, y_var):
    # Construct As
    diameter_s, com_s = smooth_params[0], smooth_params[1]
    A = np.array([
        [diameter_s, 0, 0],
        [0, com_s, 0],
        [0, 0, com_s]
    ])

    # Construct cov_matrix Q
    Q = np.array([
        [diameters_var * (1 - (A[0, 0] ** 2)), 0, 0],
        [0, x_var * (1 - A[1, 1] ** 2), 0],
        [0, 0, y_var * (1 - (A[2, 2] ** 2))]
    ])

    # Run filtering with the current smooth_param
    mf, Vf, S, innovs, innov_cov = forward_pass(y, m0, S0, C, R, A, Q, ensemble_vars)

    # Compute the negative log-likelihood
    nll, nll_values = compute_nll(innovs, innov_cov)

    return nll
