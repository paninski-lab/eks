import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from eks.core import backward_pass, compute_nll, eks_zscore, ensemble, forward_pass
from eks.utils import crop_frames, make_dlc_pandas_index


# -----------------------
# funcs for kalman pupil
# -----------------------
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
    top, bottom, left, right = [np.vstack((dlc[f'pupil_{point}_r_x'], dlc[f'pupil_{point}_r_y']))
                                for point in ['top', 'bottom', 'left', 'right']]
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


def ensemble_kalman_smoother_ibl_pupil(
    markers_list,
    keypoint_names,
    tracker_name,
    smooth_params,
    s_frames,
    likelihood_default=np.nan,
    zscore_threshold=2,
):
    """

    Parameters
    ----------
    markers_list : list of pd.DataFrames
        each list element is a dataframe of predictions from one ensemble member
    keypoint_names: list
    tracker_name : str
        tracker name for constructing final dataframe
    smooth_params : [float, float]
        contains smoothing parameters for diameter and center of mass
    likelihood_default
        value to store in likelihood column; should be np.nan or int in [0, 1]
    zscore_threshold:
        Minimum std threshold to reduce the effect of low ensemble std on a zscore metric
        (default 2).

    Returns
    -------
    dict
        markers_df: dataframe containing smoothed markers; same format as input dataframes
        latents_df: dataframe containing 3d latents: pupil diameter and pupil center of mass

    """

    # compute ensemble median
    keys = ['pupil_top_r_x', 'pupil_top_r_y', 'pupil_bottom_r_x', 'pupil_bottom_r_y',
            'pupil_right_r_x', 'pupil_right_r_y', 'pupil_left_r_x', 'pupil_left_r_y']
    ensemble_preds, ensemble_vars, ensemble_stacks, keypoints_mean_dict, keypoints_var_dict, \
        keypoints_stack_dict = ensemble(markers_list, keys)
    # ## Set parameters
    # compute center of mass
    pupil_locations = get_pupil_location(keypoints_mean_dict)
    pupil_diameters = get_pupil_diameter(keypoints_mean_dict)
    diameters = []
    for i in range(len(markers_list)):
        keypoints_dict = keypoints_stack_dict[i]
        diameter = get_pupil_diameter(keypoints_dict)
        diameters.append(diameter)

    mean_x_obs = np.mean(pupil_locations[:, 0])
    mean_y_obs = np.mean(pupil_locations[:, 1])
    # make the mean zero
    x_t_obs, y_t_obs = pupil_locations[:, 0] - mean_x_obs, pupil_locations[:, 1] - mean_y_obs

    # latent variables (observed)
    # latent variables - diameter, com_x, com_y
    # z_t_obs = np.vstack((pupil_diameters, x_t_obs, y_t_obs))

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
    C = np.asarray(
        [[0, 1, 0], [-.5, 0, 1], [0, 1, 0], [.5, 0, 1], [.5, 1, 0], [0, 0, 1], [-.5, 1, 0],
         [0, 0, 1]])

    # placeholder diagonal matrix for ensemble variance
    R = np.eye(8)

    scaled_ensemble_preds = ensemble_preds.copy()
    scaled_ensemble_stacks = ensemble_stacks.copy()
    # subtract COM means from the ensemble predictions
    for i in range(ensemble_preds.shape[1]):
        if i % 2 == 0:
            scaled_ensemble_preds[:, i] -= mean_x_obs
        else:
            scaled_ensemble_preds[:, i] -= mean_y_obs
    # subtract COM means from all the predictions
    for i in range(ensemble_preds.shape[1]):
        if i % 2 == 0:
            scaled_ensemble_stacks[:, :, i] -= mean_x_obs
        else:
            scaled_ensemble_stacks[:, :, i] -= mean_y_obs
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
    # save out marker info
    pdindex = make_dlc_pandas_index(keypoint_names,
                                    labels=["x", "y", "likelihood", "x_var", "y_var", "zscore"])
    processed_arr_dict = add_mean_to_array(y_m_smooth, keys, mean_x_obs, mean_y_obs)
    key_pair_list = [['pupil_top_r_x', 'pupil_top_r_y'],
                     ['pupil_right_r_x', 'pupil_right_r_y'],
                     ['pupil_bottom_r_x', 'pupil_bottom_r_y'],
                     ['pupil_left_r_x', 'pupil_left_r_y']]
    ensemble_indices = [(0, 1), (4, 5), (2, 3), (6, 7)]
    pred_arr = []
    for i, key_pair in enumerate(key_pair_list):
        pred_arr.append(processed_arr_dict[key_pair[0]])
        pred_arr.append(processed_arr_dict[key_pair[1]])
        var = np.empty(processed_arr_dict[key_pair[0]].shape)
        var[:] = likelihood_default
        pred_arr.append(var)
        x_var = y_v_smooth[:, i, i]
        y_var = y_v_smooth[:, i + 1, i + 1]
        pred_arr.append(x_var)
        pred_arr.append(y_var)
        # compute zscore for EKS to see how it deviates from the ensemble
        eks_predictions = \
            np.asarray([processed_arr_dict[key_pair[0]], processed_arr_dict[key_pair[1]]]).T
        ensemble_preds_curr = ensemble_preds[:, ensemble_indices[i][0]: ensemble_indices[i][1] + 1]
        ensemble_vars_curr = ensemble_vars[:, ensemble_indices[i][0]: ensemble_indices[i][1] + 1]
        zscore = eks_zscore(eks_predictions, ensemble_preds_curr, ensemble_vars_curr,
                            min_ensemble_std=zscore_threshold)
        pred_arr.append(zscore)

    pred_arr = np.asarray(pred_arr)
    markers_df = pd.DataFrame(pred_arr.T, columns=pdindex)
    # save out latents info: pupil diam, center of mass
    pred_arr2 = []
    pred_arr2.append(ms[:, 0])
    pred_arr2.append(ms[:, 1] + mean_x_obs)  # add back x mean of pupil location
    pred_arr2.append(ms[:, 2] + mean_y_obs)  # add back ys mean of pupil location
    pred_arr2 = np.asarray(pred_arr2)
    arrays = [[tracker_name, tracker_name, tracker_name], ['diameter', 'com_x', 'com_y']]
    pd_index2 = pd.MultiIndex.from_arrays(arrays, names=('scorer', 'latent'))
    latents_df = pd.DataFrame(pred_arr2.T, columns=pd_index2)

    return {'markers_df': markers_df, 'latents_df': latents_df}, smooth_params, nll_values


def pupil_optimize_smooth(
        y, m0, S0, C, R, ensemble_vars, diameters_var, x_var, y_var,
        s_frames=[(1, 2000)],
        smooth_params=[None, None]):
    """Optimize-and-smooth function for the pupil example script."""
    # Optimize smooth_param
    if smooth_params[0] is None or smooth_params[1] is None:

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
