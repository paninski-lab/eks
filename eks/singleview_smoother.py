import numpy as np
import pandas as pd
import jax.numpy as jnp
from jax import random, jit
from eks.utils import make_dlc_pandas_index
from eks.core import ensemble, eks_zscore, vectorized_ensemble
from eks.autosmooth import singlecam_multicam_optimize_and_smooth, vectorized_singlecam_multicam_optimize_and_smooth


# -----------------------
# funcs for single-view
# -----------------------
def ensemble_kalman_smoother_single_view(
        markers_list, keypoint_ensemble, smooth_param, s_frames, ensembling_mode='median',
        zscore_threshold=2, verbose=False):
    """ Use an identity observation matrix and smoothes by adjusting the smoothing parameter in the
    state-covariance matrix.

    Parameters
    ----------
    markers_list : list of list of pd.DataFrames
        each list element is a list of dataframe predictions from one ensemble member.
    keypoint_ensemble : str
        the name of the keypoint to be ensembled and smoothed
    smooth_param : float
        ranges from .01-20 (smaller values = more smoothing)
    s_frames : list of tuples or int
        specifies frames to be used for smoothing parameter auto-tuning
    ensembling_mode:
        the function used for ensembling ('mean', 'median', or 'confidence_weighted_mean')
    zscore_threshold:
        Minimum std threshold to reduce the effect of low ensemble std on a zscore metric
        (default 2).
    verbose: bool
        If True, progress will be printed for the user.
    Returns
    -------

    Returns
    -------
    dict
        keypoint_df: dataframe containing smoothed markers for one keypoint; same format as input
        dataframes
    smooth_param_final
        the optimized smooth param (or the user-input)
    nll_values
        the negative log likelihoods (EKS likelihoods) for plotting
    """

    # --------------------------------------------------------------
    # interpolate right cam markers to left cam timestamps
    # --------------------------------------------------------------
    keys = [keypoint_ensemble + '_x', keypoint_ensemble + '_y']
    x_key = keys[0]
    y_key = keys[1]
    # compute ensemble median
    ensemble_preds, ensemble_vars, keypoints_mean_dict = \
        ensemble(markers_list, keys, mode=ensembling_mode)

    mean_x_obs = np.nanmean(keypoints_mean_dict[x_key])
    mean_y_obs = np.nanmean(keypoints_mean_dict[y_key])
    x_t_obs, y_t_obs = \
        keypoints_mean_dict[x_key] - mean_x_obs, keypoints_mean_dict[y_key] - mean_y_obs

    # ------ Set values for kalman filter ------
    m0 = np.asarray([0.0, 0.0])  # initial state: mean
    S0 = np.asarray([[np.nanvar(x_t_obs), 0.0], [0.0 , np.nanvar(y_t_obs)]])  # diagonal: var

    A = np.asarray([[1.0, 0], [0, 1.0]])  # state-transition matrix,
    cov_matrix = np.asarray([[1, 0], [0, 1]])  # state covariance matrix; smaller = more smoothing
    C = np.asarray([[1, 0], [0, 1]])  # Measurement function
    R = np.eye(2)  # placeholder diagonal matrix for ensemble variance

    scaled_ensemble_preds = ensemble_preds.copy()
    scaled_ensemble_preds[:, 0] -= mean_x_obs
    scaled_ensemble_preds[:, 1] -= mean_y_obs

    y_obs = scaled_ensemble_preds

    # Optimize smooth_param before filtering and smoothing
    smooth_param_final, ms, Vs, nll, nll_values = \
        singlecam_multicam_optimize_and_smooth(
            cov_matrix, y_obs, m0, S0, C, A, R, ensemble_vars, s_frames, smooth_param)

    print(f"NLL is {nll} for {keypoint_ensemble}, smooth_param={smooth_param_final}")

    # Smoothed posterior over y
    y_m_smooth = np.dot(C, ms.T).T
    y_v_smooth = np.swapaxes(np.dot(C, np.dot(Vs, C.T)), 0, 1)

    # compute zscore for EKS to see how it deviates from the ensemble
    eks_predictions = y_m_smooth.copy()
    eks_predictions = \
        np.asarray([eks_predictions.T[0] + mean_x_obs, eks_predictions.T[1] + mean_y_obs]).T
    zscore = \
        eks_zscore(eks_predictions, ensemble_preds, ensemble_vars,
                   min_ensemble_std=zscore_threshold)

    # --------------------------------------
    # final cleanup
    # --------------------------------------
    pdindex = make_dlc_pandas_index([keypoint_ensemble],
                                    labels=["x", "y", "likelihood", "x_var", "y_var", "zscore"])
    var = np.empty(y_m_smooth.T[0].shape)
    var[:] = np.nan
    pred_arr = np.vstack([
        y_m_smooth.T[0] + mean_x_obs,
        y_m_smooth.T[1] + mean_y_obs,
        var,
        y_v_smooth[:, 0, 0],
        y_v_smooth[:, 1, 1],
        zscore,
    ]).T
    df = pd.DataFrame(pred_arr, columns=pdindex)
    return {keypoint_ensemble + '_df': df}, smooth_param_final, nll_values


def vectorized_ensemble_kalman_smoother_single_view(
        markers_3d_array, keys, bodypart_list, smooth_param, s_frames, ensembling_mode='median',
        zscore_threshold=2, verbose=False):
    """Vectorized single view ensemble Kalman smoother for multiple keypoints."""

    # Compute ensemble statistics
    ensemble_preds, ensemble_vars, keypoints_avg_dict = vectorized_ensemble(
        markers_3d_array, keys, mode=ensembling_mode)

    # Calculate mean and adjusted observations directly from keypoints_avg_dict
    mean_obs_dict = {}
    adjusted_obs_dict = {}
    scaled_ensemble_preds = ensemble_preds.copy()

    for i in range(0, len(keys), 2):
        x_key = keys[i]
        y_key = keys[i + 1]

        mean_x_obs = np.nanmean(keypoints_avg_dict[x_key])
        mean_y_obs = np.nanmean(keypoints_avg_dict[y_key])

        adjusted_x_obs = keypoints_avg_dict[x_key] - mean_x_obs
        adjusted_y_obs = keypoints_avg_dict[y_key] - mean_y_obs

        mean_obs_dict[x_key] = mean_x_obs
        mean_obs_dict[y_key] = mean_y_obs
        adjusted_obs_dict[x_key] = adjusted_x_obs
        adjusted_obs_dict[y_key] = adjusted_y_obs

        # Scale ensemble predictions
        scaled_ensemble_preds[:, i // 2, 0] -= mean_x_obs
        scaled_ensemble_preds[:, i // 2, 1] -= mean_y_obs


    # Initialize Kalman filter values for each keypoint
    n_keypoints = len(bodypart_list)
    T = scaled_ensemble_preds.shape[0]
    n_coords = scaled_ensemble_preds.shape[2]
    m0_array = np.zeros((n_keypoints, n_coords))
    S0_array = np.zeros((n_keypoints, n_coords, n_coords))
    A_array = np.zeros((n_keypoints, n_coords, n_coords))
    cov_matrix_array = np.zeros((n_keypoints, n_coords, n_coords))
    C_array = np.zeros((n_keypoints, n_coords, n_coords))
    R_array = np.zeros((n_keypoints, n_coords, n_coords))
    y_obs_array = np.zeros((n_keypoints, T, n_coords))

    for i in range(n_keypoints):
        x_key = keys[2 * i]
        y_key = keys[2 * i + 1]

        adjusted_x_obs = adjusted_obs_dict[x_key]
        adjusted_y_obs = adjusted_obs_dict[y_key]

        m0 = np.asarray([0.0, 0.0])  # initial state: mean
        S0 = np.asarray(
            [[np.nanvar(adjusted_x_obs), 0.0], [0.0, np.nanvar(adjusted_y_obs)]])  # diagonal: var
        A = np.asarray([[1.0, 0], [0, 1.0]])  # state-transition matrix
        cov_matrix = np.asarray(
            [[1, 0], [0, 1]])  # state covariance matrix; smaller = more smoothing
        C = np.asarray([[1, 0], [0, 1]])  # Measurement function
        R = np.eye(2)  # placeholder diagonal matrix for ensemble variance

        m0_array[i] = m0
        S0_array[i] = S0
        A_array[i] = A
        cov_matrix_array[i] = cov_matrix
        C_array[i] = C
        R_array[i] = R
        y_obs_array[i] = scaled_ensemble_preds[:, i, :]

    # Optimize smooth_param
    s_finals, ms_array, Vs_array, nll_array, nll_values_array = \
        vectorized_singlecam_multicam_optimize_and_smooth(
            cov_matrix_array, y_obs_array, m0_array, S0_array, C_array, A_array, R_array,
            ensemble_vars,
            s_frames, smooth_param)

    y_m_smooth_array = np.zeros((n_keypoints, T, n_coords))
    y_v_smooth_array = np.zeros((n_keypoints, T, n_coords, n_coords))
    eks_predictions_array = np.zeros(y_m_smooth_array.shape)
    df_array = []
    df_dict_array = []

    for k in range(n_keypoints):
        print(f"NLL is {nll_array[k]} for {bodypart_list[k]}, smooth_param={smooth_param}")
        y_m_smooth_array[k] = np.dot(C_array[k], ms_array[k].T).T
        y_v_smooth_array[k] = np.swapaxes(np.dot(C_array[k], np.dot(Vs_array[k], C_array[k].T)), 0, 1)

        # Computing zscore
        eks_predictions_array[k] = y_m_smooth_array[k].copy()
        eks_predictions_array[k] = np.asarray([eks_predictions_array[k].T[0] + mean_x_obs,
                                               eks_predictions_array[k].T[1] + mean_y_obs]).T
        zscore = eks_zscore(eks_predictions_array[k],
                            ensemble_preds[:, k, :],
                            ensemble_vars[:, k, :],
                            min_ensemble_std=zscore_threshold)
        # Final Cleanup
        pdindex = make_dlc_pandas_index([bodypart_list[k]],
                                        labels=["x", "y", "likelihood", "x_var", "y_var", "zscore"])
        var = np.empty(y_m_smooth_array[k].T[0].shape)
        var[:] = np.nan
        pred_arr = np.vstack([
            y_m_smooth_array[k].T[0] + mean_x_obs,
            y_m_smooth_array[k].T[1] + mean_y_obs,
            var,
            y_v_smooth_array[k][:, 0, 0],
            y_v_smooth_array[k][:, 1, 1],
            zscore,
        ]).T
        df = pd.DataFrame(pred_arr, columns=pdindex)
        df_array.append(df)
        df_dict_array.append({bodypart_list[k] + '_df': df})
    return df_dict_array, s_finals, nll_values_array

'''
def jax_ensemble_kalman_smoother_single_view(keypoints_data, keypoint_ensemble, s_frames, ensembling_mode='median', zscore_threshold=2, verbose=False):
    """
    Adapted Kalman smoother to handle batch operations using JAX.
    """
    print(f"Processing for keypoint: {keypoint_ensemble}")
    print(f"KeyPoints Data Shape: {keypoints_data.shape}")

    # Check if data dimensionality might be causing the indexing issue
    if keypoints_data.ndim == 1:
        keypoints_data = keypoints_data.reshape(-1, 1)

    # Ensembling based on the specified mode
    if ensembling_mode == 'median':
        ensemble_preds = jnp.median(keypoints_data, axis=0)
    elif ensembling_mode == 'mean':
        ensemble_preds = jnp.mean(keypoints_data, axis=0)
    else:
        ensemble_preds = jnp.mean(keypoints_data, axis=0)  # Default to mean if other methods aren't implemented

    # Prevent indexing errors by checking dimensions
    if ensemble_preds.ndim < 2:
        ensemble_preds = ensemble_preds[:, None]  # Ensure at least two dimensions

    mean_x_obs = jnp.mean(ensemble_preds[:, 0]) if ensemble_preds.shape[1] > 0 else 0
    mean_y_obs = jnp.mean(ensemble_preds[:, 1]) if ensemble_preds.shape[1] > 1 else 0

    y_obs = jnp.column_stack((ensemble_preds[:, 0] - mean_x_obs, ensemble_preds[:, 1] - mean_y_obs)) if ensemble_preds.shape[1] > 1 else ensemble_preds

    # Kalman filter setup (placeholders for actual computation)
    m0 = jnp.zeros(2)
    S0 = jnp.array([[jnp.var(y_obs[:, 0]), 0], [0, jnp.var(y_obs[:, 1])]])
    A = jnp.eye(2)
    C = jnp.eye(2)
    R = jnp.eye(2)

    # Filtering and smoothing placeholders
    mf, Vf = y_obs, jnp.tile(jnp.eye(2), (y_obs.shape[0], 1, 1))
    ms, Vs = y_obs, jnp.tile(jnp.eye(2), (y_obs.shape[0], 1, 1))

    nll = -jnp.sum(jnp.log(jnp.linalg.det(Vs)))

    results = {
        'x': y_obs[:, 0] + mean_x_obs,
        'y': y_obs[:, 1] + mean_y_obs,
        'likelihood': jnp.ones_like(y_obs[:, 0])  # placeholder for likelihood values
    }
    smooth_param = 3.1415  # Example fixed value for smoothing parameter
    return results, smooth_param, [nll]
'''