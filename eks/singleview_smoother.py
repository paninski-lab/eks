import numpy as np
import pandas as pd
from eks.utils import make_dlc_pandas_index
from eks.core import eks_zscore, vectorized_ensemble
from eks.autotune_smooth_param import vectorized_singlecam_multicam_optimize_and_smooth


# -----------------------
# Smoother function for single-view
# -----------------------


def vectorized_ensemble_kalman_smoother_single_view(
        markers_3d_array, bodypart_list, smooth_param, s_frames, ensembling_mode='median',
        zscore_threshold=2, verbose=False):
    """Vectorized single view ensemble Kalman smoother for multiple keypoints."""
    n_keypoints = markers_3d_array.shape[2] // 3

    # Compute ensemble statistics
    ensemble_preds, ensemble_vars, keypoints_avg_dict = vectorized_ensemble(
        markers_3d_array, mode=ensembling_mode)
    # Calculate mean and adjusted observations directly from keypoints_avg_dict
    mean_obs_dict = {}
    adjusted_obs_dict = {}
    scaled_ensemble_preds = ensemble_preds.copy()

    for i in range(n_keypoints):
        x_key = 3 * i
        y_key = 3 * i + 1

        mean_x_obs = np.nanmean(keypoints_avg_dict[x_key])
        mean_y_obs = np.nanmean(keypoints_avg_dict[y_key])

        adjusted_x_obs = keypoints_avg_dict[x_key] - mean_x_obs
        adjusted_y_obs = keypoints_avg_dict[y_key] - mean_y_obs

        mean_obs_dict[x_key] = mean_x_obs
        mean_obs_dict[y_key] = mean_y_obs
        adjusted_obs_dict[x_key] = adjusted_x_obs
        adjusted_obs_dict[y_key] = adjusted_y_obs

        # Scale ensemble predictions
        scaled_ensemble_preds[:, i, 0] -= mean_x_obs
        scaled_ensemble_preds[:, i, 1] -= mean_y_obs
    # Initialize Kalman filter values for each keypoint
    n_keypoints = len(bodypart_list)
    T = scaled_ensemble_preds.shape[0]
    n_coords = scaled_ensemble_preds.shape[2]
    m0s = np.zeros((n_keypoints, n_coords))
    S0s = np.zeros((n_keypoints, n_coords, n_coords))
    As = np.zeros((n_keypoints, n_coords, n_coords))
    cov_mats = np.zeros((n_keypoints, n_coords, n_coords))
    Cs = np.zeros((n_keypoints, n_coords, n_coords))
    Rs = np.zeros((n_keypoints, n_coords, n_coords))
    y_obs_array = np.zeros((n_keypoints, T, n_coords))

    for i in range(n_keypoints):
        x_key = 3 * i
        y_key = 3 * i + 1

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

        m0s[i] = m0
        S0s[i] = S0
        As[i] = A
        cov_mats[i] = cov_matrix
        Cs[i] = C
        Rs[i] = R
        y_obs_array[i] = scaled_ensemble_preds[:, i, :]

    # Optimize smooth_param for each keypoint separately
    s_finals, ms_array, Vs_array, nll_array, nll_values = \
        vectorized_singlecam_multicam_optimize_and_smooth(
            cov_mats, y_obs_array, m0s, S0s, Cs, As, Rs,
            ensemble_vars,
            s_frames, smooth_param)

    y_m_smooths = np.zeros((n_keypoints, T, n_coords))
    y_v_smooths = np.zeros((n_keypoints, T, n_coords, n_coords))
    eks_preds_array = np.zeros(y_m_smooths.shape)
    dfs = []
    df_dicts = []

    for k in range(n_keypoints):
        print(f"NLL is {nll_array[k]} for {bodypart_list[k]}, smooth_param={s_finals[k]}")
        y_m_smooths[k] = np.dot(Cs[k], ms_array[k].T).T
        y_v_smooths[k] = np.swapaxes(np.dot(Cs[k], np.dot(Vs_array[k], Cs[k].T)), 0, 1)

        # Computing zscore
        eks_preds_array[k] = y_m_smooths[k].copy()
        eks_preds_array[k] = np.asarray([eks_preds_array[k].T[0] + mean_x_obs,
                                         eks_preds_array[k].T[1] + mean_y_obs]).T
        zscore = eks_zscore(eks_preds_array[k],
                            ensemble_preds[:, k, :],
                            ensemble_vars[:, k, :],
                            min_ensemble_std=zscore_threshold)
        # Final Cleanup
        pdindex = make_dlc_pandas_index(
            [bodypart_list[k]], labels=["x", "y", "likelihood", "x_var", "y_var", "zscore"])
        var = np.empty(y_m_smooths[k].T[0].shape)
        var[:] = np.nan
        pred_arr = np.vstack([
            y_m_smooths[k].T[0] + mean_x_obs,
            y_m_smooths[k].T[1] + mean_y_obs,
            var,
            y_v_smooths[k][:, 0, 0],
            y_v_smooths[k][:, 1, 1],
            zscore,
        ]).T
        df = pd.DataFrame(pred_arr, columns=pdindex)
        dfs.append(df)
        df_dicts.append({bodypart_list[k] + '_df': df})
    return df_dicts, s_finals, nll_values
