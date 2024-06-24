import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.decomposition import PCA

from eks.core import backward_pass, eks_zscore, ensemble, forward_pass
from eks.utils import make_dlc_pandas_index


def remove_camera_means(ensemble_stacks, camera_means):
    scaled_ensemble_stacks = ensemble_stacks.copy()
    for k in range(len(ensemble_stacks)):
        for camera_id, camera_mean in enumerate(camera_means):
            scaled_ensemble_stacks[k][:, camera_id] = \
                ensemble_stacks[k][:, camera_id] - camera_mean
    return scaled_ensemble_stacks


def add_camera_means(ensemble_stacks, camera_means):
    scaled_ensemble_stacks = ensemble_stacks.copy()
    for k in range(len(ensemble_stacks)):
        for camera_id, camera_mean in enumerate(camera_means):
            scaled_ensemble_stacks[k][:, camera_id] = \
                ensemble_stacks[k][:, camera_id] + camera_mean
    return scaled_ensemble_stacks


def pca(S, n_comps):
    pca_ = PCA(n_components=n_comps)
    return pca_.fit(S), pca_.explained_variance_ratio_


def ensemble_kalman_smoother_ibl_paw(
        markers_list_left_cam, markers_list_right_cam, timestamps_left_cam,
        timestamps_right_cam, keypoint_names, smooth_param, quantile_keep_pca,
        ensembling_mode='median',
        zscore_threshold=2, img_width=128):
    """
    --(IBL-specific)-
    -Use multi-view constraints to fit a 3d latent subspace for each body part with 2
    asynchronous cameras.

    Parameters
    ----------
    markers_list_left_cam : list of pd.DataFrames
        each list element is a dataframe of predictions from one ensemble member (left view)
    markers_list_right_cam : list of pd.DataFrames
        each list element is a dataframe of predictions from one ensemble member (right view)
    timestamps_left_cam : np.array
        same length as dfs in markers_list_left_cam
    timestamps_right_cam : np.array
        same length as dfs in markers_list_right_cam
    keypoint_names : list
        list of names in the order they appear in marker dfs
    smooth_param : float
        ranges from .01-2 (smaller values = more smoothing)
    quantile_keep_pca
        percentage of the points are kept for multi-view PCA (lowest ensemble variance)
    ensembling_mode:
        the function used for ensembling ('mean', 'median', or 'confidence_weighted_mean')
    zscore_threshold:
        Minimum std threshold to reduce the effect of low ensemble std on a zscore metric
        (default 2).
    img_width
        The width of the image being smoothed (128 default, IBL-specific).
    Returns
    -------

    Returns
    -------
    dict
        left: dataframe containing smoothed left paw markers; same format as input dataframes
        right: dataframe containing smoothed right paw markers; same format as input dataframes

    """

    # --------------------------------------------------------------
    # interpolate right cam markers to left cam timestamps
    # --------------------------------------------------------------
    markers_list_stacked_interp = []
    markers_list_interp = [[], []]
    for model_id in range(len(markers_list_left_cam)):
        bl_markers_curr = []
        left_markers_curr = []
        right_markers_curr = []
        bl_left_np = markers_list_left_cam[model_id].to_numpy()
        bl_right_np = markers_list_right_cam[model_id].to_numpy()
        bl_right_interp = []
        for i in range(bl_left_np.shape[1]):
            bl_right_interp.append(interp1d(timestamps_right_cam, bl_right_np[:, i]))
        for i, ts in enumerate(timestamps_left_cam):
            if ts > timestamps_right_cam[-1]:
                break
            if ts < timestamps_right_cam[0]:
                continue
            left_markers = np.array(bl_left_np[i, [0, 1, 3, 4]])
            left_markers_curr.append(left_markers)
            right_markers = np.array([bl_right_interp[j](ts) for j in [0, 1, 3, 4]])
            right_markers[0] = img_width - right_markers[0]  # flip points to match left camera
            right_markers[2] = img_width - right_markers[2]  # flip points to match left camera
            right_markers_curr.append(right_markers)
            # combine paw 1 predictions for both cameras
            bl_markers_curr.append(np.concatenate((left_markers[:2], right_markers[:2])))
            # combine paw 2 predictions for both cameras
            bl_markers_curr.append(np.concatenate((left_markers[2:4], right_markers[2:4])))
        markers_list_stacked_interp.append(bl_markers_curr)
        markers_list_interp[0].append(left_markers_curr)
        markers_list_interp[1].append(right_markers_curr)
    # markers_list_stacked_interp = np.asarray(markers_list_stacked_interp)
    markers_list_interp = np.asarray(markers_list_interp)

    keys = ['paw_l_x', 'paw_l_y', 'paw_r_x', 'paw_r_y']
    markers_list_left_cam = []
    for k in range(len(markers_list_interp[0])):
        markers_left_cam = pd.DataFrame(markers_list_interp[0][k], columns=keys)
        markers_list_left_cam.append(markers_left_cam)

    markers_list_right_cam = []
    for k in range(len(markers_list_interp[1])):
        markers_right_cam = pd.DataFrame(markers_list_interp[1][k], columns=keys)
        markers_list_right_cam.append(markers_right_cam)

    # compute ensemble median left camera
    left_cam_ensemble_preds, left_cam_ensemble_vars, left_cam_ensemble_stacks, \
        left_cam_keypoints_mean_dict, left_cam_keypoints_var_dict, \
        left_cam_keypoints_stack_dict = \
        ensemble(markers_list_left_cam, keys, mode=ensembling_mode)

    # compute ensemble median right camera
    right_cam_ensemble_preds, right_cam_ensemble_vars, right_cam_ensemble_stacks, \
        right_cam_keypoints_mean_dict, right_cam_keypoints_var_dict, \
        right_cam_keypoints_stack_dict = \
        ensemble(markers_list_right_cam, keys, mode=ensembling_mode)

    # keep percentage of the points for multi-view PCA based lowest ensemble variance
    hstacked_vars = np.hstack((left_cam_ensemble_vars, right_cam_ensemble_vars))
    max_vars = np.max(hstacked_vars, 1)
    good_frames = np.where(max_vars <= np.percentile(max_vars, quantile_keep_pca))[0]

    good_left_cam_ensemble_preds = left_cam_ensemble_preds[good_frames]
    good_right_cam_ensemble_preds = right_cam_ensemble_preds[good_frames]
    good_left_cam_ensemble_vars = left_cam_ensemble_vars[good_frames]
    good_right_cam_ensemble_vars = right_cam_ensemble_vars[good_frames]

    # stack left and right camera predictions and variances for both paws on all the good
    # frames
    good_stacked_ensemble_preds = np.zeros(
        (good_frames.shape[0] * 2, left_cam_ensemble_preds.shape[1]))
    good_stacked_ensemble_vars = np.zeros(
        (good_frames.shape[0] * 2, left_cam_ensemble_vars.shape[1]))
    i, j = 0, 0
    while i < good_right_cam_ensemble_preds.shape[0]:
        good_stacked_ensemble_preds[j] = np.concatenate(
            (good_left_cam_ensemble_preds[i][:2], good_right_cam_ensemble_preds[i][:2]))
        good_stacked_ensemble_vars[j] = np.concatenate(
            (good_left_cam_ensemble_vars[i][:2], good_right_cam_ensemble_vars[i][:2]))
        j += 1
        good_stacked_ensemble_preds[j] = np.concatenate(
            (good_left_cam_ensemble_preds[i][2:4], good_right_cam_ensemble_preds[i][2:4]))
        good_stacked_ensemble_vars[j] = np.concatenate(
            (good_left_cam_ensemble_vars[i][2:4], good_right_cam_ensemble_vars[i][2:4]))
        i += 1
        j += 1

    # combine left and right camera predictions and variances for both paws
    left_paw_ensemble_preds = np.zeros(
        (left_cam_ensemble_preds.shape[0], left_cam_ensemble_preds.shape[1]))
    right_paw_ensemble_preds = np.zeros(
        (right_cam_ensemble_preds.shape[0], right_cam_ensemble_preds.shape[1]))
    left_paw_ensemble_vars = np.zeros(
        (left_cam_ensemble_vars.shape[0], left_cam_ensemble_vars.shape[1]))
    right_paw_ensemble_vars = np.zeros(
        (right_cam_ensemble_vars.shape[0], right_cam_ensemble_vars.shape[1]))
    for i in range(len(left_cam_ensemble_preds)):
        left_paw_ensemble_preds[i] = np.concatenate(
            (left_cam_ensemble_preds[i][:2], right_cam_ensemble_preds[i][:2]))
        right_paw_ensemble_preds[i] = np.concatenate(
            (left_cam_ensemble_preds[i][2:4], right_cam_ensemble_preds[i][2:4]))
        left_paw_ensemble_vars[i] = np.concatenate(
            (left_cam_ensemble_vars[i][:2], right_cam_ensemble_vars[i][:2]))
        right_paw_ensemble_vars[i] = np.concatenate(
            (left_cam_ensemble_vars[i][2:4], right_cam_ensemble_vars[i][2:4]))

    # get mean of each paw
    mean_camera_l_x = good_stacked_ensemble_preds[:, 0].mean()
    mean_camera_l_y = good_stacked_ensemble_preds[:, 1].mean()
    mean_camera_r_x = good_stacked_ensemble_preds[:, 2].mean()
    mean_camera_r_y = good_stacked_ensemble_preds[:, 3].mean()
    means_camera = [mean_camera_l_x, mean_camera_l_y, mean_camera_r_x, mean_camera_r_y]

    left_paw_ensemble_stacks = np.concatenate(
        (left_cam_ensemble_stacks[:, :, :2], right_cam_ensemble_stacks[:, :, :2]), 2)

    remove_camera_means(left_paw_ensemble_stacks, means_camera)

    right_paw_ensemble_stacks = np.concatenate(
        (right_cam_ensemble_stacks[:, :, :2], right_cam_ensemble_stacks[:, :, :2]), 2)

    remove_camera_means(right_paw_ensemble_stacks, means_camera)

    good_scaled_stacked_ensemble_preds = \
        remove_camera_means(good_stacked_ensemble_preds[None, :, :], means_camera)[0]
    ensemble_pca, ensemble_ex_var = pca(good_scaled_stacked_ensemble_preds, 3)

    scaled_left_paw_ensemble_preds = \
        remove_camera_means(left_paw_ensemble_preds[None, :, :], means_camera)[0]
    ensemble_pcs_left_paw = ensemble_pca.transform(scaled_left_paw_ensemble_preds)
    good_ensemble_pcs_left_paw = ensemble_pcs_left_paw[good_frames]

    scaled_right_paw_ensemble_preds = \
        remove_camera_means(right_paw_ensemble_preds[None, :, :], means_camera)[0]
    ensemble_pcs_right_paw = ensemble_pca.transform(scaled_right_paw_ensemble_preds)
    good_ensemble_pcs_right_paw = ensemble_pcs_right_paw[good_frames]

    # --------------------------------------------------------------
    # kalman filtering + smoothing
    # --------------------------------------------------------------
    # $z_t = (d_t, x_t, y_t)$
    # $z_t = As z_{t-1} + e_t, e_t ~ N(0,E)$
    # $O_t = B z_t + n_t, n_t ~ N(0,D_t)$

    dfs = {}
    for paw in ['left', 'right']:

        # --------------------------------------
        # Set values for kalman filter
        # --------------------------------------
        if paw == 'left':
            save_keypoint_name = keypoint_names[0]
            good_ensemble_pcs = good_ensemble_pcs_left_paw
            ensemble_vars = left_paw_ensemble_vars
            y = scaled_left_paw_ensemble_preds
            # ensemble_stacks = scaled_left_paw_ensemble_stacks
        else:
            save_keypoint_name = keypoint_names[1]
            good_ensemble_pcs = good_ensemble_pcs_right_paw
            ensemble_vars = right_paw_ensemble_vars
            y = scaled_right_paw_ensemble_preds
            # ensemble_stacks = scaled_right_paw_ensemble_stacks

        # compute center of mass
        # latent variables (observed)
        good_z_t_obs = good_ensemble_pcs  # latent variables - true 3D pca

        # Set values for kalman filter #
        # initial state: mean
        m0 = np.asarray([0.0, 0.0, 0.0])

        # diagonal: var
        S0 = np.asarray([
            [np.nanvar(good_z_t_obs[:, 0]), 0.0, 0.0],
            [0.0, np.nanvar(good_z_t_obs[:, 1]), 0.0],
            [0.0, 0.0, np.nanvar(good_z_t_obs[:, 2])]
        ])

        # state-transition matrix
        A = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

        # state covariance matrix
        d_t = good_z_t_obs[1:] - good_z_t_obs[:-1]
        Q = smooth_param * np.cov(d_t.T)

        # measurement function is inverse transform of PCA
        C = ensemble_pca.components_.T

        # placeholder diagonal matrix for ensemble variance
        R = np.eye(ensemble_pca.components_.shape[1])

        # --------------------------------------
        # perform filtering
        # --------------------------------------
        # do filtering pass with time-varying ensemble variances
        print(f"filtering {paw} paw...")
        mf, Vf, S, _, _ = forward_pass(y, m0, S0, C, R, A, Q, ensemble_vars)
        print("done filtering")

        # --------------------------------------
        # perform smoothing
        # --------------------------------------
        # Do the smoothing step
        print(f"smoothing {paw} paw...")
        ms, Vs, _ = backward_pass(y, mf, Vf, S, A)
        print("done smoothing")
        # Smoothed posterior over ys
        y_m_smooth = np.dot(C, ms.T).T
        y_v_smooth = np.swapaxes(np.dot(C, np.dot(Vs, C.T)), 0, 1)

        # --------------------------------------
        # cleanup for this paw
        # --------------------------------------
        save_keypoint_names = ['l_cam_' + save_keypoint_name, 'r_cam_' + save_keypoint_name]
        pdindex = make_dlc_pandas_index(
            save_keypoint_names,
            labels=["x", "y", "likelihood", "x_var", "y_var", "zscore"]
        )

        scaled_y_m_smooth = add_camera_means(y_m_smooth[None, :, :], means_camera)[0]
        scaled_y = add_camera_means(y[None, :, :], means_camera)[0]
        pred_arr = []
        for i in range(len(save_keypoint_names)):
            pred_arr.append(scaled_y_m_smooth.T[0 + 2 * i])
            pred_arr.append(scaled_y_m_smooth.T[1 + 2 * i])
            var = np.empty(scaled_y_m_smooth.T[0 + 2 * i].shape)
            var[:] = np.nan
            pred_arr.append(var)
            x_var = y_v_smooth[:, 0 + 2 * i, 0 + 2 * i]
            y_var = y_v_smooth[:, 1 + 2 * i, 1 + 2 * i]
            pred_arr.append(x_var)
            pred_arr.append(y_var)
            ###
            eks_predictions = np.asarray([scaled_y_m_smooth.T[0 + 2 * i],
                                          scaled_y_m_smooth.T[1 + 2 * i]]).T
            ensemble_preds = scaled_y[:, 2 * i:2 * (i + 1)]
            ensemble_vars_curr = ensemble_vars[:, 2 * i:2 * (i + 1)]
            zscore = eks_zscore(eks_predictions, ensemble_preds, ensemble_vars_curr,
                                min_ensemble_std=4)
            pred_arr.append(zscore)
            ###
        pred_arr = np.asarray(pred_arr)
        dfs[paw] = pd.DataFrame(pred_arr.T, columns=pdindex)

    # --------------------------------------
    # final cleanup
    # --------------------------------------
    pdindex = make_dlc_pandas_index(keypoint_names,
                                    labels=["x", "y", "likelihood", "x_var", "y_var", "zscore"])

    # make left cam dataframe
    pred_arr = np.hstack([
        dfs['left'].loc[:, ('ensemble-kalman_tracker', 'l_cam_paw_l', 'x')].to_numpy()
        [:, None],
        dfs['left'].loc[:, ('ensemble-kalman_tracker', 'l_cam_paw_l', 'y')].to_numpy()
        [:, None],
        dfs['left'].loc[:, ('ensemble-kalman_tracker', 'l_cam_paw_l', 'likelihood')].to_numpy()
        [:, None],
        dfs['left'].loc[:, ('ensemble-kalman_tracker', 'l_cam_paw_l', 'x_var')].to_numpy()
        [:, None],
        dfs['left'].loc[:, ('ensemble-kalman_tracker', 'l_cam_paw_l', 'y_var')].to_numpy()
        [:, None],
        dfs['left'].loc[:, ('ensemble-kalman_tracker', 'l_cam_paw_l', 'zscore')].to_numpy()
        [:, None],
        dfs['right'].loc[:, ('ensemble-kalman_tracker', 'l_cam_paw_r', 'x')].to_numpy()[:, None],
        dfs['right'].loc[:, ('ensemble-kalman_tracker', 'l_cam_paw_r', 'y')].to_numpy()[:, None],
        dfs['right'].loc[:, ('ensemble-kalman_tracker', 'l_cam_paw_r', 'likelihood')].to_numpy()
        [:, None],
        dfs['right'].loc[:, ('ensemble-kalman_tracker', 'l_cam_paw_r', 'x_var')].to_numpy()
        [:, None],
        dfs['right'].loc[:, ('ensemble-kalman_tracker', 'l_cam_paw_r', 'y_var')].to_numpy()
        [:, None],
        dfs['right'].loc[:, ('ensemble-kalman_tracker', 'l_cam_paw_r', 'zscore')].to_numpy()
        [:, None],
    ])
    df_left = pd.DataFrame(pred_arr, columns=pdindex)

    # make right cam dataframe
    # note we swap left and right paws to match dlc/lp convention
    # note we flip the paws horizontally to match lp convention
    pred_arr = np.hstack([
        img_width - dfs['right'].loc[:, ('ensemble-kalman_tracker', 'r_cam_paw_r', 'x')].to_numpy()
        [:, None],
        dfs['right'].loc[:, ('ensemble-kalman_tracker', 'r_cam_paw_r', 'y')].to_numpy()
        [:, None],
        dfs['right'].loc[:, ('ensemble-kalman_tracker', 'r_cam_paw_r', 'likelihood')].to_numpy()
        [:, None],
        dfs['right'].loc[:, ('ensemble-kalman_tracker', 'r_cam_paw_r', 'x_var')].to_numpy()
        [:, None],
        dfs['right'].loc[:, ('ensemble-kalman_tracker', 'r_cam_paw_r', 'y_var')].to_numpy()
        [:, None],
        dfs['right'].loc[:, ('ensemble-kalman_tracker', 'r_cam_paw_r', 'zscore')].to_numpy()
        [:, None],
        img_width - dfs['left'].loc[:, ('ensemble-kalman_tracker', 'r_cam_paw_l', 'x')].to_numpy()
        [:, None],
        dfs['left'].loc[:, ('ensemble-kalman_tracker', 'r_cam_paw_l', 'y')].to_numpy()
        [:, None],
        dfs['left'].loc[:, ('ensemble-kalman_tracker', 'r_cam_paw_l', 'likelihood')].to_numpy()
        [:, None],
        dfs['left'].loc[:, ('ensemble-kalman_tracker', 'r_cam_paw_l', 'x_var')].to_numpy()
        [:, None],
        dfs['left'].loc[:, ('ensemble-kalman_tracker', 'r_cam_paw_l', 'y_var')].to_numpy()
        [:, None],
        dfs['left'].loc[:, ('ensemble-kalman_tracker', 'r_cam_paw_l', 'zscore')].to_numpy()
        [:, None],
    ])
    df_right = pd.DataFrame(pred_arr, columns=pdindex)

    return {'left_df': df_left, 'right_df': df_right}, \
        markers_list_left_cam, markers_list_right_cam
