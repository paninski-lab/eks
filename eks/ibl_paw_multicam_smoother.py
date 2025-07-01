import numpy as np
import os
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.decomposition import PCA
from typeguard import typechecked

from eks.core import backward_pass, eks_zscore, ensemble, forward_pass
from eks.marker_array import (
    MarkerArray,
    input_dfs_to_markerArray,
    mA_to_stacked_array,
    stacked_array_to_mA,
)
from eks.multicam_smoother import ensemble_kalman_smoother_multicam
from eks.stats import compute_pca
from eks.utils import make_dlc_pandas_index, convert_lp_dlc


def remove_camera_means(ensemble_stacks, camera_means):
    centered_ensemble_stacks = ensemble_stacks.copy()
    for k in range(len(ensemble_stacks)):
        for camera_id, camera_mean in enumerate(camera_means):
            centered_ensemble_stacks[k][:, camera_id] = \
                ensemble_stacks[k][:, camera_id] - camera_mean
    return centered_ensemble_stacks


def add_camera_means(ensemble_stacks, camera_means):
    centered_ensemble_stacks = ensemble_stacks.copy()
    for k in range(len(ensemble_stacks)):
        for camera_id, camera_mean in enumerate(camera_means):
            centered_ensemble_stacks[k][:, camera_id] = \
                ensemble_stacks[k][:, camera_id] + camera_mean
    return centered_ensemble_stacks


def pca(S, n_comps):
    pca_ = PCA(n_components=n_comps)
    return pca_.fit(S), pca_.explained_variance_ratio_

@typechecked
def fit_eks_multicam_ibl_paw(
    input_source: str | list,
    save_dir: str,
    smooth_param: float | list | None = None,
    s_frames: list | None = None,
    quantile_keep_pca: float = 95.0,
    avg_mode: str = 'median',
    var_mode: str = 'confidence_weighted_var',
    verbose: bool = False,
    img_width: int = 128,
) -> tuple:
    """
        Fit the Ensemble Kalman Smoother for IBL multi-camera paw data.

        Args:
            input_source: Directory path or list of CSV file paths with columns for all cameras.
            save_dir: Directory to save output DataFrame.
            smooth_param: Value in (0, Inf); smaller values lead to more smoothing.
            s_frames: Frames for automatic optimization if smooth_param is not provided.
            quantile_keep_pca: Percentage of points kept for PCA (default: 95).
            avg_mode: Mode for averaging across ensemble ('median', 'mean').
            var_mode: mode for computing ensemble variance
                'var' | 'confidence_weighted_var'
            verbose: True to print out details
            img_width: The width of the image being smoothed (128 default, IBL-specific).

        Returns:
                tuple:
                        camera_dfs (list): List of Output Dataframes
                        s_finals (list): List of optimized smoothing factors for each keypoint.
                        input_dfs (list): List of input DataFrames for plotting.
                        bodypart_list (list): List of body parts used.
        """
    # IBL paw smoother only works for a pre-specified set of points
    bodypart_list = ['paw_l', 'paw_r']
    camera_names = ["left", "right"]

    # load files and put them in correct format
    input_dfs_left = []
    input_dfs_right = []
    timestamps_left = None
    timestamps_right = None
    filenames = os.listdir(input_source)
    for filename in filenames:
        # Prediction files
        if 'timestamps' not in filename:
            input_df = pd.read_csv(
                os.path.join(input_source, filename), header=[0, 1, 2], index_col=0)
            input_df = convert_lp_dlc(input_df, bodypart_list)
            if 'left' in filename:
                input_dfs_left.append(input_df)
            else:
                # switch right camera paws
                columns = {
                    'paw_l_x': 'paw_r_x', 'paw_l_y': 'paw_r_y',
                    'paw_l_likelihood': 'paw_r_likelihood',
                    'paw_r_x': 'paw_l_x', 'paw_r_y': 'paw_l_y',
                    'paw_r_likelihood': 'paw_l_likelihood'
                }
                input_df = input_df.rename(columns=columns)
                # reorder columns
                input_df = input_df.loc[:, columns.keys()]
                input_dfs_right.append(input_df)
        # Timestamp files
        else:
            if 'left' in filename:
                timestamps_left = np.load(os.path.join(input_source, filename))
            else:
                timestamps_right = np.load(os.path.join(input_source, filename))

    # file checks
    if timestamps_left is None or timestamps_right is None:
        raise ValueError('Need timestamps for both cameras')
    if len(input_dfs_right) != len(input_dfs_left) or len(input_dfs_left) == 0:
        raise ValueError(
            'There must be the same number of left and right camera models and >=1 model for each.')
    input_dfs_list_original = [input_dfs_left, input_dfs_right]
    # Interpolate right cam markers to left cam timestamps
    markers_list_stacked_interp = []
    markers_list_interp = [[], []]
    for model_id in range(len(input_dfs_left)):
        bl_markers_curr = []
        left_markers_curr = []
        right_markers_curr = []
        bl_left_np = input_dfs_left[model_id].to_numpy()
        bl_right_np = input_dfs_right[model_id].to_numpy()
        bl_right_interp = []
        n_beg_nans = 0
        n_end_nans = 0
        for i in range(bl_left_np.shape[1]):
            bl_right_interp.append(interp1d(timestamps_right, bl_right_np[:, i]))
        for i, ts in enumerate(timestamps_left):
            if ts > timestamps_right[-1]:
                n_end_nans += 1
                continue
            if ts < timestamps_right[0]:
                n_beg_nans += 1
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

    markers_list_interp = np.asarray(markers_list_interp)

    # Add column names back into new dfs
    keys = ['paw_l_x', 'paw_l_y', 'paw_r_x', 'paw_r_y']
    input_dfs_list = [[] for _ in camera_names]
    for c, _ in enumerate(camera_names):
        for k in range(len(markers_list_interp[c])):
            input_df = pd.DataFrame(markers_list_interp[c][k], columns=keys)
            input_dfs_list[c].append(input_df)

    # Combine synced dfs into MarkerArray
    marker_array = input_dfs_to_markerArray(
        input_dfs_list, bodypart_list, camera_names, data_fields=["x", "y"])

    # Add likelihood data field to MarkerArray
    dummy_likelihood_shape = np.array(marker_array.shape)
    dummy_likelihood_shape[-1] = 1
    marker_array = MarkerArray.stack_fields(
        marker_array,
        MarkerArray(shape=dummy_likelihood_shape, data_fields=["likelihood"])
    )

    # run eks
    camera_dfs, smooth_params_final = ensemble_kalman_smoother_multicam(
        marker_array=marker_array,
        keypoint_names=bodypart_list,
        smooth_param=smooth_param,
        quantile_keep_pca=quantile_keep_pca,
        camera_names=camera_names,
        s_frames=s_frames,
        avg_mode=avg_mode,
        var_mode=var_mode,
        verbose=verbose
    )
    # Save output DataFrames to CSVs (one per camera view)
    os.makedirs(save_dir, exist_ok=True)
    for c, camera in enumerate(camera_names):
        save_filename = f'multicam_{camera}_results.csv'
        camera_dfs[c].to_csv(os.path.join(save_dir, save_filename))
    return camera_dfs, smooth_params_final, input_dfs_list, bodypart_list

@typechecked()
def ensemble_kalman_smoother_ibl_paw(
    marker_array: MarkerArray,
    keypoint_names: list,
    smooth_param: float | list | None = None,
    quantile_keep_pca: float = 95.0,
    camera_names: list | None = None,
    s_frames: list | None = None,
    avg_mode: str = 'median',
    var_mode: str = 'confidence_weighted_var',
    verbose: bool = False,
    img_width=128,
) -> tuple:
    """
    (IBL-specific)
    Use multi-view constraints to fit a 3d latent subspace for each body part with 2
    asynchronous cameras.

    Args:
        marker_array: MarkerArray object containing marker data for left and right views
            Shape (n_models, n_cameras=2, n_frames, n_keypoints, 3 (for x, y, likelihood)
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

    Returns
    -------
    dict
        left: dataframe containing smoothed left paw markers; same format as input dataframes
        right: dataframe containing smoothed right paw markers; same format as input dataframes

    """

    # compute ensemble median left camera
    left_cam_ensemble_preds, left_cam_ensemble_vars, _, left_cam_ensemble_stacks = ensemble(
        markers_list_left_cam, keys, avg_mode=ensembling_mode, var_mode='var',
    )

    # compute ensemble median right camera
    right_cam_ensemble_preds, right_cam_ensemble_vars, _, right_cam_ensemble_stacks = ensemble(
        markers_list_right_cam, keys, avg_mode=ensembling_mode, var_mode='var',
    )

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

    good_centered_stacked_ensemble_preds = \
        remove_camera_means(good_stacked_ensemble_preds[None, :, :], means_camera)[0]
    ensemble_pca, ensemble_ex_var = pca(good_centered_stacked_ensemble_preds, 3)

    centered_left_paw_ensemble_preds = \
        remove_camera_means(left_paw_ensemble_preds[None, :, :], means_camera)[0]
    ensemble_pcs_left_paw = ensemble_pca.transform(centered_left_paw_ensemble_preds)
    good_ensemble_pcs_left_paw = ensemble_pcs_left_paw[good_frames]

    centered_right_paw_ensemble_preds = \
        remove_camera_means(right_paw_ensemble_preds[None, :, :], means_camera)[0]
    ensemble_pcs_right_paw = ensemble_pca.transform(centered_right_paw_ensemble_preds)
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
            y = centered_left_paw_ensemble_preds
            # ensemble_stacks = centered_left_paw_ensemble_stacks
        else:
            save_keypoint_name = keypoint_names[1]
            good_ensemble_pcs = good_ensemble_pcs_right_paw
            ensemble_vars = right_paw_ensemble_vars
            y = centered_right_paw_ensemble_preds
            # ensemble_stacks = centered_right_paw_ensemble_stacks

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

        centered_y_m_smooth = add_camera_means(y_m_smooth[None, :, :], means_camera)[0]
        centered_y = add_camera_means(y[None, :, :], means_camera)[0]
        pred_arr = []
        for i in range(len(save_keypoint_names)):
            pred_arr.append(centered_y_m_smooth.T[0 + 2 * i])
            pred_arr.append(centered_y_m_smooth.T[1 + 2 * i])
            var = np.empty(centered_y_m_smooth.T[0 + 2 * i].shape)
            var[:] = np.nan
            pred_arr.append(var)
            x_var = y_v_smooth[:, 0 + 2 * i, 0 + 2 * i]
            y_var = y_v_smooth[:, 1 + 2 * i, 1 + 2 * i]
            pred_arr.append(x_var)
            pred_arr.append(y_var)
            ###
            eks_predictions = np.asarray([centered_y_m_smooth.T[0 + 2 * i],
                                          centered_y_m_smooth.T[1 + 2 * i]]).T
            ensemble_preds = centered_y[:, 2 * i:2 * (i + 1)]
            ensemble_vars_curr = ensemble_vars[:, 2 * i:2 * (i + 1)]
            zscore, _ = eks_zscore(eks_predictions, ensemble_preds, ensemble_vars_curr,
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

    # add nans to beginning/end so that the returned dataframes match the number of left timestamps
    if n_beg_nans > 0:
        nan_rows = pd.DataFrame(np.nan, index=range(-n_beg_nans, 0), columns=df_right.columns)
        df_right = pd.concat([nan_rows, df_right])
        df_left = pd.concat([nan_rows, df_left])
    if n_end_nans > 0:
        curr_len = df_right.shape[0] - n_beg_nans
        nan_rows = pd.DataFrame(
            np.nan, index=range(curr_len, curr_len + n_end_nans), columns=df_right.columns)
        df_right = pd.concat([df_right, nan_rows])
        df_left = pd.concat([df_left, nan_rows])

    return {'left_df': df_left, 'right_df': df_right}, \
        markers_list_left_cam, markers_list_right_cam
