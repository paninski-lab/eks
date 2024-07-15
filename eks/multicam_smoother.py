import numpy as np
import pandas as pd
from scipy.optimize import minimize

from eks.core import ensemble, eks_zscore, compute_initial_guesses, forward_pass, backward_pass, \
    compute_nll
from eks.ibl_paw_multiview_smoother import remove_camera_means, pca
from eks.utils import make_dlc_pandas_index, crop_frames


def ensemble_kalman_smoother_multicam(
    markers_list_cameras, keypoint_ensemble, smooth_param, quantile_keep_pca, camera_names,
        s_frames, ensembling_mode='median', zscore_threshold=2):
    """Use multi-view constraints to fit a 3d latent subspace for each body part.

    Parameters
    ----------
    markers_list_cameras : list of list of pd.DataFrames
        each list element is a list of dataframe predictions from one ensemble member for each
        camera.
    keypoint_ensemble : str
        the name of the keypoint to be ensembled and smoothed
    smooth_param : float
        ranges from .01-2 (smaller values = more smoothing)
    quantile_keep_pca
        percentage of the points are kept for multi-view PCA (lowest ensemble variance)
    camera_names: list
        the camera names (should be the same length as markers_list_cameras).
    s_frames : list of tuples or int
        specifies frames to be used for smoothing parameter auto-tuning
        the function used for ensembling ('mean', 'median', or 'confidence_weighted_mean')
    zscore_threshold:
        Minimum std threshold to reduce the effect of low ensemble std on a zscore metric
        (default 2).

    Returns
    -------
    dict
        camera_dfs: dataframe containing smoothed markers for each camera; same format as input
        dataframes
    """

    # --------------------------------------------------------------
    # interpolate right cam markers to left cam timestamps
    # --------------------------------------------------------------
    num_cameras = len(camera_names)
    markers_list_stacked_interp = []
    markers_list_interp = [[] for i in range(num_cameras)]
    camera_likelihoods_stacked = []
    for model_id in range(len(markers_list_cameras[0])):
        bl_markers_curr = []
        camera_markers_curr = [[] for i in range(num_cameras)]
        camera_likelihoods = [[] for i in range(num_cameras)]
        for i in range(markers_list_cameras[0][0].shape[0]):
            curr_markers = []
            for camera in range(num_cameras):
                markers = np.array(markers_list_cameras[camera][model_id].to_numpy()[i, [0, 1]])
                likelihood = np.array(markers_list_cameras[camera][model_id].to_numpy()[i, [2]])[0]
                camera_markers_curr[camera].append(markers)
                curr_markers.append(markers)
                camera_likelihoods[camera].append(likelihood)
            # combine predictions for all cameras
            bl_markers_curr.append(np.concatenate(curr_markers))
        markers_list_stacked_interp.append(bl_markers_curr)
        camera_likelihoods_stacked.append(camera_likelihoods)
        camera_likelihoods = np.asarray(camera_likelihoods)
        for camera in range(num_cameras):
            markers_list_interp[camera].append(camera_markers_curr[camera])
            camera_likelihoods[camera] = np.asarray(camera_likelihoods[camera])
    markers_list_stacked_interp = np.asarray(markers_list_stacked_interp)
    markers_list_interp = np.asarray(markers_list_interp)
    camera_likelihoods_stacked = np.asarray(camera_likelihoods_stacked)

    keys = [keypoint_ensemble + '_x', keypoint_ensemble + '_y']
    markers_list_cams = [[] for i in range(num_cameras)]
    for k in range(len(markers_list_interp[0])):
        for camera in range(num_cameras):
            markers_cam = pd.DataFrame(markers_list_interp[camera][k], columns=keys)
            markers_cam[f'{keypoint_ensemble}_likelihood'] = camera_likelihoods_stacked[k][camera]
            markers_list_cams[camera].append(markers_cam)
    # compute ensemble median for each camera
    cam_ensemble_preds = []
    cam_ensemble_vars = []
    cam_ensemble_stacks = []
    cam_keypoints_mean_dict = []
    cam_keypoints_var_dict = []
    cam_keypoints_stack_dict = []
    for camera in range(num_cameras):
        cam_ensemble_preds_curr, cam_ensemble_vars_curr, cam_ensemble_stacks_curr, \
            cam_keypoints_mean_dict_curr, cam_keypoints_var_dict_curr, \
            cam_keypoints_stack_dict_curr = \
            ensemble(markers_list_cams[camera], keys, mode=ensembling_mode)
        cam_ensemble_preds.append(cam_ensemble_preds_curr)
        cam_ensemble_vars.append(cam_ensemble_vars_curr)
        cam_ensemble_stacks.append(cam_ensemble_stacks_curr)
        cam_keypoints_mean_dict.append(cam_keypoints_mean_dict_curr)
        cam_keypoints_var_dict.append(cam_keypoints_var_dict_curr)
        cam_keypoints_stack_dict.append(cam_keypoints_stack_dict_curr)

    # filter by low ensemble variances
    hstacked_vars = np.hstack(cam_ensemble_vars)
    max_vars = np.max(hstacked_vars, 1)
    quantile_keep = quantile_keep_pca
    good_frames = np.where(max_vars <= np.percentile(max_vars, quantile_keep))[0]

    good_cam_ensemble_preds = []
    good_cam_ensemble_vars = []
    for camera in range(num_cameras):
        good_cam_ensemble_preds.append(cam_ensemble_preds[camera][good_frames])
        good_cam_ensemble_vars.append(cam_ensemble_vars[camera][good_frames])

    good_ensemble_preds = np.hstack(good_cam_ensemble_preds)
    # good_ensemble_vars = np.hstack(good_cam_ensemble_vars)
    means_camera = []
    for i in range(good_ensemble_preds.shape[1]):
        means_camera.append(good_ensemble_preds[:, i].mean())

    ensemble_preds = np.hstack(cam_ensemble_preds)
    ensemble_vars = np.hstack(cam_ensemble_vars)
    ensemble_stacks = np.concatenate(cam_ensemble_stacks, 2)
    remove_camera_means(ensemble_stacks, means_camera)

    good_scaled_ensemble_preds = remove_camera_means(
        good_ensemble_preds[None, :, :], means_camera)[0]
    ensemble_pca, ensemble_ex_var = pca(
        good_scaled_ensemble_preds, 3)

    scaled_ensemble_preds = remove_camera_means(ensemble_preds[None, :, :], means_camera)[0]
    ensemble_pcs = ensemble_pca.transform(scaled_ensemble_preds)
    good_ensemble_pcs = ensemble_pcs[good_frames]

    y_obs = scaled_ensemble_preds

    # compute center of mass
    # latent variables (observed)
    good_z_t_obs = good_ensemble_pcs  # latent variables - true 3D pca

    # ------ Set values for kalman filter ------
    m0 = np.asarray([0.0, 0.0, 0.0])  # initial state: mean
    S0 = np.asarray([[np.var(good_z_t_obs[:, 0]), 0.0, 0.0],
                     [0.0, np.var(good_z_t_obs[:, 1]), 0.0],
                     [0.0, 0.0, np.var(good_z_t_obs[:, 2])]])  # diagonal: var

    A = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])  # state-transition matrix,

    # Q = np.asarray([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]) <-- state-cov matrix?

    d_t = good_z_t_obs[1:] - good_z_t_obs[:-1]

    C = ensemble_pca.components_.T  # Measurement function is inverse transform of PCA
    R = np.eye(ensemble_pca.components_.shape[1])  # placeholder diagonal matrix for ensemble var

    cov_matrix = np.cov(d_t.T)

    # Call functions from ensemble_kalman to optimize smooth_param before filtering and smoothing
    smooth_param, ms, Vs, nll, nll_values = multicam_optimize_smooth(
        cov_matrix, y_obs, m0, S0, C, A, R, ensemble_vars, s_frames, smooth_param)
    print(f"NLL is {nll} for {keypoint_ensemble}, smooth_param={smooth_param}")
    smooth_param_final = smooth_param

    # Smoothed posterior over ys
    y_m_smooth = np.dot(C, ms.T).T
    y_v_smooth = np.swapaxes(np.dot(C, np.dot(Vs, C.T)), 0, 1)

    # --------------------------------------
    # final cleanup
    # --------------------------------------
    pdindex = make_dlc_pandas_index([keypoint_ensemble],
                                    labels=["x", "y", "likelihood", "x_var", "y_var", "zscore"])
    camera_indices = []
    for camera in range(num_cameras):
        camera_indices.append([camera * 2, camera * 2 + 1])
    camera_dfs = {}
    for camera, camera_name in enumerate(camera_names):
        var = np.empty(y_m_smooth.T[camera_indices[camera][0]].shape)
        var[:] = np.nan
        eks_pred_x = \
            y_m_smooth.T[camera_indices[camera][0]] + means_camera[camera_indices[camera][0]]
        eks_pred_y = \
            y_m_smooth.T[camera_indices[camera][1]] + means_camera[camera_indices[camera][1]]
        # compute zscore for EKS to see how it deviates from the ensemble
        eks_predictions = np.asarray([eks_pred_x, eks_pred_y]).T
        zscore = eks_zscore(eks_predictions, cam_ensemble_preds[camera], cam_ensemble_vars[camera],
                            min_ensemble_std=zscore_threshold)
        pred_arr = np.vstack([
            eks_pred_x,
            eks_pred_y,
            var,
            y_v_smooth[:, camera_indices[camera][0], camera_indices[camera][0]],
            y_v_smooth[:, camera_indices[camera][1], camera_indices[camera][1]],
            zscore,
        ]).T
        camera_dfs[camera_name + '_df'] = pd.DataFrame(pred_arr, columns=pdindex)
    return camera_dfs, smooth_param_final, nll_values


def multicam_optimize_smooth(
        cov_matrix, y, m0, s0, C, A, R, ensemble_vars,
        s_frames=[(None, None)],
        smooth_param=None):
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
        print(f'Optimal at s={smooth_param}')

    # Final smooth with optimized s
    ms, Vs, nll, nll_values = multicam_smooth_final(
        cov_matrix, smooth_param, y, m0, s0, C, A, R, ensemble_vars)

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
