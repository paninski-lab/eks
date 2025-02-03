import os

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typeguard import typechecked

from eks.core import backward_pass, compute_initial_guesses, compute_nll, ensemble, forward_pass, \
    jax_ensemble
from eks.ibl_paw_multiview_smoother import pca, remove_camera_means
from eks.stats import compute_mahalanobis
from eks.utils import crop_frames, format_data, make_dlc_pandas_index
from eks.marker_array import MarkerArray


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

    markers_list = []
    for keypoint_ensemble in bodypart_list:
        # Separate body part predictions by camera view
        marker_list_by_cam = [[] for _ in range(len(camera_names))]
        for markers_curr in input_dfs_list:
            for c, camera_name in enumerate(camera_names):
                non_likelihood_keys = [
                    key for key in markers_curr.keys()
                    if camera_names[c] in key and keypoint_ensemble in key
                ]
                marker_list_by_cam[c].append(markers_curr[non_likelihood_keys])
        markers_list.append(marker_list_by_cam)

    # Run the ensemble Kalman smoother for multi-camera data
    camera_dfs, smooth_params_final = ensemble_kalman_smoother_multicam(
        markers_list=markers_list,
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
        print(f'Input data loaded for keypoints:\n{bodypart_list}')

    markers_list = []
    for keypoint in bodypart_list:
        # Separate body part predictions by camera view
        markers_list_cameras = [[] for _ in range(len(camera_names))]
        for c, camera_name in enumerate(camera_names):
            ensemble_members = input_dfs_list[c]
            for markers_curr in ensemble_members:
                non_likelihood_keys = [
                    key for key in markers_curr.keys()
                    if keypoint in key
                ]
                markers_list_cameras[c].append(markers_curr[non_likelihood_keys])
        markers_list.append(markers_list_cameras)

    # Run the ensemble Kalman smoother for multi-camera data
    camera_dfs, smooth_params_final = jax_ensemble_kalman_smoother_multicam(
        markers_list=markers_list,
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
        save_filename = f'multicam_{camera}_results.csv'
        camera_dfs[c].to_csv(os.path.join(save_dir, save_filename))
    return camera_dfs, smooth_params_final, input_dfs_list, bodypart_list


@typechecked
def ensemble_kalman_smoother_multicam(
    markers_list: list,
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
        markers_list: List of lists of pd.DataFrames, where each inner list contains
            DataFrame predictions for a single camera.
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
        verbose: True to print out details

    Returns:
        tuple: Dataframes with smoothed predictions, final smoothing parameters.
    """

    # Collection array for marker output by camera view
    camera_arrs = [[] for _ in camera_names]
    # Loop over keypoints; apply EKS to each individually
    for k, keypoint in enumerate(keypoint_names):
        # Setup: Interpolate right cam markers to left cam timestamps
        markers_list_cameras = markers_list[k]
        num_cameras = len(camera_names)

        markers_list_stacked_interp = []
        markers_list_interp = [[] for _ in range(num_cameras)]
        camera_likelihoods_stacked = []

        for model_id in range(len(markers_list_cameras[0])):
            bl_markers_curr = []
            camera_markers_curr = [[] for _ in range(num_cameras)]
            camera_likelihoods = [[] for _ in range(num_cameras)]

            for i in range(markers_list_cameras[0][0].shape[0]):
                curr_markers = []

                for camera in range(num_cameras):
                    markers = np.array(
                        markers_list_cameras[camera][model_id].to_numpy()[i, [0, 1]]
                    )
                    likelihood = np.array(
                        markers_list_cameras[camera][model_id].to_numpy()[i, [2]]
                    )[0]

                    camera_markers_curr[camera].append(markers)
                    curr_markers.append(markers)
                    camera_likelihoods[camera].append(likelihood)

                # Combine predictions for all cameras
                bl_markers_curr.append(np.concatenate(curr_markers))
            markers_list_stacked_interp.append(bl_markers_curr)
            camera_likelihoods_stacked.append(camera_likelihoods)

            camera_likelihoods = np.asarray(camera_likelihoods)
            for camera in range(num_cameras):
                markers_list_interp[camera].append(camera_markers_curr[camera])
                camera_likelihoods[camera] = np.asarray(camera_likelihoods[camera])

        # Shape (n_models, n_frames, n_coords * n_cameras)
        markers_list_stacked_interp = np.asarray(markers_list_stacked_interp)

        # Shape (n_cameras, n_models, n_frames, n_coords)
        markers_list_interp = np.asarray(markers_list_interp)

        # Shape (n_models, n_cameras, n_frames)
        camera_likelihoods_stacked = np.asarray(camera_likelihoods_stacked)

        keys = [f"{keypoint}_x", f"{keypoint}_y"]
        markers_list_cams = [[] for _ in range(num_cameras)]

        for k in range(len(markers_list_interp[0])):
            for camera in range(num_cameras):
                markers_cam = pd.DataFrame(markers_list_interp[camera][k], columns=keys)
                markers_cam[f'{keypoint}_likelihood'] = camera_likelihoods_stacked[k][camera]
                markers_list_cams[camera].append(markers_cam)

        # Compute ensemble median for each camera
        cam_ensemble_preds = []
        cam_ensemble_vars = []
        cam_ensemble_likes = []
        cam_ensemble_stacks = []

        for camera in range(num_cameras):
            (
                cam_ensemble_preds_curr,
                cam_ensemble_vars_curr,
                cam_ensemble_likes_curr,
                cam_ensemble_stacks_curr,
            ) = ensemble(markers_list_cameras[camera], keys, avg_mode=avg_mode, var_mode=var_mode)

            cam_ensemble_preds.append(cam_ensemble_preds_curr)
            cam_ensemble_vars.append(cam_ensemble_vars_curr)
            cam_ensemble_likes.append(cam_ensemble_likes_curr)
            cam_ensemble_stacks.append(cam_ensemble_stacks_curr)
        print(cam_ensemble_stacks[0].shape)
        # Filter by low ensemble variances
        hstacked_vars = np.hstack(cam_ensemble_vars)
        max_vars = np.max(hstacked_vars, 1)
        good_frames = np.where(max_vars <= np.percentile(max_vars, quantile_keep_pca))[0]

        good_cam_ensemble_preds = [
            cam_ensemble_preds[camera][good_frames] for camera in range(num_cameras)
        ]

        good_ensemble_preds = np.hstack(good_cam_ensemble_preds)
        means_camera = [good_ensemble_preds[:, i].mean() for i in
                        range(good_ensemble_preds.shape[1])]

        ensemble_preds = np.hstack(cam_ensemble_preds)
        ensemble_vars = np.hstack(cam_ensemble_vars)
        ensemble_likes = np.hstack(cam_ensemble_likes)
        ensemble_stacks = np.concatenate(cam_ensemble_stacks, 2)
        remove_camera_means(ensemble_stacks, means_camera)
        good_scaled_ensemble_preds = remove_camera_means(
            good_ensemble_preds[None, :, :], means_camera
        )[0]

        ensemble_pca, ensemble_ex_var = pca(good_scaled_ensemble_preds, 3)
        scaled_ensemble_preds = remove_camera_means(
            ensemble_preds[None, :, :], means_camera
        )[0]

        ensemble_pcs = ensemble_pca.transform(scaled_ensemble_preds)
        good_ensemble_pcs = ensemble_pcs[good_frames]
        y_obs = scaled_ensemble_preds

        if inflate_vars:
            inflated = True
            tmp_vars = ensemble_vars
            while inflated:
                # Compute Mahalanobis distances
                maha_results = compute_mahalanobis(y_obs, tmp_vars, likelihoods=ensemble_likes)
                # Inflate variances based on Mahalanobis distances
                inflated_ens_vars, inflated = inflate_variance(
                    tmp_vars, maha_results['mahalanobis'], threshold=5, scalar=2,
                )
                tmp_vars = inflated_ens_vars
        else:
            inflated_ens_vars = ensemble_vars

        # Kalman Filter
        m0 = np.asarray([0.0, 0.0, 0.0])
        S0 = np.asarray([[np.var(good_ensemble_pcs[:, 0]), 0.0, 0.0],
                         [0.0, np.var(good_ensemble_pcs[:, 1]), 0.0],
                         [0.0, 0.0, np.var(good_ensemble_pcs[:, 2])]])  # diagonal: var
        A = np.eye(3)
        d_t = good_ensemble_pcs[1:] - good_ensemble_pcs[:-1]
        C = ensemble_pca.components_.T
        R = np.eye(ensemble_pca.components_.shape[1])
        cov_matrix = np.cov(d_t.T)
        smooth_param_final, ms, Vs, _, _ = multicam_optimize_smooth(
            cov_matrix, y_obs, m0, S0, C, A, R, inflated_ens_vars, s_frames, smooth_param
        )
        if verbose:
            print(f"Smoothed {keypoint} at smooth_param={smooth_param_final}")

        y_m_smooth = np.dot(C, ms.T).T
        y_v_smooth = np.swapaxes(np.dot(C, np.dot(Vs, C.T)), 0, 1)

        # Final cleanup
        c_i = [[camera * 2, camera * 2 + 1] for camera in range(num_cameras)]
        for c, camera in enumerate(camera_names):
            data_arr = camera_arrs[c]
            x_i, y_i = c_i[c]

            data_arr.extend([
                y_m_smooth.T[x_i] + means_camera[x_i],
                y_m_smooth.T[y_i] + means_camera[y_i],
                ensemble_likes[:, x_i],
                ensemble_preds[:, x_i],
                ensemble_preds[:, y_i],
                inflated_ens_vars[:, x_i],
                inflated_ens_vars[:, y_i],
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


@typechecked
def jax_ensemble_kalman_smoother_multicam(
        markers_list: list,
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
        markers_list: List of lists of pd.DataFrames, where each inner list contains
            DataFrame predictions for a single camera.
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
        verbose: True to print out details

    Returns:
        tuple: Dataframes with smoothed predictions, final smoothing parameters.
    """

    # Get the dimensions
    n_keypoints = len(markers_list)  # Number of keypoints
    n_cameras = len(markers_list[0])  # Number of cameras
    n_models = len(markers_list[0][0])  # Number of models

    # Infer number of frames and check consistency
    n_frames = markers_list[0][0][0].shape[
        0]  # Assuming all DataFrames have the same number of frames

    # Initialize array
    markers_array = np.zeros((n_models, n_cameras, n_frames, n_keypoints, 3))

    # Fill in the array
    for k, keypoint in enumerate(markers_list):  # Loop over keypoints
        for c, camera in enumerate(keypoint):  # Loop over cameras
            for m, model_df in enumerate(camera):  # Loop over models
                markers_array[m, c, :, k,
                :] = model_df.to_numpy()  # Convert DataFrame to NumPy and assign

    # Check the final shape
    print(markers_array.shape)  # Should be (n_models, n_cameras, n_frames, n_keypoints, 3)

    n_cameras = len(camera_names)  # remove later

    ensemble_preds = []
    ensemble_vars = []
    ensemble_likes = []
    ensemble_stacks = []
    for c, camera in enumerate(camera_names):
        (
            cam_ensemble_preds,
            cam_ensemble_vars,
            cam_ensemble_likes,
        ) = jax_ensemble(markers_array[:, c], avg_mode=avg_mode, var_mode=var_mode)

    # Going back to Numpy for PCA
    cam_ensemble_preds = np.array(cam_ensemble_preds)
    cam_ensemble_vars = np.array(cam_ensemble_vars)
    cam_ensemble_likes = np.array(cam_ensemble_likes)

    for k, keypoint in enumerate(keypoint_names):
        # Filter by low ensemble variances
        hstacked_vars = np.hstack(cam_ensemble_vars)
        max_vars = np.max(hstacked_vars, 1)
        good_frames = np.where(max_vars <= np.percentile(max_vars, quantile_keep_pca))[0]

        good_cam_ensemble_preds = [
            cam_ensemble_preds[k][camera][good_frames] for camera in range(n_cameras)
        ]

        good_ensemble_preds = np.hstack(good_cam_ensemble_preds)
        means_camera = [good_ensemble_preds[:, i].mean() for i in
                        range(good_ensemble_preds.shape[1])]

        ensemble_preds = np.hstack(cam_ensemble_preds[k])
        ensemble_vars = np.hstack(cam_ensemble_vars)
        ensemble_likes = np.hstack(cam_ensemble_likes)

        good_scaled_ensemble_preds = remove_camera_means(
            good_ensemble_preds[None, :, :], means_camera
        )[0]

        ensemble_pca, ensemble_ex_var = pca(good_scaled_ensemble_preds, 3)
        scaled_ensemble_preds = remove_camera_means(
            ensemble_preds[None, :, :], means_camera
        )[0]

        ensemble_pcs = ensemble_pca.transform(scaled_ensemble_preds)
        good_ensemble_pcs = ensemble_pcs[good_frames]
        y_obs = scaled_ensemble_preds

        if inflate_vars:
            inflated = True
            tmp_vars = ensemble_vars
            while inflated:
                # Compute Mahalanobis distances
                maha_results = compute_mahalanobis(y_obs, tmp_vars, likelihoods=ensemble_likes)
                # Inflate variances based on Mahalanobis distances
                inflated_ens_vars, inflated = inflate_variance(
                    tmp_vars, maha_results['mahalanobis'], threshold=5, scalar=2,
                )
                tmp_vars = inflated_ens_vars
        else:
            inflated_ens_vars = ensemble_vars


        # Kalman Filter
        m0 = np.asarray([0.0, 0.0, 0.0])
        S0 = np.asarray([[np.var(good_ensemble_pcs[:, 0]), 0.0, 0.0],
                         [0.0, np.var(good_ensemble_pcs[:, 1]), 0.0],
                         [0.0, 0.0, np.var(good_ensemble_pcs[:, 2])]])  # diagonal: var
        A = np.eye(3)
        d_t = good_ensemble_pcs[1:] - good_ensemble_pcs[:-1]
        C = ensemble_pca.components_.T
        R = np.eye(ensemble_pca.components_.shape[1])
        cov_matrix = np.cov(d_t.T)
        smooth_param_final, ms, Vs, _, _ = multicam_optimize_smooth(
            cov_matrix, y_obs, m0, S0, C, A, R, inflated_ens_vars, s_frames, smooth_param
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
                y_m_smooth.T[x_i] + means_camera[x_i],
                y_m_smooth.T[y_i] + means_camera[y_i],
                ensemble_likes[:, x_i],
                ensemble_preds[:, x_i],
                ensemble_preds[:, y_i],
                inflated_ens_vars[:, x_i],
                inflated_ens_vars[:, y_i],
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
