import os

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.decomposition import PCA
from typeguard import typechecked

from eks.marker_array import (
    MarkerArray,
    input_dfs_to_markerArray,
    mA_to_stacked_array,
    stacked_array_to_mA,
)
from eks.multicam_smoother import ensemble_kalman_smoother_multicam
from eks.stats import compute_pca
from eks.utils import convert_lp_dlc, make_dlc_pandas_index


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
    inflate_vars: bool = False,
    n_latent: int = 3
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
            inflate_vars: True to use Mahalanobis distance thresholding to inflate ensemble variance
            n_latent: number of dimensions to keep from PCA

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
        verbose=verbose,
        inflate_vars=inflate_vars,
        n_latent=n_latent,
        inflate_vars_kwargs={'likelihoods': None}
    )
    # Save output DataFrames to CSVs (one per camera view)
    os.makedirs(save_dir, exist_ok=True)
    for c, camera in enumerate(camera_names):
        save_filename = f'multicam_{camera}_results.csv'
        camera_dfs[c].to_csv(os.path.join(save_dir, save_filename))
    return camera_dfs, smooth_params_final, input_dfs_list, bodypart_list
