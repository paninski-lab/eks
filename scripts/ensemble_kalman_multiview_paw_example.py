import numpy as np
import os
import pandas as pd

from eks.utils import convert_lp_dlc
from eks.multiview_pca_smoother import ensemble_kalman_smoother_paw_asynchronous


base_path = '/media/cat/cole/ibl-paw_ensembling/'
video_name = '032ffcdf-7692-40b3-b9ff-8def1fc18b2e'
s = 2  # smoothing param, Ranges from 2-10 (needs more exploration)
quantile_keep_pca = 25  # percentage of the points are kept for multi-view PCA (lowest ensemble variance)

keypoint_names = ['paw_l', 'paw_r']
markers_list_left_cam = []
markers_list_right_cam = []
num_models = 10
for i in range(num_models):
    marker_path_left_cam = os.path.join(base_path, f'{video_name}.left.rng={i}.csv')
    markers_tmp_left_cam = pd.read_csv(marker_path_left_cam, header=[0, 1, 2], index_col=0)
    marker_path_right_cam = os.path.join(base_path, f'{video_name}.right.rng={i}.csv')
    markers_tmp_right_cam = pd.read_csv(marker_path_right_cam, header=[0, 1, 2], index_col=0)
    if '.dlc' not in marker_path_left_cam:
        markers_tmp_left_cam = convert_lp_dlc(markers_tmp_left_cam, keypoint_names)
    if '.dlc' not in marker_path_right_cam:
        markers_tmp_right_cam = convert_lp_dlc(markers_tmp_right_cam, keypoint_names)
    markers_list_left_cam.append(markers_tmp_left_cam)
    # switch right camera paws
    columns = {
        'paw_l_x': 'paw_r_x', 'paw_l_y': 'paw_r_y',
        'paw_l_likelihood': 'paw_r_likelihood',
        'paw_r_x': 'paw_l_x', 'paw_r_y': 'paw_l_y',
        'paw_r_likelihood': 'paw_l_likelihood'
    }
    markers_tmp_right_cam = markers_tmp_right_cam.rename(columns=columns)
    # reorder columns
    markers_tmp_right_cam = markers_tmp_right_cam.loc[:, columns.keys()]
    markers_list_right_cam.append(markers_tmp_right_cam)

time_stamps_left_cam = np.load(os.path.join(base_path, f'{video_name}.timestamps.left.npy'))
time_stamps_right_cam = np.load(os.path.join(base_path, f'{video_name}.timestamps.right.npy'))

df_dict = ensemble_kalman_smoother_paw_asynchronous(
    markers_list_left_cam=markers_list_left_cam,
    markers_list_right_cam=markers_list_right_cam,
    timestamps_left_cam=time_stamps_left_cam,
    timestamps_right_cam=time_stamps_right_cam,
    keypoint_names=keypoint_names,
    smooth_param=s,
    quantile_keep_pca=quantile_keep_pca,
)

# save smoothed markers from each view
for view in ['left', 'right']:
    save_path = base_path + f'/kalman_smoothed_paw_traces_{video_name}.{view}.csv'
    print(f'saving smoothed markers from {view} view to ' + save_path)
    df_dict[f'{view}_df'].to_csv(save_path)
