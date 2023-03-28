import numpy as np
import os
import pandas as pd
from eks.utils import convert_lp_dlc
from eks.multiview_pca_smoother import ensemble_kalman_smoother_paw_asynchronous
import argparse
import glob

parser = argparse.ArgumentParser()
parser.add_argument("-model-dir", required=True, help="directory of models for ensembling",
                    type=str)
parser.add_argument("--save-dir", help="save directory for outputs (default is model-dir)",
                    default=None, type=float)
parser.add_argument("--s", help="smoothing parameter ranges from .01-2 (smaller values = more smoothing)",
                    default=2, type=float)
parser.add_argument("--quantile_keep_pca", help="percentage of the points are kept for multi-view PCA (lowest ensemble variance)",
                    default=25, type=float)
args = parser.parse_args()

model_dir = args.model_dir
if not os.path.isdir(model_dir):
    raise ValueError("model-dir must be a valid path to a directory")

if args.save_dir is None:
    save_dir = args.model_dir + '/outputs'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

s = args.s
quantile_keep_pca = args.quantile_keep_pca

keypoint_names = ['paw_l', 'paw_r']
markers_list_left_cam = []
markers_list_right_cam = []
time_stamps_left_cam = None
time_stamps_right_cam = None
paths = [path for path in glob.glob(model_dir + '/*') if not os.path.isdir(path)]
for i, path in enumerate(paths):
    if 'timestamps' not in path:
        marker_path = path
        print(f"model: {marker_path}")
        markers_tmp = pd.read_csv(marker_path, header=[0, 1, 2], index_col=0)
        if '.dlc' not in marker_path:
            markers_tmp = convert_lp_dlc(markers_tmp, keypoint_names)
        if 'left' in marker_path:
            markers_list_left_cam.append(markers_tmp)
        else:
            # switch right camera paws
            columns = {
                'paw_l_x': 'paw_r_x', 'paw_l_y': 'paw_r_y',
                'paw_l_likelihood': 'paw_r_likelihood',
                'paw_r_x': 'paw_l_x', 'paw_r_y': 'paw_l_y',
                'paw_r_likelihood': 'paw_l_likelihood'
            }
            markers_tmp = markers_tmp.rename(columns=columns)
            # reorder columns
            markers_tmp = markers_tmp.loc[:, columns.keys()]
            markers_list_right_cam.append(markers_tmp)
    else:
        if 'left' in path:
            time_stamps_left_cam = np.load(path)
        else:
            time_stamps_right_cam = np.load(path)
            
if time_stamps_left_cam is None or time_stamps_right_cam is None:
    raise ValueError('Need timestamps for both cameras')
    
if len(markers_list_right_cam) != len(markers_list_left_cam) or len(markers_list_left_cam) == 0:
    raise ValueError("There must be the same number of left and right camera models and >=1 model for each.")

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
    save_path = save_dir + f'/kalman_smoothed_paw_traces.{view}.csv'
    print(f'saving smoothed markers from {view} view to ' + save_path)
    df_dict[f'{view}_df'].to_csv(save_path)
