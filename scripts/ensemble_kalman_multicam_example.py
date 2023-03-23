import numpy as np
import os
import pandas as pd
from eks.utils import convert_lp_dlc
from eks.multiview_pca_smoother import ensemble_kalman_smoother_multi_cam
import argparse
import glob

parser = argparse.ArgumentParser()
parser.add_argument("-parquet-path", required=True, help="path to parquet file with model preds",
                    type=str)
parser.add_argument("-video-name", required=True, help="video predictions to be ensembled and smoothed",
                    type=str)
parser.add_argument("-train-frames", required=True, help="train frames of models to be ensembled",
                    type=int)
parser.add_argument("-model-type", required=True, help="can be 'baseline', 'context', 'semi-super', 'semi-super context'.",
                    type=str)
parser.add_argument("-keypoint-ensemble-list", required=True,  nargs='+', help="the list of keypoints to be ensembled and smoothed")
parser.add_argument("-camera-names", required=True,  nargs='+', help="the camera names")
parser.add_argument("--save-dir", help="save directory for outputs (default is model-dir)",
                    default=None, type=float)
parser.add_argument("--s", help="smoothing parameter ranges from .01-2 (smaller values = more smoothing)",
                    default=.01, type=float)
parser.add_argument("--quantile_keep_pca", help="percentage of the points are kept for multi-view PCA (lowest ensemble variance)",
                    default=25, type=float)
args = parser.parse_args()

parquet_path = args.parquet_path
if not os.path.isfile(parquet_path):
    raise ValueError("parquet-path must be a valid path to a file")

if args.save_dir is None:
    save_dir = os.getcwd() + '/outputs'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

s = args.s
quantile_keep_pca = args.quantile_keep_pca
df_ = pd.read_parquet(parquet_path)

video_name = args.video_name
rng_seed_data_pt_list = ['0', '1', '2', '3', '4']
train_frames = str(args.train_frames)
model_type = args.model_type #{'baseline', 'context', 'semi-super', 'semi-super context'}

keypoint_names = set()
for key_pair in list(df_.keys()):
    if key_pair[0] not in ['video_name', 'model_path', 'rng_seed_data_pt', 'train_frames', 'model_type']:
        keypoint_names.add(key_pair[0])
keypoint_names = list(keypoint_names)
keypoint_ensemble_list = args.keypoint_ensemble_list
num_models = len(rng_seed_data_pt_list)
num_cameras = len(args.camera_names)
camera_names = args.camera_names

cameras_df_list = []
for keypoint_ensemble in keypoint_ensemble_list:
    markers_list_cameras = [[] for i in range(num_cameras)]
    likelihood_list_cameras = [[] for i in range(num_cameras)]
    for seed in rng_seed_data_pt_list:
        df = df_[
            (df_.video_name==video_name)
            & (df_.rng_seed_data_pt==seed)
            & (df_.train_frames==train_frames)
            & (df_.model_type==model_type)
        ]
        df = df.drop(
                columns=['model_path', 'rng_seed_data_pt', 'train_frames', 'model_type']
        )
        markers_tmp = convert_lp_dlc(df, keypoint_names, model_name=None)
        for camera in range(num_cameras):
            markers_list_cameras[camera].append(markers_tmp[[key for key in markers_tmp.keys() if camera_names[camera] in key and 'likelihood' not in key and keypoint_ensemble in key]])
            likelihood_list_cameras[camera].append(markers_tmp[[key for key in markers_tmp.keys() if camera_names[camera] in key and 'likelihood' in key and keypoint_ensemble in key]])

    cameras_df = ensemble_kalman_smoother_multi_cam(markers_list_cameras, keypoint_ensemble, s, quantile_keep_pca, camera_names)
    for view in camera_names:
        save_path = save_dir + f'/kalman_smoothed_{keypoint_ensemble}_traces.{view}.csv'
        print(f'saving smoothed markers from {view} view to ' + save_path)
        cameras_df[f'{view}_df'].to_csv(save_path)

