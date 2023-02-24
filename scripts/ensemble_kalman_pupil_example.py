import numpy as np
import os
import pandas as pd
from eks.utils import convert_lp_dlc
from eks.pupil_smoother import ensemble_kalman_smoother_pupil
import argparse
import glob

parser = argparse.ArgumentParser()
parser.add_argument("-model-dir", help="directory of models for ensembling",
                    type=str)
parser.add_argument("--save-dir", help="save directory for outputs (default is model-dir)",
                    default=None, type=float)
parser.add_argument("--diameter-s", help="smoothing parameter for diameter (closer to 1 = more smoothing)",
                    default=.9999, type=float)
parser.add_argument("--com-s", help="smoothing parameter for center of mass (closer to 1 = more smoothing)",
                    default=.999, type=float)
args = parser.parse_args()

model_dir = args.model_dir
if args.save_dir is None:
    save_dir = args.model_dir + '/outputs'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
#parameters hand-picked for smoothing purposes (diameter_s, com_s, com_s)
state_transition_matrix = np.asarray([[args.diameter_s, 0, 0], [0, args.com_s, 0], [0, 0, args.com_s]]) 
print(f"Smoothing matrix: {state_transition_matrix}")

markers_list = []
keypoint_names = ['pupil_top_r', 'pupil_right_r', 'pupil_bottom_r', 'pupil_left_r']
marker_paths = [path for path in glob.glob(model_dir + '/*') if not os.path.isdir(path)]
for i, marker_path in enumerate(marker_paths):
    print(f"model {i}: {marker_path}")
    markers_tmp = pd.read_csv(marker_path, header=[0, 1, 2], index_col=0)
    if '.dlc' not in marker_path:
        markers_tmp = convert_lp_dlc(markers_tmp, keypoint_names, model_name='heatmap_tracker')
    markers_list.append(markers_tmp)
    
df_dicts = ensemble_kalman_smoother_pupil(markers_list, keypoint_names, tracker_name='heatmap_tracker', state_transition_matrix=state_transition_matrix)

save_path = save_dir + f'/kalman_smoothed_pupil_traces.csv'
print("saving smoothed predictions to " + save_path)
df_dicts['markers_df'].to_csv(save_path)

save_path = save_dir + f'/kalman_smoothed_latents.csv'
print("saving latents to " + save_path)
df_dicts['latents_df'].to_csv(save_path)
