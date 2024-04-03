"""Example script for ibl-paw dataset."""

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from eks.utils import convert_lp_dlc
from eks.multiview_pca_smoother import ensemble_kalman_smoother_paw_asynchronous
from scripts.general_scripts import handle_io

parser = argparse.ArgumentParser()
parser.add_argument(
    '--csv-dir',
    required=True,
    help='directory of models for ensembling',
    type=str,
)
parser.add_argument(
    '--save-dir',
    help='save directory for outputs (default is csv-dir)',
    default=None,
    type=str,
)
parser.add_argument(
    '--s',
    help='smoothing parameter ranges from .01-2 (smaller values = more smoothing)',
    default=1,
    type=float,
)
parser.add_argument(
    '--quantile_keep_pca',
    help='percentage of the points are kept for multi-view PCA (lowest ensemble variance)',
    default=25,
    type=float,
)
args = parser.parse_args()

# collect user-provided args
csv_dir = os.path.abspath(args.csv_dir)
save_dir = args.save_dir
s = args.s
quantile_keep_pca = args.quantile_keep_pca


# ---------------------------------------------
# run EKS algorithm
# ---------------------------------------------

# handle I/O
save_dir = handle_io(csv_dir, save_dir)

# load files and put them in correct format
markers_list_left = []
markers_list_right = []
timestamps_left = None
timestamps_right = None
filenames = os.listdir(csv_dir)
for filename in filenames:
    if 'timestamps' not in filename:
        markers_curr = pd.read_csv(os.path.join(csv_dir, filename), header=[0, 1, 2], index_col=0)
        keypoint_names = [c[1] for c in markers_curr.columns[::3]]
        model_name = markers_curr.columns[0][0]
        markers_curr_fmt = convert_lp_dlc(markers_curr, keypoint_names, model_name=model_name)
        if 'left' in filename:
            markers_list_left.append(markers_curr_fmt)
        else:
            # switch right camera paws
            columns = {
                'paw_l_x': 'paw_r_x', 'paw_l_y': 'paw_r_y',
                'paw_l_likelihood': 'paw_r_likelihood',
                'paw_r_x': 'paw_l_x', 'paw_r_y': 'paw_l_y',
                'paw_r_likelihood': 'paw_l_likelihood'
            }
            markers_curr_fmt = markers_curr_fmt.rename(columns=columns)
            # reorder columns
            markers_curr_fmt = markers_curr_fmt.loc[:, columns.keys()]
            markers_list_right.append(markers_curr_fmt)
    else:
        if 'left' in filename:
            timestamps_left = np.load(os.path.join(csv_dir, filename))
        else:
            timestamps_right = np.load(os.path.join(csv_dir, filename))

# file checks
if timestamps_left is None or timestamps_right is None:
    raise ValueError('Need timestamps for both cameras')
    
if len(markers_list_right) != len(markers_list_left) or len(markers_list_left) == 0:
    raise ValueError(
        'There must be the same number of left and right camera models and >=1 model for each.')

# run eks
df_dicts, markers_list_left_cam, markers_list_right_cam = ensemble_kalman_smoother_paw_asynchronous(
    markers_list_left_cam=markers_list_left,
    markers_list_right_cam=markers_list_right,
    timestamps_left_cam=timestamps_left,
    timestamps_right_cam=timestamps_right,
    keypoint_names=keypoint_names,
    smooth_param=s,
    quantile_keep_pca=quantile_keep_pca,
)

# save smoothed markers from each view
for view in ['left', 'right']:
    save_file = os.path.join(save_dir, f'kalman_smoothed_paw_traces.{view}.csv')
    df_dicts[f'{view}_df'].to_csv(save_file)


# ---------------------------------------------
# plot results
# ---------------------------------------------

# select example keypoint from example camera view
kp = keypoint_names[0] 
view = 'left'  # NOTE: if you want to use right view, must swap paw identities
idxs = (0, 200)
kp_swap = [x for x in keypoint_names if x != kp][0]
if view == 'right':
    markers_list_curr = markers_list_right_cam
else:
    markers_list_curr = markers_list_left_cam
    
fig, axes = plt.subplots(4, 1, figsize=(9, 6))

for ax, coord in zip(axes, ['x', 'y', 'likelihood', 'zscore']):
    ax.set_ylabel(coord, fontsize=12)
    if coord == 'zscore':
        zscores = df_dicts[f'{view}_df'].loc[slice(*idxs), ('ensemble-kalman_tracker', kp, coord)].values
        ax.plot(
            df_dicts[f'{view}_df'].loc[slice(*idxs), ('ensemble-kalman_tracker', kp, coord)],
            color=[0.5, 0.5, 0.5]
        )
        ax.set_xlabel('Time (frames)', fontsize=12)
        continue
    # plot individual models
    for m, markers_curr in enumerate(markers_list_curr):
        if coord == 'likelihood':
            ax.plot(
                markers_list_left[m].loc[slice(*idxs), f'{kp}_{coord}'], color=[0.5, 0.5, 0.5],
                label='Individual models' if m == 0 else None,
            )
        else:
            if view == 'right':
                if coord == 'x':
                    ax.plot(
                        128 - markers_curr.loc[slice(*idxs), f'{kp_swap}_{coord}'], color=[0.5, 0.5, 0.5],
                        label='Individual models' if m == 0 else None,
                    )
                else:
                    ax.plot(
                        markers_curr.loc[slice(*idxs), f'{kp_swap}_{coord}'], color=[0.5, 0.5, 0.5],
                        label='Individual models' if m == 0 else None,
                    )
            else:
                ax.plot(
                    markers_curr.loc[slice(*idxs), f'{kp}_{coord}'], color=[0.5, 0.5, 0.5],
                    label='Individual models' if m == 0 else None,
                )
    # plot eks
    if coord == 'likelihood':
        continue
    ax.plot(
        df_dicts[f'{view}_df'].loc[slice(*idxs), ('ensemble-kalman_tracker', kp, coord)],
        color='k', linewidth=2, label='EKS',
    )
    if coord == 'x':
        ax.legend()

plt.suptitle(f'EKS results for {kp} ({view} view)', fontsize=14)
plt.tight_layout()

save_file = os.path.join(save_dir, 'example_multiview_paw_eks_result.pdf')
plt.savefig(save_file)
plt.close()
print(f'see example EKS output at {save_file}')
