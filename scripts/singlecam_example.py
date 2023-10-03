"""Example script for single-camera datasets."""

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from eks.utils import convert_lp_dlc
from eks.singleview_smoother import ensemble_kalman_smoother_single_view


parser = argparse.ArgumentParser()
parser.add_argument(
    '--csv-dir',
    required=True,
    help='directory of model prediction csv files',
    type=str,
)
parser.add_argument(
    '--bodypart-list',
    required=True,
    nargs='+',
    help='the list of body parts to be ensembled and smoothed',
)
parser.add_argument(
    '--save-dir',
    help='save directory for outputs (default is csv-dir)',
    default=None,
    type=str,
)
parser.add_argument(
    '--s',
    help='smoothing parameter ranges from .01-20 (smaller values = more smoothing)',
    default=10,
    type=float,
)
args = parser.parse_args()

# collect user-provided args
csv_dir = os.path.abspath(args.csv_dir)
bodypart_list = args.bodypart_list
save_dir = args.save_dir
s = args.s

# ---------------------------------------------
# run EKS algorithm
# ---------------------------------------------

# handle I/O
if not os.path.isdir(csv_dir):
    raise ValueError('--csv-dir must be a valid directory containing prediction csv files')

if save_dir is None:
    save_dir = os.path.join(os.getcwd(), 'outputs')
    os.makedirs(save_dir, exist_ok=True)

# load files and put them in correct format
csv_files = os.listdir(csv_dir)
markers_list = []
for csv_file in csv_files:
    if not csv_file.endswith('csv'):
        continue
    markers_curr = pd.read_csv(os.path.join(csv_dir, csv_file), header=[0, 1, 2], index_col=0)
    keypoint_names = [c[1] for c in markers_curr.columns[::3]]
    model_name = markers_curr.columns[0][0]
    markers_curr_fmt = convert_lp_dlc(markers_curr, keypoint_names, model_name=model_name)
    markers_list.append(markers_curr_fmt)
if len(markers_list) == 0:
    raise FileNotFoundError(f'No marker csv files found in {csv_dir}')

# make empty dataframe to write eks results into
markers_eks = markers_curr.copy()
markers_eks.columns = markers_eks.columns.set_levels(['ensemble-kalman_tracker'], level=0)
for col in markers_eks.columns:
    if col[-1] == 'likelihood':
        # set this to 1.0 so downstream filtering functions don't get
        # tripped up
        markers_eks[col].values[:] = 1.0
    else:
        markers_eks[col].values[:] = np.nan

# loop over keypoints; apply eks to each individually
for keypoint_ensemble in bodypart_list:
    # run eks
    keypoint_df_dict = ensemble_kalman_smoother_single_view(
        markers_list=markers_list,
        keypoint_ensemble=keypoint_ensemble,
        smooth_param=s,
    )
    keypoint_df = keypoint_df_dict[keypoint_ensemble+'_df']
    # put results into new dataframe
    for coord in ['x', 'y', 'zscore']:
        src_cols = ('ensemble-kalman_tracker', f'{keypoint_ensemble}', coord)
        dst_cols = ('ensemble-kalman_tracker', f'{keypoint_ensemble}', coord)
        markers_eks.loc[:, dst_cols] = keypoint_df.loc[:, src_cols]

# save eks results
markers_eks.to_csv(os.path.join(save_dir, 'eks.csv'))


# ---------------------------------------------
# plot results
# ---------------------------------------------

# select example keypoint
kp = bodypart_list[0]
idxs = (0, 500)

fig, axes = plt.subplots(4, 1, figsize=(9, 8))

for ax, coord in zip(axes, ['x', 'y', 'likelihood', 'zscore']):
    # plot individual models
    ax.set_ylabel(coord, fontsize=12)
    if coord == 'zscore':
        ax.plot(
        markers_eks.loc[slice(*idxs), ('ensemble-kalman_tracker', f'{kp}', coord)],
        color='k', linewidth=2)
        ax.set_xlabel('Time (frames)', fontsize=12)
        continue
    for m, markers_curr in enumerate(markers_list):
        ax.plot(
            markers_curr.loc[slice(*idxs), f'{kp}_{coord}'], color=[0.5, 0.5, 0.5],
            label='Individual models' if m == 0 else None,
        )
    # plot eks
    if coord == 'likelihood':
        continue
    ax.plot(
        markers_eks.loc[slice(*idxs), ('ensemble-kalman_tracker', f'{kp}', coord)],
        color='k', linewidth=2, label='EKS',
    )
    if coord == 'x':
        ax.legend()
    

plt.suptitle(f'EKS results for {kp}', fontsize=14)
plt.tight_layout()

save_file = os.path.join(save_dir, 'example_singlecam_eks_result.pdf')
plt.savefig(save_file)
plt.close()
print(f'see example EKS output at {save_file}')
