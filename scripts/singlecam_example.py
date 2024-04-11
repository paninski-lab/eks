"""Example script for single-camera datasets."""

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from eks.utils import convert_lp_dlc
from eks.singleview_smoother import ensemble_kalman_smoother_single_view, get_nll_values
from scripts.general_scripts import handle_io, handle_parse_args

# collect user-provided args
args = handle_parse_args('singlecam')
csv_dir = os.path.abspath(args.csv_dir)
bodypart_list = args.bodypart_list
save_dir = args.save_dir
s = args.s

# ---------------------------------------------
# run EKS algorithm
# ---------------------------------------------

# handle I/O
save_dir = handle_io(csv_dir, save_dir)

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

# make empty list for nll values
nll_values = []

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

# get NLL values from the smoother object
nll_values = get_nll_values()
nll_values_subset = nll_values[idxs[0]:idxs[1]]

fig, axes = plt.subplots(5, 1, figsize=(9, 10))  # Increased the number of subplots to accommodate nll_values

for ax, coord in zip(axes, ['x', 'y', 'likelihood', 'zscore']):
    # Rename axes label for likelihood and zscore coordinates
    if coord == 'likelihood':
        ylabel = 'model likelihoods'
    elif coord == 'zscore':
        ylabel = 'EKS disagreement'
    else:
        ylabel = coord


    # plot individual models
    ax.set_ylabel(ylabel, fontsize=12)
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

    # Plot nll_values_subset against the time axis
    axes[-1].plot(range(*idxs), nll_values_subset, color='k', linewidth=2)
    axes[-1].set_ylabel('EKS NLL', fontsize=12)


plt.suptitle(f'EKS results for {kp}, smoothing = {s}', fontsize=14)
plt.tight_layout()

save_file = os.path.join(save_dir, f'singlecam s={s}.pdf')
plt.savefig(save_file)
plt.close()
print(f'see example EKS output at {save_file}')
