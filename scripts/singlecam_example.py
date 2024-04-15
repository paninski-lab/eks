"""Example script for single-camera datasets."""

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from scripts.general_scripting import handle_io, handle_parse_args, format_csv, populate_output_dataframe
from smoothers.singleview_smoother import ensemble_kalman_smoother_single_view


# Collect User-Provided Args
smoother_type = 'singlecam'
args = handle_parse_args(smoother_type)

csv_dir = os.path.abspath(args.csv_dir)
# Find save directory if specified, otherwise defaults to outputs\
save_dir = handle_io(csv_dir, args.save_dir)

bodypart_list = args.bodypart_list
s = args.s  # optional, defaults to automatic optimization

# Load and format input files and prepare an empty DataFrame for output.
# markers_list : list of input DataFrames
# markers_eks : empty DataFrame for EKS output
markers_list, markers_eks = format_csv(csv_dir, 'lp')


# ---------------------------------------------
# Run EKS Algorithm
# ---------------------------------------------

# loop over keypoints; apply eks to each individually
for keypoint_ensemble in bodypart_list:
    # run eks
    keypoint_df_dict, s_final, nll_values = ensemble_kalman_smoother_single_view(
        markers_list,
        keypoint_ensemble,
        s,
    )
    keypoint_df = keypoint_df_dict[keypoint_ensemble + '_df']

    # put results into new dataframe
    markers_eks = populate_output_dataframe(keypoint_df, keypoint_ensemble, markers_eks)

# save optimized smoothing param for plot title
s = s_final

# save eks results
markers_eks.to_csv(os.path.join(save_dir, f'{smoother_type}, s={s}_.csv'))


# ---------------------------------------------
# plot results
# ---------------------------------------------

# select example keypoint
kp = bodypart_list[-1]
idxs = (0, 500)

# crop NLL values
nll_values_subset = nll_values[idxs[0]:idxs[1]]

fig, axes = plt.subplots(5, 1, figsize=(9, 10))

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
