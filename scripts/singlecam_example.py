"""Example script for single-camera datasets."""

import matplotlib.pyplot as plt
import os

from general_scripting import handle_io, handle_parse_args
from eks.utils import format_data, populate_output_dataframe
from eks.singleview_smoother import ensemble_kalman_smoother_single_view


# Collect User-Provided Args
smoother_type = 'singlecam'
args = handle_parse_args(smoother_type)

input_dir = os.path.abspath(args.input_dir)

# Note: LP and DLC are .csv, SLP is .slp
data_type = args.data_type

# Find save directory if specified, otherwise defaults to outputs\
save_dir = handle_io(input_dir, args.save_dir)
save_filename = args.save_filename

bodypart_list = args.bodypart_list
s = args.s  # optional, defaults to automatic optimization

# Load and format input files and prepare an empty DataFrame for output.
input_dfs_list, output_df = format_data(args.input_dir, data_type)


# ---------------------------------------------
# Run EKS Algorithm
# ---------------------------------------------

# loop over keypoints; apply eks to each individually
for keypoint in bodypart_list:
    # run eks
    keypoint_df_dict, s_final, nll_values = ensemble_kalman_smoother_single_view(
        input_dfs_list,
        keypoint,
        s,
    )
    keypoint_df = keypoint_df_dict[keypoint + '_df']

    # put results into new dataframe
    output_df = populate_output_dataframe(keypoint_df, keypoint, output_df)
    output_df.to_csv('populated_output.csv', index=False)
    print(f"DataFrame successfully converted to CSV")
# save optimized smoothing param for plot title

# save eks results
save_filename = save_filename or f'{smoother_type}.csv'  # use type and s if no user input
output_df.to_csv(os.path.join(save_dir, save_filename))


# ---------------------------------------------
# plot results
# ---------------------------------------------

# select example keypoint
kp = bodypart_list[-1]
idxs = (0, 1990)

# crop NLL values
# nll_values_subset = nll_values[idxs[0]:idxs[1]]

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
            output_df.loc[slice(*idxs), ('ensemble-kalman_tracker', f'{kp}', coord)],
            color='k', linewidth=2)
        ax.set_xlabel('Time (frames)', fontsize=12)
        continue
    for m, markers_curr in enumerate(input_dfs_list):
        ax.plot(
            markers_curr.loc[slice(*idxs), f'{kp}_{coord}'], color=[0.5, 0.5, 0.5],
            label='Individual models' if m == 0 else None,
        )
    # plot eks
    if coord == 'likelihood':
        continue
    ax.plot(
        output_df.loc[slice(*idxs), ('ensemble-kalman_tracker', f'{kp}', coord)],
        color='k', linewidth=2, label='EKS',
    )
    if coord == 'x':
        ax.legend()

    # Plot nll_values_subset against the time axis
    # axes[-1].plot(range(*idxs), nll_values_subset, color='k', linewidth=2)
    # axes[-1].set_ylabel('EKS NLL', fontsize=12)


plt.suptitle(f'EKS results for {kp}, smoothing = {s}', fontsize=14)
plt.tight_layout()

save_file = os.path.join(save_dir, f'singlecam_s={s}.pdf')
plt.savefig(save_file)
plt.close()
print(f'see example EKS output at {save_file}')
