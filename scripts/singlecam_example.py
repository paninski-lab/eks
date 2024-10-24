"""Example script for single-camera datasets."""
import os

import numpy as np
import pandas as pd

from eks.command_line_args import handle_io, handle_parse_args
from eks.singlecam_smoother import ensemble_kalman_smoother_singlecam
from eks.utils import format_data, plot_results, populate_output_dataframe

# Collect User-Provided Args
smoother_type = 'singlecam'
args = handle_parse_args(smoother_type)
input_dir = os.path.abspath(args.input_dir)
data_type = args.data_type  # Note: LP and DLC are .csv, SLP is .slp
save_dir = handle_io(input_dir, args.save_dir)  # defaults to outputs\
save_filename = args.save_filename
bodypart_list = args.bodypart_list
s = args.s  # defaults to automatic optimization
s_frames = args.s_frames  # frames to be used for automatic optimization (only if no --s flag)
blocks = args.blocks
ensembling_mode = 'median'


# Load and format input files and prepare an empty DataFrame for output.
input_dfs, output_df, keypoint_names = format_data(args.input_dir, data_type)
if bodypart_list is None:
    bodypart_list = keypoint_names
print(f'Input data has been read in for the following keypoints:\n{bodypart_list}')

# Convert list of DataFrames to a 3D NumPy array
data_arrays = [df.to_numpy() for df in input_dfs]
markers_3d_array = np.stack(data_arrays, axis=0)

# Map keypoint names to keys in input_dfs and crop markers_3d_array
keypoint_is = {}
keys = []
for i, col in enumerate(input_dfs[0].columns):
    keypoint_is[col] = i
for part in bodypart_list:
    keys.append(keypoint_is[part + '_x'])
    keys.append(keypoint_is[part + '_y'])
    keys.append(keypoint_is[part + '_likelihood'])
key_cols = np.array(keys)
markers_3d_array = markers_3d_array[:, :, key_cols]

# Call the smoother function
df_dicts, s_finals = ensemble_kalman_smoother_singlecam(
    markers_3d_array=markers_3d_array,
    bodypart_list=bodypart_list,
    smooth_param=s,
    s_frames=s_frames,
    blocks=blocks,
    ensembling_mode=ensembling_mode,
)

# Save eks results in new DataFrames and .csv output files
for k in range(len(bodypart_list)):
    keypoint_name = bodypart_list[k]
    df = df_dicts[k][keypoint_name + '_df']
    output_df = populate_output_dataframe(df, keypoint_name, output_df)
    # update likelihoods using ensemble
    dst_cols = ('ensemble-kalman_tracker', keypoint_name, 'likelihood')
    df_likelihoods = pd.concat([df[f'{keypoint_name}_likelihood'] for df in input_dfs], axis=1)
    if ensembling_mode == 'median':
        l_vals = df_likelihoods.median(axis=1).values
    elif ensembling_mode == 'confidence_weighted_mean':
        l_vals = df_likelihoods.mean(axis=1).values
    else:
        raise NotImplementedError
    output_df.loc[:, dst_cols] = l_vals

# save out dataframe
keypoint_i = -1  # keypoint to be plotted
save_filename = save_filename or f'{smoother_type}_{s_finals[keypoint_i]}.csv'
output_df.to_csv(os.path.join(save_dir, save_filename))
print("DataFrames successfully converted to CSV")

# Plot results
plot_results(
    output_df=output_df,
    input_dfs_list=input_dfs,
    key=f'{bodypart_list[keypoint_i]}',
    idxs=(0, 500),
    s_final=s_finals[keypoint_i],
    nll_values=None,
    save_dir=save_dir,
    smoother_type=smoother_type,
)
