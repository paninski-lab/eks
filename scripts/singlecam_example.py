"""Example script for single-camera datasets."""
import os
import time
import numpy as np

from eks.command_line_args import handle_io, handle_parse_args
from eks.utils import format_data, populate_output_dataframe, plot_results
from eks.singleview_smoother import vectorized_ensemble_kalman_smoother_single_view
from eks.jax_singleview_smoother import jax_ensemble_kalman_smoother_single_view

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
jax = args.jax  # whether jax is used for speedups

# Load and format input files and prepare an empty DataFrame for output.
input_dfs, output_df, _ = format_data(args.input_dir, data_type)
print('Input data has been read in.')

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

# Initialize
df_dicts, s_finals, nll_values = [], [], []

# Call the smoother function
if jax == "True":
    df_dicts, s_finals, nll_values_array = jax_ensemble_kalman_smoother_single_view(
        markers_3d_array,
        bodypart_list,
        s,
        s_frames
    )
else:
    df_dicts, s_finals, nll_values_array = vectorized_ensemble_kalman_smoother_single_view(
        markers_3d_array,
        bodypart_list,
        s,
        s_frames
    )

# Save eks results in new DataFrames and .csv output files
for k in range(len(bodypart_list)):
    df = df_dicts[k][bodypart_list[k] + '_df']
    output_df = populate_output_dataframe(df, bodypart_list[k], output_df)
    save_filename = save_filename or f'{smoother_type}_{s_finals[k]}.csv'
    output_df.to_csv(os.path.join(save_dir, save_filename))
print("DataFrames successfully converted to CSV")

# Plot results
keypoint_i = -1  # keypoint to be plotted
plot_results(output_df=output_df,
             input_dfs_list=input_dfs,
             key=f'{bodypart_list[keypoint_i]}',
             idxs=(0, 500),
             s_final=s_finals[keypoint_i],
             nll_values=nll_values_array[keypoint_i],
             save_dir=save_dir,
             smoother_type=smoother_type
             )
