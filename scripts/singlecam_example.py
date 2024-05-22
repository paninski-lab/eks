"""Example script for single-camera datasets."""
import os
import numpy as np
import pandas as pd

from eks.command_line_args import handle_io, handle_parse_args
from eks.utils import format_data, format_data_jax, populate_output_dataframe, plot_results, \
    batch_process_ensemble_kalman, jax_populate_output_dataframe
from eks.singleview_smoother import ensemble_kalman_smoother_single_view, \
    vectorized_ensemble_kalman_smoother_single_view

# Collect User-Provided Args
smoother_type = 'singlecam'
args = handle_parse_args(smoother_type)
input_dir = os.path.abspath(args.input_dir)
data_type = args.data_type  # Note: LP and DLC are .csv, SLP is .slp
save_dir = handle_io(input_dir, args.save_dir)  # defaults to outputs\
save_filename = args.save_filename
bodypart_list = args.bodypart_list
s = args.s  # defaults to automatic optimization
s_frames = args.s_frames # frames to be used for automatic optimization (only if no --s flag)
vectorize = args.vectorize # whether vectorization is used for speedups

# Load and format input files and prepare an empty DataFrame for output.
input_dfs_list, output_df, _ = format_data(args.input_dir, data_type)
print('Input data has been read in.')

# Assuming you have a dictionary that maps keypoint names to indices
keypoint_indices = {}
for i, col in enumerate(input_dfs_list[0].columns):
        keypoint_indices[col] = i

# Vectorization if specified
if vectorize == 'True':
    print('Vectorizing!')

    # Convert list of DataFrames to a 3D NumPy array
    data_arrays = [df.to_numpy() for df in input_dfs_list]
    all_keypoints_data = np.stack(data_arrays, axis=0)

    # Prepare keys using indices from the dictionary
    indices = []
    for part in bodypart_list:
        indices.append(keypoint_indices[part + '_x'])
        indices.append(keypoint_indices[part + '_y'])

    # Call the vectorized function with numerical indices
    df_dict_array, s_finals, nll_values_array = vectorized_ensemble_kalman_smoother_single_view(
        all_keypoints_data,
        indices,
        bodypart_list,
        s,
        s_frames
    )
    # save eks results
    for k in range(len(bodypart_list)):
        df = df_dict_array[k][bodypart_list[k] + '_df']
        # put results into new dataframe
        output_df = populate_output_dataframe(df, bodypart_list[k], output_df)
        save_filename = save_filename or f'{smoother_type}_{s_finals[k]}.csv'
        output_df.to_csv(os.path.join(save_dir, save_filename))
    print(f"DataFrame successfully converted to CSV")

    # plot results
    keypoint_i = -1
    plot_results(output_df=output_df,
                 input_dfs_list=input_dfs_list,
                 key=f'{bodypart_list[keypoint_i]}',
                 idxs=(0, 500),
                 s_final=s_finals[keypoint_i],
                 nll_values=nll_values_array[keypoint_i],
                 save_dir=save_dir,
                 smoother_type=smoother_type
                 )

else:
    # loop over keypoints; apply eks to each individually
    for i, keypoint in enumerate(bodypart_list):
        keypoint_index = keypoint_indices[keypoint + '_x']  # Adjust index if your function expects different inputs

        # Check if smooth_param given by user as float or array
        if s is None or len(s) == 1:
            s = s
        elif len(s) > 1:
            s = s[i]

        # run eks
        keypoint_df_dict, s_final, nll_values = ensemble_kalman_smoother_single_view(
            input_dfs_list,
            keypoint,
            s,
            s_frames
        )
        keypoint_df = keypoint_df_dict[keypoint + '_df']
        # put results into new dataframe
        output_df = populate_output_dataframe(keypoint_df, keypoint, output_df)

    # save eks results
    save_filename = save_filename or f'{smoother_type}_{s_final}.csv'
    output_df.to_csv(os.path.join(save_dir, save_filename))
    print(f"DataFrame successfully converted to CSV")

    # plot results
    plot_results(output_df=output_df,
                 input_dfs_list=input_dfs_list,
                 key=f'{bodypart_list[-1]}',
                 idxs=(0, 500),
                 s_final=s_final,
                 nll_values=nll_values,
                 save_dir=save_dir,
                 smoother_type=smoother_type
                 )




'''
    # Prepare JAX arrays
    data_by_keypoint = format_data_jax(input_dfs_list, bodypart_list)
    keypoints_jax_arrays = jnp.array([data_by_keypoint[kp] for kp in bodypart_list])
    optimize_flags = jnp.array(
        [1 if s[i] is None else 0 for i in
         range(len(bodypart_list) - 1)])  # 1 for optimize, 0 for fixed
    print(keypoints_jax_arrays)
    print("Function type before calling batch_process_ensemble_kalman:",
          type(jax_ensemble_kalman_smoother_single_view))

    # Run the batch processing Kalman smoother
    results = batch_process_ensemble_kalman(
        jax_ensemble_kalman_smoother_single_view,
        keypoints_jax_arrays,
        bodypart_list,
        s_frames
    )
    s_final = 3.14
    output_df = jax_populate_output_dataframe(results)
    print(f"DataFrame successfully converted to CSV")
'''