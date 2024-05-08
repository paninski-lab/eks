"""Example script for single-camera datasets."""
import os

from general_scripting import handle_io, handle_parse_args
from eks.utils import format_data, populate_output_dataframe, plot_results
from eks.singleview_smoother import ensemble_kalman_smoother_single_view


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

# Load and format input files and prepare an empty DataFrame for output.
input_dfs_list, output_df, _ = format_data(args.input_dir, data_type)

# loop over keypoints; apply eks to each individually
for i, keypoint in enumerate(bodypart_list):
    # run eks
    keypoint_df_dict, s_final, nll_values = ensemble_kalman_smoother_single_view(
        input_dfs_list,
        keypoint,
        s[i],
        s_frames
    )
    keypoint_df = keypoint_df_dict[keypoint + '_df']

    # put results into new dataframe
    output_df = populate_output_dataframe(keypoint_df, keypoint, output_df)
    print(f"DataFrame successfully converted to CSV")

# save eks results
save_filename = save_filename or f'{smoother_type}_{s_final}.csv'
output_df.to_csv(os.path.join(save_dir, save_filename))

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