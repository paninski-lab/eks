"""Example script for ibl-pupil dataset."""
import os

from eks.command_line_args import handle_io, handle_parse_args
from eks.utils import format_data, plot_results
from eks.pupil_smoother import ensemble_kalman_smoother_pupil

# Collect User-Provided Args
smoother_type = 'pupil'
args = handle_parse_args(smoother_type)
input_dir = os.path.abspath(args.input_dir)
data_type = args.data_type  # Note: LP and DLC are .csv, SLP is .slp
save_dir = handle_io(input_dir, args.save_dir)  # defaults to outputs\
save_filename = args.save_filename
diameter_s = args.diameter_s  # defaults to automatic optimization
com_s = args.com_s  # defaults to automatic optimization
s_frames = args.s_frames # frames to be used for automatic optimization (only if no --s flag)

# Load and format input files and prepare an empty DataFrame for output.
input_dfs_list, output_df, keypoint_names = format_data(input_dir, data_type)

# run eks
df_dicts, smooth_params, nll_values = ensemble_kalman_smoother_pupil(
    markers_list=input_dfs_list,
    keypoint_names=keypoint_names,
    tracker_name='ensemble-kalman_tracker',
    smooth_params=[diameter_s, com_s],
    s_frames=s_frames
    )

save_file = os.path.join(save_dir, 'kalman_smoothed_pupil_traces.csv')
print(f'saving smoothed predictions to {save_file}')
df_dicts['markers_df'].to_csv(save_file)

save_file = os.path.join(save_dir, 'kalman_smoothed_latents.csv')
print(f'saving latents to {save_file}')
df_dicts['latents_df'].to_csv(save_file)


# ---------------------------------------------
# plot results
# ---------------------------------------------

# plot results
plot_results(output_df=df_dicts['markers_df'],
             input_dfs_list=input_dfs_list,
             key=f'{keypoint_names[-1]}',
             idxs=(0, 500),
             s_final=(smooth_params[0], smooth_params[1]),
             nll_values=nll_values,
             save_dir=save_dir,
             smoother_type=smoother_type
             )
