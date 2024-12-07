"""Example script for ibl-pupil dataset."""
import os

from eks.command_line_args import handle_io, handle_parse_args
from eks.ibl_pupil_smoother import fit_eks_pupil
from eks.utils import format_data, plot_results

# Collect User-Provided Arguments
smoother_type = 'pupil'
args = handle_parse_args(smoother_type)

# Determine input source (directory or list of files)
input_source = args.input_dir if isinstance(args.input_dir, str) else args.input_files
data_type = args.data_type  # LP and DLC are .csv, SLP is .slp

# Set up the save directory
if isinstance(input_source, str):
    input_dir = os.path.abspath(input_source)
else:
    input_dir = os.path.abspath(os.path.dirname(input_source[0]))
save_dir = handle_io(input_dir, args.save_dir)
save_filename = args.save_filename

# Parameters for smoothing
diameter_s = args.diameter_s
com_s = args.com_s
s_frames = args.s_frames

# Run the smoothing function
df_dicts, smooth_params, input_dfs_list, keypoint_names, nll_values = fit_eks_pupil(
    input_source=input_source,
    data_type=data_type,
    save_dir=save_dir,
    smooth_params=[diameter_s, com_s],
    s_frames=s_frames
)

# Save the results
print("Saving smoothed predictions and latents...")
markers_save_file = os.path.join(save_dir, 'kalman_smoothed_pupil_traces.csv')
latents_save_file = os.path.join(save_dir, 'kalman_smoothed_latents.csv')
df_dicts['markers_df'].to_csv(markers_save_file)
print(f'Smoothed predictions saved to {markers_save_file}')
df_dicts['latents_df'].to_csv(latents_save_file)
print(f'Latents saved to {latents_save_file}')

# Plot results
plot_results(
    output_df=df_dicts['markers_df'],
    input_dfs_list=input_dfs_list,
    key=f'{keypoint_names[-1]}',
    idxs=(0, 500),
    s_final=(smooth_params[0], smooth_params[1]),
    nll_values=nll_values,
    save_dir=save_dir,
    smoother_type=smoother_type
)
