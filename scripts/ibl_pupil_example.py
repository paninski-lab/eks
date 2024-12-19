"""Example script for ibl-pupil dataset."""

import os

from eks.command_line_args import handle_io, handle_parse_args
from eks.ibl_pupil_smoother import fit_eks_pupil
from eks.utils import plot_results


smoother_type = 'ibl_pupil'

# Collect User-Provided Arguments
args = handle_parse_args(smoother_type)

# Determine input source (directory or list of files)
input_source = args.input_dir if isinstance(args.input_dir, str) else args.input_files

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
df_smoothed, smooth_params, input_dfs_list, keypoint_names, nll_values = fit_eks_pupil(
    input_source=input_source,
    save_file=os.path.join(save_dir, save_filename or 'eks_pupil.csv'),
    smooth_params=[diameter_s, com_s],
    s_frames=s_frames,
)

# Plot results
plot_results(
    output_df=df_smoothed,
    input_dfs_list=input_dfs_list,
    key=f'{keypoint_names[-1]}',
    idxs=(0, 500),
    s_final=(smooth_params[0], smooth_params[1]),
    nll_values=nll_values,
    save_dir=save_dir,
    smoother_type=smoother_type
)
