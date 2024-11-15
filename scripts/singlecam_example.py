"""Example script for single-camera datasets."""
import os
from eks.command_line_args import handle_io, handle_parse_args
from eks.singlecam_smoother import fit_eks_singlecam
from eks.utils import plot_results

# Collect User-Provided Args
smoother_type = 'singlecam'
args = handle_parse_args(smoother_type)
input_source = args.input_dir if isinstance(args.input_dir, str) else args.input_files
data_type = args.data_type  # Note: LP and DLC are .csv, SLP is .slp
# Determine the input directory path
if isinstance(input_source, str):
    input_dir = os.path.abspath(input_source)
else:
    input_dir = os.path.abspath(os.path.dirname(input_source[0]))
save_filename = args.save_filename
# Set up the save directory
save_dir = handle_io(input_dir, args.save_dir)
bodypart_list = args.bodypart_list
s = args.s  # Defaults to automatic optimization
s_frames = args.s_frames  # Frames to be used for automatic optimization if s is not provided
blocks = args.blocks
verbose = True if args.verbose == 'True' else False

# Fit EKS using the provided input data
output_df, s_finals, input_dfs, bodypart_list = fit_eks_singlecam(
    input_source=input_source,
    data_type=data_type,
    save_dir=save_dir,
    save_filename=save_filename,
    bodypart_list=bodypart_list,
    s=s,
    s_frames=s_frames,
    blocks=blocks,
    verbose=verbose
)

# Plot results for a specific keypoint (default to last keypoint)
keypoint_i = -1
plot_results(
    output_df=output_df,
    input_dfs_list=input_dfs,
    key=f'{bodypart_list[keypoint_i]}',
    idxs=(0, 500),
    s_final=s_finals[keypoint_i],
    nll_values=None,
    save_dir=save_dir,
    smoother_type=smoother_type
)

print("Ensemble Kalman Smoothing complete. Results saved and plotted successfully.")
