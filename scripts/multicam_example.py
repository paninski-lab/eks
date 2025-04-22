"""Example script for multi-camera datasets."""

import os

from eks.command_line_args import handle_io, handle_parse_args
from eks.multicam_smoother import fit_eks_multicam
from eks.utils import plot_results

smoother_type = 'multicam'

# Collect User-Provided Args
args = handle_parse_args(smoother_type)
input_source = args.input_dir if isinstance(args.input_dir, str) else args.input_files
# Determine the input directory path
if isinstance(input_source, str):
    input_dir = os.path.abspath(input_source)
else:
    input_dir = os.path.abspath(os.path.dirname(input_source[0]))
# Set up the save directory
save_filename = args.save_filename
save_dir = handle_io(input_dir, args.save_dir)
bodypart_list = args.bodypart_list
s = args.s  # Defaults to automatic optimization
s_frames = args.s_frames  # Frames to be used for automatic optimization if s is not provided
camera_names = args.camera_names
quantile_keep_pca = args.quantile_keep_pca
verbose = True if args.verbose == 'True' else False
inflate_vars = True if args.inflate_vars == 'True' else False
n_latent = args.n_latent

# Fit EKS using the provided input data
camera_dfs, s_finals, input_dfs, bodypart_list = fit_eks_multicam(
    input_source=input_source,
    save_dir=save_dir,
    bodypart_list=bodypart_list,
    smooth_param=s,
    s_frames=s_frames,
    camera_names=camera_names,
    quantile_keep_pca=quantile_keep_pca,
    verbose=verbose,
    inflate_vars=inflate_vars,
    n_latent=args.n_latent
)


# Plot results for a specific keypoint (default to last keypoint of last camera view)
keypoint_i = -1
camera_c = -1
plot_results(
    output_df=camera_dfs[camera_c],
    input_dfs_list=input_dfs[camera_c],
    key=f'{bodypart_list[keypoint_i]}',
    idxs=(0, 500),
    s_final=s_finals[keypoint_i],
    nll_values=None,
    save_dir=save_dir,
    smoother_type=smoother_type,
)

print("Ensemble Kalman Smoothing complete. Results saved and plotted successfully.")
