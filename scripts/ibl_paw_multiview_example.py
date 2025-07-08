"""Example script for ibl-paw dataset."""
import os

from eks.command_line_args import handle_io, handle_parse_args
from eks.ibl_paw_multicam_smoother import fit_eks_multicam_ibl_paw
from eks.utils import plot_results

smoother_type = 'ibl_paw'

# Collect User-Provided Args
args = handle_parse_args(smoother_type)
input_source = args.input_dir if isinstance(args.input_dir, str) else args.input_files
# Determine the input directory path
if isinstance(input_source, str):
    input_dir = os.path.abspath(input_source)
else:
    input_dir = os.path.abspath(os.path.dirname(input_source[0]))
save_filename = args.save_filename
save_dir = handle_io(input_dir, args.save_dir)
s = args.s
s_frames = args.s_frames  # Frames to be used for automatic optimization if s is not provided
quantile_keep_pca = args.quantile_keep_pca
inflate_vars = True if args.inflate_vars == 'True' else False
n_latent = args.n_latent
verbose = True if args.verbose == 'True' else False

# Fit EKS using the provided input data
camera_dfs, s_finals, input_dfs, bodypart_list = fit_eks_multicam_ibl_paw(
    input_source=input_source,
    save_dir=save_dir,
    smooth_param=s,
    s_frames=s_frames,
    quantile_keep_pca=quantile_keep_pca,
    var_mode='var',
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
    coords=['x', 'y']
)

print("Ensemble Kalman Smoothing complete. Results saved and plotted successfully.")
