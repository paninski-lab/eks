"""Example script for multi-camera datasets."""
import os

from general_scripting import handle_io, handle_parse_args
from eks.utils import format_data, populate_output_dataframe, plot_results
from eks.multiview_pca_smoother import ensemble_kalman_smoother_multi_cam



# Collect User-Provided Args
smoother_type = 'multicam'
args = handle_parse_args(smoother_type)
input_dir = os.path.abspath(args.input_dir)
data_type = args.data_type  # Note: LP and DLC are .csv, SLP is .slp
save_dir = handle_io(input_dir, args.save_dir)  # defaults to outputs\
save_filename = args.save_filename
bodypart_list = args.bodypart_list
s = args.s  # defaults to automatic optimization
s_frames = args.s_frames # frames to be used for automatic optimization (only if no --s flag)
camera_names = args.camera_names
quantile_keep_pca = args.quantile_keep_pca

# Load and format input files and prepare an empty DataFrame for output.
input_dfs_list, output_df, _ = format_data(input_dir, data_type)

# loop over keypoints; apply eks to each individually
# Note: all camera views must be stored in the same csv file
for keypoint_ensemble in bodypart_list:
    # Separate body part predictions by camera view
    marker_list_by_cam = [[] for _ in range(len(camera_names))]
    for markers_curr in input_dfs_list:
        for c, camera_name in enumerate(camera_names):
            non_likelihood_keys = [
                key for key in markers_curr.keys()
                if camera_names[c] in key and keypoint_ensemble in key
            ]
            marker_list_by_cam[c].append(markers_curr[non_likelihood_keys])

    # run eks
    cameras_df_dict, s_final, nll_values = ensemble_kalman_smoother_multi_cam(
        markers_list_cameras=marker_list_by_cam,
        keypoint_ensemble=keypoint_ensemble,
        smooth_param=s,
        quantile_keep_pca=quantile_keep_pca,
        camera_names=camera_names,
        s_frames=s_frames
    )

    # put results into new dataframe
    for camera in camera_names:
        cameras_df = cameras_df_dict[f'{camera}_df']
        populate_output_dataframe(cameras_df, keypoint_ensemble, output_df,
                                  key_suffix=f'_{camera}')

# save eks results
save_filename = save_filename or f'{smoother_type}_{s_final}.csv'
output_df.to_csv(os.path.join(save_dir, save_filename))

# plot results
plot_results(output_df=output_df,
             input_dfs_list=input_dfs_list,
             key=f'{bodypart_list[-1]}_{camera_names[0]}',
             idxs=(0, 500),
             s_final=s_final,
             nll_values=nll_values,
             save_dir=save_dir,
             smoother_type=smoother_type
             )
