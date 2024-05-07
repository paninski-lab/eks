"""Example script for multi-camera datasets."""

import matplotlib.pyplot as plt
import os

from eks.multiview_pca_smoother import ensemble_kalman_smoother_multi_cam
from general_scripting import handle_io, handle_parse_args
from eks.utils import format_data

# collect user-provided args
smoother_type = 'multicam'
args = handle_parse_args(smoother_type)

input_dir = os.path.abspath(args.input_dir)

# Note: LP and DLC are .csv, SLP is .slp
data_type = args.data_type

# Find save directory if specified, otherwise defaults to outputs\
save_dir = handle_io(input_dir, args.save_dir)

bodypart_list = args.bodypart_list
camera_names = args.camera_names
num_cameras = len(camera_names)
quantile_keep_pca = args.quantile_keep_pca
s = args.s  # optional, defaults to automatic optimization
s_frames = args.s_frames

# Load and format input files and prepare an empty DataFrame for output.
# markers_list : list of input DataFrames
# markers_eks : empty DataFrame for EKS output
markers_list, markers_eks = format_data(input_dir, data_type)

# loop over keypoints; apply eks to each individually
for keypoint_ensemble in bodypart_list:
    # this structure assumes all camera views are stored in the same csv file
    # here we separate body part predictions by camera view
    marker_list_by_cam = [[] for _ in range(num_cameras)]
    for markers_curr in markers_list:
        for c, camera_name in enumerate(camera_names):
            non_likelihood_keys = [
                key for key in markers_curr.keys()
                if camera_names[c] in key and keypoint_ensemble in key
            ]
            marker_list_by_cam[c].append(markers_curr[non_likelihood_keys])
    # run eks
    cameras_df, s_final, nll_values = ensemble_kalman_smoother_multi_cam(
        markers_list_cameras=marker_list_by_cam,
        keypoint_ensemble=keypoint_ensemble,
        smooth_param=s,
        quantile_keep_pca=quantile_keep_pca,
        camera_names=camera_names,
        s_frames=s_frames
    )
    # put results into new dataframe
    for camera in camera_names:
        df_tmp = cameras_df[f'{camera}_df']
        for coord in ['x', 'y', 'zscore']:
            src_cols = ('ensemble-kalman_tracker', f'{keypoint_ensemble}', coord)
            dst_cols = ('ensemble-kalman_tracker', f'{keypoint_ensemble}_{camera}', coord)
            markers_eks.loc[:, dst_cols] = df_tmp.loc[:, src_cols]

# save eks results
markers_eks.to_csv(os.path.join(save_dir, 'eks.csv'))

# ---------------------------------------------
# plot results
# ---------------------------------------------

# select example keypoint from example camera view
kp = bodypart_list[0]
cam = camera_names[0]
idxs = (0, 500)

fig, axes = plt.subplots(4, 1, figsize=(9, 6))

for ax, coord in zip(axes, ['x', 'y', 'likelihood', 'zscore']):
    # plot individual models
    ax.set_ylabel(coord, fontsize=12)
    if coord == 'zscore':
        ax.plot(
            markers_eks.loc[slice(*idxs), ('ensemble-kalman_tracker', f'{kp}_{cam}', coord)],
            color=[0.5, 0.5, 0.5])
        ax.set_xlabel('Time (frames)', fontsize=12)
        continue
    for m, markers_curr in enumerate(markers_list):
        ax.plot(
            markers_curr.loc[slice(*idxs), f'{kp}_{cam}_{coord}'], color=[0.5, 0.5, 0.5],
            label='Individual models' if m == 0 else None,
        )
    # plot eks
    if coord == 'likelihood':
        continue
    ax.plot(
        markers_eks.loc[slice(*idxs), ('ensemble-kalman_tracker', f'{kp}_{cam}', coord)],
        color='k', linewidth=2, label='EKS',
    )
    if coord == 'x':
        ax.legend()

plt.suptitle(f'EKS results for {kp} ({cam} view), smoothing = {s_final}', fontsize=14)
plt.tight_layout()

save_file = os.path.join(save_dir, 'example_multicam_eks_result.pdf')
plt.savefig(save_file)
plt.close()
print(f'see example EKS output at {save_file}')
