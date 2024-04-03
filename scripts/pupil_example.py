"""Example script for ibl-pupil dataset."""

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from eks.utils import convert_lp_dlc
from eks.pupil_smoother import ensemble_kalman_smoother_pupil
from scripts.general_scripts import handle_io


parser = argparse.ArgumentParser()
parser.add_argument(
    '--csv-dir',
    required=True,
    help='directory of models for ensembling',
    type=str
)
parser.add_argument(
    '--save-dir',
    help='save directory for outputs (default is csv-dir)',
    default=None,
    type=str,
)
parser.add_argument(
    '--diameter-s',
    help='smoothing parameter for diameter (closer to 1 = more smoothing)',
    default=.9999,
    type=float
)
parser.add_argument(
    '--com-s',
    help='smoothing parameter for center of mass (closer to 1 = more smoothing)',
    default=.999,
    type=float
)
args = parser.parse_args()

# collect user-provided args
csv_dir = os.path.abspath(args.csv_dir)
save_dir = args.save_dir


# ---------------------------------------------
# run EKS algorithm
# ---------------------------------------------

# handle I/O
save_dir = handle_io(csv_dir, save_dir)

# load files and put them in correct format
csv_files = os.listdir(csv_dir)
markers_list = []
for csv_file in csv_files:
    if not csv_file.endswith('csv'):
        continue
    markers_curr = pd.read_csv(os.path.join(csv_dir, csv_file), header=[0, 1, 2], index_col=0)
    keypoint_names = [c[1] for c in markers_curr.columns[::3]]
    model_name = markers_curr.columns[0][0]
    markers_curr_fmt = convert_lp_dlc(markers_curr, keypoint_names, model_name=model_name)
    markers_list.append(markers_curr_fmt)
if len(markers_list) == 0:
    raise FileNotFoundError(f'No marker csv files found in {csv_dir}')

# parameters hand-picked for smoothing purposes (diameter_s, com_s, com_s)
state_transition_matrix = np.asarray([
    [args.diameter_s, 0, 0],
    [0, args.com_s, 0],
    [0, 0, args.com_s]
])
print(f'Smoothing matrix: {state_transition_matrix}')

# run eks
df_dicts = ensemble_kalman_smoother_pupil(
    markers_list=markers_list,
    keypoint_names=keypoint_names,
    tracker_name='ensemble-kalman_tracker',
    state_transition_matrix=state_transition_matrix,
)

save_file = os.path.join(save_dir, 'kalman_smoothed_pupil_traces.csv')
print(f'saving smoothed predictions to {save_file }')
df_dicts['markers_df'].to_csv(save_file)

save_file = os.path.join(save_dir, 'kalman_smoothed_latents.csv')
print(f'saving latents to {save_file}')
df_dicts['latents_df'].to_csv(save_file)


# ---------------------------------------------
# plot results
# ---------------------------------------------

# select example keypoint
kp = keypoint_names[0]
idxs = (0, 500)

fig, axes = plt.subplots(4, 1, figsize=(9, 8))

for ax, coord in zip(axes, ['x', 'y', 'likelihood', 'zscore']):
    # plot individual models
    ax.set_ylabel(coord, fontsize=12)
    if coord == 'zscore':
        ax.plot(
        df_dicts['markers_df'].loc[slice(*idxs), ('ensemble-kalman_tracker', f'{kp}', coord)],
        color='k', linewidth=2)
        ax.set_xlabel('Time (frames)', fontsize=12)
        continue
    for m, markers_curr in enumerate(markers_list):
        ax.plot(
            markers_curr.loc[slice(*idxs), f'{kp}_{coord}'], color=[0.5, 0.5, 0.5],
            label='Individual models' if m == 0 else None,
        )
    # plot eks
    if coord == 'likelihood':
        continue
    ax.plot(
        df_dicts['markers_df'].loc[slice(*idxs), ('ensemble-kalman_tracker', kp, coord)],
        color='k', linewidth=2, label='EKS',
    )
    if coord == 'x':
        ax.legend()

plt.suptitle(f'EKS results for {kp}', fontsize=14)
plt.tight_layout()

save_file = os.path.join(save_dir, 'example_pupil_eks_result.pdf')
plt.savefig(save_file)
plt.close()
print(f'see example EKS output at {save_file}')
