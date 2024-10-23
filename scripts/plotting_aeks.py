import copy
import os
import sys

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../../tracking-diagnostics')))

from diagnostics.video import get_frames_from_idxs

from eks.utils import convert_lp_dlc, format_data


def format_data(ensemble_dir):
    input_files = os.listdir(ensemble_dir)
    markers_list = []
    for input_file in input_files:
        markers_curr = pd.read_csv(
            os.path.join(ensemble_dir, input_file), header=[0, 1, 2], index_col=0)
        keypoint_names = [c[1] for c in markers_curr.columns[::3]]
        model_name = markers_curr.columns[0][0]
        markers_curr_fmt = convert_lp_dlc(
            markers_curr, keypoint_names, model_name=model_name)
        markers_curr_fmt.to_csv('fmt_input.csv', index=False)
        markers_list.append(markers_curr_fmt)
    return markers_list


import os
import subprocess


def save_video(save_file, tmp_dir, framerate, frame_pattern='frame_%06d.jpeg'):
    call_str = f'ffmpeg -r {framerate} -i {os.path.join(tmp_dir, frame_pattern)} -c:v libx264 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" {save_file}'

    if os.name == 'nt':  # If the OS is Windows
        subprocess.run(['ffmpeg', '-r', str(framerate), '-i', f'{tmp_dir}/frame_%06d.jpeg',
                        '-c:v', 'libx264', '-vf', "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                        save_file],
                       check=True)
    else:  # If the OS is Unix/Linux
        subprocess.run(['/bin/bash', '-c', call_str], check=True)


# load eks
eks_path = f'/eks/outputs/eks_test_vid.csv'
markers_curr = pd.read_csv(eks_path, header=[0, 1, 2], index_col=0)
keypoint_names = [c[1] for c in markers_curr.columns[::3]]
model_name = markers_curr.columns[0][0]
eks_pd = convert_lp_dlc(markers_curr, keypoint_names, model_name)

# load aeks
eks_path = f'/eks/outputs/aeks_test_vid.csv'
markers_curr = pd.read_csv(eks_path, header=[0, 1, 2], index_col=0)
keypoint_names = [c[1] for c in markers_curr.columns[::3]]
model_name = markers_curr.columns[0][0]
eks_pd2 = convert_lp_dlc(markers_curr, keypoint_names, model_name)

# load ensembles
ensemble_dir = f'/eks/data/mirror-mouse-aeks/expanded-networks'
ensemble_pd_list = format_data(ensemble_dir)
animal_ids = [1]
body_parts = ['paw1LH_top', 'paw2LF_top', 'paw3RF_top', 'paw4RH_top', 'tailBase_top',
              'tailMid_top', 'nose_top', 'obs_top',
              'paw1LH_bot', 'paw2LF_bot', 'paw3RF_bot', 'paw4RH_bot', 'tailBase_bot',
              'tailMid_bot', 'nose_bot', 'obsHigh_bot', 'obsLow_bot'
              ]
to_plot = []
for animal_id in animal_ids:
    for body_part in body_parts:
        to_plot.append(body_part)

save_path = '/eks/videos'
video_name = 'test_vid.mp4'
video_path = f'/eks/videos/{video_name}'
cap = cv2.VideoCapture(video_path)

start_frame = 0
frame_idxs = None
n_frames = 993
idxs = np.arange(start_frame, start_frame + n_frames)
framerate = 20


def plot_video_markers(markers_pd, ax, n, body_part, color, alphas, markers, model_id=0,
                       markersize=8):
    x_key = body_part + '_x'
    y_key = body_part + '_y'
    markers_x = markers_pd[x_key][n]
    markers_y = markers_pd[y_key][n]
    ax.scatter(markers_x, markers_y, alpha=alphas[model_id], marker="o", color=color)


colors = ['cyan', 'pink', 'purple']
alphas = [.8] * len(ensemble_pd_list) + [1.0]
markers = ['.'] * len(ensemble_pd_list) + ['x']
model_labels = ['expanded-network rng0', 'eks', 'aeks']
model_colors = colors
fr = 60

for body_part in to_plot:
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    tmp_dir = os.path.join(save_path, f'tmp_{body_part}')
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    save_file = os.path.join(save_path, f'test_vid_{body_part}.mp4')

    txt_fr_kwargs = {
        'fontsize': 14, 'color': [1, 1, 1], 'horizontalalignment': 'left',
        'verticalalignment': 'top', 'fontname': 'monospace',
        'bbox': dict(facecolor='k', alpha=0.25, edgecolor='none'),
        'transform': ax.transAxes
    }
    save_imgs = True
    if save_imgs:
        markersize = 18
    else:
        markersize = 12
    for idx in tqdm(range(len(idxs))):
        n = idxs[idx]
        ax.clear()
        frame = get_frames_from_idxs(cap, [n])
        ax.imshow(frame[0, 0], vmin=0, vmax=255, cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])
        patches = []
        # ensemble
        for model_id, markers_pd in enumerate(ensemble_pd_list):
            markers_pd_copy = copy.deepcopy(markers_pd)
            plot_video_markers(markers_pd_copy, ax, n, body_part, colors[0], alphas, markers,
                               model_id=model_id, markersize=markersize)
        # eks_ind
        for model_id, markers_pd in enumerate([eks_pd]):
            markers_pd_copy = copy.deepcopy(markers_pd)
            plot_video_markers(markers_pd_copy, ax, n, body_part, colors[1], alphas, markers,
                               model_id=model_id, markersize=markersize)
        # eks_cdnm
        for model_id, markers_pd in enumerate([eks_pd2]):
            markers_pd_copy = copy.deepcopy(markers_pd)
            plot_video_markers(markers_pd_copy, ax, n, body_part, colors[2], alphas, markers,
                               model_id=model_id, markersize=markersize)
        # legend
        for i, model_label in enumerate(model_labels):
            patches.append(mpatches.Patch(color=model_colors[i], label=model_label))
        ax.legend(handles=patches, prop={'size': 12}, loc='upper right')
        im = ax.text(0.02, 0.98, f'frame {n}', **txt_fr_kwargs)
        plt.savefig(os.path.join(tmp_dir, 'frame_%06d.jpeg' % idx))
    save_video(save_file, tmp_dir, framerate, frame_pattern='frame_%06d.jpeg')
    # Clean up temporary directory
    for file in os.listdir(tmp_dir):
        os.remove(os.path.join(tmp_dir, file))
    os.rmdir(tmp_dir)
