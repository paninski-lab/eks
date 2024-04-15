import os
import argparse
import pandas as pd
import numpy as np

from smoothers.utils import convert_lp_dlc

# -------------------------------------------------------------
""" Collection of General Functions for EKS Scripting
The functions here are called by individual example scripts """
# -------------------------------------------------------------


# ---------------------------------------------
# Command Line Arguments and File I/O
# ---------------------------------------------

def handle_io(csv_dir, save_dir): # Finds + returns save directory if specified, otherwise defaults to outputs\
    if not os.path.isdir(csv_dir):
        raise ValueError('--csv-dir must be a valid directory containing prediction csv files')
    if save_dir is None:
        save_dir = os.path.join(os.getcwd(), 'outputs')
        os.makedirs(save_dir, exist_ok=True)
    return save_dir


def handle_parse_args(script_type):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--csv-dir',
        required=True,
        help='directory of model prediction csv files',
        type=str,
    )
    parser.add_argument(
        '--save-dir',
        help='save directory for outputs (default is csv-dir)',
        default=None,
        type=str,
    )
    parser.add_argument(
        '--save-filename',
        help='filename for outputs (default uses smoother type and s parameter)',
        default=None,
        type=str,
    )
    if script_type == 'singlecam':
        add_bodyparts(parser)
        add_s(parser)
    elif script_type == 'multicam':
        add_bodyparts(parser)
        add_camera_names(parser)
        add_quantile_keep_pca(parser)
        add_s(parser)
    elif script_type == 'pupil':
        add_diameter_s(parser)
        add_com_s(parser)
    elif script_type == 'paw':
        add_s(parser)
        add_quantile_keep_pca(parser)
    else:
        raise ValueError("Unrecognized script type.")
    args = parser.parse_args()
    return args


# Helper Functions for handle_parse_args
def add_bodyparts(parser):
    parser.add_argument(
        '--bodypart-list',
        required=True,
        nargs='+',
        help='the list of body parts to be ensembled and smoothed',
    )
    return parser


def add_s(parser):
    parser.add_argument(
        '--s',
        help='smoothing parameter ranges from .01-20 (smaller values = more smoothing)',
        type=float,
    )
    return parser


def add_camera_names(parser):
    parser.add_argument(
        '--camera-names',
        required=True,
        nargs='+',
        help='the camera names',
    )
    return parser


def add_quantile_keep_pca(parser):
    parser.add_argument(
        '--quantile_keep_pca',
        help='percentage of the points are kept for multi-view PCA (lowest ensemble variance)',
        default=25,
        type=float,
    )
    return parser


def add_diameter_s(parser):
    parser.add_argument(
        '--diameter-s',
        help='smoothing parameter for diameter (closer to 1 = more smoothing)',
        default=.9999,
        type=float
    )
    return parser


def add_com_s(parser):
    parser.add_argument(
        '--com-s',
        help='smoothing parameter for center of mass (closer to 1 = more smoothing)',
        default=.999,
        type=float
    )
    return parser


# ---------------------------------------------
# Loading + Formatting CSV<->DataFrame
# ---------------------------------------------

def format_csv(csv_dir, data_type='lp'):
    csv_files = os.listdir(csv_dir)
    markers_list = []

    # Extracting markers from data. Applies correct format conversion and stores each file's markers in a list
    for csv_file in csv_files:
        if not csv_file.endswith('csv'):
            continue
        markers_curr = pd.read_csv(os.path.join(csv_dir, csv_file), header=[0, 1, 2], index_col=0)
        keypoint_names = [c[1] for c in markers_curr.columns[::3]]
        model_name = markers_curr.columns[0][0]
        if data_type == 'lp':
            markers_curr_fmt = convert_lp_dlc(markers_curr, keypoint_names, model_name=model_name)
        else:
            markers_curr_fmt = markers_curr

        markers_list.append(markers_curr_fmt)

    if len(markers_list) == 0:
        raise FileNotFoundError(f'No marker csv files found in {csv_dir}')

    markers_eks = make_output_dataframe(markers_curr)

    return markers_list, markers_eks  # returns both the formatted marker data and the empty dataframe for EKS output


# Making empty DataFrame for EKS output
def make_output_dataframe(markers_curr):
    markers_eks = markers_curr.copy()
    markers_eks.columns = markers_eks.columns.set_levels(['ensemble-kalman_tracker'], level=0)
    for col in markers_eks.columns:
        if col[-1] == 'likelihood':
            # set this to 1.0 so downstream filtering functions don't get
            # tripped up
            markers_eks[col].values[:] = 1.0
        else:
            markers_eks[col].values[:] = np.nan

    return markers_eks


def populate_output_dataframe(keypoint_df, keypoint_ensemble, markers_eks):
    for coord in ['x', 'y', 'zscore']:
        src_cols = ('ensemble-kalman_tracker', f'{keypoint_ensemble}', coord)
        dst_cols = ('ensemble-kalman_tracker', f'{keypoint_ensemble}', coord)
        markers_eks.loc[:, dst_cols] = keypoint_df.loc[:, src_cols]
    return markers_eks


