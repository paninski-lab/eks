import argparse
import os
import re

# -------------------------------------------------------------
""" Collection of General Functions for EKS Scripting
The functions here are called by individual example scripts """
# -------------------------------------------------------------


# ---------------------------------------------
# Command Line Arguments and File I/O
# ---------------------------------------------

# Finds + returns save directory if specified, otherwise defaults to outputs
def handle_io(input_dir, save_dir):
    if not os.path.isdir(input_dir):
        raise ValueError('--input-dir must be a valid directory containing prediction files')
    if save_dir is None:
        save_dir = os.path.join(os.getcwd(), 'outputs')
        os.makedirs(save_dir, exist_ok=True)
    return save_dir


# Handles extraction of arguments from command-line flags
def handle_parse_args(script_type):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input-dir',
        required=True,
        help='directory of model prediction csv files',
        type=str,
    )
    parser.add_argument(
        '--save-dir',
        help='save directory for outputs (default is input-dir)',
        default=None,
        type=str,
    )
    parser.add_argument(
        '--save-filename',
        help='filename for outputs (default uses smoother type and s parameter)',
        default=None,
        type=str,
    )
    parser.add_argument(
        '--data-type',
        help='format of input data (Lightning Pose = lp, SLEAP = slp), dlc by default.',
        default='lp',
        type=str,
    )
    parser.add_argument(
        '--s-frames',
        help='frames to be considered for smoothing '
             'parameter optimization, first 2k frames by default. Moot if --s is specified. '
             'Format: "[(start_int, end_int), (start_int, end_int), ... ]" or int. '
             'Inputting a single int uses all frames from 1 to the int. '
             '(None, end_int) starts from first frame; (start_int, None) proceeds to last frame.',
        default=[(None, 10000)],
        type=parse_s_frames,
    )
    parser.add_argument(
        '--blocks',
        help='keypoints to be blocked for correlated noise. Generates on smoothing param per '
             'block, as opposed to per keypoint. Specified by the form "x1, x2, x3; y1, y2"'
             ' referring to keypoint indices (starting at 0)',
        default=[],
        type=parse_blocks,
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


# Helper function for parsing s-frames
def parse_s_frames(input_string):
    try:
        # First, check if the input is a single integer
        if input_string.isdigit():
            end = int(input_string)
            return [(1, end)]  # Handle as from first to 'end'

        # Remove spaces, replace with nothing
        cleaned = re.sub(r'\s+', '', input_string)
        # Match tuples in the form of (x,ys), (x,), (,ys)
        tuple_pattern = re.compile(r'\((\d*),(\d*)\)')
        matches = tuple_pattern.findall(cleaned)

        if not matches:
            raise ValueError("No valid tuples found.")

        tuples = []
        for start, end in matches:
            # Convert numbers to integers or None if empty
            start = int(start) if start else None
            end = int(end) if end else None
            if start is not None and end is not None and start > end:
                raise ValueError("Start index cannot be greater than end index.")
            tuples.append((start, end))

        return tuples
    except Exception as e:
        raise argparse.ArgumentTypeError(f"Invalid format for --s-frames: {e}")


# Helper function for parsing blocks
def parse_blocks(blocks_str):
    try:
        # Split the input string by ';' to separate each block
        blocks = blocks_str.split(';')
        # Split each block by ',' to get individual integers and convert to lists of integers
        parsed_blocks = [list(map(int, block.split(','))) for block in blocks]
        return parsed_blocks
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Invalid format for --blocks: {blocks_str}. Error: {e}")


# --------------------------------------
# Helper Functions for handle_parse_args
# --------------------------------------


def add_bodyparts(parser):
    parser.add_argument(
        '--bodypart-list',
        nargs='+',
        help='the list of body parts to be ensembled and smoothed. If not specified, uses all.',
    )
    return parser


def add_s(parser):
    parser.add_argument(
        '--s',
        help='Specifying a smoothing parameter overrides the auto-tuning function. '
             'Providing multiple args will set each additional bodypart to the next s param',
        nargs='+',
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
        type=float
    )
    return parser


def add_com_s(parser):
    parser.add_argument(
        '--com-s',
        help='smoothing parameter for center of mass (closer to 1 = more smoothing)',
        type=float
    )
    return parser
