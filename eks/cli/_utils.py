"""Shared utilities for eks CLI subcommands."""

import argparse
import re
from pathlib import Path


def handle_io(input_dir: str | Path, save_dir: str | Path | None) -> Path:
    """Validate input directory and resolve save directory.

    Args:
        input_dir: Path to directory containing prediction CSV files.
        save_dir: Path to save output files. Defaults to ./outputs if None.

    Returns:
        Resolved path to the save directory.

    Raises:
        ValueError: If input_dir is not a valid directory.
    """
    if not Path(input_dir).is_dir():
        raise ValueError('--input-dir must be a valid directory containing prediction files')
    if save_dir is None:
        save_dir = Path.cwd() / 'outputs'
        save_dir.mkdir(parents=True, exist_ok=True)
    return Path(save_dir)


def parse_s_frames(input_string: str) -> list[tuple[int | None, int | None]]:
    """Parse frame range specifications for smoothing parameter optimization.

    Args:
        input_string: Frame range string, e.g. '[(0,100),(200,300)]'.

    Returns:
        List of (start, end) tuples with int or None values.

    Raises:
        argparse.ArgumentTypeError: If the format is invalid.
    """
    try:
        if input_string.isdigit():
            return [(1, int(input_string))]

        cleaned = re.sub(r'\s+', '', input_string)
        matches = re.compile(r'\((\d*),(\d*)\)').findall(cleaned)

        if not matches:
            raise ValueError('no valid tuples found')

        tuples = []
        for start, end in matches:
            start = int(start) if start else None
            end = int(end) if end else None
            if start is not None and end is not None and start > end:
                raise ValueError('start index cannot be greater than end index')
            tuples.append((start, end))

        return tuples
    except Exception as e:
        raise argparse.ArgumentTypeError(f'invalid format for --s-frames: {e}')


def parse_blocks(blocks_str: str) -> list[list[int]]:
    """Parse keypoint block specifications for correlated noise.

    Args:
        blocks_str: Block string, e.g. '0,1,2;3,4'.

    Returns:
        List of lists of keypoint indices.

    Raises:
        argparse.ArgumentTypeError: If the format is invalid.
    """
    try:
        return [list(map(int, block.split(','))) for block in blocks_str.split(';')]
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f'invalid format for --blocks: {blocks_str}. Error: {e}'
        )


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add arguments shared by all subcommands.

    Args:
        parser: The argument parser to add arguments to.

    Returns:
        The modified parser.
    """
    parser.add_argument(
        '--input-dir',
        help='directory of model prediction csv files',
        type=str,
    )
    parser.add_argument(
        '--input-files',
        help='list of model prediction csv files from various directories',
        nargs='+',
    )
    parser.add_argument(
        '--save-dir',
        help='save directory for outputs (default: ./outputs)',
        default=None,
        type=str,
    )
    parser.add_argument(
        '--save-filename',
        help='filename for outputs (default uses smoother type)',
        default=None,
        type=str,
    )
    parser.add_argument(
        '--s-frames',
        help=(
            'frames considered for smoothing parameter optimization; moot if --s is specified. '
            'format: "[(start_int,end_int),(start_int,end_int),...]". '
            '(None,end_int) starts from first frame; (start_int,None) proceeds to last frame.'
        ),
        default=None,
        type=parse_s_frames,
    )
    parser.add_argument(
        '--blocks',
        help=(
            'keypoints grouped for correlated noise, yielding one smoothing param per block '
            'rather than per keypoint. format: "x1,x2,x3;y1,y2" (keypoint indices from 0)'
        ),
        default=[],
        type=parse_blocks,
    )
    parser.add_argument(
        '--verbose',
        help='display smoothing parameter optimization iterations',
        action='store_true',
    )
    parser.add_argument(
        '--make-plot',
        help='generate and save diagnostic plots after smoothing',
        action='store_true',
    )
    return parser


def add_bodyparts(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add --bodypart-list argument to a parser.

    Args:
        parser: The argument parser to add the argument to.

    Returns:
        The modified parser.
    """
    parser.add_argument(
        '--bodypart-list',
        nargs='+',
        help='body parts to ensemble and smooth; uses all if not specified',
    )
    return parser


def add_s(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add --s argument to a parser.

    Args:
        parser: The argument parser to add the argument to.

    Returns:
        The modified parser.
    """
    parser.add_argument(
        '--s',
        help=(
            'smoothing parameter; overrides auto-tuning when specified. '
            'multiple values assign one per bodypart in order'
        ),
        nargs='+',
        type=float,
    )
    return parser


def add_camera_names(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add --camera-names argument to a parser.

    Args:
        parser: The argument parser to add the argument to.

    Returns:
        The modified parser.
    """
    parser.add_argument(
        '--camera-names',
        required=False,
        nargs='+',
        help=(
            'camera names corresponding to input files; each name must appear as a substring '
            'of the matching filenames. required for multicam without --calibration and for '
            'mirrored-multicam; ignored when --calibration is provided'
        ),
    )
    return parser


def add_quantile_keep_pca(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add --quantile-keep-pca argument to a parser.

    Args:
        parser: The argument parser to add the argument to.

    Returns:
        The modified parser.
    """
    parser.add_argument(
        '--quantile-keep-pca',
        help='percentage of points kept for multi-view PCA (lowest ensemble variance)',
        default=95,
        type=float,
    )
    return parser


def add_inflate_vars(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add --inflate-vars argument to a parser.

    Args:
        parser: The argument parser to add the argument to.

    Returns:
        The modified parser.
    """
    parser.add_argument(
        '--no-inflate-vars',
        dest='inflate_vars',
        action='store_false',
        default=True,
        help='disable Mahalanobis distance-based variance inflation (enabled by default)',
    )
    return parser


def add_n_latent(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add --n-latent argument to a parser.

    Args:
        parser: The argument parser to add the argument to.

    Returns:
        The modified parser.
    """
    parser.add_argument(
        '--n-latent',
        help='number of latent PCA dimensions to retain',
        default=3,
        type=int,
    )
    return parser


def add_calibration(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add --calibration argument to a parser.

    Args:
        parser: The argument parser to add the argument to.

    Returns:
        The modified parser.
    """
    parser.add_argument(
        '--calibration',
        help='path to calibration.toml file',
        default=None,
        type=str,
    )
    return parser


def add_diameter_s(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add --diameter-s argument to a parser.

    Args:
        parser: The argument parser to add the argument to.

    Returns:
        The modified parser.
    """
    parser.add_argument(
        '--diameter-s',
        help='smoothing parameter for pupil diameter (closer to 1 = more smoothing)',
        type=float,
    )
    return parser


def add_com_s(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add --com-s argument to a parser.

    Args:
        parser: The argument parser to add the argument to.

    Returns:
        The modified parser.
    """
    parser.add_argument(
        '--com-s',
        help='smoothing parameter for pupil center of mass (closer to 1 = more smoothing)',
        type=float,
    )
    return parser
