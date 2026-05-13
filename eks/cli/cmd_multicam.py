"""Subcommand for multi-camera ensemble Kalman smoothing."""

import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

from eks.cli._utils import (
    add_bodyparts,
    add_calibration,
    add_camera_names,
    add_common_args,
    add_inflate_vars,
    add_n_latent,
    add_quantile_keep_pca,
    add_s,
    handle_io,
)
from eks.multicam_smoother import fit_eks_multicam
from eks.utils import plot_results


def register(subparsers: argparse._SubParsersAction) -> None:
    """Register the multicam subcommand.

    Args:
        subparsers: The subparsers action to register this command with.
    """
    parser = subparsers.add_parser(
        'multicam',
        help='run ensemble Kalman smoothing on multi-camera pose data',
    )
    add_common_args(parser)
    add_bodyparts(parser)
    add_camera_names(parser)
    add_s(parser)
    add_quantile_keep_pca(parser)
    add_inflate_vars(parser)
    add_n_latent(parser)
    add_calibration(parser)
    parser.set_defaults(handler=cmd_multicam)


def cmd_multicam(args: argparse.Namespace) -> None:
    """Run ensemble Kalman smoothing on multi-camera pose data.

    Args:
        args: Parsed command-line arguments.
    """
    if args.calibration is None and args.camera_names is None:
        raise ValueError('--camera-names is required when --calibration is not provided')
    if args.calibration is not None and args.camera_names is not None:
        logger.warning(
            '--camera-names is ignored when --calibration is provided; '
            'camera names will be read from the calibration file'
        )

    input_source = args.input_dir if args.input_dir is not None else args.input_files
    if isinstance(input_source, str):
        input_dir = Path(input_source).resolve()
    else:
        input_dir = Path(input_source[0]).resolve().parent

    save_dir = handle_io(input_dir, args.save_dir)

    camera_dfs, s_finals, input_dfs, bodypart_list, df_3d = fit_eks_multicam(
        input_source=input_source,
        save_dir=str(save_dir),
        bodypart_list=args.bodypart_list,
        smooth_param=args.s,
        s_frames=args.s_frames,
        camera_names=args.camera_names,
        quantile_keep_pca=args.quantile_keep_pca,
        verbose=args.verbose,
        inflate_vars=args.inflate_vars,
        n_latent=args.n_latent,
        calibration=args.calibration,
    )

    if args.make_plot:
        plot_results(
            output_df=camera_dfs[-1],
            input_dfs_list=input_dfs[-1],
            key=f'{bodypart_list[-1]}',
            idxs=(0, 500),
            s_final=s_finals[-1],
            nll_values=None,
            save_dir=str(save_dir),
            smoother_type='multicam',
        )
