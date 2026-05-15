"""Subcommand for IBL pupil ensemble Kalman smoothing."""

import argparse
from pathlib import Path

from eks.cli._utils import (
    add_com_s,
    add_common_args,
    add_diameter_s,
    handle_io,
    plot_results,
)
from eks.ibl_pupil_smoother import fit_eks_pupil


def register(subparsers: argparse._SubParsersAction) -> None:
    """Register the ibl-pupil subcommand.

    Args:
        subparsers: The subparsers action to register this command with.
    """
    parser = subparsers.add_parser(
        'ibl-pupil',
        help='run ensemble Kalman smoothing on IBL pupil tracking data',
    )
    add_common_args(parser)
    add_diameter_s(parser)
    add_com_s(parser)
    parser.set_defaults(handler=cmd_ibl_pupil)


def cmd_ibl_pupil(args: argparse.Namespace) -> None:
    """Run ensemble Kalman smoothing on IBL pupil tracking data.

    Args:
        args: Parsed command-line arguments.
    """
    input_source = args.input_dir if args.input_dir is not None else args.input_files
    if isinstance(input_source, str):
        input_dir = Path(input_source).resolve()
    else:
        input_dir = Path(input_source[0]).resolve().parent

    save_dir = handle_io(input_dir, args.save_dir)
    save_file = save_dir / (args.save_filename or 'eks_ibl_pupil.csv')

    df_smoothed, smooth_params, input_dfs_list, keypoint_names = fit_eks_pupil(
        input_source=input_source,
        save_file=str(save_file),
        smooth_params=[args.diameter_s, args.com_s],
        s_frames=args.s_frames,
    )

    if args.make_plot:
        plot_results(
            output_df=df_smoothed,
            input_dfs_list=input_dfs_list,
            key=f'{keypoint_names[-1]}',
            idxs=(0, 500),
            s_final=(smooth_params[0], smooth_params[1]),
            nll_values=None,
            save_dir=str(save_dir),
            smoother_type='ibl_pupil',
        )
