"""Subcommand for single-camera ensemble Kalman smoothing."""

import argparse
from pathlib import Path

from eks.cli._utils import (
    add_bodyparts,
    add_common_args,
    add_s,
    handle_io,
)
from eks.singlecam_smoother import fit_eks_singlecam
from eks.utils import plot_results


def register(subparsers: argparse._SubParsersAction) -> None:
    """Register the singlecam subcommand.

    Args:
        subparsers: The subparsers action to register this command with.
    """
    parser = subparsers.add_parser(
        'singlecam',
        help='run ensemble Kalman smoothing on single-camera pose data',
    )
    add_common_args(parser)
    add_bodyparts(parser)
    add_s(parser)
    parser.set_defaults(handler=cmd_singlecam)


def cmd_singlecam(args: argparse.Namespace) -> None:
    """Run ensemble Kalman smoothing on single-camera pose data.

    Args:
        args: Parsed command-line arguments.
    """
    input_source = args.input_dir if args.input_dir is not None else args.input_files
    if isinstance(input_source, str):
        input_dir = Path(input_source).resolve()
    else:
        input_dir = Path(input_source[0]).resolve().parent

    save_dir = handle_io(input_dir, args.save_dir)
    save_file = save_dir / (args.save_filename or 'eks_singlecam.csv')

    output_df, s_finals, input_dfs, bodypart_list = fit_eks_singlecam(
        input_source=input_source,
        save_file=str(save_file),
        bodypart_list=args.bodypart_list,
        smooth_param=args.s,
        s_frames=args.s_frames,
        blocks=args.blocks,
        verbose=args.verbose,
    )

    if args.make_plot:
        plot_results(
            output_df=output_df,
            input_dfs_list=input_dfs,
            key=f'{bodypart_list[-1]}',
            idxs=(0, 500),
            s_final=s_finals[-1],
            nll_values=None,
            save_dir=str(save_dir),
            smoother_type='singlecam',
        )
