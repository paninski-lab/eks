"""Subcommand for mirrored multi-camera ensemble Kalman smoothing."""

import argparse
from pathlib import Path

from eks.cli._utils import (
    add_bodyparts,
    add_camera_names,
    add_common_args,
    add_inflate_vars,
    add_n_latent,
    add_quantile_keep_pca,
    add_s,
    handle_io,
    plot_results,
)
from eks.multicam_smoother import fit_eks_mirrored_multicam


def register(subparsers: argparse._SubParsersAction) -> None:
    """Register the mirrored-multicam subcommand.

    Args:
        subparsers: The subparsers action to register this command with.
    """
    parser = subparsers.add_parser(
        'mirrored-multicam',
        help='run ensemble Kalman smoothing on mirrored multi-camera pose data',
    )
    add_common_args(parser)
    add_bodyparts(parser)
    add_camera_names(parser)
    add_s(parser)
    add_quantile_keep_pca(parser)
    add_inflate_vars(parser)
    add_n_latent(parser)
    parser.set_defaults(handler=cmd_mirrored_multicam)


def cmd_mirrored_multicam(args: argparse.Namespace) -> None:
    """Run ensemble Kalman smoothing on mirrored multi-camera pose data.

    Args:
        args: Parsed command-line arguments.
    """
    input_source = args.input_dir if args.input_dir is not None else args.input_files
    if isinstance(input_source, str):
        input_dir = Path(input_source).resolve()
    else:
        input_dir = Path(input_source[0]).resolve().parent

    save_dir = handle_io(input_dir, args.save_dir)
    save_file = save_dir / (args.save_filename or 'eks_mirrored_multicam.csv')

    output_df, s_finals, input_dfs, bodypart_list = fit_eks_mirrored_multicam(
        input_source=input_source,
        save_file=str(save_file),
        bodypart_list=args.bodypart_list,
        smooth_param=args.s,
        s_frames=args.s_frames,
        camera_names=args.camera_names,
        quantile_keep_pca=args.quantile_keep_pca,
        inflate_vars=args.inflate_vars,
        n_latent=args.n_latent,
    )

    if args.make_plot:
        plot_results(
            output_df=output_df,
            input_dfs_list=input_dfs,
            key=f'{bodypart_list[-1]}_{args.camera_names[0]}',
            idxs=(0, 500),
            s_final=s_finals[-1],
            nll_values=None,
            save_dir=str(save_dir),
            smoother_type='multicam',
        )
