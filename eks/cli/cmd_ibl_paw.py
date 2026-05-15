"""Subcommand for IBL paw multi-view ensemble Kalman smoothing."""

import argparse
from pathlib import Path

from eks.cli._utils import (
    add_common_args,
    add_inflate_vars,
    add_n_latent,
    add_quantile_keep_pca,
    add_s,
    handle_io,
    plot_results,
)
from eks.ibl_paw_multicam_smoother import fit_eks_multicam_ibl_paw


def register(subparsers: argparse._SubParsersAction) -> None:
    """Register the ibl-paw subcommand.

    Args:
        subparsers: The subparsers action to register this command with.
    """
    parser = subparsers.add_parser(
        'ibl-paw',
        help='run ensemble Kalman smoothing on IBL paw multi-view tracking data',
    )
    add_common_args(parser)
    add_s(parser)
    add_quantile_keep_pca(parser)
    add_inflate_vars(parser)
    add_n_latent(parser)
    parser.set_defaults(handler=cmd_ibl_paw)


def cmd_ibl_paw(args: argparse.Namespace) -> None:
    """Run ensemble Kalman smoothing on IBL paw multi-view tracking data.

    Args:
        args: Parsed command-line arguments.
    """
    input_source = args.input_dir if args.input_dir is not None else args.input_files
    if isinstance(input_source, str):
        input_dir = Path(input_source).resolve()
    else:
        input_dir = Path(input_source[0]).resolve().parent

    save_dir = handle_io(input_dir, args.save_dir)

    camera_dfs, s_finals, input_dfs, bodypart_list = fit_eks_multicam_ibl_paw(
        input_source=input_source,
        save_dir=str(save_dir),
        smooth_param=args.s,
        s_frames=args.s_frames,
        quantile_keep_pca=args.quantile_keep_pca,
        var_mode='var',
        inflate_vars=args.inflate_vars,
        n_latent=args.n_latent,
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
            smoother_type='ibl_paw',
            coords=['x', 'y'],
        )
