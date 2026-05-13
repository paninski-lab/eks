"""Entry point for the eks CLI."""

import argparse
import importlib
import logging
from pathlib import Path


def main() -> None:
    """Run the eks command-line interface."""
    parser = argparse.ArgumentParser(
        prog='eks',
        description='Ensemble Kalman Smoother for pose estimation.',
    )
    subparsers = parser.add_subparsers(
        title='subcommands',
        dest='subcommand',
    )
    subparsers.required = True

    cli_dir = Path(__file__).parent
    for module_path in sorted(cli_dir.glob('cmd_*.py')):
        module_name = module_path.stem
        module = importlib.import_module(f'eks.cli.{module_name}')
        module.register(subparsers)

    args = parser.parse_args()
    if getattr(args, 'verbose', False):
        logging.basicConfig(format='%(levelname)s:%(name)s: %(message)s')
        logging.getLogger('eks').setLevel(logging.DEBUG)
    else:
        logging.basicConfig(format='%(message)s')
        logging.getLogger('eks').setLevel(logging.INFO)
    args.handler(args)


if __name__ == '__main__':
    main()
