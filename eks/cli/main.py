"""Entry point for the eks CLI."""

import argparse
import importlib
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
    args.handler(args)


if __name__ == '__main__':
    main()
