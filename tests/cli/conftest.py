"""Shared fixtures for the CLI tests."""

import argparse
import copy
from collections.abc import Callable
from typing import Any

import pytest

# arguments shared by every subcommand handler
_BASE_ARGS: dict[str, Any] = {
    'input_dir': '/tmp/input',
    'input_files': None,
    'save_dir': '/tmp/output',
    'save_filename': None,
    's_frames': None,
    'blocks': [],
    'verbose': False,
    'make_plot': False,
}


@pytest.fixture
def make_cli_args() -> Callable[[dict[str, Any] | None], Callable[..., argparse.Namespace]]:
    """Return a factory for building the argparse.Namespace a subcommand handler receives.

    Call the factory with the subcommand-specific defaults to get a builder; call the builder with
    keyword overrides. Unknown keywords raise a TypeError, mirroring a real function signature.
    """

    def _factory(
        command_defaults: dict[str, Any] | None = None,
    ) -> Callable[..., argparse.Namespace]:
        defaults = {**_BASE_ARGS, **(command_defaults or {})}

        def _build(**overrides: Any) -> argparse.Namespace:
            unknown = sorted(set(overrides) - set(defaults))
            if unknown:
                raise TypeError(f'unexpected keyword arguments: {unknown}')
            return argparse.Namespace(**{**copy.deepcopy(defaults), **overrides})

        return _build

    return _factory
