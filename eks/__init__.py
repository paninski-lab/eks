"""Ensemble Kalman Smoother (EKS) package for pose estimation post-processing."""

import importlib.metadata
from typing import Any

# Public API — importable directly from `eks`
from eks.ibl_paw_multicam_smoother import fit_eks_multicam_ibl_paw
from eks.ibl_pupil_smoother import fit_eks_pupil
from eks.marker_array import MarkerArray
from eks.multicam_smoother import fit_eks_mirrored_multicam, fit_eks_multicam
from eks.singlecam_smoother import fit_eks_singlecam

__all__ = [
    # data structure
    'MarkerArray',
    # single-camera smoother
    'fit_eks_singlecam',
    # multi-camera smoother
    'fit_eks_multicam',
    'fit_eks_mirrored_multicam',
    # IBL pupil smoother
    'fit_eks_pupil',
    # IBL paw smoother
    'fit_eks_multicam_ibl_paw',
]

# from eks import *
__package_version = "unknown"


def __get_package_version() -> str:
    """Find the version of this package."""
    global __package_version

    if __package_version != "unknown":
        # We already set it at some point in the past,
        # so return that previous value without any
        # extra work.
        return __package_version

    try:
        # Try to get the version of the current package if
        # it is running from a distribution.
        __package_version = importlib.metadata.version("ensemble-kalman-smoother")
    except importlib.metadata.PackageNotFoundError:
        # Fall back on getting it from a local pyproject.toml.
        # This works in a development environment where the
        # package has not been installed from a distribution.
        import warnings
        from pathlib import Path

        import toml
        warnings.warn(
            "ensemble-kalman-smoother not pip-installed, getting version from pyproject.toml.",
            stacklevel=2,
        )
        pyproject_toml_file = Path(__file__).parent.parent / "pyproject.toml"
        __package_version = toml.load(pyproject_toml_file)["project"]["version"]

    return __package_version


def __getattr__(name: str) -> Any:
    """Get package attributes."""
    if name in ("version", "__version__"):
        return __get_package_version()
    else:
        raise AttributeError(f"No attribute {name} in module {__name__}.")
