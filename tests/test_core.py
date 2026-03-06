import jax.numpy as jnp
import numpy as np

from eks.core import ensemble
from eks.marker_array import MarkerArray


def test_jax_ensemble():
    # Basic test data
    n_models = 4
    n_cameras = 2
    n_frames = 5
    n_keypoints = 3

    # Creating a random MarkerArray
    markers_5d_array = np.random.rand(n_models, n_cameras, n_frames, n_keypoints, 3)
    marker_array = MarkerArray(markers_5d_array, data_fields=['x', 'y', 'likelihood'])

    # ---------------------------------------------
    # Run jax_ensemble in median mode
    # ---------------------------------------------
    ensemble_marker_array = ensemble(marker_array, avg_mode='median')

    # Check output shape
    expected_shape = (1, n_cameras, n_frames, n_keypoints, 5)
    assert ensemble_marker_array.array.shape == expected_shape, \
        f"Expected shape {expected_shape}, got {ensemble_marker_array.array.shape}"

    # Check that output values are finite
    assert jnp.isfinite(
        ensemble_marker_array.array).all(), "Expected finite values in ensemble output"

    # ---------------------------------------------
    # Run jax_ensemble in mean mode
    # ---------------------------------------------
    ensemble_marker_array = ensemble(marker_array, avg_mode='mean')

    # Check output shape
    assert ensemble_marker_array.array.shape == expected_shape, \
        f"Expected shape {expected_shape}, got {ensemble_marker_array.array.shape}"

    # Check that output values are finite
    assert jnp.isfinite(
        ensemble_marker_array.array).all(), "Expected finite values in ensemble output"

    # ---------------------------------------------
    # Test confidence-weighted variance mode
    # ---------------------------------------------
    ensemble_marker_array = ensemble(marker_array, var_mode='confidence_weighted_var')

    # Check output shape
    assert ensemble_marker_array.array.shape == expected_shape, \
        f"Expected shape {expected_shape}, got {ensemble_marker_array.array.shape}"

    # Check that output values are finite
    assert jnp.isfinite(
        ensemble_marker_array.array).all(), "Expected finite values in ensemble output"


def test_jax_ensemble_nan_variance():
    """Test that NaN values in variance fields are replaced by nan_replacement."""
    n_models = 3
    n_cameras = 2
    n_frames = 5
    n_keypoints = 4
    nan_replacement = 1000.0

    # Create test data with NaNs in x and y
    data = np.random.rand(n_models, n_cameras, n_frames, n_keypoints, 3)
    data[..., 0] = np.nan  # Force NaNs in x-coordinates
    data[..., 1] = np.nan  # Force NaNs in y-coordinates

    marker_array = MarkerArray(data, data_fields=["x", "y", "likelihood"])
    ensemble_marker_array = ensemble(marker_array, nan_replacement=nan_replacement)

    var_x = ensemble_marker_array.array[..., 2]  # Extract var_x
    var_y = ensemble_marker_array.array[..., 3]  # Extract var_y

    # Assert that NaNs were replaced with the specified replacement value
    assert np.all(var_x == nan_replacement), "NaNs in var_x were not replaced"
    assert np.all(var_y == nan_replacement), "NaNs in var_y were not replaced"


def test_jax_ensemble_single_network():
    """
    Test that ensemble() produces valid (positive, finite) variance estimates when n_models=1.

    With a single network there is no cross-model spread to measure, so nanvar returns 0.
    The function must fall back to a likelihood-based variance proxy so that downstream
    Kalman filtering receives non-zero observation noise.
    """
    n_models = 1
    n_cameras = 2
    n_frames = 10
    n_keypoints = 3

    rng = np.random.default_rng(0)
    data = rng.random((n_models, n_cameras, n_frames, n_keypoints, 3))
    # Likelihoods in (0, 1) — not all ones, so a likelihood-based fallback can be non-trivial
    data[..., 2] = rng.uniform(0.5, 1.0, size=(n_models, n_cameras, n_frames, n_keypoints))

    marker_array = MarkerArray(data, data_fields=["x", "y", "likelihood"])

    for avg_mode in ("median", "mean"):
        for var_mode in ("var", "confidence_weighted_var"):
            result = ensemble(marker_array, avg_mode=avg_mode, var_mode=var_mode)

            # Shape must still be correct
            expected_shape = (1, n_cameras, n_frames, n_keypoints, 5)
            assert result.array.shape == expected_shape, (
                f"[{avg_mode}, {var_mode}] Expected shape {expected_shape}, "
                f"got {result.array.shape}"
            )

            var_x = np.array(result.array[..., 2])
            var_y = np.array(result.array[..., 3])

            # All variance values must be finite
            assert np.all(np.isfinite(var_x)), \
                f"[{avg_mode}, {var_mode}] var_x contains non-finite values"
            assert np.all(np.isfinite(var_y)), \
                f"[{avg_mode}, {var_mode}] var_y contains non-finite values"

            # Variance must be strictly positive — zero variance from nanvar(single sample)
            # is not acceptable as observation noise in the Kalman filter
            assert np.all(var_x > 0), \
                f"[{avg_mode}, {var_mode}] var_x is zero (nanvar of single sample fallback needed)"
            assert np.all(var_y > 0), \
                f"[{avg_mode}, {var_mode}] var_y is zero (nanvar of single sample fallback needed)"


def test_jax_ensemble_zero_likelihood():
    """Test that zero likelihood does not cause NaNs in variance calculations."""
    n_models = 3
    n_cameras = 2
    n_frames = 5
    n_keypoints = 4
    nan_replacement = 1000.0

    # Create test data with likelihood = 0
    data = np.random.rand(n_models, n_cameras, n_frames, n_keypoints, 3)
    data[..., 2] = 0  # Set all likelihood values to 0

    marker_array = MarkerArray(data, data_fields=["x", "y", "likelihood"])
    ensemble_marker_array = ensemble(marker_array, nan_replacement=nan_replacement)

    var_x = ensemble_marker_array.array[..., 2]  # Extract var_x
    var_y = ensemble_marker_array.array[..., 3]  # Extract var_y

    # Check that variance fields contain no NaNs
    assert np.all(np.isfinite(var_x)), "var_x contains NaNs due to zero likelihood"
    assert np.all(np.isfinite(var_y)), "var_y contains NaNs due to zero likelihood"
