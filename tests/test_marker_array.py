import pytest
import numpy as np
import jax.numpy as jnp
from eks.marker_array import MarkerArray

def generate_test_marker_array(n_models=2, n_cameras=2, n_frames=10, n_keypoints=3, n_fields=3):
    """ Generate a test MarkerArray with structured values for easy validation. """
    data = np.zeros((n_models, n_cameras, n_frames, n_keypoints, n_fields), dtype=np.float32)
    for k in range(n_keypoints):
        for c in range(n_cameras):
            for e in range(n_models):
                data[e, c, :, k, 0] = k * c * e * np.arange(1, 11)  # X values
                data[e, c, :, k, 1] = k * c * e * np.arange(11, 21)  # Y values
                data[e, c, :, k, 2] = k * c * e * np.arange(21, 31)  # Likelihood values
    return MarkerArray(data, data_fields=["x", "y", "likelihood"])

def test_marker_array_shapes():
    marker_array = generate_test_marker_array()
    expected_shape = (2, 2, 10, 3, 3)
    assert marker_array.shape == expected_shape, \
        f"Expected shape {expected_shape}, got {marker_array.shape}"

def test_marker_array_field_slicing():
    marker_array = generate_test_marker_array()
    sliced_array = marker_array.slice_fields("x")
    expected_shape = (2, 2, 10, 3, 1)
    assert sliced_array.shape == expected_shape, \
        f"Expected shape {expected_shape}, got {sliced_array.shape}"

def test_marker_array_axis_slicing():
    marker_array = generate_test_marker_array()
    sliced_array = marker_array.slice("cameras", 1)
    expected_shape = (2, 1, 10, 3, 3)
    assert sliced_array.shape == expected_shape, \
        f"Expected shape {expected_shape}, got {sliced_array.shape}"

def test_marker_array_values():
    marker_array = generate_test_marker_array()
    k, c, e = 1, 1, 1
    expected_x_values = k * c * e * np.arange(1, 11)
    retrieved_values = marker_array.array[e, c, :, k, 0]
    assert np.array_equal(retrieved_values, expected_x_values), \
        "X values do not match expected pattern"

def test_marker_array_clone():
    marker_array = generate_test_marker_array()
    cloned_array = MarkerArray(marker_array=marker_array)
    assert np.array_equal(marker_array.array, cloned_array.array), \
        "Cloned MarkerArray does not match original"

def test_invalid_slice():
    marker_array = generate_test_marker_array()
    with pytest.raises(AssertionError, match="Invalid slice axis"):
        marker_array.slice("invalid_axis", 0)

def test_invalid_field():
    marker_array = generate_test_marker_array()
    with pytest.raises(AssertionError, match="Field 'invalid' not found"):
        marker_array.slice_fields("invalid")
