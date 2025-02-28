import numpy as np
import pytest

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


def test_stack_fields_success():
    """Test successful stacking of MarkerArray instances along the fields axis."""
    shape = (10, 10, 3, 5, 2)  # (x, y, z, t, fields)
    array1 = np.random.rand(*shape)
    array2 = np.random.rand(*shape)

    marker1 = MarkerArray(array1, data_fields=['field1', 'field2'])
    marker2 = MarkerArray(array2, data_fields=['field3', 'field4'])

    stacked_marker = MarkerArray.stack_fields(marker1, marker2)

    # Check shape: fields should be doubled
    assert stacked_marker.array.shape == (10, 10, 3, 5, 4)  # 2+2 fields

    # Check data_fields concatenation
    assert stacked_marker.data_fields == ['field1', 'field2', 'field3', 'field4']


def test_stack_fields_shape_mismatch():
    """Test that an assertion error is raised when shape mismatch occurs."""
    shape1 = (10, 10, 3, 5, 2)
    shape2 = (12, 10, 3, 5, 2)  # Mismatched first dimension

    array1 = np.random.rand(*shape1)
    array2 = np.random.rand(*shape2)

    marker1 = MarkerArray(array1, data_fields=['field1', 'field2'])
    marker2 = MarkerArray(array2, data_fields=['field3', 'field4'])

    with pytest.raises(AssertionError, match="Shape mismatch"):
        MarkerArray.stack_fields(marker1, marker2)


def test_stack_fields_missing_data_fields():
    """Test that an assertion error is raised when a MarkerArray lacks data_fields."""
    shape = (10, 10, 3, 5, 2)
    array1 = np.random.rand(*shape)
    array2 = np.random.rand(*shape)

    marker1 = MarkerArray(array1, data_fields=['field1', 'field2'])
    marker2 = MarkerArray(array2, data_fields=None)  # Missing data_fields

    with pytest.raises(AssertionError, match="All MarkerArrays must have data_fields defined"):
        MarkerArray.stack_fields(marker1, marker2)


def test_stack_fields_single_input():
    """Test that stack_fields works with a single MarkerArray."""
    shape = (10, 10, 3, 5, 2)
    array = np.random.rand(*shape)
    marker = MarkerArray(array, data_fields=['field1', 'field2'])

    stacked_marker = MarkerArray.stack_fields(marker)

    assert stacked_marker.array.shape == shape  # No change in shape
    assert stacked_marker.data_fields == ['field1', 'field2']


def test_stack_fields_no_input():
    """Test that an assertion error is raised when no MarkerArray is provided."""
    with pytest.raises(AssertionError,
                       match="At least one MarkerArray must be provided for stacking."):
        MarkerArray.stack_fields()


def test_reorder_data_fields_success():
    """Test successful reordering of data fields."""
    shape = (10, 10, 3, 5, 4)  # (x, y, z, t, fields)
    array = np.random.rand(*shape)
    original_fields = ['A', 'B', 'C', 'D']
    new_order = ['C', 'A', 'D', 'B']

    marker = MarkerArray(array, data_fields=original_fields)
    reordered_marker = marker.reorder_data_fields(new_order)

    # Check the new data fields order
    assert reordered_marker.data_fields == new_order


def test_reorder_data_fields_invalid_order():
    """Test that an assertion error is raised when new_order does not match existing fields."""
    shape = (10, 10, 3, 5, 4)
    array = np.random.rand(*shape)
    original_fields = ['A', 'B', 'C', 'D']
    invalid_order = ['A', 'B', 'X', 'D']  # 'X' is not in original fields

    marker = MarkerArray(array, data_fields=original_fields)

    with pytest.raises(AssertionError, match="Mismatch in data fields"):
        marker.reorder_data_fields(invalid_order)


def test_reorder_data_fields_partial_order():
    """Test that an assertion error is raised when `new_order` has missing fields."""
    shape = (10, 10, 3, 5, 4)
    array = np.random.rand(*shape)
    original_fields = ['A', 'B', 'C', 'D']
    partial_order = ['A', 'B', 'C']  # Missing 'D'

    marker = MarkerArray(array, data_fields=original_fields)

    with pytest.raises(AssertionError, match="Mismatch in data fields"):
        marker.reorder_data_fields(partial_order)


def test_reorder_data_fields_duplicate_fields():
    """Test that an assertion error is raised when `new_order` has duplicate fields."""
    shape = (10, 10, 3, 5, 4)
    array = np.random.rand(*shape)
    original_fields = ['A', 'B', 'C', 'D']
    duplicate_order = ['A', 'A', 'C', 'D']  # 'A' appears twice

    marker = MarkerArray(array, data_fields=original_fields)

    with pytest.raises(AssertionError, match="Mismatch in data fields"):
        marker.reorder_data_fields(duplicate_order)
