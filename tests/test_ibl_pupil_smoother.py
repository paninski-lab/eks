import numpy as np
import pandas as pd
import pytest

from eks.ibl_pupil_smoother import (
    add_mean_to_array,
    ensemble_kalman_smoother_ibl_pupil,
    get_pupil_diameter,
    get_pupil_location,
)


@pytest.fixture
def mock_dlc_data():
    """
    Fixture to generate mock DLC data for testing the get_pupil_location function.
    """
    n_samples = 10

    # Generate random data for pupil coordinates
    dlc_data = {
        'pupil_top_r_x': np.random.rand(n_samples),
        'pupil_top_r_y': np.random.rand(n_samples),
        'pupil_bottom_r_x': np.random.rand(n_samples),
        'pupil_bottom_r_y': np.random.rand(n_samples),
        'pupil_left_r_x': np.random.rand(n_samples),
        'pupil_left_r_y': np.random.rand(n_samples),
        'pupil_right_r_x': np.random.rand(n_samples),
        'pupil_right_r_y': np.random.rand(n_samples),
    }

    # Introduce some NaN values randomly
    dlc_data['pupil_top_r_x'][2] = np.nan
    dlc_data['pupil_left_r_y'][5] = np.nan

    return dlc_data


def test_get_pupil_location(mock_dlc_data):
    """
    Test the get_pupil_location function using mock data.
    """
    dlc = mock_dlc_data
    center = get_pupil_location(dlc)

    # Assertions
    assert isinstance(center, np.ndarray), "Expected center to be a numpy array"
    assert center.shape == (len(dlc['pupil_top_r_x']), 2), \
        f"Expected shape to be {(len(dlc['pupil_top_r_x']), 2)}, got {center.shape}"

    # Check that the output does not contain any NaNs where expected
    assert np.isfinite(center).all(), "Expected no NaN values in center"

    # Check if the median calculations return finite values when data is complete
    non_nan_center = get_pupil_location({key: np.random.rand(10) for key in dlc.keys()})
    assert np.isfinite(non_nan_center).all(), "Expected no NaN values when data is complete"

    print("Test for get_pupil_location passed successfully.")


def test_get_pupil_diameter(mock_dlc_data):
    """
    Test the get_pupil_diameter function using mock data.
    """
    dlc = mock_dlc_data
    diameters = get_pupil_diameter(dlc)

    # Assertions
    assert isinstance(diameters, np.ndarray), "Expected output to be a numpy array"
    assert diameters.shape == (len(dlc['pupil_top_r_x']),), \
        f"Expected shape to be {(len(dlc['pupil_top_r_x']),)}, got {diameters.shape}"

    # Check that the output does not contain any NaNs where expected
    assert np.isfinite(diameters).all(), "Expected no NaN values in diameters"

    # Check if the median calculations return finite values when data is complete
    non_nan_dlc = {key: np.random.rand(10) for key in dlc.keys()}
    non_nan_diameters = get_pupil_diameter(non_nan_dlc)
    assert np.isfinite(non_nan_diameters).all(), "Expected no NaN values with complete data"

    # Test with completely NaN input
    nan_dlc = {key: np.full(10, np.nan) for key in dlc.keys()}
    nan_diameters = get_pupil_diameter(nan_dlc)
    assert np.isnan(nan_diameters).all(), "Expected NaN values with all NaN input"

    print("Test for get_pupil_diameter passed successfully.")


@pytest.fixture
def mock_data_1():
    """
    Fixture to generate mock data for testing the add_mean_to_array function.
    """
    # Generate a random array of shape (10, 4) with some example keys
    pred_arr = np.random.randn(10, 4)
    keys = ['key1_x', 'key2_y', 'key3_x', 'key4_y']
    mean_x = 2.0
    mean_y = 3.0
    return pred_arr, keys, mean_x, mean_y


def test_add_mean_to_array(mock_data_1):
    """
    Test the add_mean_to_array function using mock data.
    """
    pred_arr, keys, mean_x, mean_y = mock_data_1

    # Run the function with the mock data
    result = add_mean_to_array(pred_arr, keys, mean_x, mean_y)

    # Assertions to verify the result
    assert isinstance(result, dict), "Expected output to be a dictionary"
    assert len(result) == len(keys), f"Expected dict to have {len(keys)} keys, got {len(result)}"

    # Check that the dictionary keys match the input keys
    assert set(result.keys()) == set(keys), "Keys in the output dictionary do not match input keys"

    # Verify the values in the dictionary are correctly offset by mean_x and mean_y
    for i, key in enumerate(keys):
        if 'x' in key:
            expected = pred_arr[:, i] + mean_x
        else:
            expected = pred_arr[:, i] + mean_y
        np.testing.assert_array_almost_equal(
            result[key], expected, err_msg=f"Mismatch for key '{key}'"
        )


def test_add_mean_to_array_empty():
    """
    Test the add_mean_to_array function with empty input arrays.
    """
    pred_arr = np.array([]).reshape(0, 0)
    keys = []
    mean_x = 2.0
    mean_y = 3.0

    result = add_mean_to_array(pred_arr, keys, mean_x, mean_y)

    # Assertions for empty inputs
    assert isinstance(result, dict), "Expected output to be a dictionary"
    assert len(result) == 0, "Expected empty dictionary for empty input"


def test_add_mean_to_array_single_row():
    """
    Test the add_mean_to_array function with a single row of data.
    """
    pred_arr = np.array([[1.0, 2.0, 3.0, 4.0]])
    keys = ['key1_x', 'key2_y', 'key3_x', 'key4_y']
    mean_x = 2.0
    mean_y = 3.0

    result = add_mean_to_array(pred_arr, keys, mean_x, mean_y)

    # Expected output
    expected_dict = {
        'key1_x': np.array([1.0 + mean_x]),
        'key2_y': np.array([2.0 + mean_y]),
        'key3_x': np.array([3.0 + mean_x]),
        'key4_y': np.array([4.0 + mean_y]),
    }

    # Assertions to verify the result
    for key, expected_value in expected_dict.items():
        np.testing.assert_array_almost_equal(
            result[key], expected_value, err_msg=f"Mismatch for key '{key}'"
        )

    print("All tests for add_mean_to_array passed successfully.")


def test_ensemble_kalman_smoother_ibl_pupil():

    def _check_outputs(df, params, nlls):
        assert isinstance(df, pd.DataFrame), "first return arg should be a DataFrame"
        assert df.shape[0] == 100, "markers_df should have 100 rows"
        assert len(params) == 2, "Expected 2 smooth parameters"
        assert params[0] < 1, "Expected diameter smoothing parameter to be less than 1"
        assert params[1] < 1, "Expected COM smoothing parameter to be less than 1"
        assert isinstance(nlls, list), "Expected nll_values to be a list"

    # Create mock data
    columns = [
        'pupil_top_r_x', 'pupil_top_r_y', 'pupil_bottom_r_x', 'pupil_bottom_r_y',
        'pupil_right_r_x', 'pupil_right_r_y', 'pupil_left_r_x', 'pupil_left_r_y'
    ]
    markers_list = [
        pd.DataFrame(np.random.randn(100, 8), columns=columns),
        pd.DataFrame(np.random.randn(100, 8), columns=columns),
    ]
    s_frames = [(1, 20)]

    # Run with fixed smooth params
    smooth_params = [0.5, 0.5]
    smoothed_df, smooth_params_out, nll_values = ensemble_kalman_smoother_ibl_pupil(
        markers_list, smooth_params, s_frames, avg_mode='mean', var_mode='var',
    )
    _check_outputs(smoothed_df, smooth_params_out, nll_values)
    assert smooth_params == smooth_params_out

    # Run with [None, None] smooth params
    smoothed_df, smooth_params_out, nll_values = ensemble_kalman_smoother_ibl_pupil(
        markers_list, [None, None], s_frames, avg_mode='mean', var_mode='var',
    )
    _check_outputs(smoothed_df, smooth_params_out, nll_values)

    # Run with None smooth params
    smoothed_df, smooth_params_out, nll_values = ensemble_kalman_smoother_ibl_pupil(
        markers_list, None, s_frames, avg_mode='mean', var_mode='var',
    )
    _check_outputs(smoothed_df, smooth_params_out, nll_values)

    # CURRENTLY NOT SUPPORTED: fix one smooth param

    # Run with diameter smooth param
    # smooth_params = [0.9, None]
    # smoothed_df, smooth_params_out, nll_values = ensemble_kalman_smoother_ibl_pupil(
    #     markers_list, [0.9, None], s_frames, avg_mode='mean', var_mode='var',
    # )
    # _check_outputs(smoothed_df, smooth_params_out, nll_values)
    # assert smooth_params[0] == smooth_params_out[0]

    # Run with COM smooth param
    # smoothed_df, smooth_params_out, nll_values = ensemble_kalman_smoother_ibl_pupil(
    #     markers_list, [None, 0.9], s_frames, avg_mode='mean', var_mode='var',
    # )
    # _check_outputs(smoothed_df, smooth_params_out, nll_values)
    # assert smooth_params[1] == smooth_params_out[1]
