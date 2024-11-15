import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from eks.ibl_pupil_smoother import get_pupil_location, get_pupil_diameter, add_mean_to_array, ensemble_kalman_smoother_ibl_pupil


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
        'pupil_right_r_y': np.random.rand(n_samples)
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
    assert len(result) == len(keys), f"Expected dictionary to have {len(keys)} keys, got {len(result)}"

    # Check that the dictionary keys match the input keys
    assert set(result.keys()) == set(keys), "Keys in the output dictionary do not match input keys"

    # Verify the values in the dictionary are correctly offset by mean_x and mean_y
    for i, key in enumerate(keys):
        if 'x' in key:
            expected = pred_arr[:, i] + mean_x
        else:
            expected = pred_arr[:, i] + mean_y
        np.testing.assert_array_almost_equal(result[key], expected, err_msg=f"Mismatch for key '{key}'")


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
        np.testing.assert_array_almost_equal(result[key], expected_value, err_msg=f"Mismatch for key '{key}'")

    print("All tests for add_mean_to_array passed successfully.")


@pytest.fixture
def mock_data():
    """
    Fixture to provide mock data for testing.
    """
    markers_list = [
        pd.DataFrame(
            np.random.randn(100, 8),
            columns=[
                'pupil_top_r_x', 'pupil_top_r_y', 'pupil_bottom_r_x', 'pupil_bottom_r_y',
                'pupil_right_r_x', 'pupil_right_r_y', 'pupil_left_r_x', 'pupil_left_r_y'
            ]
        )
    ]
    keypoint_names = [
        'pupil_top_r', 'pupil_bottom_r', 'pupil_right_r', 'pupil_left_r'
    ]
    tracker_name = 'ensemble-kalman_tracker'
    smooth_params = [0.5, 0.5]
    s_frames = [10, 20, 30]
    return markers_list, keypoint_names, tracker_name, smooth_params, s_frames


@patch('eks.core.ensemble')
@patch('eks.ibl_pupil_smoother.get_pupil_location')
@patch('eks.ibl_pupil_smoother.get_pupil_diameter')
@patch('eks.ibl_pupil_smoother.pupil_optimize_smooth')
@patch('eks.utils.make_dlc_pandas_index')
@patch('eks.ibl_pupil_smoother.add_mean_to_array')
@patch('eks.core.eks_zscore')
def test_ensemble_kalman_smoother_ibl_pupil(
    mock_zscore, mock_add_mean, mock_index, mock_smooth,
    mock_get_diameter, mock_get_location, mock_ensemble,
    mock_data
):
    # Unpack mock data
    markers_list, keypoint_names, tracker_name, smooth_params, s_frames = mock_data

    # Mock the ensemble function
    ensemble_preds = np.random.randn(100, 8)
    ensemble_vars = np.random.rand(100, 8) * 0.1
    ensemble_stacks = np.random.randn(5, 100, 8)
    keypoints_mean_dict = {k: np.random.randn(100) for k in [
        'pupil_top_r_x', 'pupil_top_r_y', 'pupil_bottom_r_x', 'pupil_bottom_r_y',
        'pupil_right_r_x', 'pupil_right_r_y', 'pupil_left_r_x', 'pupil_left_r_y']}
    keypoints_var_dict = keypoints_mean_dict.copy()
    keypoints_stack_dict = {i: keypoints_mean_dict for i in range(5)}

    mock_ensemble.return_value = (ensemble_preds, ensemble_vars, ensemble_stacks,
                                  keypoints_mean_dict, keypoints_var_dict, keypoints_stack_dict)

    # Mock the get_pupil_location and get_pupil_diameter functions
    mock_get_location.return_value = np.random.randn(100, 2)
    mock_get_diameter.return_value = np.random.rand(100)

    # Mock the pupil_optimize_smooth function
    mock_smooth.return_value = ([0.5, 0.6], np.random.randn(100, 3), np.random.rand(100, 3, 3), 0.05, [0.1, 0.2])

    # Mock the make_dlc_pandas_index function
    mock_index.return_value = pd.MultiIndex.from_product(
        [['ensemble-kalman_tracker'], keypoint_names, ['x', 'y', 'likelihood', 'x_var', 'y_var', 'zscore']]
    )

    # Mock the add_mean_to_array function
    mock_add_mean.return_value = {f'{k}_x': np.random.randn(100) for k in keypoint_names}
    mock_add_mean.return_value.update({f'{k}_y': np.random.randn(100) for k in keypoint_names})

    # Mock the eks_zscore function
    mock_zscore.return_value = np.random.randn(100), None

    # Run the function with mocked data
    result, smooth_params_out, nll_values = ensemble_kalman_smoother_ibl_pupil(
        markers_list, keypoint_names, tracker_name, smooth_params, s_frames
    )

    # Assertions
    assert isinstance(result, dict), "Expected result to be a dictionary"
    assert 'markers_df' in result, "Expected 'markers_df' in result"
    assert 'latents_df' in result, "Expected 'latents_df' in result"

    markers_df = result['markers_df']
    latents_df = result['latents_df']

    assert isinstance(markers_df, pd.DataFrame), "markers_df should be a DataFrame"
    assert isinstance(latents_df, pd.DataFrame), "latents_df should be a DataFrame"

    # Verify the shape of the output DataFrames
    assert markers_df.shape[0] == 100, "markers_df should have 100 rows"
    assert latents_df.shape[0] == 100, "latents_df should have 100 rows"

    # Check if the smooth parameters and NLL values are correctly returned
    assert len(smooth_params_out) == 2, "Expected 2 smooth parameters"
    assert isinstance(nll_values, list), "Expected nll_values to be a list"

    print("All tests passed successfully.")


if __name__ == "__main__":
    pytest.main([__file__])