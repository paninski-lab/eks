import jax.numpy as jnp
import numpy as np
import pandas as pd

from eks.marker_array import MarkerArray, input_dfs_to_markerArray
from eks.singlecam_smoother import ensemble_kalman_smoother_singlecam, initialize_kalman_filter


def test_ensemble_kalman_smoother_singlecam():

    def _check_outputs(df, params):
        # Basic checks to ensure the function runs and returns expected types
        assert isinstance(df_smoothed, pd.DataFrame), \
            f"Expected first return value to be a pd.DataFrame, got {type(df_smoothed)}"
        assert isinstance(s_finals, (list, np.ndarray)), \
            "Expected s_finals to be a list or an ndarray"

        # Check for different return values in the correct level of the columns
        for v in ['x', 'y', 'likelihood']:
            assert v in df_smoothed.columns.get_level_values('coords'), \
                "Expected 'likelihood' in DataFrame columns at the 'coords' level"

    # Create mock data
    keypoint_names = ['kp1', 'kp2']
    columns = [f'{kp}_{coord}' for kp in keypoint_names for coord in ['x', 'y', 'likelihood']]
    markers_list = [
        pd.DataFrame(np.random.randn(100, len(columns)), columns=columns),
        pd.DataFrame(np.random.randn(100, len(columns)), columns=columns),
    ]
    s_frames = None
    blocks = []
    marker_array = input_dfs_to_markerArray([markers_list], keypoint_names, [""])

    # run with fixed smooth param (float)
    smooth_param = 0.1
    df_smoothed, s_finals = ensemble_kalman_smoother_singlecam(
        marker_array=marker_array,
        keypoint_names=keypoint_names,
        smooth_param=smooth_param,
        s_frames=s_frames,
        blocks=blocks,
    )
    _check_outputs(df_smoothed, s_finals)
    assert s_finals == [smooth_param]

    # run with fixed smooth param (int)
    smooth_param = 5
    df_smoothed, s_finals = ensemble_kalman_smoother_singlecam(
        marker_array=marker_array,
        keypoint_names=keypoint_names,
        smooth_param=smooth_param,
        s_frames=s_frames,
        blocks=blocks,
    )
    _check_outputs(df_smoothed, s_finals)
    assert s_finals == [smooth_param]

    # run with fixed smooth param (single-entry list)
    smooth_param = [0.1]
    df_smoothed, s_finals = ensemble_kalman_smoother_singlecam(
        marker_array=marker_array,
        keypoint_names=keypoint_names,
        smooth_param=smooth_param,
        s_frames=s_frames,
        blocks=blocks,
    )
    _check_outputs(df_smoothed, s_finals)
    assert s_finals == smooth_param

    # run with fixed smooth param (list)
    smooth_param = [0.1, 0.4]
    df_smoothed, s_finals = ensemble_kalman_smoother_singlecam(
        marker_array=marker_array,
        keypoint_names=keypoint_names,
        smooth_param=smooth_param,
        s_frames=s_frames,
        blocks=blocks,
    )
    _check_outputs(df_smoothed, s_finals)
    assert np.all(s_finals == smooth_param)

    # run with None smooth param
    smooth_param = None
    df_smoothed, s_finals = ensemble_kalman_smoother_singlecam(
        marker_array=marker_array,
        keypoint_names=keypoint_names,
        smooth_param=smooth_param,
        s_frames=s_frames,
        blocks=blocks,
    )
    _check_outputs(df_smoothed, s_finals)


def test_initialize_kalman_filter():
    # Define test parameters
    n_frames = 10
    n_keypoints = 3

    # Generate random centered ensemble predictions, Shape (1, 1, T, K, 2)
    centered_ensemble_preds = np.random.randn(1, 1, n_frames, n_keypoints, 2)

    # Convert to MarkerArray
    emA_centered_preds = MarkerArray(centered_ensemble_preds, data_fields=["x", "y"])

    # Run the function
    m0s, S0s, As, cov_mats, Cs = initialize_kalman_filter(emA_centered_preds)

    # Assertions to verify the function output
    assert m0s.shape == (n_keypoints, 2), \
        f"Expected shape {(n_keypoints, 2)}, got {m0s.shape}"
    assert S0s.shape == (n_keypoints, 2, 2), \
        f"Expected shape {(n_keypoints, 2, 2)}, got {S0s.shape}"
    assert As.shape == (n_keypoints, 2, 2), \
        f"Expected shape {(n_keypoints, 2, 2)}, got {As.shape}"
    assert cov_mats.shape == (n_keypoints, 2, 2), \
        f"Expected shape {(n_keypoints, 2, 2)}, got {cov_mats.shape}"
    assert Cs.shape == (n_keypoints, 2, 2), \
        f"Expected shape {(n_keypoints, 2, 2)}, got {Cs.shape}"

    # Check that the diagonal of S0s contains non-negative values (variance cannot be negative)
    assert jnp.all(S0s[:, 0, 0] >= 0), "S0s diagonal should have non-negative variances"
    assert jnp.all(S0s[:, 1, 1] >= 0), "S0s diagonal should have non-negative variances"

    # Check that the state transition matrix is correctly initialized as identity
    expected_A = jnp.tile(jnp.eye(2), (n_keypoints, 1, 1))
    assert jnp.allclose(As, expected_A), "State transition matrix A should be identity"

    # Check that the measurement function matrix C is correctly initialized
    expected_C = jnp.tile(jnp.eye(2), (n_keypoints, 1, 1))
    assert jnp.allclose(Cs, expected_C), "Measurement function matrix C should be identity"

    # Check that covariance matrices are initialized correctly as identity
    expected_cov = jnp.tile(jnp.eye(2), (n_keypoints, 1, 1))
    assert jnp.allclose(cov_mats, expected_cov), "Covariance matrices should be identity"

    print("Test for initialize_kalman_filter passed successfully.")
