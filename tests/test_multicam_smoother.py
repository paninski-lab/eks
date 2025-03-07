import numpy as np

from eks.marker_array import MarkerArray
from eks.multicam_smoother import ensemble_kalman_smoother_multicam, inflate_variance


def test_ensemble_kalman_smoother_multicam():
    """Test the basic functionality of ensemble_kalman_smoother_multicam."""

    # Mock inputs
    keypoint_names = ['kp1', 'kp2']
    data_fields = ['x', 'y', 'likelihood']
    n_cameras = 2
    n_frames = 100
    num_fields = len(data_fields)

    # Create mock MarkerArray with explicit data_fields
    markers_array = np.random.randn(5, n_cameras, n_frames, len(keypoint_names), num_fields)
    marker_array = MarkerArray(markers_array, data_fields=data_fields)

    camera_names = ['cam1', 'cam2']
    smooth_param = 0.1
    quantile_keep_pca = 95
    s_frames = None

    # Run the smoother
    camera_dfs, smooth_params_final = ensemble_kalman_smoother_multicam(
        marker_array=marker_array,
        keypoint_names=keypoint_names,
        smooth_param=smooth_param,
        quantile_keep_pca=quantile_keep_pca,
        camera_names=camera_names,
        s_frames=s_frames,
        avg_mode='median',
        inflate_vars=False,
    )
    assert isinstance(camera_dfs, list), "Expected output to be a list"
    assert len(camera_dfs) == len(camera_names), \
        f"Expected {len(camera_names)} entries in camera_dfs, got {len(camera_dfs)}"
    assert isinstance(smooth_params_final, float), \
        f"Expected smooth_param_final to be a float, got {type(smooth_params_final)}"
    assert smooth_params_final == smooth_param, \
        f"Expected smooth_param_final to match input smooth_param ({smooth_param}), " \
        f"got {smooth_params_final}"

    # Run with variance inflation
    camera_dfs, smooth_params_final = ensemble_kalman_smoother_multicam(
        marker_array=marker_array,
        keypoint_names=keypoint_names,
        smooth_param=smooth_param,
        quantile_keep_pca=quantile_keep_pca,
        camera_names=camera_names,
        s_frames=s_frames,
        avg_mode='median',
        inflate_vars=True,
        inflate_vars_kwargs={'likelihood_threshold': None}
    )
    # Assertions
    assert isinstance(camera_dfs, list), "Expected output to be a list"
    assert len(camera_dfs) == len(camera_names), \
        f"Expected {len(camera_names)} entries in camera_dfs, got {len(camera_dfs)}"
    assert isinstance(smooth_params_final, float), \
        f"Expected smooth_param_final to be a float, got {type(smooth_params_final)}"
    assert smooth_params_final == smooth_param, \
        f"Expected smooth_param_final to match input smooth_param ({smooth_param}), " \
        f"got {smooth_params_final}"

    # Run with variance inflation + more maha kwargs
    camera_dfs, smooth_params_final = ensemble_kalman_smoother_multicam(
        marker_array=marker_array,
        keypoint_names=keypoint_names,
        smooth_param=smooth_param,
        quantile_keep_pca=quantile_keep_pca,
        camera_names=camera_names,
        s_frames=s_frames,
        avg_mode='mean',
        inflate_vars=True,
        inflate_vars_kwargs={
            'loading_matrix': np.random.randn(2 * len(camera_names), 3),
            'mean': np.random.randn(2 * len(camera_names))
        }
    )
    assert isinstance(camera_dfs, list), "Expected output to be a list"
    assert len(camera_dfs) == len(camera_names), \
        f"Expected {len(camera_names)} entries in camera_dfs, got {len(camera_dfs)}"
    assert isinstance(smooth_params_final, float), \
        f"Expected smooth_param_final to be a float, got {type(smooth_params_final)}"
    assert smooth_params_final == smooth_param, \
        f"Expected smooth_param_final to match input smooth_param ({smooth_param}), " \
        f"got {smooth_params_final}"


def test_ensemble_kalman_smoother_multicam_no_smooth_param():
    """Test ensemble_kalman_smoother_multicam with no smooth_param provided."""
    # Mock inputs
    keypoint_names = ['kp1', 'kp2']
    data_fields = ['x', 'y', 'likelihood']
    n_cameras = 2
    n_frames = 100
    num_fields = len(data_fields)

    # Create mock MarkerArray with explicit data_fields
    markers_array = np.random.randn(3, n_cameras, n_frames, len(keypoint_names), num_fields)
    markerArray = MarkerArray(markers_array, data_fields=data_fields)  # Ensure data_fields is set

    camera_names = ['cam1', 'cam2']
    quantile_keep_pca = 90
    s_frames = [(0, 10)]

    # Run the smoother without providing smooth_param
    camera_dfs, smooth_params_final = ensemble_kalman_smoother_multicam(
        marker_array=markerArray,
        keypoint_names=keypoint_names,
        smooth_param=None,
        quantile_keep_pca=quantile_keep_pca,
        camera_names=camera_names,
        s_frames=s_frames,
        avg_mode='median',
        inflate_vars=False,
    )

    # Assertions
    assert smooth_params_final is not None, "Expected smooth_param_final to be not None"
    assert isinstance(smooth_params_final, float), \
        f"Expected smooth_param_final to be a float, got {type(smooth_params_final)}"


def test_inflate_variance():

    # ------------------------------------------------------------------------
    # test no inflation
    # ------------------------------------------------------------------------
    n_time = 10
    n_cams = 3
    maha_dict = {c: np.ones((n_time, 1)) for c in range(n_cams)}
    v = np.ones((n_time, 2 * n_cams))
    v_new, inflated = inflate_variance(v=v, maha_dict=maha_dict, threshold=5, scalar=2)
    assert not inflated
    assert np.allclose(v, v_new)

    # ------------------------------------------------------------------------
    # test inflation
    # ------------------------------------------------------------------------
    scalar = 2
    v_new, inflated = inflate_variance(v=v, maha_dict=maha_dict, threshold=0.5, scalar=scalar)
    assert inflated
    assert np.allclose(scalar * v, v_new)

    # ------------------------------------------------------------------------
    # test inflation for one view but not the other (>2 cams)
    # ------------------------------------------------------------------------
    n_cams = 3
    maha_dict = {
        0: np.ones((n_time, 1)),
        1: 2 * np.ones((n_time, 1)),
        2: 4 * np.ones((n_time, 1)),
    }
    v = np.ones((n_time, 2 * n_cams))
    scalar = 3
    v_new, inflated = inflate_variance(v=v, maha_dict=maha_dict, threshold=1.5, scalar=scalar)
    assert inflated
    assert np.allclose(v[:, :2], v_new[:, :2])
    assert np.allclose(scalar * v[:, 2:4], v_new[:, 2:4])
    assert np.allclose(scalar * v[:, 4:], v_new[:, 4:])

    # ------------------------------------------------------------------------
    # redo, but with 2 view; variance should be inflated in both
    # ------------------------------------------------------------------------
    n_cams = 2
    maha_dict = {
        0: np.ones((n_time, 1)),
        1: 2 * np.ones((n_time, 1)),
    }
    v = np.ones((n_time, 2 * n_cams))
    scalar = 3
    v_new, inflated = inflate_variance(v=v, maha_dict=maha_dict, threshold=1.5, scalar=scalar)
    assert inflated
    assert np.allclose(scalar * v, v_new)
