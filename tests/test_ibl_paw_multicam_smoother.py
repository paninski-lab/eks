"""Tests for eks.ibl_paw_multicam_smoother."""

import numpy as np

from eks.ibl_paw_multicam_smoother import add_camera_means, remove_camera_means


class TestRemoveCameraAndAddCameraMeans:
    """Test remove_camera_means and add_camera_means."""

    def _make_stacks(self, n_keypoints=2, n_frames=10, n_cameras=2, seed=0):
        """Return a list of (n_frames, n_cameras) arrays, one per keypoint."""
        rng = np.random.default_rng(seed)
        return [rng.standard_normal((n_frames, n_cameras)) for _ in range(n_keypoints)]

    def test_remove_camera_means_subtracts_per_camera(self):
        # Arrange
        stacks = self._make_stacks()
        originals = [s.copy() for s in stacks]
        camera_means = [3.0, 7.0]

        # Act
        result = remove_camera_means(stacks, camera_means)

        # Assert
        for k in range(len(stacks)):
            np.testing.assert_allclose(result[k][:, 0], originals[k][:, 0] - 3.0)
            np.testing.assert_allclose(result[k][:, 1], originals[k][:, 1] - 7.0)

    def test_add_camera_means_adds_per_camera(self):
        # Arrange
        stacks = self._make_stacks()
        originals = [s.copy() for s in stacks]
        camera_means = [3.0, 7.0]

        # Act
        result = add_camera_means(stacks, camera_means)

        # Assert
        for k in range(len(stacks)):
            np.testing.assert_allclose(result[k][:, 0], originals[k][:, 0] + 3.0)
            np.testing.assert_allclose(result[k][:, 1], originals[k][:, 1] + 7.0)

    def test_remove_then_add_is_identity(self):
        # Arrange
        stacks = self._make_stacks()
        originals = [s.copy() for s in stacks]
        camera_means = [3.0, 7.0]

        # Act
        result = add_camera_means(remove_camera_means(stacks, camera_means), camera_means)

        # Assert
        for k in range(len(stacks)):
            np.testing.assert_allclose(result[k], originals[k])
