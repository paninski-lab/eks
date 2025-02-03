import numpy as np
import jax.numpy as jnp


class MarkerArray:
    def __init__(self, array):
        """
        Initialize MarkerArray with a structured numpy array.

        Expected shape: (n_models, n_cameras, n_frames, n_keypoints, 3)
        """
        assert isinstance(array, (np.ndarray, jnp.ndarray)), "Input must be a NumPy or JAX array."
        assert array.ndim == 5, "Expected shape (n_models, n_cameras, n_frames, n_keypoints, 3)"
        assert array.shape[-1] == 3, "Last dimension must represent (x, y, likelihood)"

        self.array = array
        self.n_models, self.n_cameras, self.n_frames, self.n_keypoints, _ = array.shape

    ## --- Axis Length Getters --- ##
    def num_models(self):
        """Returns the number of models."""
        return self.n_models

    def num_cameras(self):
        """Returns the number of cameras."""
        return self.n_cameras

    def num_frames(self):
        """Returns the number of frames."""
        return self.n_frames

    def num_keypoints(self):
        """Returns the number of keypoints."""
        return self.n_keypoints

    ## --- Data Access Methods --- ##
    def get_model(self, model_idx):
        """Retrieve data for a specific model."""
        return self.array[model_idx]  # Shape: (n_cameras, n_frames, n_keypoints, 3)

    def get_camera(self, camera_idx):
        """Retrieve data for a specific camera."""
        return self.array[:, camera_idx]  # Shape: (n_models, n_frames, n_keypoints, 3)

    def get_frame(self, frame_idx):
        """Retrieve all data for a specific frame across all models and cameras."""
        return self.array[:, :, frame_idx]  # Shape: (n_models, n_cameras, n_keypoints, 3)

    def get_keypoint(self, keypoint_idx):
        """Retrieve all data for a specific keypoint across all models, cameras, and frames."""
        return self.array[:, :, :, keypoint_idx]  # Shape: (n_models, n_cameras, n_frames, 3)

    def get_point(self, model_idx, camera_idx, frame_idx, keypoint_idx):
        """Retrieve (x, y, likelihood) for a specific keypoint at a given frame."""
        return self.array[model_idx, camera_idx, frame_idx, keypoint_idx]  # Shape: (3,)

    ## --- Convert to JAX Array --- ##
    def jaxify(self):
        """Return a JAX version of the MarkerArray."""
        return MarkerArray(jnp.array(self.array))

    def __repr__(self):
        return f"MarkerArray(shape={self.array.shape}, " \
               f"type={'JAX' if isinstance(self.array, jnp.ndarray) else 'NumPy'})"


def markers_list_to_numpy(markers_list):
    """
    Converts markers_list (list of list of list of DataFrames) into a NumPy array
    with shape (n_models, n_cameras, n_frames, n_keypoints, 3).
    """
    # Get dimensions
    n_keypoints = len(markers_list)  # Number of keypoints
    n_cameras = len(markers_list[0])  # Number of cameras
    n_models = len(markers_list[0][0])  # Number of models
    n_frames = markers_list[0][0][0].shape[0]

    # Initialize array
    markers_array = np.zeros((n_models, n_cameras, n_frames, n_keypoints, 3))

    # Fill the array
    for k, keypoint in enumerate(markers_list):
        for c, camera in enumerate(keypoint):
            for m, model_df in enumerate(camera):
                markers_array[m, c, :, k, :] = model_df.to_numpy()

    return markers_array
