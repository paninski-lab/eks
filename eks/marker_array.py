import numpy as np
import jax.numpy as jnp

class MarkerArray:
    def __init__(self, array: np.ndarray | jnp.ndarray, data_fields: list[str]):
        """
        Initialize MarkerArray with a structured numpy or JAX array.

        Args:
            array: NumPy/JAX array of shape (n_models, n_cameras, n_frames, n_keypoints, n_fields).
            data_fields: List of strings specifying what each channel in the last axis represents.
                         ['x', 'y', 'likelihood'] or ['x', 'y', 'var_x', 'var_y', 'likelihood'].

        Raises:
            AssertionError: If array dimensions do not match expected format.
        """
        assert isinstance(array, (np.ndarray, jnp.ndarray)), "Input must be a NumPy or JAX array."
        assert array.ndim == 5, \
            "Expected shape (n_models, n_cameras, n_frames, n_keypoints, n_fields)."
        assert len(data_fields) == array.shape[-1], \
            "data_fields length must match last axis of array."

        self.array = array
        self.data_fields = data_fields
        self.n_models, self.n_cameras, self.n_frames, self.n_keypoints, self.n_fields = array.shape

    ## --- Data Access Methods --- ##

    def get_shape(self):
        """Retrieve shape of array. """
        return self.array.shape

    def get_model(self, model_idx: int):
        """Retrieve data for a specific model. """
        tmp_array = self.array[model_idx]
        tmp_array = tmp_array[None]
        return MarkerArray(tmp_array, self.data_fields)
        # Shape: (n_cameras, n_frames, n_keypoints, n_fields)

    def get_camera(self, camera_idx: int):
        """Retrieve data for a specific camera."""
        tmp_array = self.array[:, camera_idx]
        tmp_array = tmp_array[:, None]
        return MarkerArray(tmp_array, self.data_fields)
        # Shape: (n_models, n_frames, n_keypoints, n_fields)

    def get_frame(self, frame_idx: int):
        """Retrieve all data for a specific frame across all models and cameras."""
        tmp_array = self.array[:, :, frame_idx]
        tmp_array = tmp_array[:, :, None]
        return MarkerArray(tmp_array, self.data_fields)
        # Shape: (n_models, n_cameras, n_keypoints, n_fields)

    def get_keypoint(self, keypoint_idx: int):
        """Retrieve all data for a specific keypoint across all models, cameras, and frames."""
        tmp_array = self.array[:, :, :, keypoint_idx]
        tmp_array = tmp_array[:, :, :, None]
        return MarkerArray(tmp_array, self.data_fields)
        # Shape: (n_models, n_cameras, n_frames, n_fields)

    def get_point(self, model_idx: int, camera_idx: int, frame_idx: int, keypoint_idx: int):
        """Retrieve all stored values for a specific keypoint at a given frame."""
        return self.array[model_idx, camera_idx, frame_idx, keypoint_idx]
        # Shape: (n_fields,)

    ## --- Data Field Specific Methods --- ##
    def get_field(self, field_name: str):
        """Retrieve all data for a specific field (e.g., 'x', 'y', 'likelihood', 'var_x', 'var_y')."""
        assert field_name in self.data_fields, f"Field '{field_name}' not found in data_fields."
        field_idx = self.data_fields.index(field_name)
        return self.array[..., field_idx]
        # Shape: (n_models, n_cameras, n_frames, n_keypoints)

    def get_fields(self, field_names: list[str]):
        """Retrieve multiple fields as a new MarkerArray."""
        field_indices = [self.data_fields.index(f) for f in field_names]
        return MarkerArray(self.array[..., field_indices], field_names)

    ## --- Convert to JAX Array --- ##
    def jaxify(self):
        """Return a JAX version of the MarkerArray."""
        return MarkerArray(jnp.array(self.array), self.data_fields)

    def __repr__(self):
        return f"MarkerArray(shape={self.array.shape}, data_fields={self.data_fields}, " \
               f"type={'JAX' if isinstance(self.array, jnp.ndarray) else 'NumPy'})"


def input_dfs_to_markerArray(input_dfs_list, bodypart_list, camera_names):
    """
    Converts input_dfs_list (list of list of DataFrames) into a NumPy array
    with shape (n_models, n_cameras, n_frames, n_keypoints, 3).
    """
    # Get dimensions
    n_keypoints = len(bodypart_list)  # Number of keypoints
    n_cameras = len(camera_names)  # Number of cameras
    n_models = len(input_dfs_list[0])  # Number of models
    n_frames = input_dfs_list[0][0].shape[0]

    # Initialize array
    marker_array = np.zeros((n_models, n_cameras, n_frames, n_keypoints, 3))

    # Fill the array
    for c, camera_name in enumerate(camera_names):
        for m in range(n_models):
            model_df = input_dfs_list[c][m]
            for k, keypoint in enumerate(bodypart_list):
                # Extract keypoint-specific columns
                marker_array[m, c, :, k, 0] = model_df[f"{keypoint}_x"].to_numpy()
                marker_array[m, c, :, k, 1] = model_df[f"{keypoint}_y"].to_numpy()
                marker_array[m, c, :, k, 2] = model_df[f"{keypoint}_likelihood"].to_numpy()

    # Convert to MarkerArray
    marker_array = MarkerArray(marker_array, data_fields=["x", "y", "likelihood"])
    return marker_array
