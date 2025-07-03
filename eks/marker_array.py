from typing import List, Optional, Tuple, Union

import jax.numpy as jnp
import numpy as np


class MarkerArray:
    def __init__(
        self,
        array: Optional[Union[np.ndarray, jnp.ndarray]] = None,
        shape: Optional[Tuple[int, int, int, int, int]] = None,
        data_fields: Optional[List[str]] = None,
        marker_array: Optional["MarkerArray"] = None,
        dtype: type = np.float32
    ):
        """
        Initialize a MarkerArray with a structured NumPy or JAX array.

        Supports:
        - Initializing from an existing array.
        - Creating an empty array with a specified shape.
        - Cloning an existing MarkerArray.

        Args:
            array (Optional[Union[np.ndarray, jnp.ndarray]]): NumPy/JAX array with shape
                (n_models, n_cameras, n_frames, n_keypoints, n_fields).
            shape (Optional[Tuple[int, int, int, int, int]]): If `array` is None, this specifies
                the shape of the new array.
            data_fields (Optional[List[str]]): Field names (e.g., ["x", "y", "likelihood"]).
            marker_array (Optional[MarkerArray]): Existing MarkerArray to copy.
            dtype (type): Data type for the new array if created from `shape`.
            Defaults to np.float32.

        Raises:
            AssertionError: If neither `array`, `shape`, nor `marker_array` is provided.
        """
        if marker_array is not None:
            # Clone an existing MarkerArray
            assert isinstance(marker_array, MarkerArray), "marker_array must be a MarkerArray."
            self.array: np.ndarray = np.array(marker_array.array, dtype=dtype)
            self.data_fields = marker_array.data_fields if data_fields is None else data_fields

        elif array is not None:
            assert isinstance(array, (np.ndarray, jnp.ndarray)), \
                "Input must be a NumPy or JAX array."
            assert array.ndim == 5, \
                "Expected shape (n_models, n_cameras, n_frames, n_keypoints, n_fields)."
            self.array: Union[np.ndarray, jnp.ndarray] = array
            self.data_fields: List[str] = data_fields

        elif shape is not None:
            assert len(shape) == 5, \
                "Shape must be (n_models, n_cameras, n_frames, n_keypoints, n_fields)."
            self.array: np.ndarray = np.zeros(shape, dtype=dtype)
            self.data_fields: List[str] = data_fields

        else:
            raise AssertionError("Provide either `array`, `shape`, or `marker_array`.")

        self.n_models, self.n_cameras, self.n_frames, self.n_keypoints, self.n_fields = \
            self.array.shape

        # Create a dictionary mapping names to indices
        self.axis_map: dict[str, int] = {
            "models": 0,
            "cameras": 1,
            "frames": 2,
            "keypoints": 3,
            "fields": 4
        }

    @property
    def shape(self):
        """Returns shape of array."""
        return self.array.shape

    def get_array(self, squeeze=False):
        """Returns array with squeezed singleton axes if squeeze=True."""
        return np.squeeze(self.array) if squeeze else self.array

    def slice(self, axis: str, indices: Union[int, List[int], np.ndarray]) -> "MarkerArray":
        """
        Slice the MarkerArray dynamically along a single named axis.

        Args:
            axis (str): One of "models", "cameras", "frames", "keypoints", or a field name.
            indices (Union[int, List[int], np.ndarray]): Single index or list of indices to keep.

        Returns:
            MarkerArray: A new sliced MarkerArray.
        """
        assert axis in self.axis_map, \
            f"Invalid slice axis: {axis}. Must be one of {list(self.axis_map.keys())}."

        # Convert single integer index to a list
        if isinstance(indices, int):
            indices = [indices]

        sliced_array = np.take(self.array, indices, axis=self.axis_map[axis])
        return MarkerArray(sliced_array, data_fields=self.data_fields)

    def slice_fields(self, *fields: str) -> "MarkerArray":
        """
        Slice the MarkerArray to keep only specific fields.

        Args:
            *fields (str): Field names to keep. Pass multiple fields as separate arguments.

        Returns:
            MarkerArray: A new MarkerArray with only the selected fields.

        MAYBE build-in array extractor + maybe squeezer as well (based on usage)
        """
        # Validate fields
        for field in fields:
            assert field in self.data_fields, \
                f"Field '{field}' not found in data_fields: {self.data_fields}"

        # Get field indices
        field_indices = [self.data_fields.index(field) for field in fields]

        # Slice the last axis (fields)
        sliced_array = np.take(self.array, field_indices, axis=4)

        return MarkerArray(sliced_array, data_fields=list(fields))

    @staticmethod
    def stack(others: List["MarkerArray"], axis: str) -> "MarkerArray":
        """
        Stack multiple MarkerArrays along a specified axis.

        Args:
            others (List[MarkerArray]): List of MarkerArrays to stack together.
            axis (str): One of "models", "cameras", "frames", "keypoints", or a field name.

        Returns:
            MarkerArray: A new MarkerArray with stacked arrays along the specified axis.
        """
        assert len(others) > 0, "At least one MarkerArray must be provided for stacking."

        # Use the first element as reference for shape and data fields
        reference = others[0]
        assert axis in reference.axis_map, \
            f"Invalid stack axis: {axis}. Must be one of {list(reference.axis_map.keys())}."

        # Ensure all MarkerArrays have the same shape except for the stacking axis
        for other in others[1:]:
            assert isinstance(other, MarkerArray), \
                "All elements in 'others' must be MarkerArray instances."
            assert reference.array.shape[:reference.axis_map[axis]] + \
                   reference.array.shape[reference.axis_map[axis] + 1:] \
                   == other.array.shape[:reference.axis_map[axis]] + \
                   other.array.shape[reference.axis_map[axis] + 1:], \
                   f"Shape mismatch: Cannot stack along '{axis}' due to differing dimensions."

        # Stack all arrays along the specified axis
        stacked_array = np.concatenate([other.array for other in others],
                                       axis=reference.axis_map[axis])

        return MarkerArray(stacked_array, data_fields=reference.data_fields)

    def stack_fields(*marker_arrays: "MarkerArray") -> "MarkerArray":
        """
        Stack multiple MarkerArrays along the 'fields' axis, ensuring they are identical
        in all other dimensions.

        Args:
            *marker_arrays (MarkerArray): Variable number of MarkerArray instances to stack.

        Returns:
            MarkerArray: A new MarkerArray with the stacked data along the fields axis.

        Raises:
            AssertionError: If input MarkerArrays have mismatched shapes in axis besides 'fields'.
        """
        assert len(marker_arrays) > 0, "At least one MarkerArray must be provided for stacking."

        # Use the first element as reference for shape and data fields
        reference = marker_arrays[0]

        # Ensure all MarkerArrays have the same shape except for the fields axis
        for other in marker_arrays[1:]:
            assert isinstance(other, MarkerArray), "All inputs must be MarkerArray instances."
            assert reference.array.shape[:4] == other.array.shape[:4], \
                "Shape mismatch: Cannot stack along 'fields' due to differing dimensions."

        # Stack all arrays along the fields axis (last axis, index 4)
        stacked_array = np.concatenate([other.array for other in marker_arrays], axis=4)

        # Concatenate the data_fields from all MarkerArrays
        stacked_fields = []
        for other in marker_arrays:
            assert other.data_fields is not None, "All MarkerArrays must have data_fields defined."
            stacked_fields.extend(other.data_fields)

        return MarkerArray(stacked_array, data_fields=stacked_fields)

    def reorder_data_fields(self, new_order: List[str]) -> "MarkerArray":
        """
        Reorder the fields dimension of the MarkerArray to match the specified order.

        Args:
            new_order (List[str]): List specifying the new order of the data fields.

        Returns:
            MarkerArray: A new MarkerArray with reordered data fields.

        Raises:
            AssertionError: If `new_order` does not have exactly the same fields as `data_fields`.
        """
        assert set(new_order) == set(self.data_fields), \
            f"Mismatch in data fields: Expected {self.data_fields}, but got {new_order}"

        # Get the indices of the new order
        field_indices = [self.data_fields.index(field) for field in new_order]

        # Reorder the last axis (fields)
        reordered_array = np.take(self.array, field_indices, axis=4)

        # Clone with reordered fields
        return MarkerArray(marker_array=self, data_fields=new_order, array=reordered_array)

    def __repr__(self) -> str:
        axis_names = ["models", "cameras", "frames", "keypoints", "fields"]
        shape_str = ", ".join(f"{name}={size}" for name, size in zip(axis_names, self.array.shape))

        return (
            f"MarkerArray({shape_str}, data_fields={self.data_fields}, "
            f"type={'JAX' if isinstance(self.array, jnp.ndarray) else 'NumPy'})"
        )


def input_dfs_to_markerArray(
        input_dfs_list,
        bodypart_list,
        camera_names,
        data_fields=["x", "y", "likelihood"]
):
    """
    Converts input_dfs_list (list of list of DataFrames) into a NumPy array
    with shape (n_models, n_cameras, n_frames, n_keypoints, n_data_fields).
    """
    # Get dimensions
    n_keypoints = len(bodypart_list)  # Number of keypoints
    n_cameras = len(camera_names)  # Number of cameras
    n_models = len(input_dfs_list[0])  # Number of models
    n_frames = input_dfs_list[0][0].shape[0]
    n_fields = len(data_fields)

    # Initialize array
    marker_array = np.zeros((n_models, n_cameras, n_frames, n_keypoints, n_fields))

    # Fill the array
    for c, camera_name in enumerate(camera_names):
        for m in range(n_models):
            model_df = input_dfs_list[c][m]
            for k, keypoint in enumerate(bodypart_list):
                for d, data_field in enumerate(data_fields):
                    marker_array[m, c, :, k, d] = model_df[f"{keypoint}_{data_field}"].to_numpy()

    # Convert to MarkerArray
    marker_array = MarkerArray(marker_array, data_fields=data_fields)
    return marker_array


def mA_to_stacked_array(marker_array, keypoint_idx):
    """
    Reshapes a single-model MarkerArray object into the required format for compute_mahalanobis,
    selecting only a specific keypoint index.

    Args:
        marker_array (np.ndarray): MarkerArray containing multiple fields for keypoints.
            Expected shape: (1, n_cameras, n_frames, n_keypoints, n_fields).
        keypoint_idx (int): Index of the keypoint to extract.

    Returns:
        np.ndarray: Reshaped array with shape (n_frames, n_cameras * n_fields).
    """
    n_models, n_cameras, n_frames, n_keypoints, num_fields = marker_array.shape

    assert 0 <= keypoint_idx < n_keypoints, \
        f"keypoint_idx {keypoint_idx} is out of range (0-{n_keypoints - 1})"

    # Slice by keypoint_idx and extract array Shape: (n_cameras, n_frames, 1, n_fields)
    selected_array = marker_array.slice("keypoints", keypoint_idx).get_array()[0]
    # Reshape into (n_frames, n_cameras * num_fields)
    reshaped_array = selected_array.transpose(1, 0, 2, 3).reshape(-1, n_cameras * num_fields)
    return reshaped_array


def stacked_array_to_mA(reshaped_x, n_cameras, data_fields):
    """
    Reshapes a (n_frames, n_cameras * num_fields) array back into a MarkerArray format of shape
    (1, n_cameras, n_frames, 1, num_fields).

    Args:
        reshaped_x (np.ndarray): Array of shape (n_frames, n_cameras * num_fields),
            where each frame contains concatenated fields (e.g., x, y) for all cameras.
        n_cameras (int): Number of cameras.
        data_fields (str): names of data_fields in MarkerArray output

    Returns:
        np.ndarray: MarkerArray with shape (1, n_cameras, n_frames, 1, num_fields).
    """
    n_frames, total_fields = reshaped_x.shape
    num_fields = total_fields // n_cameras
    assert total_fields % n_cameras == 0, \
        "Input shape mismatch: total fields must be divisible by n_cameras."

    # Reshape into (n_cameras, n_frames, num_fields)
    reshaped_x = reshaped_x.reshape(n_frames, n_cameras, num_fields).transpose(1, 0, 2)
    # Add extra dimensions for model and keypoint
    reshaped_x = reshaped_x[None, :, :, None, :]
    mA_x = MarkerArray(reshaped_x, data_fields=data_fields)
    return mA_x
