import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sleap_io.io.slp import read_labels


def make_dlc_pandas_index(keypoint_names, labels=["x", "y", "likelihood"]):
    pdindex = pd.MultiIndex.from_product(
        [["ensemble-kalman_tracker"], keypoint_names, labels],
        names=["scorer", "bodyparts", "coords"],
    )
    return pdindex


def convert_lp_dlc(df_lp, keypoint_names, model_name=None):
    df_dlc = {}
    for feat in keypoint_names:
        for feat2 in ['x', 'y', 'likelihood']:
            try:
                if model_name is None:
                    col_tuple = (feat, feat2)
                else:
                    col_tuple = (model_name, feat, feat2)

                # Skip columns with any unnamed level
                if any(level.startswith('Unnamed') for level in col_tuple if
                       isinstance(level, str)):
                    continue

                df_dlc[f'{feat}_{feat2}'] = df_lp.loc[:, col_tuple]
            except KeyError:
                # If the specified column does not exist, skip it
                continue

    df_dlc = pd.DataFrame(df_dlc, index=df_lp.index)
    return df_dlc


def convert_slp_dlc(base_dir, slp_file):
    # Read data from .slp file
    filepath = os.path.join(base_dir, slp_file)
    labels = read_labels(filepath)

    # Determine the maximum number of instances and keypoints
    max_instances = len(labels[0].instances)
    keypoint_names = [node.name for node in labels[0].instances[0].points.keys()]
    print(keypoint_names)
    num_keypoints = len(keypoint_names)

    # Initialize a NumPy array to store the data
    num_frames = len(labels.labeled_frames)
    data = np.zeros((num_frames, max_instances, num_keypoints, 3))  # 3 for x, y, likelihood

    # Fill the NumPy array with data
    for i, labeled_frame in enumerate(labels.labeled_frames):
        for j, instance in enumerate(labeled_frame.instances):
            if j >= max_instances:
                break
            for k, keypoint_node in enumerate(instance.points.keys()):
                point = instance.points[keypoint_node]
                data[i, j, k, 0] = point.x if not np.isnan(point.x) else 0
                data[i, j, k, 1] = point.y if not np.isnan(point.y) else 0
                # Check if 'score' exists, otherwise leave as 0
                data[i, j, k, 2] = getattr(point, 'score', 0) + 1e-6

    # Reshape data to 2D array for DataFrame creation
    reshaped_data = data.reshape(num_frames, -1)
    columns = []
    for j in range(max_instances):
        for keypoint_name in keypoint_names:
            columns.append(f"{j + 1}_{keypoint_name}_x")
            columns.append(f"{j + 1}_{keypoint_name}_y")
            columns.append(f"{j + 1}_{keypoint_name}_likelihood")

    # Create DataFrame from the reshaped data
    df = pd.DataFrame(reshaped_data, columns=columns)
    df.to_csv(f'{slp_file}.csv', index=False)
    print(f"File read. See read-in data at {slp_file}.csv")
    return df


def format_data(input_source, data_type):
    """
    Load and format input files from a directory or a list of file paths.

    Args:
        input_source (str or list): Directory path or list of file paths.
        data_type (str): Type of data (e.g., 'csv', 'slp').

    Returns:
        input_dfs_list (list): List of formatted DataFrames.
        output_df (DataFrame): Empty DataFrame for storing results.
        keypoint_names (list): List of keypoint names.
    """
    input_dfs_list = []
    keypoint_names = None

    # Determine if input_source is a directory or a list of file paths
    if isinstance(input_source, str) and os.path.isdir(input_source):
        # If it's a directory, list all files in the directory
        input_files = os.listdir(input_source)
        file_paths = [os.path.join(input_source, file) for file in input_files]
    elif isinstance(input_source, list):
        # If it's a list of file paths, use it directly
        file_paths = input_source
    else:
        raise ValueError("input_source must be a directory path or a list of file paths")

    # Process each file based on the data type
    for file_path in file_paths:
        if data_type == 'slp' and file_path.endswith('.slp'):
            markers_curr = convert_slp_dlc(os.path.dirname(file_path), os.path.basename(file_path))
            keypoint_names = [c[1] for c in markers_curr.columns[::3]]
            markers_curr_fmt = markers_curr

        elif data_type in ['lp', 'dlc'] and file_path.endswith('.csv'):
            markers_curr = pd.read_csv(file_path, header=[0, 1, 2], index_col=0)
            keypoint_names = [c[1] for c in markers_curr.columns[::3]]
            model_name = markers_curr.columns[0][0]

            if data_type == 'lp':
                markers_curr_fmt = convert_lp_dlc(markers_curr, keypoint_names,
                                                  model_name=model_name)
            else:
                markers_curr_fmt = markers_curr

        else:
            continue

        input_dfs_list.append(markers_curr_fmt)

    # Check if we found any valid input files
    if len(input_dfs_list) == 0:
        raise FileNotFoundError(f'No valid marker input files found in {input_source}')

    # Create an empty output DataFrame using the last processed DataFrame as a template
    output_df = make_output_dataframe(input_dfs_list[0])

    return input_dfs_list, output_df, keypoint_names


def make_output_dataframe(markers_curr):
    ''' Makes empty DataFrame for EKS output, including x_var and y_var '''
    markers_eks = markers_curr.copy()

    # Check if the columns Index is a MultiIndex
    if isinstance(markers_eks.columns, pd.MultiIndex):
        # Set the first level of the MultiIndex to 'ensemble-kalman_tracker'
        markers_eks.columns = markers_eks.columns.set_levels(['ensemble-kalman_tracker'], level=0)
    else:
        # Convert the columns Index to a MultiIndex with three levels
        new_columns = []

        for col in markers_eks.columns:
            # Extract instance number, keypoint name, and feature from the column name
            parts = col.split('_')
            instance_num = parts[0]
            keypoint_name = '_'.join(parts[1:-1])  # Combine parts for keypoint name
            feature = parts[-1]

            # Construct new column names with desired MultiIndex structure
            new_columns.append(
                ('ensemble-kalman_tracker', f'{instance_num}_{keypoint_name}', feature))

        # Convert the columns Index to a MultiIndex with three levels
        markers_eks.columns = pd.MultiIndex.from_tuples(new_columns,
                                                        names=['scorer', 'bodyparts', 'coords'])

    # Iterate over columns and set initial values for likelihood and variance
    for col in markers_eks.columns:
        if col[-1] == 'likelihood':
            # Set likelihood values to 1.0
            markers_eks[col].values[:] = 1.0
        elif col[-1] in ['x_var', 'y_var']:
            markers_eks[col].values[:] = np.nan
        else:
            # Set other values to NaN
            markers_eks[col].values[:] = np.nan

    return markers_eks


def dataframe_to_csv(df, filename):
    """
    Converts a DataFrame to a CSV file.

    Parameters:
        df (pandas.DataFrame): The DataFrame to be converted.
        filename (str): The name of the CSV file to be created.

    Returns:
        None
    """
    try:
        df.to_csv(filename, index=False)
    except Exception as e:
        print("Error:", e)


def populate_output_dataframe(keypoint_df, keypoint_ensemble, output_df, key_suffix=''):
    # Include 'x', 'y', 'zscore', 'nll', 'x_var', and 'y_var' in the coordinates to transfer
    for coord in ['x', 'y', 'zscore', 'nll', 'x_var', 'y_var']:
        src_cols = ('ensemble-kalman_tracker', f'{keypoint_ensemble}', coord)
        dst_cols = ('ensemble-kalman_tracker', f'{keypoint_ensemble}' + key_suffix, coord)
        output_df.loc[:, dst_cols] = keypoint_df.loc[:, src_cols]

    return output_df


def plot_results(output_df, input_dfs_list,
                 key, s_final, nll_values, idxs, save_dir, smoother_type):
    if nll_values is None:
        fig, axes = plt.subplots(4, 1, figsize=(9, 10))
    else:
        fig, axes = plt.subplots(5, 1)

    for ax, coord in zip(axes, ['x', 'y', 'likelihood', 'zscore']):
        # Rename axes label for likelihood and zscore coordinates
        if coord == 'likelihood':
            ylabel = 'model likelihoods'
        elif coord == 'zscore':
            ylabel = 'EKS disagreement'
        else:
            ylabel = coord

        # plot individual models
        ax.set_ylabel(ylabel, fontsize=12)
        if coord == 'zscore':
            ax.plot(output_df.loc[slice(*idxs), ('ensemble-kalman_tracker', key, coord)],
                    color='k', linewidth=2)
            ax.set_xlabel('Time (frames)', fontsize=12)
            continue
        for m, markers_curr in enumerate(input_dfs_list):
            ax.plot(
                markers_curr.loc[slice(*idxs), key + f'_{coord}'], color=[0.5, 0.5, 0.5],
                label='Individual models' if m == 0 else None,
            )
        # plot eks
        if coord == 'likelihood':
            continue
        ax.plot(
            output_df.loc[slice(*idxs), ('ensemble-kalman_tracker', key, coord)],
            color='k', linewidth=2, label='EKS',
        )
        if coord == 'x':
            ax.legend()

        # Plot nll_values against the time axis
        if nll_values is not None:
            nll_values_subset = nll_values[idxs[0]:idxs[1]]
            axes[-1].plot(range(*idxs), nll_values_subset, color='k', linewidth=2)
            axes[-1].set_ylabel('EKS NLL', fontsize=12)

    plt.suptitle(f'EKS results for {key}, smoothing = {s_final}', fontsize=14)
    plt.tight_layout()
    save_file = os.path.join(save_dir,
                             f'{smoother_type}_{key}.pdf')
    plt.savefig(save_file)
    plt.close()
    print(f'see example EKS output at {save_file}')


def crop_frames(y, s_frames):
    """ Crops frames as specified by s_frames to be used for auto-tuning s."""
    # Create an empty list to store arrays
    result = []

    for frame in s_frames:
        # Unpack the frame, setting defaults for empty start or end
        start, end = frame
        # Default start to 0 if not specified (and adjust for zero indexing)
        start = start - 1 if start is not None else 0
        # Default end to the length of ys if not specified
        end = end if end is not None else len(y)

        # Cap the indices within valid range
        start = max(0, start)
        end = min(len(y), end)

        # Validate the keys
        if start >= end:
            raise ValueError(f"Index range ({start + 1}, {end}) "
                             f"is out of bounds for the list of length {len(y)}.")

        # Use numpy slicing to preserve the data structure
        result.append(y[start:end])

    # Concatenate all slices into a single numpy array
    return np.concatenate(result)
