import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import jax.numpy as jnp

from jax import device_put, vmap
from sleap_io.io.slp import read_labels

def make_dlc_pandas_index(keypoint_names, labels=["x", "y", "likelihood"]):
    """
    Create a pandas MultiIndex for DLC data.

    Parameters:
        keypoint_names (list of str): List of keypoint names.
        labels (list of str): List of labels for each keypoint.

    Returns:
        pd.MultiIndex: MultiIndex object for the DataFrame.
    """
    pdindex = pd.MultiIndex.from_product(
        [["ensemble-kalman_tracker"], keypoint_names, labels],
        names=["scorer", "bodyparts", "coords"],
    )
    return pdindex

def convert_lp_dlc(df_lp, keypoint_names, model_name=None):
    """
    Convert labels-plus (LP) format to DeepLabCut (DLC) format.

    Parameters:
        df_lp (pd.DataFrame): Input DataFrame in LP format.
        keypoint_names (list of str): List of keypoint names.
        model_name (str, optional): Model name for the columns. Defaults to None.

    Returns:
        pd.DataFrame: DataFrame in DLC format.
    """
    df_dlc = {}
    for feat in keypoint_names:
        for feat2 in ['x', 'y', 'likelihood']:
            try:
                if model_name is None:
                    col_tuple = (feat, feat2)
                else:
                    col_tuple = (model_name, feat, feat2)

                # Skip columns with any unnamed level
                if any(level.startswith('Unnamed') for level in col_tuple if isinstance(level, str)):
                    continue

                df_dlc[f'{feat}_{feat2}'] = df_lp.loc[:, col_tuple]
            except KeyError:
                # If the specified column does not exist, skip it
                continue

    df_dlc = pd.DataFrame(df_dlc, index=df_lp.index)
    return df_dlc

def convert_slp_dlc(base_dir, slp_file):
    """
    Convert SLEAP (.slp) file format to DeepLabCut (DLC) format.

    Parameters:
        base_dir (str): Base directory containing the SLEAP file.
        slp_file (str): SLEAP file name.

    Returns:
        pd.DataFrame: DataFrame with converted data.
    """
    filepath = os.path.join(base_dir, slp_file)
    labels = read_labels(filepath)

    max_instances = len(labels[0].instances)
    keypoint_names = [node.name for node in labels[0].instances[0].points.keys()]
    print(keypoint_names)
    num_keypoints = len(keypoint_names)

    num_frames = len(labels.labeled_frames)
    data = np.zeros((num_frames, max_instances, num_keypoints, 3))  # 3 for x, y, likelihood

    for i, labeled_frame in enumerate(labels.labeled_frames):
        for j, instance in enumerate(labeled_frame.instances):
            if j >= max_instances:
                break
            for k, keypoint_node in enumerate(instance.points.keys()):
                point = instance.points[keypoint_node]
                data[i, j, k, 0] = point.x if not np.isnan(point.x) else 0
                data[i, j, k, 1] = point.y if not np.isnan(point.y) else 0
                data[i, j, k, 2] = point.score + 1e-6

    reshaped_data = data.reshape(num_frames, -1)
    columns = []
    for j in range(max_instances):
        for keypoint_name in keypoint_names:
            columns.append(f"{j + 1}_{keypoint_name}_x")
            columns.append(f"{j + 1}_{keypoint_name}_y")
            columns.append(f"{j + 1}_{keypoint_name}_likelihood")

    df = pd.DataFrame(reshaped_data, columns=columns)
    df.to_csv(f'{slp_file}.csv', index=False)
    print(f"File read. See read-in data at {slp_file}.csv")
    return df

def format_data(input_dir, data_type):
    """
    Format data from a directory containing input files.

    Parameters:
        input_dir (str): Directory containing input files.
        data_type (str): Type of data to process ('slp', 'lp', or 'dlc').

    Returns:
        tuple: A tuple containing a list of formatted DataFrames, an empty output DataFrame, and a list of keypoint names.
    """
    input_files = os.listdir(input_dir)
    input_dfs_list = []

    for input_file in input_files:
        if data_type == 'slp':
            if not input_file.endswith('.slp'):
                continue
            markers_curr = convert_slp_dlc(input_dir, input_file)
            keypoint_names = [c[1] for c in markers_curr.columns[::3]]
            markers_curr_fmt = markers_curr
        elif data_type == 'lp' or 'dlc':
            if not input_file.endswith('csv'):
                continue
            markers_curr = pd.read_csv(os.path.join(input_dir, input_file), header=[0, 1, 2], index_col=0)
            keypoint_names = [c[1] for c in markers_curr.columns[::3]]
            model_name = markers_curr.columns[0][0]
            if data_type == 'lp':
                markers_curr_fmt = convert_lp_dlc(markers_curr, keypoint_names, model_name=model_name)
            else:
                markers_curr_fmt = markers_curr

        input_dfs_list.append(markers_curr_fmt)

    if len(input_dfs_list) == 0:
        raise FileNotFoundError(f'No marker input files found in {input_dir}')

    output_df = make_output_dataframe(markers_curr)
    return input_dfs_list, output_df, keypoint_names

def make_output_dataframe(markers_curr):
    """
    Create an empty DataFrame for EKS output.

    Parameters:
        markers_curr (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: An empty DataFrame formatted for EKS output.
    """
    markers_eks = markers_curr.copy()

    if isinstance(markers_eks.columns, pd.MultiIndex):
        markers_eks.columns = markers_eks.columns.set_levels(['ensemble-kalman_tracker'], level=0)
    else:
        new_columns = []
        for col in markers_eks.columns:
            parts = col.split('_')
            instance_num = parts[0]
            keypoint_name = '_'.join(parts[1:-1])
            feature = parts[-1]
            new_columns.append(('ensemble-kalman_tracker', f'{instance_num}_{keypoint_name}', feature))

        markers_eks.columns = pd.MultiIndex.from_tuples(new_columns, names=['scorer', 'bodyparts', 'coords'])

    for col in markers_eks.columns:
        if col[-1] == 'likelihood':
            markers_eks[col].values[:] = 1.0
        else:
            markers_eks[col].values[:] = np.nan

    return markers_eks

def dataframe_to_csv(df, filename):
    """
    Convert a DataFrame to a CSV file.

    Parameters:
        df (pd.DataFrame): The DataFrame to be converted.
        filename (str): The name of the CSV file to be created.

    Returns:
        None
    """
    try:
        df.to_csv(filename, index=False)
    except Exception as e:
        print("Error:", e)

def populate_output_dataframe(keypoint_df, keypoint_ensemble, output_df, key_suffix=''):
    """
    Populate the output DataFrame with the ensemble keypoints.

    Parameters:
        keypoint_df (pd.DataFrame): DataFrame containing the keypoints.
        keypoint_ensemble (str): Name of the keypoint ensemble.
        output_df (pd.DataFrame): Output DataFrame to populate.
        key_suffix (str): Suffix for the keypoint name, used for multi-camera setups. Defaults to ''.

    Returns:
        pd.DataFrame: Populated output DataFrame.
    """
    for coord in ['x', 'y', 'zscore']:
        src_cols = ('ensemble-kalman_tracker', f'{keypoint_ensemble}', coord)
        dst_cols = ('ensemble-kalman_tracker', f'{keypoint_ensemble}' + key_suffix, coord)
        output_df.loc[:, dst_cols] = keypoint_df.loc[:, src_cols]
    return output_df

def plot_results(output_df, input_dfs_list, key, s_final, nll_values, idxs, save_dir, smoother_type):
    """
    Plot the results of the EKS.

def plot_results(output_df, input_dfs_list,
                 key, s_final, nll_values, idxs, save_dir, smoother_type):
    
    if nll_values is None:
        fig, axes = plt.subplots(4, 1, figsize=(9, 10))
    else:
        fig, axes = plt.subplots(5, 1)

    for ax, coord in zip(axes, ['x', 'y', 'likelihood', 'zscore']):
        if coord == 'likelihood':
            ylabel = 'model likelihoods'
        elif coord == 'zscore':
            ylabel = 'EKS disagreement'
        else:
            ylabel = coord

        ax.set_ylabel(ylabel, fontsize=12)
        if coord == 'zscore':
            ax.plot(output_df.loc[slice(*idxs), ('ensemble-kalman_tracker', key, coord)], color='k', linewidth=2)
            ax.set_xlabel('Time (frames)', fontsize=12)
            continue
        for m, markers_curr in enumerate(input_dfs_list):
            ax.plot(markers_curr.loc[slice(*idxs), key + f'_{coord}'], color=[0.5, 0.5, 0.5],
                    label='Individual models' if m == 0 else None)
        if coord == 'likelihood':
            continue
        ax.plot(output_df.loc[slice(*idxs), ('ensemble-kalman_tracker', key, coord)], color='k', linewidth=2, label='EKS')
        if coord == 'x':
            ax.legend()

        # Plot nll_values against the time axis
        if nll_values is not None:
            nll_values_subset = nll_values[idxs[0]:idxs[1]]
            axes[-1].plot(range(*idxs), nll_values_subset, color='k', linewidth=2)
            axes[-1].set_ylabel('EKS NLL', fontsize=12)

    plt.suptitle(f'EKS results for {key}, smoothing = {s_final}', fontsize=14)
    plt.tight_layout()

    save_file = os.path.join(save_dir, f'{smoother_type}_s={s_final}.pdf')
    plt.savefig(save_file)
    plt.close()
    print(f'see example EKS output at {save_file}')

def crop_frames(y, s_frames):
    """
    Crop frames as specified by s_frames to be used for auto-tuning s.

    Parameters:
        y (np.ndarray): Array of data to be cropped.
        s_frames (list of tuples): List of start and end indices for cropping.

    Returns:
        np.ndarray: Concatenated array of cropped frames.
    """
    result = []

    for frame in s_frames:
        start, end = frame
        start = start - 1 if start is not None else 0
        end = end if end is not None else len(y)

        start = max(0, start)
        end = min(len(y), end)

        if start >= end:
            raise ValueError(f"Index range ({start + 1}, {end}) is out of bounds for the list of length {len(y)}.")

        result.append(y[start:end])

    return np.concatenate(result)
