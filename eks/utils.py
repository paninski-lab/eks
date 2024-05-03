import os
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
            if model_name is None:
                df_dlc[f'{feat}_{feat2}'] = df_lp.loc[:, (feat, feat2)]
            else:
                df_dlc[f'{feat}_{feat2}'] = df_lp.loc[:, (model_name, feat, feat2)]
    df_dlc = pd.DataFrame(df_dlc, index=df_lp.index)

    return df_dlc


def convert_slp_dlc(base_dir, slp_file):
    print(f'Reading {base_dir}/{slp_file}')
    # Read data from .slp file
    filepath = os.path.join(base_dir, slp_file)
    print(filepath)
    labels = read_labels(filepath)

    # Determine the maximum number of instances
    max_instances = len(labels[0].instances)

    data = []  # List to store data for DataFrame
    for i, labeled_frame in enumerate(labels.labeled_frames):
        frame_data = {}  # Dictionary to store data for current frame
        for j, instance in enumerate(labeled_frame.instances):
            # Check if the instance number exceeds the maximum expected
            if j >= max_instances:
                break

            for keypoint_node in instance.points.keys():
                # Extract the name from keypoint_node
                keypoint_name = keypoint_node.name
                # Extract x, y, and likelihood from the PredictedPoint object
                point = instance.points[keypoint_node]

                # Ensure x and y are floats, handle blank entries by converting to 0
                x = point.x  # if not np.isnan(point.x) else 0
                y = point.y  # if not np.isnan(point.y) else 0
                likelihood = point.score + 1e-6

                # Construct the column name based on instance number and keypoint name
                column_name_x = f"{j + 1}_{keypoint_name}_x"
                column_name_y = f"{j + 1}_{keypoint_name}_y"
                column_name_likelihood = f"{j + 1}_{keypoint_name}_likelihood"

                # Add data to frame_data dictionary
                frame_data[column_name_x] = x
                frame_data[column_name_y] = y
                frame_data[column_name_likelihood] = likelihood

        # Append frame_data to the data list
        data.append(frame_data)

    # Create DataFrame from the list of frame data
    df = pd.DataFrame(data)
    df.to_csv(f'{slp_file}.csv', index=False)
    print(f"DataFrame successfully converted to CSV: input.csv")
    return df


# ---------------------------------------------
# Loading + Formatting CSV<->DataFrame
# ---------------------------------------------


def format_data(input_dir, data_type):
    input_files = os.listdir(input_dir)
    markers_list = []
    # Extracting markers from data
    # Applies correct format conversion and stores each file's markers in a list
    for input_file in input_files:

        if data_type == 'slp':
            if not input_file.endswith('.slp'):
                continue
            markers_curr = convert_slp_dlc(input_dir, input_file)
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

        markers_list.append(markers_curr_fmt)

    if len(markers_list) == 0:
        raise FileNotFoundError(f'No marker input files found in {input_dir}')

    markers_eks = make_output_dataframe(markers_curr)
    # returns both the formatted marker data and the empty dataframe for EKS output
    return markers_list, markers_eks


# Making empty DataFrame for EKS output
def make_output_dataframe(markers_curr):
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

    # Iterate over columns and set values
    for col in markers_eks.columns:
        if col[-1] == 'likelihood':
            # Set likelihood values to 1.0
            markers_eks[col].values[:] = 1.0
        else:
            # Set other values to NaN
            markers_eks[col].values[:] = np.nan

    # Write DataFrame to CSV
    output_csv = 'output_dataframe.csv'
    dataframe_to_csv(markers_eks, output_csv)

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
        print(f"DataFrame successfully converted to CSV: {filename}")
    except Exception as e:
        print("Error:", e)


def populate_output_dataframe(keypoint_df, keypoint_ensemble, markers_eks):
    for coord in ['x', 'y', 'zscore']:
        src_cols = ('ensemble-kalman_tracker', f'{keypoint_ensemble}', coord)
        dst_cols = ('ensemble-kalman_tracker', f'{keypoint_ensemble}', coord)
        markers_eks.loc[:, dst_cols] = keypoint_df.loc[:, src_cols]
    return markers_eks
