import os
import numpy as np
import pandas as pd


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


# ---------------------------------------------
# Loading + Formatting CSV<->DataFrame
# ---------------------------------------------


def format_csv(csv_dir, data_type='lp'):
    csv_files = os.listdir(csv_dir)
    markers_list = []

    # Extracting markers from data
    # Applies correct format conversion and stores each file's markers in a list
    for csv_file in csv_files:
        if not csv_file.endswith('csv'):
            continue
        markers_curr = pd.read_csv(os.path.join(csv_dir, csv_file), header=[0, 1, 2], index_col=0)
        keypoint_names = [c[1] for c in markers_curr.columns[::3]]
        model_name = markers_curr.columns[0][0]
        if data_type == 'lp':
            markers_curr_fmt = convert_lp_dlc(markers_curr, keypoint_names, model_name=model_name)
        else:
            markers_curr_fmt = markers_curr

        markers_list.append(markers_curr_fmt)

    if len(markers_list) == 0:
        raise FileNotFoundError(f'No marker csv files found in {csv_dir}')

    markers_eks = make_output_dataframe(markers_curr)

    # returns both the formatted marker data and the empty dataframe for EKS output
    return markers_list, markers_eks


# Making empty DataFrame for EKS output
def make_output_dataframe(markers_curr):
    markers_eks = markers_curr.copy()
    markers_eks.columns = markers_eks.columns.set_levels(['ensemble-kalman_tracker'], level=0)
    for col in markers_eks.columns:
        if col[-1] == 'likelihood':
            # set this to 1.0 so downstream filtering functions don't get
            # tripped up
            markers_eks[col].values[:] = 1.0
        else:
            markers_eks[col].values[:] = np.nan

    return markers_eks


def populate_output_dataframe(keypoint_df, keypoint_ensemble, markers_eks):
    for coord in ['x', 'y', 'zscore']:
        src_cols = ('ensemble-kalman_tracker', f'{keypoint_ensemble}', coord)
        dst_cols = ('ensemble-kalman_tracker', f'{keypoint_ensemble}', coord)
        markers_eks.loc[:, dst_cols] = keypoint_df.loc[:, src_cols]
    return markers_eks
