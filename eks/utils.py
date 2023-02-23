import pandas as pd

def make_dlc_pandas_index(keypoint_names):
    xyl_labels = ["x", "y", "likelihood"]
    pdindex = pd.MultiIndex.from_product(
        [["ensemble-kalman_tracker"], keypoint_names, xyl_labels],
        names=["scorer", "bodyparts", "coords"],
    )
    return pdindex

def convert_lp_dlc(df_lp, keypoint_names, model_name='heatmap_tracker'):
    df_dlc = {} 
    for feat in keypoint_names: 
        for feat2 in ['x', 'y', 'likelihood']: 
            df_dlc[f'{feat}_{feat2}'] = df_lp.loc[:, (model_name, feat, feat2)] 
    df_dlc = pd.DataFrame(df_dlc, index=df_lp.index)
    return df_dlc