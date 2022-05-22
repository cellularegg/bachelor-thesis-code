import itertools
import multiprocessing as mp
import os
from datetime import datetime
from typing import List

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import outlier_detection_methods as odm

random_seed = 1
np.random.seed(random_seed)


def threshold_outlier_prediction(input_df, window, center_window,
                                 method):
    od_df = input_df.copy()
    if method == 'mean':
        od_df = odm.mean_outlier_detection(od_df, window,
                                           center_window)
    elif method == 'median':
        od_df = odm.median_outlier_detection(od_df, window,
                                             center_window)
    elif method == 'mad':
        od_df = odm.mad_outlier_detection(od_df, window,
                                          center_window)
    elif method == 'z-score':
        od_df = odm.z_score_outlier_detection(od_df, window,
                                              center_window)
    elif method == 'delta-z-score':
        od_df = odm.delta_z_score_outlier_detection(od_df, window,
                                                    center_window)
    elif method == 'madn-z-score':
        od_df = odm.madn_z_score_outlier_detection(od_df, window,
                                                   center_window)
    else:
        raise ValueError(f'Method ({method}) not supported')
    # od_df['result'] = od_df['result'].replace(np.inf, np.nan)
    # od_df['result'] = od_df['result'].replace(-np.inf, np.nan)
    od_df['result'] = od_df['result'].fillna(0)
    od_df['result'] = od_df['result'].replace(np.inf, np.finfo(
        np.float64).max)
    od_df['result'] = od_df['result'].replace(-np.inf, np.finfo(
        np.float64).min)
    return {'window_size': window, 'center_window': center_window,
            'method': method, 'df': od_df}


def run_grid_search_parallely(df: pd.DataFrame, windows: list,
                              center_windows: list,
                              methods: List[str],
                              target_dir: str):
    with mp.Pool(processes=12) as executor:
        results = executor.starmap(threshold_outlier_prediction,
                                   itertools.product([df], windows,
                                                     center_windows,
                                                     methods))

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    for res in results:
        res['df'].to_parquet(
            f'{target_dir}{res["window_size"]}_'
            f'{"cw" if res["center_window"] else "nocw"}_'
            f'{res["method"]}.parquet')


stations_df = pd.read_csv('./data/stations.csv')
stations_dict = stations_df.groupby(['common_id']).first().to_dict(
    'index')

common_ids = ['36022-ie', '39003-ie', '2386-ch', '42960105-de',
              '2720050000-de']
methods = ['median', 'mean', 'mad', 'z-score', 'madn-z-score', 'delta-z-score']
windows = [None] + list(range(2, 51))
center_windows = [False, True]
for common_id in common_ids:
    print(
        f'{datetime.now().isoformat()} - Processing {common_id} (regular, not preprocessed)')
    tex_plots_path = f'../bachelor-thesis/plots/pdfs/{common_id}/'
    df = pd.read_parquet(
        f'data/classified_raw/{common_id}_outliers_classified.parquet')
    run_grid_search_parallely(df, windows, center_windows, methods,
                              f'./data/predictions/raw/regular/{common_id}/')
    print(
        f'{datetime.now().isoformat()} - Processing {common_id} (regular, preprocessed)')
    df = pd.read_parquet(
        f'data/classified/{common_id}_outliers_classified.parquet')
    run_grid_search_parallely(df, windows, center_windows, methods,
                              f'./data/predictions/raw_preprocessed/regular/{common_id}/')

    print(
        f'{datetime.now().isoformat()} - Processing {common_id} (normalized, not preprocessed)')
    df = pd.read_parquet(
        f'data/classified_raw/{common_id}_outliers_classified.parquet')
    scaler = StandardScaler()
    df['water_level'] = scaler.fit_transform(df[['water_level']])
    run_grid_search_parallely(df, windows, center_windows, methods,
                              f'./data/predictions/raw/normalized/{common_id}/')
    print(
        f'{datetime.now().isoformat()} - Processing {common_id} (normalized, preprocessed)')
    df = pd.read_parquet(
        f'data/classified/{common_id}_outliers_classified.parquet')
    scaler = StandardScaler()
    df['water_level'] = scaler.fit_transform(df[['water_level']])
    run_grid_search_parallely(df, windows, center_windows, methods,
                              f'./data/predictions/raw_preprocessed/normalized/{common_id}/')
