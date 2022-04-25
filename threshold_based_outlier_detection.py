import itertools
import multiprocessing as mp
import os
from typing import List

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

random_seed = 1
np.random.seed(random_seed)


def threshold_outlier_prediction(input_df, window, center_window,
                                 method):
    od_df = input_df.copy()
    if method == 'mean':
        if window is None:
            od_df['x_hat'] = od_df['water_level'].mean()
        else:
            od_df['x_hat'] = od_df['water_level'].rolling(window=window,
                                                          center=center_window,
                                                          min_periods=1).mean()
        od_df['result'] = np.abs(od_df['water_level'] - od_df['x_hat'])
    elif method == 'median':
        if window is None:
            od_df['x_hat'] = od_df['water_level'].median()
        else:
            od_df['x_hat'] = od_df['water_level'].rolling(window=window,
                                                          center=center_window,
                                                          min_periods=1).median()
        od_df['result'] = np.abs(od_df['water_level'] - od_df['x_hat'])
    elif method == 'mad':
        if window is None:
            od_df['x_hat'] = np.median(
                np.abs(od_df['water_level'] - np.median()))
        else:
            od_df['x_hat'] = od_df['water_level'].rolling(window=window,
                                                          center=center_window,
                                                          min_periods=1).apply(
                lambda x: np.median(np.abs(x - np.median(x))))
        od_df['result'] = np.abs(od_df['water_level'] - od_df['x_hat'])
    elif method == 'z-score':
        if window is None:
            od_df['mean'] = od_df['water_level'].mean()
            od_df['std'] = od_df['water_level'].std()
        else:
            od_df['mean'] = od_df['water_level'].rolling(window=window,
                                                         center=center_window,
                                                         min_periods=1).mean()
            od_df['std'] = od_df['water_level'].rolling(window=window,
                                                        center=center_window,
                                                        min_periods=1).std()
        od_df['result'] = (od_df['water_level'] - od_df['mean']) / od_df['std']
    elif method == 'delta-z-score':
        od_df['water_level_delta'] = od_df['water_level'].diff().fillna(0)
        if window is None:
            od_df['mean'] = od_df['water_level_delta'].mean()
            od_df['std'] = od_df['water_level_delta'].std()
        else:
            od_df['mean'] = od_df['water_level_delta'].rolling(window=window,
                                                               center=center_window,
                                                               min_periods=1).mean()
            od_df['std'] = od_df['water_level_delta'].rolling(window=window,
                                                              center=center_window,
                                                              min_periods=1).std()
        od_df['result'] = (od_df['water_level_delta'] - od_df['mean']) / od_df[
            'std']
    elif method == 'mad-z-score':
        if window is None:
            od_df['median'] = od_df['water_level'].median()
            od_df['mad'] = np.median(
                np.abs(od_df['water_level'] - od_df['median']))
            od_df['madn'] = od_df['mad'] / 0.6745
        else:
            od_df['median'] = od_df['water_level'].rolling(window=window,
                                                           min_periods=1,
                                                           center=center_window).median()
            od_df['mad'] = od_df['water_level'].rolling(window=window,
                                                        min_periods=1,
                                                        center=center_window).apply(
                lambda x: np.median(np.abs(x - np.median(x))))
            od_df['madn'] = od_df['mad'] / 0.6745
        od_df['result'] = (od_df['water_level'] - od_df['median']).divide(
            od_df['madn'])
    else:
        raise ValueError(f'Method ({method}) not supported')
    return {'window_size': window, 'center_window': center_window,
            'method': method, 'df': od_df}


def run_grid_search_parallely(df: pd.DataFrame, windows: list,
                              center_windows: list, methods: List[str],
                              target_dir: str):
    with mp.Pool(processes=12) as executor:
        results = executor.starmap(threshold_outlier_prediction,
                                   itertools.product([df], windows,
                                                     center_windows, methods))

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    for res in results:
        res['df'].to_parquet(
            f'{target_dir}{res["window_size"]}_'
            f'{"cw" if res["center_window"] else "nocw"}_'
            f'{res["method"]}.parquet')


stations_df = pd.read_csv('./data/stations.csv')
stations_dict = stations_df.groupby(['common_id']).first().to_dict('index')

# common_ids = ['39003-ie', '2386-ch', '42960105-de', '2720050000-de', '36022-ie']
common_ids = ['36022-ie']
methods = ['median', 'mean', 'mad', 'z-score', 'delta-z-score', 'mad-z-score']
windows = [None] + list(range(2, 51))
center_windows = [False, True]
# regular
for common_id in common_ids:
    print(f'Processing {common_id} (regular, not preprocessed)')
    tex_plots_path = f'../bachelor-thesis/plots/pdfs/{common_id}/'
    df = pd.read_parquet(
        f'data/classified_raw/{common_id}_outliers_classified.parquet')
    run_grid_search_parallely(df, windows, center_windows, methods,
                              f'./data/predictions/raw/regular/{common_id}/')
    print(f'Processing {common_id} (regular, preprocessed)')
    df = pd.read_parquet(
        f'data/classified/{common_id}_outliers_classified.parquet')
    run_grid_search_parallely(df, windows, center_windows, methods,
                              f'./data/predictions/raw_preprocessed/regular/{common_id}/')

    print(f'Processing {common_id} (regular, not preprocessed)')
    df = pd.read_parquet(
        f'data/classified_raw/{common_id}_outliers_classified.parquet')
    scaler = StandardScaler()
    df['water_level'] = scaler.fit_transform(df[['water_level']])
    run_grid_search_parallely(df, windows, center_windows, methods,
                              f'./data/predictions/raw/normalized/{common_id}/')
    print(f'Processing {common_id} (regular, preprocessed)')
    df = pd.read_parquet(
        f'data/classified/{common_id}_outliers_classified.parquet')
    scaler = StandardScaler()
    df['water_level'] = scaler.fit_transform(df[['water_level']])
    run_grid_search_parallely(df, windows, center_windows, methods,
                              f'./data/predictions/raw_preprocessed/normalized/{common_id}/')
