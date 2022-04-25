import itertools
import multiprocessing as mp
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

random_seed = 1
np.random.seed(random_seed)


def get_x_hat(X: np.ndarray, method: str, window: int = None,
              center_window: bool = False) -> np.ndarray:
    if window is None:
        if method == 'mean':
            return np.full(X.shape, X.mean())
        elif method == 'median':
            return np.full(X.shape, np.median(X))
        elif method == 'mad':
            return np.full(X.shape, np.median(np.abs(X - np.median(X))))
        else:
            raise ValueError(f'Method {method} not supported')
    else:
        if method == 'mean':
            tmp_df = pd.DataFrame({'X': X.reshape(-1)})
            return tmp_df['X'].rolling(window=window, min_periods=1,
                                       center=center_window).mean().to_numpy().reshape(
                -1, 1)
        elif method == 'median':
            tmp_df = pd.DataFrame({'X': X.reshape(-1)})
            return tmp_df['X'].rolling(window=window, min_periods=1,
                                       center=center_window).median().to_numpy().reshape(
                -1, 1)
        elif method == 'mad':
            tmp_df = pd.DataFrame({'X': X.reshape(-1)})
            return tmp_df['X'].rolling(window=window, min_periods=1,
                                       center=center_window).apply(
                lambda x: np.median(
                    np.abs(x - np.median(x)))).to_numpy().reshape(-1, 1)
        else:
            raise ValueError(f'Method {method} not supported')


def get_z_score(X: np.ndarray, window: int = None,
                center_window: bool = False) -> np.ndarray:
    if window is None:
        return (X - X.mean()) / X.std()
    else:
        # https://stackoverflow.com/questions/47164950/compute-rolling-z-score-in-pandas-dataframe
        x = pd.Series(X.reshape(-1))
        r = x.rolling(window=window, center=center_window)
        m = r.mean().shift(1)
        s = r.std(ddof=0).shift(1)
        z = (x - m) / s
        return z
        # tmp_df = pd.DataFrame({'X': X.reshape(-1)})
        #
        # return tmp_df['X'].rolling(window=window, min_periods=1,
        #                             center=center_window).apply(
        #     lambda x: (x - np.mean(x)) / np.std(x)).to_numpy().reshape(-1, 1)


def get_delta_z_score(X: np.ndarray, window: int = None,
                      center_window: bool = False) -> np.ndarray:
    reshaped_X = X.reshape(-1)
    diff_X = np.diff(reshaped_X, prepend=[reshaped_X[0]])
    if window is None:
        return (diff_X - diff_X.mean()) / diff_X.std()
    else:
        # https://stackoverflow.com/questions/47164950/compute-rolling-z-score-in-pandas-dataframe
        x = pd.Series(diff_X)
        r = x.rolling(window=window, center=center_window)
        m = r.mean().shift(1)
        s = r.std(ddof=0).shift(1)
        z = (x - m) / s
        return z


def get_mad_z_score(X: np.ndarray, window: int = None,
                    center_window: bool = False) -> np.ndarray:
    if window is None:
        median = np.median(X)
        mad = np.median(np.abs(X - median))
        madn = mad / 0.6745
        return (X - median) / madn
    else:
        tmp_df = pd.DataFrame({'X': X.reshape(-1)})
        tmp_df['Median'] = tmp_df['X'].rolling(window=window, min_periods=1,
                                               center=center_window).median()
        tmp_df['MAD'] = tmp_df['X'].rolling(window=window, min_periods=1,
                                            center=center_window).apply(
            lambda x: np.median(np.abs(x - np.median(x))))
        tmp_df['MADN'] = tmp_df['MAD'] / 0.6745
        tmp_df['M'] = tmp_df.apply(
            lambda row: np.abs(row['X'] - row['Median']) / row['MADN'] if row[
                                                                              'MADN'] != 0 else 0,
            axis=1)
        return tmp_df['M'].to_numpy().reshape(-1, 1)


def threshold_outlier_prediction(input_df, window, center_window,
                                 method):
    if method in ['mean', 'median', 'mad']:
        x_hat = get_x_hat(df['water_level'].to_numpy(), method, window,
                          center_window)
        input_df['m'] = (df['water_level'] - x_hat.reshape(-1)).abs()
    elif method == 'z-score':
        z_score = get_z_score(df['water_level'].to_numpy(), window,
                              center_window)
        input_df['m'] = z_score
    elif method == 'delta-z-score':
        z_score = get_delta_z_score(df['water_level'].to_numpy(), window,
                                    center_window)
        input_df['m'] = z_score
    elif method == 'mad-z-score':
        z_score = get_mad_z_score(df['water_level'].to_numpy(), window,
                                  center_window)
        input_df['m'] = z_score
    else:
        raise ValueError(f'Method ({method}) not supported')
    # https://stackoverflow.com/questions/33275461/specificity-in-scikit-learn
    return {'window_size': window, 'center_window': center_window,
            'method': method, 'df': input_df}
    # return {'pred': y_pred, 'truth': y}


stations_df = pd.read_csv('./data/stations.csv')
stations_dict = stations_df.groupby(['common_id']).first().to_dict('index')

common_ids = ['39003-ie', '2386-ch', '42960105-de', '2720050000-de', '36022-ie']
methods = ['median', 'mean', 'mad', 'z-score', 'delta-z-score', 'mad-z-score']
windows = [None] + list(range(2, 50))
center_windows = [False, True]
# regular
for common_id in common_ids:
    print(f'Processing {common_id} (regular)')
    tex_plots_path = f'../bachelor-thesis/plots/pdfs/{common_id}/'
    df = pd.read_parquet(
        f'data/classified/{common_id}_outliers_classified.parquet')
    df.info()
    with mp.Pool(processes=12) as executor:
        results = executor.starmap(threshold_outlier_prediction,
                                   itertools.product([df], windows,
                                                     center_windows, methods))

    fp = f'././data/predictions/raw_preprocessed/regular/{common_id}/'
    if not os.path.exists(fp):
        os.makedirs(fp)
    for res in results:
        res['df'].to_parquet(
            f'{fp}{res["window_size"]}_{"cw" if res["center_window"] else "nocw"}_{res["method"]}.parquet')

# normalized
for common_id in common_ids:
    print(f'Processing {common_id} (regular)')
    df = pd.read_parquet(
        f'data/classified/{common_id}_outliers_classified.parquet')
    df.info()

    scaler = StandardScaler()
    df['water_level'] = scaler.fit_transform(df[['water_level']])

    with mp.Pool(processes=12) as executor:
        results = executor.starmap(threshold_outlier_prediction,
                                   itertools.product([df], windows,
                                                     center_windows, methods))

    fp = f'././data/predictions/raw_preprocessed/normalized/{common_id}/'
    if not os.path.exists(fp):
        os.makedirs(fp)
    for res in results:
        res['df'].to_parquet(
            f'{fp}{res["window_size"]}_{"cw" if res["center_window"] else "nocw"}_{res["method"]}.parquet')
