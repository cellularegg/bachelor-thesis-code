from typing import Union

import numpy as np
import pandas as pd


def mean_outlier_detection(input_df: pd.DataFrame,
                           window: Union[int, None],
                           center_window: bool):
    """
    Detects outliers in a dataframe using a (moving) average.
    :param input_df: the input dataframe where the values are stored
                     in the column water_level
    :param window: the size of the window, None if no window should
                   be used
    :param center_window: whether the window should be centered or not
    :return: a copy of the input dataframe where the column result
             should be compared to a threshold to detect outliers
    """
    od_df = input_df.copy()
    if window is None:
        od_df['x_hat'] = od_df['water_level'].mean()
    else:
        od_df['x_hat'] = \
            od_df['water_level'].rolling(window=window,
                                         center=center_window,
                                         min_periods=1).mean()
    od_df['result'] = np.abs(od_df['water_level'] - od_df['x_hat'])
    return od_df


def median_outlier_detection(input_df: pd.DataFrame,
                             window: Union[int, None],
                             center_window: bool):
    """
    Detects outliers in a dataframe using a (moving) average.
    :param input_df: the input dataframe where the values are stored
                     in the column water_level
    :param window: the size of the window, None if no window should
                   be used
    :param center_window: whether the window should be centered or not
    :return: a copy of the input dataframe where the column result
             should be compared to a threshold to detect outliers
    """
    od_df = input_df.copy()
    if window is None:
        od_df['x_hat'] = od_df['water_level'].median()
    else:
        od_df['x_hat'] = \
            od_df['water_level'].rolling(window=window,
                                         center=center_window,
                                         min_periods=1).median()
    od_df['result'] = np.abs(od_df['water_level'] - od_df['x_hat'])
    return od_df


def mad_outlier_detection(input_df: pd.DataFrame,
                          window: Union[int, None],
                          center_window: bool):
    """
    Detects outliers in a dataframe using a (moving) average.
    :param input_df: the input dataframe where the values are stored
                     in the column water_level
    :param window: the size of the window, None if no window should
                   be used
    :param center_window: whether the window should be centered or not
    :return: a copy of the input dataframe where the column result
             should be compared to a threshold to detect outliers
    """
    od_df = input_df.copy()
    if window is None:
        od_df['x_hat'] = np.median(
            np.abs(od_df['water_level'] - np.median(
                od_df['water_level'])))
    else:
        od_df['x_hat'] = \
            od_df['water_level'].rolling(window=window,
                                         center=center_window,
                                         min_periods=1).apply(
                lambda x: np.median(np.abs(x - np.median(x))))
    od_df['result'] = np.abs(od_df['water_level'] - od_df['x_hat'])
    return od_df


def z_score_outlier_detection(input_df: pd.DataFrame,
                              window: Union[int, None],
                              center_window: bool):
    """
    Detects outliers in a dataframe using a (moving) average.
    :param input_df: the input dataframe where the values are stored
                     in the column water_level
    :param window: the size of the window, None if no window should
                   be used
    :param center_window: whether the window should be centered or not
    :return: a copy of the input dataframe where the column result
             should be compared to a threshold to detect outliers
    """
    od_df = input_df.copy()
    if window is None:
        od_df['mean'] = od_df['water_level'].mean()
        od_df['std'] = od_df['water_level'].std()
    else:
        od_df['mean'] = \
            od_df['water_level'].rolling(window=window,
                                         center=center_window,
                                         min_periods=1).mean()
        od_df['std'] = \
            od_df['water_level'].rolling(window=window,
                                         center=center_window,
                                         min_periods=1).std()
    od_df['result'] = \
        (od_df['water_level'] - od_df['mean']).divide(od_df['std'])
    return od_df


def delta_z_score_outlier_detection(input_df: pd.DataFrame,
                                    window: Union[int, None],
                                    center_window: bool):
    """
    Detects outliers in a dataframe using a (moving) average.
    :param input_df: the input dataframe where the values are stored
                     in the column water_level
    :param window: the size of the window, None if no window should
                   be used
    :param center_window: whether the window should be centered or not
    :return: a copy of the input dataframe where the column result
             should be compared to a threshold to detect outliers
    """
    od_df = input_df.copy()
    od_df['water_level_delta'] = od_df['water_level'].diff().fillna(0)
    if window is None:
        od_df['mean'] = od_df['water_level_delta'].mean()
        od_df['std'] = od_df['water_level_delta'].std()
    else:
        od_df['mean'] = \
            od_df['water_level_delta'].rolling(window=window,
                                               center=center_window,
                                               min_periods=1).mean()
        od_df['std'] = \
            od_df['water_level_delta'].rolling(window=window,
                                               center=center_window,
                                               min_periods=1).std()
    od_df['result'] = \
        (od_df['water_level_delta'] - od_df['mean']).divide(
            od_df['std'])
    return od_df


def madn_z_score_outlier_detection(input_df: pd.DataFrame,
                                   window: Union[int, None],
                                   center_window: bool):
    """
    Detects outliers in a dataframe using a (moving) average.
    :param input_df: the input dataframe where the values are stored
                     in the column water_level
    :param window: the size of the window, None if no window should
                   be used
    :param center_window: whether the window should be centered or not
    :return: a copy of the input dataframe where the column result
             should be compared to a threshold to detect outliers
    """
    od_df = input_df.copy()
    if window is None:
        od_df['median'] = od_df['water_level'].median()
        od_df['mad'] = np.median(
            np.abs(od_df['water_level'] - od_df['median']))
        od_df['madn'] = od_df['mad'] / 0.6745
    else:
        od_df['median'] = \
            od_df['water_level'].rolling(window=window,
                                         min_periods=1,
                                         center=center_window).median()
        od_df['mad'] = \
            od_df['water_level'].rolling(window=window,
                                         min_periods=1,
                                         center=center_window).apply(
                lambda x: np.median(np.abs(x - np.median(x))))
        od_df['madn'] = od_df['mad'] / 0.6745
    od_df['result'] = \
        (od_df['water_level'] - od_df['median']).abs() \
            .divide(od_df['madn'])
    return od_df
