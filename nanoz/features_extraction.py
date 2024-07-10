#!/usr/bin/env python3

import logging

import numpy as np
import pandas as pd
from scipy.stats import median_abs_deviation, skew, kurtosis

from nanoz.utils import copy_doc

# TODO: don't use this module, will be integrated for Algoz 3.0.0
# TODO: more options

def _available_features():
    """
    Available features name:
        'MIN': Minimum
        'MAX': Maximum
        'MAD': Median absolute deviation
        'SKEW': Skewness
        'KURT': Kurtosis
    """


@copy_doc(_available_features)
def rolling_features(df, columns_name, features_name, window_size):
    """
    Calculate features from features_name list for the dataframe.
    
    TODO: dispatch dict of features.
    TODO: features extraction in threads.
    
    Parameters
    ----------
    df: DataFrame
        Pandas dataframe.
    columns_name: list
        Features are calculated for these names of the dataframe columns.
    features_name: list
        Features name as string.
        [COPYDOC]
    window_size: int
        Size of the rolling window for calculated features.
    
    Returns
    -------
    DataFrame
        Dataframe with features columns added.
    """

    if 'MIN' in features_name:
        for col in columns_name:
            df[col+'_MIN'] = df[col].rolling(window_size, min_periods=1).min()
        logging.debug('Statistical feature "MIN" calculated')
    
    if 'MAX' in features_name:
        for col in columns_name:
            df[col+'_MAX'] = df[col].rolling(window_size, min_periods=1).max()
        logging.debug('Statistical feature "MAX" calculated')
    
    if 'MAD' in features_name:
        for col in columns_name:
            df[col+'_MAD'] = pd.DataFrame(
                median_abs_deviation(
                    np.lib.stride_tricks.sliding_window_view(df[col].to_numpy(), window_size), axis=-1
                )
            )
        logging.debug('Statistical feature "MAD" calculated')
    
    if 'SKEW' in features_name:
        for col in columns_name:
            df[col+'_SKEW'] = pd.DataFrame(
                skew(
                    np.lib.stride_tricks.sliding_window_view(df[col].to_numpy(), window_size), axis=-1
                )
            )
        logging.debug('Statistical feature "SKEW" calculated')
    
    if 'KURT' in features_name:
        for col in columns_name:
            df[col+'_KURT'] = pd.DataFrame(
                kurtosis(
                    np.lib.stride_tricks.sliding_window_view(df[col].to_numpy(), window_size), axis=-1
                )
            )
        logging.debug('Statistical feature "KURT" calculated')
    return df
