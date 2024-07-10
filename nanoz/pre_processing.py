#!/usr/bin/env python3

import logging

import pandas as pd


class SkipData(object):
    """
    Skip a percentage of rows from the end of a DataFrame.

    The `DataSkip` class is used to skip a specified percentage of rows from the end of a DataFrame. It modifies the
    DataFrame in-place by dropping the specified number of rows from the end.

    Attributes
    ----------
    percentage : float, optional
        The percentage of rows to skip from the end of the DataFrame. Default is 0.

    TODO: Add support for skipping rows from the beginning or in random order (need to handle available idx).
    """
    def __init__(self, **kwargs):
        """
        Initializes a new instance of the `DataSkip` class.

        Parameters
        ----------
        percentage : float, optional
            The percentage of rows to skip from the end of the DataFrame. Default is 0.
        """
        self.percentage = kwargs.get("percentage", 0)

    def __call__(self, df):
        """
        Skip a percentage of rows from the end of the DataFrame.

        Parameters
        ----------
        df : pandas.DataFrame
            The input DataFrame to skip rows from.

        Returns
        -------
        pandas.DataFrame
            The modified DataFrame with skipped rows.
        """
        if self.percentage > 0:
            size = df.shape[0]
            # skipfooter option is unsupported with engine='c', and engine 'c' is faster than 'python' engine.
            df.drop(df.tail(int(self.percentage / 100 * df.shape[0])).index, inplace=True)
            logging.debug(f"Skipped {size - df.shape[0]} rows ({self.percentage}%) from the end of the dataset.")
        else:
            logging.debug(f"Data skip percentage is zero. No rows will be skipped.")
        return df


class LimitOfDetection(object):
    """
    Applies the limit of detection (LOD) filter to a DataFrame for specified gases.

    The `LimitOfDetection` class is used to apply a filter to a DataFrame by setting the values of gases below the LOD
    threshold to zero. The LOD threshold is defined for each gas. The class takes a DataFrame as input and modifies it
    in-place.

    Attributes
    ----------
    gases : list or None, optional
        A list of gases for which the LOD filter should be applied. If `None`, no filtering is performed.

    lod : list or None, optional
        A list of LOD thresholds corresponding to the gases. The order of LOD values should match the order of gases.
        If `None`, no filtering is performed.
    """
    def __init__(self, **kwargs):
        """
        Initializes a new instance of the `LimitOfDetection` class.

        Parameters
        ----------
        gases : list or None, optional
            A list of gases for which the LOD filter should be applied. If `None`, no filtering is performed.

        lod : list or None, optional
            A list of LOD thresholds corresponding to the gases.
            The order of LOD values should match the order of gases.
            If `None`, no filtering is performed.

        new_value : int, optional
            The value to replace the values below the LOD threshold with. Default is 0.

        """
        self.gases = kwargs.get("gases", None)
        self.lod = kwargs.get("lod", None)
        self.new_value = kwargs.get("new_value", 0)

    def __call__(self, df):
        """
        Applies the LOD filter to the input DataFrame.

        Parameters
        ----------
        df : pandas.DataFrame
            The input DataFrame to be filtered.

        Returns
        -------
        pandas.DataFrame
            The filtered DataFrame with values below the LOD threshold set to the new_value.

        """
        if self.gases and self.lod and len(self.gases) == len(self.lod):
            for gas, lod in zip(self.gases, self.lod):
                df.loc[df[gas] < lod, gas] = self.new_value
            logging.debug(f"LOD applied to gases: {self.gases} with {self.lod}.")
            return df
        else:
            logging.warning(f"LOD not applied. Please provide gases and LOD values.")
            return df


class OrdinalEncoderFromInterval(object):
    """
    Apply ordinal encoding to numerical values based on defined intervals.

    The `OrdinalEncoderFromInterval` class is used to apply ordinal encoding to numerical values in a DataFrame based on
    predefined intervals. It modifies the DataFrame in-place by replacing the numerical values with corresponding
    ordinal codes based on the intervals.

    Attributes
    ----------
    gases : list, optional
        The list of column names representing the gases to encode. Default is None.
    intervals : list of tuples, optional
        The list of intervals defining the mapping from numerical values to ordinal codes. Each interval is defined as
        a tuple (start, end), where start and end are the inclusive lower and upper bounds of the interval,
        respectively. Default is None.
    """
    def __init__(self, **kwargs):
        """
       Initialize a new instance of the `OrdinalEncoderFromInterval` class.

       Parameters
       ----------
       gases : list, optional
           The list of column names representing the gases to encode. Default is None.
       intervals : list of tuples, optional
           The list of intervals defining the mapping from numerical values to ordinal codes. Each interval is defined
           as a tuple (start, end), where start and end are the inclusive lower and upper bounds of the interval,
           respectively. Default is None.
       """
        self.gases = kwargs.get("gases", None)
        self.intervals = kwargs.get("intervals", None)
        if self.intervals:
            self.intervals = pd.IntervalIndex.from_tuples([tuple(interval) for interval in self.intervals])
        else:
            pass  # TODO: Add support for automatic interval generation.

    def __call__(self, df):
        """
        Apply ordinal encoding to numerical values in the DataFrame.

        Parameters
        ----------
        df : pandas.DataFrame
            The input DataFrame to apply ordinal encoding to.

        Returns
        -------
        pandas.DataFrame
            The modified DataFrame with ordinal encoded values.
        """
        if self.gases and self.intervals is not None:
            for gas in self.gases:
                df[gas+"_raw"] = df[gas]
                df[gas] = pd.cut(df[gas], bins=self.intervals).cat.codes
                if any(df[gas] == -1):
                    logging.debug(f"Indexes list of ordinal encoding resulted in -1: {df.index[df[gas] == -1]}")
                    logging.warning(f"Ordinal encoding applied to {gas} resulted in -1 values. "
                                    f"Please check the intervals.")
            logging.debug(f"Ordinal encoding applied to gases: {self.gases} with {self.intervals}.")
            return df
        else:
            logging.warning(f"Ordinal encoding not applied. Please provide a list of gases and intervals as a list of "
                            f"list with start and end values.")
            return df
