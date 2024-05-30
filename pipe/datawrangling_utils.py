import pandas as pd
import numpy as np

def remove_outliers(df, column_name, threshold=3):
    """
    Remove rows with outliers in a specific column of a DataFrame.
    
    Parameters:
        - df: pandas DataFrame
            The DataFrame containing the data.
        - column_name: str
            The name of the column to check for outliers.
        - threshold: int or float, optional (default=3)
            The number of standard deviations away from the mean to consider as an outlier.
    
    Returns:
        - pandas DataFrame
            The DataFrame with the outliers removed.
    """
    mean = np.mean(df[column_name])
    std = np.std(df[column_name])
    lower_bound = mean - threshold * std
    upper_bound = mean + threshold * std
    
    return df[(df[column_name] >= lower_bound) & (df[column_name] <= upper_bound)]

def is_high_season(date):
    year = date.year
        #verano, asueto revolución, invierno, asueto constitución, asueto petróleo
    if (pd.Timestamp(year=year, month=7, day=17) <= date <= pd.Timestamp(year=year, month=8, day=27)) or \
       (pd.Timestamp(year=year, month=11, day=18) <= date <= pd.Timestamp(year=year, month=11, day=20)) or \
       (pd.Timestamp(year=year, month=12, day=18) <= date <= pd.Timestamp(year=year+1, month=1, day=5)) or \
       (pd.Timestamp(year=year, month=2, day=3) <= date <= pd.Timestamp(year=year, month=2, day=5)) or \
       (pd.Timestamp(year=year, month=3, day=16) <= date <= pd.Timestamp(year=year, month=3, day=18)) or \
        (pd.Timestamp(year=year, month=3, day=25) <= date <= pd.Timestamp(year=year, month=4, day=7)):
        return 'temporada alta'
    return 'temporada baja'
