import pandas as pd
import numpy as np
import pyodbc

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


def get_sql_type(dtype):
    """Returns a string with the type in sql syntaxis"""
    
    if pd.api.types.is_integer_dtype(dtype):
        return 'INT'
    elif pd.api.types.is_float_dtype(dtype):
        return 'FLOAT'
    elif pd.api.types.is_datetime64_any_dtype(dtype):
        return 'DATETIME'
    else:
        return 'VARCHAR(MAX)'
    
def create_table_sql(df, table_name):
    """Creates a new table based on some df schema"""
    cols = []
    for col in df.columns:
        sql_type = get_sql_type(df[col].dtype)
        cols.append(f"[{col}] {sql_type}")
    return f"CREATE TABLE {table_name} ({', '.join(cols)});"

def load_df_to_sql_server(df:pd.DataFrame, table_name:str, conn_str:str, overwrite=False):
    """Uploads a dataframe to the database"""
    
    conn = pyodbc.connect(conn_str)
    cursor = conn.cursor()
    
    table_exists_query = f"""
    IF OBJECT_ID(N'{table_name}', 'U') IS NOT NULL
        SELECT 1
    ELSE
        SELECT 0
    """
    
    cursor.execute(table_exists_query)
    table_exists = cursor.fetchone()[0]
    
    if not table_exists:
        create_table_query = create_table_sql(df, table_name)
        cursor.execute(create_table_query)
        conn.commit()
    
    if overwrite:
        cursor.execute(f"DELETE FROM {table_name}")
    
    for index, row in df.iterrows():
        sql_query = f"INSERT INTO {table_name} ({', '.join(df.columns)}) VALUES ({', '.join(['?'] * len(row))})"
        cursor.execute(sql_query, tuple(row))
    
    conn.commit()
    cursor.close()
    conn.close()
