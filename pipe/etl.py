#%%
import pandas as pd
import pyodbc
import numpy as np
import SECRETS
from datawrangling_utils import remove_outliers, is_high_season 

#%% Performing the query

conn_str = (
    f"DRIVER=ODBC Driver 17 for SQL Server;"
    f"SERVER={SECRETS.SERVER};"
    f"DATABASE={SECRETS.DATABASE};"
    f"UID={SECRETS.UID};"
    f"PWD={SECRETS.PWD}"
)

conn = pyodbc.connect(conn_str)

with open(r'Query-Reservaciones.sql', 'r') as file:
    sql_query = file.read()
    
df = pd.read_sql(sql_query, conn)

conn.close()
conn_str = None

#%% Data Wrangling

df = remove_outliers(df, "IngresoMto", threshold=3)

df['FechaLlegada'] = pd.to_datetime(df['FechaLlegada'])

df['Tipo_temporada'] = df['FechaLlegada'].apply(is_high_season)

df['Diferencia_reservacion_llegada'] = pd.to_datetime(df['FechaLlegada']) - \
                                        pd.to_datetime(df['FechaRegistro'])

#%% Saving results in file system
df.to_parquet(r'..\data\Transformed_df.parquet')