#%%
import pandas as pd
import pyodbc
import numpy as np
import json
import utils.SECRETS as SECRETS
from utils.datawrangling_utils import remove_outliers, is_high_season 

#%%Getting run parameters

# Load the run information from the JSON file
with open(r'utils\run_parameters.json', 'r') as f:
    run_data = json.load(f)

today = run_data["timestamp"]
run_uuid = run_data["uuid"]


#%% Performing the query

conn_str = (
    f"DRIVER=ODBC Driver 17 for SQL Server;"
    f"SERVER={SECRETS.SERVER};"
    f"DATABASE={SECRETS.DATABASE};"
    f"UID={SECRETS.UID};"
    f"PWD={SECRETS.PWD}"
)

conn = pyodbc.connect(conn_str)

with open(r'utils\Query-Reservaciones.sql', 'r') as file:
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
df.to_parquet(f'..\data\Base_Data\Transformed_df_{today}_{run_uuid}.parquet')