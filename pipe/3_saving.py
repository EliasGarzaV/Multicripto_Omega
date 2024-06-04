#%%
import pandas as pd
import pyodbc
import numpy as np
import json
import utils.SECRETS as SECRETS
from utils.datawrangling_utils import load_df_to_sql_server

#%%Getting run parameters

# Load the run information from the JSON file
with open(r'utils\run_parameters.json', 'r') as f:
    run_data = json.load(f)

today = run_data["timestamp"]
run_uuid = run_data["uuid"]


#%%Loading Clustering results

df = pd.read_parquet(f'..\data\Clustered_Data\Clustered_df_{today}_{run_uuid}.parquet')

#%%Creating Ocupation Dataframe
df_ocupation = df[['FechaLlegada', 'FechaSalida']].copy()

df_ocupation['FechaLlegada'] = pd.to_datetime(df_ocupation['FechaLlegada'])
df_ocupation['FechaSalida'] = pd.to_datetime(df_ocupation['FechaSalida'])

# Create a date range for each reservation
df_ocupation['stay_dates'] = df_ocupation.apply(
    lambda row: pd.date_range(start=row['FechaLlegada'], end=row['FechaSalida'] - pd.Timedelta(days=1)), axis=1
)

# Explode the stay_dates into separate rows
stay_dates_exploded = df_ocupation.explode('stay_dates')

# Count the number of rooms occupied for each date
occupied_rooms = stay_dates_exploded['stay_dates'].value_counts().sort_index()

# Create a DataFrame to display the results
occupied_rooms_df = occupied_rooms.reset_index()
occupied_rooms_df.columns = ['date', 'rooms_occupied']

#%%Combining it with reservations df

# Function to calculate the sum of occupation between arrival and departure
def calculate_total_occupation(arrival, departure, occupation_df):
    mask = (occupation_df['date'] >= arrival) & (occupation_df['date'] <= departure)
    return occupation_df.loc[mask, 'rooms_occupied'].sum() / (occupation_df.loc[mask, 'rooms_occupied'].count() * 735)

# Apply the function to each row in reservations_df
df['OCUPACION'] = df.apply(lambda row: calculate_total_occupation(row['FechaLlegada'],
                                                                         row['FechaSalida'],
                                                                         occupied_rooms_df), axis=1)
#%%Saving backup table and writing to Database

df.to_parquet(f'..\data\Historic_Data\Historic_df_{today}_{run_uuid}.parquet')

conn_str = (
    f"DRIVER=ODBC Driver 17 for SQL Server;"
    f"SERVER={SECRETS.SERVER};"
    f"DATABASE={SECRETS.DATABASE};"
    f"UID={SECRETS.UID};"
    f"PWD={SECRETS.PWD}"
)

load_df_to_sql_server(df.astype(str), 'iar_ClusterResults', conn_str, overwrite=True)

conn_str = None

#%%

