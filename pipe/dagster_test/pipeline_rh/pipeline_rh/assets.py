from dagster import Definitions, asset
import json
import uuid
from datetime import datetime
import pandas as pd
import pyodbc
import numpy as np
import utils.SECRETS as SECRETS
from utils.datawrangling_utils import remove_outliers, is_high_season, load_df_to_sql_server
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn import mixture 
from kneed import KneeLocator
import pickle


@asset(description='This asset defines the parameters for all the job run.')
def run_params():
    # Generate timestamp and UUID
    run_data = {
        "timestamp": datetime.now().strftime('%Y-%m-%d_H%H-M%M-S%S'),
        "uuid": str(uuid.uuid4())
    }

    # Save to run_parameters file
    with open(r'utils\run_parameters.json', 'w') as f:
        json.dump(run_data, f)
    return "Success"

@asset(deps=[run_params], description='Performs the query to the Azure SQL DB and cleans the data.')
def etl():
    # Load the run information from the JSON file
    with open(r'utils\run_parameters.json', 'r') as f:
        run_data = json.load(f)

    today = run_data["timestamp"]
    run_uuid = run_data["uuid"]

    # Performing the query

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

    #  Data Wrangling

    df = remove_outliers(df, "IngresoMto", threshold=3)

    df['FechaLlegada'] = pd.to_datetime(df['FechaLlegada'])

    df['Tipo_temporada'] = df['FechaLlegada'].apply(is_high_season)

    df['Diferencia_reservacion_llegada'] = pd.to_datetime(df['FechaLlegada']) - \
                                            pd.to_datetime(df['FechaRegistro'])

    #% Saving results in file system
    df.to_parquet(f'data\Base_Data\Transformed_df_{today}_{run_uuid}.parquet')

    return "Success"
    
@asset(deps=[etl], description='Tests various clustering techniques and selects the best fit.')
def clustering():

    # Load the run information from the JSON file
    with open(r'utils\run_parameters.json', 'r') as f:
        run_data = json.load(f)

    today = run_data["timestamp"]
    run_uuid = run_data["uuid"]

    #% Reading transformed Data and defining parameters
    df = pd.read_parquet(f'data\Base_Data\Transformed_df_{today}_{run_uuid}.parquet')

    columns_2_use = ['Tipo_Habitacion', 'Clasificacion_tipo_habitacion', 'Paquete', 'Canal', 'Estatus_res', 'Capacidad_hotel', 'Numero_personas', 'Numero_adultos', 'Numero_noches', 'IngresoMto', 'Tipo_temporada']
    df_selected = df[columns_2_use]

    categorical_features = ['Tipo_Habitacion', 'Clasificacion_tipo_habitacion', 'Paquete', 'Canal', 'Estatus_res', "Tipo_temporada"]
    numeric_features = ['Capacidad_hotel', 'Numero_personas', 'Numero_adultos', 'Numero_noches', 'IngresoMto']

    #% Transforming Data

    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Preprocesar los datos
    data_transformed = preprocessor.fit_transform(df_selected)

    #%Iterating over clustering algorithms

    labels, models = dict(), dict()

    #%K-Means

    # Elbow Analysis
    sum_of_squared_distances = []
    K = range(1, 10)  # Adjust according to needs and resources
    for k in K:
        km = KMeans(n_clusters=k, random_state=42)
        km = km.fit(data_transformed)
        sum_of_squared_distances.append(km.inertia_)
        
    # Find knee using maximum curvature method (KneeLocator)
    kn = KneeLocator(K, sum_of_squared_distances, curve='convex', direction='decreasing')
    n_clusters_optimo = kn.knee if kn.knee else 3  # It uses 3 as default if no knee is found

    # Fitting optimas Kmeans clustering
    kmeans_final = KMeans(n_clusters=n_clusters_optimo, random_state=42)
    kmeans = Pipeline(steps=[('preprocessor', preprocessor),
                            ('clusterer', kmeans_final)])

    kmeans.fit(df_selected)

    #Saving model and results
    labels['K-Means'] = kmeans.predict(df_selected)
    models['K-Means'] = kmeans

    #%Gaussian Mixture

    #'data_transformed' might be sparse due to one-hot encoding
    if isinstance(data_transformed, np.ndarray):
        data_dense = data_transformed  
    else:
        data_dense = data_transformed.toarray()  

    #Knee analisis
    n_components = np.arange(1, 11)  
    bics = []
    for n in n_components:
        gmm = mixture.GaussianMixture(n_components = n, random_state = 13)
        gmm.fit(data_dense)
        bics.append(gmm.bic(data_dense))
        
    knee_bic = KneeLocator(n_components, bics, curve = 'convex', direction = 'decreasing')

    def to_dense(X):
        if isinstance(X, np.ndarray):
            return X
        else:
            return X.toarray()

    #Define dense_transformer to prevent sparcity
    dense_transformer = FunctionTransformer(to_dense, accept_sparse=True)

    numeric_transformer = Pipeline([
        ('scaler', StandardScaler()),
        ('to_dense', dense_transformer)  
    ])

    categorical_transformer = Pipeline([
        ('encoder', OneHotEncoder(handle_unknown = 'ignore')),
        ('to_dense', dense_transformer)  
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    #Define final model pipe
    pipeline_final = Pipeline([
        ('preprocessor', preprocessor),
        ('to_dense', dense_transformer), 
        ('clusterer', mixture.GaussianMixture(n_components = knee_bic.elbow, random_state = 13))
    ])

    pipeline_final.fit(df_selected)

    #Saving model and results
    labels['Mixture'] = pipeline_final.predict(df_selected)
    models['Mixture'] = pipeline_final

    #%DBSCAN

    def find_dbscan_clusters(df_scaled, max_clusters, start_eps=0.1, end_eps=2.0, step_eps=0.1, min_samples=5):
        for eps in np.arange(start_eps, end_eps, step_eps):
            db = DBSCAN(eps=eps, min_samples=min_samples).fit(df_scaled)
            labels = db.labels_
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)  # Exclude noise label (-1)
            print(f"eps: {eps:.2f}, clusters: {n_clusters}")
            if n_clusters <= max_clusters:
                return labels, eps
        return None, None

    db = DBSCAN(eps=0.5, min_samples=5).fit(data_transformed)
    labels['DBSCAN'] = db.labels_
    models['DBSCAN'] = db

    #%Spectral Clustering

    # # Elbow Analysis
    # # Define a range for the number of clusters
    # range_n_clusters = list(range(2, 10))

    # # List to store silhouette scores
    # silhouette_avg_scores = []

    # for n_clusters in range_n_clusters:
    #     # Apply Spectral Clustering
    #     spectral_clustering = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', random_state=42)
    #     cluster_labels = spectral_clustering.fit_predict(data_transformed)
        
    #     # Compute the silhouette score
    #     silhouette_avg = silhouette_score(data_transformed, cluster_labels)
    #     silhouette_avg_scores.append(silhouette_avg)
        

    # # Find knee using maximum curvature method (KneeLocator)
    # sc = KneeLocator(range_n_clusters, silhouette_avg_scores, curve='convex', direction='decreasing')
    # n_clusters_optimo = sc.knee if sc.knee else 3  # Utiliza 3 como valor predeterminado si el método no encuentra un codo

    # labels['Spectral'] = sc.labels_
    # models['Spectral'] = sc

    #%Select the best clustering fit

    davies_bouldin_dict = dict()
    for model, labels_result in labels.items():
        davies_bouldin_dict[model] = davies_bouldin_score(data_transformed.toarray(), labels_result)

    best_model = min(davies_bouldin_dict, key=davies_bouldin_dict.get)
    best_model = 'Mixture'

    #%Add label to original df and saving
    df['CLUSTER'] = labels[best_model]

    df.to_parquet(f'data\Clustered_Data\Clustered_df_{today}_{run_uuid}.parquet')

    #saving the best model
    # model_filename = f'models\\final_model_{best_model}_{today}_{run_uuid}.pkl'
    # with open(model_filename, 'wb') as file:
    #     pickle.dump(models[best_model], file)
   
@asset(deps=[clustering], description='Add OcupationPct to the data and loads it to the Azure DB.')
def saving():

    # Load the run information from the JSON file
    with open(r'utils\run_parameters.json', 'r') as f:
        run_data = json.load(f)

    today = run_data["timestamp"]
    run_uuid = run_data["uuid"]


    #%Loading Clustering results

    df = pd.read_parquet(f'data\Clustered_Data\Clustered_df_{today}_{run_uuid}.parquet')

    #%Creating Ocupation Dataframe
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

    #%Combining it with reservations df

    # Function to calculate the sum of occupation between arrival and departure
    def calculate_total_occupation(arrival, departure, occupation_df):
        mask = (occupation_df['date'] >= arrival) & (occupation_df['date'] <= departure)
        return occupation_df.loc[mask, 'rooms_occupied'].sum() / (occupation_df.loc[mask, 'rooms_occupied'].count() * 735)

    # Apply the function to each row in reservations_df
    df['OCUPACION'] = df.apply(lambda row: calculate_total_occupation(row['FechaLlegada'],
                                                                            row['FechaSalida'],
                                                                            occupied_rooms_df), axis=1)
    #%Saving backup table and writing to Database

    df.to_parquet(f'data\Historic_Data\Historic_df_{today}_{run_uuid}.parquet')

    conn_str = (
        f"DRIVER=ODBC Driver 17 for SQL Server;"
        f"SERVER={SECRETS.SERVER};"
        f"DATABASE={SECRETS.DATABASE};"
        f"UID={SECRETS.UID};"
        f"PWD={SECRETS.PWD}"
    )

    load_df_to_sql_server(df.astype(str), 'iar_ClusterResults', conn_str, overwrite=True)

    conn_str = None 

defs = Definitions(assets=[run_params, etl, clustering, saving])