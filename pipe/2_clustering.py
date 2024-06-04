#%% Importing libraries
import pandas as pd
import numpy as np
import pickle
import json
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn import mixture 
from kneed import KneeLocator

#%%Getting run parameters

# Load the run information from the JSON file
with open(r'utils\run_parameters.json', 'r') as f:
    run_data = json.load(f)

today = run_data["timestamp"]
run_uuid = run_data["uuid"]

#%% Reading transformed Data and defining parameters
df = pd.read_parquet(f'..\data\Transformed_df_{today}_{run_uuid}.parquet')

columns_2_use = ['Tipo_Habitacion', 'Clasificacion_tipo_habitacion', 'Paquete', 'Canal', 'Estatus_res', 'Capacidad_hotel', 'Numero_personas', 'Numero_adultos', 'Numero_noches', 'IngresoMto', 'Tipo_temporada']
df_selected = df[columns_2_use]

categorical_features = ['Tipo_Habitacion', 'Clasificacion_tipo_habitacion', 'Paquete', 'Canal', 'Estatus_res', "Tipo_temporada"]
numeric_features = ['Capacidad_hotel', 'Numero_personas', 'Numero_adultos', 'Numero_noches', 'IngresoMto']

#%% Transforming Data

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

#%%Iterating over clustering algorithms

labels, models = dict(), dict()

#%%K-Means

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

#%%Gaussian Mixture

#'data_transformed' might be sparse due to one-hot encoding
if isinstance(data_transformed, np.ndarray):
    data_dense = data_transformed  
else:
    data_dense = data_transformed.toarray()  

#Knee analisis
n_components = np.arange(1, 11)  
bics = []
for n in n_components:
    gmm = mixture.GaussianMixture(n_components = n, random_state = 42)
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
    ('clusterer', mixture.GaussianMixture(n_components = knee_bic.elbow, random_state = 42))
])

pipeline_final.fit(df_selected)

#Saving model and results
labels['Mixture'] = pipeline_final.predict(df_selected)
models['Mixture'] = pipeline_final

#%%DBSCAN

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

#%%Spectral Clustering

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

#%%Select the best clustering fit

davies_bouldin_dict = dict()
for model, labels_result in labels.items():
    davies_bouldin_dict[model] = davies_bouldin_score(data_transformed.toarray(), labels_result)

best_model = min(davies_bouldin_dict, key=davies_bouldin_dict.get)

#%%Add label to original df
df['CLUSTER'] = labels[best_model]

df.to_parquet(f'..\data\Clustered_Data\Clustered_df_{today}_{run_uuid}.parquet')

#saving the best model
model_filename = f'..\\models\\final_model_{best_model}_{today}_{run_uuid}.pkl'
with open(model_filename, 'wb') as file:
    pickle.dump(models[best_model], file)
    
#%%