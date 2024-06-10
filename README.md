# Multicripto_Omega
El final de una era, el inicio de 6 profesionales

# Estructura del Repositorio
Este repositorio es el template de trabajo para el projecto final de Ciencia de Datos del equipo Multicripto. Esta dividido en 4 carpetas las cuales sirven para lo siguiente:
  ```
    📦multicripto_omega
     ┣ 📂data
     ┃ ┣ 📂Base_Data
     ┃ ┣ 📂Clustered_Data
     ┃ ┣ 📂Historic_Data
     ┣ 📂models
     ┣ 📂pipe
     ┃ ┣ 📂test_dagster
     ┃ ┣ 📂utils
     ┃ ┣ 📜0_run_parameters.py
     ┃ ┣ 📜1_etl.py
     ┃ ┣ 📜2_clustering.py
     ┃ ┣ 📜3_saving.py
     ┣ 📂test
  ```

- [Data](data): Aqui se guardan los datos historicos y respaldos cuando se ejecuta la pipeline.

- [Models](models): Se guardan historicamente los .pkl con los mejores modelos de clustering entrenados en cada ejecucion.

- [Pipe](pipe): Esta es la carpeta importante y a revisar. Esta el código de la pipeline pero sera explicado más a detalle abajo.

- [Test](test): Es una carpeta en donde tenemos algunos intentos de pruebas varias que hicimos. No hay ningun código oficial aqui.


# Pipeline

Esta es la carpeta principal y esta dividida en 2 partes. Primero tenemos el código que se ejecutaria si quisieras correr la pipeline en una terminal (Ir script por script ejecutando manualmente) y creemos que puede ser la forma más amigable de revisar el código. Luego tenemos otra carpeta [dagster_test](pipe/dagster_test) la cual tiene lo mismo pero escrito para ser montado con dagster para orquestrar el flujo. 

## Creación de la pipeline
Tenemos 4 scripts importantes y una carpeta [utils](pipe/utils) que hacen lo siguiente:

- [Carpeta utils](pipe/utils): Guardamos codigo auxiliar a la pipeline y tiene la estructura:
    ```
   📂utils
    ┣ 📜datawrangling_utils.py
    ┣ 📜Query-Reservaciones.sql
    ┣ 📜run_parameters.json
    ```
    Aqui guardamos la query de SQL que se ejecuta para traer la información, algunas funciones auxiliares de data wrangling y aqui tambien se guarda el .json con los parametros de la ejecucion actual.

- [0_run_parameters](pipe/0_run_parameters.py): Definimos metaparamentros de la ejecucion como el tiempo de inicio y un uuid.
- [1_etl.py](pipe/1_etl.py): Nos conectamos a la base de datos, hacemos la query y limpiamos los datos. Guardamos esto por persistencia en la carpeta de `data/Base_Data`.
- [2_clustering.py](pipe/2_clustering.py): Se prueban varios algoritmos distintos de clustering y se define el mejor. Luego guardamos el modelo resultante en `models` y los resultados del clustering en `data/Clustered_Data`.
- [3_saving.py](pipe/3_saving.py): Agregamos a la información la data de ocupación. Nos conectamos de nuevo a la base de datos y actualizamos la tabla de resultados con el dataframe final. Tambien guardamos un backup en `data/Historic_Data`.
  

## Implementacion con Dagster
Aqui tenemos una carpeta llamada `pipeline_rh` la cual contiene la pipeline. Tambien aqui tenemos la informacion de un venv que para ejecutar la pipeline pero esto se omitió del repositorio. (los requirements estan en la carpeta principal). 
La estructura es la siguiente:

  ```
    📦pipe
     ┣ 📂dagster_test
     ┃ ┣ 📂pipeline_rh
     ┃ ┃ ┣ 📂data
     ┃ ┃ ┃ ┣ 📂Base_Data
     ┃ ┃ ┃ ┃ ┣ 📜.gitkeep
     ┃ ┃ ┃ ┃ ┣ 📜Transformed_df_2024-06-03_H22-M03-S52_ccd8ec6a-3cc1-48e6-b05b-2f5783fa601f.parquet
     ┃ ┃ ┃ ┃ ┣ 📜Transformed_df_2024-06-06_H12-M20-S16_e297b21a-9803-4d71-b0bb-76fe9e3d6865.parquet
     ┃ ┃ ┃ ┣ 📂Clustered_Data
     ┃ ┃ ┃ ┃ ┣ 📜.gitkeep
     ┃ ┃ ┃ ┃ ┣ 📜Clustered_df_2024-06-03_H22-M03-S52_ccd8ec6a-3cc1-48e6-b05b-2f5783fa601f.parquet
     ┃ ┃ ┃ ┃ ┣ 📜Clustered_df_2024-06-06_H12-M20-S16_e297b21a-9803-4d71-b0bb-76fe9e3d6865.parquet
     ┃ ┃ ┃ ┣ 📂Historic_Data
     ┃ ┃ ┃ ┃ ┣ 📜.gitkeep
     ┃ ┃ ┃ ┃ ┣ 📜Historic_df_2024-06-03_H22-M03-S52_ccd8ec6a-3cc1-48e6-b05b-2f5783fa601f.parquet
     ┃ ┃ ┃ ┃ ┣ 📜Historic_df_2024-06-06_H12-M20-S16_e297b21a-9803-4d71-b0bb-76fe9e3d6865.parquet
     ┃ ┃ ┣ 📂models
     ┃ ┃ ┃ ┣ 📜.gitkeep
     ┃ ┃ ┃ ┣ 📜final_model_DBSCAN_2024-06-03_H22-M03-S52_ccd8ec6a-3cc1-48e6-b05b-2f5783fa601f.pkl
     ┃ ┃ ┃ ┗ 📜final_model_Mixture_2024-06-06_H12-M20-S16_e297b21a-9803-4d71-b0bb-76fe9e3d6865.pkl
     ┃ ┃ ┣ 📂utils
     ┃ ┃ ┣ 📂pipeline_rh
     ┃ ┃ ┃ ┣ 📜assets.py
     ┃ ┃ ┃ ┗ 📜__init__.py
     ┃ ┃ ┣ 📂pipeline_rh.egg-info
     ┃ ┃ ┣ 📜pyproject.toml
     ┃ ┃ ┣ 📜README.md
     ┃ ┃ ┣ 📜setup.cfg
     ┃ ┃ ┗ 📜setup.py
  ```

Tenemos las mismas carpetas `data`, `models` y `utils` que son analogas a las otras en el repositorio. Para guardar datos y definir funciones. 

luego esta la carpetas
- [pipeline_rh](pipe/dagster_test/pipeline_rh/pipeline_rh) Tenemos 2 códigos.
    - [__init__.py](pipe/dagster_test/pipeline_rh/pipeline_rh/__init__.py): Inicializador de todos los assets y schedules que se ejecutaran en dagster.
    - [assets.py](pipe/dagster_test/pipeline_rh/pipeline_rh/assets.py): Definimos los assets que son objetos de dagster para modificar una pipeline. En este caso nuestros assets son los scripts.

El resto de carpetas y archivos son de setup y configuracion para dagster y fueron generados por la libreria. El unico cambio que se hizó fue en [setup.py](pipe/dagster_test/pipeline_rh/setup.py) en donde tenemos que agregar todas las librerias en nuestro requirements.txt. 





