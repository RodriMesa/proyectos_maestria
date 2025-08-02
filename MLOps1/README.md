# Proyecto MLOps: Pipeline de Entrenamiento y Registro de MNIST con Airflow y MLflow

Este proyecto implementa un pipeline completo de MLOps para el entrenamiento, evaluación y registro de modelos usando el dataset MNIST. Utiliza Docker, Airflow, MLflow, MinIO (S3 compatible) y PostgreSQL para orquestar y almacenar los experimentos y artefactos.

## Descripción

- Airflow: Orquesta el pipeline de entrenamiento, evaluación y registro del modelo RandomForest sobre MNIST.
- MLflow: Almacena los experimentos, métricas y modelos entrenados.
- MinIO: Almacena los artefactos (datasets, modelos serializados, datos preprocesados) en formato S3.
- PostgreSQL: Backend para MLflow y Airflow.
- Docker Compose: Levanta todos los servicios necesarios de forma sencilla.
- bootstrap.sh: Descarga el dataset MNIST desde Kaggle y lo sube a MinIO automáticamente.

## Estructura

- emnist_dag.py: DAG principal que descarga datos, entrena el modelo, evalúa y registra en MLflow.
- Dockerfiles: Dockerfiles personalizados para MLflow y Airflow.
- bootstrap.sh: Script para inicializar el bucket y cargar los datos en MinIO.
- init.sql: Script para crear las bases de datos necesarias en PostgreSQL.
- .env: Variables de entorno para configuración de servicios.

## Cómo correr el proyecto

1. Clona el repositorio y entra al directorio del proyecto

```bash
git clone <repo-url>
cd <repo-directory>
```

2. Configura las variables de entorno

Edita el archivo .env si necesitas cambiar usuarios, contraseñas o puertos.

3. Inicializa las carpetas necesarias

```bash
./init.sh
```

Esto crea las carpetas de Airflow y levanta los servicios con Docker Compose.

4. Verifica que los servicios estén corriendo

```bash
docker-compose ps
```

Deberías ver los servicios: postgres, minio, mlflow, airflow-webserver, airflow-scheduler, etc.

5. Accede a las interfaces web (puertos en `.env`)

- Airflow: http://localhost:{AIRFLOW_WEB_PORT}
- MLflow: http://localhost:{MLFLOW_PORT}
- MinIO: http://localhost:{MINIO_UI_PORT}

6. Ejecuta el DAG en Airflow

Ingresa a la UI de Airflow, habilita y ejecuta el DAG `emnist_airflow_pipeline`. Esto descargará los datos, entrenará el modelo y registrará los resultados en MLflow.

## Requisitos

- Docker y Docker Compose
- Acceso a Kaggle (para descargar EMNIST)
- Python (solo si quieres modificar los DAGs o scripts)

## Notas

- Los datos y modelos se almacenan en MinIO bajo el bucket mlflow.
- Los experimentos y métricas se visualizan en MLflow.
- El pipeline es reproducible y modular, se puede modificar el DAG para experimentar con otros modelos o datasets.
