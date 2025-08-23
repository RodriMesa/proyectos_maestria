# Proyecto MLOps: Pipeline de Entrenamiento y Registro de EMNIST con Airflow y MLflow

Este proyecto implementa un pipeline completo de MLOps para el entrenamiento, evaluación y registro de modelos usando el dataset EMNIST. Utiliza Docker, Airflow, MLflow, MinIO (S3 compatible) y PostgreSQL para orquestar y almacenar los experimentos y artefactos.

## ¿Qué hace el proyecto?

- **Descarga automática del dataset EMNIST** desde Kaggle y lo almacena en MinIO (S3).
- **Orquesta el pipeline de ML** con Airflow: descarga datos, entrena un modelo RandomForest, evalúa y registra el modelo en MLflow.
- **Almacena experimentos y artefactos** en MLflow y MinIO.
- **Expone interfaces web** para Airflow, MLflow y MinIO.
- **Incluye una app de inferencia** (`draw_app`) para servir el modelo registrado.

## Descripción de los servicios

- **Airflow:** Orquestación del pipeline de ML.
- **MLflow:** Seguimiento de experimentos y registro de modelos.
- **MinIO:** Almacenamiento S3 para datasets y modelos.
- **PostgreSQL:** Backend para MLflow y Airflow.
- **Data Loader:** Descarga y carga automática de datos a MinIO.
- **MLflow Serve Model:** Sirve el modelo registrado para inferencia.
- **Draw App:** Interfaz web para probar el modelo.
- **Docker Compose:** Levanta todos los servicios necesarios de forma sencilla.
- **bootstrap.sh:** Descarga el dataset EMNIST desde Kaggle y lo sube a MinIO automáticamente. Este sistema se corre desde una imagen Debian Bullseye.

## Estructura del pipeline

- `./airflow/dags/emnist_dag.py`: DAG principal que descarga datos, entrena el modelo, evalúa y registra en MLflow.
- `./Dockerfiles`: Dockerfiles personalizados para los servicios de MLflow y Airflow.
- `./scripts/bootstrap.sh`: Script para inicializar el bucket y cargar los datos en MinIO.
- `./init/init.sql`: Script para crear las bases de datos necesarias en PostgreSQL.
- `.env`: Variables de entorno para configuración de servicios. Si bien por seguridad no deberían encontrarse en el repositorio, se adjuntan para asegurar una ejecución fluida ya que este es un caso de prueba.

## Cómo correr el proyecto

1. Clonar el repositorio y entrar al directorio del proyecto

```bash
git clone <repo-url>
cd <repo-directory>
```

2. Configurar las variables de entorno

Editar el archivo `.env` si se necesita cambiar usuarios, contraseñas o puertos.

3. Inicializar las carpetas necesarias

```bash
./init.sh
```

Esto crea las carpetas de Airflow y levanta los servicios con Docker Compose.

4. Verificar que los servicios estén corriendo

```bash
docker-compose ps
```

o

```bash
docker compose ps
```

dependiendo de la version de docker-compose que tenga instalada.

Se deberían ver los siguientes servicios activos: postgres, minio, mlflow, airflow-webserver, airflow-scheduler, draw_app, mlflow_serve_model.

5. Acceder a las interfaces web (puertos en `.env`)

- Airflow: [http://localhost:{AIRFLOW_WEB_PORT}](http://localhost:{AIRFLOW_WEB_PORT})
  - Default User: `admin`
  - Default Password: `admin`
- MLflow: [http://localhost:{MLFLOW_PORT}](http://localhost:{MLFLOW_PORT})
- MinIO: [http://localhost:{MINIO_UI_PORT}](http://localhost:{MINIO_UI_PORT})
  - Credenciales en `.env`
- Draw App: [http://localhost:{DRAW_APP_PORT}](http://localhost:{DRAW_APP_PORT})

6. Ejecutar el DAG en Airflow

Ingresar a la UI de Airflow, habilitar y ejecutar el DAG `emnist_airflow_pipeline`. Esto descargará los datos, entrenará el modelo y registrará los resultados en MLflow.

## Requisitos

- Docker y Docker Compose
- Acceso a Kaggle (para descargar EMNIST)
- Python (solo si quieres modificar los DAGs o scripts)

## Notas

- Los datos y modelos se almacenan en MinIO bajo el bucket mlflow.
- Los experimentos y métricas se visualizan en MLflow.
- El pipeline es reproducible y modular, se puede modificar el DAG para experimentar con otros modelos o datasets.
- La aplicación al completo ha sido probada en sistemas Linux, Mac y Windows y se brinda soporte para los tres.
