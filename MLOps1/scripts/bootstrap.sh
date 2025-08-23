#!/bin/bash
set -e

# Install mc, curl, unzip & wget
apt-get update && apt-get install -y curl unzip wget jq ca-certificates && \
  curl -sLO https://dl.min.io/client/mc/release/linux-amd64/mc && \
  chmod +x mc && mv mc /usr/local/bin/

# Configure alias
mc alias set local ${MINIO_ENDPOINT} ${MINIO_ACCESS_KEY} ${MINIO_SECRET_KEY}

# --- DATA BLOCK (skip when no files are missing) ---
NEED_DATA=0
if ! mc stat local/mlflow/emnist/emnist-digits-train.csv >/dev/null 2>&1; then
  NEED_DATA=1
fi
if ! mc stat local/mlflow/emnist/emnist-digits-test.csv >/dev/null 2>&1; then
  NEED_DATA=1
fi

if [ "$NEED_DATA" -eq 1 ]; then
  echo "Faltan archivos. Preparando datos EMNIST..."

  # Create bucket  if not existing
  mc mb local/mlflow || echo "El bucket ya existe"

  # Temporal folder to download dataset
  mkdir -p /tmp/emnist && cd /tmp/emnist

  # Download datasets from Kaggle
  curl --progress-bar -L -o emnist.zip https://www.kaggle.com/api/v1/datasets/download/crawford/emnist

  # Unzip
  unzip -o emnist.zip || echo "Error descomprimiendo archivo"

  # Upload datasets to bucket (if available)
  [ -f emnist-digits-train.csv ] && mc cp emnist-digits-train.csv local/mlflow/emnist/
  [ -f emnist-digits-test.csv ] && mc cp emnist-digits-test.csv  local/mlflow/emnist/

  # Clean temporal folders
  rm -rf /tmp/emnist
else
  echo "Los archivos ya existen en el bucket. Saltando descarga/subida."
fi
# --- END DATA BLOCK ---

# -----------------------------
# Check for a 'Production' model in MLFlow Model Registry using the 'stage' tag
# -----------------------------
MODEL_EXISTS=$(curl -s -o /dev/null -w "%{http_code}" \
  "${MLFLOW_API}/api/2.0/mlflow/registered-models/get?name=${MODEL_NAME}")

if [[ "$MODEL_EXISTS" == "200" ]]; then
  echo "El modelo '${MODEL_NAME}' ya existe en MLflow. Saltando despliegue."
  exit 0
fi

# -----------------------------
# If no model is created or detected on 'Production', automatically run DAG
# -----------------------------
echo "Esperando que Airflow esté disponible (webserver:8080/health)..."
until curl -s -o /dev/null -w "%{http_code}" http://airflow-webserver:8080/health | grep -q 200; do
  echo "Aún no disponible, reintentando en 5s..."
  sleep 5
done

echo "Airflow está listo. Disparando DAG ${DAG_ID}..."

# CAUTION: use AIRFLOW_API (it points to :8080/api/v1)
RUN_ID=$(
  curl -s -u "${AIRFLOW_USER}:${AIRFLOW_PASS}" \
    -H "Content-Type: application/json" \
    -X POST "${AIRFLOW_API}/dags/${DAG_ID}/dagRuns" \
    -d '{"conf": {}}' | jq -r '.dag_run_id'
)

if [[ -z "${RUN_ID}" || "${RUN_ID}" == "null" ]]; then
  echo "No se obtuvo dag_run_id. Respuesta:"
  curl -s -u "${AIRFLOW_USER}:${AIRFLOW_PASS}" -H "Content-Type: application/json" \
    -X POST "${AIRFLOW_API}/dags/${DAG_ID}/dagRuns" -d '{"conf": {}}'
  exit 1
fi

echo "DAG Run ID: $RUN_ID"

STATUS="running"
while [[ "$STATUS" == "running" || "$STATUS" == "queued" ]]; do
  STATUS=$(
    curl -s -u "${AIRFLOW_USER}:${AIRFLOW_PASS}" \
      "${AIRFLOW_API}/dags/${DAG_ID}/dagRuns/${RUN_ID}" | jq -r '.state'
  )
  echo "Estado actual del DAG: $STATUS"
  sleep 10
done

if [[ "$STATUS" != "success" ]]; then
  echo "El DAG falló con estado: $STATUS"
  exit 1
fi

echo "DAG finalizado correctamente."