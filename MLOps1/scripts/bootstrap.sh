#!/bin/bash
set -e

# Instala mc, curl, unzip y wget
apt-get update && apt-get install -y curl unzip wget jq ca-certificates && \
  curl -sLO https://dl.min.io/client/mc/release/linux-amd64/mc && \
  chmod +x mc && mv mc /usr/local/bin/

# Configura alias
mc alias set local ${MINIO_ENDPOINT} ${MINIO_ACCESS_KEY} ${MINIO_SECRET_KEY}

# --- BLOQUE DE DATOS (se ejecuta solo si faltan archivos) ---
NEED_DATA=0
if ! mc stat local/mlflow/emnist/emnist-digits-train.csv >/dev/null 2>&1; then
  NEED_DATA=1
fi
if ! mc stat local/mlflow/emnist/emnist-digits-test.csv >/dev/null 2>&1; then
  NEED_DATA=1
fi

if [ "$NEED_DATA" -eq 1 ]; then
  echo "Faltan archivos. Preparando datos EMNIST..."

  # Crea el bucket si no existe (no falla si ya existe)
  mc mb local/mlflow || echo "El bucket ya existe"

  # Carpeta temporal
  mkdir -p /tmp/emnist && cd /tmp/emnist

  # Descarga desde Kaggle
  curl --progress-bar -L -o emnist.zip https://www.kaggle.com/api/v1/datasets/download/crawford/emnist

  # Descomprime
  unzip -o emnist.zip || echo "Error descomprimiendo archivo"

  # Sube al bucket (sólo si existen localmente)
  [ -f emnist-digits-train.csv ] && mc cp emnist-digits-train.csv local/mlflow/emnist/
  [ -f emnist-digits-test.csv ] && mc cp emnist-digits-test.csv  local/mlflow/emnist/

  # Limpieza
  rm -rf /tmp/emnist
else
  echo "Los archivos ya existen en el bucket. Saltando descarga/subida."
fi
# --- FIN BLOQUE DE DATOS ---

echo "Esperando que Airflow esté disponible (webserver:8080/health)..."
until curl -s -o /dev/null -w "%{http_code}" http://airflow-webserver:8080/health | grep -q 200; do
  echo "Aún no disponible, reintentando en 5s..."
  sleep 5
done

echo "Airflow está listo. Disparando DAG ${DAG_ID}..."

# OJO: usamos AIRFLOW_API (que ya apunta a :8080/api/v1)
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