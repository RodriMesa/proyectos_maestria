#!/bin/bash
set -e

# Instala mc, curl, unzip y wget
apt-get update && apt-get install -y curl unzip wget && \
  curl -sLO https://dl.min.io/client/mc/release/linux-amd64/mc && \
  chmod +x mc && mv mc /usr/local/bin/

# Configura alias
mc alias set local ${MINIO_ENDPOINT} ${MINIO_ACCESS_KEY} ${MINIO_SECRET_KEY}

# Verifica si los archivos ya están presentes en el bucket
if mc stat local/mlflow/emnist/emnist-balanced-train.csv >/dev/null 2>&1 && \
   mc stat local/mlflow/emnist/emnist-balanced-test.csv >/dev/null 2>&1; then
  echo "Los archivos ya existen en el bucket. No se realiza ninguna acción."
  exit 0
fi

# Crea el bucket si no existe
mc mb local/mlflow || echo "El bucket ya existe"

# Crear carpeta temporal
mkdir -p /tmp/emnist && cd /tmp/emnist

# Descarga desde Kaggle
curl --progress-bar -L -o emnist.zip https://www.kaggle.com/api/v1/datasets/download/crawford/emnist

# Descomprime
unzip emnist.zip || echo "Error descomprimiendo archivo"

# Sube al bucket
mc cp emnist-balanced-train.csv local/mlflow/emnist/
mc cp emnist-balanced-test.csv local/mlflow/emnist/

# Limpieza
rm -rf /tmp/emnist
