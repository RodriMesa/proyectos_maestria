#!/bin/bash

mkdir -p ./airflow/dags ./airflow/logs ./airflow/plugins

# Asegurar que cualquier usuario pueda usarlo (ideal en entornos colaborativos)
chmod -R 775 ./airflow
chown -R 50000:50000 ./airflow