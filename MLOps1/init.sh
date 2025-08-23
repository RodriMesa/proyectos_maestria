#!/bin/bash

mkdir -p ./airflow/dags ./airflow/logs ./airflow/plugins ./postgres_data
chmod -R 777 ./
# Grant edition permissions to any user/docker system
chmod -R 775 ./airflow
chmod -R 777 ./postgres_data
chmod -R 777 ./airflow/logs
chown -R 50000:50000 ./airflow

sudo docker compose up -d --build
