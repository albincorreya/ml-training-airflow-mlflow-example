#!/bin/bash

airflow standalone & mlflow server \
    --host $MLFLOW_VIRTUAL_HOST \
    --port $MLFLOW_VIRTUAL_PORT \
    --backend-store-uri $MLFLOW_DB_URI \
    --default-artifact-root $MLFLOW_ARTIFACT_PATH
