#!/bin/bash

set -o errexit
set -o nounset
set -o pipefail

# uncomment below when you see any issue with db...
# https://github.com/ymym3412/mlflow-docker-compose/issues/4
# mlflow db upgrade $MLFLOW_DB_URI

mlflow server \
    --host $MLFLOW_VIRTUAL_HOST \
    --port $MLFLOW_VIRTUAL_PORT \
    --backend-store-uri $MLFLOW_DB_URI \
    --default-artifact-root $MLFLOW_ARTIFACT_PATH
