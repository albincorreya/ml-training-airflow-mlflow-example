# Demo MLOps framework with custom Audio Classifier python package

![example workflow](https://github.com/albincorreya/mlops-training-pipeline-demo/actions/workflows/push.yml/badge.svg)

This is an example of setting up MLOPs training pipeline infrastructure with production ready ML research code on a local server.


Why both Airflow and MLflow?

- Airflow is a standard scalable worlfow manager in the industry with good community support. It has a easy-to-use web UI which can imporove the workflow productivity of data scientists without moving away from python ecosystem. It also gives flexibity to run many combinations of DAGs with different operators (eg, docker, python, bash, remote etc). 

- MLFlow is a easy-to-use open-source ML experiment and model artifact tracker. built-in autolog support for our Tensorlfow/Keras models.


Note, that this is a solution strictly considering the restrictions of local (on-premise) server setup using open-source technolgies. There are better alternative like AWS SageMaker etc if you want end-2-end on-cloud solutions.


The project is structured as below
- [audio_classifier](./audio_classifier): Custom python library for training deep learning audio classifiers. 
  Check [here](./audio_classifier/README.md) for more details.
- [dags](./dags): Custom Airflow DAGs and example json configurations to run ML pipeline
- [docker](./docker): Custom Dockerfiles for various containers used in our docker-compose stack.
- [env_files](./env_files): Files with environment variables defined for the docker-compose stack. (Note: dont use these secrets in production)
- [scripts](./scripts): A bunch of bash and python scripts along with some template configuration files.
- [data](./data): Input raw audio dataset provided from the reference repository


## docker-compose stack


Tools used:

- Airflow: Tracking, Orchestration of DAG pipelines.
- MLFlow: Experiment tracking, versioning, model artfacts etc.
- Tensorflow, TF record, Essentia: Data processing, versioning and validation.
- Docker and compose: Containerisation all code and easy local development setup.
- Poetry: For better python dependency management of audio_classifier library.


![alt text](./assets/sketch.png)


## Getting started 

- One command to spin up everything

```
docker-compose up airflow --build
```

### Access dashboard

- MLFlow UI: http://localhost:5000
  
- Airflow UI: http://localhost:8080

>> Note: the login credentials for airflow are sent by email. PLease reachout if you haven't received it.


#### Optional 

- Setup SSH port forwarding on your local system if we want to access the service runnign on a remote server.
  
  
```bash
ssh -L 5000:35.217.35.118:5000 ubuntu@35.217.35.118

ssh -L 8080:35.217.35.118:8080 ubuntu@35.217.35.118
```

> Important: The remote server provided has some memory limitations so the service is not up.



## Run DAG using Airflow

#### Using Airflow Web UI

- Go to http://localhost:8080

![alt text](./assets/dags-list.png)

You will see a list of dags we predefined defined on the [./dags]() directory.


We can also see a visualisation of ML training pipeline DAG 

![alt text](./assets/dag-example.png)


- Trigger our [ml_pipeline_dag.py](./dags/ml_pipeline_dag.py) using the 
  [ml_pipeline_config.json](./dags/ml_pipeline_config.json) example on the web UI.
  
![alt text](./assets/trigger-job-example.png)


## Experiment and model artifact tracking with MLFlow

![alt text](./assets/tracking-lists.png)

![alt text](./assets/artifacts.png)

![alt text](./assets/metrics.png)

## Local development Setup

- Launch our MLOps infrastructure with the following command 

```bash
docker-compose up mlflow --build
```

Once it is up you can access the web interfaces of both Airflow and MLFlow at the following addresses 

http://localhost:8080 and http://localhost:5000


> NOTE: For simplicity a reverse proxy wasn't added to the stack. But, a nginx server or similar could be easily 
> setup up on the current stack in future
> 
### Run your scripts from command line

Once the Airflow server is up run the following command

```bash
docker-compose run airflow bash
# once inside the shell
airflow trigger_dag 'example_dag_conf' -r 'run_id' --conf '{"message":"value"}'
```

#### Using only MLFlow Tracking

Once the MlFlow server is up run the following command

```bash
docker-compose run mlflow bash
# once inside the shell
python scripts/ml/train.py --help
```

#### Using Airflow CLI with MLFlow tracking


## Things nice-to-have in future iterations

- Full fledge CI/CD pipeline on the repository (Eg: Github actions)
- More elaborate doc strings in python code of `audio_classifier`.
- Deploying Airflow more production ready mode (Now we are using SQLite backend for simplicity).
- Adding reverse-proxy for both remote MLFlow and Airflow tracking servers so that it can be accessed from internet,


