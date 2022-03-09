import uuid
from datetime import datetime
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.utils.trigger_rule import TriggerRule
from airflow.operators.python import BranchPythonOperator
from airflow.operators.dummy_operator import DummyOperator


# Correlation id for training job (this can be also found on MLFLow tracking)
correlation_id = uuid.uuid4()


def choose_to_do_preprocessing(**kwargs):
    to_preprocess = bool(kwargs['dag_run'].conf.get('enable_preprocess'))
    if to_preprocess:
        return "preprocess_raw_audio"
    else:
        return "train_job"


with DAG(
        dag_id="audiocnn_ml_pipeline_dag",
        schedule_interval=None,
        start_date=datetime(2022, 3, 3,),
        catchup=False) as dag:

    start = DummyOperator(task_id="start")

    branch = BranchPythonOperator(
        task_id="check_to_preprocess_or_not",
        python_callable=choose_to_do_preprocessing
    )

    # Task for running data preprocessing task
    preprocessing_task = BashOperator(
        task_id="preprocess_raw_audio",
        bash_command="python ${MY_LOCAL_ASSETS}/preprocess_audio.py "
                     "--dataset-version {{ dag_run.conf['preprocess']['dataset_version'] }} "
                     "--audio-dir {{ dag_run.conf['preprocess']['audio_dir'] }} "
                     "--output-dir {{ dag_run.conf['preprocess']['output_dir'] }}",
        dag=dag
    )

    script_path = "python ${MY_LOCAL_ASSETS}/train.py" + f" --correlation-id {correlation_id}"
    script_args = " --dataset-path {{ dag_run.conf['train']['dataset_path'] }} " \
                  "--n-epochs {{ dag_run.conf['train']['n_epochs'] }} " \
                  "--data-batch-size {{ dag_run.conf['train']['data_batch_size'] }} " \
                  "--model-yaml-config {{ dag_run.conf['train']['model_yaml_config'] }}"
    # Task running our ML training job
    training_task = BashOperator(
        task_id=f"train_job",
        bash_command=script_path + script_args,
        dag=dag,
        trigger_rule=TriggerRule.ONE_SUCCESS
    )

    complete = DummyOperator(task_id="complete", trigger_rule=TriggerRule.ONE_SUCCESS)

    # DAG which define steps to run data preprocessing and training job pipeline with input options
    start >> branch
    preprocessing_task.set_upstream(branch)
    training_task.set_upstream([branch, preprocessing_task])
    complete.set_upstream(training_task)
