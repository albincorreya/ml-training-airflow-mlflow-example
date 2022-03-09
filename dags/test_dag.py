
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.dummy_operator import DummyOperator

default_args = {
    'owner': 'airflow',
    'description': 'A dummy workflow',
    'depend_on_past': False,
    'start_date': datetime(2021, 5, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 0,
    'retry_delay': timedelta(minutes=5)
}

with DAG(dag_id="test_dummy_dag", default_args=default_args, catchup=False) as dag:
    start_op = DummyOperator(task_id="dummy_task_run_1")

    mid_op = DummyOperator(task_id="dummy_task_run_2")

    start_op >> mid_op
