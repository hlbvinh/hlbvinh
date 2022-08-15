import os
import socket
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash_operator import BashOperator

dag_name = "climate_models"
cmd_name = dag_name
airflow_dir = "/opt/ai/airflow"
dag_dir = os.path.join(airflow_dir, "dags")
log_dir = os.path.join(airflow_dir, "log")
release_dir = os.path.join(dag_dir, dag_name, "current")
config_path = os.path.join(release_dir, "config.yml")
pex_path = os.path.join(release_dir, "skynet.pex")
email = ["martinb@ambiclimate.com", "pallav@ambiclimate.com"]
if "tryambi" in socket.gethostname():
    interval = timedelta(hours=24)
    email += ["j8w6d8x5k4i0f2t8@ambilabs.slack.com"]
else:
    interval = timedelta(hours=20)
    email += ["d8t6u1e5d3m4p4v7@ambilabs.slack.com"]

default_args = {
    "owner": "ambi",
    "depends_on_past": False,
    "start_date": datetime(2017, 3, 23),
    "execution_timeout": timedelta(hours=15),
    "email": email,
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

dag = DAG(
    "climate_models",
    catchup=False,
    default_args=default_args,
    schedule_interval=interval,
)

t1 = BashOperator(
    task_id="sample_generator",
    bash_command=(
        f"{pex_path} sample_generator --config {config_path} "
        f"--log_directory {log_dir}"
    ),
    dag=dag,
)

t2 = BashOperator(
    task_id="mode_model_training",
    bash_command=(
        f"{pex_path} mode_model --config {config_path} " f"--log_directory {log_dir}"
    ),
    dag=dag,
)

t3 = BashOperator(
    task_id="climate_model_training",
    bash_command=(
        f"{pex_path} micro_models --config {config_path} " f"--log_directory {log_dir}"
    ),
    dag=dag,
)

dag >> t1 >> t2
t1 >> t3
