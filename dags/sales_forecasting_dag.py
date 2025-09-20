"""
Sales Forecasting Pipeline DAG
Contains only operators, dependencies, retry policy and scheduling.
All business logic is delegated to tasks.* functions.
"""

import os
import sys
from datetime import datetime, timedelta
import json

import pandas as pd

from airflow import DAG
from airflow.decorators import task
from airflow.models import Variable
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import ShortCircuitOperator
from airflow.utils.trigger_rule import TriggerRule
from airflow.exceptions import AirflowSkipException


import logging
from src.logger import logger

logger = logging.getLogger(__name__)

from dags.tasks.extract_data_task import extract_data_task, validate_data_task
from dags.tasks.prepare_and_train_task import prepare_and_train_task, evaluate_models_task
from tasks.model_deployement_tasks import (
    register_best_models_task, 
    transition_to_production_task,
    generate_performance_report_task,
    cleanup_task
)


# Default arguments from configuration
default_args = {
    "owner": "PeaceAi",
    "depends_on_past": False,
    "start_date": datetime(2025, 7, 24),
    "email_on_failure": True,
    "email_on_retry": False,
    "email": ["aakuskar.980@gmail.com"],
    "retries": 1,
    "retry_delay": timedelta(minutes=3),
}


DATA_DIR = Variable.get("sales_data_dir", "/tmp/sales_data")
ARTIFACT_DIR = Variable.get("artifacts_dir", "/tmp/artifacts")
APPROVAL_VAR = Variable.get("approve_production_var", "approve_production")


with DAG(
    dag_id="sales_forecasting",
    description="Thin orchestration DAG for sales forecasting (refactored)",
    schedule="@weekly",
    default_args=default_args,
    catchup=False,
    max_active_runs=1,
    tags=["ml", "training", "sales"],
) as dag:

    start = EmptyOperator(task_id="start")
    # -------------------------
    # Extract & Generate Data
    # -------------------------
     
    @task(task_id="extract_data")
    def extract_data_wrapper():
        return extract_data_task(data_dir=DATA_DIR)
    
    # -------------------------
    # Data Validation
    # -------------------------
    @task(task_id="validate_data")
    def validate_data_wrapper(extract_result):
        return validate_data_task(extract_result, sample_n=10)
    # -------------------------
    # Prepare & Train (calls tasks wrapper)
    # -------------------------
    @task(task_id="prepare_and_train")
    def prepare_and_train_wrapper(extract_result, validation_summary):
        return prepare_and_train_task(extract_result, validation_summary)

    # -------------------------
    # Evaluate (calls tasks wrapper)
    # -------------------------
    @task(task_id="evaluate_models")
    def evaluate_models_wrapper(training_result):
        return evaluate_models_task(training_result)

    # -------------------------
    # Manual approval short-circuit
    # -------------------------
    def _approval_check(**kwargs):
        """
        Proceed only when:
         - AUTO_PROMOTE Variable is true OR
         - APPROVAL_VAR Airflow Variable is set to "true" (manual confirmation)
        """
        APPROVAL_VAR = Variable.get("approve_production_var", "approve_production")

        auto_promote = Variable.get("auto_promote", "false").lower() == "true"
        if auto_promote:
            logger.info("auto_promote is true: skipping manual approval.")
            return True
        approval = Variable.get(APPROVAL_VAR, "false").lower() == "true"
        if approval:
            logger.info(f"Approval variable {APPROVAL_VAR} is true -> proceed with promotion.")
            return True
        logger.info(f"Approval missing. Set Airflow Variable '{APPROVAL_VAR}' to 'true' to promote models.")
        return False
    approval_check = ShortCircuitOperator(
        task_id="manual_approval_check",
        python_callable=_approval_check,
    )
    # -------------------------
    # Register & Transition (calls tasks wrapper)
    # -------------------------
    @task(task_id="register_best_models")
    def register_best_models_wrapper(evaluation_result: dict) -> dict:
        from tasks.training_tasks import register_best_models
        return register_best_models(evaluation_result=evaluation_result)

    @task(task_id="transition_to_production")
    def transition_to_production_wrapper(registration_result: dict) -> str:
        from tasks.training_tasks import transition_to_production
        return transition_to_production(registration_result=registration_result)

    # -------------------------
    # Generate performance report (calls tasks wrapper)
    # -------------------------
    @task(task_id="generate_performance_report")
    def generate_report_wrapper(training_result: dict, validation_summary: dict) -> dict:
        from tasks.training_tasks import generate_performance_report
        return generate_performance_report(training_result=training_result, validation_summary=validation_summary)

    # -------------------------
    # Cleanup (always run) - uses wrapper
    # -------------------------
    @task(task_id="cleanup", trigger_rule=TriggerRule.ALL_DONE)
    def cleanup_wrapper(temp_dir: str = DATA_DIR, artifact_dir: str = ARTIFACT_DIR) -> str:
        from tasks.training_tasks import cleanup
        return cleanup(temp_dir=temp_dir, artifact_dir=artifact_dir)
    

    # -------------------------
    # DAG wiring
    # -------------------------
    extract_result = extract_data_wrapper()
    validation_summary = validate_data_wrapper(extract_result)

    training_result = prepare_and_train_wrapper(extract_result, validation_summary)
    evaluation_result = evaluate_models_wrapper(training_result)

    # approval gate
    approval = approval_check

    registration_result = register_best_models_wrapper(evaluation_result)
    transition_result = transition_to_production_wrapper(registration_result)

    report = generate_report_wrapper(training_result, validation_summary)
    cleanup = cleanup_wrapper()

    # dependencies
    start >> extract_result >> validation_summary >> training_result >> evaluation_result
    evaluation_result >> approval
    approval >> registration_result >> transition_result
    evaluation_result >> report
    report >> cleanup
    registration_result >> cleanup
    transition_result >> cleanup

    end = EmptyOperator(task_id="end", trigger_rule=TriggerRule.NONE_FAILED)
    cleanup >> end
