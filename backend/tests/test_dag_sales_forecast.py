import pytest
import os
from airflow.models import DagBag

# Path to your DAG file
DAG_PATH = os.path.join(os.path.dirname(__file__), "../dags/sales_forecast_training.py")

@pytest.fixture(scope="module")
def dag_bag():
    return DagBag(dag_folder=os.path.dirname(DAG_PATH), include_examples=False)

def test_dag_import(dag_bag):
    assert dag_bag.dags is not None, "DAGs not loaded"
    assert dag_bag.import_errors == {}, f"DAG import errors: {dag_bag.import_errors}"

def test_sales_forecast_dag_loaded(dag_bag):
    dag_id = "sales_forecast_training"
    assert dag_id in dag_bag.dags, f"{dag_id} is not in DagBag"
    dag = dag_bag.get_dag(dag_id)
    assert dag.default_args is not None
    assert len(dag.tasks) > 0, "DAG has no tasks"

def test_task_dependencies(dag_bag):
    dag = dag_bag.get_dag("sales_forecast_training")
    expected_tasks = [
        "prepare_and_train_task",
        "evaluate_model_task",
        "generate_forecast_task"
    ]
    for task_id in expected_tasks:
        assert task_id in dag.task_ids, f"Missing task: {task_id}"

    # Example: ensure one task depends on another
    assert "evaluate_model_task" in dag.downstream_task_ids("prepare_and_train_task")
