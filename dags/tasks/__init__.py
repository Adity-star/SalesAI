# Task wrappers for Airflow DAGs
"""
Task package for Airflow DAGs.

Each task here should be a thin wrapper that calls into core logic from `src/`.
For now, tasks may import directly from `include/`, but the plan is to migrate
everything into a properly modular `src/` package.

Example:
    from dags.tasks.data import generate_data_task
"""
