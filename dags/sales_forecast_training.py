import os
import sys
import pandas as pd

from datetime import datetime, timedelta
from airflow.decorators import dag, task
from include.logger import logger
from airflow.utils.log.logging_mixin import LoggingMixin


# Add include path
sys.path.append("/usr/local/airflow/include")

log = LoggingMixin().log


default_args = {
    "owner": "PeaceAi",
    "depends_on_past": False,
    "start_date": datetime(2025, 7, 24),
    "email_on_failure": True,
    "email_on_retry": False,
    "email": ["aakuskar.980@gmail.com"],
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

@dag(
    schedule='@weekly',
    start_date=datetime(2025, 7, 24),
    catchup=False,
    description="DAG for training sales forecasting model",
    tags = ['ml', 'training', 'sales'],
    )
def sales_forecast_training():

    @task()
    def extract_data_task():
        log.info("Starting synthetic sales data generation")

        from include.utils.data_generator import SyntheticDataGenerator

        data_output_dir = "tmp/sales_data"
        generator = SyntheticDataGenerator(
            start_date="2023-01-01", end_date="2023-12-31"
        )
        file_paths = generator.generate_sales_data(output_dir=data_output_dir)
        total_files = sum(len(paths) for paths in file_paths.values())
        
        log.info(f"Generated {total_files} files:")
        for data_type, paths in file_paths.items():
            log.info(f"  - {data_type}: {len(paths)} files")
        
        return {
            "data_output_dir": data_output_dir,
            "file_paths": file_paths,
            "total_files": total_files,
        }

    # You can chain or return the task(s) here if you want
    extract_result = extract_data_task()


sales_forecast_training()
