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

    @task()
    def validate_data_task(extract_result):
        import glob

        file_paths = extract_result["file_paths"]
        total_rows = 0
        issues_found = []
        print(f"Validating {len(file_paths['sales'])} sales files...")
        for i, sales_file in enumerate(file_paths["sales"][:10]):
            df = pd.read_parquet(sales_file)
            if i == 0:
                print(f"Sales data columns: {df.columns.tolist()}")
            if df.empty:
                issues_found.append(f"Empty file: {sales_file}")
                continue
            required_cols = [
                "date",
                "store_id",
                "product_id",
                "quantity_sold",
                "revenue",
            ]
            missing_cols = set(required_cols) - set(df.columns)
            if missing_cols:
                issues_found.append(f"Missing columns in {sales_file}: {missing_cols}")
            total_rows += len(df)
            if df["quantity_sold"].min() < 0:
                issues_found.append(f"Negative quantities in {sales_file}")
            if df["revenue"].min() < 0:
                issues_found.append(f"Negative revenue in {sales_file}")
        for data_type in ["promotions", "store_events", "customer_traffic"]:
            if data_type in file_paths and file_paths[data_type]:
                sample_file = file_paths[data_type][0]
                df = pd.read_parquet(sample_file)
                print(f"{data_type} data shape: {df.shape}")
                print(f"{data_type} columns: {df.columns.tolist()}")
        validation_summary = {
            "total_files_validated": len(file_paths["sales"][:10]),
            "total_rows": total_rows,
            "issues_found": len(issues_found),
            "issues": issues_found[:5],
        }
        if issues_found:
            print(f"Validation completed with {len(issues_found)} issues:")
            for issue in issues_found[:5]:
                print(f"  - {issue}")
        else:
            print(f"Validation passed! Total rows: {total_rows}")
        return validation_summary
    
    extract_result = extract_data_task()
    validation_summary = validate_data_task(extract_result)


sales_forecast_training()
