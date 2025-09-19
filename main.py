# from include.utils.data_generator import SyntheticDataGenerator
# import sys
# from include.logger import logger
# import pandas as pd

# import os
# print(os.__file__)
# print(hasattr(os, "register_at_fork"))



# def extract_data_task():
#         logger.info("Starting synthetic sales data generation")

#         from include.utils.data_generator import SyntheticDataGenerator

#         data_output_dir = "tmp/sales_data"
#         generator = SyntheticDataGenerator(
#             start_date="2023-01-01", end_date="2023-12-31"
#         )
#         file_paths = generator.generate_sales_data(output_dir=data_output_dir)
#         total_files = sum(len(paths) for paths in file_paths.values())
        
#         logger.info(f"Generated {total_files} files:")
#         for data_type, paths in file_paths.items():
#             logger.info(f"  - {data_type}: {len(paths)} files")
        
#         return {
#             "data_output_dir": data_output_dir,
#             "file_paths": file_paths,
#             "total_files": total_files,
#         }

# def validate_data_task(extract_result):
#     import glob

#     file_paths = extract_result["file_paths"]
#     total_rows = 0
#     issues_found = []
#     print(f"Validating {len(file_paths['sales'])} sales files...")
#     for i, sales_file in enumerate(file_paths["sales"][:10]):
#         df = pd.read_parquet(sales_file)
#         if i == 0:
#             print(f"Sales data columns: {df.columns.tolist()}")
#         if df.empty:
#             issues_found.append(f"Empty file: {sales_file}")
#             continue
#         required_cols = [
#             "date",
#             "store_id",
#             "product_id",
#             "quantity_sold",
#             "revenue",
#         ]
#         missing_cols = set(required_cols) - set(df.columns)
#         if missing_cols:
#             issues_found.append(f"Missing columns in {sales_file}: {missing_cols}")
#         total_rows += len(df)
#         if df["quantity_sold"].min() < 0:
#             issues_found.append(f"Negative quantities in {sales_file}")
#         if df["revenue"].min() < 0:
#             issues_found.append(f"Negative revenue in {sales_file}")
#     for data_type in ["promotions", "store_events", "customer_traffic"]:
#         if data_type in file_paths and file_paths[data_type]:
#             sample_file = file_paths[data_type][0]
#             df = pd.read_parquet(sample_file)
#             print(f"{data_type} data shape: {df.shape}")
#             print(f"{data_type} columns: {df.columns.tolist()}")
#     validation_summary = {
#         "total_files_validated": len(file_paths["sales"][:10]),
#         "total_rows": total_rows,
#         "issues_found": len(issues_found),
#         "issues": issues_found[:5],
#     }
#     if issues_found:
#         print(f"Validation completed with {len(issues_found)} issues:")
#         for issue in issues_found[:5]:
#             print(f"  - {issue}")
#     else:
#         print(f"Validation passed! Total rows: {total_rows}")
#     return validation_summary


# #extract_data_task()
# extract_result = extract_data_task()
# validate_data_task(extract_result)


# test_mlflow_run.py

import os
from include.utils.mlflow_utils import MLflowManager
import mlflow

mlflow.set_tracking_uri("http://localhost:5001")
mlflow.set_experiment("test-exp")

with mlflow.start_run(run_name="test-run"):
    mlflow.log_param("param1", 123)
    mlflow.log_metric("metric1", 0.95)


# if __name__ == "__main__":
#     main()
