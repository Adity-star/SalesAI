from src.logger import logger
import os
from airflow.models import Variable
import pandas as pd

DATA_DIR = Variable.get("sales_data_dir", "/tmp/sales_data")
def extract_data_task(data_dir: str = DATA_DIR) -> dict:
        logger.info("Starting synthetic sales data generation")
        from src.data.data_generator import SyntheticDataGenerator

        os.makedirs(data_dir, exist_ok=True)

        generator = SyntheticDataGenerator(start_date="2023-01-01", end_date="2023-01-21")
        file_paths = generator.generate_sales_data(output_dir=data_dir)

        total_files = sum(len(paths) for paths in file_paths.values())
        logger.info(f"Generated {total_files} files under {data_dir}")

        return {"data_output_dir": data_dir, "file_paths": file_paths, "total_files": total_files}


def validate_data_task(extract_result: dict, sample_n: int = 10) -> dict:
        file_paths = extract_result["file_paths"]
        issues_found = []
        total_rows = 0

        sales_files = file_paths.get("sales", [])[:sample_n]
        if not sales_files:
            issues_found.append("No sales files found for validation")

        for i, sales_file in enumerate(sales_files):
            try:
                df = pd.read_parquet(sales_file)
                total_rows += len(df)
                if df.empty:
                    issues_found.append(f"Empty file: {sales_file}")
                    continue
                required_cols = ["date", "store_id", "product_id", "quantity_sold", "revenue"]
                missing = set(required_cols) - set(df.columns)
                if missing:
                    issues_found.append(f"Missing cols in {sales_file}: {missing}")
                if "quantity_sold" in df.columns and df["quantity_sold"].min() < 0:
                    issues_found.append(f"Negative quantity in {sales_file}")
            except Exception as e:
                issues_found.append(f"Failed to read {sales_file}: {e}")

        summary = {
            "total_files_validated": len(sales_files),
            "total_rows": int(total_rows),
            "issues_found": len(issues_found),
            "issues": issues_found[:20],
        }

        if issues_found:
            logger.warning(f"Validation found {len(issues_found)} issues. Sample: {issues_found[:5]}")
        else:
            logger.info(f"Validation passed. Total rows scanned: {total_rows}")

        return summary