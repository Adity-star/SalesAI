from src.logger import logger
import os
from airflow.models import Variable
import pandas as pd


def extract_data_task() -> dict:
        logger.info("🚀 Starting M5 dataset ingestion pipeline")
        from src.data_pipelines.ingester import DatasetProcessor
        from src.entity.ingest_entity import DatasetConfig

        config = DatasetConfig()

        processor = DatasetProcessor(config)

        df ,quality_metrics = processor.run_full_pipeline()


        logger.info(f"🎉 Processing completed! Dataset shape: {df.shape}, "
                f"Date range: {df['date'].min()} to {df['date'].max()}")

        del df 

        import gc; gc.collect()
        return {
                "quality_score": quality_metrics.data_completeness_score,
               "total_time_series": quality_metrics.total_time_series
    }

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