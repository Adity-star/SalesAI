import os
import sys
import logging
from datetime import datetime
import pandas as pd
from include.logger import logger

# Setup logger (similar to your Airflow logger)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add include path if needed (adjust if different locally)
sys.path.append(os.path.abspath("include"))

DATA_DIR = os.getenv("SALES_DATA_DIR", "/tmp/sales_data")
ARTIFACT_DIR = os.getenv("ARTIFACT_DIR", "/tmp/artifacts")


def extract_data_task(data_dir=DATA_DIR):
    logger.info("Starting synthetic sales data generation")
    from include.utils.data_generator import SyntheticDataGenerator

    os.makedirs(data_dir, exist_ok=True)
    generator = SyntheticDataGenerator(start_date="2023-01-01", end_date="2023-01-30")
    file_paths = generator.generate_sales_data(output_dir=data_dir)
    total_files = sum(len(paths) for paths in file_paths.values())
    logger.info(f"Generated {total_files} files under {data_dir}")

    return {"data_output_dir": data_dir, "file_paths": file_paths, "total_files": total_files}


def validate_data_task(extract_result, sample_n=10):
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

    if issues_found:
        logger.warning(f"Validation found {len(issues_found)} issues. Sample: {issues_found[:5]}")
    else:
        logger.info(f"Validation passed. Total rows scanned: {total_rows}")

    return {
        "total_files_validated": len(sales_files),
        "total_rows": int(total_rows),
        "issues_found": len(issues_found),
        "issues": issues_found[:20],
    }


def prepare_and_train_task(extract_result, validation_summary):
    from include.models.train_models import ModelTrainer

    file_paths = extract_result["file_paths"]
    logger.info(f"Loading sales data from multiple files...")
    sales_dfs = []
    max_files = 50
    for i, sales_file in enumerate(file_paths["sales"][:max_files]):
        df = pd.read_parquet(sales_file)
        sales_dfs.append(df)
        if (i + 1) % 10 == 0:
            logger.info(f"  Loaded {i + 1} files...")
    sales_df = pd.concat(sales_dfs, ignore_index=True)
    logger.info(f"Combined sales data shape: {sales_df.shape}")

    daily_sales = (
        sales_df.groupby(["date", "store_id", "product_id", "category"])
        .agg(
            {
                "quantity_sold": "sum",
                "revenue": "sum",
                "cost": "sum",
                "profit": "sum",
                "discount_percent": "mean",
                "unit_price": "mean",
            }
        )
        .reset_index()
    )
    daily_sales = daily_sales.rename(columns={"revenue": "sales"})

    if file_paths.get("promotions"):
        promo_df = pd.read_parquet(file_paths["promotions"][0])
        promo_summary = (
            promo_df.groupby(["date", "product_id"])["discount_percent"]
            .max()
            .reset_index()
        )
        promo_summary["has_promotion"] = 1
        daily_sales = daily_sales.merge(
            promo_summary[["date", "product_id", "has_promotion"]],
            on=["date", "product_id"],
            how="left",
        )
        daily_sales["has_promotion"] = daily_sales["has_promotion"].fillna(0)

    if file_paths.get("customer_traffic"):
        traffic_dfs = []
        for traffic_file in file_paths["customer_traffic"][:10]:
            traffic_dfs.append(pd.read_parquet(traffic_file))
        traffic_df = pd.concat(traffic_dfs, ignore_index=True)
        traffic_summary = (
            traffic_df.groupby(["date", "store_id"])
            .agg({"customer_traffic": "sum", "is_holiday": "max"})
            .reset_index()
        )
        daily_sales = daily_sales.merge(
            traffic_summary, on=["date", "store_id"], how="left"
        )

    logger.info(f"Final training data shape: {daily_sales.shape}")
    logger.info(f"Columns: {daily_sales.columns.tolist()}")

    trainer = ModelTrainer()
    store_daily_sales = (
        daily_sales.groupby(["date", "store_id"])
        .agg(
            {
                "sales": "sum",
                "quantity_sold": "sum",
                "profit": "sum",
                "has_promotion": "mean",
                "customer_traffic": "first",
                "is_holiday": "first",
            }
        )
        .reset_index()
    )
    store_daily_sales["date"] = pd.to_datetime(store_daily_sales["date"])

    train_df, val_df, test_df = trainer.prepare_data(
        store_daily_sales,
        target_col="sales",
        date_col="date",
        group_cols=["store_id"],
        categorical_cols=["store_id"],
    )
    logger.info(
        f"Train shape: {train_df.shape}, Val shape: {val_df.shape}, Test shape: {test_df.shape}"
    )

    results = trainer.train_all_models(
        train_df, val_df, test_df, target_col="sales", use_optuna=True
    )

    for model_name, model_results in results.items():
        if "metrics" in model_results:
            logger.info(f"\n{model_name} metrics:")
            for metric, value in model_results["metrics"].items():
                logger.info(f"  {metric}: {value:.4f}")

    return {"models": results}


def evaluate_models_task(training_result):
    models = training_result.get("models", {})
    best_model = None
    best_rmse = float("inf")

    for mname, mres in models.items():
        metrics = mres.get("metrics", {})
        rmse = metrics.get("rmse")
        if rmse is not None and rmse < best_rmse:
            best_rmse = rmse
            best_model = mname

    if best_model is None:
        logger.error("No valid model found during evaluation.")
        raise RuntimeError("No valid model found during evaluation.")

    logger.info(f"Selected best model: {best_model} (RMSE: {best_rmse})")
    return {"best_model": best_model, "best_rmse": best_rmse}


def generate_performance_report_task(training_result, validation_summary):
    import json

    report = {
        "timestamp": datetime.now().isoformat(),
        "data_summary": validation_summary or {},
        "model_performance": training_result.get("models", {}),
    }
    report_path = os.path.join(ARTIFACT_DIR, "performance_report.json")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Saved performance report to {report_path}")
    return report


def cleanup_task(data_dir=DATA_DIR, artifact_dir=ARTIFACT_DIR):
    import shutil

    try:
        if os.path.exists(data_dir):
            shutil.rmtree(data_dir)
            logger.info(f"Removed data dir: {data_dir}")
    except Exception as e:
        logger.warning(f"Cleanup data dir failed: {e}")

    try:
        temp_art = os.path.join(artifact_dir, "tmp")
        if os.path.exists(temp_art):
            shutil.rmtree(temp_art)
            logger.info(f"Removed temp artifact dir: {temp_art}")
    except Exception as e:
        logger.warning(f"Cleanup artifacts failed: {e}")

    return "cleanup_done"


def main():
    extract_result = extract_data_task()
    validation_summary = validate_data_task(extract_result)
    training_result = prepare_and_train_task(extract_result, validation_summary)
    evaluation_result = evaluate_models_task(training_result)
    generate_performance_report_task(training_result, validation_summary)
    cleanup_task()


if __name__ == "__main__":
    main()
