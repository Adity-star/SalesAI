"""
Task wrappers for training/registration/reporting/cleanup.

Place at: dags/tasks/training_tasks.py
This module is intentionally a thin wrapper around your existing 'include' modules.
Functions are written to be callable by Airflow (via a @task wrapper) or locally
for testing:

    from tasks.training_tasks import prepare_and_train
    res = prepare_and_train(extract_result=my_extract, validation_summary=my_val)
"""

import os
import json
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def prepare_and_train(extract_result: Dict[str, Any], validation_summary: Dict[str, Any]) -> Dict[str, Any]:
    """
    Load generated files, prepare training table, and train models.
    Delegates to include.models.train_models.ModelTrainer (your existing trainer).
    Returns a serializable dict with training results and MLflow run id.
    """
    try:
        from src.models.train_models import ModelTrainer  # uses your current code location
    except Exception as e:
        logger.exception("Could not import ModelTrainer from include.models.train_models: %s", e)
        raise

    file_paths = extract_result.get("file_paths", {})
    logger.info("Loading sales data from %d sales files", len(file_paths.get("sales", [])))
    import pandas as pd

    sales_dfs = []
    max_files = 50
    for i, sales_file in enumerate(file_paths.get("sales", [])[:max_files]):
        try:
            df = pd.read_parquet(sales_file)
            sales_dfs.append(df)
        except Exception as e:
            logger.warning("Failed to read %s : %s", sales_file, e)
        if (i + 1) % 10 == 0:
            logger.info("Loaded %d files...", i + 1)

    if not sales_dfs:
        raise ValueError("No sales files loaded for training.")

    sales_df = pd.concat(sales_dfs, ignore_index=True)
    logger.info("Combined sales data shape: %s", sales_df.shape)

    # aggregate to daily-level as in previous DAG
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

    # optional promotions
    if file_paths.get("promotions"):
        try:
            promo_df = pd.read_parquet(file_paths["promotions"][0])
            promo_summary = (
                promo_df.groupby(["date", "product_id"])["discount_percent"].max().reset_index()
            )
            promo_summary["has_promotion"] = 1
            daily_sales = daily_sales.merge(
                promo_summary[["date", "product_id", "has_promotion"]],
                on=["date", "product_id"],
                how="left",
            )
            daily_sales["has_promotion"] = daily_sales["has_promotion"].fillna(0)
        except Exception as e:
            logger.warning("Failed to load promotions: %s", e)

    # optional traffic
    if file_paths.get("customer_traffic"):
        try:
            traffic_dfs = []
            for traffic_file in file_paths["customer_traffic"][:10]:
                traffic_dfs.append(pd.read_parquet(traffic_file))
            traffic_df = pd.concat(traffic_dfs, ignore_index=True)
            traffic_summary = (
                traffic_df.groupby(["date", "store_id"])
                .agg({"customer_traffic": "sum", "is_holiday": "max"})
                .reset_index()
            )
            daily_sales = daily_sales.merge(traffic_summary, on=["date", "store_id"], how="left")
        except Exception as e:
            logger.warning("Failed to load customer traffic: %s", e)

    logger.info("Final training table shape: %s", daily_sales.shape)
    trainer = ModelTrainer()

    # aggregate at store-level for training as previous DAG
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

    # prepare and train
    train_df, val_df, test_df = trainer.prepare_data(
        store_daily_sales,
        target_col="sales",
        date_col="date",
        group_cols=["store_id"],
        categorical_cols=["store_id"],
    )
    logger.info("Train shape: %s, Val shape: %s, Test shape: %s", train_df.shape, val_df.shape, test_df.shape)

    results = trainer.train_all_models(train_df, val_df, test_df, target_col="sales", use_optuna=True)

    # Make results serializable: keep only metrics summary
    serializable_results = {}
    for model_name, model_results in results.items():
        serializable_results[model_name] = {"metrics": model_results.get("metrics", {})}

    # capture mlflow run id if present
    try:
        import mlflow

        current_run_id = mlflow.active_run().info.run_id if mlflow.active_run() else None
    except Exception:
        current_run_id = None

    logger.info("Training done. MLflow run id: %s", current_run_id)
    return {"models": serializable_results, "mlflow_run_id": current_run_id}


def evaluate_models(training_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Pick best model by RMSE (or other metric). Returns selected model metadata.
    """
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
        raise ValueError("No valid model found during evaluation.")

    logger.info("Selected best model: %s (RMSE: %s)", best_model, best_rmse)
    return {"best_model": best_model, "best_rmse": best_rmse, "mlflow_run_id": training_result.get("mlflow_run_id")}

