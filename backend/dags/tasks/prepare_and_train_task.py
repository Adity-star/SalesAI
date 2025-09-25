
from src.logger import logger
import logging
import pandas as pd
from airflow.exceptions import AirflowSkipException



def prepare_and_train_task(extract_result, validation_summary):
        from src.models.train_models import ModelTrainer

        file_paths = extract_result["file_paths"]
        logger.info(f"Loading sales data from multiple files...")
        sales_dfs = []
        max_files = 50
        for i, sales_file in enumerate(file_paths["sales"][:max_files]):
            df = pd.read_parquet(sales_file)
            sales_dfs.append(df)
            if (i + 1) % 10 == 0:
                print(f"  Loaded {i + 1} files...")
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
                print(f"\n{model_name} metrics:")
                for metric, value in model_results["metrics"].items():
                    print(f"  {metric}: {value:.4f}")
        print("\nVisualization charts have been generated and saved to MLflow/MinIO")
        print("Charts include:")
        print("  - Model metrics comparison")
        print("  - Predictions vs actual values")
        print("  - Residuals analysis")
        print("  - Error distribution")
        print("  - Feature importance comparison")
        serializable_results = {}
        for model_name, model_results in results.items():
            serializable_results[model_name] = {
                "metrics": model_results.get("metrics", {})
            }
        import mlflow

        current_run_id = (
            mlflow.active_run().info.run_id if mlflow.active_run() else None
        )
        return {
            "training_results": serializable_results,
            "mlflow_run_id": current_run_id,
        }

def evaluate_models_task(training_result: dict) -> dict:
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
            logger.error("No valid model found during evaluation. Skipping downstream tasks.")
            # You can skip downstream tasks or raise an error as fits your workflow:
            raise AirflowSkipException("No valid model found")

        logger.info(f"Selected best model: {best_model} (RMSE: {best_rmse})")
        return {
            "best_model": best_model,
            "best_rmse": best_rmse,
            "mlflow_run_id": training_result.get("mlflow_run_id")
        }