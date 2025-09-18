import os
import sys
import json
import shutil
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from airflow import DAG
from airflow.decorators import task, task_group
from airflow.models import Variable
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import ShortCircuitOperator, PythonOperator
from airflow.utils.trigger_rule import TriggerRule
from airflow.utils.log.logging_mixin import LoggingMixin

from include.models.train_models import ModelTrainer
from include.utils.mlflow_utils import MLflowManager

import logging
from include.logger import logger

logger = logging.getLogger(__name__)

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

# DAG config from Airflow Variables (fallback defaults)
DATA_DIR = Variable.get("sales_data_dir", "/tmp/sales_data")
ARTIFACT_DIR = Variable.get("artifacts_dir", "/tmp/artifacts")
AUTO_PROMOTE = Variable.get("auto_promote", "false").lower() == "true"
APPROVAL_VAR = Variable.get("approve_production_var", "approve_production")


with DAG(
    dag_id="sales_forecast_training_v3",
    description="Generate, validate, train, evaluate, and promote sales forecasting models",
    schedule_interval="@weekly",
    default_args=default_args,
    catchup=False,
    max_active_runs=1,
    tags=["ml", "training", "sales"],
) as dag:

    start = EmptyOperator(task_id="start")
     # -------------------------
    # Extract & Generate Data
    # -------------------------
    @task(task_id="extract_data")
    def extract_data_task(data_dir: str = DATA_DIR) -> dict:
        log.info("Starting synthetic sales data generation")
        from include.utils.data_generator import SyntheticDataGenerator

        os.makedirs(data_dir, exist_ok=True)

        generator = SyntheticDataGenerator(start_date="2023-01-01", end_date="2023-02-28")
        file_paths = generator.generate_sales_data(output_dir=data_dir)

        total_files = sum(len(paths) for paths in file_paths.values())
        log.info(f"Generated {total_files} files under {data_dir}")

        return {"data_output_dir": data_dir, "file_paths": file_paths, "total_files": total_files}


    # -------------------------
    # Data Validation
    # -------------------------
    @task(task_id="validate_data")
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
            log.warning(f"Validation found {len(issues_found)} issues. Sample: {issues_found[:5]}")
        else:
            log.info(f"Validation passed. Total rows scanned: {total_rows}")

        return summary

    
    # -------------------------
    # Prepare & Train (TaskGroup)
    # -------------------------
    @task(task_id="prepare_and_train")
    def prepare_and_train_task(extract_result: dict, validation_summary: dict, artifacts_dir: str = ARTIFACT_DIR) -> dict:
        from include.models.train_models import ModelTrainer

        file_paths = extract_result["file_paths"]
        sales_dfs = []
        max_files = int(Variable.get("max_sales_files", 50))

        for i, sales_file in enumerate(file_paths.get("sales", [])[:max_files]):
            try:
                df = pd.read_parquet(sales_file)
                sales_dfs.append(df)
            except Exception as e:
                log.warning(f"Skipping file {sales_file} due to read error: {e}")

        if not sales_dfs:
            raise ValueError("No sales files loaded, aborting training")

        sales_df = pd.concat(sales_dfs, ignore_index=True)
        log.info(f"Loaded combined sales data shape: {sales_df.shape}")

        # Aggregate
        daily_sales = (
            sales_df.groupby(["date", "store_id", "product_id", "category"], dropna=False)
            .agg({
                "quantity_sold": "sum",
                "revenue": "sum",
                "cost": "sum",
                "profit": "sum",
                "discount_percent": "mean",
                "unit_price": "mean",
            })
            .reset_index()
        )
        daily_sales = daily_sales.rename(columns={"revenue": "sales"})

        # Merge promotions
        if file_paths.get("promotions"):
            try:
                promo_df = pd.read_parquet(file_paths["promotions"][0])
                promo_summary = promo_df.groupby(["date", "product_id"])["discount_percent"].max().reset_index()
                promo_summary["has_promotion"] = 1
                daily_sales = daily_sales.merge(
                    promo_summary[["date", "product_id", "has_promotion"]],
                    on=["date", "product_id"], how="left"
                )
                daily_sales["has_promotion"] = daily_sales["has_promotion"].fillna(0)
            except Exception as e:
                log.warning(f"Promotion merge failed: {e}")

        if file_paths.get("customer_traffic"):
            try:
                traffic_dfs = [pd.read_parquet(f) for f in file_paths["customer_traffic"][:10]]
                traffic_df = pd.concat(traffic_dfs, ignore_index=True)
                traffic_summary = traffic_df.groupby(["date", "store_id"]).agg({"customer_traffic": "sum", "is_holiday": "max"}).reset_index()
                daily_sales = daily_sales.merge(traffic_summary, on=["date", "store_id"], how="left")
            except Exception as e:
                log.warning(f"Traffic merge failed: {e}")

        # Training
        trainer = ModelTrainer()

        store_daily = (
            daily_sales.groupby(["date", "store_id"])
            .agg({
                "sales": "sum",
                "quantity_sold": "sum",
                "profit": "sum",
                "has_promotion": "mean",
                "customer_traffic": "first",
                "is_holiday": "first",
            })
            .reset_index()
        )

        store_daily["date"] = pd.to_datetime(store_daily["date"])

        train_df, val_df, test_df = trainer.prepare_data(
            store_daily,
            target_col=Variable.get("target_col", "sales"),
            date_col=Variable.get("date_col", "date"),
            group_cols=["store_id"],
            categorical_cols=["store_id"],
        )

        results = trainer.train_all_models(
            train_df,
            val_df,
            test_df,
            target_col=Variable.get("target_col", "sales"),
            use_optuna=Variable.get("use_optuna", "true").lower() == "true"
        )

        trainer.save_artifacts(version=datetime.now().strftime("%Y%m%d_%H%M%S"))
        import mlflow
        run_id = mlflow.active_run().info.run_id if mlflow.active_run() else None

        return {
            "models": {k: {"metrics": v.get("metrics", {})} for k, v in results.items()},
            "mlflow_run_id": run_id
        }

    
    # -------------------------
    # Evaluate & Choose Best
    # -------------------------
    @task(task_id="evaluate_models")
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
        log.info(f"Selected best model: {best_model} (RMSE: {best_rmse})")
        return {"best_model": best_model, "best_rmse": best_rmse, "mlflow_run_id": training_result.get("mlflow_run_id")}
    

    # -------------------------
    # Manual approval gate
    # -------------------------
    def _approval_check(**kwargs):
        """
        Proceed only when:
         - AUTO_PROMOTE Variable is true OR
         - APPROVAL_VAR Airflow Variable is set to "true" (manual confirmation)
        """
        auto_promote = Variable.get("auto_promote", "false").lower() == "true"
        if auto_promote:
            log.info("auto_promote is true: skipping manual approval.")
            return True
        approval = Variable.get(APPROVAL_VAR, "false").lower() == "true"
        if approval:
            log.info(f"Approval variable {APPROVAL_VAR} is true -> proceed with promotion.")
            return True
        log.info(f"Approval missing. Set Airflow Variable '{APPROVAL_VAR}' to 'true' to promote models.")
        return False
    approval_check = ShortCircuitOperator(
        task_id="manual_approval_check",
        python_callable=_approval_check,
        provide_context=True,
    )
    

    # -------------------------
    # Register & Transition Models
    # -------------------------
    @task(task_id="register_best_models")
    def register_best_models_task(evaluation_result: dict) -> dict:
        best_model = evaluation_result.get("best_model")
        run_id = evaluation_result.get("mlflow_run_id")
        mlflow_manager = MLflowManager()
        versions = {}
        # register both tree-based models if present: xgboost, lightgbm
        for model_name in ["xgboost", "lightgbm"]:
            try:
                ver = mlflow_manager.register_model(run_id, model_name, model_name)
                versions[model_name] = ver
                log.info(f"Registered {model_name} version {ver}")
            except Exception as e:
                log.warning(f"Failed to register {model_name}: {e}")
        return {"registered_versions": versions}


    @task(task_id="transition_to_production")
    def transition_to_production_task(registration_result: dict) -> str:
        mlflow_manager = MLflowManager()
        out = []
        for model_name, version in registration_result.get("registered_versions", {}).items():
            try:
                mlflow_manager.transition_model_stage(model_name, version, "Production")
                out.append(f"{model_name}:v{version}")
                log.info(f"Transitioned {model_name} v{version} to Production")
            except Exception as e:
                log.warning(f"Failed to transition {model_name} v{version}: {e}")
        return ";".join(out)

     # -------------------------
    # Generate Performance Report
    # -------------------------
    @task(task_id="generate_performance_report")
    def generate_performance_report_task(training_result: dict, validation_summary: dict) -> dict:
        report = {
            "timestamp": datetime.now().isoformat(),
            "data_summary": validation_summary or {},
            "model_performance": training_result.get("models", {}),
        }
        report_path = Variable.get("performance_report_path", "/tmp/performance_report.json")
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        log.info(f"Saved performance report to {report_path}")
        # optionally push to MLflow as artifact
        try:
            import mlflow
            mlflow.log_artifact(report_path, artifact_path="reports")
        except Exception as e:
            log.warning(f"Failed to log performance report to MLflow: {e}")
        return report
    
     # -------------------------
    # Cleanup (always run)
    # -------------------------
    @task(task_id="cleanup", trigger_rule=TriggerRule.ALL_DONE)
    def cleanup_task(temp_dir: str = DATA_DIR, artifact_dir: str = ARTIFACT_DIR):
        # remove tmp directories safely
        try:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                log.info(f"Removed data dir: {temp_dir}")
        except Exception as e:
            log.warning(f"Cleanup data dir failed: {e}")

        try:
            # optionally keep artifacts, but remove temporary intermediate dirs
            temp_art = os.path.join(artifact_dir, "tmp")
            if os.path.exists(temp_art):
                shutil.rmtree(temp_art)
                log.info(f"Removed temp artifact dir: {temp_art}")
        except Exception as e:
            log.warning(f"Cleanup artifacts failed: {e}")

        return "cleanup_done"

    # -------------------------
    # DAG wiring
    # -------------------------
    extract_result = extract_data_task()
    validation_summary = validate_data_task(extract_result)
    training_result = prepare_and_train_task(extract_result, validation_summary)
    evaluation_result = evaluate_models_task(training_result)

    # Manual approval short-circuit: if returns False, downstream (register/transition) will be skipped
    approval = approval_check

    registration_result = register_best_models_task(evaluation_result)
    transition_result = transition_to_production_task(registration_result)

    report = generate_performance_report_task(training_result, validation_summary)
    cleanup = cleanup_task()

    # dependencies
    start >> extract_result >> validation_summary >> training_result >> evaluation_result
    evaluation_result >> approval  # approval gate
    approval >> registration_result >> transition_result
    # If approval fails, skip register/transition; we still generate report and cleanup
    evaluation_result >> report
    report >> cleanup
    registration_result >> cleanup
    transition_result >> cleanup

    end = EmptyOperator(task_id="end", trigger_rule=TriggerRule.NONE_FAILED)
    cleanup >> end



