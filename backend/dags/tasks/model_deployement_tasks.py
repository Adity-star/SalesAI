
import logging
from src import logger
import os
import sys
from datetime import datetime, timedelta
from airflow.models import Variable


logger = logging.getLogger(__name__)


APPROVAL_VAR = Variable.get("approve_production_var", "approve_production")


def register_best_models_task(evaluation_result: dict) -> dict:
        from src.utils.mlflow_utils import MLflowManager

        best_model = evaluation_result.get("best_model")
        run_id = evaluation_result.get("mlflow_run_id")
        mlflow_manager = MLflowManager()
        versions = {}
        # register both tree-based models if present: xgboost, lightgbm
        for model_name in ["xgboost", "lightgbm"]:
            try:
                ver = mlflow_manager.register_model(run_id, model_name, model_name)
                versions[model_name] = ver
                logger.info(f"Registered {model_name} version {ver}")
            except Exception as e:
                logger.warning(f"Failed to register {model_name}: {e}")
        return {"registered_versions": versions}


def transition_to_production_task(registration_result: dict) -> str:
        from src.utils.mlflow_utils import MLflowManager

        mlflow_manager = MLflowManager()
        out = []
        for model_name, version in registration_result.get("registered_versions", {}).items():
            try:
                mlflow_manager.transition_model_stage(model_name, version, "Production")
                out.append(f"{model_name}:v{version}")
                logger.info(f"Transitioned {model_name} v{version} to Production")
            except Exception as e:
                logger.warning(f"Failed to transition {model_name} v{version}: {e}")
        return ";".join(out)

def generate_performance_report_task(training_result: dict, validation_summary: dict) -> dict:
        import json
        report = {
            "timestamp": datetime.now().isoformat(),
            "data_summary": validation_summary or {},
            "model_performance": training_result.get("models", {}),
        }
        report_path = Variable.get("performance_report_path", "/tmp/performance_report.json")
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Saved performance report to {report_path}")
        # optionally push to MLflow as artifact
        try:
            import mlflow
            mlflow.log_artifact(report_path, artifact_path="reports")
        except Exception as e:
            logger.warning(f"Failed to log performance report to MLflow: {e}")
        return report


# def cleanup_task():
       
#         # remove tmp directories safely
#         import shutil
#         try:
#             if os.path.exists(temp_dir):
#                 shutil.rmtree(temp_dir)
#                 logger.info(f"Removed data dir: {temp_dir}")
#         except Exception as e:
#             logger.warning(f"Cleanup data dir failed: {e}")

#         try:
#             # optionally keep artifacts, but remove temporary intermediate dirs
#             temp_art = os.path.join(artifact_dir, "tmp")
#             if os.path.exists(temp_art):
#                 shutil.rmtree(temp_art)
#                 logger.info(f"Removed temp artifact dir: {temp_art}")
#         except Exception as e:
#             logger.warning(f"Cleanup artifacts failed: {e}")

#         return "cleanup_done"