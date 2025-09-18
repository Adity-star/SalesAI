import os
import logging
import yaml
import mlflow
import joblib
import pandas as pd
import numpy as np

from datetime import datetime
from typing import Optional, Dict, Any
from mlflow.tracking import MlflowClient
from mlflow import pyfunc
from urllib.parse import urljoin
import traceback

from .service_discovery import get_mlflow_endpoint, get_minio_endpoint

import logging
from include.logger import logger

logger = logging.getLogger(__name__)

class MLflowManager:
    def __init__(self, config_path: str = "/usr/local/airflow/include/config/ml_config.yaml"):
        self._load_config(config_path)
        self._setup_mlflow()
        self._setup_minio()
        self.client = MlflowClient(tracking_uri=self.tracking_uri)
       
    def _load_config(self, config_path:str):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        mlflow_cfg = self.config.get('mlflow', {})
        self.experiment_name = os.getenv("MLFLOW_EXPERIMENT", mlflow_cfg.get("experiment_name", "default"))
        self.registry_name = os.getenv("MLFLOW_REGISTRY", mlflow_cfg.get("registry_name", "default_registry"))

    def _setup_mlflow(self):
        self.tracking_uri = get_mlflow_endpoint(retries=5, backoff=1.5)
        mlflow.set_tracking_uri(self.tracking_uri)

        try:
            mlflow.set_experiment(self.experiment_name)
            logger.info(f"Experiment set: {self.experiment_name}")
        except Exception as e:
            logger.error(f"Failed to set MLflow experiment: {e}\n{traceback.format_exc()}")
            raise RuntimeError("Failed to connect or set MLflow experiment")
    
    def _setup_minio(self):
        os.environ['MLFLOW_S3_ENDPOINT_URL'] = get_minio_endpoint()
        os.environ['AWS_ACCESS_KEY_ID'] = os.getenv('AWS_ACCESS_KEY_ID', 'minioadmin')
        os.environ['AWS_SECRET_ACCESS_KEY'] = os.getenv('AWS_SECRET_ACCESS_KEY', 'minioadmin')


    def start_run(self, run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None) -> str:
        if mlflow.active_run():
            mlflow.end_run()

        run_name = run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run = mlflow.start_run(run_name=run_name, tags=tags)
        logger.info(f"Started run: {run.info.run_id}, name={run_name}")
        return run.info.run_id
    
    def log_params(self, params: Dict[str, Any]):
        for k, v in params.items():
            mlflow.log_param(k, v)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        for k, v in metrics.items():
            mlflow.log_metric(k, v, step=step)
    
    def log_model(
        self, model, model_name: str, 
        input_example: Optional[pd.DataFrame] = None,
        signature: Optional[Any] = None,
        registered_model_name: Optional[str] = None
    ):
        try:
            import tempfile
            with tempfile.TemporaryDirectory() as tmpdir:
                model_path = os.path.join(tmpdir, f"{model_name}_model.pkl")
                joblib.dump(model, model_path)

                mlflow.log_artifact(model_path, artifact_path=f"models/{model_name}")
                logger.info(f"Logged model {model_name} as artifact")

                metadata = {
                    "model_type": model_name,
                    "framework": type(model).__module__,
                    "class": type(model).__name__,
                    "timestamp": datetime.now().isoformat()
                }
                metadata_path = os.path.join(tmpdir, f"{model_name}_metadata.yaml")
                with open(metadata_path, 'w') as f:
                    yaml.dump(metadata, f)
                mlflow.log_artifact(metadata_path, artifact_path=f"models/{model_name}")
        except Exception as e:
            logger.error(f"Error logging model {model_name}: {e}\n{traceback.format_exc()}")

    def log_artifacts(self, artifact_path: str):
        mlflow.log_artifacts(artifact_path)

    def log_figure(self, figure, artifact_file: str):
        mlflow.log_figure(figure, artifact_file)

    def end_run(self, status: str = "FINISHED"):
        run = mlflow.active_run()
        run_id = run.info.run_id if run else None

        mlflow.end_run(status=status)
        logger.info(f"Ended run {run_id} with status {status}")

        # Optional: Sync to S3
        if run_id and status == "FINISHED":
            try:
                from utils.mlflow_s3_utils import MLflowS3Manager
                s3_manager = MLflowS3Manager()
                s3_manager.sync_mlflow_artifacts_to_s3(run_id)
                logger.info(f"S3 sync completed for run {run_id}")
            except Exception as e:
                logger.warning(f"Artifact sync failed: {e}")
    
    def get_best_model(self, metric: str = "rmse", ascending: bool = True) -> Dict[str, Any]:
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[f"metrics.{metric} {'ASC' if ascending else 'DESC'}"],
            max_results=1
        )
        if runs.empty:
            raise ValueError("No runs found.")

        best = runs.iloc[0]
        return {
            "run_id": best["run_id"],
            "metrics": {k.replace("metrics.", ""): v for k, v in best.items() if k.startswith("metrics.")},
            "params": {k.replace("params.", ""): v for k, v in best.items() if k.startswith("params.")}
        }
    
    def load_model(self, model_uri: str):
        try:
            return pyfunc.load_model(model_uri)
        except:
            if "runs:/" in model_uri:
                run_id = model_uri.split("/")[1]
                artifact_path = "/".join(model_uri.split("/")[2:])
                local_path = mlflow.artifacts.download_artifacts(
                    run_id=run_id,
                    artifact_path=f"{artifact_path}_model.pkl"
                )
                return joblib.load(local_path)
            raise

    def register_model(self, run_id: str, model_name: str, artifact_path: str) -> str:
        try:
            uri = f"runs:/{run_id}/{artifact_path}"
            version = mlflow.register_model(uri, f"{self.registry_name}_{model_name}")
            logger.info(f"Registered model version {version.version}")
            return version.version
        except Exception as e:
            logger.warning(f"Model registration failed: {e}")
            return run_id  # fallback

    def transition_model_stage(self, model_name: str, version: str, stage: str):
        try:
            self.client.transition_model_version_stage(
                name=f"{self.registry_name}_{model_name}",
                version=version,
                stage=stage
            )
            logger.info(f"Transitioned model {model_name} v{version} to stage {stage}")
        except Exception as e:
            logger.warning(f"Stage transition failed: {e}")

    def get_latest_model_version(self, model_name: str, stage: Optional[str] = None) -> Dict[str, Any]:
        try:
            filter_string = f"name='{self.registry_name}_{model_name}'"
            if stage:
                filter_string += f" AND current_stage='{stage}'"
            versions = self.client.search_model_versions(filter_string)
            if not versions:
                raise ValueError("No model versions found.")
            latest = max(versions, key=lambda x: int(x.version))
            return {
                "version": latest.version,
                "stage": latest.current_stage,
                "run_id": latest.run_id,
                "source": latest.source
            }
        except Exception:
            logger.warning("Model version lookup failed, falling back to best run.")
            best = self.get_best_model()
            return {
                "version": best["run_id"],
                "stage": "None",
                "run_id": best["run_id"],
                "source": f"runs:/{best['run_id']}/models"
            }

    def get_run_ui_link(self, run_id: str) -> Optional[str]:
        if not self.tracking_uri or "http" not in self.tracking_uri:
            return None
        return urljoin(self.tracking_uri, f"#/experiments/{self.experiment_name}/runs/{run_id}")