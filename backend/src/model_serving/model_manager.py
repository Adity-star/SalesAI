import yaml
import mlflow
import json
import hashlib
import time
import asyncio
import pandas as pd
import numpy as np
from fastapi import HTTPException
from typing import Any, Dict, Optional
from datetime import datetime
from src.logger import logger
from prometheus_client import Counter, Histogram, Gauge
from src.utils.mlflow_utils import MLflowManager
from src.entity.prediction_entity import PredictionRequest,PredictionResponse,BatchPredictionRequest,BatchPredictionResponse


# Prometheus metrics
PREDICTION_COUNTER = Counter('predictions_total', 'Total predictions made', ['model_name', 'horizon', 'status'])
PREDICTION_DURATION = Histogram('prediction_duration_seconds', 'Time spent making predictions', ['model_name'])
MODEL_LOAD_GAUGE = Gauge('model_load_timestamp', 'Timestamp of last model load', ['model_name'])
ACTIVE_REQUESTS = Gauge('active_requests', 'Number of active prediction requests')



class ModelManager:
    def __init__(self, config_path: str = "config/ml_config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.mlflow_manager = MLflowManager(config_path)  # your helper class

        # loaded models dict, keyed by model name
        self.models: Dict[str, Any] = {}
        # metadata for models
        self.metadata: Dict[str, Dict[str, Any]] = {}
        self.model_version: Optional[str] = None

    def load_models(self, model_stage: str = "Production"):
        """Load models from MLflow registry and caching them."""
        logger.info(f"Loading models from stage = {model_stage}")
        try:
            # Example: load XGBoost
            xgb_ver = self.mlflow_manager.get_latest_model_version("xgboost", stage=model_stage)
            xgb_uri = f"models:/{self.mlflow_manager.registry_name}_xgboost/{xgb_ver['version']}"
            self.models["xgboost"] = mlflow.xgboost.load_model(xgb_uri)
            self.metadata["xgboost"] = xgb_ver

            # Load LightGBM
            lgb_ver = self.mlflow_manager.get_latest_model_version("lightgbm", stage=model_stage)
            lgb_uri = f"models:/{self.mlflow_manager.registry_name}_lightgbm/{lgb_ver['version']}"
            self.models["lightgbm"] = mlflow.lightgbm.load_model(lgb_uri)
            self.metadata["lightgbm"] = lgb_ver

            # Load other models (prophet etc) similarly
            # ...

            # Optionally, set a “default” or version
            self.model_version = xgb_ver["version"]
            logger.info("Models loaded successfully", extra={"version": self.model_version})

        except Exception as e:
            logger.error("Error loading models", extra={"error": str(e)})
            raise HTTPException(status_code=500, detail=f"Model loading failed: {e}")

    def get_model(self, model_name: str) -> Any:
        """Return the loaded model or error if not available."""
        if model_name not in self.models:
            raise HTTPException(status_code=404, detail=f"Model {model_name} not loaded")
        return self.models[model_name]

    def get_model_metadata(self, model_name: str) -> Dict[str, Any]:
        return self.metadata.get(model_name, {})

    def get_model_version(self, model_name: str) -> str:
        meta = self.get_model_metadata(model_name)
        return meta.get("version", "")

    def list_models(self) -> list[str]:
        return list(self.models.keys())




# -------------------------------
# Prediction Service
# ------------------------------

class PredictionService:
    def __init__(self, model_manager: ModelManager, feature_engineer, data_validator):
        self.model_manager = model_manager
        self.feature_engineer = feature_engineer
        self.data_validator = data_validator

    def _make_features_df(self, request: PredictionRequest) -> pd.DataFrame:
        # validate input if needed
        # self.data_validator.validate(...)  # optional

        # build dataframe
        data = {
            "store_id": [request.store_id],
            "item_id": [request.item_id],
            "date": [pd.to_datetime(request.date)],
            "sales": [0],  # dummy if needed
        }
        # include additional features if present
        if hasattr(request, "additional_features") and request.additional_features:
            for k, v in request.additional_features.items():
                data[k] = [v]

        df = pd.DataFrame(data)
        return df

    def _predict_from_model(self, model, X: np.ndarray) -> np.ndarray:
        # you might need to adjust depending on model interface
        return model.predict(X)

    def predict_single(self, request: PredictionRequest) -> PredictionResponse:
        start = time.time()
        # 1. Load the model you want
        model = self.model_manager.get_model(request.model_name)

        # 2. Feature engineering
        df = self._make_features_df(request)
        X = self.feature_engineer.prepare_features(df)  # your method

        # 3. Predict
        preds = self._predict_from_model(model, X)

        # 4. Prepare metadata
        version = self.model_manager.get_model_version(request.model_name)

        duration = time.time() - start
        logger.info("Prediction done", extra={"model": request.model_name, "duration": duration})

        # build and return response
        return PredictionResponse(
            store_id=request.store_id,
            item_id=request.item_id,
            date=request.date,
            prediction=float(preds[0]),
            confidence_interval_80=[0.0, 0.0],  # placeholder or compute
            confidence_interval_95=[0.0, 0.0],
            model_version=version,
            prediction_timestamp=datetime.now().isoformat()
        )

    async def predict_batch(self, request: BatchPredictionRequest) -> BatchPredictionResponse:
        # process each prediction (could parallelize)
        results = []
        failed = 0
        for pr in request.predictions:
            try:
                res = self.predict_single(pr)
                results.append(res)
            except Exception as e:
                logger.error("Predict batch item failed", extra={"error": str(e), "item": pr})
                failed += 1

        total = len(request.predictions)
        return BatchPredictionResponse(
            request_id=hashlib.md5(json.dumps([r.dict() for r in request.predictions]).encode()).hexdigest(),
            total_predictions=total,
            successful_predictions=total - failed,
            failed_predictions=failed,
            predictions=results,
            processing_time_seconds=0.0,
            model_name=request.predictions[0].model_name if results else "",
            model_version=self.model_manager.get_model_version(request.predictions[0].model_name) if results else "",
        )

    async def predict_single_async(self, request: PredictionRequest) -> PredictionResponse:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.predict_single, request)
