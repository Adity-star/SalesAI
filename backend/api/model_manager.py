

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta, date
import json
import pickle
import asyncio
from functools import lru_cache
import hashlib
import time

# FastAPI and dependencies
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, status, Query
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field, validator
import uvicorn

# ML libraries
import lightgbm as lgb
import joblib

# Monitoring and observability
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from prometheus_client.exposition import make_wsgi_app
import structlog

# Configuration
from src.utils.config_loader import ConfigLoader
from api.prediction import ModelInfo, PredictionRequest,PredictionResponse,BatchPredictionRequest,BatchPredictionResponse



from pathlib import Path
from src.logger import logger

# Prometheus metrics
PREDICTION_COUNTER = Counter('predictions_total', 'Total predictions made', ['model_name', 'horizon', 'status'])
PREDICTION_DURATION = Histogram('prediction_duration_seconds', 'Time spent making predictions', ['model_name'])
MODEL_LOAD_GAUGE = Gauge('model_load_timestamp', 'Timestamp of last model load', ['model_name'])
ACTIVE_REQUESTS = Gauge('active_requests', 'Number of active prediction requests')


# -------------------------------
# Model Manager
# -------------------------------

class ModelManager:
    """Manages model loading, caching, and versioning."""
    
    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.models = {}
        self.model_metadata = {}
        self.feature_pipeline = None
        
        logger.info("Initialized ModelManager", model_dir=str(self.model_dir))
    
    @lru_cache(maxsize=10)
    def load_model(self, model_name: str, version: str = "latest") -> Any:
        """Load model with caching."""
        cache_key = f"{model_name}_{version}"
        
        if cache_key in self.models:
            logger.debug("Model loaded from cache", model_name=model_name, version=version)
            return self.models[cache_key]
        
        model_path = self._get_model_path(model_name, version)
        
        if not model_path.exists():
            raise HTTPException(
                status_code=404, 
                detail=f"Model {model_name} version {version} not found"
            )
        
        logger.info("Loading model from disk", model_path=str(model_path))
        start_time = time.time()
        
        try:
            if model_name == "lightgbm":
                model = lgb.Booster(model_file=str(model_path / "model.txt"))
            else:
                model = joblib.load(model_path / "model.pkl")
            
            # Load metadata
            metadata_path = model_path / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path) as f:
                    metadata = json.load(f)
                self.model_metadata[cache_key] = metadata
            
            self.models[cache_key] = model
            
            load_time = time.time() - start_time
            MODEL_LOAD_GAUGE.labels(model_name=model_name).set(time.time())
            
            logger.info("Model loaded successfully", 
                       model_name=model_name, version=version, load_time=load_time)
            
            return model
            
        except Exception as e:
            logger.error("Failed to load model", 
                        model_name=model_name, version=version, error=str(e))
            raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")
    
    def _get_model_path(self, model_name: str, version: str) -> Path:
        """Get path to model artifacts."""
        if version == "latest":
            # Find latest version
            model_dirs = [d for d in (self.model_dir / model_name).iterdir() 
                         if d.is_dir() and d.name.startswith("v")]
            if not model_dirs:
                raise HTTPException(404, f"No versions found for model {model_name}")
            
            latest_dir = max(model_dirs, key=lambda x: x.name)
            return latest_dir
        else:
            return self.model_dir / model_name / f"v{version}"
    
    def get_model_info(self, model_name: str, version: str = "latest") -> ModelInfo:
        """Get model information."""
        cache_key = f"{model_name}_{version}"
        metadata = self.model_metadata.get(cache_key, {})
        
        return ModelInfo(
            model_name=model_name,
            model_version=version,
            model_type=metadata.get("model_type", "unknown"),
            training_date=datetime.fromisoformat(metadata.get("training_date", "2024-01-01T00:00:00")),
            feature_count=metadata.get("feature_count", 0),
            performance_metrics=metadata.get("performance_metrics", {}),
            last_loaded=datetime.now()
        )
    
    def list_models(self) -> List[str]:
        """List available models."""
        if not self.model_dir.exists():
            return []
        
        return [d.name for d in self.model_dir.iterdir() if d.is_dir()]


# -------------------------------
# Feature Engineering Pipeline
# -------------------------------

class PredictionFeatureEngineer:
    """Real-time feature engineering for predictions."""
    
    def __init__(self, feature_config: Dict = None):
        self.config = feature_config or {}
        self.feature_cache = {}
        
    @lru_cache(maxsize=1000)
    def engineer_features(self, store_id: str, item_id: str, pred_date: date) -> pd.DataFrame:
        """Engineer features for a single prediction."""
        logger.debug("Engineering features", store_id=store_id, item_id=item_id, pred_date=str(pred_date))
        
        # Create base feature row
        features = {
            'store_id': store_id,
            'item_id': item_id,
            'date': pd.to_datetime(pred_date),
        }
        
        # Add date features
        features.update(self._add_date_features(pred_date))
        
        # Add historical features (would need historical data store)
        features.update(self._add_historical_features(store_id, item_id, pred_date))
        
        # Convert to DataFrame
        df = pd.DataFrame([features])
        return df
    
    def _add_date_features(self, pred_date: date) -> Dict[str, Any]:
        """Add date-based features."""
        dt = pd.to_datetime(pred_date)
        
        return {
            'year': dt.year,
            'month': dt.month,
            'day': dt.day,
            'dayofweek': dt.dayofweek,
            'quarter': dt.quarter,
            'weekofyear': dt.isocalendar().week,
            'is_weekend': int(dt.dayofweek >= 5),
            'is_month_start': int(dt.day <= 5),
            'is_month_end': int(dt.day >= 25),
            'month_sin': np.sin(2 * np.pi * dt.month / 12),
            'month_cos': np.cos(2 * np.pi * dt.month / 12),
            'dow_sin': np.sin(2 * np.pi * dt.dayofweek / 7),
            'dow_cos': np.cos(2 * np.pi * dt.dayofweek / 7)
        }
    
    def _add_historical_features(self, store_id: str, item_id: str, pred_date: date) -> Dict[str, Any]:
        """Add historical features (simplified - would need historical data store)."""
        # In production, this would query historical data from feature store
        # For now, return placeholder features
        return {
            'sales_lag_7': 10.0,    # Would be actual historical values
            'sales_lag_14': 8.0,
            'sales_lag_28': 12.0,
            'sales_roll_7_mean': 9.5,
            'sales_roll_14_mean': 10.2,
            'price_lag_7': 5.99,
            'price_vs_category': 1.05
        }
    

# -------------------------------
# Prediction Service
# -------------------------------

class PredictionService:
    """Core prediction service."""
    
    def __init__(self, model_manager: ModelManager, feature_engineer: PredictionFeatureEngineer):
        self.model_manager = model_manager
        self.feature_engineer = feature_engineer
        
    async def predict_single(self, request: PredictionRequest) -> PredictionResponse:
        """Make single prediction."""
        start_time = time.time()
        ACTIVE_REQUESTS.inc()
        
        try:
            # Load model
            model = self.model_manager.load_model(request.model_name)
            
            # Engineer features
            features_df = self.feature_engineer.engineer_features(
                request.store_id, request.item_id, request.date
            )
            
            # Make prediction
            with PREDICTION_DURATION.labels(model_name=request.model_name).time():
                if request.model_name == "lightgbm":
                    prediction = model.predict(features_df.select_dtypes(include=[np.number]))
                    prediction = max(0, prediction[0])  # Ensure non-negative
                else:
                    prediction = max(0, model.predict(features_df)[0])
            
            # Get model metadata
            model_info = self.model_manager.get_model_info(request.model_name)
            
            PREDICTION_COUNTER.labels(
                model_name=request.model_name, 
                horizon=request.horizon, 
                status='success'
            ).inc()
            
            processing_time = time.time() - start_time
            logger.info("Prediction completed", 
                       store_id=request.store_id, item_id=request.item_id,
                       prediction=prediction, processing_time=processing_time)
            
            return PredictionResponse(
                store_id=request.store_id,
                item_id=request.item_id,
                date=str(request.date),
                prediction=float(prediction),
                model_name=request.model_name,
                model_version=model_info.model_version,
                prediction_timestamp=datetime.now(),
                features_used=list(features_df.select_dtypes(include=[np.number]).columns)
            )
            
        except Exception as e:
            PREDICTION_COUNTER.labels(
                model_name=request.model_name, 
                horizon=request.horizon, 
                status='error'
            ).inc()
            
            logger.error("Prediction failed", 
                        store_id=request.store_id, item_id=request.item_id, 
                        error=str(e))
            raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
            
        finally:
            ACTIVE_REQUESTS.dec()
    
    async def predict_batch(self, request: BatchPredictionRequest) -> BatchPredictionResponse:
        """Make batch predictions."""
        start_time = time.time()
        request_id = hashlib.md5(json.dumps(request.dict(), sort_keys=True).encode()).hexdigest()[:8]
        
        logger.info("Starting batch prediction", 
                   request_id=request_id, total_items=len(request.items))
        
        predictions = []
        successful = 0
        failed = 0
        
        # Load model once for batch
        model = self.model_manager.load_model(request.model_name)
        model_info = self.model_manager.get_model_info(request.model_name)
        
        # Process items
        for item in request.items:
            try:
                pred_request = PredictionRequest(
                    store_id=item['store_id'],
                    item_id=item['item_id'],
                    date=item['date'],
                    horizon=request.horizon,
                    model_name=request.model_name
                )
                
                response = await self.predict_single(pred_request)
                predictions.append(response)
                successful += 1
                
            except Exception as e:
                logger.warning("Item prediction failed", 
                              item=item, error=str(e))
                failed += 1
                continue
        
        processing_time = time.time() - start_time
        
        logger.info("Batch prediction completed",
                   request_id=request_id, successful=successful, failed=failed,
                   processing_time=processing_time)
        
        return BatchPredictionResponse(
            request_id=request_id,
            total_predictions=len(request.items),
            successful_predictions=successful,
            failed_predictions=failed,
            predictions=predictions,
            processing_time_seconds=processing_time,
            model_name=request.model_name,
            model_version=model_info.model_version
        )
