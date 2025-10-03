
from typing import List
from datetime import datetime
from functools import lru_cache
import os
from dotenv import load_dotenv
import sys

# FastAPI and dependencies
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
import uvicorn


from src.logger import logger

# Configuration
from src.monitoring.drift import M5MonitoringSystem, MonitoringConfig
from src.model_serving.model_manager import ModelManager,PredictionService
from src.entity.prediction_entity import ModelInfo, PredictionRequest,PredictionResponse,BatchPredictionRequest,BatchPredictionResponse

load_dotenv()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))


# -------------------------------
# Dependency Injection Providers
# -------------------------------

@lru_cache(maxsize=1)
def get_model_manager() -> ModelManager:
    """Get a singleton ModelManager instance and preload the default model."""
    model_manager = ModelManager()
    try:
        models = model_manager.list_models()
        if models:
            default_model = models[0]
            model_manager.load_model(default_model)
            logger.info("Preloaded default model", model_name=default_model)
    except Exception as e:
        logger.warning("Failed to preload model during startup", error=str(e))
    return model_manager



@lru_cache(maxsize=1)
def get_monitoring_system() -> M5MonitoringSystem:
    """Get a singleton M5MonitoringSystem instance."""
    config = MonitoringConfig(
        db_path="monitoring_db.sqlite",
        drift_threshold=0.05,
        accuracy_window=7
    )
    return M5MonitoringSystem(config)

def get_prediction_service(
    model_manager: ModelManager = Depends(get_model_manager),
) -> PredictionService:
    """Get a PredictionService instance."""
    return PredictionService(model_manager)


# -------------------------------
# FastAPI Application
# -------------------------------

app = FastAPI(
    title="Forecasting API",
    description="Production API for M5 Walmart sales forecasting",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("FRONTEND_URL", "http://localhost:3000")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Authentication
security = HTTPBearer()
API_TOKEN = os.getenv("API_TOKEN")

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify authentication token."""
    if not API_TOKEN or credentials.credentials != API_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials

# -------------------------------
# API Endpoints
# -------------------------------

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "M5 Walmart Forecasting API",
        "version": "1.0.0",
        "description": "Production API for sales forecasting",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "predict_batch": "/predict/batch",
            "models": "/models",
            "metrics": "/metrics"
        }
    }

@app.get("/health")
async def health_check(model_manager: ModelManager = Depends(get_model_manager)):
    """Health check endpoint."""
    try:
        models = model_manager.list_models()
        return {
            "status": "healthy",
            "timestamp": datetime.now(),
            "available_models": models,
            "version": "1.0.0"
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {e}")

@app.post("/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    prediction_service: PredictionService = Depends(get_prediction_service),
    monitoring_system: M5MonitoringSystem = Depends(get_monitoring_system),
    credentials: HTTPAuthorizationCredentials = Depends(verify_token)
):
    """Make single prediction."""
    response = await prediction_service.predict_single(request)
    try:
        monitoring_system.database.store_prediction(
            model_name=response.model_name,
            prediction_value=response.prediction,
            actual_value=None,
            features=request.features,
            prediction_timestamp=datetime.now()
        )
    except Exception as e:
        logger.error("Failed to log prediction to monitoring", error=str(e))
    return response

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    request: BatchPredictionRequest,
    prediction_service: PredictionService = Depends(get_prediction_service),
    monitoring_system: M5MonitoringSystem = Depends(get_monitoring_system),
    credentials: HTTPAuthorizationCredentials = Depends(verify_token)
):
    """Make batch predictions."""
    response = await prediction_service.predict_batch(request)
    try:
        prediction_timestamp = datetime.now()
        for row_features, row_prediction in zip(request.instances, response.predictions):
            monitoring_system.database.store_prediction(
                model_name=response.model_name,
                prediction_value=row_prediction,
                actual_value=None,
                features=row_features,
                prediction_timestamp=prediction_timestamp
            )
        logger.info("Batch predictions logged to monitoring", total=len(response.predictions))
    except Exception as e:
        logger.error("Failed to log batch predictions to monitoring", error=str(e))
    return response

@app.get("/models", response_model=List[str])
async def list_models(
    model_manager: ModelManager = Depends(get_model_manager),
    credentials: HTTPAuthorizationCredentials = Depends(verify_token)
):
    """List available models."""
    return model_manager.list_models()

@app.get("/models/{model_name}", response_model=ModelInfo)
async def get_model_info(
    model_name: str,
    version: str = "latest",
    model_manager: ModelManager = Depends(get_model_manager),
    credentials: HTTPAuthorizationCredentials = Depends(verify_token)
):
    """Get model information."""
    try:
        return model_manager.get_model_info(model_name, version)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Model not found: {e}")

@app.get("/metrics")
async def get_metrics():
    """Get Prometheus metrics."""
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    from fastapi import Response
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/monitor/report/{model_name}")
async def monitoring_report(
    model_name: str,
    days: int = 7,
    monitoring_system: M5MonitoringSystem = Depends(get_monitoring_system)
):
    """Generate a monitoring report for a given model."""
    report = monitoring_system.generate_monitoring_report(model_name, days)
    return report

@app.get("/monitor/dashboard/{model_name}")
async def monitoring_dashboard(
    model_name: str,
    days: int = 30,
    monitoring_system: M5MonitoringSystem = Depends(get_monitoring_system)
):
    """Generate a monitoring dashboard and return its path."""
    output_dir = os.getenv("MONITORING_DASHBOARD_DIR", "/tmp")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{model_name}_dashboard.html")
    
    monitoring_system.create_monitoring_dashboard(model_name, days, output_path)
    
    return {"message": "Dashboard created", "path": output_path}

# -------------------------------
# CLI Runner
# -------------------------------

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Forecasting API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--log-level", default="info", help="Log level")
    
    args = parser.parse_args()
    
    uvicorn.run(
        "main:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload,
        log_level=args.log_level
    )