"""
Centralized configuration module for SalesAI project.
Single source of truth for paths, URIs, and hyperparameters.
"""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Data directories
DATA_DIR = os.getenv("DATA_DIR", str(PROJECT_ROOT / "data"))
PROCESSED_DIR = os.getenv("PROCESSED_DIR", str(PROJECT_ROOT / "data" / "processed"))
RAW_DATA_DIR = os.getenv("RAW_DATA_DIR", str(PROJECT_ROOT / "data" / "raw"))

# Model directories
MODEL_DIR = os.getenv("MODEL_DIR", str(PROJECT_ROOT / "models"))
ARTIFACTS_DIR = os.getenv("ARTIFACTS_DIR", str(PROJECT_ROOT / "artifacts"))

# MLflow configuration
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "sales_forecasting")

# Feature store
FEATURE_STORE_PATH = os.getenv("FEATURE_STORE_PATH", str(PROJECT_ROOT / "data" / "feature_store"))

# Logging configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///sales_forecasting.db")






# Airflow configuration
AIRFLOW_PARAMS = {
    "dag_owner": "sales_team",
    "retry_delay_minutes": 5,
    "max_retries": 2,
    "email_on_failure": True,
    "email_on_retry": False
}