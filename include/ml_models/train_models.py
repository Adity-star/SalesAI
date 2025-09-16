import yaml
import joblib
from include.logger import logger
from typing import Dict, Any

import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

import xgboost as xbg
import lightgbm as lgb
from prophet import Prophet
import optuna 
import mlflow


class ModelTrainer:
    def __init__(self, config_path: str = r"C:\Users\Administrator\OneDrive\Desktop\SalesAI\include\config\ml_config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.model_config = self.config['models']
        self.training_config = self.config['training']
        