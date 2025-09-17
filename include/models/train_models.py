import yaml
import joblib
from include.logger import logger
from typing import Dict, List, Tuple, Optional, Any

import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, LabelEncoder

import xgboost as xgb
import lightgbm as lgb
from prophet import Prophet
import optuna 
import mlflow

from utils.mlflow_utils import MLflowManager
from include.feature_engineering.feature_pipeline import FeaturePipeline
from include.data_validation.validators import DataValidator

logger = logger.getLogger(__name__)


class ModelTrainer:
    def __init__(self, config_path: str = r"C:\Users\Administrator\OneDrive\Desktop\SalesAI\include\config\ml_config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.model_config = self.config['models']
        self.training_config = self.config['training']

        self.mlflow_manager = MLflowManager(config_path)
        self.feature_engineer = FeaturePipeline(config_path)

        self.data_validator = DataValidator(config_path)

        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, Any] = {}
        self.encoders: Dict[str, Any] = {}
        self.feature_cols: List[str] = []

    def prepare_data(
        self, df: pd.DataFrame, target_col: str = "sales",
        date_col: str = "date", group_cols: Optional[List[str]] = None,
        categorical_cols: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

        logger.info("Preparing data for training")

        required_cols = [date_col, target_col]
        if group_cols:
            required_cols.extend(group_cols)

        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns for training: {missing_cols}")

        
        df_features = self.feature_engineer.create_all_features(
            df, target_col, date_col, group_cols, categorical_cols)
        
         # Chronological split
        df_sorted = df_features.sort_values(date_col)
        train_size = int(len(df_sorted) * (1 - self.training_config["test_size"] - self.training_config["validation_size"]))
        val_size = int(len(df_sorted) * self.training_config["validation_size"])

        train_df = df_sorted[:train_size]
        val_df = df_sorted[train_size:train_size + val_size]
        test_df = df_sorted[train_size + val_size:]

        # Drop rows with missing target
        train_df = train_df.dropna(subset=[target_col])
        val_df = val_df.dropna(subset=[target_col])
        test_df = test_df.dropna(subset=[target_col])

        logger.info(f"Data split → Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        return train_df, val_df, test_df
    

    import pandas as pd
import numpy as np
import yaml
from typing import List, Tuple, Optional, Dict, Any
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler, RobustScaler

class ModelTrainer:
    def __init__(self, config_path: str = "/usr/local/airflow/include/config/ml_config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.model_config = self.config["models"]
        self.training_config = self.config["training"]
        self.mlflow_manager = MLflowManager(config_path)
        self.feature_engineer = FeaturePipeline(config_path)
        self.data_validator = DataValidator(config_path)

        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, Any] = {}
        self.encoders: Dict[str, Any] = {}
        self.feature_cols: List[str] = []

    

    def preprocess_features(
        self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
        target_col: str, exclude_cols: List[str] = ["date"]
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:

        feature_cols = [col for col in train_df.columns if col not in exclude_cols + [target_col]]
        self.feature_cols = feature_cols

        X_train, X_val, X_test = train_df[feature_cols].copy(), val_df[feature_cols].copy(), test_df[feature_cols].copy()
        y_train, y_val, y_test = train_df[target_col].values, val_df[target_col].values, test_df[target_col].values

        # Handle categorical encoding
        categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
        for col in categorical_cols:
            if self.training_config.get("encoder", "label") == "label":
                if col not in self.encoders:
                    self.encoders[col] = LabelEncoder()
                    X_train.loc[:, col] = self.encoders[col].fit_transform(X_train[col].astype(str))
                else:
                    X_train.loc[:, col] = self.encoders[col].transform(X_train[col].astype(str))
                X_val.loc[:, col] = self.encoders[col].transform(X_val[col].astype(str))
                X_test.loc[:, col] = self.encoders[col].transform(X_test[col].astype(str))
            elif self.training_config["encoder"] == "onehot":
                ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
                encoded = ohe.fit_transform(X_train[[col]])
                encoded_cols = [f"{col}_{cat}" for cat in ohe.categories_[0]]
                X_train = X_train.drop(col, axis=1).join(pd.DataFrame(encoded, columns=encoded_cols, index=X_train.index))
                X_val = X_val.drop(col, axis=1).join(pd.DataFrame(ohe.transform(X_val[[col]]), columns=encoded_cols, index=X_val.index))
                X_test = X_test.drop(col, axis=1).join(pd.DataFrame(ohe.transform(X_test[[col]]), columns=encoded_cols, index=X_test.index))
                self.encoders[col] = ohe

        # Scale numeric features
        scaler_type = self.training_config.get("scaler", "standard")
        if scaler_type == "standard":
            scaler = StandardScaler()
        elif scaler_type == "minmax":
            scaler = MinMaxScaler()
        elif scaler_type == "robust":
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unsupported scaler type: {scaler_type}")

        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
        X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns, index=X_val.index)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

        self.scalers["scaler"] = scaler
        logger.info(f"Preprocessing complete. Features: {len(self.feature_cols)}")

        return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test
    

    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
            'r2': r2_score(y_true, y_pred)
        }
        return metrics
    

    def train_xgboost(self, 
                  X_train: np.ndarray, y_train: np.ndarray,
                  X_val: np.ndarray, y_val: np.ndarray,
                  use_optuna: bool = True) -> xgb.XGBRegressor:
        """
        Train an XGBoost regressor with optional Optuna optimization.
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            use_optuna: Whether to run Optuna hyperparameter search
        
        Returns:
            Trained XGBRegressor model
        """
        logger.info("Training XGBoost model")

        best_params = None

        if use_optuna:
            logger.info("Starting Optuna hyperparameter search for XGBoost")

            def objective(trial):
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 12),
                    'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
                    'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                    'gamma': trial.suggest_float('gamma', 0, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 5.0),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 5.0),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                    'random_state': 42,
                    'tree_method': 'hist'  # faster training, GPU if available
                }

                model = xgb.XGBRegressor(**params)
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    early_stopping_rounds=50,
                    verbose=False
                )
                y_pred = model.predict(X_val)
                rmse = np.sqrt(mean_squared_error(y_val, y_pred))
                return rmse

            study = optuna.create_study(
                direction="minimize",
                sampler=optuna.samplers.TPESampler(seed=42),
                pruner=optuna.pruners.MedianPruner()
            )
            study.optimize(objective, n_trials=self.training_config.get('optuna_trials', 50))

            best_params = study.best_params
            logger.info(f"Optuna best params: {best_params}")
        else:
            best_params = self.model_config['xgboost']['params']
            logger.info("Using config params for XGBoost")

        # Final model training with best params
        best_params.update({
            "random_state": 42,
            "tree_method": "hist"  # switch to "gpu_hist" if GPU is available
        })

        model = xgb.XGBRegressor(**best_params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=50,
            verbose=True
        )

        # Save trained model
        self.models['xgboost'] = model

        return model
    
    def train_lightgbm(self, 
                   X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray,
                   use_optuna: bool = True) -> lgb.LGBMRegressor:
        """
        Train a LightGBM regressor with optional Optuna optimization.
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            use_optuna: Whether to run Optuna hyperparameter search
        
        Returns:
            Trained LGBMRegressor model
        """
        logger.info("Training LightGBM model")

        best_params = None

        if use_optuna:
            logger.info("Starting Optuna hyperparameter search for LightGBM")

            def objective(trial):
                params = {
                    'num_leaves': trial.suggest_int('num_leaves', 31, 256),
                    'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
                    'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                    'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                    'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 5.0),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 5.0),
                    'min_split_gain': trial.suggest_float('min_split_gain', 0, 1.0),
                    'random_state': 42,
                    'verbosity': -1,
                    'objective': 'regression',
                    'metric': 'rmse',
                    'boosting_type': 'gbdt'
                }

                model = lgb.LGBMRegressor(**params)
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
                )

                y_pred = model.predict(X_val)
                rmse = np.sqrt(mean_squared_error(y_val, y_pred))
                return rmse

            study = optuna.create_study(
                direction="minimize",
                sampler=optuna.samplers.TPESampler(seed=42),
                pruner=optuna.pruners.MedianPruner()
            )
            study.optimize(objective, n_trials=self.training_config.get('optuna_trials', 50))

            best_params = study.best_params
            logger.info(f"Optuna best params: {best_params}")

            best_params.update({
                'random_state': 42,
                'verbosity': -1,
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt'
            })
        else:
            best_params = self.model_config['lightgbm']['params']
            logger.info("Using config params for LightGBM")

        # Final model training
        model = lgb.LGBMRegressor(**best_params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)]
        )

        self.models['lightgbm'] = model

        return model
    

    def train_prophet(self, 
                  train_df: pd.DataFrame, 
                  val_df: pd.DataFrame,
                  date_col: str = 'date', 
                  target_col: str = 'sales') -> Prophet:
        """
        Train a Prophet model with optional regressors and log results to MLflow.
        
        Args:
            train_df: Training DataFrame
            val_df: Validation DataFrame
            date_col: Column name for dates
            target_col: Column name for target variable
        
        Returns:
            Trained Prophet model
        """
        logger.info("Training Prophet model")

        # Prepare training data
        prophet_train = train_df[[date_col, target_col]].rename(
            columns={date_col: 'ds', target_col: 'y'}
        ).dropna().sort_values('ds')

        # Load params
        prophet_params = self.model_config['prophet']['params'].copy()

        # Override defaults for stability
        prophet_params.update({
            'stan_backend': 'CMDSTANPY',
            'mcmc_samples': 0,         # faster, no Bayesian sampling
            'uncertainty_samples': 100 # smaller uncertainty for speed
        })

        try:
            model = Prophet(**prophet_params)

            # --- Regressor selection ---
            numeric_cols = train_df.select_dtypes(include=[np.number]).columns
            regressor_cols = [c for c in numeric_cols if c not in [target_col, 'year', 'month', 'day', 'week', 'quarter']]

            if len(regressor_cols) > 5:
                variances = {c: train_df[c].var() for c in regressor_cols}
                regressor_cols = sorted(variances, key=variances.get, reverse=True)[:5]

            for col in regressor_cols:
                if train_df[col].std() > 0:
                    model.add_regressor(col)
                    prophet_train[col] = train_df[col]

            # --- Fit model ---
            model.fit(prophet_train)
            self.models['prophet'] = model

        except Exception as e:
            logger.error(f"Prophet training failed: {e}")
            logger.info("Retrying Prophet with minimal config...")

            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                changepoint_prior_scale=0.05,
                seasonality_prior_scale=10.0,
                uncertainty_samples=50,
                mcmc_samples=0
            )
            model.fit(prophet_train[['ds', 'y']])
            self.models['prophet'] = model

        # --- Validation evaluation ---
        prophet_val = val_df[[date_col, target_col]].rename(
            columns={date_col: 'ds', target_col: 'y'}
        ).dropna().sort_values('ds')

        # Add regressors if available
        for col in regressor_cols if 'regressor_cols' in locals() else []:
            if col in val_df:
                prophet_val[col] = val_df[col]

        forecast = model.predict(prophet_val)
        y_true = prophet_val['y'].values
        y_pred = forecast['yhat'].values

        val_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        val_mae = mean_absolute_error(y_true, y_pred)
        val_r2 = r2_score(y_true, y_pred)

        logger.info(f"Prophet Val RMSE: {val_rmse:.4f}, MAE: {val_mae:.4f}, R2: {val_r2:.4f}")
        return model





