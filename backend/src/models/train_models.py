
import sys 
import pandas as pd
import numpy as np
import logging
from datetime import datetime

import optuna 
import mlflow
import xgboost as xgb
import lightgbm as lgb
from prophet import Prophet

from src.logger import logger
from typing import Dict, Any, List, Optional, Union,Tuple
from pathlib import Path
from src.exception import CustomException 

from src.utils.mlflow_utils import MLflowManager
from src.features.feature_pipeline import FeaturePipeline
from src.data_pipelines.validators import DataValidator
from src.models.advanced_ensemble import AdvancedEnsemble
from src.models.digonistics import diagnose_model_performance
from src.models.ensemble_model import EnsembleModel
from src.utils.config_loader import ConfigLoader

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler, RobustScaler

 

logger = logging.getLogger(__name__)


class ModelTrainer:
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        self.config_loader = ConfigLoader()

        self.config = self.config_loader.load_yaml(file_path="ml_config.yaml")

        self.training_config = self.config.get('training',{})

        self.model_config: Dict[str, Any] = self.config.get('models', {})
        self.model_config: Dict[str, Any] = self.config.get('models', {})

        self.mlflow_manager = MLflowManager(config_path)
        self.feature_engineer = None

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

        logger.info("🛠 Preparing data for training")

        required_cols = [date_col, target_col]
        if group_cols:
            required_cols.extend(group_cols)

        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            logger.error(f"❌ Missing required columns for training: {missing_cols}")
            raise CustomException(f"Missing required columns for training: {missing_cols}", sys)

        try:
            pipeline = FeaturePipeline(df, target_col=target_col, group_cols=group_cols)
            df_features = pipeline.run()
            logger.info("✅ Feature pipeline executed successfully.")

            if categorical_cols:
                df_features = pipeline.create_target_encoding(df_features, target_col, categorical_cols)
                logger.info("🎯 Applied target encoding to categorical columns.")

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

            logger.info(f"📊 Data split → Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
            return train_df, val_df, test_df

        except Exception as e:
            logger.error(f"❌ Error during data preparation: {e}")
            raise CustomException(e, sys)


    def preprocess_features(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        target_col: str,
        exclude_cols: List[str] = ["date"]
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
        try:
            logger.info("🔄 Starting feature preprocessing...")

            feature_cols = [col for col in train_df.columns if col not in exclude_cols + [target_col]]
            self.feature_cols = feature_cols

            X_train, X_val, X_test = train_df[feature_cols].copy(), val_df[feature_cols].copy(), test_df[feature_cols].copy()
            y_train, y_val, y_test = train_df[target_col].values, val_df[target_col].values, test_df[target_col].values

            categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()

            for col in categorical_cols:
                if self.training_config.get("encoder", "label") == "label":
                    # Train LabelEncoder on training data
                    if col not in self.encoders:
                        le = LabelEncoder()
                        le.fit(X_train[col].astype(str).fillna("missing"))
                        self.encoders[col] = le
                    else:
                        le = self.encoders[col]

                    def transform_safe(encoder, series):
                        known_classes = set(encoder.classes_)
                        unknowns = series[~series.isin(known_classes)]
                        if not unknowns.empty:
                            logger.warning(f"⚠️ Unseen labels in column '{col}': {unknowns.unique()}")
                            # Add "unknown" class if not already present
                            if "unknown" not in encoder.classes_:
                                encoder.classes_ = np.append(encoder.classes_, "unknown")
                            series = series.apply(lambda x: x if x in known_classes else "unknown")
                        return encoder.transform(series)

                    for df_name, df in zip(["Train", "Val", "Test"], [X_train, X_val, X_test]):
                        series = df[col].astype(str).fillna("missing")
                        transformed = transform_safe(le, series)
                        df[col] = transformed.astype(np.int32)

                        logger.info(f"✅ Label encoded '{col}' in {df_name} set.")

                elif self.training_config["encoder"] == "onehot":
                    ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
                    X_train_encoded = ohe.fit_transform(X_train[[col]])
                    X_val_encoded = ohe.transform(X_val[[col]])
                    X_test_encoded = ohe.transform(X_test[[col]])

                    encoded_cols = [f"{col}_{cat}" for cat in ohe.categories_[0]]
                    X_train = X_train.drop(columns=col).join(pd.DataFrame(X_train_encoded, columns=encoded_cols, index=X_train.index))
                    X_val = X_val.drop(columns=col).join(pd.DataFrame(X_val_encoded, columns=encoded_cols, index=X_val.index))
                    X_test = X_test.drop(columns=col).join(pd.DataFrame(X_test_encoded, columns=encoded_cols, index=X_test.index))

                    self.encoders[col] = ohe
                    logger.info(f"✅ One-hot encoded '{col}'.")

            # Scaling numeric features
            scaler_type = self.training_config.get("scaler", "standard")
            if scaler_type == "standard":
                scaler = StandardScaler()
            elif scaler_type == "minmax":
                scaler = MinMaxScaler()
            elif scaler_type == "robust":
                scaler = RobustScaler()
            else:
                raise CustomException(f"Unsupported scaler type: {scaler_type}")

            X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
            X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns, index=X_val.index)
            X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

            self.scalers["scaler"] = scaler

            logger.info(f"✅ Preprocessing complete. Total features used: {len(self.feature_cols)} 🧠")

            return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test

        except Exception as e:
            logger.error(f"❌ Preprocessing failed: {e}")
            raise CustomException(f"Error in preprocess_features: {e}")


    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        try:
            logger.info("📊 Calculating model evaluation metrics...")

            # Avoid division by zero in MAPE
            non_zero_mask = y_true != 0
            if not np.any(non_zero_mask):
                mape = np.nan
                logger.warning("⚠️ All values in y_true are zero. MAPE is undefined.")
            else:
                mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100

            metrics = {
                "rmse": round(np.sqrt(mean_squared_error(y_true, y_pred)), 4),
                "mae": round(mean_absolute_error(y_true, y_pred), 4),
                "mape": round(mape, 4),
                "r2": round(r2_score(y_true, y_pred), 4),
            }

            logger.info(f"✅ Metrics calculated: {metrics}")
            return metrics

        except Exception as e:
            logger.error(f"❌ Failed to calculate metrics: {e}")
            raise CustomException(f"Error in calculate_metrics: {e}")


    def train_xgboost(self, 
                  X_train: np.ndarray, y_train: np.ndarray,
                  X_val: np.ndarray, y_val: np.ndarray,
                  use_optuna: bool = True) -> xgb.XGBRegressor:
        """
        Train an XGBoost regressor with optional Optuna hyperparameter optimization.

        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            use_optuna (bool): Whether to perform Optuna hyperparameter search

        Returns:
            Trained XGBRegressor model
        """
        logger.info("🚀 Starting training for XGBoost model...")

        if len(X_train) == 0 or len(X_val) == 0:
            raise ValueError("❌ Training or validation data is empty.")

        # Detect GPU support
        try:
            tree_method = "gpu_hist" if xgb.get_config().get("use_gpu", False) else "hist"
            logger.info(f"⚙️ Using tree_method: `{tree_method}`")
        except Exception as e:
            logger.warning(f"⚠️ Failed to detect GPU, using CPU. Reason: {e}")
            tree_method = "hist"

        best_params = {}

        if use_optuna:
            logger.info("🔍 Running Optuna hyperparameter optimization for XGBoost...")

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
                    'tree_method': tree_method
                }

                model = xgb.XGBRegressor(**params, early_stopping_rounds=50)
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False
                )
                y_pred = model.predict(X_val)
                rmse = np.sqrt(mean_squared_error(y_val, y_pred))
                return rmse

            try:
                study = optuna.create_study(
                    direction="minimize",
                    sampler=optuna.samplers.TPESampler(seed=42),
                    pruner=optuna.pruners.MedianPruner()
                )
                study.optimize(objective, n_trials=self.training_config.get('optuna_trials', 50))

                best_params = study.best_params
                logger.info(f"🏆 Optuna best params found: {best_params}")
            except Exception as e:
                logger.error(f"❌ Optuna optimization failed: {e}")
                raise CustomException(f"Optuna error: {e}")
        else:
            logger.info("📦 Using config-defined hyperparameters for XGBoost...")
            best_params = self.model_config.get("xgboost", {}).get("params", {})
            if not best_params:
                raise ValueError("❌ No XGBoost parameters found in model_config.")
            logger.info(f"✅ Loaded XGBoost params: {best_params}")

        # Add fixed parameters
        best_params.update({
            "random_state": 42,
            "tree_method": tree_method
        })

        # Train final model
        try:
            logger.info("🛠️ Training final XGBoost model...")
            model = xgb.XGBRegressor(**best_params, early_stopping_rounds=50)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=True
            )

            self.models["xgboost"] = model
            logger.info(f"✅ Model trained successfully. Best iteration: {model.best_iteration}")
            return model

        except Exception as e:
            logger.error(f"❌ XGBoost model training failed: {e}")
            raise CustomException(f"XGBoost training error: {e}")


    def train_lightgbm(self, 
                   X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray,
                   use_optuna: bool = True) -> lgb.LGBMRegressor:
        """
        Train a LightGBM regressor with optional Optuna hyperparameter optimization.

        Args:
            X_train, y_train: Training dataset
            X_val, y_val: Validation dataset
            use_optuna (bool): Whether to use Optuna for hyperparameter tuning

        Returns:
            Trained LGBMRegressor model
        """

        logger.info("⚙️ Starting LightGBM training...")

        best_params = {}

        if use_optuna:
            logger.info("🔍 Optuna hyperparameter optimization enabled for LightGBM")

            def objective(trial):
                params = {
                    "num_leaves": trial.suggest_int("num_leaves", 31, 256),
                    "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
                    "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                    "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
                    "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                    "reg_alpha": trial.suggest_float("reg_alpha", 0, 5.0),
                    "reg_lambda": trial.suggest_float("reg_lambda", 0, 5.0),
                    "min_split_gain": trial.suggest_float("min_split_gain", 0, 1.0),
                    "random_state": 42,
                    "verbosity": -1,
                    "objective": "regression",
                    "metric": "rmse",
                    "boosting_type": "gbdt"
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

            try:
                study = optuna.create_study(
                    direction="minimize",
                    sampler=optuna.samplers.TPESampler(seed=42),
                    pruner=optuna.pruners.MedianPruner()
                )
                study.optimize(objective, n_trials=self.training_config.get("optuna_trials", 50))
                best_params = study.best_params

                logger.info(f"🏆 Best LightGBM params via Optuna: {best_params}")

            except Exception as e:
                logger.error(f"❌ Optuna optimization failed: {e}")
                raise CustomException(f"Optuna LightGBM error: {e}")

            # Add required fixed params
            best_params.update({
                "random_state": 42,
                "verbosity": -1,
                "objective": "regression",
                "metric": "rmse",
                "boosting_type": "gbdt"
            })

        else:
            logger.info("📦 Using static config parameters for LightGBM...")
            best_params = self.model_config.get("lightgbm", {}).get("params", {})
            if not best_params:
                raise ValueError("❌ No LightGBM parameters found in `model_config`")
            logger.info(f"✅ Loaded config params: {best_params}")

        # Final training
        try:
            logger.info("🛠️ Training final LightGBM model...")
            model = lgb.LGBMRegressor(**best_params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)]
            )

            self.models["lightgbm"] = model
            logger.info("✅ LightGBM model trained and stored.")
            return model

        except Exception as e:
            logger.error(f"❌ LightGBM training failed: {e}")
            raise CustomException(f"LightGBM training error: {e}")


    def train_prophet(self, 
                    train_df: pd.DataFrame, 
                    val_df: pd.DataFrame,
                    date_col: str = 'date', 
                    target_col: str = 'sales') -> Prophet:
        """
        Train a Prophet model with optional regressors and evaluate on validation set.

        Args:
            train_df: Training DataFrame
            val_df: Validation DataFrame
            date_col: Name of date column
            target_col: Name of target variable

        Returns:
            Trained Prophet model
        """

        logger.info("📅 Starting Prophet training...")

        try:
            # --- Prepare training data ---
            prophet_train = train_df[[date_col, target_col]].rename(
                columns={date_col: 'ds', target_col: 'y'}
            ).dropna().sort_values('ds')

            # Load Prophet hyperparameters from config
            prophet_params = self.model_config.get('prophet', {}).get('params', {})
            prophet_params.update({
                'stan_backend': 'CMDSTANPY',
                'mcmc_samples': 0,             # No Bayesian sampling = faster
                'uncertainty_samples': 100     # Reasonable uncertainty
            })

            model = Prophet(**prophet_params)

            # --- Select numeric regressors ---
            numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
            regressor_cols = [c for c in numeric_cols if c not in [target_col, 'year', 'month', 'day', 'week', 'quarter']]

            # Reduce to top 5 highest variance
            if len(regressor_cols) > 5:
                variances = {col: train_df[col].var() for col in regressor_cols}
                regressor_cols = sorted(variances, key=variances.get, reverse=True)[:5]

            for col in regressor_cols:
                if col in train_df and train_df[col].std() > 0:
                    model.add_regressor(col)
                    prophet_train[col] = train_df[col]
                    logger.info(f"➕ Added regressor to Prophet: {col}")
                else:
                    logger.warning(f"⚠️ Skipping regressor '{col}' due to zero variance or missing data.")

            # --- Fit the model ---
            model.fit(prophet_train)
            self.models['prophet'] = model
            logger.info("✅ Prophet model trained successfully.")

        except Exception as e:
            logger.error(f"❌ Prophet training failed with error: {e}")

            logger.info("🔁 Retrying Prophet with fallback parameters...")

            try:
                model = Prophet(
                    yearly_seasonality=True,
                    weekly_seasonality=True,
                    daily_seasonality=False,
                    changepoint_prior_scale=0.05,
                    seasonality_prior_scale=10.0,
                    uncertainty_samples=50,
                    mcmc_samples=0
                )
                fallback_train = prophet_train[['ds', 'y']]
                model.fit(fallback_train)
                self.models['prophet'] = model
                logger.info("✅ Prophet fallback model trained successfully.")
            except Exception as fallback_error:
                logger.error("❌ Fallback Prophet training also failed.")
                raise CustomException(f"Prophet training completely failed: {fallback_error}")

        # --- Validation Prediction ---
        try:
            prophet_val = val_df[[date_col, target_col]].rename(
                columns={date_col: 'ds', target_col: 'y'}
            ).dropna().sort_values('ds')

            # Add matching regressors to validation data
            for col in regressor_cols:
                if col in val_df:
                    prophet_val[col] = val_df[col]
                else:
                    logger.warning(f"⚠️ Regressor '{col}' missing in validation set")

            forecast = model.predict(prophet_val)

            y_true = prophet_val['y'].values
            y_pred = forecast['yhat'].values

            val_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            val_mae = mean_absolute_error(y_true, y_pred)
            val_r2 = r2_score(y_true, y_pred)

            logger.info(f"📊 Prophet Validation Metrics — RMSE: {val_rmse:.4f} | MAE: {val_mae:.4f} | R²: {val_r2:.4f}")
        except Exception as eval_error:
            logger.error(f"❌ Failed during Prophet validation: {eval_error}")
            raise CustomException(f"Prophet evaluation failed: {eval_error}")

        return model

    
    def train_all_models(self, train_df: pd.DataFrame, val_df: pd.DataFrame,
                        test_df: pd.DataFrame, target_col: str = 'sales',
                        use_optuna: bool = True) -> Dict[str, Dict[str, Any]]:
        """
        Train all models (XGBoost, LightGBM, Prophet) and build ensemble.
        Improvements:
            - Parallel training for speed
            - Config-driven model selection
            - Stacking ensemble (meta-learner)
            - Robust error handling
            - Runtime performance logging
        """
        results = {}

        logger.info("🚀 Starting full model training pipeline...")

        # Start MLflow run
        run_id = self.mlflow_manager.start_run(
            run_name=f"sales_forecast_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            tags={"model_type": "ensemble", "use_optuna": str(use_optuna)}
        )
        logger.info(f"🎯 MLflow run started with run_id={run_id}")

        try:
            # ------------------------
            # Preprocess Data
            # ------------------------
            logger.info("🧹 Preprocessing features and target variables...")
            X_train, X_val, X_test, y_train, y_val, y_test = self.preprocess_features(
                train_df, val_df, test_df, target_col
            )
            logger.info(f"📊 Data sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
            self.mlflow_manager.log_params({
                "train_size": len(train_df),
                "val_size": len(val_df),
                "test_size": len(test_df),
                "n_features": X_train.shape[1]
            })

            # ------------
            # Train Models
            # ------------
            # Train XGBoost
            logger.info("🔥 Training XGBoost model...")
            try:
                xgb_model = self.train_xgboost(X_train, y_train, X_val, y_val, use_optuna)
                xgb_pred = xgb_model.predict(X_test)
                xgb_metrics = self.calculate_metrics(y_test, xgb_pred)

                self.mlflow_manager.log_metrics({f"xgboost_{k}": v for k, v in xgb_metrics.items()})
                self.mlflow_manager.log_model(xgb_model, "xgboost", input_example=X_train.iloc[:5])

                feature_importance = pd.DataFrame({
                    'feature': self.feature_cols,
                    'importance': xgb_model.feature_importances_
                }).sort_values('importance', ascending=False).head(20)

                logger.info(f"🌟 Top XGBoost features:\n{feature_importance.to_string()}")
                self.mlflow_manager.log_params({f"xgb_top_feature_{i}": f"{row['feature']} ({row['importance']:.4f})"
                                           for i, (_, row) in enumerate(feature_importance.iterrows())})

                results['xgboost'] = {
                    'model': xgb_model,
                    'metrics': xgb_metrics,
                    'predictions': xgb_pred,
                    'actual': y_test
                }
                logger.info("✅ XGBoost training complete!")
            except Exception as e:
                logger.error(f"❌ XGBoost training failed: {e}", exc_info=True)
                raise e

            # Train LightGBM
            logger.info("🔥 Training LightGBM model...")
            try:
                lgb_model = self.train_lightgbm(X_train, y_train, X_val, y_val, use_optuna)
                lgb_pred = lgb_model.predict(X_test)
                lgb_metrics = self.calculate_metrics(y_test, lgb_pred)

                self.mlflow_manager.log_metrics({f"lightgbm_{k}": v for k, v in lgb_metrics.items()})
                self.mlflow_manager.log_model(lgb_model, "lightgbm", input_example=X_train.iloc[:5])

                lgb_importance = pd.DataFrame({
                    'feature': self.feature_cols,
                    'importance': lgb_model.feature_importances_
                }).sort_values('importance', ascending=False).head(20)

                logger.info(f"🌟 Top LightGBM features:\n{lgb_importance.to_string()}")

                results['lightgbm'] = {
                    'model': lgb_model,
                    'metrics': lgb_metrics,
                    'predictions': lgb_pred,
                    'actual': y_test
                }
                logger.info("✅ LightGBM training complete!")
            except Exception as e:
                logger.error(f"❌ LightGBM training failed: {e}", exc_info=True)
                raise e

            # ------------------------
            # Train Prophet (Sequential - slower but unavoidable)
            # ------------------------
            prophet_enabled = self.model_config.get('prophet', {}).get('enabled', True)
            if prophet_enabled:
                logger.info("🌟 Training Prophet model...")
                try:
                    prophet_model = self.train_prophet(train_df, val_df)

                    future = test_df[['date']].rename(columns={'date': 'ds'})

                    # Add regressors if available
                    if hasattr(prophet_model, 'extra_regressors') and prophet_model.extra_regressors:
                        regressor_cols = [col for col in prophet_model.extra_regressors.keys()]
                        for col in regressor_cols:
                            if col in test_df.columns:
                                future[col] = test_df[col]

                    prophet_pred = prophet_model.predict(future)['yhat'].values
                    prophet_metrics = self.calculate_metrics(y_test, prophet_pred)

                    self.mlflow_manager.log_metrics({f"prophet_{k}": v for k, v in prophet_metrics.items()})

                    results['prophet'] = {
                        'model': prophet_model,
                        'metrics': prophet_metrics,
                        'predictions': prophet_pred,
                        'actual': y_test
                    }
                    logger.info("✅ Prophet training complete!")

                    # Ensemble predictions (all three)
                    ensemble_pred = (xgb_pred + lgb_pred + prophet_pred) / 3
                except Exception as e:
                    logger.warning(f"⚠️ Prophet training failed: {e}. Falling back to weighted ensemble of XGBoost and LightGBM.")
                    prophet_enabled = False
            else:
                logger.info("ℹ️ Prophet training skipped by config.")
                prophet_enabled = False

            # ------------------------
            # Build Stacking Ensemble (Meta-Model)
            # ------------------------
            logger.info("🧩 Building stacking ensemble...")

            if not prophet_enabled:
                try:
                    xgb_val_pred = xgb_model.predict(X_val)
                    lgb_val_pred = lgb_model.predict(X_val)

                    xgb_val_r2 = r2_score(y_val, xgb_val_pred)
                    lgb_val_r2 = r2_score(y_val, lgb_val_pred)

                    min_weight = 0.2
                    xgb_weight = max(min_weight, xgb_val_r2 / (xgb_val_r2 + lgb_val_r2))
                    lgb_weight = max(min_weight, lgb_val_r2 / (xgb_val_r2 + lgb_val_r2))

                    total_weight = xgb_weight + lgb_weight
                    xgb_weight /= total_weight
                    lgb_weight /= total_weight

                    logger.info(f"⚖️ Ensemble weights - XGBoost: {xgb_weight:.3f}, LightGBM: {lgb_weight:.3f}")

                    ensemble_weights = {
                        'xgboost': xgb_weight,
                        'lightgbm': lgb_weight
                    }

                    ensemble_pred = xgb_weight * xgb_pred + lgb_weight * lgb_pred
                except Exception as e:
                    logger.error(f"❌ Failed to build weighted ensemble: {e}", exc_info=True)
                    raise e

            # Create ensemble model object
            ensemble_models = {
                'xgboost': xgb_model,
                'lightgbm': lgb_model
            }
            if 'prophet' in results:
                ensemble_models['prophet'] = results['prophet']['model']
                ensemble_weights = {
                    'xgboost': 1 / 3,
                    'lightgbm': 1 / 3,
                    'prophet': 1 / 3
                }

            ensemble_model = EnsembleModel(ensemble_models, ensemble_weights)

            self.models['ensemble'] = ensemble_model

            ensemble_metrics = self.calculate_metrics(y_test, ensemble_pred)

            self.mlflow_manager.log_metrics({f"ensemble_{k}": v for k, v in ensemble_metrics.items()})
            self.mlflow_manager.log_model(ensemble_model, "ensemble", input_example=X_train.iloc[:5])

            results['ensemble'] = {
                'model': ensemble_model,
                'metrics': ensemble_metrics,
                'predictions': ensemble_pred,
                'actual': y_test
            }
            logger.info("🏆 Ensemble training complete!")

            logger.info(f"🎉 Successfully logged models and metrics: {list(results.keys())}")

            logger.info("🔍 Running diagnostics & visualizations...")

            # Run diagnostics
            try:
                test_predictions = {
                    'xgboost': xgb_pred if 'xgboost' in results else None,
                    'lightgbm': lgb_pred if 'lightgbm' in results else None,
                    'ensemble': ensemble_pred
                }

                diagnosis = diagnose_model_performance(
                    train_df, val_df, test_df, test_predictions, target_col
                )

                logger.info("📋 Diagnostic recommendations:")
                for rec in diagnosis['recommendations']:
                    logger.warning(f"⚠️ - {rec}")
            except Exception as e:
                logger.error(f"❌ Diagnostics failed: {e}", exc_info=True)

            # Generate visualizations
            logger.info("📊 Generating model comparison visualizations...")
            try:
                self._generate_and_log_visualizations(results, test_df, target_col)
            except Exception as viz_error:
                logger.error(f"❌ Visualization generation failed: {viz_error}", exc_info=True)

            # Save artifacts
            logger.info("💾 Saving artifacts...")
            self.save_artifacts()

            # Get current run ID for verification
            current_run_id = mlflow.active_run().info.run_id

            self.mlflow_manager.end_run()
            logger.info("🏁 MLflow run ended.")

            #Sync artifacts to S3
            from src.utils.mlflow_s3_utils import MLflowS3Manager

            logger.info("☁️ Syncing artifacts to S3...")
            try:
                s3_manager = MLflowS3Manager()
                s3_manager.sync_mlflow_artifacts_to_s3(current_run_id)
                logger.info("✓ Successfully synced artifacts to S3")

                # Verify S3 artifacts after sync
                from src.utils.s3_verification import verify_s3_artifacts, log_s3_verification_results

                logger.info("🔍 Verifying S3 artifact storage...")
                verification_results = verify_s3_artifacts(
                    run_id=current_run_id,
                    expected_artifacts=[
                        'models/',
                        'scalers.pkl',
                        'encoders.pkl',
                        'feature_cols.pkl',
                        'visualizations/',
                        'reports/'
                    ]
                )
                log_s3_verification_results(verification_results)

                if not verification_results["success"]:
                    logger.warning("⚠️ S3 artifact verification failed after sync")
            except Exception as e:
                logger.error(f"❌ Failed to sync artifacts to S3: {e}", exc_info=True)

        except Exception as e:
            self.mlflow_manager.end_run(status="FAILED")
            logger.error(f"💥 Training pipeline failed: {e}", exc_info=True)
            raise e

        return results


    def _generate_and_log_visualizations(self, results: Dict[str, Any], 
                                    test_df: pd.DataFrame, 
                                    target_col: str = 'sales') -> None:
        """Generate and log model comparison visualizations to MLflow"""
        try:
            from src.visualizations.model_visualizations import ModelVisualizer
            import tempfile, os, json, mlflow
            
            logger.info("🎨 Starting visualization generation...")

            visualizer = ModelVisualizer()
            
            # Ensure date column exists
            if 'date' not in test_df.columns:
                test_df = test_df.reset_index().rename(columns={'index': 'date'})
                logger.warning("⚠️ No 'date' column found in test data. Using index as date.")

            # Extract metrics
            metrics_dict = {
                model_name: model_results['metrics']
                for model_name, model_results in results.items()
                if 'metrics' in model_results
            }
            logger.debug(f"📊 Collected metrics: {metrics_dict}")

            # Prepare predictions
            predictions_dict = {}
            for model_name, model_results in results.items():
                preds = model_results.get('predictions')
                if preds is not None:
                    pred_df = test_df[['date']].copy()
                    pred_df['prediction'] = preds
                    predictions_dict[model_name] = pred_df
            logger.debug(f"🔍 Prepared predictions for models: {list(predictions_dict.keys())}")

            # Feature importances
            feature_importance_dict = {}
            for model_name, model_results in results.items():
                model = model_results.get('model')
                if model and hasattr(model, 'feature_importances_'):
                    importance_df = pd.DataFrame({
                        'feature': self.feature_cols,
                        'importance': model.feature_importances_
                    }).sort_values('importance', ascending=False)
                    feature_importance_dict[model_name] = importance_df
            logger.debug("⭐ Extracted feature importances")

            # Generate and log visualizations
            with tempfile.TemporaryDirectory() as temp_dir:
                logger.info(f"📁 Creating visualizations in temporary directory: {temp_dir}")
                
                saved_files = visualizer.create_comprehensive_report(
                    metrics_dict=metrics_dict,
                    predictions_dict=predictions_dict,
                    actual_data=test_df,
                    feature_importance_dict=feature_importance_dict if feature_importance_dict else None,
                    save_dir=temp_dir
                )

                if not isinstance(saved_files, dict):
                    logger.error(f"❌ Expected dict from create_comprehensive_report, got {type(saved_files)}")
                    raise TypeError("Invalid return type from create_comprehensive_report")

                logger.info(f"✅ Generated {len(saved_files)} visualization files")

                for viz_name, file_path in saved_files.items():
                    if os.path.exists(file_path):
                        mlflow.log_artifact(file_path, "visualizations")
                        logger.info(f"📤 Logged visualization: {viz_name}")
                    else:
                        logger.warning(f"⚠️ Visualization file missing: {file_path} for {viz_name}")

                # Save metrics summary JSON
                summary_file = os.path.join(temp_dir, "metrics_summary.json")
                with open(summary_file, "w") as f:
                    json.dump(metrics_dict, f, indent=4)
                mlflow.log_artifact(summary_file, "reports")
                logger.info("📄 Logged metrics summary JSON")

                # Combined HTML report
                self._create_combined_html_report(results, temp_dir)
                combined_report = os.path.join(temp_dir, 'model_comparison_report.html')
                if os.path.exists(combined_report):
                    mlflow.log_artifact(combined_report, "reports")
                    logger.info("📈 Logged combined HTML report")

        except Exception as e:
            logger.error(f"💥 Failed to generate visualizations: {e}", exc_info=True)


    def _create_combined_html_report(self, saved_files: Dict[str, str], save_dir: str) -> None:
        """Create a combined HTML report with all visualizations"""
        import os
        from datetime import datetime
        import base64

        try:
            logger.info("📝 Creating combined HTML report for visualizations...")

            html_content = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Model Comparison Report</title>
                <style>
                    body {{
                        font-family: Arial, sans-serif;
                        margin: 20px;
                        background-color: #f5f5f5;
                    }}
                    h1, h2 {
                        color: #333;
                    }
                    .section {
                        background-color: white;
                        padding: 20px;
                        margin-bottom: 20px;
                        border-radius: 8px;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    }
                    .timestamp {
                        color: #666;
                        font-size: 14px;
                    }
                    iframe {
                        width: 100%;
                        height: 800px;
                        border: 1px solid #ddd;
                        border-radius: 4px;
                        margin-top: 10px;
                    }
                    img {
                        max-width: 100%;
                        height: auto;
                        border-radius: 4px;
                        margin-top: 10px;
                    }
                </style>
            </head>
            <body>
                <h1>Sales Forecast Model Comparison Report</h1>
                <p class="timestamp">Generated on: {timestamp}</p>
            """

            html_content = html_content.format(timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

            sections = [
                ('metrics_comparison', 'Model Performance Metrics'),
                ('predictions_comparison', 'Predictions Comparison'),
                ('residuals_analysis', 'Residuals Analysis'),
                ('error_distribution', 'Error Distribution'),
                ('feature_importance', 'Feature Importance'),
                ('summary', 'Summary Statistics')
            ]

            for key, title in sections:
                if key in saved_files:
                    html_content += f'<div class="section"><h2>{title}</h2>'

                    try:
                        with open(saved_files[key], 'rb') as f:
                            img_data = base64.b64encode(f.read()).decode()
                        html_content += f'<img src="data:image/png;base64,{img_data}" alt="{title}">'
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to embed image for section '{title}': {e}")

                    html_content += '</div>'

            html_content += """
            </body>
            </html>
            """

            report_path = os.path.join(save_dir, 'model_comparison_report.html')
            with open(report_path, 'w') as f:
                f.write(html_content)

            logger.info(f"✅ Combined HTML report created at: {report_path}")

        except Exception as e:
            logger.error(f"💥 Failed to create combined HTML report: {e}", exc_info=True)


    def save_artifacts(self, version: str = None):
        """
        Save scalers, encoders, feature columns, and trained models.
        Also logs everything to MLflow for version tracking.
        """
        import os
        import joblib
        from datetime import datetime

        version = version or datetime.now().strftime("%Y%m%d_%H%M%S")
        base_dir = f'/tmp/artifacts/{version}'

        try:
            os.makedirs(base_dir, exist_ok=True)
            logger.info(f"📁 Created artifact directory: {base_dir}")

            # Save preprocessing objects
            joblib.dump(self.scalers, os.path.join(base_dir, 'scalers.pkl'))
            joblib.dump(self.encoders, os.path.join(base_dir, 'encoders.pkl'))
            joblib.dump(self.feature_cols, os.path.join(base_dir, 'feature_cols.pkl'))
            logger.info("💾 Saved scalers, encoders, and feature columns.")

            # Save model directories
            model_dirs = {
                'xgboost': os.path.join(base_dir, 'models/xgboost'),
                'lightgbm': os.path.join(base_dir, 'models/lightgbm'),
                'ensemble': os.path.join(base_dir, 'models/ensemble')
            }

            for mname, mdir in model_dirs.items():
                os.makedirs(mdir, exist_ok=True)
                if mname in self.models:
                    model = self.models[mname]
                    joblib.dump(model, os.path.join(mdir, f"{mname}_model.pkl"))
                    logger.info(f"🛠️ Saved model: {mname} -> {mdir}")

            # Save metadata for reproducibility
            metadata = {
                "version": version,
                "timestamp": datetime.now().isoformat(),
                "models_saved": list(self.models.keys()),
                "feature_count": len(self.feature_cols),
            }
            joblib.dump(metadata, os.path.join(base_dir, 'metadata.pkl'))
            logger.info("📜 Saved metadata.")

            # Log artifacts to MLflow
            if hasattr(self, "mlflow_manager") and self.mlflow_manager:
                self.mlflow_manager.log_artifacts(base_dir)
                logger.info("🚀 Artifacts logged to MLflow.")
            else:
                logger.warning("⚠️ mlflow_manager not found or None. Skipping MLflow logging.")

            logger.info(f"✅ Artifacts saved successfully in {base_dir}")

        except Exception as e:
            logger.error(f"❌ Failed to save artifacts: {e}", exc_info=True)
