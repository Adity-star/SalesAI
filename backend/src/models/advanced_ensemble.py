import numpy as np
import pandas as pd
from scipy.optimize import minimize


from typing import Dict, List, Tuple, Any
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

import logging
from src.logger import logger

logger = logging.getLogger(__name__)


class AdvancedEnsemble:
    """Advanced ensemble techniques for better model performance"""

    def __init__(self, save_dir: str = "./models/ensemble/"):
        self.meta_model = None
        self.base_models = {}
        self.model_weights = {}
        self.save_dir = save_dir

    def create_stacking_ensemble(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        base_predictions: Dict[str, Dict[str, np.ndarray]],
        meta_model_type: str = "ridge",
         ) -> np.ndarray:
        """
        Create a stacking ensemble using base model predictions.
        
        Parameters
        ----------
        base_predictions : dict
            Dictionary containing {model_name: {"train": np.ndarray, "val": np.ndarray}}
            where "train" are OOF predictions and "val" are validation predictions.
        """
        logger.info("Creating stacking ensemble")

        # Prepare meta features
        train_meta_features = [base_predictions[m]["train"] for m in base_predictions]
        val_meta_features = [base_predictions[m]["val"] for m in base_predictions]

        X_meta_train = np.column_stack(train_meta_features)
        X_meta_val = np.column_stack(val_meta_features)

        # Add diversity features (pairwise differences between base models)
        if len(base_predictions) > 1:
            model_names = list(base_predictions.keys())
            for i in range(len(model_names)):
                for j in range(i + 1, len(model_names)):
                    diff_train = (
                        base_predictions[model_names[i]]["train"]
                        - base_predictions[model_names[j]]["train"]
                    )
                    diff_val = (
                        base_predictions[model_names[i]]["val"]
                        - base_predictions[model_names[j]]["val"]
                    )
                    X_meta_train = np.column_stack([X_meta_train, diff_train])
                    X_meta_val = np.column_stack([X_meta_val, diff_val])

        # Select meta-model
        if meta_model_type == "ridge":
            self.meta_model = Ridge(alpha=1.0, random_state=42)
        elif meta_model_type == "lasso":
            self.meta_model = Lasso(alpha=0.01, random_state=42)
        elif meta_model_type == "elastic":
            self.meta_model = ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=42)
        elif meta_model_type == "rf":
            self.meta_model = RandomForestRegressor(
                n_estimators=200, max_depth=7, random_state=42, n_jobs=-1
            )
        else:
            logger.warning(f"Unknown meta_model_type={meta_model_type}, defaulting to Ridge")
            self.meta_model = Ridge(alpha=1.0, random_state=42)

        # Train meta-model
        self.meta_model.fit(X_meta_train, y_train)
        logger.info(f"Meta-model ({meta_model_type}) trained successfully")

        # Validation predictions
        meta_predictions = self.meta_model.predict(X_meta_val)

        # Evaluate ensemble
        val_rmse = np.sqrt(mean_squared_error(y_val, meta_predictions))
        logger.info(f"Stacking ensemble validation RMSE: {val_rmse:.4f}")

        return meta_predictions
    
     # =========================
    # Optimal Weighted Blending
    # =========================
    def create_blended_ensemble(
        self,
        predictions: Dict[str, np.ndarray],
        y_true: np.ndarray,
        optimization_metric: str = "rmse",
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Create optimally weighted blend of predictions
        """
        logger.info(f"Creating blended ensemble optimized for {optimization_metric}")

        def objective(weights):
            # Ensure weights sum to 1
            weights = np.clip(weights, 0, 1)
            weights = weights / np.sum(weights)

            # Weighted prediction
            blended = np.zeros_like(y_true, dtype=float)
            for i, (model_name, pred) in enumerate(predictions.items()):
                blended += weights[i] * pred

            # Calculate chosen metric
            if optimization_metric == "rmse":
                return np.sqrt(np.mean((blended - y_true) ** 2))
            elif optimization_metric == "mae":
                return np.mean(np.abs(blended - y_true))
            else:  # r2
                return -1 * self._r2_score(y_true, blended)

        # Equal weights initialization
        n_models = len(predictions)
        initial_weights = np.ones(n_models) / n_models
        bounds = [(0, 1) for _ in range(n_models)]

        # Optimize weights
        result = minimize(objective, initial_weights, bounds=bounds, method="SLSQP")
        optimal_weights = result.x / np.sum(result.x)

        # Final blended predictions
        blended_pred = np.zeros_like(y_true, dtype=float)
        weight_dict = {}
        for i, (model_name, pred) in enumerate(predictions.items()):
            blended_pred += optimal_weights[i] * pred
            weight_dict[model_name] = float(optimal_weights[i])

        # Log results
        rmse = np.sqrt(np.mean((blended_pred - y_true) ** 2))
        mae = np.mean(np.abs(blended_pred - y_true))
        r2 = self._r2_score(y_true, blended_pred)

        logger.info(f"Blended Ensemble RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")
        logger.info(f"Optimal blend weights: {weight_dict}")

        return blended_pred, weight_dict
    
     # =========================
    # Dynamic Ensemble
    # =========================
    def create_dynamic_ensemble(
        self,
        base_predictions: Dict[str, Dict[str, np.ndarray]],
        y_val: np.ndarray,
        window_size: int = 30,
    ) -> np.ndarray:
        """
        Create ensemble with dynamic weights that change over time
        """
        logger.info(f"Creating dynamic ensemble with window_size={window_size}")

        val_preds = {name: preds["val"] for name, preds in base_predictions.items()}
        n_samples = len(y_val)
        dynamic_predictions = np.zeros(n_samples)

        for i in range(n_samples):
            # Sliding window
            start_idx = max(0, i - window_size)
            end_idx = min(n_samples, i + window_size)

            if end_idx - start_idx < 10:
                # Not enough data, equal weights
                for pred in val_preds.values():
                    dynamic_predictions[i] += pred[i] / len(val_preds)
            else:
                # Local error-based weights
                local_weights = {}
                total_weight = 0

                for name, pred in val_preds.items():
                    local_error = np.mean(
                        np.abs(pred[start_idx:end_idx] - y_val[start_idx:end_idx])
                    )
                    weight = 1.0 / (local_error + 1e-6)
                    local_weights[name] = weight
                    total_weight += weight

                # Normalize weights
                for name, pred in val_preds.items():
                    dynamic_predictions[i] += (
                        local_weights[name] / total_weight
                    ) * pred[i]

        rmse = np.sqrt(np.mean((dynamic_predictions - y_val) ** 2))
        mae = np.mean(np.abs(dynamic_predictions - y_val))
        r2 = self._r2_score(y_val, dynamic_predictions)

        logger.info(
            f"Dynamic Ensemble RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}"
        )
        return dynamic_predictions

    # =========================
    # Helper metric
    # =========================
    def _r2_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate R-squared score"""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot)
