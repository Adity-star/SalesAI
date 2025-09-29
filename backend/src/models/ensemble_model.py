"""
Ensemble Model for combining predictions from multiple models
"""

import numpy as np
import joblib
from typing import Dict, Any
from src.logger import logger


class EnsembleModel:
    """Ensemble model that combines predictions from multiple models"""

    def __init__(self, models: Dict[str, Any], weights: Dict[str, float] = None):
        """
        Initialize ensemble model

        Args:
            models: Dictionary of model_name -> model object (must have .predict method)
            weights: Dictionary of model_name -> weight (if None, uses equal weights)
        """
        self.models = models

        if not models:
            logger.error("❌ No models provided to EnsembleModel")
            raise ValueError("At least one model is required")

        if weights is None:
            n_models = len(models)
            self.weights = {name: 1.0 / n_models for name in models.keys()}
            logger.info("⚖️ No weights provided. Using equal weights for all models.")
        else:
            total_weight = sum(weights.values())
            if total_weight == 0:
                logger.error("❌ Sum of weights is zero")
                raise ValueError("Weights must sum to a positive value")

            self.weights = {name: w / total_weight for name, w in weights.items()}
            logger.info(f"⚖️ Custom weights provided. Normalized weights: {self.weights}")

        # Ensure weights exist for all models
        for model_name in self.models.keys():
            if model_name not in self.weights:
                self.weights[model_name] = 0.0
                logger.warning(f"⚠️ No weight provided for '{model_name}'. Defaulting to 0.0")

        logger.info(f"✅ Initialized EnsembleModel with models: {list(self.models.keys())}")

    def predict(self, X):
        """Make ensemble predictions"""
        predictions = []
        weights = []

        for model_name, model in self.models.items():
            weight = self.weights.get(model_name, 0.0)
            if weight > 0:
                try:
                    pred = model.predict(X)
                    predictions.append(pred)
                    weights.append(weight)
                    logger.debug(f"🔍 Prediction collected from '{model_name}'")
                except Exception as e:
                    logger.error(f"❌ Error predicting with '{model_name}': {e}")

        if not predictions:
            logger.error("❌ No models available for prediction")
            raise ValueError("No models available for prediction")

        predictions = np.array(predictions)
        weights = np.array(weights)

        # Normalize weights again to be safe
        weights = weights / weights.sum()

        ensemble_pred = np.average(predictions, axis=0, weights=weights)

        logger.info("✅ Ensemble prediction completed successfully")
        return ensemble_pred

    def get_params(self, deep=True):
        """Get parameters for sklearn compatibility"""
        return {
            "models": self.models,
            "weights": self.weights,
        }

    def set_params(self, **params):
        """Set parameters for sklearn compatibility"""
        for key, value in params.items():
            setattr(self, key, value)
            logger.info(f"⚙️ Set parameter '{key}' = {value}")
        return self

    def save(self, filepath: str):
        """Save ensemble model to file"""
        try:
            joblib.dump(self, filepath)
            logger.info(f"💾 Ensemble model saved at '{filepath}'")
        except Exception as e:
            logger.error(f"❌ Failed to save ensemble model: {e}")

    @classmethod
    def load(cls, filepath: str):
        """Load ensemble model from file"""
        try:
            model = joblib.load(filepath)
            logger.info(f"📂 Ensemble model loaded from '{filepath}'")
            return model
        except Exception as e:
            logger.error(f"❌ Failed to load ensemble model: {e}")
            raise

    def __repr__(self):
        model_info = [f"{name}: {weight:.3f}" for name, weight in self.weights.items()]
        return f"EnsembleModel({', '.join(model_info)})"
