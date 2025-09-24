import lightgbm as lgb
from src.models.metrics import rmse
import numpy as np
import logging

logger = logging.getLogger(__name__)

def train_baseline_model(X_train, y_train, X_valid, y_valid, params=None, early_stopping_rounds=50, verbose_eval=10):
    """
    Train a LightGBM baseline model with early stopping.

    Args:
        X_train (pd.DataFrame or np.ndarray): Training features.
        y_train (pd.Series or np.ndarray): Training target.
        X_valid (pd.DataFrame or np.ndarray): Validation features.
        y_valid (pd.Series or np.ndarray): Validation target.
        params (dict, optional): LightGBM parameters.
        early_stopping_rounds (int): Early stopping rounds.
        verbose_eval (int): Frequency of logging during training.

    Returns:
        model: Trained LightGBM model.
        preds: Predictions on validation set.
        metric: RMSE on validation set.
        feature_importance (pd.DataFrame): Optional feature importance dataframe.
    """
    if params is None:
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'verbose': -1,
            'seed': 42
        }

    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)

    logger.info("Starting training LightGBM baseline model...")

    model = lgb.train(
        params,
        train_data,
        valid_sets=[train_data, valid_data],
        valid_names=['train', 'valid'],
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=verbose_eval
    )

    preds = model.predict(X_valid, num_iteration=model.best_iteration)
    metric = rmse(y_valid, preds)
    logger.info(f"Validation RMSE: {metric:.4f}")

    # Optional: Extract feature importance
    importance_df = None
    try:
        importance_df = (
            pd.DataFrame({
                'feature': model.feature_name(),
                'importance': model.feature_importance(importance_type='gain')
            }).sort_values(by='importance', ascending=False)
        )
        logger.info("Extracted feature importance")
    except Exception as e:
        logger.warning(f"Could not extract feature importance: {e}")

    return model, preds, metric, importance_df
