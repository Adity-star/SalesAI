import lightgbm as lgb
from src.models.metrics import rmse
import numpy as np

def train_baseline_model(X_train, y_train, X_valid, y_valid, params=None):
    """
    Train a LightGBM baseline
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
    model = lgb.train(params, train_data, valid_sets=[valid_data], early_stopping_rounds=50)
    preds = model.predict(X_valid)
    metric = rmse(y_valid, preds)
    return model, preds, metric
