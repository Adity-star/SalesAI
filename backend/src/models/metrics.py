import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

def rmse(y_true, y_pred):
    """Root Mean Squared Error"""
    return mean_squared_error(y_true, y_pred, squared=False)

def mae(y_true, y_pred):
    """Mean Absolute Error"""
    return mean_absolute_error(y_true, y_pred)

def smape(y_true, y_pred):
    """Symmetric Mean Absolute Percentage Error (in %)"""
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_pred - y_true) / denom
    diff[denom == 0] = 0.0  # avoid division by zero
    return np.mean(diff) * 100

def mase(y_true, y_pred, naive_forecast):
    """
    Mean Absolute Scaled Error.
    naive_forecast: array-like baseline predictions, usually lag-1 forecast (previous period values).
    """
    y_true, y_pred, naive_forecast = np.array(y_true), np.array(y_pred), np.array(naive_forecast)
    n = len(y_true)
    # Denominator: mean absolute error of naive forecast on training or historical data
    denom = np.mean(np.abs(y_true[1:] - y_true[:-1]))
    if denom == 0:
        return np.nan
    return np.mean(np.abs(y_true - y_pred)) / denom

def prediction_interval_coverage(y_true, y_lower, y_upper):
    """
    Compute coverage of prediction intervals.

    Args:
        y_true: actual values
        y_lower: lower bound of prediction interval
        y_upper: upper bound of prediction interval

    Returns:
        coverage: fraction of true values within intervals
    """
    y_true = np.array(y_true)
    y_lower = np.array(y_lower)
    y_upper = np.array(y_upper)
    covered = (y_true >= y_lower) & (y_true <= y_upper)
    coverage = np.mean(covered)
    return coverage

def compute_prediction_intervals(preds, residuals, alpha=0.05):
    """
    Compute prediction intervals assuming normal distribution of residuals.

    Args:
        preds: predicted values
        residuals: residuals from training set (y_train - y_train_pred)
        alpha: significance level (default 0.05 for 95% intervals)

    Returns:
        lower_bound, upper_bound: arrays of prediction interval bounds
    """
    import scipy.stats as st
    std_dev = np.std(residuals)
    z = st.norm.ppf(1 - alpha/2)
    lower = preds - z * std_dev
    upper = preds + z * std_dev
    return lower, upper
