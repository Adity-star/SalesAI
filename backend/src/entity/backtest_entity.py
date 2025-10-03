
from typing import List, Optional, Tuple
from dataclasses import dataclass, field
import pandas as pd 
import numpy as np

@dataclass
class BacktestConfig:
    """Configuration for backtesting parameters."""
    # Time periods
    train_start_date: str = "2013-01-01"
    validation_start_date: str = "2013-03-01" 
    test_start_date: str = "2015-06-01"
    test_end_date: str = "2015-09-01"
    
    # Rolling window settings
    initial_train_days: int = 90  # 2 years initial training
    step_size_days: int = 20       # Retrain monthly
    min_train_days: int = 30      # Minimum training period
    
    # Forecast horizons
    horizons: List[int] = field(default_factory=lambda: [7, 14, 28])
    
    # Model settings
    models_to_evaluate: List[str] = field(default_factory=lambda: ['lightgbm'])
    hyperparameter_tuning: bool = False
    
    # Evaluation settings
    stratify_by: List[str] = field(default_factory=lambda: ['cat_id', 'dept_id', 'store_id', 'state_id'])
    save_predictions: bool = True
    memory_efficient: bool = True


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""
    rmse: float = 0.0
    mae: float = 0.0 
    mase: float = 0.0
    smape: float = 0.0
    mape: float = 0.0
    mean_prediction: float = 0.0
    mean_actual: float = 0.0
    samples: int = 0
    zero_predictions_pct: float = 0.0


@dataclass
class BacktestResults:
    """Container for backtest results."""
    model_name: str
    horizon: int
    metrics: EvaluationMetrics
    predictions: Optional[pd.DataFrame] = None
    feature_importance: Optional[pd.DataFrame] = None
    training_time: float = 0.0
    prediction_time: float = 0.0


def convert_np_types(obj):
    # Recursively convert numpy types to native Python types
    if isinstance(obj, dict):
        return {k: convert_np_types(v) for k, v in obj.items()}
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, list):
        return [convert_np_types(i) for i in obj]
    else:
        return obj
            
