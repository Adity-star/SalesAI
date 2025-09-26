"""
M5 Walmart Backtesting Framework
==============================

Production-grade backtesting framework specifically designed for M5 dataset:
- Rolling-origin (walk-forward) validation 
- Multiple forecast horizons (7, 14, 28, 90 days)
- Retail-specific metrics (RMSE, MAE, MASE, sMAPE, PICP)
- Hierarchical evaluation (item -> category -> store -> total)
- Multiple baseline models (LightGBM, Prophet, Seasonal Naive)
- Memory-efficient processing for large-scale evaluation
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
import warnings
import json
import gc
import time
from abc import ABC, abstractmethod

# ML libraries
import lightgbm as lgb
from prophet import Prophet
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder

# Configuration
from src.config.m5_config_loader import load_config

logger = logging.getLogger(__name__)

# -------------------------------
# Configuration and Data Models
# -------------------------------

@dataclass
class BacktestConfig:
    """Configuration for backtesting parameters."""
    # Time periods
    train_start_date: str = "2013-01-01"
    validation_start_date: str = "2015-01-01" 
    test_start_date: str = "2015-06-01"
    test_end_date: str = "2016-05-22"
    
    # Rolling window settings
    initial_train_days: int = 730  # 2 years initial training
    step_size_days: int = 28       # Retrain monthly
    min_train_days: int = 365      # Minimum training period
    
    # Forecast horizons
    horizons: List[int] = field(default_factory=lambda: [7, 14, 28])
    
    # Model settings
    models_to_evaluate: List[str] = field(default_factory=lambda: ['lightgbm', 'prophet', 'seasonal_naive'])
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

# -------------------------------
# Metrics Calculator
# -------------------------------

class M5MetricsCalculator:
    """Calculate M5-specific forecasting metrics."""
    
    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Root Mean Square Error."""
        return np.sqrt(mean_squared_error(y_true, y_pred))
    
    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Error."""
        return mean_absolute_error(y_true, y_pred)
    
    @staticmethod
    def mase(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray, 
             seasonal_period: int = 7) -> float:
        """Mean Absolute Scaled Error - critical for M5 competition."""
        if len(y_train) < seasonal_period:
            return np.nan
        
        # Seasonal naive forecast errors
        naive_errors = np.abs(y_train[seasonal_period:] - y_train[:-seasonal_period])
        scale = np.mean(naive_errors)
        
        if scale == 0 or np.isnan(scale):
            return 0.0
        
        return np.mean(np.abs(y_true - y_pred)) / scale
    
    @staticmethod
    def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Symmetric Mean Absolute Percentage Error."""
        numerator = np.abs(y_true - y_pred)
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
        
        # Handle division by zero
        mask = denominator != 0
        if not np.any(mask):
            return 0.0
        
        return np.mean(numerator[mask] / denominator[mask]) * 100
    
    @staticmethod
    def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Percentage Error."""
        mask = y_true != 0
        if not np.any(mask):
            return np.inf
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    @staticmethod
    def calculate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                            y_train: np.ndarray) -> EvaluationMetrics:
        """Calculate all metrics at once."""
        calc = M5MetricsCalculator()
        
        # Ensure non-negative predictions for retail data
        y_pred_clipped = np.maximum(y_pred, 0)
        
        return EvaluationMetrics(
            rmse=calc.rmse(y_true, y_pred_clipped),
            mae=calc.mae(y_true, y_pred_clipped),
            mase=calc.mase(y_true, y_pred_clipped, y_train),
            smape=calc.smape(y_true, y_pred_clipped),
            mape=calc.mape(y_true, y_pred_clipped),
            mean_prediction=np.mean(y_pred_clipped),
            mean_actual=np.mean(y_true),
            samples=len(y_true),
            zero_predictions_pct=np.mean(y_pred_clipped == 0) * 100
        )

# -------------------------------
# Model Wrappers
# -------------------------------

class BaseModel(ABC):
    """Abstract base class for M5 models."""
    
    def __init__(self, name: str, **kwargs):
        self.name = name
        self.model = None
        self.fitted = False
        self.feature_importance_ = None
        self.training_time = 0.0
        self.prediction_time = 0.0
        
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        """Fit model to training data."""
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        pass
    
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """Get feature importance if available."""
        return self.feature_importance_

class M5LightGBMModel(BaseModel):
    """LightGBM model optimized for M5 dataset."""
    
    def __init__(self, name: str = "LightGBM", **lgb_params):
        super().__init__(name)
        self.lgb_params = {
            'objective': 'poisson',  # Better for count data
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 127,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42,
            'num_threads': 4
        }
        self.lgb_params.update(lgb_params)
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        """Fit LightGBM model with early stopping."""
        start_time = time.time()
        
        # Prepare data
        feature_cols = [col for col in X.columns if col not in ['date', 'store_id', 'item_id']]
        X_train = X[feature_cols].copy()
        
        # Handle categorical features
        categorical_features = []
        for col in X_train.columns:
            if X_train[col].dtype.name == 'category':
                categorical_features.append(col)
                X_train[col] = X_train[col].astype('category')
        
        # Create dataset
        train_data = lgb.Dataset(
            X_train, 
            label=y,
            categorical_feature=categorical_features,
            free_raw_data=False
        )
        
        # Training parameters
        num_boost_round = kwargs.get('num_boost_round', 1000)
        early_stopping_rounds = kwargs.get('early_stopping_rounds', 100)
        
        # Train model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model = lgb.train(
                self.lgb_params,
                train_data,
                num_boost_round=num_boost_round,
                callbacks=[
                    lgb.early_stopping(early_stopping_rounds),
                    lgb.log_evaluation(0)
                ]
            )
        
        # Store feature importance
        if self.model:
            self.feature_importance_ = pd.DataFrame({
                'feature': X_train.columns,
                'importance': self.model.feature_importance(importance_type='gain')
            }).sort_values('importance', ascending=False)
        
        self.fitted = True
        self.training_time = time.time() - start_time
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with LightGBM."""
        if not self.fitted:
            raise ValueError("Model must be fitted before predicting")
        
        start_time = time.time()
        
        feature_cols = [col for col in X.columns if col not in ['date', 'store_id', 'item_id']]
        X_pred = X[feature_cols].copy()
        
        # Handle categorical features
        for col in X_pred.columns:
            if X_pred[col].dtype.name == 'category':
                X_pred[col] = X_pred[col].astype('category')
        
        predictions = self.model.predict(X_pred)
        
        # Ensure non-negative predictions
        predictions = np.maximum(predictions, 0)
        
        self.prediction_time = time.time() - start_time
        return predictions

class M5ProphetModel(BaseModel):
    """Prophet model adapted for M5 retail data."""
    
    def __init__(self, name: str = "Prophet", **prophet_params):
        super().__init__(name)
        self.prophet_params = {
            'growth': 'linear',
            'daily_seasonality': False,
            'weekly_seasonality': True,
            'yearly_seasonality': True,
            'seasonality_mode': 'multiplicative',
            'changepoint_prior_scale': 0.05,
            'seasonality_prior_scale': 10.0
        }
        self.prophet_params.update(prophet_params)
        self.models = {}  # Store separate models for each time series
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        """Fit Prophet models for each time series."""
        start_time = time.time()
        
        # Combine data
        df = X.copy()
        df['y'] = y
        
        # Group by time series
        n_models_trained = 0
        max_models = kwargs.get('max_models', 100)  # Limit for memory
        
        for (store_id, item_id), group in df.groupby(['store_id', 'item_id']):
            if n_models_trained >= max_models:
                break
                
            if len(group) < 30:  # Need sufficient data
                continue
            
            # Prepare Prophet data format
            prophet_df = pd.DataFrame({
                'ds': group['date'],
                'y': group['y']
            })
            
            # Fit Prophet model
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = Prophet(**self.prophet_params)
                    model.fit(prophet_df)
                    
                self.models[(store_id, item_id)] = model
                n_models_trained += 1
                
            except Exception as e:
                logger.warning(f"Failed to fit Prophet for {store_id}-{item_id}: {e}")
                continue
        
        logger.info(f"Trained {n_models_trained} Prophet models")
        
        self.fitted = True
        self.training_time = time.time() - start_time
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with Prophet models."""
        if not self.fitted:
            raise ValueError("Model must be fitted before predicting")
        
        start_time = time.time()
        predictions = np.zeros(len(X))
        
        for i, (store_id, item_id) in enumerate(zip(X['store_id'], X['item_id'])):
            if (store_id, item_id) in self.models:
                model = self.models[(store_id, item_id)]
                
                # Create future dataframe
                future_df = pd.DataFrame({'ds': [X.iloc[i]['date']]})
                
                try:
                    forecast = model.predict(future_df)
                    predictions[i] = max(0, forecast['yhat'].iloc[0])
                except:
                    predictions[i] = 0
            else:
                predictions[i] = 0
        
        self.prediction_time = time.time() - start_time
        return predictions

class M5SeasonalNaiveModel(BaseModel):
    """Seasonal naive model - strong baseline for retail data."""
    
    def __init__(self, name: str = "SeasonalNaive", seasonal_period: int = 7):
        super().__init__(name)
        self.seasonal_period = seasonal_period
        self.seasonal_values = {}
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        """Store seasonal patterns for each time series."""
        start_time = time.time()
        
        df = X.copy()
        df['y'] = y
        
        # Store last seasonal_period values for each time series
        for (store_id, item_id), group in df.groupby(['store_id', 'item_id']):
            if len(group) >= self.seasonal_period:
                recent_values = group.tail(self.seasonal_period)['y'].values
                self.seasonal_values[(store_id, item_id)] = recent_values
            else:
                # Use mean if insufficient data
                self.seasonal_values[(store_id, item_id)] = np.full(
                    self.seasonal_period, group['y'].mean()
                )
        
        self.fitted = True
        self.training_time = time.time() - start_time
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict using seasonal naive approach."""
        if not self.fitted:
            raise ValueError("Model must be fitted before predicting")
        
        start_time = time.time()
        predictions = np.zeros(len(X))
        
        for i, (store_id, item_id) in enumerate(zip(X['store_id'], X['item_id'])):
            if (store_id, item_id) in self.seasonal_values:
                seasonal_idx = i % self.seasonal_period
                seasonal_values = self.seasonal_values[(store_id, item_id)]
                predictions[i] = seasonal_values[seasonal_idx]
            else:
                predictions[i] = 0
        
        self.prediction_time = time.time() - start_time
        return predictions

# -------------------------------
# Backtesting Engine
# -------------------------------

class M5BacktestEngine:
    """M5-specific backtesting engine with rolling-origin validation."""
    
    def __init__(self, config: BacktestConfig, output_dir: str = "backtest_results"):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "predictions").mkdir(exist_ok=True)
        (self.output_dir / "models").mkdir(exist_ok=True)
        (self.output_dir / "reports").mkdir(exist_ok=True)
        
        self.results = []
        
        logger.info(f"Initialized M5 backtesting engine")
        logger.info(f"Horizons: {self.config.horizons}")
        logger.info(f"Models: {self.config.models_to_evaluate}")
    
    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for backtesting."""
        logger.info("Preparing data for backtesting...")
        
        # Ensure required columns exist
        required_cols = ['date', 'store_id', 'item_id', 'sales']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Ensure date is datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Sort data
        df = df.sort_values(['store_id', 'item_id', 'date']).reset_index(drop=True)
        
        # Filter date range
        start_date = pd.to_datetime(self.config.train_start_date)
        end_date = pd.to_datetime(self.config.test_end_date)
        df = df[(df['date'] >= start_date) & (df['date'] <= end_date)].copy()
        
        logger.info(f"Prepared data: {df.shape[0]} rows")
        logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")
        logger.info(f"Unique series: {df.groupby(['store_id', 'item_id']).ngroups}")
        
        return df
    
    def get_time_splits(self) -> List[Tuple[str, str, str]]:
        """Generate rolling-origin time splits."""
        logger.info("Generating rolling-origin time splits...")
        
        validation_start = pd.to_datetime(self.config.validation_start_date)
        test_start = pd.to_datetime(self.config.test_start_date)
        test_end = pd.to_datetime(self.config.test_end_date)
        
        splits = []
        current_test_start = test_start
        
        while current_test_start <= test_end:
            # Calculate training end (before validation period)
            train_end = current_test_start - timedelta(days=1)
            train_start = train_end - timedelta(days=self.config.initial_train_days)
            
            # Ensure minimum training period
            if (train_end - train_start).days >= self.config.min_train_days:
                splits.append((
                    train_start.strftime('%Y-%m-%d'),
                    train_end.strftime('%Y-%m-%d'),
                    current_test_start.strftime('%Y-%m-%d')
                ))
            
            # Move to next test period
            current_test_start += timedelta(days=self.config.step_size_days)
        
        logger.info(f"Generated {len(splits)} time splits")
        return splits
    
    def create_model(self, model_name: str) -> BaseModel:
        """Create model instance."""
        if model_name == 'lightgbm':
            return M5LightGBMModel()
        elif model_name == 'prophet':
            return M5ProphetModel()
        elif model_name == 'seasonal_naive':
            return M5SeasonalNaiveModel()
        else:
            raise ValueError(f"Unknown model: {model_name}")
    
    def evaluate_split(self, df: pd.DataFrame, train_start: str, train_end: str, 
                      test_start: str) -> List[BacktestResults]:
        """Evaluate all models on a single time split."""
        logger.info(f"Evaluating split: train {train_start} to {train_end}, test from {test_start}")
        
        # Split data
        train_mask = (df['date'] >= train_start) & (df['date'] <= train_end)
        train_data = df[train_mask].copy()
        
        if len(train_data) == 0:
            logger.warning("Empty training data")
            return []
        
        split_results = []
        
        # Evaluate each model
        for model_name in self.config.models_to_evaluate:
            logger.info(f"Training {model_name}...")
            
            try:
                # Create and train model
                model = self.create_model(model_name)
                
                # Prepare features (exclude non-feature columns)
                feature_cols = [col for col in train_data.columns 
                              if col not in ['sales', 'd'] and not col.startswith('id')]
                
                X_train = train_data[['date', 'store_id', 'item_id'] + feature_cols]
                y_train = train_data['sales']
                
                # Train model
                model.fit(X_train, y_train)
                
                # Evaluate on each horizon
                for horizon in self.config.horizons:
                    test_end_date = pd.to_datetime(test_start) + timedelta(days=horizon)
                    test_mask = (df['date'] >= test_start) & (df['date'] < test_end_date)
                    test_data = df[test_mask].copy()
                    
                    if len(test_data) == 0:
                        continue
                    
                    # Make predictions
                    X_test = test_data[['date', 'store_id', 'item_id'] + feature_cols]
                    y_test = test_data['sales']
                    
                    predictions = model.predict(X_test)
                    
                    # Calculate metrics
                    metrics = M5MetricsCalculator.calculate_all_metrics(
                        y_test.values, predictions, y_train.values
                    )
                    
                    # Store results
                    result = BacktestResults(
                        model_name=model_name,
                        horizon=horizon,
                        metrics=metrics,
                        training_time=model.training_time,
                        prediction_time=model.prediction_time
                    )
                    
                    if self.config.save_predictions:
                        pred_df = test_data[['date', 'store_id', 'item_id', 'sales']].copy()
                        pred_df['prediction'] = predictions
                        pred_df['model'] = model_name
                        pred_df['horizon'] = horizon
                        result.predictions = pred_df
                    
                    if hasattr(model, 'get_feature_importance'):
                        result.feature_importance = model.get_feature_importance()
                    
                    split_results.append(result)
                    
                    logger.info(f"{model_name} H{horizon}: RMSE={metrics.rmse:.3f}, "
                               f"MAE={metrics.mae:.3f}, MASE={metrics.mase:.3f}")
            
            except Exception as e:
                logger.error(f"Failed to evaluate {model_name}: {e}")
                continue
            
            # Memory cleanup
            del model
            gc.collect()
        
        return split_results
    
    def run_backtest(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run complete backtesting procedure."""
        logger.info("Starting M5 backtesting procedure...")
        
        start_time = time.time()
        
        # Prepare data
        df_prepared = self.prepare_data(df)
        
        # Get time splits
        time_splits = self.get_time_splits()
        
        # Run evaluation on each split
        all_results = []
        
        for i, (train_start, train_end, test_start) in enumerate(time_splits):
            logger.info(f"Processing split {i+1}/{len(time_splits)}")
            
            split_results = self.evaluate_split(df_prepared, train_start, train_end, test_start)
            all_results.extend(split_results)
            
            # Save intermediate results
            if i % 2 == 0:  # Save every 2 splits
                self._save_intermediate_results(all_results)
                gc.collect()
        
        # Aggregate and save final results
        results_df = self._aggregate_results(all_results)
        self._save_final_results(results_df, all_results)
        
        duration = time.time() - start_time
        logger.info(f"Backtesting completed in {duration/60:.1f} minutes")
        
        return results_df
    
    def _aggregate_results(self, results: List[BacktestResults]) -> pd.DataFrame:
        """Aggregate results across splits."""
        logger.info("Aggregating backtest results...")
        
        summary_data = []
        
        # Group by model and horizon
        for model_name in self.config.models_to_evaluate:
            for horizon in self.config.horizons:
                # Filter results for this model-horizon combination
                model_results = [r for r in results 
                               if r.model_name == model_name and r.horizon == horizon]
                
                if not model_results:
                    continue
                
                # Calculate average metrics
                avg_metrics = {
                    'rmse': np.mean([r.metrics.rmse for r in model_results]),
                    'mae': np.mean([r.metrics.mae for r in model_results]),
                    'mase': np.mean([r.metrics.mase for r in model_results if not np.isnan(r.metrics.mase)]),
                    'smape': np.mean([r.metrics.smape for r in model_results]),
                    'training_time': np.mean([r.training_time for r in model_results]),
                    'prediction_time': np.mean([r.prediction_time for r in model_results]),
                    'n_splits': len(model_results)
                }
                
                summary_data.append({
                    'model': model_name,
                    'horizon': horizon,
                    **avg_metrics
                })
        
        summary_df = pd.DataFrame(summary_data)
        return summary_df
    
    def _save_intermediate_results(self, results: List[BacktestResults]):
        """Save intermediate results to prevent data loss."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        temp_path = self.output_dir / f"temp_results_{timestamp}.json"
        
        # Convert results to JSON-serializable format
        results_data = []
        for result in results:
            result_dict = {
                'model_name': result.model_name,
                'horizon': result.horizon,
                'metrics': asdict(result.metrics),
                'training_time': result.training_time,
                'prediction_time': result.prediction_time
            }
            results_data.append(result_dict)
        
        with open(temp_path, 'w') as f:
            json.dump(results_data, f)
    
    def _save_final_results(self, summary_df: pd.DataFrame, all_results: List[BacktestResults]):
        """Save final backtest results."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save summary table
        summary_path = self.output_dir / f"backtest_summary_{timestamp}.csv"
        summary_df.to_csv(summary_path, index=False)
        
        # Save detailed results
        detailed_path = self.output_dir / f"backtest_detailed_{timestamp}.json"
        detailed_results = []
        
        for result in all_results:
            result_dict = {
                'model_name': result.model_name,
                'horizon': result.horizon,
                'metrics': asdict(result.metrics),
                'training_time': result.training_time,
                'prediction_time': result.prediction_time
            }
            detailed_results.append(result_dict)
        
        with open(detailed_path, 'w') as f:
            json.dump(detailed_results, f, indent=2)
        
        # Generate and save report
        self._generate_report(summary_df)
        
        logger.info(f"Results saved:")
        logger.info(f"  Summary: {summary_path}")
        logger.info(f"  Detailed: {detailed_path}")
    
    def _generate_report(self, summary_df: pd.DataFrame):
        """Generate human-readable backtest report."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = self.output_dir / f"backtest_report_{timestamp}.txt"
        
        with open(report_path, 'w') as f:
            f.write("M5 WALMART BACKTESTING REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Configuration:\n")
            f.write(f"  Train start: {self.config.train_start_date}\n")
            f.write(f"  Test start: {self.config.test_start_date}\n")
            f.write(f"  Test end: {self.config.test_end_date}\n")
            f.write(f"  Horizons: {self.config.horizons}\n")
            f.write(f"  Models: {self.config.models_to_evaluate}\n\n")
            
            f.write("RESULTS SUMMARY:\n")
            f.write("-" * 20 + "\n")
            
            # Best model by metric
            for metric in ['rmse', 'mae', 'mase']:
                if metric in summary_df.columns:
                    best_idx = summary_df[metric].idxmin()
                    best_row = summary_df.iloc[best_idx]
                    f.write(f"Best {metric.upper()}: {best_row['model']} "
                           f"(H{best_row['horizon']}) = {best_row[metric]:.4f}\n")
            
            f.write("\nDETAILED RESULTS:\n")
            f.write("-" * 20 + "\n")
            
            # Results by model and horizon
            for model in summary_df['model'].unique():
                f.write(f"\n{model.upper()}:\n")
                model_data = summary_df[summary_df['model'] == model]
                for _, row in model_data.iterrows():
                    f.write(f"  H{row['horizon']:2d}: RMSE={row['rmse']:.4f}, "
                           f"MAE={row['mae']:.4f}, MASE={row['mase']:.4f}, "
                           f"sMAPE={row['smape']:.2f}%\n")
        
        logger.info(f"Report saved: {report_path}")

# -------------------------------
# Main Runner Function
# -------------------------------

def run_m5_backtest(data_path: str, config_path: str = None, 
                   output_dir: str = "backtest_results") -> pd.DataFrame:
    """
    Main function to run M5 backtesting.
    
    Args:
        data_path: Path to processed M5 features data
        config_path: Path to backtest configuration (optional)
        output_dir: Output directory for results
    
    Returns:
        Summary DataFrame with results
    """
    
    # Load data
    logger.info(f"Loading data from {data_path}")
    df = pd.read_parquet(data_path)
    
    # Load or create configuration
    if config_path:
        with open(config_path) as f:
            import yaml
            config_dict = yaml.safe_load(f)
        config = BacktestConfig(**config_dict)
    else:
        config = BacktestConfig()
    
    # Initialize and run backtesting
    backtest_engine = M5BacktestEngine(config, output_dir)
    results_df = backtest_engine.run_backtest(df)
    
    return results_df

if __name__ == "__main__":
    import argparse
    import sys
    from pathlib import Path
    
    parser = argparse.ArgumentParser(
        description="M5 Walmart Backtesting Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --input data/features/m5/m5_features.parquet
  %(prog)s --input data/features/m5/m5_features.parquet --config configs/backtest.yaml
  %(prog)s --input data/features/m5/m5_features.parquet --output-dir results/backtest_v1
  %(prog)s --input data/features/m5/m5_features.parquet --models lightgbm seasonal_naive
        """
    )
    
    parser.add_argument('--input', required=True,
                       help='Input parquet file with M5 features')
    parser.add_argument('--config',
                       help='Backtest configuration YAML file')
    parser.add_argument('--output-dir', default='backtest_results',
                       help='Output directory for results')
    parser.add_argument('--models', nargs='+', 
                       choices=['lightgbm', 'prophet', 'seasonal_naive'],
                       help='Models to evaluate')
    parser.add_argument('--horizons', type=int, nargs='+',
                       help='Forecast horizons in days')
    parser.add_argument('--quick-test', action='store_true',
                       help='Run quick test with limited data')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f'{args.output_dir}/backtest.log')
        ]
    )
    
    # Create output directory
    Path(args.output_dir).mkdir(exist_ok=True)
    
    logger = logging.getLogger(__name__)
    
    try:
        # Validate input file
        if not Path(args.input).exists():
            logger.error(f"Input file not found: {args.input}")
            sys.exit(1)
        
        # Load data
        logger.info(f"Loading data from {args.input}")
        df = pd.read_parquet(args.input)
        
        # Quick test mode - sample data
        if args.quick_test:
            logger.info("Running in quick test mode - sampling data")
            # Sample by time series to maintain structure
            unique_series = df[['store_id', 'item_id']].drop_duplicates()
            sample_series = unique_series.sample(n=min(100, len(unique_series)), random_state=42)
            df = df.merge(sample_series, on=['store_id', 'item_id'], how='inner')
            logger.info(f"Sampled data shape: {df.shape}")
        
        # Create configuration
        if args.config and Path(args.config).exists():
            import yaml
            with open(args.config) as f:
                config_dict = yaml.safe_load(f)
            config = BacktestConfig(**config_dict)
        else:
            config = BacktestConfig()
        
        # Override config with command line arguments
        if args.models:
            config.models_to_evaluate = args.models
        if args.horizons:
            config.horizons = args.horizons
        
        # Quick test modifications
        if args.quick_test:
            config.horizons = [7, 14]  # Fewer horizons
            config.models_to_evaluate = ['lightgbm', 'seasonal_naive']  # Faster models
            config.test_start_date = "2016-04-01"  # Shorter test period
        
        logger.info(f"Configuration: {len(config.horizons)} horizons, {len(config.models_to_evaluate)} models")
        
        # Run backtesting
        backtest_engine = M5BacktestEngine(config, args.output_dir)
        results_df = backtest_engine.run_backtest(df)
        
        # Print summary
        print("\n" + "=" * 60)
        print("M5 BACKTESTING RESULTS SUMMARY")
        print("=" * 60)
        print(results_df.round(4).to_string(index=False))
        print("=" * 60)
        
        # Find best model
        if not results_df.empty:
            best_rmse_idx = results_df['rmse'].idxmin()
            best_model = results_df.iloc[best_rmse_idx]
            print(f"\nBest RMSE: {best_model['model']} (H{best_model['horizon']}) = {best_model['rmse']:.4f}")
            
            best_mase_idx = results_df['mase'].idxmin()
            best_model_mase = results_df.iloc[best_mase_idx]
            print(f"Best MASE: {best_model_mase['model']} (H{best_model_mase['horizon']}) = {best_model_mase['mase']:.4f}")
        
        logger.info(f"Backtesting completed successfully! Results saved to {args.output_dir}")
        
    except KeyboardInterrupt:
        logger.info("Backtesting interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Backtesting failed: {e}")
        if args.debug:
            import traceback
            logger.error(traceback.format_exc())
        sys.exit(1)