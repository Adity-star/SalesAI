"""
Backtesting Framework
==============================

Backtesting framework  designed for dataset:
- Rolling-origin (walk-forward) validation 
- Multiple forecast horizons (7, 14, 28, 90 days)
- Retail-specific metrics (RMSE, MAE, MASE, sMAPE, PICP)
- Hierarchical evaluation (item -> category -> store -> total)
- Multiple baseline models (LightGBM, Prophet, Seasonal Naive)
- Memory-efficient processing for large-scale evaluation
"""

import json
import gc
import time
import warnings

import pandas as pd
import numpy as np
from pathlib import Path
import logging

from src.entity.backtest_entity import BacktestConfig, BacktestResults,convert_np_types
from typing import List, Optional, Tuple
from datetime import datetime, timedelta
from abc import ABC, abstractmethod

import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error

logger = logging.getLogger(__name__)


# -------------------------------
# Metrics Calculator
# -------------------------------


logger = logging.getLogger(__name__)

class MetricsCalculator:
    """Calculate M5-specific forecasting metrics."""

    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Root Mean Square Error."""
        result = np.sqrt(mean_squared_error(y_true, y_pred))
        logger.debug(f"🔢 RMSE calculated: {result:.4f}")
        return result
    
    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Error."""
        result = mean_absolute_error(y_true, y_pred)
        logger.debug(f"✂️ MAE calculated: {result:.4f}")
        return result
    
    @staticmethod
    def mase(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray, 
             seasonal_period: int = 7) -> float:
        """Mean Absolute Scaled Error - critical for M5 competition."""
        if len(y_train) < seasonal_period:
            logger.warning("⚠️ MASE: y_train length less than seasonal_period, returning NaN")
            return np.nan
        
        naive_errors = np.abs(y_train[seasonal_period:] - y_train[:-seasonal_period])
        scale = np.mean(naive_errors)
        
        if scale == 0 or np.isnan(scale):
            logger.warning("⚠️ MASE: scale is zero or NaN, returning 0.0")
            return 0.0
        
        result = np.mean(np.abs(y_true - y_pred)) / scale
        logger.debug(f"📏 MASE calculated: {result:.4f}")
        return result
    
    @staticmethod
    def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Symmetric Mean Absolute Percentage Error."""
        numerator = np.abs(y_true - y_pred)
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
        
        mask = denominator != 0
        if not np.any(mask):
            logger.warning("⚠️ SMAPE: denominator zero for all samples, returning 0.0")
            return 0.0
        
        result = np.mean(numerator[mask] / denominator[mask]) * 100
        logger.debug(f"📉 SMAPE calculated: {result:.4f}%")
        return result
    
    @staticmethod
    def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Percentage Error."""
        mask = y_true != 0
        if not np.any(mask):
            logger.warning("⚠️ MAPE: y_true contains all zeros, returning infinity")
            return np.inf
        
        result = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        logger.debug(f"📊 MAPE calculated: {result:.4f}%")
        return result
    
    @staticmethod
    def calculate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                              y_train: np.ndarray):
        """Calculate all metrics at once."""
        logger.info("🚀 Calculating all evaluation metrics...")
        calc = MetricsCalculator()
        
        y_pred_clipped = np.maximum(y_pred, 0)  # Ensure no negative predictions
        
        metrics = dict(
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
        
        logger.info(f"✅ Metrics calculated: RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}, MASE={metrics['mase']:.4f}, SMAPE={metrics['smape']:.2f}%, MAPE={metrics['mape']:.2f}%")
        logger.debug(f"Mean prediction: {metrics['mean_prediction']:.4f}, Mean actual: {metrics['mean_actual']:.4f}, Zero predictions: {metrics['zero_predictions_pct']:.2f}%")
        
        return metrics


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



class LightGBMModel(BaseModel):
    """LightGBM model optimized for dataset."""

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
        self.fitted = False
        self.feature_importance_ = None
        self.training_time = None
        self.prediction_time = None

    def fit(self, X: pd.DataFrame, y: pd.Series,  X_valid: pd.DataFrame = None, y_valid: pd.Series = None, **kwargs):
        """Fit LightGBM model with early stopping."""
        logger.info("🚀 Starting LightGBM training...")
        start_time = time.time()

        feature_cols = [col for col in X.columns if col not in ['date', 'store_id', 'item_id']]
        X_train = X[feature_cols].copy()

        # Identify categorical features
        categorical_features = []
        for col in X_train.columns:
            if X_train[col].dtype.name == 'category':
                categorical_features.append(col)
                X_train[col] = X_train[col].astype('category')

        train_data = lgb.Dataset(
            X_train,
            label=y,
            categorical_feature=categorical_features,
            free_raw_data=False
        )

        valid_sets = None
        valid_names = None

        if X_valid is not None and y_valid is not None:
            X_val = X_valid[feature_cols].copy()
            for col in X_val.columns:
                if X_val[col].dtype.name == 'category':
                    X_val[col] = X_val[col].astype('category')
            
            valid_data = lgb.Dataset(X_val, label=y_valid, categorical_feature=categorical_features, free_raw_data=False)
            valid_sets = [train_data, valid_data]
            valid_names = ['train', 'valid']
        else:
            valid_sets = [train_data]
            valid_names = ['train']

        num_boost_round = kwargs.get('num_boost_round', 1000)
        early_stopping_rounds = kwargs.get('early_stopping_rounds', 100)

         
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model = lgb.train(
                self.lgb_params,
                train_data,
                num_boost_round=num_boost_round,
                valid_sets=valid_sets,
                valid_names=valid_names,
                callbacks=[
                lgb.early_stopping(early_stopping_rounds),
                lgb.log_evaluation(0)
            ])
        

        if self.model:
            self.feature_importance_ = pd.DataFrame({
                'feature': X_train.columns,
                'importance': self.model.feature_importance(importance_type='gain')
            }).sort_values('importance', ascending=False)
            logger.info(f"📊 Feature importance computed for {len(self.feature_importance_)} features")

        self.fitted = True
        self.training_time = time.time() - start_time
        logger.info(f"✅ Training completed in {self.training_time:.2f} seconds")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with LightGBM."""
        if not self.fitted:
            logger.error("❌ Prediction attempted before model was fitted")
            raise ValueError("Model must be fitted before predicting")

        logger.info("⚡ Starting prediction...")
        start_time = time.time()

        feature_cols = [col for col in X.columns if col not in ['date', 'store_id', 'item_id']]
        X_pred = X[feature_cols].copy()

        for col in X_pred.columns:
            if X_pred[col].dtype.name == 'category':
                X_pred[col] = X_pred[col].astype('category')

        predictions = self.model.predict(X_pred)
        predictions = np.maximum(predictions, 0)  

        self.prediction_time = time.time() - start_time
        logger.info(f"✅ Prediction completed in {self.prediction_time:.2f} seconds")
        return predictions
    

# -------------------------------
# Backtesting Engine
# -------------------------------
class BacktestEngine:
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
        logger.info("🛠️ Preparing data for backtesting...")

        # Check for NaN values
        nan_counts = df.isnull().sum()
        nan_cols = nan_counts[nan_counts > 0]
        if not nan_cols.empty:
            logger.warning(f"⚠️ Found NaN values in the following columns:\n{nan_cols.to_string()}")
        else:
            logger.info("✅ No NaN values found in the input dataframe.")
        
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
        
        logger.info(f"✅ Prepared data: {df.shape[0]} rows")
        logger.info(f"📅 Date range: {df['date'].min()} to {df['date'].max()}")
        logger.info(f"🧩 Unique series count: {df.groupby(['store_id', 'item_id'],observed=False).ngroups}")
        return df
    
    def get_time_splits(self) -> List[Tuple[str, str, str]]:
        """Generate rolling-origin time splits."""
        logger.info("🔄 Generating rolling-origin time splits...")
        
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
        
        logger.info(f"📊 Generated {len(splits)} time splits")
        return splits
    
    def create_model(self, model_name: str) -> BaseModel:
        """Create model instance."""
        if model_name == 'lightgbm':
            return LightGBMModel()
        else:
            raise ValueError(f"Unknown model: {model_name}")
    
    
    def evaluate_split(self, df: pd.DataFrame, train_start: str, train_end: str, test_start: str) -> List[BacktestResults]:
        from types import SimpleNamespace

        logger.info(f"Evaluating split: train {train_start} to {train_end}, test from {test_start}")
        
        train_mask = (df['date'] >= train_start) & (df['date'] <= train_end)
        train_data = df[train_mask].copy()
        
        if len(train_data) == 0:
            logger.warning("Empty training data")
            return []
        
        split_results = []
        
        for model_name in self.config.models_to_evaluate:
            logger.info(f"Training {model_name}...")
            
            try:
                model = self.create_model(model_name)
                
                feature_cols = [col for col in train_data.columns if col not in ['sales', 'd', 'date', 'store_id', 'item_id']]
                X_train = train_data[['date', 'store_id', 'item_id'] + feature_cols]
                y_train = train_data['sales']
                
                model.fit(X_train, y_train)
                
                for horizon in self.config.horizons:
                    test_end_date = pd.to_datetime(test_start) + timedelta(days=horizon)
                    test_mask = (df['date'] >= test_start) & (df['date'] < test_end_date)
                    test_data = df[test_mask].copy()
                    
                    if len(test_data) == 0:
                        continue
                    
                    X_test = test_data[['date', 'store_id', 'item_id'] + feature_cols]
                    y_test = test_data['sales']
                    
                    predictions = model.predict(X_test)
                    
                    # Wrap the dict into an object with attribute access
                    metrics_dict = MetricsCalculator.calculate_all_metrics(y_test.values, predictions, y_train.values)
                    metrics = SimpleNamespace(**metrics_dict)
                    
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
            
            del model
            gc.collect()
        
        return split_results

      
    def run_backtest(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run complete backtesting procedure."""
        logger.info("🚀 Starting backtesting procedure...")
        
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
        logger.info(f"✅ Backtesting completed in {duration / 60:.1f} minutes")
        
        return results_df
    
    def _aggregate_results(self, results: List[BacktestResults]) -> pd.DataFrame:
        """Aggregate results across splits."""
        logger.info("📊 Aggregating backtest results...")
        
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
        logger.info("💾 Saving intermediate results to prevent data loss...")

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        temp_path = self.output_dir / f"temp_results_{timestamp}.json"
        
        # Convert results to JSON-serializable format
        results_data = []
        for result in results:
            result_dict = {
                'model_name': result.model_name,
                'horizon': result.horizon,
                'metrics': convert_np_types(vars(result.metrics)),
                'training_time': float(result.training_time),
                'prediction_time': float(result.prediction_time)
            }
            results_data.append(result_dict)
        
        with open(temp_path, 'w') as f:
            json.dump(results_data, f)
    
    def _save_final_results(self, summary_df: pd.DataFrame, all_results: List[BacktestResults]):
        """Save final backtest results."""

        logger.info("💾 Saving final backtest results and generating report...")

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
                'metrics': convert_np_types(vars(result.metrics)),
                'training_time': float(result.training_time),
                'prediction_time': float(result.prediction_time)
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

        logger.info("📝 Generating human-readable backtest report...")

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
        logger.info("✅ Report generation completed.")
