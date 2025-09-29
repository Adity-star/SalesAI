from abc import ABC,abstractmethod
import time
import pandas as pd
from prophet import Prophet
import lightgbm as lgb
import numpy as np
from typing import Optional
from src.logger import logger
import warnings

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

def flatten_columns(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in df.columns]
    return df

class ProphetModel(BaseModel):
    """Prophet model adapted for retail data."""

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
        self.models = {}  
        self.fitted = False
        self.training_time = None
        self.prediction_time = None

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        logger.info("🚀 Starting Prophet training...")
        start_time = time.time()

        df = X.copy()
        df = flatten_columns(df)
        df['y'] = y

        logger.info(f"📊 Columns after flattening: {list(df.columns)}")

        # Auto-detect column names with case-insensitive matching
        store_cols = [c for c in df.columns if 'store_id' in c.lower()]
        item_cols = [c for c in df.columns if 'item_id' in c.lower()]

        if not store_cols or not item_cols:
            logger.error(f"❌ Could not find 'store_id' (found: {store_cols}) or 'item_id' (found: {item_cols}) columns after flattening.")
            raise ValueError("Could not find 'store_id' or 'item_id' columns after flattening.")

        # Handle duplicate columns
        if len(store_cols) > 1:
            logger.warning(f"⚠️ Multiple 'store_id' columns found: {store_cols}. Using first: {store_cols[0]}")
        store_col = store_cols[0]

        if len(item_cols) > 1:
            logger.warning(f"⚠️ Multiple 'item_id' columns found: {item_cols}. Using first: {item_cols[0]}")
        item_col = item_cols[0]

        # Validate required columns
        required_cols = ['date', store_col, item_col]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"❌ Missing required columns: {missing_cols}")
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Assign and clean store_id and item_id using iloc to ensure Series
        store_loc = df.columns.get_loc(store_col)
        item_loc = df.columns.get_loc(item_col)
        logger.debug(f"Assigning store_id from {store_col} (loc {store_loc}) and item_id from {item_col} (loc {item_loc})")
        df['store_id'] = df.iloc[:, store_loc].astype(str).str.strip()
        df['item_id'] = df.iloc[:, item_loc].astype(str).str.strip()

        n_models_trained = 0
        max_models = kwargs.get('max_models', 100)

        for (store_id, item_id), group in df.groupby(['store_id', 'item_id']):
            if n_models_trained >= max_models:
                logger.info(f"⏹️ Max models limit reached ({max_models}), stopping training.")
                break

            if len(group) < 30:
                logger.warning(f"⚠️ Skipping {store_id}-{item_id} due to insufficient data (<30 rows).")
                continue

            prophet_df = pd.DataFrame({'ds': group['date'], 'y': group['y']})

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = Prophet(**self.prophet_params)
                    # Optionally add feature-engineered columns as regressors
                    for col in ['price_lag_1', 'price_change_1d', 'zero_sales_flag']:
                        if col in group.columns:
                            prophet_df[col] = group[col]
                            model.add_regressor(col)
                    model.fit(prophet_df)

                self.models[(store_id, item_id)] = model
                n_models_trained += 1
                logger.info(f"✅ Trained Prophet model for {store_id}-{item_id}")

            except Exception as e:
                logger.warning(f"❌ Failed to fit Prophet for {store_id}-{item_id}: {e}")

        self.fitted = True
        self.training_time = time.time() - start_time
        logger.info(f"🏁 Prophet training completed: {n_models_trained} models trained in {self.training_time:.2f} seconds")
        return self


    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.fitted:
            logger.error("❌ Prediction attempted before fitting the Prophet model")
            raise ValueError("Model must be fitted before predicting")

        logger.info("⚡ Starting Prophet prediction...")
        start_time = time.time()

        X = X.copy().reset_index(drop=True)
        X = flatten_columns(X)

        store_cols = [c for c in X.columns if 'store_id' in c.lower()]
        item_cols = [c for c in X.columns if 'item_id' in c.lower()]

        if not store_cols or not item_cols:
            logger.error(f"❌ Could not find 'store_id' (found: {store_cols}) or 'item_id' (found: {item_cols}) columns in predict.")
            raise ValueError("Could not find 'store_id' or 'item_id' columns in predict.")

        # Handle duplicate columns
        if len(store_cols) > 1:
            logger.warning(f"⚠️ Multiple 'store_id' columns found: {store_cols}. Using first: {store_cols[0]}")
        store_col = store_cols[0]

        if len(item_cols) > 1:
            logger.warning(f"⚠️ Multiple 'item_id' columns found: {item_cols}. Using first: {item_cols[0]}")
        item_col = item_cols[0]

        # Assign and clean store_id and item_id using iloc to ensure Series
        store_loc = X.columns.get_loc(store_col)
        item_loc = X.columns.get_loc(item_col)
        logger.debug(f"Assigning store_id from {store_col} (loc {store_loc}) and item_id from {item_col} (loc {item_loc})")
        X['store_id'] = X.iloc[:, store_loc].astype(str).str.strip()
        X['item_id'] = X.iloc[:, item_loc].astype(str).str.strip()

        predictions = np.zeros(len(X))

        for (store_id, item_id), group in X.groupby(['store_id', 'item_id']):
            if (store_id, item_id) in self.models:
                model = self.models[(store_id, item_id)]
                future_df = pd.DataFrame({'ds': group['date']})
                # Optionally add feature-engineered columns as regressors
                for col in ['price_lag_1', 'price_change_1d', 'zero_sales_flag']:
                    if col in group.columns:
                        future_df[col] = group[col]
                try:
                    forecast = model.predict(future_df)
                    predictions[group.index] = np.maximum(0, forecast['yhat'].values)
                except Exception as e:
                    logger.warning(f"⚠️ Prediction failed for {store_id}-{item_id}: {e}")
                    predictions[group.index] = 0
            else:
                predictions[group.index] = 0
                logger.warning(f"⚠️ No model for {store_id}-{item_id}")

        self.prediction_time = time.time() - start_time
        logger.info(f"✅ Prophet prediction completed in {self.prediction_time:.2f} seconds")
        return predictions

class SeasonalNaiveModel(BaseModel):
    """Seasonal naive model - strong baseline for retail data."""

    def __init__(self, name: str = "SeasonalNaive", seasonal_period: int = 7):
        super().__init__(name)
        self.seasonal_period = seasonal_period
        self.seasonal_values = {}
        self.fitted = False
        self.training_time = None
        self.prediction_time = None

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        logger.info("🚀 Starting SeasonalNaive model training...")
        start_time = time.time()

        df = X.copy()
        df = flatten_columns(df)
        df['y'] = y

        logger.info(f"📊 Columns after flattening: {list(df.columns)}")

        store_cols = [c for c in df.columns if 'store_id' in c.lower()]
        item_cols = [c for c in df.columns if 'item_id' in c.lower()]

        if not store_cols or not item_cols:
            logger.error(f"❌ Could not find 'store_id' (found: {store_cols}) or 'item_id' (found: {item_cols}) columns after flattening.")
            raise ValueError("Could not find 'store_id' or 'item_id' columns after flattening.")

        # Handle duplicate columns
        if len(store_cols) > 1:
            logger.warning(f"⚠️ Multiple 'store_id' columns found: {store_cols}. Using first: {store_cols[0]}")
        store_col = store_cols[0]

        if len(item_cols) > 1:
            logger.warning(f"⚠️ Multiple 'item_id' columns found: {item_cols}. Using first: {item_cols[0]}")
        item_col = item_cols[0]

        # Validate required columns
        required_cols = ['date', store_col, item_col]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"❌ Missing required columns: {missing_cols}")
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Assign and clean store_id and item_id
        logger.debug(f"Assigning store_id from {store_col} and item_id from {item_col}")
        df['store_id'] = df[store_col].astype(str).str.strip()
        df['item_id'] = df[item_col].astype(str).str.strip()

        for (store_id, item_id), group in df.groupby(['store_id', 'item_id']):
            if len(group) >= self.seasonal_period:
                recent_values = group.tail(self.seasonal_period)['y'].values
                self.seasonal_values[(store_id, item_id)] = recent_values
                logger.info(f"✅ Stored seasonal values for {store_id}-{item_id}")
            else:
                mean_value = group['y'].mean() if len(group) > 0 else 0
                self.seasonal_values[(store_id, item_id)] = np.full(self.seasonal_period, mean_value)
                logger.warning(f"⚠️ Insufficient data for {store_id}-{item_id}; using mean values")

        self.fitted = True
        self.training_time = time.time() - start_time
        logger.info(f"🏁 SeasonalNaive training completed in {self.training_time:.2f} seconds")
        return self


    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.fitted:
            logger.error("❌ Prediction attempted before fitting the SeasonalNaive model")
            raise ValueError("Model must be fitted before predicting")

        logger.info("⚡ Starting SeasonalNaive prediction...")
        start_time = time.time()

        predictions = np.zeros(len(X))

        # Apply same column handling for predict
        X = X.copy().reset_index(drop=True)
        X = flatten_columns(X)

        store_cols = [c for c in X.columns if 'store_id' in c.lower()]
        item_cols = [c for c in X.columns if 'item_id' in c.lower()]

        if not store_cols or not item_cols:
            logger.error(f"❌ Could not find 'store_id' (found: {store_cols}) or 'item_id' (found: {item_cols}) columns in predict.")
            raise ValueError("Could not find 'store_id' or 'item_id' columns in predict.")

        store_col = store_cols[0]
        item_col = item_cols[0]

        X['store_id'] = X[store_col].astype(str).str.strip()
        X['item_id'] = X[item_col].astype(str).str.strip()

        for i, (store_id, item_id) in enumerate(zip(X['store_id'], X['item_id'])):
            if (store_id, item_id) in self.seasonal_values:
                seasonal_idx = i % self.seasonal_period
                seasonal_vals = self.seasonal_values[(store_id, item_id)]
                predictions[i] = seasonal_vals[seasonal_idx]
            else:
                predictions[i] = 0
                logger.warning(f"⚠️ No seasonal values found for {store_id}-{item_id} at index {i}; predicting 0")

        self.prediction_time = time.time() - start_time
        logger.info(f"✅ SeasonalNaive prediction completed in {self.prediction_time:.2f} seconds")
        return predictions


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