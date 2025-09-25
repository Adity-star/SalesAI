"""
M5 Walmart-Specific Feature Pipeline
===================================

- Walmart-specific retail features (price elasticity, SNAP benefits, hierarchical aggregations)
- M5-specific calendar features (events, holidays, promotions)
- Advanced time series features for retail forecasting
- Memory-optimized processing for large-scale data
"""

import os
import numpy as np
import pandas as pd
import holidays
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
import logging
import warnings
from pathlib import Path
import gc
from src.logger import logger
from src.config.m5_config_loader import load_features_config, config_loader


logger = logging.getLogger(__name__)

class M5WalmartFeaturePipeline:
    """M5 Walmart-specific feature engineering pipeline with configuration support."""
    
    def __init__(self, df: pd.DataFrame, config_path: Optional[str] = None, 
                 memory_efficient: bool = True):
        """
        Initialize M5 feature pipeline.
        
        Args:
            df: Input DataFrame with M5 data
            config_path: Path to features configuration file
            memory_efficient: Enable memory optimization for large datasets
        """
        self.df = df.copy()
        self.memory_efficient = memory_efficient
        
        # Load configuration
        try:
            self.config = load_features_config(config_path)
            logger.info(f"Loaded configuration: {config_loader.get_config_summary(self.config)}")
        except Exception as e:
            logger.warning(f"Failed to load configuration: {e}. Using defaults.")
            self.config = self._get_fallback_config()
        
        # Extract configuration values
        dataset_config = self.config.get('dataset', {})
        self.target_col = dataset_config.get('target_column', 'sales')
        self.group_cols = dataset_config.get('group_columns', ['store_id', 'item_id'])
        self.hierarchy_cols = dataset_config.get('hierarchy_columns', 
                                               ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'])
        self.date_col = dataset_config.get('date_column', 'date')
        
        # Processing configuration
        processing_config = self.config.get('processing', {})
        self.chunk_size = processing_config.get('chunk_size', 100000)
        self.n_jobs = processing_config.get('n_jobs', -1)
        
        # Validate input data
        self._validate_m5_data()
        
        logger.info(f"Initialized M5WalmartFeaturePipeline with {len(df)} rows")
        logger.info(f"Target: {self.target_col}, Memory efficient: {memory_efficient}")
        
    def _get_fallback_config(self) -> Dict:
        """Get fallback configuration if loading fails."""
        return {
            'dataset': {
                'target_column': 'date'
            },
            'date_features': {'enabled': True},
            'lag_features': {
                'enabled': True,
                'sales_lags': {'windows': [1, 7, 14, 28]}
            },
            'rolling_features': {
                'enabled': True,
                'windows': [7, 14, 28],
                'statistics': [{'name': 'mean'}, {'name': 'std'}]
            },
            'walmart_features': {
                'snap_features': {'enabled': True},
                'price_features': {'enabled': True},
                'event_featu': 'sales',
                'group_columns': ['store_id', 'item_id'],
                'date_column': 'date',
                'event_features': {'enabled': True}
            }
        }
    def _get_default_config(self) -> Dict:
        """Get default configuration optimized for M5 dataset."""
        return {
            'date_features': {
                'enabled': True,
                'cols': ['year', 'month', 'day', 'dayofweek', 'quarter', 
                        'weekofyear', 'is_weekend', 'is_month_start', 'is_month_end']
            },
            'lag_features': {
                'enabled': True,
                'sales_lags': [1, 2, 3, 7, 14, 21, 28],  # Critical for M5
                'price_lags': [1, 7, 14, 28],
                'revenue_lags': [7, 14, 28]
            },
            'rolling_features': {
                'enabled': True,
                'windows': [7, 14, 28, 56],
                'functions': ['mean', 'std', 'min', 'max']
            },
            'walmart_features': {
                'enabled': True,
                'snap_features': True,
                'price_features': True,
                'event_features': True,
                'hierarchical_features': True
            },
            'advanced_features': {
                'enabled': True,
                'ewm_spans': [7, 14, 28],
                'trend_features': True,
                'ratio_features': True
            }
        }
    
    def _validate_data(self):
        """Validate that DataFrame has required M5 columns."""
        required_cols = [self.date_col] + self.group_cols + [self.target_col]
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required M5 columns: {missing_cols}")
        
        # Check for M5-specific columns
        m5_cols = ['dept_id', 'cat_id', 'state_id', 'sell_price']
        available_m5_cols = [col for col in m5_cols if col in self.df.columns]
        logger.info(f"Available M5 columns: {available_m5_cols}")
        
        # Ensure date is datetime
        if not pd.api.types.is_datetime64_any_dtype(self.df[self.date_col]):
            self.df[self.date_col] = pd.to_datetime(self.df[self.date_col])
        
        # Sort data for efficient processing
        self.df = self.df.sort_values(self.group_cols + [self.date_col]).reset_index(drop=True)
    def add_date_features(self) -> pd.DataFrame:
        """Add M5-specific date features based on configuration."""
        date_config = self.config.get('date_features', {})
        if not date_config.get('enabled', True):
            logger.info("Date features disabled in configuration")
            return self.df
        
        logger.info("Adding M5-specific date features...")
        
        # Basic date features from config
        basic_features = date_config.get('basic_features', [])
        for feature in basic_features:
            if feature == 'year':
                self.df['year'] = self.df[self.date_col].dt.year
            elif feature == 'month':
                self.df['month'] = self.df[self.date_col].dt.month
            elif feature == 'day':
                self.df['day'] = self.df[self.date_col].dt.day
            elif feature == 'dayofweek':
                self.df['dayofweek'] = self.df[self.date_col].dt.dayofweek
            elif feature == 'quarter':
                self.df['quarter'] = self.df[self.date_col].dt.quarter
            elif feature == 'weekofyear':
                self.df['weekofyear'] = self.df[self.date_col].dt.isocalendar().week
        
        # Derived features from config
        derived_features = date_config.get('derived_features', [])
        for feature in derived_features:
            if feature == 'is_weekend' and 'dayofweek' in self.df.columns:
                self.df['is_weekend'] = (self.df['dayofweek'] >= 5).astype('int8')
            elif feature == 'is_month_start' and 'day' in self.df.columns:
                self.df['is_month_start'] = (self.df['day'] <= 5).astype('int8')
            elif feature == 'is_month_end' and 'day' in self.df.columns:
                self.df['is_month_end'] = (self.df['day'] >= 25).astype('int8')
            elif feature == 'week_of_month' and 'day' in self.df.columns:
                self.df['week_of_month'] = ((self.df['day'] - 1) // 7 + 1).astype('int8')
            elif feature == 'is_payday_week' and 'day' in self.df.columns:
                self.df['is_payday_week'] = ((self.df['day'] <= 7) | 
                                           ((self.df['day'] >= 14) & (self.df['day'] <= 21))).astype('int8')
        
        # Holiday features from config
        holiday_config = date_config.get('holiday_features', {})
        if holiday_config.get('enabled', True):
            try:
                country = holiday_config.get('country', 'US')
                years = holiday_config.get('years', list(range(2011, 2017)))
                
                country_holidays = holidays.country_holidays(country, years=years)
                self.df['is_holiday'] = self.df[self.date_col].dt.date.isin(country_holidays).astype('int8')
                
                # Custom holidays from config
                custom_holidays = holiday_config.get('custom_holidays', [])
                for holiday in custom_holidays:
                    if holiday == 'christmas_period':
                        self.df['christmas_period'] = ((self.df['month'] == 12) & 
                                                     (self.df['day'] >= 15)).astype('int8')
                    elif holiday == 'thanksgiving_week':
                        self.df['thanksgiving_week'] = ((self.df['month'] == 11) & 
                                                      (self.df['day'] >= 22)).astype('int8')
                    elif holiday == 'back_to_school':
                        self.df['back_to_school'] = ((self.df['month'] == 8) | 
                                                   ((self.df['month'] == 9) & (self.df['day'] <= 15))).astype('int8')
                
            except Exception as e:
                logger.warning(f"Failed to add holiday features: {e}")
                self.df['is_holiday'] = 0
        
        # Cyclical encoding from config
        cyclical_config = date_config.get('cyclical_encoding', {})
        if cyclical_config.get('enabled', True):
            for feature_name, feature_config in cyclical_config.get('features', {}).items():
                if feature_name in self.df.columns:
                    period = feature_config['period']
                    prefix = feature_config['prefix']
                    
                    self.df[f'{prefix}_sin'] = np.sin(2 * np.pi * self.df[feature_name] / period).astype('float32')
                    self.df[f'{prefix}_cos'] = np.cos(2 * np.pi * self.df[feature_name] / period).astype('float32')
        
        logger.info("M5 date features added successfully")
        return self.df
    
    def add_lag_features(self) -> pd.DataFrame:
        """Add lag features based on configuration."""
        lag_config = self.config.get('lag_features', {})
        if not lag_config.get('enabled', True):
            logger.info("Lag features disabled in configuration")
            return self.df
        
        logger.info("Adding M5 lag features...")
        
        # Sales lags from config
        sales_config = lag_config.get('sales_lags', {})
        sales_lags = sales_config.get('windows', [1, 7, 14, 28])
        sales_dtype = sales_config.get('dtype', 'int16')
        
        for lag in sales_lags:
            col_name = f'{self.target_col}_lag_{lag}'
            self.df[col_name] = (self.df.groupby(self.group_cols)[self.target_col]
                               .shift(lag)
                               .astype(sales_dtype))
            logger.debug(f"Created lag feature: {col_name}")
        
        # Price lags from config (if price column available)
        if 'sell_price' in self.df.columns:
            price_config = lag_config.get('price_lags', {})
            if price_config:
                price_lags = price_config.get('windows', [1, 7, 14, 28])
                price_dtype = price_config.get('dtype', 'float32')
                
                for lag in price_lags:
                    col_name = f'price_lag_{lag}'
                    self.df[col_name] = (self.df.groupby(self.group_cols)['sell_price']
                                       .shift(lag)
                                       .astype(price_dtype))
        
        # Revenue lags from config
        if 'sell_price' in self.df.columns:
            revenue_config = lag_config.get('revenue_lags', {})
            if revenue_config:
                # Create revenue column if not exists
                if 'revenue' not in self.df.columns:
                    self.df['revenue'] = self.df[self.target_col] * self.df['sell_price']
                
                revenue_lags = revenue_config.get('windows', [7, 14, 28])
                revenue_dtype = revenue_config.get('dtype', 'float32')
                
                for lag in revenue_lags:
                    col_name = f'revenue_lag_{lag}'
                    self.df[col_name] = (self.df.groupby(self.group_cols)['revenue']
                                       .shift(lag)
                                       .astype(revenue_dtype))
        
        logger.info(f"Added {len(sales_lags)} sales lag features")
        return self.df
    
    def add_rolling_features(self) -> pd.DataFrame:
        """Add rolling statistical features based on configuration."""
        rolling_config = self.config.get('rolling_features', {})
        if not rolling_config.get('enabled', True):
            logger.info("Rolling features disabled in configuration")
            return self.df
        
        logger.info("Adding M5 rolling features...")
        
        windows = rolling_config.get('windows', [7, 14, 28])
        min_periods = rolling_config.get('min_periods', 1)
        columns = rolling_config.get('columns', [self.target_col])
        statistics = rolling_config.get('statistics', [{'name': 'mean'}, {'name': 'std'}])
        
        for col in columns:
            if col not in self.df.columns:
                continue
                
            for window in windows:
                for stat_config in statistics:
                    stat_name = stat_config['name']
                    stat_dtype = stat_config.get('dtype', 'float32')
                    fill_na = stat_config.get('fill_na', None)
                    
                    col_name = f'{col}_roll_{window}_{stat_name}'
                    
                    try:
                        if stat_name == 'mean':
                            result = (self.df.groupby(self.group_cols)[col]
                                    .rolling(window, min_periods=min_periods)
                                    .mean()
                                    .reset_index(0, drop=True))
                        elif stat_name == 'std':
                            result = (self.df.groupby(self.group_cols)[col]
                                    .rolling(window, min_periods=min_periods)
                                    .std()
                                    .reset_index(0, drop=True))
                            if fill_na is not None:
                                result = result.fillna(fill_na)
                        elif stat_name == 'min':
                            result = (self.df.groupby(self.group_cols)[col]
                                    .rolling(window, min_periods=min_periods)
                                    .min()
                                    .reset_index(0, drop=True))
                        elif stat_name == 'max':
                            result = (self.df.groupby(self.group_cols)[col]
                                    .rolling(window, min_periods=min_periods)
                                    .max()
                                    .reset_index(0, drop=True))
                        else:
                            logger.warning(f"Unknown statistic: {stat_name}")
                            continue
                        
                        self.df[col_name] = result.astype(stat_dtype)
                        logger.debug(f"Created rolling feature: {col_name}")
                        
                    except Exception as e:
                        logger.error(f"Failed to create rolling feature {col_name}: {e}")
        
        logger.info(f"Added rolling features for {len(windows)} windows")
        return self.df
    
    def add_walmart_specific_features(self) -> pd.DataFrame:
        """Add Walmart-specific features based on configuration."""
        walmart_config = self.config.get('walmart_features', {})
        
        # SNAP features
        if walmart_config.get('snap_features', {}).get('enabled', True):
            self._add_snap_features(walmart_config['snap_features'])
        
        # Price features  
        if walmart_config.get('price_features', {}).get('enabled', True):
            self._add_price_features(walmart_config['price_features'])
        
        # Event features
        if walmart_config.get('event_features', {}).get('enabled', True):
            self._add_event_features(walmart_config['event_features'])
        
        return self.df
    
    def add_snap_features(self) -> pd.DataFrame:
        """Add SNAP (food assistance) benefit features - critical for Walmart."""
        logger.info("Adding SNAP benefit features...")
        
        snap_cols = ['snap_CA', 'snap_TX', 'snap_WI']
        available_snap = [col for col in snap_cols if col in self.df.columns]
        
        if not available_snap:
            logger.warning("No SNAP columns found - skipping SNAP features")
            return self.df
        
        # SNAP benefits are distributed at beginning of month for most recipients
        self.df['snap_any'] = 0
        for col in available_snap:
            self.df['snap_any'] = (self.df['snap_any'] | self.df[col]).astype('int8')
        
        # SNAP interaction with grocery categories (if available)
        if 'cat_id' in self.df.columns:
            # Food categories typically benefit most from SNAP
            food_categories = ['FOODS_1', 'FOODS_2', 'FOODS_3']  # Common M5 food categories
            for cat in food_categories:
                col_name = f'snap_{cat.lower()}_interaction'
                if (self.df['cat_id'] == cat).any():
                    self.df[col_name] = (self.df['snap_any'] * 
                                       (self.df['cat_id'] == cat).astype('int8')).astype('int8')
        
        # SNAP benefit timing features
        self.df['snap_benefit_period'] = ((self.df['snap_any'] == 1) & 
                                         (self.df['day'] <= 10)).astype('int8')
        
        logger.info(f"SNAP features added using columns: {available_snap}")
        return self.df
    
    def _add_price_features(self, price_config: Dict):
        """Add price-related features."""
        logger.info("Adding M5 price features...")
        
        price_col = price_config.get('price_column', 'sell_price')
        if price_col not in self.df.columns:
            logger.warning(f"Price column {price_col} not found")
            return
        
        # Fill missing prices
        self.df[price_col] = (self.df.groupby(self.group_cols)[price_col]
                             .transform(lambda x: x.fillna(method='ffill').fillna(method='bfill')))
        
        # Price change features
        change_config = price_config.get('change_features', {})
        if change_config.get('enabled', True):
            windows = change_config.get('windows', [1, 7, 14, 28])
            thresholds = change_config.get('thresholds', {'increase': 0.05, 'decrease': -0.05})
            
            for window in windows:
                lag_col = f'price_lag_{window}'
                change_col = f'price_change_{window}d'
                
                if lag_col in self.df.columns:
                    self.df[change_col] = ((self.df[price_col] - self.df[lag_col]) / 
                                         (self.df[lag_col] + 0.01)).astype('float32')
                    
                    # Price change flags
                    self.df[f'price_increased_{window}d'] = (
                        self.df[change_col] > thresholds['increase']).astype('int8')
                    self.df[f'price_decreased_{window}d'] = (
                        self.df[change_col] < thresholds['decrease']).astype('int8')
        
        # Relative price features
        relative_config = price_config.get('relative_features', {})
        if relative_config.get('enabled', True):
            for comparison in relative_config.get('comparisons', []):
                level_col = comparison['level']
                name = comparison['name']
                
                if level_col in self.df.columns:
                    avg_price = (self.df.groupby([level_col, self.date_col])[price_col]
                               .transform('mean'))
                    self.df[f'price_vs_{name}'] = (self.df[price_col] / 
                                                  (avg_price + 0.01)).astype('float32')
        
        logger.info("Price features added successfully")
        return self.df
                                      
    
    def add_price_features(self) -> pd.DataFrame:
        """Add price-related features specific to retail forecasting."""
        logger.info("Adding M5 price features...")
        
        if 'sell_price' not in self.df.columns:
            logger.warning("No sell_price column found - skipping price features")
            return self.df
        
        # Fill missing prices with forward/backward fill within item-store groups
        self.df['sell_price'] = (self.df.groupby(['store_id', 'item_id'])['sell_price']
                                .transform(lambda x: x.fillna(method='ffill').fillna(method='bfill')))
        
        # Price change features
        price_lags = self.config.get('lag_features', {}).get('price_lags', [1, 7, 14, 28])
        
        for lag in price_lags:
            lag_col = f'price_lag_{lag}'
            self.df[lag_col] = (self.df.groupby(['store_id', 'item_id'])['sell_price']
                               .shift(lag).astype('float32'))
            
            # Price change indicators
            change_col = f'price_change_{lag}d'
            self.df[change_col] = ((self.df['sell_price'] - self.df[lag_col]) / 
                                  (self.df[lag_col] + 0.01)).astype('float32')
            
            # Price increase/decrease flags
            self.df[f'price_increased_{lag}d'] = (self.df[change_col] > 0.05).astype('int8')
            self.df[f'price_decreased_{lag}d'] = (self.df[change_col] < -0.05).astype('int8')
        
        # Relative price features (compared to category/department averages)
        if 'cat_id' in self.df.columns:
            # Category-level price positioning
            cat_avg_price = (self.df.groupby(['cat_id', 'date'])['sell_price']
                           .transform('mean'))
            self.df['price_vs_category'] = (self.df['sell_price'] / 
                                          (cat_avg_price + 0.01)).astype('float32')
            
            # Price rank within category
            self.df['price_rank_in_category'] = (self.df.groupby(['cat_id', 'date'])['sell_price']
                                               .rank(pct=True)).astype('float32')
        
        if 'dept_id' in self.df.columns:
            # Department-level price positioning
            dept_avg_price = (self.df.groupby(['dept_id', 'date'])['sell_price']
                            .transform('mean'))
            self.df['price_vs_department'] = (self.df['sell_price'] / 
                                            (dept_avg_price + 0.01)).astype('float32')
        
        # Price volatility (rolling standard deviation)
        for window in [7, 28]:
            vol_col = f'price_volatility_{window}d'
            self.df[vol_col] = (self.df.groupby(['store_id', 'item_id'])['sell_price']
                               .rolling(window, min_periods=1)
                               .std()
                               .reset_index(0, drop=True)
                               .astype('float32'))
        
        logger.info("M5 price features added successfully")
        return self.df
    
    def add_event_features(self) -> pd.DataFrame:
        """Add event-related features from M5 calendar."""
        logger.info("Adding M5 event features...")
        
        event_cols = ['event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']
        available_events = [col for col in event_cols if col in self.df.columns]
        
        if not available_events:
            logger.warning("No event columns found - skipping event features")
            return self.df
        
        # Has any event flag
        self.df['has_event'] = 0
        for col in ['event_name_1', 'event_name_2']:
            if col in self.df.columns:
                self.df['has_event'] = (self.df['has_event'] | 
                                       self.df[col].notna()).astype('int8')
        
        # Event type flags
        for col in ['event_type_1', 'event_type_2']:
            if col in self.df.columns:
                event_types = self.df[col].dropna().unique()
                for event_type in event_types:
                    if pd.notna(event_type):
                        flag_col = f'event_{event_type.lower().replace(" ", "_")}'
                        self.df[flag_col] = (self.df[col] == event_type).astype('int8')
        
        # Days since/until major events
        major_events = ['Christmas', 'Thanksgiving', 'Easter', 'SuperBowl']
        for event in major_events:
            event_col = f'event_{event.lower()}'
            if event_col in self.df.columns:
                # Create days until/since event features
                event_dates = self.df[self.df[event_col] == 1]['date']
                if len(event_dates) > 0:
                    for event_date in event_dates:
                        days_diff = (self.df['date'] - event_date).dt.days
                        # Days until event (negative) or since event (positive)
                        self.df[f'days_to_{event.lower()}'] = np.minimum(
                            np.abs(days_diff), 30  # Cap at 30 days
                        ) * np.sign(days_diff)
        
        logger.info(f"Event features added using columns: {available_events}")
        return self.df

    def handle_missing_values(self) -> pd.DataFrame:
        """Handle missing values based on configuration strategies."""
        logger.info("Handling missing values...")
        
        missing_config = self.config.get('missing_values', {})
        strategy_config = missing_config.get('strategy', {})
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if self.df[col].isnull().any():
                missing_count = self.df[col].isnull().sum()
                missing_pct = missing_count / len(self.df) * 100
                
                # Apply strategy based on column type and configuration
                if any(pattern in col for pattern in ['lag', 'roll', 'ewm']):
                    strategy = strategy_config.get('time_series_features', 'forward_fill')
                elif 'price' in col:
                    strategy = strategy_config.get('price_features', 'group_forward_fill')
                elif col.endswith('_flag') or 'is_' in col:
                    strategy = strategy_config.get('flag_features', 'zero_fill')
                else:
                    strategy = strategy_config.get('other_features', 'mean_fill')
                
                # Apply the strategy
                if strategy == 'forward_fill':
                    self.df[col] = (self.df.groupby(self.group_cols)[col]
                                   .transform(lambda x: x.fillna(method='ffill').fillna(method='bfill')))
                elif strategy == 'group_forward_fill':
                    self.df[col] = (self.df.groupby(self.group_cols)[col]
                                   .transform(lambda x: x.fillna(method='ffill')
                                             .fillna(method='bfill').fillna(x.mean())))
                elif strategy == 'zero_fill':
                    self.df[col] = self.df[col].fillna(0)
                elif strategy == 'mean_fill':
                    self.df[col] = self.df[col].fillna(self.df[col].mean())
                
                logger.debug(f"Filled {missing_count} missing values in {col} using {strategy}")
        
        logger.info("Missing value handling completed")
        return self.df
    
    def add_hierarchical_features(self) -> pd.DataFrame:
        """Add hierarchical aggregation features (item->dept->category->store->state)."""
        logger.info("Adding M5 hierarchical features...")
        
        if self.memory_efficient:
            # Process in smaller chunks to manage memory
            return self._add_hierarchical_features_chunked()
        else:
            return self._add_hierarchical_features_full()
    
    def _add_hierarchical_features_chunked(self) -> pd.DataFrame:
        """Add hierarchical features with memory management."""
        chunk_size = 100000
        total_rows = len(self.df)
        
        for start_idx in range(0, total_rows, chunk_size):
            end_idx = min(start_idx + chunk_size, total_rows)
            
            logger.debug(f"Processing hierarchical features chunk: {start_idx} to {end_idx}")
            
            # Process chunk
            chunk = self.df.iloc[start_idx:end_idx].copy()
            chunk = self._add_hierarchical_features_to_chunk(chunk)
            
            # Update main dataframe
            self.df.iloc[start_idx:end_idx] = chunk
            
            # Cleanup
            del chunk
            if start_idx % (chunk_size * 5) == 0:  # Every 5 chunks
                gc.collect()
        
        return self.df
    
    def _add_hierarchical_features_to_chunk(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """Add hierarchical features to a data chunk."""
        
        # Aggregation levels for M5
        agg_levels = [
            (['store_id'], 'store'),
            (['cat_id'], 'category'), 
            (['dept_id'], 'department'),
            (['state_id'], 'state'),
            (['store_id', 'cat_id'], 'store_cat'),
            (['store_id', 'dept_id'], 'store_dept')
        ]
        
        # Rolling windows for aggregations
        windows = [7, 28]
        
        for group_cols, prefix in agg_levels:
            # Check if all required columns exist
            if not all(col in chunk.columns for col in group_cols):
                continue
            
            for window in windows:
                # Sales aggregations
                agg_col = f'{prefix}_sales_mean_{window}d'
                chunk[agg_col] = (chunk.groupby(group_cols)['sales']
                                 .rolling(window, min_periods=1)
                                 .mean()
                                 .reset_index(0, drop=True)
                                 .astype('float32'))
                
                # Revenue aggregations (if price available)
                if 'sell_price' in chunk.columns:
                    chunk['revenue'] = chunk['sales'] * chunk['sell_price']
                    revenue_col = f'{prefix}_revenue_mean_{window}d'
                    chunk[revenue_col] = (chunk.groupby(group_cols)['revenue']
                                         .rolling(window, min_periods=1)
                                         .mean()
                                         .reset_index(0, drop=True)
                                         .astype('float32'))
        
        return chunk
    
    
    def add_rolling_features(self) -> pd.DataFrame:
        """Add rolling statistical features for M5 dataset."""
        logger.info("Adding M5 rolling features...")
        
        roll_cfg = self.config.get('rolling_features', {})
        if not roll_cfg.get('enabled', True):
            return self.df
        
        windows = roll_cfg.get('windows', [7, 14, 28, 56])
        functions = roll_cfg.get('functions', ['mean', 'std'])
        
        for window in windows:
            for func in functions:
                col_name = f'sales_roll_{window}_{func}'
                
                try:
                    if func == 'mean':
                        self.df[col_name] = (self.df.groupby(['store_id', 'item_id'])['sales']
                                            .rolling(window, min_periods=1)
                                            .mean()
                                            .reset_index(0, drop=True)
                                            .astype('float32'))
                    elif func == 'std':
                        self.df[col_name] = (self.df.groupby(['store_id', 'item_id'])['sales']
                                            .rolling(window, min_periods=1)
                                            .std()
                                            .reset_index(0, drop=True)
                                            .fillna(0)  # Fill NaN std with 0
                                            .astype('float32'))
                    elif func == 'min':
                        self.df[col_name] = (self.df.groupby(['store_id', 'item_id'])['sales']
                                            .rolling(window, min_periods=1)
                                            .min()
                                            .reset_index(0, drop=True)
                                            .astype('int16'))
                    elif func == 'max':
                        self.df[col_name] = (self.df.groupby(['store_id', 'item_id'])['sales']
                                            .rolling(window, min_periods=1)
                                            .max()
                                            .reset_index(0, drop=True)
                                            .astype('int16'))
                    
                    logger.debug(f"Created rolling feature: {col_name}")
                    
                except Exception as e:
                    logger.error(f"Failed to create rolling feature {col_name}: {e}")
        
        logger.info(f"Added rolling features for {len(windows)} windows")
        return self.df
    
    def add_advanced_features(self) -> pd.DataFrame:
        """Add advanced M5-specific features."""
        logger.info("Adding M5 advanced features...")
        
        adv_cfg = self.config.get('advanced_features', {})
        if not adv_cfg.get('enabled', True):
            return self.df
        
        # Exponentially weighted moving averages
        ewm_spans = adv_cfg.get('ewm_spans', [7, 14, 28])
        for span in ewm_spans:
            col_name = f'sales_ewm_{span}'
            self.df[col_name] = (self.df.groupby(['store_id', 'item_id'])['sales']
                                .ewm(span=span, adjust=False)
                                .mean()
                                .reset_index(0, drop=True)
                                .astype('float32'))
        
        # Trend features
        if adv_cfg.get('trend_features', True):
            # Linear trend within each time series
            self.df['time_index'] = (self.df.groupby(['store_id', 'item_id'])
                                    .cumcount()
                                    .astype('int16'))
            
            # Sales velocity and acceleration
            self.df['sales_velocity'] = (self.df.groupby(['store_id', 'item_id'])['sales']
                                        .diff()
                                        .astype('float32'))
            
            self.df['sales_acceleration'] = (self.df.groupby(['store_id', 'item_id'])['sales_velocity']
                                            .diff()
                                            .astype('float32'))
        
        # Ratio features
        if adv_cfg.get('ratio_features', True):
            # Ratio to recent averages
            for window in [7, 28]:
                avg_col = f'sales_roll_{window}_mean'
                if avg_col in self.df.columns:
                    ratio_col = f'sales_ratio_to_{window}d_avg'
                    self.df[ratio_col] = ((self.df['sales'] / (self.df[avg_col] + 1))
                                         .astype('float32'))
        
        # Item lifecycle features
        self.df['days_since_first_sale'] = (self.df.groupby(['store_id', 'item_id'])['date']
                                           .rank(method='min')
                                           .astype('int16'))
        
        # Zero sales patterns
        self.df['zero_sales_flag'] = (self.df['sales'] == 0).astype('int8')
        
        # Consecutive zero sales days
        zero_groups = (self.df.groupby(['store_id', 'item_id'])['zero_sales_flag']
                      .apply(lambda x: (x != x.shift()).cumsum()))
        
        self.df['consecutive_zero_days'] = (self.df.groupby(['store_id', 'item_id', zero_groups])
                                           .cumcount()
                                           .where(self.df['zero_sales_flag'] == 1, 0)
                                           .astype('int8'))
        
        logger.info("M5 advanced features added successfully")
        return self.df
    
    def optimize_dtypes(self) -> pd.DataFrame:
        """Optimize data types for memory efficiency."""
        logger.info("Optimizing data types for memory efficiency...")
        
        # Integer downcasting
        int_cols = self.df.select_dtypes(include=['int']).columns
        for col in int_cols:
            if col.endswith('_flag') or 'is_' in col:
                self.df[col] = self.df[col].astype('int8')
            elif 'lag' in col and 'sales' in col:
                self.df[col] = pd.to_numeric(self.df[col], downcast='integer')
            elif col in ['year', 'month', 'day', 'dayofweek']:
                self.df[col] = self.df[col].astype('int16')
        
        # Float downcasting
        float_cols = self.df.select_dtypes(include=['float']).columns
        for col in float_cols:
            self.df[col] = pd.to_numeric(self.df[col], downcast='float')
        
        # Categorical encoding for string columns with limited unique values
        str_cols = self.df.select_dtypes(include=['object']).columns
        for col in str_cols:
            unique_ratio = self.df[col].nunique() / len(self.df)
            if unique_ratio < 0.5:  # Less than 50% unique values
                self.df[col] = self.df[col].astype('category')
        
        logger.info("Data type optimization completed")
        return self.df
    
    def run(self) -> pd.DataFrame:
        """Run the complete M5 feature engineering pipeline."""
        logger.info("Starting M5 Walmart Feature Pipeline...")
        
        start_time = datetime.now()
        
        try:
            # Step 1: M5-specific date features
            self.df = self.add_date_features()
            
            # Step 2: SNAP benefit features
            if self.config.get('walmart_features', {}).get('snap_features', True):
                self.df = self.add_snap_features()
            
            # Step 3: Price features
            if self.config.get('walmart_features', {}).get('price_features', True):
                self.df = self.add_price_features()
            
            # Step 4: Event features
            if self.config.get('walmart_features', {}).get('event_features', True):
                self.df = self.add_event_features()
            
            # Step 5: Lag features
            self.df = self.add_lag_features()
            
            # Step 6: Rolling statistical features
            self.df = self.add_rolling_features()
            
            # Step 7: Hierarchical aggregation features
            if self.config.get('walmart_features', {}).get('hierarchical_features', True):
                self.df = self.add_hierarchical_features()
            
            # Step 8: Advanced features
            self.df = self.add_advanced_features()
            
            # Step 9: Handle missing values
            self.df = self.handle_missing_values()
            
            # Step 10: Optimize data types for memory efficiency
            self.df = self.optimize_dtypes()
            
            duration = datetime.now() - start_time
            logger.info(f"M5 Feature Pipeline completed in {duration.total_seconds():.1f} seconds")
            logger.info(f"Final dataset shape: {self.df.shape}")
            logger.info(f"Feature columns: {len([col for col in self.df.columns if col not in ['date', 'store_id', 'item_id', 'sales']])}")
            
            return self.df
            
        except Exception as e:
            logger.error(f"M5 Feature Pipeline failed: {e}")
            raise
    
    def get_feature_importance(self, target_col: str = None, sample_size: int = 100000) -> pd.DataFrame:
        """Get feature importance using RandomForest on a sample."""
        logger.info("Calculating feature importance...")
        
        target_col = target_col or self.target_col
        
        # Sample data for faster computation
        if len(self.df) > sample_size:
            sample_df = self.df.sample(n=sample_size, random_state=42)
        else:
            sample_df = self.df.copy()
        
        # Prepare features
        feature_cols = [col for col in sample_df.columns 
                       if col not in ['date', 'store_id', 'item_id', target_col, 'd']]
        
        X = sample_df[feature_cols]
        y = sample_df[target_col]
        
        # Handle categorical features
        for col in X.select_dtypes(include=['category', 'object']).columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
        
        # Drop rows with missing target
        mask = y.notna()
        X, y = X[mask], y[mask]
        
        # Fit RandomForest
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        
        # Create importance DataFrame
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info(f"Feature importance calculated for {len(importance_df)} features")
        return importance_df
    
    def save_features(self, output_path: str, include_metadata: bool = True):
        """Save engineered features to parquet file."""
        logger.info(f"Saving M5 features to {output_path}")
        
        # Create output directory
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        if include_metadata:
            # Add metadata
            metadata = {
                'created_at': datetime.now().isoformat(),
                'pipeline_version': 'M5_v1.0',
                'original_shape': f"{self.df.shape[0]}x{self.df.shape[1]}",
                'target_column': self.target_col,
                'feature_count': len([col for col in self.df.columns 
                                    if col not in ['date', 'store_id', 'item_id', 'sales']]),
                'memory_efficient': self.memory_efficient
            }
            
            # Save with metadata
            import pyarrow as pa
            import pyarrow.parquet as pq
            
            table = pa.Table.from_pandas(self.df)
            table = table.replace_schema_metadata({
                key: str(value) for key, value in metadata.items()
            })
            
            pq.write_table(table, output_path, compression='snappy')
        else:
            self.df.to_parquet(output_path, compression='snappy', index=False)
        
        file_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
        logger.info(f"Features saved: {output_path} ({file_size_mb:.1f} MB)")
