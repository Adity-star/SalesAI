"""
M5 Walmart-Specific Feature Pipeline
===================================

Enhanced feature pipeline specifically designed for M5 Walmart dataset:
- Walmart-specific retail features (price elasticity, SNAP benefits, hierarchical aggregations)
- M5-specific calendar features (events, holidays, promotions)
- Advanced time series features for retail forecasting
- Memory-optimized processing for large-scale data
"""

import os
import yaml
import numpy as np
import pandas as pd
import holidays
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
import logging
import warnings
from pathlib import Path
import gc

logger = logging.getLogger(__name__)

class M5WalmartFeaturePipeline:
    """M5 Walmart-specific feature engineering pipeline."""
    
    def __init__(self, df: pd.DataFrame, target_col: str = 'sales', 
                 config: Dict = None, memory_efficient: bool = True):
        """
        Initialize M5 feature pipeline.
        
        Args:
            df: Input DataFrame with M5 data
            target_col: Target column name (default: 'sales')
            config: Feature configuration dictionary
            memory_efficient: Enable memory optimization for large datasets
        """
        self.df = df.copy()
        self.target_col = target_col
        self.memory_efficient = memory_efficient
        
        # M5-specific group columns
        self.group_cols = ['store_id', 'item_id']
        self.hierarchy_cols = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
        
        # Default configuration optimized for M5
        self.config = config or self._get_default_m5_config()
        
        # Validate input data
        self._validate_m5_data()
        
        logger.info(f"Initialized M5WalmartFeaturePipeline with {len(df)} rows")
        logger.info(f"Target: {target_col}, Memory efficient: {memory_efficient}")
    
    def _get_default_m5_config(self) -> Dict:
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
    
    def _validate_m5_data(self):
        """Validate that DataFrame has required M5 columns."""
        required_cols = ['date', 'store_id', 'item_id', 'sales']
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required M5 columns: {missing_cols}")
        
        # Check for M5-specific columns
        m5_cols = ['dept_id', 'cat_id', 'state_id', 'sell_price']
        available_m5_cols = [col for col in m5_cols if col in self.df.columns]
        logger.info(f"Available M5 columns: {available_m5_cols}")
        
        # Ensure date is datetime
        if not pd.api.types.is_datetime64_any_dtype(self.df['date']):
            self.df['date'] = pd.to_datetime(self.df['date'])
        
        # Sort data for efficient processing
        self.df = self.df.sort_values(['store_id', 'item_id', 'date']).reset_index(drop=True)
    
    def add_m5_date_features(self) -> pd.DataFrame:
        """Add M5-specific date features including Walmart calendar patterns."""
        logger.info("Adding M5-specific date features...")
        
        date_cfg = self.config.get('date_features', {})
        if not date_cfg.get('enabled', True):
            return self.df
        
        # Basic date features
        self.df['year'] = self.df['date'].dt.year
        self.df['month'] = self.df['date'].dt.month
        self.df['day'] = self.df['date'].dt.day
        self.df['dayofweek'] = self.df['date'].dt.dayofweek  # 0=Monday
        self.df['quarter'] = self.df['date'].dt.quarter
        self.df['weekofyear'] = self.df['date'].dt.isocalendar().week
        
        # Walmart-specific date features
        self.df['is_weekend'] = (self.df['dayofweek'] >= 5).astype('int8')
        self.df['is_month_start'] = (self.df['day'] <= 5).astype('int8')
        self.df['is_month_end'] = (self.df['day'] >= 25).astype('int8')
        self.df['week_of_month'] = ((self.df['day'] - 1) // 7 + 1).astype('int8')
        
        # Paycheck cycles (important for Walmart shoppers)
        self.df['is_payday_week'] = ((self.df['day'] <= 7) | 
                                    ((self.df['day'] >= 14) & (self.df['day'] <= 21))).astype('int8')
        
        # Holiday features using US holidays
        try:
            us_holidays = holidays.US(years=range(2011, 2017))  # M5 date range
            self.df['is_holiday'] = self.df['date'].dt.date.isin(us_holidays).astype('int8')
            
            # Specific retail-important holidays
            christmas_period = ((self.df['month'] == 12) & (self.df['day'] >= 15)).astype('int8')
            thanksgiving_week = ((self.df['month'] == 11) & (self.df['day'] >= 22)).astype('int8')
            back_to_school = ((self.df['month'] == 8) | 
                             ((self.df['month'] == 9) & (self.df['day'] <= 15))).astype('int8')
            
            self.df['christmas_period'] = christmas_period
            self.df['thanksgiving_week'] = thanksgiving_week
            self.df['back_to_school'] = back_to_school
            
        except Exception as e:
            logger.warning(f"Failed to add holiday features: {e}")
            self.df['is_holiday'] = 0
        
        # Cyclical encoding for better ML performance
        self.df['month_sin'] = np.sin(2 * np.pi * self.df['month'] / 12).astype('float32')
        self.df['month_cos'] = np.cos(2 * np.pi * self.df['month'] / 12).astype('float32')
        self.df['dow_sin'] = np.sin(2 * np.pi * self.df['dayofweek'] / 7).astype('float32')
        self.df['dow_cos'] = np.cos(2 * np.pi * self.df['dayofweek'] / 7).astype('float32')
        
        logger.info("M5 date features added successfully")
        return self.df
    
    def add_m5_snap_features(self) -> pd.DataFrame:
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
    
    def add_m5_price_features(self) -> pd.DataFrame:
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
    
    def add_m5_event_features(self) -> pd.DataFrame:
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
    
    def add_m5_hierarchical_features(self) -> pd.DataFrame:
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
    
    def add_m5_lag_features(self) -> pd.DataFrame:
        """Add lag features optimized for M5 retail patterns."""
        logger.info("Adding M5 lag features...")
        
        lag_cfg = self.config.get('lag_features', {})
        if not lag_cfg.get('enabled', True):
            return self.df
        
        # Sales lags (most important for forecasting)
        sales_lags = lag_cfg.get('sales_lags', [1, 2, 3, 7, 14, 21, 28])
        
        for lag in sales_lags:
            col_name = f'sales_lag_{lag}'
            self.df[col_name] = (self.df.groupby(['store_id', 'item_id'])['sales']
                                .shift(lag)
                                .astype('int16'))  # Use int16 for sales lags
            
            logger.debug(f"Created lag feature: {col_name}")
        
        # Revenue lags (if price available)
        if 'sell_price' in self.df.columns:
            revenue_lags = lag_cfg.get('revenue_lags', [7, 14, 28])
            
            if 'revenue' not in self.df.columns:
                self.df['revenue'] = self.df['sales'] * self.df['sell_price']
            
            for lag in revenue_lags:
                col_name = f'revenue_lag_{lag}'
                self.df[col_name] = (self.df.groupby(['store_id', 'item_id'])['revenue']
                                    .shift(lag)
                                    .astype('float32'))
        
        logger.info(f"Added {len(sales_lags)} sales lag features")
        return self.df
    
    def add_m5_rolling_features(self) -> pd.DataFrame:
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
    
    def add_m5_advanced_features(self) -> pd.DataFrame:
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
    
    def handle_missing_values(self) -> pd.DataFrame:
        """Handle missing values with M5-specific strategies."""
        logger.info("Handling missing values...")
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if self.df[col].isnull().any():
                missing_count = self.df[col].isnull().sum()
                missing_pct = missing_count / len(self.df) * 100
                
                if missing_pct > 50:
                    logger.warning(f"High missing percentage in {col}: {missing_pct:.1f}%")
                
                # Strategy based on column type
                if 'lag' in col or 'roll' in col or 'ewm' in col:
                    # Forward fill then backward fill for time series features
                    self.df[col] = (self.df.groupby(['store_id', 'item_id'])[col]
                                   .transform(lambda x: x.fillna(method='ffill')
                                             .fillna(method='bfill')))
                elif 'price' in col:
                    # Forward fill prices within item-store groups
                    self.df[col] = (self.df.groupby(['store_id', 'item_id'])[col]
                                   .transform(lambda x: x.fillna(method='ffill')
                                             .fillna(method='bfill')
                                             .fillna(x.mean())))
                else:
                    # Fill with 0 for flags, mean for other features
                    if col.endswith('_flag') or 'is_' in col:
                        self.df[col] = self.df[col].fillna(0)
                    else:
                        self.df[col] = self.df[col].fillna(self.df[col].mean())
                
                logger.debug(f"Filled {missing_count} missing values in {col}")
        
        logger.info("Missing value handling completed")
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
            self.df = self.add_m5_date_features()
            
            # Step 2: SNAP benefit features
            if self.config.get('walmart_features', {}).get('snap_features', True):
                self.df = self.add_m5_snap_features()
            
            # Step 3: Price features
            if self.config.get('walmart_features', {}).get('price_features', True):
                self.df = self.add_m5_price_features()
            
            # Step 4: Event features
            if self.config.get('walmart_features', {}).get('event_features', True):
                self.df = self.add_m5_event_features()
            
            # Step 5: Lag features
            self.df = self.add_m5_lag_features()
            
            # Step 6: Rolling statistical features
            self.df = self.add_m5_rolling_features()
            
            # Step 7: Hierarchical aggregation features
            if self.config.get('walmart_features', {}).get('hierarchical_features', True):
                self.df = self.add_m5_hierarchical_features()
            
            # Step 8: Advanced features
            self.df = self.add_m5_advanced_features()
            
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

# Convenience function for easy usage
def create_m5_features(df: pd.DataFrame, config: Dict = None, 
                      memory_efficient: bool = True) -> pd.DataFrame:
    """
    Convenience function to create M5 Walmart features.
    
    Args:
        df: Input DataFrame with M5 data
        config: Feature configuration dictionary
        memory_efficient: Enable memory optimization
    
    Returns:
        DataFrame with engineered features
    """
    pipeline = M5WalmartFeaturePipeline(df, config=config, memory_efficient=memory_efficient)
    return pipeline.run()

# Example configuration for M5
M5_FEATURE_CONFIG = {
    'date_features': {
        'enabled': True,
        'cols': ['year', 'month', 'day', 'dayofweek', 'quarter', 
                'weekofyear', 'is_weekend', 'is_month_start', 'is_month_end']
    },
    'lag_features': {
        'enabled': True,
        'sales_lags': [1, 2, 3, 7, 14, 21, 28, 35, 42],
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

if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="M5 Walmart Feature Engineering")
    parser.add_argument("--input", required=True, help="Input parquet file path")
    parser.add_argument("--output", required=True, help="Output parquet file path")
    parser.add_argument("--config", help="Configuration JSON file path")
    parser.add_argument("--memory-efficient", action="store_true", help="Enable memory efficient processing")
    parser.add_argument("--importance", action="store_true", help="Calculate feature importance")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    try:
        # Load data
        logger.info(f"Loading data from {args.input}")
        df = pd.read_parquet(args.input)
        
        # Load config if provided
        config = M5_FEATURE_CONFIG
        if args.config:
            import json
            with open(args.config) as f:
                config = json.load(f)
        
        # Create features
        logger.info("Starting feature engineering...")
        pipeline = M5WalmartFeaturePipeline(df, config=config, memory_efficient=args.memory_efficient)
        df_features = pipeline.run()
        
        # Save features
        pipeline.save_features(args.output)
        
        # Calculate importance if requested
        if args.importance:
            importance_df = pipeline.get_feature_importance()
            importance_path = args.output.replace('.parquet', '_importance.csv')
            importance_df.to_csv(importance_path, index=False)
            logger.info(f"Feature importance saved to {importance_path}")
        
        logger.info("Feature engineering completed successfully!")
        
    except Exception as e:
        logger.error(f"Feature engineering failed: {e}")
        raise