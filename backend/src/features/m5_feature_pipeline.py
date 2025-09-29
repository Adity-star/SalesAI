
import sys
import logging
import numpy as np
import pandas as pd
import logging

from datetime import datetime
from typing import  Dict, Optional
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import logging
from pathlib import Path
from src.logger import logger
from src.utils.config_loader import load_features_config
from src.exception import CustomException

logger = logging.getLogger(__name__)

class M5FeaturePipeline:
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
            if config_path:
               self.config = load_features_config(config_path)
               logger.info(f"Loaded configuration: {config_path}")
            else:
                self.config = self._get_default_config()
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
        self._validate_data()
        
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

        """Add date features based on configuration."""

        date_config = self.config.get('date_features', {})
        if not date_config.get('enabled', True):
            logger.info("⏸️ Date features disabled in configuration")
            return self.df

        logger.info("🗓️ Adding M5-specific date features...")

        # Ensure datetime format and handle invalid/missing values
        self.df[self.date_col] = pd.to_datetime(self.df[self.date_col], errors='coerce')
        print(self.date_col)

        if self.df[self.date_col].isna().any():
            logger.warning("⚠️ Date column contains missing values. Filling using forward-fill.")
            self.df[self.date_col] = self.df[self.date_col].fillna(method='ffill')

        # Basic date features from config
        basic_features = date_config.get('basic_features', [])
        for feature in basic_features:
            logger.info(f"🔹 Adding basic date feature: {feature}")

            if feature == 'year':
                self.df['year'] = self.df[self.date_col].dt.year.astype('Int16')
            elif feature == 'month':
                self.df['month'] = self.df[self.date_col].dt.month.astype('Int8')
            elif feature == 'day':
                self.df['day'] = self.df[self.date_col].dt.day.astype('Int8')
            elif feature == 'dayofweek':
                self.df['dayofweek'] = self.df[self.date_col].dt.dayofweek.astype('Int8')
            elif feature == 'quarter':
                self.df['quarter'] = self.df[self.date_col].dt.quarter.astype('Int8')
            elif feature == 'weekofyear':
                self.df['weekofyear'] = self.df[self.date_col].dt.isocalendar().week.astype('Int8')

        # Derived features from config
        derived_features = date_config.get('derived_features', [])
        for feature in derived_features:
            if feature == 'is_weekend' and 'dayofweek' in self.df.columns:
                self.df['is_weekend'] = (self.df['dayofweek'] >= 5).astype('Int8')
            elif feature == 'is_month_start' and 'day' in self.df.columns:
                self.df['is_month_start'] = (self.df['day'] <= 5).astype('Int8')
            elif feature == 'is_month_end' and 'day' in self.df.columns:
                self.df['is_month_end'] = (self.df['day'] >= 25).astype('Int8')
            elif feature == 'week_of_month' and 'day' in self.df.columns:
                self.df['week_of_month'] = ((self.df['day'] - 1) // 7 + 1).astype('Int8')
            elif feature == 'is_payday_week' and 'day' in self.df.columns:
                self.df['is_payday_week'] = ((self.df['day'] <= 7) |
                                            ((self.df['day'] >= 14) & (self.df['day'] <= 21))).astype('Int8')
                

        # Holiday features from config
        holiday_config = date_config.get('holiday_features', {})
        if holiday_config.get('enabled', True):
            try:
                import holidays
                country = holiday_config.get('country', 'US')
                years = holiday_config.get('years', list(range(2011, 2017)))
                country_holidays = holidays.country_holidays(country, years=years)

                self.df['is_holiday'] = self.df[self.date_col].dt.date.isin(country_holidays).astype('Int8')

                # Custom holidays from config
                custom_holidays = holiday_config.get('custom_holidays', [])
                for holiday in custom_holidays:
                    if holiday == 'christmas_period':
                        self.df['christmas_period'] = ((self.df['month'] == 12) &
                                                    (self.df['day'] >= 15)).astype('Int8')
                    elif holiday == 'thanksgiving_week':
                        self.df['thanksgiving_week'] = ((self.df['month'] == 11) &
                                                        (self.df['day'] >= 22)).astype('Int8')
                    elif holiday == 'back_to_school':
                        self.df['back_to_school'] = ((self.df['month'] == 8) |
                                                    ((self.df['month'] == 9) & (self.df['day'] <= 15))).astype('Int8')

                logger.info("🎉 Holiday features added successfully")

            except Exception as e:
                logger.warning(f"⚠️ Failed to add holiday features: {e}")
                self.df['is_holiday'] = 0
                raise CustomException(e, sys) from e

        # Cyclical encoding
        cyclical_config = date_config.get('cyclical_encoding', {})
        if cyclical_config.get('enabled', True):
            for feature_name, feature_config in cyclical_config.get('features', {}).items():
                if feature_name in self.df.columns:
                    period = feature_config['period']
                    prefix = feature_config['prefix']
                    self.df[f'{prefix}_sin'] = np.sin(2 * np.pi * self.df[feature_name] / period).astype('float32')
                    self.df[f'{prefix}_cos'] = np.cos(2 * np.pi * self.df[feature_name] / period).astype('float32')

            logger.info("🌀 Cyclical encoding applied successfully")

        logger.info("✅ Date features added successfully")
        logger.info(f"Columns after adding date features: {list(self.df.columns)}")

        return self.df
    
    
    def add_snap_features(self) -> pd.DataFrame:

        """Add SNAP (food assistance) benefit features - critical for Walmart."""

        logger.info("🛠️ Adding SNAP benefit features...")

        snap_cols = ['snap_CA', 'snap_TX', 'snap_WI']
        available_snap = [col for col in snap_cols if col in self.df.columns]

        if not available_snap:
            logger.warning("⚠️ No SNAP columns found - skipping SNAP features")
            return self.df

        # Initialize snap_any to 0
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
                    self.df[col_name] = (self.df['snap_any'] * (self.df['cat_id'] == cat).astype('int8')).astype('int8')
                    logger.info(f"✅ Added SNAP interaction feature: {col_name}")

        # SNAP benefit timing features (benefits mostly distributed early in month)
        self.df['snap_benefit_period'] = ((self.df['snap_any'] == 1) & (self.df['day'] <= 10)).astype('int8')
        logger.info("✅ Added 'snap_benefit_period' feature")

        logger.info(f"🎉 SNAP features added using columns: {available_snap}")
        return self.df
    
    
    def add_price_features(self) -> pd.DataFrame:

        logger.info("🛠️ Adding price features...")

        if 'sell_price' not in self.df.columns:
            logger.warning("⚠️ No sell_price column found - skipping price features")
            return self.df

        # Fill missing prices with forward/backward fill within item-store groups
        self.df['sell_price'] = (
            self.df.groupby(['store_id', 'item_id'], observed=False)['sell_price']
            .transform(lambda x: x.ffill().bfill())
        )
        logger.info("✅ Filled missing 'sell_price' with group forward/backward fill")

        # Extract windows and ensure they are ints
        price_lags = self.config.get('price_features', {}).get('change_features', {}).get('windows', [1, 7, 14, 28])
        price_lags = [int(w) for w in price_lags]  # <<< make sure they are ints

        increase_threshold = self.config.get('price_features', {}).get('change_features', {}).get('thresholds', {}).get('increase', 0.05)
        decrease_threshold = self.config.get('price_features', {}).get('change_features', {}).get('thresholds', {}).get('decrease', -0.05)

        for lag in price_lags:
            lag_col = f'price_lag_{lag}'
            self.df[lag_col] = (
                self.df.groupby(['store_id', 'item_id'], observed=False)['sell_price']
                .shift(lag)
                .astype('float32')
            )
            logger.info(f"✅ Added lag feature: {lag_col}")

            # Price change indicators
            change_col = f'price_change_{lag}d'
            self.df[change_col] = (
                (self.df['sell_price'] - self.df[lag_col]) /
                (self.df[lag_col] + 0.01)  # avoid division by zero
            ).astype('float32')
            logger.info(f"✅ Added price change feature: {change_col}")

            # Price increase/decrease flags
            self.df[f'price_increased_{lag}d'] = (self.df[change_col] > increase_threshold).astype('int8')
            self.df[f'price_decreased_{lag}d'] = (self.df[change_col] < decrease_threshold).astype('int8')
            logger.info(f"✅ Added price increase/decrease flags for lag {lag}d")

        # Price volatility (rolling std dev)
        volatility_windows = self.config.get('price_features', {}).get('volatility_features', {}).get('windows', [7, 28])
        volatility_windows = [int(w) for w in volatility_windows]  # <<< cast here too

        for window in volatility_windows:
            vol_col = f'price_volatility_{window}d'
            self.df[vol_col] = (
                self.df.groupby(['store_id', 'item_id'], observed=False)['sell_price']
                .rolling(window, min_periods=1)
                .std()
                .reset_index(level=[0, 1], drop=True)
                .astype('float32')
            )
            logger.info(f"✅ Added price volatility feature: {vol_col}")

        logger.info("🎉Price features added successfully")
        return self.df
    
    
    def add_event_features(self) -> pd.DataFrame:
        """Add event-related features from M5 calendar."""
        logger.info("📅 Adding event features...")

        event_cols = ['event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']
        available_events = [col for col in event_cols if col in self.df.columns]

        if not available_events:
            logger.warning("⚠️ No event columns found - skipping event features")
            return self.df

        # Has any event flag
        self.df['has_event'] = 0
        for col in ['event_name_1', 'event_name_2']:
            if col in self.df.columns:
                self.df['has_event'] = (self.df['has_event'] | self.df[col].notna()).astype('int8')
        logger.info("✅ Created 'has_event' flag")

        # Event type flags
        for col in ['event_type_1', 'event_type_2']:
            if col in self.df.columns:
                event_types = self.df[col].dropna().unique()
                for event_type in event_types:
                    if pd.notna(event_type):
                        flag_col = f'event_{event_type.lower().replace(" ", "_")}'
                        self.df[flag_col] = (self.df[col] == event_type).astype('int8')
                logger.info(f"✅ Created event type flags for column: {col}")

        # Days since/until major events
        major_events = ['Christmas', 'Thanksgiving', 'Easter', 'SuperBowl']
        for event in major_events:
            event_col = f'event_{event.lower()}'
            if event_col in self.df.columns:
                event_dates = self.df.loc[self.df[event_col] == 1, 'date'].dropna().unique()
                if len(event_dates) > 0:
                    # Calculate days difference to each event date
                    diffs = np.array([(self.df['date'] - pd.Timestamp(ed)).dt.days for ed in event_dates])
                    # Shape: (num_events, num_rows)

                    # Find closest event (min absolute difference)
                    abs_diffs = np.abs(diffs)
                    min_indices = np.argmin(abs_diffs, axis=0)
                    min_abs_diff = abs_diffs[min_indices, range(len(self.df))]
                    min_sign = np.sign(diffs[min_indices, range(len(self.df))])

                    # Cap difference at 30 days and apply sign
                    days_to_event = min_abs_diff.clip(max=30) * min_sign

                    # Handle NaNs before casting to integer
                    days_to_event_filled = pd.Series(days_to_event).fillna(0).astype('int16')
                    self.df[f'days_to_{event.lower()}'] = days_to_event_filled.values
                logger.info(f"✅ Added days to/from event feature: days_to_{event.lower()}")

        logger.info(f"🎉 Event features added using columns: {available_events}")
        return self.df

    
    def add_lag_features(self) -> pd.DataFrame:
        """Add lag features based on configuration."""

        lag_config = self.config.get('lag_features', {})
        if not lag_config.get('enabled', True):
            logger.info("⏸️ Lag features disabled in configuration")
            return self.df

        logger.info("🔄 Adding lag features...")

        # Handle sales_lags which can be a list or a dict
        sales_config = lag_config.get('sales_lags', [])
        if isinstance(sales_config, list):
            sales_lags = sales_config
            sales_dtype = 'int16'  # default dtype if only list provided

        else:
            sales_lags = sales_config.get('windows', [])
            sales_dtype = sales_config.get('dtype', 'int16')

            
        for lag in sales_lags:
            col_name = f'{self.target_col}_lag_{lag}'
            lagged = self.df.groupby(self.group_cols, observed=False)[self.target_col].shift(lag)

            if 'int' in sales_dtype:
                # Fill NaNs with -1 (or 0 if preferred) before converting to int
                lagged = lagged.fillna(-1).astype(sales_dtype)
            else:
                lagged = lagged.astype(sales_dtype)

            self.df[col_name] = lagged
            logger.debug(f"🛠️ Created lag feature: {col_name}")
        logger.info(f"✅ Added {len(sales_lags)} sales lag features")


        # Price lags
        price_config = lag_config.get('price_lags', [])
        if isinstance(price_config, list):
            price_lags = price_config
            price_dtype = 'float32'
        else:
            price_lags = price_config.get('windows', [1, 7, 14, 28])
            price_dtype = price_config.get('dtype', 'float32')

        if 'sell_price' in self.df.columns:
            for lag in price_lags:
                col_name = f'price_lag_{lag}'
                lagged = self.df.groupby(self.group_cols, observed=False)['sell_price'].shift(lag)
                self.df[col_name] = lagged.astype(price_dtype)
                logger.debug(f"🛠️ Created price lag feature: {col_name}")
        
        logger.info(f"✅ Added {len(price_lags)} sales lag features")

        # Revenue lags
        revenue_config = lag_config.get('revenue_lags', [])
        if isinstance(revenue_config, list):
            revenue_lags = revenue_config
            revenue_dtype = 'float32'
        else:
            revenue_lags = revenue_config.get('windows', [])
            revenue_dtype = revenue_config.get('dtype', 'float32')

        if 'sell_price' in self.df.columns:
            if 'revenue' not in self.df.columns:
                self.df['revenue'] = self.df[self.target_col] * self.df['sell_price']

            for lag in revenue_lags:
                col_name = f'revenue_lag_{lag}'
                lagged = self.df.groupby(self.group_cols,observed=False)['revenue'].shift(lag)
                self.df[col_name] = lagged.astype(revenue_dtype)
                logger.debug(f"🛠️ Created revenue lag feature: {col_name}")

        logger.info(f"✅ Added {len(revenue_lags)} sales lag features")
        return self.df


    def add_rolling_features(self) -> pd.DataFrame:

        """Add rolling statistical features based on configuration."""

        rolling_config = self.config.get('rolling_features', {})
        if not rolling_config.get('enabled', True):
            logger.info("ℹ️ Rolling features disabled in configuration")
            return self.df

        logger.info("🌀 Adding rolling features...")

        windows = rolling_config.get('windows', [7, 14, 28])
        windows = [int(w) for w in windows]  # Ensure they are ints

        min_periods = rolling_config.get('min_periods', 1)
        columns = rolling_config.get('columns', [self.target_col])
        statistics = rolling_config.get('statistics', [{'name': 'mean'}, {'name': 'std'}])

        for col in columns:
            if col not in self.df.columns:
                logger.warning(f"⚠️ Column '{col}' not found in dataframe, skipping rolling features for this column.")
                continue

            for window in windows:
                logger.debug(f"🔁 Using rolling window: {window} (type: {type(window)}) for column: {col}")

                for stat_config in statistics:
                    stat_name = stat_config['name']
                    stat_dtype = stat_config.get('dtype', 'float32')
                    fill_na = stat_config.get('fill_na', None)

                    col_name = f'{col}_roll_{window}_{stat_name}'

                    try:
                        grouped = self.df.groupby(self.group_cols, observed=False)[col]
                        rolling_obj = grouped.rolling(window, min_periods=min_periods)

                        if stat_name == 'mean':
                            result = rolling_obj.mean().reset_index(level=list(range(len(self.group_cols))), drop=True)
                        elif stat_name == 'std':
                            result = rolling_obj.std().reset_index(level=list(range(len(self.group_cols))), drop=True)
                            if fill_na is not None:
                                result = result.fillna(fill_na)
                        elif stat_name == 'min':
                            result = rolling_obj.min().reset_index(level=list(range(len(self.group_cols))), drop=True)
                        elif stat_name == 'max':
                            result = rolling_obj.max().reset_index(level=list(range(len(self.group_cols))), drop=True)
                        else:
                            logger.warning(f"⚠️ Unknown statistic: {stat_name}, skipping...")
                            continue

                        self.df[col_name] = result.astype(stat_dtype)
                        logger.debug(f"✅ Created rolling feature: {col_name}")

                    except Exception as e:
                        logger.error(f"❌ Failed to create rolling feature {col_name}: {e}")

        logger.info(f"✅ Finished adding rolling features for {len(windows)} windows and {len(columns)} columns")
        return self.df    


    def add_advanced_features(self) -> pd.DataFrame:
        """Add advanced M5-specific features like EWM, trend, ratio, and zero-sale patterns."""
        logger.info("🚀 Adding M5 advanced features...")
        
        adv_cfg = self.config.get('advanced_features', {})
        if not adv_cfg.get('enabled', True):
            logger.info("⛔ Advanced features disabled in config")
            return self.df

        # --- Exponentially Weighted Moving Average (EWM) ---
        ewm_spans = adv_cfg.get('ewm_spans', [7, 14, 28])
        for span in ewm_spans:
            col_name = f'sales_ewm_{span}'
            try:
                self.df[col_name] = (
                    self.df.groupby(['store_id', 'item_id'], observed=False)['sales']
                        .transform(lambda x: x.ewm(span=span, adjust=False).mean())
                        .astype('float32')
                )
                logger.debug(f"📈 Created EWM feature: {col_name}")
            except Exception as e:
                logger.error(f"❌ Failed to create {col_name}: {e}")

        # --- Trend Features ---
        if adv_cfg.get('trend_features', True):
            logger.info("📊 Adding trend features...")
            try:
                self.df['time_index'] = (
                    self.df.groupby(['store_id', 'item_id'], observed=False)
                        .cumcount()
                        .fillna(0)
                        .astype('int16')
                )

                self.df['sales_velocity'] = (
                    self.df.groupby(['store_id', 'item_id'], observed=False)['sales']
                        .diff()
                        .fillna(0)
                        .astype('float32')
                )

                self.df['sales_acceleration'] = (
                    self.df.groupby(['store_id', 'item_id'], observed=False)['sales_velocity']
                        .diff()
                        .fillna(0)
                        .astype('float32')
                )
            except Exception as e:
                logger.error(f"❌ Failed to compute trend features: {e}")

        # --- Ratio Features ---
        if adv_cfg.get('ratio_features', True):
            logger.info("🔢 Adding ratio features...")
            for window in [7, 28]:
                avg_col = f'sales_roll_{window}_mean'
                if avg_col in self.df.columns:
                    ratio_col = f'sales_ratio_to_{window}d_avg'
                    try:
                        self.df[ratio_col] = (
                            (self.df['sales'] / (self.df[avg_col] + 1))
                            .fillna(0)
                            .astype('float32')
                        )
                        logger.debug(f"✅ Created {ratio_col}")
                    except Exception as e:
                        logger.warning(f"⚠️ Could not create ratio feature {ratio_col}: {e}")

        # --- Item Lifecycle ---
        try:
            self.df['days_since_first_sale'] = (
                self.df.groupby(['store_id', 'item_id'], observed=False)['date']
                    .rank(method='min')
                    .fillna(0)
                    .astype('int16')
            )
        except Exception as e:
            logger.warning(f"⚠️ Could not compute 'days_since_first_sale': {e}")

        # --- Zero Sales Patterns ---
        try:
            self.df['zero_sales_flag'] = (self.df['sales'] == 0).astype('int8')

            zero_groups = (
                self.df.groupby(['store_id', 'item_id'],observed=False)['zero_sales_flag']
                    .transform(lambda x: (x != x.shift()).cumsum())
                    .fillna(0)
                    .astype('int16')
            )

            self.df['consecutive_zero_days'] = (
                self.df.groupby(['store_id', 'item_id', zero_groups], observed=False)
                    .cumcount()
                    .where(self.df['zero_sales_flag'] == 1, 0)
                    .fillna(0)
                    .astype('int8')
            )
        except Exception as e:
            logger.warning(f"⚠️ Could not compute zero-sales features: {e}")

        logger.info("✅ Advanced features added successfully")
        return self.df


    def handle_missing_values(self) -> pd.DataFrame:
        """Handle missing values based on column-specific strategies with fallback."""
        logger.info("Handling missing values...")

        # Define strategies
        fill_strategies = {
            'time_series_features': 'forward_fill',    # e.g., lag, rolling, ewm
            'price_features': 'group_forward_fill',    # price-related columns
            'flag_features': 'zero_fill',              # binary flags or booleans
            'other_features': 'mean_fill'              # other numeric
        }

        # Validate group_cols
        if not hasattr(self, 'group_cols') or not self.group_cols:
            logger.warning("group_cols is undefined or empty; defaulting to no grouping")
            self.group_cols = []

        numeric_cols = self.df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if self.df[col].isnull().any():
                missing_count = self.df[col].isnull().sum()
                logger.debug(f"Processing '{col}' with {missing_count} missing values")

                # Assign strategy based on column name patterns
                if any(key in col for key in ['lag', 'roll', 'ewm']):
                    strategy = fill_strategies['time_series_features']
                elif 'price' in col:
                    strategy = fill_strategies['price_features']
                elif col.endswith('_flag') or 'is_' in col:
                    strategy = fill_strategies['flag_features']
                else:
                    strategy = fill_strategies['other_features']

                # Apply the strategy
                if strategy == 'forward_fill':
                    if self.group_cols:
                        self.df[col] = self.df.groupby(self.group_cols,observed=False)[col].transform(
                            lambda x: x.ffill().bfill()
                        )
                    else:
                        self.df[col] = self.df[col].ffill().bfill()
                elif strategy == 'group_forward_fill':
                    if self.group_cols:
                        self.df[col] = self.df.groupby(self.group_cols,observed=False)[col].transform(
                            lambda x: x.ffill().bfill().fillna(x.mean())
                        )
                    else:
                        self.df[col] = self.df[col].ffill().bfill().fillna(self.df[col].mean())
                elif strategy == 'zero_fill':
                    self.df[col] = self.df[col].fillna(0)
                elif strategy == 'mean_fill':
                    self.df[col] = self.df[col].fillna(self.df[col].mean())

                # Fallback for remaining NaNs (e.g., fully missing groups)
                if self.df[col].isnull().any():
                    remaining_missing = self.df[col].isnull().sum()
                    overall_mean = self.df[col].mean()
                    if not np.isnan(overall_mean):
                        self.df[col] = self.df[col].fillna(overall_mean)
                        logger.debug(f"Filled {remaining_missing} remaining NaNs in '{col}' with overall mean ({overall_mean})")
                    else:
                        self.df[col] = self.df[col].fillna(0)
                        logger.debug(f"Filled {remaining_missing} remaining NaNs in '{col}' with 0 (no valid mean)")

                logger.debug(f"Filled {missing_count} missing values in '{col}' using {strategy} with fallback")

        # Handle categorical columns
        cat_cols = self.df.select_dtypes(include=['object', 'category']).columns
        for col in cat_cols:
            if self.df[col].isnull().any():
                missing_count = self.df[col].isnull().sum()
                mode_val = self.df[col].mode()[0] if not self.df[col].mode().empty else 'missing'
                self.df[col] = self.df[col].fillna(mode_val)
                logger.debug(f"Filled {missing_count} missing values in categorical '{col}' with mode '{mode_val}'")

        # Final check for any remaining NaNs
        if self.df.isnull().any().any():
            logger.warning(f"Remaining NaNs after handling: {self.df.isnull().sum()[self.df.isnull().sum() > 0]}")
        else:
            logger.info("All missing values handled successfully")

        return self.df


    def optimize_dtypes(self) -> pd.DataFrame:
        """🔧 Optimize data types for memory efficiency."""
        import gc

        logger.info("🧪 Optimizing data types for memory efficiency...")

        # --- Integer downcasting ---
        int_cols = self.df.select_dtypes(include=['int']).columns
        for col in int_cols:
            try:
                if col.endswith('_flag') or 'is_' in col:
                    self.df[col] = self.df[col].astype('int8')
                    logger.debug(f"✅ Downcasted '{col}' to int8")
                elif 'lag' in col and 'sales' in col:
                    self.df[col] = pd.to_numeric(self.df[col], downcast='integer')
                    logger.debug(f"✅ Downcasted '{col}' using pd.to_numeric")
                elif col in ['year', 'month', 'day', 'dayofweek']:
                    self.df[col] = self.df[col].astype('int16')
                    logger.debug(f"✅ Downcasted '{col}' to int16")
            except Exception as e:
                logger.warning(f"⚠️ Could not downcast integer column '{col}': {e}")
            finally:
                gc.collect()

        # --- Float downcasting ---
        float_cols = self.df.select_dtypes(include=['float']).columns
        for col in float_cols:
            try:
                self.df[col] = self.df[col].astype('float32')
                logger.debug(f"✅ Downcasted float column '{col}' to float32")
            except Exception as e:
                logger.warning(f"⚠️ Could not downcast float column '{col}': {e}")
            finally:
                gc.collect()

        # --- Categorical conversion ---
        str_cols = self.df.select_dtypes(include=['object']).columns
        for col in str_cols:
            try:
                unique_ratio = self.df[col].nunique(dropna=False) / len(self.df)
                if unique_ratio < 0.5:
                    self.df[col] = self.df[col].astype('category')
                    logger.debug(f"✅ Converted '{col}' to category")
            except Exception as e:
                logger.warning(f"⚠️ Could not convert '{col}' to category: {e}")
            finally:
                gc.collect()

        logger.info("🎯 Data type optimization completed successfully")
        return self.df

        
    def run(self) -> pd.DataFrame:
        """Run the complete M5 feature engineering pipeline."""
        logger.info("🚀 Starting M5 Walmart Feature Pipeline...")

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

            # Step 8: Advanced features
            self.df = self.add_advanced_features()

            # Step 9: Handle missing values
            self.df = self.handle_missing_values()

            # Step 10: Optimize data types for memory efficiency
            self.df = self.optimize_dtypes()

            duration = datetime.now() - start_time
            logger.info(f"✅ M5 Feature Pipeline completed in {duration.total_seconds():.1f} seconds")
            logger.info(f"📊 Final dataset shape: {self.df.shape}")
            logger.info(f"🔧 Feature columns count: {len([col for col in self.df.columns if col not in ['date', 'store_id', 'item_id', 'sales']])}")

            return self.df

        except Exception as e:
            logger.error(f"❌ M5 Feature Pipeline failed: {e}")
            raise

    def get_feature_importance(self, target_col: str = None, sample_size: int = 100000) -> pd.DataFrame:
        """Get feature importance using RandomForest on a sample."""
        logger.info("🔍 Calculating feature importance...")

        target_col = target_col or self.target_col

        # Sample data for faster computation
        if len(self.df) > sample_size:
            sample_df = self.df.sample(n=sample_size, random_state=42)
        else:
            sample_df = self.df.copy()

        # Prepare features, excluding columns that are identifiers or target
        exclude_cols = {'date', 'store_id', 'item_id', target_col, 'd'}
        feature_cols = [col for col in sample_df.columns if col not in exclude_cols]

        X = sample_df[feature_cols].copy()
        y = sample_df[target_col]

        # Encode categorical features using LabelEncoder
        for col in X.select_dtypes(include=['category', 'object']).columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))

        # Drop rows with missing target
        mask = y.notna()
        X, y = X.loc[mask], y.loc[mask]

        # Fit RandomForest
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X, y)

        # Create importance DataFrame
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False).reset_index(drop=True)

        logger.info(f"✅ Feature importance calculated for {len(importance_df)} features")
        return importance_df


    def save_features(self, output_path: str, include_metadata: bool = True):
        """Save engineered features to parquet file."""
        logger.info(f"💾 Saving M5 features to {output_path}")

        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        if include_metadata:
            # Add metadata info
            metadata = {
                'created_at': datetime.now().isoformat(),
                'pipeline_version': 'M5_v1.0',
                'original_shape': f"{self.df.shape[0]}x{self.df.shape[1]}",
                'target_column': self.target_col,
                'feature_count': len([col for col in self.df.columns if col not in ['date', 'store_id', 'item_id', 'sales']]),
                'memory_efficient': self.memory_efficient
            }

            import pyarrow as pa
            import pyarrow.parquet as pq

            table = pa.Table.from_pandas(self.df)
            # PyArrow expects metadata keys and values as bytes
            table = table.replace_schema_metadata({k: str(v).encode() for k, v in metadata.items()})

            pq.write_table(table, output_path, compression='snappy')
        else:
            self.df.to_parquet(output_path, compression='snappy', index=False)

        file_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
        logger.info(f"📦 Features saved: {output_path} ({file_size_mb:.1f} MB)")
