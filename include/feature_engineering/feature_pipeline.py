import os
import yaml
import numpy as np
import pandas as pd
import holidays
from datetime import datetime
from typing import List, Dict, Tuple, Optional
    
from typing import List, Optional
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from  from_root import from_root
import logging
from include.logger import logger

logger = logging.getLogger(__name__)

# from from_root import from_root
config_path = os.path.join(from_root(), "include", "config", "data_validation_config.yaml")


try:

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    features_cfg = config.get("features", {})
    holiday_cfg = config.get("holiday", {})
    logger.info("Config loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load config at {config_path}: {e}")
    raise


class FeaturePipeline:
    def __init__(self, df, target_col='sales', group_cols=None):
        self.df = df.copy()
        self.target_col = target_col
        self.group_cols = group_cols if group_cols else []
        self.country = holiday_cfg.get("country", "us")
        logger.info(f"Initialized FeaturePipeline with target: {target_col}, groups: {self.group_cols}, country: {self.country}")


    def add_date_features(self) -> pd.DataFrame:
        """Add date-based features from config"""
        logger.info("Adding date features...")
        date_cfg = features_cfg.get("date_features", {})
        if not date_cfg.get("enabled", False):
            logger.warning("Date features are disabled in config.")
            return self.df

        values = date_cfg.get("cols", [])

        if "date" not in self.df.columns:
            logger.error("Column 'date' not found in DataFrame.")
            raise ValueError("Missing 'date' column in data.")

        try:
            self.df["date"] = pd.to_datetime(self.df["date"])
        except Exception as e:
            logger.error(f"Failed to convert 'date' column to datetime: {e}")
            raise

        if "year" in values:
            self.df["year"] = self.df["date"].dt.year
        if "month" in values:
            self.df["month"] = self.df["date"].dt.month
        if "day" in values:
            self.df["day"] = self.df["date"].dt.day
        if "dayofweek" in values:
            self.df["dayofweek"] = self.df["date"].dt.dayofweek
        if "quarter" in values:
            self.df["quarter"] = self.df["date"].dt.quarter
        if "weekofyear" in values:
            self.df["weekofyear"] = self.df["date"].dt.isocalendar().week
        if "is_weekend" in values:
            self.df["is_weekend"] = (self.df["date"].dt.dayofweek >= 5).astype(int)
        if "is_holiday" in values:
            try:
                holiday_set = holidays.country_holidays(self.country)
                self.df["is_holiday"] = self.df["date"].dt.date.isin(holiday_set).astype(int)
            except Exception as e:
                logger.warning(f"⚠️ Failed to fetch holidays for {self.country}: {e}")
                self.df["is_holiday"] = 0

        logger.info(f"Date features added: {', '.join(values)}")
        return self.df

    def add_lag_features(self) -> pd.DataFrame:
        """Add lag features based on config settings."""
        logger.info("Adding lag features...")

        lag_cfg = features_cfg.get("lag_features", {})
        if not lag_cfg.get("enabled", False):
            logger.warning("Lag features are disabled in config.")
            return self.df

        lags = lag_cfg.get("lags", [])
        if not lags:
            logger.warning("No lags specified in config.")
            return self.df

        if self.target_col not in self.df.columns:
            logger.error(f"Target column '{self.target_col}' not found in DataFrame.")
            raise ValueError(f"Target column '{self.target_col}' is missing.")

        if not self.group_cols:
            logger.error("group_cols must be defined to generate lag features.")
            raise ValueError("group_cols must be provided for lag feature generation.")

        try:
            self.df = self.df.sort_values(self.group_cols + ["date"])
        except Exception as e:
            logger.error(f"Failed to sort DataFrame by group_cols and 'date': {e}")
            raise

        for lag in lags:
            col_name = f"{self.target_col}_lag_{lag}"
            try:
                self.df[col_name] = self.df.groupby(self.group_cols)[self.target_col].shift(lag)
                logger.info(f"Created lag feature: {col_name}")
            except Exception as e:
                logger.error(f"Failed to create lag {lag}: {e}")
                raise

        logger.info(f"Total lag features created: {len(lags)}")
        return self.df

    def add_rolling_features(self) -> pd.DataFrame:
        """Add rolling window statistical features to the DataFrame."""
        logger.info("⏳ Adding rolling features...")

        roll_cfg = features_cfg.get("rolling_features", {})
        if not roll_cfg.get("enabled", False):
            logger.warning("Rolling features are disabled in config.")
            return self.df

        windows = roll_cfg.get("windows", [])
        funcs = roll_cfg.get("functions", [])

        if not windows or not funcs:
            logger.warning("No windows or functions defined for rolling features.")
            return self.df

        if self.target_col not in self.df.columns:
            logger.error(f"Target column '{self.target_col}' not found in DataFrame.")
            raise ValueError(f"Target column '{self.target_col}' is missing.")

        if not self.group_cols:
            logger.error("group_cols must be defined to compute rolling features.")
            raise ValueError("group_cols must be provided for rolling features.")

        try:
            self.df = self.df.sort_values(self.group_cols + ["date"])
        except Exception as e:
            logger.error(f"Failed to sort DataFrame for rolling features: {e}")
            raise

        count = 0
        for w in windows:
            for func in funcs:
                col_name = f"{self.target_col}_roll_{w}_{func}"
                try:
                    self.df[col_name] = (
                        self.df.groupby(self.group_cols)[self.target_col]
                        .transform(lambda x: getattr(x.rolling(window=w, min_periods=1), func)())
                    )
                    logger.info(f"Created rolling feature: {col_name}")
                    count += 1
                except Exception as e:
                    logger.error(f"Failed to create rolling feature '{col_name}': {e}")
                    raise

        logger.info(f"Total rolling features created: {count}")
        return self.df

    def add_cyclical_features(self) -> pd.DataFrame:
        if not features_cfg.get("cyclical_features", {}).get("enabled", False):
            logger.info("Cyclical features are disabled in config.")
            return self.df

        required_cols = ["month", "dayofweek"]
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        if missing_cols:
            logger.error(f"Missing columns for cyclical features: {missing_cols}")
            raise KeyError(f"Required columns not found: {missing_cols}")

        logger.info("Adding cyclical features (month and dayofweek)...")

        try:
            self.df["month_sin"] = np.sin(2 * np.pi * self.df["month"] / 12)
            self.df["month_cos"] = np.cos(2 * np.pi * self.df["month"] / 12)
            self.df["dow_sin"] = np.sin(2 * np.pi * self.df["dayofweek"] / 7)
            self.df["dow_cos"] = np.cos(2 * np.pi * self.df["dayofweek"] / 7)
            logger.info("Cyclical features added successfully.")
        except Exception as e:
            logger.error(f"Error adding cyclical features: {e}")
            raise

        return self.df

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        logger.info(f"Handling missing values for {len(numeric_columns)} numeric columns.")

        for col in numeric_columns:
            if df[col].isnull().any():
                num_missing = df[col].isnull().sum()
                if 'lag' in col or 'roll' in col:
                    df[col] = df[col].ffill().bfill()
                    logger.info(f"Filled {num_missing} missing values in time-based feature '{col}' using forward/backward fill.")
                else:
                    mean_val = df[col].mean()
                    df[col] = df[col].fillna(mean_val)
                    logger.info(f"Filled {num_missing} missing values in feature '{col}' using mean value {mean_val:.4f}.")

        logger.info("Completed handling missing values.")
        return df

    def run(self) -> pd.DataFrame:
        logger.info("Starting Feature Pipeline...")

        try:
            self.df = self.add_date_features()
            logger.info("Date features added.")

            self.df = self.add_lag_features()
            logger.info("Lag features added.")

            self.df = self.add_rolling_features()
            logger.info("Rolling features added.")

            self.df = self.add_cyclical_features()
            logger.info("Cyclical features added.")

            self.df = self.handle_missing_values(self.df)
            logger.info("Handled missing values.")

            logger.info("Feature Pipeline completed successfully.")
        except Exception as e:
            logger.error(f"Feature Pipeline failed: {e}")
            raise

        return self.df

    def select_features(self, df: pd.DataFrame, target_col: str,
                        importance_threshold: float = 0.001) -> List[str]:
        """Select important features using Random Forest feature importances."""
        X = df.drop(columns=[target_col])
        y = df[target_col]

        # Encode categorical features
        label_encoders = {}
        for col in X.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le

        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X, y)

        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)

        selected_features = feature_importance[
            feature_importance['importance'] >= importance_threshold
        ]['feature'].tolist()

        logger.info(f"Selected {len(selected_features)} features out of {len(X.columns)} total features")
        return selected_features

    def create_advanced_features(self, df: pd.DataFrame, target_col: str,
                                 date_col: str, group_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """Create advanced features to improve model performance."""
        df = df.copy()

        # Exponentially weighted moving averages
        ewm_spans = [7, 14]
        for span in ewm_spans:
            if group_cols:
                df[f'{target_col}_ewm_{span}'] = df.groupby(group_cols)[target_col].transform(
                    lambda x: x.ewm(span=span, adjust=False).mean()
                )
            else:
                df[f'{target_col}_ewm_{span}'] = df[target_col].ewm(span=span, adjust=False).mean()

        # Trend features
        if group_cols:
            df['trend'] = df.groupby(group_cols).cumcount()
            df['trend_squared'] = df['trend'] ** 2
        else:
            df['trend'] = np.arange(len(df))
            df['trend_squared'] = df['trend'] ** 2

        # Velocity and acceleration
        df[f'{target_col}_velocity'] = df[target_col].diff()
        df[f'{target_col}_acceleration'] = df[f'{target_col}_velocity'].diff()

        # Ratio features to rolling means
        for window in [7, 30]:
            rolling_mean = df[target_col].rolling(window, min_periods=1).mean()
            df[f'{target_col}_ratio_to_{window}d_avg'] = df[target_col] / (rolling_mean + 1)

        # Date-derived features
        df['day_of_month'] = df[date_col].dt.day
        df['is_month_start'] = (df['day_of_month'] <= 5).astype(int)
        df['is_month_end'] = (df['day_of_month'] >= 25).astype(int)
        df['week_of_month'] = (df['day_of_month'] - 1) // 7 + 1
        df['quarter_progress'] = (df[date_col].dt.month - 1) % 3 + 1
        df['is_quarter_end'] = (df['quarter_progress'] == 3).astype(int)

        # Interaction feature example
        if 'has_promotion' in df.columns and 'is_weekend' in df.columns:
            df['promotion_weekend'] = df['has_promotion'] * df['is_weekend']

        # Additional ratio features (duplicated rolling ratios in original? Removed duplicates here)
        # (If you want both ratio_to_window and ratio_to_window_avg, keep both; else remove duplicates)

        # Days since month start for monthly patterns
        df['days_since_month_start'] = df['day_of_month']

        logger.info("Created advanced features")
        return df
    
    def create_target_encoding(self, df: pd.DataFrame, target_col: str, 
                           categorical_cols: List[str], smoothing: float = 1.0) -> pd.DataFrame:
        """
        Create smoothed target encoding for categorical columns.
        
        Parameters:
            df (pd.DataFrame): Input dataframe.
            target_col (str): Column to be predicted.
            categorical_cols (List[str]): List of categorical columns to encode.
            smoothing (float): Smoothing factor to balance category mean with global mean.
        
        Returns:
            pd.DataFrame: DataFrame with additional target-encoded columns.
        """
        df = df.copy()

        if not categorical_cols:
            logger.warning("No categorical columns provided for target encoding.")
            return df

        try:
            global_mean = df[target_col].mean()
            logger.info(f"Global mean of target '{target_col}': {global_mean:.4f}")

            for col in categorical_cols:
                if col not in df.columns:
                    logger.warning(f"Column '{col}' not found in DataFrame. Skipping target encoding for it.")
                    continue

                logger.info(f"Applying target encoding for column: {col}")
                means = df.groupby(col)[target_col].mean()
                counts = df[col].value_counts()

                smooth_mean = {
                    cat: (counts[cat] * means[cat] + smoothing * global_mean) / (counts[cat] + smoothing)
                    for cat in counts.index
                }

                encoded_col = f'{col}_target_encoded'
                df[encoded_col] = df[col].map(smooth_mean).fillna(global_mean)

                logger.info(f"Created target-encoded feature: {encoded_col}")

            logger.info(f"Target encoding completed for {len(categorical_cols)} columns.")
        except Exception as e:
            logger.error(f"Failed to apply target encoding: {e}")
            raise

        return df


        
