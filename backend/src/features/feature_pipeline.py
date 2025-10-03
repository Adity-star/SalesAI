
import numpy as np
import pandas as pd
import holidays
from typing import List, Optional
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import logging

from src.logger import logger
from src.utils.config_loader import ConfigLoader
from src.exception import CustomException 

logger = logging.getLogger(__name__)


try:
    config_loader = ConfigLoader()
    config = config_loader.load_yaml(file_path="ml_config.yaml")
    features_cfg = config.get("features", {})
    holiday_cfg = config.get("holiday", {})
    logger.info("✅ Config loaded successfully using config_loader")
except Exception as e:
    logger.error(f"❌ Failed to load config: {e}")
    raise



class FeaturePipeline:
    def __init__(self, df, target_col='sales', group_cols=None):
        self.df = df.copy()
        self.target_col = target_col
        self.group_cols = group_cols if group_cols else []
        self.country = holiday_cfg.get("country", "us")
        logger.info(f"🚀 Initialized FeaturePipeline with target: {target_col}, groups: {self.group_cols}, country: {self.country}")


    def add_date_features(self) -> pd.DataFrame:
        logger.info("📅 Adding date features")
        date_cfg = features_cfg.get("date_features", {})
        if not date_cfg.get("enabled", False):
            logger.info("⚠️ Date features are disabled in config")
            return self.df

        cols = date_cfg.get("cols", [])
        if "date" not in self.df.columns:
            logger.error("❌ Missing 'date' column in DataFrame")
            raise CustomException("Missing 'date' column in DataFrame")

        try:
            self.df["date"] = pd.to_datetime(self.df["date"])
        except Exception as e:
            logger.error(f"❌ Error converting 'date' column: {e}")
            raise CustomException(f"Error converting 'date' column: {e}") from e

        if "year" in cols:
            self.df["year"] = self.df["date"].dt.year
        if "month" in cols:
            self.df["month"] = self.df["date"].dt.month
        if "day" in cols:
            self.df["day"] = self.df["date"].dt.day
        if "dayofweek" in cols:
            self.df["dayofweek"] = self.df["date"].dt.dayofweek
        if "quarter" in cols:
            self.df["quarter"] = self.df["date"].dt.quarter
        if "weekofyear" in cols:
            self.df["weekofyear"] = self.df["date"].dt.isocalendar().week
        if "is_weekend" in cols:
            self.df["is_weekend"] = (self.df["date"].dt.dayofweek >= 5).astype(int)
        if "is_holiday" in cols:
            try:
                holiday_set = holidays.country_holidays(self.country)
                self.df["is_holiday"] = self.df["date"].dt.date.isin(holiday_set).astype(int)
            except Exception as e:
                logger.warning(f"⚠️ Could not fetch holidays for {self.country}: {e}")
                self.df["is_holiday"] = 0

        logger.info(f"✅ Added date features: {cols}")
        return self.df

    def add_lag_features(self) -> pd.DataFrame:
        logger.info("🕰 Adding lag features")
        lag_cfg = features_cfg.get("lag_features", {})
        if not lag_cfg.get("enabled", False):
            logger.info("⚠️ Lag features disabled in config")
            return self.df

        lags = lag_cfg.get("lags", [])
        if not lags:
            logger.info("⚠️ No lags specified")
            return self.df

        if self.target_col not in self.df.columns:
            logger.error(f"❌ Target column '{self.target_col}' missing")
            raise CustomException(f"Target column '{self.target_col}' missing")

        if not self.group_cols:
            logger.error("❌ group_cols must be provided for lag features")
            raise CustomException("group_cols must be provided for lag features")

        try:
            self.df = self.df.sort_values(self.group_cols + ["date"])
        except Exception as e:
            logger.error(f"❌ Sorting error: {e}")
            raise CustomException(f"Sorting error: {e}") from e

        for lag in lags:
            col_name = f"{self.target_col}_lag_{lag}"
            try:
                self.df[col_name] = self.df.groupby(self.group_cols, observed=False)[self.target_col].shift(lag)
                logger.info(f"✅ Created lag feature: {col_name}")
            except Exception as e:
                logger.error(f"❌ Error creating lag {lag}: {e}")
                raise CustomException(f"Error creating lag {lag}: {e}") from e

        return self.df

    def add_rolling_features(self) -> pd.DataFrame:
        logger.info("📈 Adding rolling features")
        roll_cfg = features_cfg.get("rolling_features", {})
        if not roll_cfg.get("enabled", False):
            logger.info("⚠️ Rolling features disabled in config")
            return self.df

        windows = roll_cfg.get("windows", [])
        funcs = roll_cfg.get("functions", [])
        if not windows or not funcs:
            logger.info("⚠️ No rolling windows or functions defined")
            return self.df

        if self.target_col not in self.df.columns:
            logger.error(f"❌ Target column '{self.target_col}' missing")
            raise CustomException(f"Target column '{self.target_col}' missing")

        if not self.group_cols:
            logger.error("❌ group_cols must be provided for rolling features")
            raise CustomException("group_cols must be provided for rolling features")

        try:
            self.df = self.df.sort_values(self.group_cols + ["date"])
        except Exception as e:
            logger.error(f"❌ Sorting error: {e}")
            raise CustomException(f"Sorting error: {e}") from e

        for window in windows:
            for func in funcs:
                col_name = f"{self.target_col}_roll_{window}_{func}"
                try:
                    self.df[col_name] = (
                        self.df.groupby(self.group_cols, observed=False)[self.target_col]
                        .transform(lambda x: getattr(x.rolling(window=window, min_periods=1), func)())
                    )
                    logger.info(f"✅ Created rolling feature: {col_name}")
                except Exception as e:
                    logger.error(f"❌ Error creating rolling feature {col_name}: {e}")
                    raise CustomException(f"Error creating rolling feature {col_name}: {e}") from e

        return self.df
    
    def add_cyclical_features(self) -> pd.DataFrame:
        if not features_cfg.get("cyclical_features", {}).get("enabled", False):
            logger.info("⚠️ Cyclical features are disabled in config.")
            return self.df

        required_cols = ["month", "dayofweek"]
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        if missing_cols:
            logger.error(f"❌ Missing columns for cyclical features: {missing_cols}")
            raise KeyError(f"Required columns not found: {missing_cols}")

        logger.info("🔄 Adding cyclical features (month and dayofweek)...")

        try:
            self.df["month_sin"] = np.sin(2 * np.pi * self.df["month"] / 12)
            self.df["month_cos"] = np.cos(2 * np.pi * self.df["month"] / 12)
            self.df["dow_sin"] = np.sin(2 * np.pi * self.df["dayofweek"] / 7)
            self.df["dow_cos"] = np.cos(2 * np.pi * self.df["dayofweek"] / 7)
            logger.info("✅ Cyclical features added successfully.")
        except Exception as e:
            logger.error(f"💥 Error adding cyclical features: {e}")
            raise

        return self.df


    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        logger.info(f"🧹 Handling missing values for {len(numeric_columns)} numeric columns.")

        for col in numeric_columns:
            if df[col].isnull().any():
                num_missing = df[col].isnull().sum()
                if 'lag' in col or 'roll' in col:
                    df[col] = df[col].ffill().bfill()
                    logger.info(f"🔄 Filled {num_missing} missing values in time-based feature '{col}' using forward/backward fill.")
                else:
                    mean_val = df[col].mean()
                    df[col] = df[col].fillna(mean_val)
                    logger.info(f"🟰 Filled {num_missing} missing values in feature '{col}' using mean value {mean_val:.4f}.")

        logger.info("✅ Completed handling missing values.")
        return df


    def run(self) -> pd.DataFrame:
        logger.info("🚦 Starting Feature Pipeline...")

        try:
            self.df = self.add_date_features()
            logger.info("📅 Date features added.")

            self.df = self.add_lag_features()
            logger.info("🕰 Lag features added.")

            self.df = self.add_rolling_features()
            logger.info("📈 Rolling features added.")

            self.df = self.add_cyclical_features()
            logger.info("🔄 Cyclical features added.")

            self.df = self.handle_missing_values(self.df)
            logger.info("🧹 Handled missing values.")

            logger.info("🏁 Feature Pipeline completed successfully.")
        except Exception as e:
            logger.error(f"💥 Feature Pipeline failed: {e}")
            raise

        return self.df


    def select_features(self, df: pd.DataFrame, target_col: str,
                        importance_threshold: float = 0.001) -> List[str]:
        logger.info("🔍 Selecting important features using Random Forest")

        X = df.drop(columns=[target_col])
        y = df[target_col]

        # Encode categorical features
        label_encoders = {}
        for col in X.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le
            logger.info(f"🔤 Encoded categorical feature: {col}")

        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X, y)

        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)

        selected_features = feature_importance[
            feature_importance['importance'] >= importance_threshold
        ]['feature'].tolist()

        logger.info(f"✅ Selected {len(selected_features)} features out of {len(X.columns)} total features")
        return selected_features


    def create_advanced_features(self, df: pd.DataFrame, target_col: str,
                                date_col: str, group_cols: Optional[List[str]] = None) -> pd.DataFrame:
        logger.info("✨ Creating advanced features")

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
            logger.info(f"📊 Created EWM feature with span {span}")

        # Trend features
        if group_cols:
            df['trend'] = df.groupby(group_cols).cumcount()
        else:
            df['trend'] = np.arange(len(df))
        df['trend_squared'] = df['trend'] ** 2
        logger.info("📈 Created trend and trend_squared features")

        # Velocity and acceleration
        df[f'{target_col}_velocity'] = df[target_col].diff()
        df[f'{target_col}_acceleration'] = df[f'{target_col}_velocity'].diff()
        logger.info("⚡ Created velocity and acceleration features")

        # Ratio features to rolling means
        for window in [7, 30]:
            rolling_mean = df[target_col].rolling(window, min_periods=1).mean()
            df[f'{target_col}_ratio_to_{window}d_avg'] = df[target_col] / (rolling_mean + 1)
            logger.info(f"📉 Created ratio to {window}-day rolling average feature")

        # Date-derived features
        df['day_of_month'] = df[date_col].dt.day
        df['is_month_start'] = (df['day_of_month'] <= 5).astype(int)
        df['is_month_end'] = (df['day_of_month'] >= 25).astype(int)
        df['week_of_month'] = (df['day_of_month'] - 1) // 7 + 1
        df['quarter_progress'] = (df[date_col].dt.month - 1) % 3 + 1
        df['is_quarter_end'] = (df['quarter_progress'] == 3).astype(int)
        logger.info("📅 Created date-derived features")

        # Interaction feature example
        if 'has_promotion' in df.columns and 'is_weekend' in df.columns:
            df['promotion_weekend'] = df['has_promotion'] * df['is_weekend']
            logger.info("🎉 Created interaction feature: promotion_weekend")

        # Days since month start for monthly patterns
        df['days_since_month_start'] = df['day_of_month']

        logger.info("✅ Advanced features created")
        return df


    def create_target_encoding(self, df: pd.DataFrame, target_col: str, 
                            categorical_cols: List[str], smoothing: float = 1.0) -> pd.DataFrame:
        logger.info("🎯 Creating target encoding for categorical columns")

        df = df.copy()

        if not categorical_cols:
            logger.warning("⚠️ No categorical columns provided for target encoding.")
            return df

        try:
            global_mean = df[target_col].mean()
            logger.info(f"📊 Global mean of target '{target_col}': {global_mean:.4f}")

            for col in categorical_cols:
                if col not in df.columns:
                    logger.warning(f"⚠️ Column '{col}' not found in DataFrame. Skipping target encoding.")
                    continue

                logger.info(f"🔤 Applying target encoding for column: {col}")
                means = df.groupby(col)[target_col].mean()
                counts = df[col].value_counts()

                smooth_mean = {
                    cat: (counts[cat] * means[cat] + smoothing * global_mean) / (counts[cat] + smoothing)
                    for cat in counts.index
                }

                encoded_col = f'{col}_target_encoded'
                df[encoded_col] = df[col].map(smooth_mean).fillna(global_mean)

                logger.info(f"✅ Created target-encoded feature: {encoded_col}")

            logger.info(f"🏁 Target encoding completed for {len(categorical_cols)} columns.")
        except Exception as e:
            logger.error(f"💥 Failed to apply target encoding: {e}")
            raise

        return df
