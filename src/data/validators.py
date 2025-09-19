import yaml
import pandas as pd
import numpy as np
import pandera as pa

from src.logger import logger
from typing import Dict, List, Tuple, Optional, Any
from pandera import Column, DataFrameSchema, Check


import pandas as pd
import numpy as np
import yaml
import logging
from typing import Tuple, List, Dict, Any


logger = logging.getLogger(__name__)

class DataValidator:
    def __init__(self, config_path: str = r'C:\Users\Administrator\OneDrive\Desktop\SalesAI\include\config\data_validation_config.yaml'):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.validation_config = self.config.get("validation", {})
        self.required_columns = self.validation_config.get("required_columns", [])
        self.data_types = self.validation_config.get("data_types", {})
        self.value_ranges = self.validation_config.get("value_ranges", {})
        self.check_duplicates = self.validation_config.get("check_duplicates", True)
        self.outlier_method = self.validation_config.get("outlier_method", "zscore")

    def validate_schema(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        errors = []

        # Required columns
        missing_columns = set(self.required_columns) - set(df.columns)
        if missing_columns:
            msg = f"Missing required columns: {missing_columns}"
            errors.append(msg)
            logger.error(msg)

        # Data types
        for col, expected_type in self.data_types.items():
            if col in df.columns:
                actual_type = df[col].dtype
                try:
                    if expected_type == "datetime64[ns]":
                        df[col] = pd.to_datetime(df[col], errors="coerce")
                    else:
                        df[col] = df[col].astype(expected_type)
                except Exception as e:
                    msg = f"Column {col}: Cannot convert {actual_type} → {expected_type} ({e})"
                    errors.append(msg)
                    logger.error(msg)

        is_valid = len(errors) == 0
        return is_valid, errors

    def validate_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        quality_report = {
            "total_rows": len(df),
            "column_stats": {},
            "quality_issues": []
        }

        # Duplicates
        if self.check_duplicates:
            duplicates = df.duplicated().sum()
            if duplicates > 0:
                msg = f"Found {duplicates} duplicate rows"
                quality_report["quality_issues"].append(msg)
                logger.warning(msg)

        # Column-wise stats
        for col in df.columns:
            col_stats = {
                "null_count": df[col].isnull().sum(),
                "null_percentage": (df[col].isnull().sum() / len(df)) * 100,
                "unique_values": df[col].nunique()
            }

            if np.issubdtype(df[col].dtype, np.number):
                col_stats.update({
                    "mean": df[col].mean(),
                    "std": df[col].std(),
                    "min": df[col].min(),
                    "max": df[col].max(),
                    "outliers": self._detect_outliers(df[col])
                })

                # Range checks
                if col in self.value_ranges:
                    range_cfg = self.value_ranges[col]
                    if "min" in range_cfg and df[col].min() < range_cfg["min"]:
                        msg = f"{col}: Values below min ({df[col].min()} < {range_cfg['min']})"
                        quality_report["quality_issues"].append(msg)
                        logger.warning(msg)
                    if "max" in range_cfg and df[col].max() > range_cfg["max"]:
                        msg = f"{col}: Values above max ({df[col].max()} > {range_cfg['max']})"
                        quality_report["quality_issues"].append(msg)
                        logger.warning(msg)

            quality_report["column_stats"][col] = col_stats

        return quality_report

    def _detect_outliers(self, series: pd.Series) -> int:
        if self.outlier_method == "zscore":
            z_scores = (series - series.mean()) / series.std(ddof=0)
            return int((np.abs(z_scores) > 3).sum())
        elif self.outlier_method == "iqr":
            q1, q3 = series.quantile([0.25, 0.75])
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            return int(((series < lower) | (series > upper)).sum())
        else:
            return 0



def create_pandera_schema(self) -> DataFrameSchema:
    schema_dict = {}
    for col, dtype in self.data_types.items():
        checks = []
        if col in self.value_ranges:
            r = self.value_ranges[col]
            if "min" in r:
                checks.append(Check.greater_than_or_equal_to(r["min"]))
            if "max" in r:
                checks.append(Check.less_than_or_equal_to(r["max"]))

        pandera_dtype = "datetime64" if dtype == "datetime64[ns]" else dtype
        schema_dict[col] = Column(pandera_dtype, checks=checks, nullable=True)

    return DataFrameSchema(schema_dict)

def validate_schema(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
    schema = self.create_pandera_schema()
    errors = []
    try:
        schema.validate(df, lazy=True)
        return True, []
    except pa.errors.SchemaErrors as e:
        logger.error("Schema validation failed")
        errors = str(e).split("\n")
        return False, errors

def validate_time_series(self, df: pd.DataFrame, date_col: str = "date",
                         group_cols: Optional[List[str]] = None,
                         expected_freq: str = "D") -> Dict[str, Any]:
    ts_report = {"date_range": {}, "frequency_issues": [], "gaps": []}
    df[date_col] = pd.to_datetime(df[date_col])

    ts_report["date_range"] = {
        "start": df[date_col].min().strftime("%Y-%m-%d"),
        "end": df[date_col].max().strftime("%Y-%m-%d"),
        "days": (df[date_col].max() - df[date_col].min()).days
    }

    if group_cols:
        for group, group_df in df.groupby(group_cols):
            missing = pd.date_range(group_df[date_col].min(),
                                    group_df[date_col].max(),
                                    freq=expected_freq).difference(group_df[date_col])
            if len(missing) > 0:
                ts_report["gaps"].append({
                    "group": group,
                    "gap_count": len(missing),
                    "missing_dates": missing.strftime("%Y-%m-%d").tolist()
                })
                logger.warning(f"Group {group}: {len(missing)} missing {expected_freq} dates")
    else:
        missing = pd.date_range(df[date_col].min(),
                                df[date_col].max(),
                                freq=expected_freq).difference(df[date_col])
        if len(missing) > 0:
            ts_report["gaps"] = {
                "gap_count": len(missing),
                "missing_dates": missing.strftime("%Y-%m-%d").tolist()
            }
            logger.warning(f"Found {len(missing)} missing {expected_freq} dates")

    return ts_report

def validate_prediction_data(self, df: pd.DataFrame, training_stats: Dict[str, Any]) -> Tuple[bool, List[str]]:
    errors = []
    is_valid, schema_errors = self.validate_schema(df)
    errors.extend(schema_errors)

    for col in df.select_dtypes(include=[np.number]).columns:
        if col in training_stats:
            train_mean, train_std = training_stats[col]["mean"], training_stats[col]["std"]
            pred_mean, pred_std = df[col].mean(), df[col].std()
            if abs(pred_mean - train_mean) > 3 * train_std:
                msg = f"Drift in {col}: mean {train_mean:.2f} → {pred_mean:.2f}"
                errors.append(msg)
                logger.warning(msg)

    return len(errors) == 0, errors
