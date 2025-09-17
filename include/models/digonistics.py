import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from typing import Dict, Any
from pathlib import Path
from include.logger import logger

logger = logger.getLogger(__name__)
logger.setLevel(logger.INFO)


def diagnose_model_performance(train_df: pd.DataFrame, 
                              val_df: pd.DataFrame,
                              test_df: pd.DataFrame,
                              predictions: Dict[str, np.ndarray],
                              target_col: str = 'sales',
                              save_dir: str = "./diagnostics") -> Dict[str, Any]:
    """
    Diagnose why models are underperforming and export structured reports
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    diagnosis = {
        'data_quality': {},
        'distribution_shift': {},
        'prediction_analysis': {},
        'recommendations': []
    }

    # 1. Data Quality
    logger.info("Checking data quality...")
    y_train, y_val, y_test = train_df[target_col], val_df[target_col], test_df[target_col]

    diagnosis['data_quality']['train_outliers'] = detect_outliers(y_train)
    diagnosis['data_quality']['val_outliers'] = detect_outliers(y_val)
    diagnosis['data_quality']['test_outliers'] = detect_outliers(y_test)

    # 2. Distribution Shift
    logger.info("Checking distribution shift...")
    def stats(y): return {'mean': float(y.mean()), 'std': float(y.std())}
    diagnosis['distribution_shift'] = {
        'train': stats(y_train),
        'val': stats(y_val),
        'test': stats(y_test),
    }

    mean_shift_val = abs(y_val.mean() - y_train.mean()) / y_train.mean()
    mean_shift_test = abs(y_test.mean() - y_train.mean()) / y_train.mean()
    if mean_shift_val > 0.2:
        diagnosis['recommendations'].append(
            f"Validation set shows distribution shift (mean shift: {mean_shift_val:.1%})"
        )
    if mean_shift_test > 0.2:
        diagnosis['recommendations'].append(
            f"Test set shows distribution shift (mean shift: {mean_shift_test:.1%})"
        )

    # 3. Prediction Analysis
    logger.info("Analyzing predictions...")
    for model_name, pred in predictions.items():
        if pred is not None:
            residuals = y_test - pred
            diagnosis['prediction_analysis'][model_name] = {
                'pred_mean': float(pred.mean()),
                'pred_std': float(pred.std()),
                'residual_mean': float(residuals.mean()),
                'residual_std': float(residuals.std()),
                'mape': float(np.mean(np.abs(residuals / y_test)) * 100),
                'extreme_low_count': int((pred < y_test.min() * 0.5).sum()),
                'extreme_high_count': int((pred > y_test.max() * 1.5).sum())
            }

    # 4. Feature Importance Check
    feature_cols = [c for c in train_df.columns if c not in [target_col, 'date']]
    diagnosis['data_quality']['n_features'] = len(feature_cols)
    if len(feature_cols) > 50:
        diagnosis['recommendations'].append(
            f"High feature count ({len(feature_cols)}). Consider feature selection."
        )

    # 5. Data Leakage Check
    numeric_cols = [c for c in train_df.select_dtypes(include=[np.number]).columns if c != target_col]
    if numeric_cols:
        correlations = train_df[numeric_cols].corrwith(train_df[target_col])
        high_corr = correlations[abs(correlations) > 0.95]
        if len(high_corr) > 0:
            diagnosis['data_quality']['high_corr_features'] = high_corr.to_dict()
            diagnosis['recommendations'].append(
                f"Potential leakage: {len(high_corr)} features correlate >95% with target."
            )

    # 6. Sample Size
    if len(train_df) < 1000:
        diagnosis['recommendations'].append(
            f"Training set is small ({len(train_df)} samples). Consider generating more data."
        )

    # 7. Target Variable Analysis
    zeros = (y_train == 0).sum()
    if zeros > len(y_train) * 0.1:
        diagnosis['recommendations'].append(
            f"Many zero sales ({zeros} in training). Consider log transform or zero-inflated models."
        )

    # --- Save JSON ---
    json_path = Path(save_dir) / "diagnosis.json"
    with open(json_path, "w") as f:
        json.dump(diagnosis, f, indent=4)

    # --- Save Markdown Report ---
    md_path = Path(save_dir) / "diagnosis_report.md"
    with open(md_path, "w") as f:
        f.write("# 📊 Model Diagnostic Report\n\n")
        f.write("## Data Quality\n")
        f.write(json.dumps(diagnosis['data_quality'], indent=4))
        f.write("\n\n## Distribution Shift\n")
        f.write(json.dumps(diagnosis['distribution_shift'], indent=4))
        f.write("\n\n## Prediction Analysis\n")
        f.write(json.dumps(diagnosis['prediction_analysis'], indent=4))
        f.write("\n\n## Recommendations\n")
        for rec in diagnosis['recommendations']:
            f.write(f"- {rec}\n")

    logger.info(f"Diagnosis saved to {json_path} and {md_path}")
    return diagnosis


def detect_outliers(data: pd.Series, method: str = 'iqr') -> Dict[str, Any]:
    """Detect outliers in data"""
    if method == 'iqr':
        Q1, Q3 = data.quantile(0.25), data.quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        outliers = data[(data < lower) | (data > upper)]
        return {
            'count': int(len(outliers)),
            'percentage': float(len(outliers) / len(data) * 100),
            'lower_bound': float(lower),
            'upper_bound': float(upper),
            'min_outlier': float(outliers.min()) if not outliers.empty else None,
            'max_outlier': float(outliers.max()) if not outliers.empty else None
        }
    return {}
