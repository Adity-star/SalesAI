# 📊 Model Diagnostic Report

## 📦 Data Quality
{
    "train_outliers": {
        "count": 20358,
        "percentage": 8.848495912167008,
        "lower_bound": -1.5,
        "upper_bound": 2.5,
        "min_outlier": 3.0,
        "max_outlier": 91.0
    },
    "val_outliers": {
        "count": 3435,
        "percentage": 10.451212462348252,
        "lower_bound": -1.5,
        "upper_bound": 2.5,
        "min_outlier": 3.0,
        "max_outlier": 67.0
    },
    "test_outliers": {
        "count": 7939,
        "percentage": 12.077096263843252,
        "lower_bound": -1.5,
        "upper_bound": 2.5,
        "min_outlier": 3.0,
        "max_outlier": 75.0
    },
    "n_features": 109
}

## 🔁 Distribution Shift
{
    "train": {
        "mean": 0.8515775203704834,
        "std": 2.6162774562835693
    },
    "val": {
        "mean": 0.9835093021392822,
        "std": 2.533191442489624
    },
    "test": {
        "mean": 1.1463581323623657,
        "std": 2.8792829513549805
    }
}

## 📉 Prediction Analysis
{
    "xgboost": {
        "pred_mean": 1.1458598375320435,
        "pred_std": 2.8734021186828613,
        "residual_mean": 0.0004984021652489901,
        "residual_std": 0.10249742120504379,
        "mape": 1.9689277410507202,
        "extreme_low_count": 2601,
        "extreme_high_count": 0
    },
    "lightgbm": {
        "pred_mean": 1.1469232225106722,
        "pred_std": 2.871270551303252,
        "residual_mean": -0.0005650625982953736,
        "residual_std": 0.13349111892937168,
        "mape": 2.298704801417244,
        "extreme_low_count": 27,
        "extreme_high_count": 0
    },
    "ensemble": {
        "pred_mean": 1.1463913867890854,
        "pred_std": 2.8717699223224558,
        "residual_mean": -3.322687670879836e-05,
        "residual_std": 0.1044292109554549,
        "mape": 1.9469210989147132,
        "extreme_low_count": 1075,
        "extreme_high_count": 0
    }
}

## 💡 Recommendations
- ⚠️ Test set shows distribution shift (mean shift: 34.6%)
- ⚠️ High number of features (109). Consider feature selection.
- 💡 Many zero sales (160711 in training). Consider log-transform or zero-inflated models.
