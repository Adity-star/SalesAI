import pandas as pd
from datetime import timedelta
from src.models.baseline import train_baseline_model
from src.models.metrics import rmse, mae, smape, mase

def backtest(df, feature_cols, target_col, group_cols, horizons, step, initial_train_days=None):
    """
    Perform rolling-origin (walk-forward) backtesting.
    Returns dict: horizon → list of metric dicts
    """
    df = df.sort_values(group_cols + ['date']).reset_index(drop=True)
    results = {}
    min_date = df['date'].min()
    max_date = df['date'].max()

    # Determine starting point for rolling train windows
    if initial_train_days:
        start_date = min_date + timedelta(days=initial_train_days)
    else:
        start_date = min_date + timedelta(days=max(horizons))

    for h in horizons:
        results_h = []
        current_train_end = start_date

        while current_train_end + timedelta(days=h) <= max_date:
            train = df[df['date'] <= current_train_end]
            test = df[(df['date'] > current_train_end) & (df['date'] <= current_train_end + timedelta(days=h))]

            if len(test) < 1:
                break

            # Prepare features and target
            X_train = train[feature_cols].fillna(0)
            y_train = train[target_col]
            X_test = test[feature_cols].fillna(0)
            y_test = test[target_col]

            # Compute naive baseline for MASE:
            # Use the last known value from train for each group as lag-1 baseline
            # Combine train and test for consistent lag calculation
            combined = pd.concat([train, test], ignore_index=True)
            combined = combined.sort_values(group_cols + ['date']).reset_index(drop=True)
            naive = combined.groupby(group_cols)[target_col].shift(1)
            # Extract naive forecast for test period
            naive_test = naive[combined['date'] > current_train_end].values

            # Fill any NaNs in naive forecast (e.g., first records) with training last known values or zero
            if pd.isna(naive_test).any():
                # fallback to last train value per group
                last_train_values = train.groupby(group_cols)[target_col].last()
                naive_test_filled = []
                for idx, row in test.iterrows():
                    group = tuple(row[col] for col in group_cols)
                    val = last_train_values.get(group, np.nan)
                    naive_test_filled.append(val if not pd.isna(val) else 0)
                naive_test = pd.Series(naive_test).fillna(pd.Series(naive_test_filled)).values

            # Train model and get predictions
            model, preds, _ = train_baseline_model(X_train, y_train, X_test, y_test)

            # Compute metrics
            m_rmse = rmse(y_test, preds)
            m_mae = mae(y_test, preds)
            m_smape = smape(y_test, preds)
            m_mase = mase(y_test, preds, naive_test)

            results_h.append({
                'horizon': h,
                'train_end': current_train_end.strftime('%Y-%m-%d'),
                'rmse': float(m_rmse),
                'mae': float(m_mae),
                'smape': float(m_smape),
                'mase': float(m_mase)
            })

            current_train_end = current_train_end + timedelta(days=step)

        results[h] = results_h

    return results
