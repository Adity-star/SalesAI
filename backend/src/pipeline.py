import os
import pandas as pd
from datetime import datetime
import joblib 
import sys 

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.features.m5_feature_pipeline import M5FeaturePipeline

import run_ingestion
import run_features
import run_backtest
from src.logger import logger




# def main():
#     # Load config
#     config = load_config()
#     data_path = config.get("data", {}).get("data_path", "data\raw\rossmann-store-salestrain.csv")
#     target_col = config.get("model", {}).get("target_col", "sales")
#     group_cols = config.get("model", {}).get("group_cols", ["store_id", "product_id"])
#     date_col = config.get("features", {}).get("date_col", "date")
#     feature_cols = config.get("features", {}).get("feature_cols", [])  # Will fill later after FE
#     horizons = config.get("backtest", {}).get("horizons", [7, 14, 28])
#     step = config.get("backtest", {}).get("step", 7)
#     initial_train_days = config.get("backtest", {}).get("initial_train_days", 90)
#     model_output_path = config.get("model", {}).get("model_output_path", "models/lgbm_baseline.pkl")

#     # Load data
#     print(f"Loading data from {data_path}...")
#     df = pd.read_csv(data_path, parse_dates=[date_col])

#     # Feature Engineering
#     print("Running feature engineering...")
#     pipeline = FeaturePipeline(df, target_col=target_col, group_cols=group_cols)
#     df = pipeline.run()

#     # Add advanced features if enabled
#     if config.get("features", {}).get("advanced_features_enabled", False):
#         df = pipeline.create_advanced_features(df, target_col=target_col, date_col=date_col, group_cols=group_cols)

#     # Define final feature columns (excluding target, date, group cols)
#     exclude_cols = [target_col, date_col] + group_cols
#     feature_cols = [col for col in df.columns if col not in exclude_cols]

#     print(f"Using {len(feature_cols)} features for training.")

#     # Backtesting and training
#     print("Starting backtesting and training...")
#     results = backtest(df, feature_cols, target_col, group_cols, horizons, step, initial_train_days)

#     # Summarize backtest results
#     for h, metrics_list in results.items():
#         rmses = [m['rmse'] for m in metrics_list]
#         maes = [m['mae'] for m in metrics_list]
#         smapes = [m['smape'] for m in metrics_list]
#         mases = [m['mase'] for m in metrics_list]
#         print(f"Horizon: {h} days | Avg RMSE: {sum(rmses)/len(rmses):.4f} | "
#               f"Avg MAE: {sum(maes)/len(maes):.4f} | Avg sMAPE: {sum(smapes)/len(smapes):.2f}% | Avg MASE: {sum(mases)/len(mases):.4f}")

#     # Train final model on entire data
#     print("Training final model on full data...")
#     X = df[feature_cols].fillna(0)
#     y = df[target_col]

#     model, preds, metric = train_baseline_model(X, y, X, y)
#     print(f"Final model training RMSE on full data: {metric:.4f}")

#     # Save model
#     os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
#     joblib.dump(model, model_output_path)
#     print(f"Saved final model to {model_output_path}")

# if __name__ == "__main__":
#     main()


def run_full_pipeline(raw_data_path, config_path=None, output_dir="backtest_results"):
    """
    Run full pipeline: ingestion -> feature engineering -> backtesting.
    """
    logger.info("🚀 Starting full pipeline...")

    try:
        # Step 1: Data ingestion
        logger.info("📥 Running data ingestion...")
        ingested_data_path = run_ingestion(raw_data_path)  # Adjust return as needed
        logger.info(f"✅ Data ingestion completed. Output saved at: {ingested_data_path}")

        # Step 2: Feature engineering
        logger.info("⚙️ Running feature engineering...")
        feature_data_path = run_features(ingested_data_path)
        logger.info(f"✅ Feature engineering completed. Features saved at: {feature_data_path}")

        # Step 3: Backtesting
        logger.info("📊 Running backtesting...")
        results_df = run_backtest(feature_data_path, config_path, output_dir)
        logger.info("✅ Backtesting completed successfully!")
        
        print("\n🎉 Full pipeline completed! Backtest results:")
        print(results_df.head())
        return results_df

    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Run full M5 pipeline: ingestion, features, backtest")

    parser.add_argument('--raw-data', required=True, help='Path to raw input data')
    parser.add_argument('--config', default=None, help='Backtest config YAML file (optional)')
    parser.add_argument('--output-dir', default='backtest_results', help='Output directory for backtest results')

    args = parser.parse_args()

    Path(args.output_dir).mkdir(exist_ok=True)

    run_full_pipeline(args.raw_data, args.config, args.output_dir)

