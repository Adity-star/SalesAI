
import logging
import pandas as pd
from src.logger import logger
from src.backtest.backtesting import BacktestEngine
from src.entity.backtest_entity import BacktestConfig

# -------------------------------
# Main Runner Function
# -------------------------------

def run_m5_backtest(data_path: str, config_path: str = None, 
                   output_dir: str = "backtest_results") -> pd.DataFrame:
    """
    Main function to run M5 backtesting.
    
    Args:
        data_path: Path to processed M5 features data
        config_path: Path to backtest configuration (optional)
        output_dir: Output directory for results
    
    Returns:
        Summary DataFrame with results
    """
    
    logger.info(f"📥 Loading data from {data_path} ...")
    df = pd.read_parquet(data_path)
    logger.info(f"✅ Data loaded with shape {df.shape}")
    
    # Load or create configuration
    if config_path:
        with open(config_path) as f:
            import yaml
            config_dict = yaml.safe_load(f)
        config = BacktestConfig(**config_dict)
        logger.info(f"⚙️ Configuration loaded from {config_path}")
    else:
        config = BacktestConfig()
        logger.info(f"⚙️ Default configuration initialized")
    
    # Initialize and run backtesting
    logger.info(f"🚦 Initializing backtest engine with output directory: {output_dir}")
    backtest_engine = BacktestEngine(config, output_dir)
    
    logger.info(f"🏁 Starting backtesting...")
    results_df = backtest_engine.run_backtest(df)
    logger.info(f"🏁 Backtesting completed. Results obtained with shape {results_df.shape}")
    
    return results_df

if __name__ == "__main__":
    import argparse
    import sys
    from pathlib import Path
    
    parser = argparse.ArgumentParser(
        description="M5 Walmart Backtesting Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --input data/features/m5/m5_features.parquet
  %(prog)s --input data/features/m5/m5_features.parquet --config configs/backtest.yaml
  %(prog)s --input data/features/m5/m5_features.parquet --output-dir results/backtest_v1
  %(prog)s --input data/features/m5/m5_features.parquet --models lightgbm seasonal_naive
        """
    )
    
    parser.add_argument('--input', required=True,
                       help='📂 Input parquet file with M5 features')
    parser.add_argument('--config',
                       help='⚙️ Backtest configuration YAML file')
    parser.add_argument('--output-dir', default='backtest_results',
                       help='📁 Output directory for results')
    parser.add_argument('--models', nargs='+', 
                       choices=['lightgbm'],
                       help='🧠 Models to evaluate')
    parser.add_argument('--horizons', type=int, nargs='+',
                       help='⏳ Forecast horizons in days')
    parser.add_argument('--quick-test', action='store_true',
                       help='⚡ Run quick test with limited data')
    parser.add_argument('--debug', action='store_true',
                       help='🐞 Enable debug logging')
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(exist_ok=True)
    
    logger = logging.getLogger(__name__)
    
    try:
        # Validate input file
        if not Path(args.input).exists():
            logger.error(f"❌ Input file not found: {args.input}")
            sys.exit(1)
        
        logger.info(f"📥 Loading data from {args.input}")
        df = pd.read_parquet(args.input)
        
        # Quick test mode - sample data
        if args.quick_test:
            logger.info("⚡ Running in quick test mode - sampling data")
            unique_series = df[['store_id', 'item_id']].drop_duplicates()
            sample_series = unique_series.sample(n=min(100, len(unique_series)), random_state=42)
            df = df.merge(sample_series, on=['store_id', 'item_id'], how='inner')
            logger.info(f"📊 Sampled data shape: {df.shape}")
        
        # Create configuration
        if args.config and Path(args.config).exists():
            import yaml
            with open(args.config) as f:
                config_dict = yaml.safe_load(f)
            config = BacktestConfig(**config_dict)
            logger.info(f"⚙️ Configuration loaded from {args.config}")
        else:
            config = BacktestConfig()
            logger.info(f"⚙️ Default configuration initialized")
        
        # Override config with command line arguments
        if args.models:
            config.models_to_evaluate = args.models
            logger.info(f"🧠 Models to evaluate set to: {args.models}")
        if args.horizons:
            config.horizons = args.horizons
            logger.info(f"⏳ Forecast horizons set to: {args.horizons}")
        
        # Quick test modifications
        if args.quick_test:
            config.horizons = [7, 14]  
            config.models_to_evaluate = ['lightgbm','seasonal_naive','prophet'] 
            config.test_start_date = "2016-04-01"  
            logger.info(f"⚡ Quick test config applied: horizons={config.horizons}, models={config.models_to_evaluate}, test_start_date={config.test_start_date}")
        
        logger.info(f"🔍 Configuration summary: {len(config.horizons)} horizons, {len(config.models_to_evaluate)} models")
        
        # Run backtesting
        backtest_engine = BacktestEngine(config, args.output_dir)
        results_df = backtest_engine.run_backtest(df)
        
        # Print summary
        print("\n" + "=" * 60)
        print("🚀 BACKTESTING RESULTS SUMMARY 🚀")
        print("=" * 60)
        print(results_df.round(4).to_string(index=False))
        print("=" * 60)
        
        # Find best model
        if not results_df.empty:
            best_rmse_idx = results_df['rmse'].idxmin()
            best_model = results_df.iloc[best_rmse_idx]

            best_rmse_val = float(best_model['rmse']) if not isinstance(best_model['rmse'], float) else best_model['rmse']

            print(f"\n🏅 Best RMSE: {best_model['model']} (H{best_model['horizon']}) = {best_rmse_val:.4f}")
            
            best_mase_idx = results_df['mase'].idxmin()
            best_model_mase = results_df.iloc[best_mase_idx]

            best_mase_val = float(best_model_mase['mase']) if not isinstance(best_model_mase['mase'], float) else best_model_mase['mase']

            print(f"🏅 Best MASE: {best_model_mase['model']} (H{best_model_mase['horizon']}) = {best_mase_val:.4f}")
        
        logger.info(f"🎉 Backtesting completed successfully! Results saved to {args.output_dir}")
        
    except KeyboardInterrupt:
        logger.info("🛑 Backtesting interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Backtesting failed: {e}")
        if args.debug:
            import traceback
            logger.error(traceback.format_exc())
        sys.exit(1)
