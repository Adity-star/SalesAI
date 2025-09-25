
#!/usr/bin/env python3
"""
M5 Feature Engineering Runner
============================

Run feature engineering on processed M5 data.

Usage:
    python run_m5_features.py --input data/processed/m5/master.parquet --output data/features/m5/m5_features.parquet
    python run_m5_features.py --input data/processed/m5/master.parquet --output data/features/m5/m5_features.parquet --memory-efficient
    python run_m5_features.py --input data/processed/m5/master.parquet --output data/features/m5/m5_features.parquet --importance
"""

from pathlib import Path 
from datetime import datetime
import argparse
from src.features.m5_feature_pipeline import M5WalmartFeaturePipeline
import pandas as pd
from typing import Dict
import logging
from src.logger import logger
import json

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


def validate_input_data(df: pd.DataFrame) -> bool:
    """Validate input data has required columns."""
    required_cols = ['date', 'store_id', 'item_id', 'sales']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        return False
    
    logger.info(f"Input validation passed. Data shape: {df.shape}")
    logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")
    logger.info(f"Unique time series: {df.groupby(['store_id', 'item_id'],observed=False).ngroups:,}")
    
    return True

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="M5 Walmart Feature Engineering Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --input data/processed/m5/m5_master.parquet --output data/features/m5/m5_features.parquet
  %(prog)s --input data/processed/m5/m5_master.parquet --output data/features/m5/m5_features.parquet --memory-efficient
  %(prog)s --input data/processed/m5/m5_master.parquet --output data/features/m5/m5_features.parquet --importance --debug
        """
    )
    
    parser.add_argument('--input', required=True, 
                       help='Input parquet file (processed M5 data)')
    parser.add_argument('--output', required=True,
                       help='Output parquet file (features)')
    parser.add_argument('--config', 
                       help='Configuration JSON file path')
    parser.add_argument('--memory-efficient', action='store_true',
                       help='Enable memory efficient processing (recommended for large datasets)')
    parser.add_argument('--importance', action='store_true',
                       help='Calculate and save feature importance')
    parser.add_argument('--sample-size', type=int, default=None,
                       help='Process only a sample of the data (for testing)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Setup logging
    global logger
    logger = logging.getLogger(__name__)
    logger.info("Starting M5 Feature Engineering Pipeline")
    logger.info("=" * 50)
    
    try:
        # Validate input file
        input_path = Path(args.input)
        if not input_path.exists():
            logger.error(f"Input file not found: {args.input}")
            return 1
        
        # Load configuration
        if args.config:
            logger.info(f"Loading configuration from {args.config}")
            with open(args.config) as f:
                config = json.load(f)
        else:
            logger.info("Using default M5 configuration")
            config = M5_FEATURE_CONFIG
        
        # Load data
        logger.info(f"Loading data from {args.input}")
        df = pd.read_parquet(args.input)
        
        # Sample data if requested
        if args.sample_size and len(df) > args.sample_size:
            logger.info(f"Sampling {args.sample_size} rows from {len(df)} total rows")
            # Sample by time series to maintain temporal structure
            unique_series = df[['store_id', 'item_id']].drop_duplicates()
            sample_series = unique_series.sample(n=min(args.sample_size//1000, len(unique_series)), 
                                               random_state=42)
            df = df.merge(sample_series, on=['store_id', 'item_id'], how='inner')
            logger.info(f"Sampled data shape: {df.shape}")
        
        # Validate input data
        if not validate_input_data(df):
            return 1
        
        # Initialize feature pipeline
        logger.info("Initializing M5 feature pipeline...")
        pipeline = M5WalmartFeaturePipeline(
            df=df, 
            memory_efficient=args.memory_efficient
        )
        
        # Run feature engineering
        logger.info("Running feature engineering...")
        df_features = pipeline.run()
        
        # Create output directory
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save features
        logger.info(f"Saving features to {args.output}")
        pipeline.save_features(args.output)
        
        # Calculate feature importance if requested
        if args.importance:
            logger.info("Calculating feature importance...")
            try:
                importance_df = pipeline.get_feature_importance()
                importance_path = str(output_path).replace('.parquet', '_importance.csv')
                importance_df.to_csv(importance_path, index=False)
                
                logger.info(f"Feature importance saved to {importance_path}")
                logger.info("Top 10 most important features:")
                for i, row in importance_df.head(10).iterrows():
                    logger.info(f"  {i+1:2d}. {row['feature']:<30} {row['importance']:.4f}")
                    
            except Exception as e:
                logger.warning(f"Failed to calculate feature importance: {e}")
        
        # Summary statistics
        logger.info("=" * 50)
        logger.info("FEATURE ENGINEERING SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Input shape: {df.shape}")
        logger.info(f"Output shape: {df_features.shape}")
        logger.info(f"Features added: {df_features.shape[1] - df.shape[1]}")
        
        feature_cols = [col for col in df_features.columns 
                       if col not in ['date', 'store_id', 'item_id', 'sales', 'd']]
        logger.info(f"Total feature columns: {len(feature_cols)}")
        
        # Feature type breakdown
        feature_types = {
            'lag_features': len([col for col in feature_cols if 'lag' in col]),
            'rolling_features': len([col for col in feature_cols if 'roll' in col]),
            'price_features': len([col for col in feature_cols if 'price' in col]),
            'date_features': len([col for col in feature_cols if any(x in col for x in ['month', 'day', 'year', 'week'])]),
            'snap_features': len([col for col in feature_cols if 'snap' in col]),
            'event_features': len([col for col in feature_cols if 'event' in col]),
            'other_features': len([col for col in feature_cols if not any(x in col for x in ['lag', 'roll', 'price', 'month', 'day', 'year', 'week', 'snap', 'event'])])
        }
        
        logger.info("Feature type breakdown:")
        for feat_type, count in feature_types.items():
            logger.info(f"  {feat_type}: {count}")
        
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info(f"Output file size: {file_size_mb:.1f} MB")
        logger.info("=" * 50)
        
        logger.info("Feature engineering completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        logger.info("Feature engineering interrupted by user")
        return 1
        
    except Exception as e:
        logger.error(f"Feature engineering failed: {e}")
        if args.debug:
            import traceback
            logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    exit(main())