#!/usr/bin/env python3
"""
Prediction Pipeline Runner
===========================

Run sales forecasting predictions on new data.

Usage:
    python run_prediction.py --input data/new_data.csv --model-version latest
    python run_prediction.py --input data/new_data.csv --explain --save-output
"""

import sys
import argparse
import pandas as pd
import logging
from pathlib import Path
from datetime import datetime

from src.logger import logger
from src.exception import CustomException
from src.prediction_pipeline import PredictionPipeline

logger = logging.getLogger(__name__)


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Sales Forecasting Prediction Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --input data/input.csv --model-version latest
  %(prog)s --input data/input.parquet --model xgboost --explain
  %(prog)s --input data/input.csv --batch-size 500 --save-output
  %(prog)s --input data/input.csv --artifacts-dir /path/to/models --debug
        """
    )
    
    parser.add_argument('--input', required=True,
                       help='📂 Input data file (CSV or Parquet)')
    parser.add_argument('--model-version', default='latest',
                       help='🏷️ Model version to use (default: latest)')
    parser.add_argument('--artifacts-dir',
                       help='📁 Directory containing model artifacts')
    parser.add_argument('--model', default='ensemble',
                       choices=['xgboost', 'lightgbm', 'ensemble'],
                       help='🧠 Model to use for predictions')
    parser.add_argument('--explain', action='store_true',
                       help='📊 Generate SHAP explanations')
    parser.add_argument('--batch-size', type=int, default=1000,
                       help='📦 Batch size for processing')
    parser.add_argument('--save-output', action='store_true',
                       help='💾 Save predictions to file')
    parser.add_argument('--output-dir', default='predictions',
                       help='📁 Output directory for predictions')
    parser.add_argument('--debug', action='store_true',
                       help='🐞 Enable debug logging')
    
    args = parser.parse_args()
    
    logger.info("🚀 Starting Sales Forecasting Prediction Pipeline")
    logger.info("=" * 60)
    
    try:
        # Validate input file
        input_path = Path(args.input)
        if not input_path.exists():
            logger.error(f"❌ Input file not found: {args.input}")
            raise CustomException(f"Input file not found: {args.input}", sys)
        
        # Load input data
        logger.info(f"📥 Loading input data from {args.input}...")
        if args.input.endswith('.parquet'):
            input_df = pd.read_parquet(args.input)
        else:
            input_df = pd.read_csv(args.input)
        
        logger.info(f"✅ Loaded {len(input_df)} rows with {len(input_df.columns)} columns")
        
        # Initialize prediction pipeline
        logger.info(f"🔧 Initializing prediction pipeline (version: {args.model_version})...")
        pipeline = PredictionPipeline(
            model_version=args.model_version,
            artifact_dir=args.artifacts_dir
        )
        
        # Load artifacts
        pipeline.load_artifacts()
        
        # Validate input
        logger.info("🔍 Validating input data...")
        is_valid, errors = pipeline.validate_input(input_df)
        if not is_valid:
            logger.error("❌ Input validation failed:")
            for error in errors:
                logger.error(f"  - {error}")
            raise CustomException("Input validation failed", sys)
        
        logger.info("✅ Input validation passed")
        
        # Generate predictions
        if len(input_df) > args.batch_size:
            logger.info(f"📦 Processing in batches of {args.batch_size}...")
            result_df = pipeline.predict_batch(
                input_df,
                model_name=args.model,
                batch_size=args.batch_size
            )
            predictions = result_df["predicted_sales"].values
        else:
            logger.info(f"🔮 Generating predictions with {args.model} model...")
            results = pipeline.predict(
                input_df,
                model_name=args.model,
                explain=args.explain
            )
            predictions = results["predictions"]
            
            # Print explanations if requested
            if args.explain and "explanations" in results:
                logger.info("📊 SHAP Explanations:")
                explanations = results["explanations"]
                
                if "global_importance" in explanations:
                    logger.info("\n🌟 Top 10 Most Important Features:")
                    for i, feature in enumerate(explanations["global_importance"][:10], 1):
                        logger.info(f"  {i:2d}. {feature['feature']:<30} {feature['importance']:.4f}")
        
        # Summary statistics
        logger.info("=" * 60)
        logger.info("📈 PREDICTION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"🔢 Total predictions: {len(predictions)}")
        logger.info(f"📊 Prediction statistics:")
        logger.info(f"  Mean:   {predictions.mean():.2f}")
        logger.info(f"  Median: {np.median(predictions):.2f}")
        logger.info(f"  Std:    {predictions.std():.2f}")
        logger.info(f"  Min:    {predictions.min():.2f}")
        logger.info(f"  Max:    {predictions.max():.2f}")
        
        # Save output if requested
        if args.save_output:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = output_dir / f"predictions_{args.model}_{timestamp}.csv"
            
            output_df = input_df.copy()
            output_df["predicted_sales"] = predictions
            output_df["model_used"] = args.model
            output_df["prediction_timestamp"] = datetime.now()
            
            output_df.to_csv(output_file, index=False)
            logger.info(f"💾 Predictions saved to: {output_file}")
        
        logger.info("=" * 60)
        logger.info("🎉 Prediction pipeline completed successfully!")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("🛑 Prediction interrupted by user")
        return 1
        
    except Exception as e:
        logger.error(f"❌ Prediction failed: {e}")
        if args.debug:
            import traceback
            logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    import numpy as np
    exit(main())