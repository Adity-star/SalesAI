#!/usr/bin/env python3
"""
M5 Dataset Processing Runner
===========================

Memory-optimized runner for M5 Walmart dataset processing.
Handles large-scale data processing with limited memory resources.

Usage:
    python run_m5_pipeline.py --memory-limit 4 --chunk-size 2000
    python run_m5_pipeline.py --validate-only
    python run_m5_pipeline.py --sample-only --sample-size 10000
"""

import sys
import os
import argparse
import logging
from pathlib import Path
from datetime import datetime
import psutil
import gc

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Import your M5 processor
from src.data.production_ingester import M5DatasetProcessor, M5DatasetConfig

def setup_logging(debug: bool = False) -> logging.Logger:
    """Setup logging configuration."""
    log_level = logging.DEBUG if debug else logging.INFO
    
    # Create logs directory
    log_dir = Path("data/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / f"m5_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def check_system_resources():
    """Check available system resources."""
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('.')
    
    logger.info("System Resources:")
    logger.info(f"  Available RAM: {memory.available / 1024**3:.1f} GB / {memory.total / 1024**3:.1f} GB")
    logger.info(f"  Available Disk: {disk.free / 1024**3:.1f} GB / {disk.total / 1024**3:.1f} GB")
    logger.info(f"  CPU Count: {psutil.cpu_count()}")
    
    # Warnings for low resources
    if memory.available / 1024**3 < 2:
        logger.warning(" Low available memory (< 2GB). Consider closing other applications.")
    
    if disk.free / 1024**3 < 5:
        logger.warning("Low available disk space (< 5GB). Consider cleaning up disk space.")
    
    return memory.available / 1024**3

def validate_data_files(config: M5DatasetConfig) -> bool:
    """Validate that required M5 data files exist."""
    logger.info("Validating M5 data files...")
    
    required_files = [
        config.sales_train_path,
        config.prices_path,
        config.calendar_path
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
        else:
            size_mb = Path(file_path).stat().st_size / (1024 * 1024)
            logger.info(f"✓ {Path(file_path).name}: {size_mb:.1f} MB")
    
    if missing_files:
        logger.error("Missing required files:")
        for file_path in missing_files:
            logger.error(f"  - {file_path}")
        logger.error("\nPlease download M5 dataset files from:")
        logger.error("https://www.kaggle.com/competitions/m5-forecasting-accuracy/data")
        return False
    
    logger.info("All required files found")
    return True

def create_sample_dataset(config: M5DatasetConfig, sample_size: int = 10000):
    """Create a smaller sample dataset for testing."""
    logger.info(f"Creating sample dataset with {sample_size} rows...")
    
    try:
        import pandas as pd
        
        # Sample the sales data
        sales_sample = pd.read_csv(config.sales_train_path, nrows=sample_size)
        
        # Get unique items from sample
        sample_items = sales_sample['item_id'].unique()
        sample_stores = sales_sample['store_id'].unique()
        
        # Create sample directory
        sample_dir = Path("data/raw/sample")
        sample_dir.mkdir(parents=True, exist_ok=True)
        
        # Save sample sales
        sample_sales_path = sample_dir / "sales_train_evaluation.csv"
        sales_sample.to_csv(sample_sales_path, index=False)
        
        # Filter prices for sample items and stores
        logger.info("Creating sample prices...")
        prices = pd.read_csv(config.prices_path)
        sample_prices = prices[
            (prices['item_id'].isin(sample_items)) & 
            (prices['store_id'].isin(sample_stores))
        ]
        sample_prices_path = sample_dir / "sell_prices.csv"
        sample_prices.to_csv(sample_prices_path, index=False)
        
        # Copy full calendar (it's small)
        logger.info("Copying calendar...")
        calendar = pd.read_csv(config.calendar_path)
        sample_calendar_path = sample_dir / "calendar.csv"
        calendar.to_csv(sample_calendar_path, index=False)
        
        logger.info(f"Sample dataset created in {sample_dir}")
        logger.info(f"  Sales: {len(sales_sample)} rows")
        logger.info(f"  Prices: {len(sample_prices)} rows")
        logger.info(f"  Calendar: {len(calendar)} rows")
        
        # Return config for sample dataset
        sample_config = M5DatasetConfig(
            sales_train_path=str(sample_sales_path),
            prices_path=str(sample_prices_path),
            calendar_path=str(sample_calendar_path),
            max_memory_gb=config.max_memory_gb,
            chunk_size=config.chunk_size
        )
        
        return sample_config
        
    except Exception as e:
        logger.error(f"Failed to create sample dataset: {e}")
        raise

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="M5 Walmart Dataset Processing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Run full pipeline
  %(prog)s --memory-limit 4                  # Limit memory usage to 4GB
  %(prog)s --chunk-size 2000                 # Use smaller chunks
  %(prog)s --sample-only --sample-size 5000  # Create and process sample
  %(prog)s --validate-only                   # Only validate data quality
        """
    )
    
    parser.add_argument('--memory-limit', type=float, default=4.0,
                       help='Memory limit in GB (default: 4.0)')
    parser.add_argument('--chunk-size', type=int, default=5000,
                       help='Chunk size for processing (default: 5000)')
    parser.add_argument('--validate-only', action='store_true',
                       help='Only validate data quality without full processing')
    parser.add_argument('--sample-only', action='store_true',
                       help='Create and process sample dataset only')
    parser.add_argument('--sample-size', type=int, default=10000,
                       help='Size of sample dataset (default: 10000)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Setup logging
    global logger
    logger = setup_logging(args.debug)
    
    logger.info("Starting M5 Walmart Dataset Processing Pipeline")
    logger.info("=" * 60)
    
    try:
        # Check system resources
        available_memory = check_system_resources()
        
        # Adjust memory limit based on available resources
        if args.memory_limit > available_memory * 0.8:
            recommended_limit = max(1.0, available_memory * 0.6)
            logger.warning(f"Memory limit ({args.memory_limit}GB) is high for available RAM.")
            logger.warning(f"   Recommended: --memory-limit {recommended_limit:.1f}")
            
            # Auto-adjust if not enough memory
            if available_memory < 3:
                args.memory_limit = max(1.0, available_memory * 0.6)
                logger.info(f"Auto-adjusting memory limit to {args.memory_limit:.1f}GB")
        
        # Create configuration
        config = M5DatasetConfig(
            max_memory_gb=args.memory_limit,
            chunk_size=args.chunk_size
        )
        
        # Handle sample dataset creation
        if args.sample_only:
            if not validate_data_files(config):
                return 1
            
            config = create_sample_dataset(config, args.sample_size)
            logger.info("Processing sample dataset...")
        else:
            # Validate full dataset files
            if not validate_data_files(config):
                return 1
        
        # Initialize processor
        processor = M5DatasetProcessor(config)
        
        if args.validate_only:
            logger.info("Running validation-only mode...")
            
            # Load data without full processing
            if not processor.validate_file_integrity():
                return 1
            
            calendar = processor.load_calendar_data()
            prices = processor.load_prices_data()
            
            logger.info("Data validation completed successfully")
            logger.info("Data Overview:")
            logger.info(f"  Calendar: {len(calendar)} days")
            logger.info(f"  Prices: {len(prices)} price points")
            
        else:
            # Run full processing pipeline
            logger.info("Running full processing pipeline...")
            
            df, quality_metrics = processor.run_full_pipeline()
            
            logger.info("Processing completed successfully!")
            logger.info("Final Results:")
            logger.info(f"Dataset shape: {df.shape}")
            logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")
            logger.info(f"Quality score: {quality_metrics.data_completeness_score:.3f}")
            logger.info(f"Time series: {quality_metrics.total_time_series:,}")
            
            # Memory cleanup
            del df
            gc.collect()
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("\nProcessing interrupted by user")
        return 1
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        logger.error("Check the log file for detailed error information")
        return 1
        
    finally:
        # Final memory cleanup
        gc.collect()
        final_memory = psutil.virtual_memory()
        logger.info(f"Final memory usage: {(final_memory.total - final_memory.available) / 1024**3:.1f} GB")

if __name__ == "__main__":
    exit(main())