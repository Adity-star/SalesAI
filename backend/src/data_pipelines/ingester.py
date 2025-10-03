"""
Sales Forecasting Pipeline for Single Dataset
============================================================


Dataset: Forecasting (Walmart)
- 42,840 time series
- 5 years of data (2011-2016)  
- Hierarchical structure: item -> dept -> category -> store -> state
- Rich external data: prices, promotions, calendar events
"""
import gc
import sys
import time
import psutil
import json
import pandas as pd
import numpy as np
from pathlib import Path
import logging

from typing import Dict, Tuple
from dataclasses import asdict
from datetime import datetime

import pyarrow as pa
import pyarrow.parquet as pq
from contextlib import contextmanager
from src.logger import logger
from src.exception import CustomException

from src.entity.ingest_entity import DataQualityMetrics,DatasetConfig

logger = logging.getLogger(__name__)



# -------------------------------
# Memory Management
# -------------------------------

class MemoryManager:
    """memory management for large dataset processing."""
    
    def __init__(self, max_memory_gb: float = 8.0):
        self.max_memory_gb = max_memory_gb
        self.max_memory_bytes = max_memory_gb * 1024**3
        
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'rss_gb': memory_info.rss / 1024**3,
            'vms_gb': memory_info.vms / 1024**3,
            'percent': process.memory_percent(),
            'available_gb': psutil.virtual_memory().available / 1024**3
        }
    
    def check_memory_usage(self):
        """Check if memory usage is within limits."""
        memory_stats = self.get_memory_usage()
        
        if memory_stats['rss_gb'] > self.max_memory_gb:
            logger.warning(f"⚠️ Memory usage ({memory_stats['rss_gb']:.2f} GB) exceeds limit ({self.max_memory_gb} GB)")
            gc.collect()  # Force garbage collection
            
        logger.debug(f"💾 Memory usage: {memory_stats['rss_gb']:.2f} GB ({memory_stats['percent']:.1f}%)")
    
    @contextmanager
    def memory_monitor(self, operation_name: str):
        """Context manager to monitor memory usage during operations."""
        start_memory = self.get_memory_usage()
        start_time = time.time()
        
        logger.info(f"🚀 Starting {operation_name} - Memory: {start_memory['rss_gb']:.2f} GB")
        
        try:
            yield
        finally:
            end_memory = self.get_memory_usage()
            duration = time.time() - start_time
            memory_delta = end_memory['rss_gb'] - start_memory['rss_gb']
            
            logger.info(f"✅ Completed {operation_name} in {duration:.2f}s - "
                        f"Memory: {end_memory['rss_gb']:.2f} GB ({memory_delta:+.2f} GB)")
            
            self.check_memory_usage()



#------------------------------
#Downcast data
#------------------------------

def downcast(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        col_type = df[col].dtype

        if pd.api.types.is_integer_dtype(col_type):
            df[col] = pd.to_numeric(df[col], downcast='integer')

        elif pd.api.types.is_float_dtype(col_type):
            df[col] = pd.to_numeric(df[col], downcast='float')

        elif pd.api.types.is_object_dtype(col_type):
            if col == 'date':
                df[col] = pd.to_datetime(df[col], format='%Y-%m-%d', errors='coerce')
            else:
                # Check if column is numeric-like
                converted = pd.to_numeric(df[col], errors='coerce')
                num_missing = converted.isna().sum()
                total = len(df[col])

                if num_missing / total < 0.05:  # less than 5% non-convertible treated as numeric
                    df[col] = converted
                else:
                    df[col] = df[col].astype('category')
                    logger.info(f"📊 Column '{col}' converted to category")

    return df

def downcast_with_stats(df: pd.DataFrame, name: str = "DataFrame") -> pd.DataFrame:
    start_mem = df.memory_usage(deep=True).sum() / 1024**2
    df = downcast(df)
    end_mem = df.memory_usage(deep=True).sum() / 1024**2
    print(f"🔧 {name} memory usage: {start_mem:.2f} MB → {end_mem:.2f} MB ({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)")
    return df


# -------------------------------
# Custom JSON Encoder
# -------------------------------

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

# -------------------------------
# M5 Dataset Processor
# -------------------------------

class DatasetProcessor:
    """Processor for M5 Walmart dataset."""
    
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.memory_manager = MemoryManager(config.max_memory_gb)
        self.quality_metrics = None
        
        # Setup output directories
        self.setup_directories()
        
        logger.info("🚀 M5DatasetProcessor initialized for production processing")
    
    def setup_directories(self):
        """Create required directory structure."""
        dirs = [
            "data/processed/m5", 
            "data/features/m5",
            "data/quality/m5",
        ]
        
        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def validate_file_integrity(self) -> bool:
        """Validate that all required M5 files exist and are readable."""
        logger.info("🔍 Validating M5 dataset file integrity...")
        
        required_files = [
            self.config.sales_train_path,
            self.config.prices_path,
            self.config.calendar_path
        ]
        
        for file_path in required_files:
            path = Path(file_path)
            if not path.exists():
                logger.error(f"❌ Required file missing: {file_path}")
                return False
            
            # Check file size (M5 files should be substantial)
            file_size_mb = path.stat().st_size / (1024 * 1024)
            logger.info(f"📁 File {path.name}: {file_size_mb:.1f} MB")
            
            if file_size_mb < 1:  # Basic sanity check
                logger.warning(f"⚠️ File {file_path} seems too small ({file_size_mb:.1f} MB)")
        
        logger.info("✅ File integrity validation passed")
        return True
    
    def load_calendar_data(self) -> pd.DataFrame:
        """Load and validate calendar data with comprehensive error handling."""
        logger.info("📅 Loading M5 calendar data...")
        
        with self.memory_manager.memory_monitor("calendar_loading"):
            try:
                calendar_dtypes = {
                    'date': 'str',
                    'wm_yr_wk': 'int32',
                    'weekday': 'str',  # keep as str for mapping
                    'd': 'str',
                    'event_name_1': 'str',
                    'event_type_1': 'str', 
                    'event_name_2': 'str',
                    'event_type_2': 'str',
                    'snap_CA': 'int8',
                    'snap_TX': 'int8',
                    'snap_WI': 'int8'
                }
                
                calendar = pd.read_csv(self.config.calendar_path, dtype=calendar_dtypes)
                calendar = downcast_with_stats(calendar, name="Calendar")

                # Convert date column once
                calendar['date'] = pd.to_datetime(calendar['date'], errors='coerce')

                # Map weekday strings to numbers BEFORE any numeric conversion
                weekday_map = {
                    'Sunday': 7,
                    'Monday': 1,
                    'Tuesday': 2,
                    'Wednesday': 3,
                    'Thursday': 4,
                    'Friday': 5,
                    'Saturday': 6
                }
                calendar['weekday'] = calendar['weekday'].map(weekday_map).astype('int8')

                # Validate calendar length
                expected_days = (pd.to_datetime('2016-05-22') - pd.to_datetime('2011-01-29')).days + 1
                if len(calendar) != expected_days:
                    logger.warning(f"⚠️ Expected {expected_days} calendar days, found {len(calendar)}")
                
                # Date features
                calendar['year'] = calendar['date'].dt.year
                calendar['month'] = calendar['date'].dt.month  
                calendar['day'] = calendar['date'].dt.day
                calendar['quarter'] = calendar['date'].dt.quarter
                calendar['week_of_year'] = calendar['date'].dt.isocalendar().week
                
                # Weekend flag (assuming Sat=6, Sun=7 as weekend)
                calendar['is_weekend'] = calendar['weekday'].isin([6,7]).astype('int8')
                
                # Event processing
                calendar['has_event'] = (
                    calendar['event_name_1'].notna() | calendar['event_name_2'].notna()
                ).astype('int8')
                
                # SNAP benefits
                calendar['snap_any'] = (
                    (calendar['snap_CA'] == 1) |
                    (calendar['snap_TX'] == 1) |
                    (calendar['snap_WI'] == 1)
                ).astype('int8')
                
                logger.info(f"📊 Calendar loaded: {len(calendar)} days, {calendar['date'].min()} to {calendar['date'].max()}")
                
                return calendar

            
            except Exception as e:
                logger.error(f"Failed to load calendar data: {e}")
                raise CustomException(e, sys) from e

    def load_prices_data(self) -> pd.DataFrame:
        """Load and validate pricing data."""
        logger.info("💰 Loading M5 pricing data...")
        
        with self.memory_manager.memory_monitor("prices_loading"):
            try:
                # Load with memory-efficient dtypes
                prices_dtypes = {
                    'store_id': 'str',
                    'item_id': 'str', 
                    'wm_yr_wk': 'int32',
                    'sell_price': 'float32'
                }
                
                prices = pd.read_csv(self.config.prices_path, dtype=prices_dtypes)
                prices = downcast_with_stats(prices, name="Prices")

                if self.config.load_fraction:
                    n_rows = int(len(prices) * self.config.load_fraction)
                    prices = prices.iloc[:n_rows]  # Top rows
                    logger.info(f"📉 Using only {n_rows} rows ({self.config.load_fraction:.0%}) of prices data.")
                    
                # Data quality checks
                null_prices = prices['sell_price'].isna().sum()
                if null_prices > 0:
                    logger.warning(f"⚠️ Found {null_prices} null prices ({null_prices/len(prices)*100:.2f}%)")
                
                negative_prices = (prices['sell_price'] < 0).sum()
                if negative_prices > 0:
                    logger.error(f"❌ Found {negative_prices} negative prices - this is critical!")
                    # In production, you might want to raise an exception or alert
                
                zero_prices = (prices['sell_price'] == 0).sum()
                if zero_prices > 0:
                    logger.warning(f"⚠️ Found {zero_prices} zero prices ({zero_prices/len(prices)*100:.2f}%)")
                
                # Price statistics
                price_stats = prices['sell_price'].describe()
                logger.info(f"📊 Price statistics: min=${price_stats['min']:.2f}, "
                            f"max=${price_stats['max']:.2f}, mean=${price_stats['mean']:.2f}")
                
                logger.info(f"✅ Prices loaded: {len(prices)} price points for {prices['item_id'].nunique()} items")
                
                return prices
                
            except Exception as e:
                logger.error(f"❗ Failed to load prices data: {e}")
                raise CustomException(e, sys) from e

    def load_sales_data_chunked(self) -> pd.DataFrame:
        """Load sales data in chunks to handle memory efficiently."""
        logger.info("📦 Loading M5 sales data (chunked processing)...")
        
        with self.memory_manager.memory_monitor("sales_loading"):
            try:
                # First, get the column structure
                sample_df = pd.read_csv(self.config.sales_train_path, nrows=1)
                
                # Identify day columns (d_1, d_2, etc.)
                day_columns = [col for col in sample_df.columns if col.startswith('d_')]
                id_columns = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']

                total_rows = sum(1 for _ in open(self.config.sales_train_path)) - 1  # exclude header
                
                # Determine target rows based on load_fraction and optional max_rows
                if hasattr(self.config, 'max_rows') and self.config.max_rows is not None:
                    target_rows = min(int(total_rows * self.config.load_fraction), self.config.max_rows)
                else:
                    target_rows = int(total_rows * self.config.load_fraction)

                chunk_size = self.config.chunk_size

                total_chunks = (target_rows + chunk_size - 1) // chunk_size
                logger.info(f"📊 Total rows in sales: {total_rows}")
                logger.info(f"⏳ Loading {self.config.load_fraction:.0%} of dataset → {target_rows} rows in ~{total_chunks} chunks (chunk size={chunk_size})")
                
                # Load in chunks for memory efficiency
                chunks = []
                rows_loaded = 0
                chunk_iterator = pd.read_csv(
                    self.config.sales_train_path,
                    chunksize=self.config.chunk_size,
                    dtype={col: 'str' for col in id_columns}
                )
                
                for i, chunk in enumerate(chunk_iterator, start=1):
                    if rows_loaded >= target_rows:
                        logger.info(f"🛑 Reached target rows ({target_rows}), stopping further loading")
                        break

                    if rows_loaded + len(chunk) > target_rows:
                        chunk = chunk.iloc[:target_rows - rows_loaded]

                    for col in day_columns:
                        chunk.loc[:,col] = pd.to_numeric(chunk[col], downcast='integer')

                    chunks.append(chunk)
                    rows_loaded += len(chunk)

                    logger.info(f"✅ Loaded chunk {i}/{total_chunks}, rows loaded so far: {rows_loaded}")

                    self.memory_manager.check_memory_usage()

                sales_wide = pd.concat(chunks, ignore_index=True)
                del chunks
                gc.collect()

                logger.info(f"🎉 Completed loading sales data: {len(sales_wide)} items × {len(day_columns)} days")

                return sales_wide, day_columns

            except Exception as e:
                logger.error(f"❗ Failed to load sales data: {e}")
                raise CustomException(e, sys) from e

    def create_master_dataset(
        self,
        sales_wide: pd.DataFrame,
        calendar: pd.DataFrame,
        prices: pd.DataFrame,
        chunk_size: int = 1000000
    ) -> Path:
        """
        Chunked version of master dataset creation with debug and fix for merge key mismatches.
        """
        import gc

        logger.info("🛠️ Creating master dataset in chunks with enhanced key alignment and debug...")

        output_dir = Path("data/processed/m5/master_chunks")
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            max_chunks = 3
            total_rows = len(sales_wide)
            total_chunks = (total_rows + chunk_size - 1) // chunk_size
            logger.info(f"📐 Total rows: {total_rows}, Chunk size: {chunk_size}, Total chunks: {total_chunks}")

            chunk_paths = []

            for i, start in enumerate(range(0, total_rows, chunk_size), 1):
                if i > max_chunks:
                    logger.info(f"🛑 Reached max chunk limit ({max_chunks}), stopping further processing")
                    break

                end = min(start + chunk_size, total_rows)
                logger.info(f"🔄 Processing chunk {i}/{total_chunks}: rows {start} to {end}")
                chunk = sales_wide.iloc[start:end].copy()

                # Melt the chunk
                id_vars = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']

                sales_long = chunk.melt(
                    id_vars=id_vars,
                    var_name='d',
                    value_name='sales'
                )

                # Step 1: Align calendar 'd' categories with sales_long 'd'
                sales_long['d'] = sales_long['d'].astype('category')
                calendar['d'] = calendar['d'].astype('category')
                calendar['d'] = calendar['d'].cat.set_categories(sales_long['d'].cat.categories)

                # Step 2: Merge with calendar
                df = sales_long.merge(calendar, on='d', how='left')

                # Step 5: Merge with prices
                df = df.merge(
                    prices[['store_id', 'item_id', 'wm_yr_wk', 'sell_price']],
                    on=['store_id', 'item_id', 'wm_yr_wk'],
                    how='left'
                )

                # Compute revenue
                df['sales'] = pd.to_numeric(df['sales'], errors='coerce').fillna(0).astype('float32')
                df['revenue'] = (df['sales'] * df['sell_price']).astype('float32')

                # Optional: downcast to reduce memory footprint
                df = downcast_with_stats(df, name=f"Master Chunk [{start}-{end}]")

                # Save chunk to disk
                out_path = output_dir / f"master_chunk_{start}_{end}.parquet"
                df.to_parquet(out_path, index=False)
                chunk_paths.append(out_path)

                logger.info(f"💾 Saved chunk: {out_path} ({df.shape[0]} rows)")

                del chunk, sales_long, df
                gc.collect()

            logger.info(f"🎯 All master chunks saved to {output_dir}")
            return output_dir

        except Exception as e:
            logger.error(f"❗ Failed during chunked master dataset creation: {e}")
            raise CustomException(e, sys) from e


    def load_master_dataset(self, master_data_dir: Path) -> pd.DataFrame:
        import glob
        parquet_files = sorted(glob.glob(str(master_data_dir / "*.parquet")))
        if not parquet_files:
            logger.error(f"❌ No parquet files found in {master_data_dir}")
            raise FileNotFoundError(f"No parquet files found in {master_data_dir}")
        logger.info(f"📂 Loading master dataset from {len(parquet_files)} parquet files...")
        dfs = [pd.read_parquet(f) for f in parquet_files]
        logger.info(f"✅ Master dataset loaded with {len(dfs)} chunks")
        return pd.concat(dfs, ignore_index=True)


    def validate_data_quality(self, df: pd.DataFrame) -> DataQualityMetrics:
        logger.info("🔍 Performing comprehensive data quality validation...")
        with self.memory_manager.memory_monitor("data_quality_validation"):
            try:
                total_obs = len(df)
                unique_series = df.groupby(['store_id', 'item_id'], observed=False).ngroups

                missing_sales = df['sales'].isna().sum()
                missing_sales_pct = (missing_sales / total_obs) * 100 if total_obs else 0

                zero_sales = (df['sales'] == 0).sum()
                zero_sales_pct = (zero_sales / total_obs) * 100 if total_obs else 0

                negative_sales = (df['sales'] < 0).sum()

                missing_prices = df['sell_price'].isna().sum()
                price_coverage_pct = ((total_obs - missing_prices) / total_obs) * 100 if total_obs else 0

                missing_dates = df['date'].isna().sum()
                calendar_coverage_pct = ((total_obs - missing_dates) / total_obs) * 100 if total_obs else 0

                completeness_score = (
                    (100 - missing_sales_pct) * 0.4 +
                    price_coverage_pct * 0.3 +
                    calendar_coverage_pct * 0.3
                ) / 100

                temporal_gaps = 0
                sample_series = df.groupby(['store_id', 'item_id'], observed=False).head(1000)
                for (store, item), group in sample_series.groupby(['store_id', 'item_id'], observed=False):
                    date_diff = group['date'].diff().dt.days
                    gaps = (date_diff > 1).sum()
                    temporal_gaps += gaps

                temporal_consistency_score = max(0, 1 - (temporal_gaps / unique_series)) if unique_series else 0

                hierarchy_issues = 0
                item_dept = df.groupby('item_id', observed=False)['dept_id'].nunique()
                hierarchy_issues += (item_dept > 1).sum()

                dept_cat = df.groupby('dept_id', observed=False)['cat_id'].nunique()
                hierarchy_issues += (dept_cat > 1).sum()

                hierarchical_consistency_score = max(
                    0,
                    1 - (hierarchy_issues / (df['item_id'].nunique() + df['dept_id'].nunique()))
                ) if (df['item_id'].nunique() + df['dept_id'].nunique()) > 0 else 0

                series_sales_counts = df.groupby(['store_id', 'item_id'], observed=False)['sales'].count()
                valid_series = (series_sales_counts >= self.config.min_sales_per_item).sum()

                series_zero_pcts = df.groupby(['store_id', 'item_id'], observed=False)['sales'].apply(lambda x: (x == 0).mean())
                valid_series_zero = (series_zero_pcts <= self.config.max_zero_days_pct).sum()

                valid_time_series = min(valid_series, valid_series_zero)

                quality_metrics = DataQualityMetrics(
                    total_time_series=unique_series,
                    valid_time_series=valid_time_series,
                    total_observations=total_obs,
                    missing_sales_pct=missing_sales_pct,
                    zero_sales_pct=zero_sales_pct,
                    negative_sales_count=negative_sales,
                    price_coverage_pct=price_coverage_pct,
                    calendar_coverage_pct=calendar_coverage_pct,
                    data_completeness_score=completeness_score,
                    temporal_consistency_score=temporal_consistency_score,
                    hierarchical_consistency_score=hierarchical_consistency_score
                )

                self.quality_metrics = quality_metrics

                logger.info("============================================================")
                logger.info("📊 DATA QUALITY SUMMARY")
                logger.info("============================================================")
                logger.info(f"🆔 Total time series: {unique_series:,}")
                logger.info(f"✅ Valid time series: {valid_time_series:,} ({valid_time_series / unique_series * 100:.1f}%)")
                logger.info(f"🔢 Total observations: {total_obs:,}")
                logger.info(f"⚠️ Missing sales: {missing_sales_pct:.2f}%")
                logger.info(f"⭕ Zero sales: {zero_sales_pct:.1f}%")
                logger.info(f"❌ Negative sales: {negative_sales:,}")
                logger.info(f"💲 Price coverage: {price_coverage_pct:.1f}%")
                logger.info(f"📅 Calendar coverage: {calendar_coverage_pct:.1f}%")
                logger.info(f"🧮 Data completeness score: {completeness_score:.3f}")
                logger.info(f"⏳ Temporal consistency score: {temporal_consistency_score:.3f}")
                logger.info(f"🏗️ Hierarchical consistency score: {hierarchical_consistency_score:.3f}")

                return quality_metrics

            except Exception as e:
                logger.error(f"❌ Data quality validation failed: {e}")
                raise CustomException(e, sys) from e


    def save_quality_report(self, metrics: DataQualityMetrics):
        report_path = Path("data/quality/m5/quality_report.json")
        report_path.parent.mkdir(parents=True, exist_ok=True)

        report = {
            'timestamp': datetime.utcnow().isoformat(),
            'dataset': 'M5_Walmart',
            'quality_metrics': asdict(metrics),
            'validation_config': {
                'min_sales_per_item': self.config.min_sales_per_item,
                'max_zero_days_pct': self.config.max_zero_days_pct
            }
        }

        try:
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, cls=NumpyEncoder)

            logger.info(f"📝 Quality report saved to {report_path}")
        except Exception as e:
            logger.error(f"❌ Failed to save quality report: {e}", exc_info=True)
            raise CustomException(e, sys) from e


    def save_processed_data(self, df: pd.DataFrame, filename: str = "processed"):
        logger.info(f"💾 Saving processed M5 data: {filename}")

        with self.memory_manager.memory_monitor("data_saving"):
            try:
                output_path = Path(f"data/processed/m5/{filename}.parquet")

                metadata = {
                    'processed_at': datetime.utcnow().isoformat(),
                    'shape': f"{df.shape[0]}x{df.shape[1]}",
                    'date_range_start': df['date'].min().isoformat(),
                    'date_range_end': df['date'].max().isoformat(),
                    'unique_time_series': int(df.groupby(['store_id', 'item_id'], observed=False).ngroups),
                    'data_quality_score': float(self.quality_metrics.data_completeness_score) if self.quality_metrics else None
                }

                table = pa.Table.from_pandas(df)
                table = table.replace_schema_metadata({
                    key: str(value) for key, value in metadata.items()
                })

                pq.write_table(
                    table,
                    output_path,
                    compression=self.config.compression,
                    row_group_size=50000,
                    use_dictionary=True
                )

                file_size_mb = output_path.stat().st_size / (1024 * 1024)
                logger.info(f"✅ Saved processed data: {output_path} ({file_size_mb:.1f} MB)")

                return output_path

            except Exception as e:
                logger.error(f"❌ Failed to save processed data: {e}")
                raise CustomException(e, sys) from e


    def run_full_pipeline(self) -> Tuple[pd.DataFrame, DataQualityMetrics]:
        logger.info("🚀 Starting FAANG-level M5 dataset processing pipeline...")

        start_time = time.time()

        try:
            if not self.validate_file_integrity():
                raise ValueError("File integrity validation failed ❌")

            calendar = self.load_calendar_data()
            prices = self.load_prices_data()
            sales_wide, day_columns = self.load_sales_data_chunked()
            master_data_dir = self.create_master_dataset(sales_wide, calendar, prices)
            master_df = self.load_master_dataset(master_data_dir)

            del sales_wide, calendar, prices
            gc.collect()

            quality_metrics = self.validate_data_quality(master_df)
            self.save_quality_report(quality_metrics)

            if self.config.save_interim:
                self.save_processed_data(master_df, "master")

            duration = time.time() - start_time

            logger.info("=" * 80)
            logger.info("🎉 M5 DATASET PROCESSING COMPLETED SUCCESSFULLY 🎉")
            logger.info("=" * 80)
            logger.info(f"⏱ Processing time: {duration / 60:.1f} minutes")
            logger.info(f"📊 Final dataset shape: {master_df.shape}")
            logger.info(f"💾 Memory usage: {self.memory_manager.get_memory_usage()['rss_gb']:.2f} GB")
            logger.info(f"⭐ Data quality score: {quality_metrics.data_completeness_score:.3f}")
            logger.info("=" * 80)

            return master_df, quality_metrics

        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}")
            raise CustomException(e, sys) from e
