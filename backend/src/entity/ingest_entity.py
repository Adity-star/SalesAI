
from typing import Dict
from dataclasses import dataclass


@dataclass
class DatasetConfig:
    """Configuration specific to M5 Walmart dataset."""
    # Data paths
    sales_train_path: str = "data/raw/sales_train_evaluation.csv"
    prices_path: str = "data/raw/sell_prices.csv"
    calendar_path: str = "data/raw/calendar.csv"
    
    # Schema definitions
    sales_schema: Dict = None
    prices_schema: Dict = None
    calendar_schema: Dict = None
    
    # Processing parameters
    chunk_size: int = 10000
    max_memory_gb: float = 8.0
    
    # Validation thresholds
    min_sales_per_item: int = 100  # Minimum sales history required
    max_zero_days_pct: float = 0.95  # Max percentage of zero-sales days allowed
    
    # Output configuration
    save_interim: bool = True
    save_features: bool = True
    compression: str = "snappy"
    load_fraction: float = 0.05
    max_rows: int = None

@dataclass
class DataQualityMetrics:
    """Comprehensive data quality metrics for M5 dataset."""
    total_time_series: int
    valid_time_series: int
    total_observations: int
    missing_sales_pct: float
    zero_sales_pct: float
    negative_sales_count: int
    price_coverage_pct: float
    calendar_coverage_pct: float
    data_completeness_score: float
    temporal_consistency_score: float
    hierarchical_consistency_score: float
