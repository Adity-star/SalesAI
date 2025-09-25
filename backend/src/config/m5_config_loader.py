"""
Configuration Loader for M5 Pipeline
====================================

Centralized configuration management for M5 Walmart feature engineering pipeline.
Supports YAML and JSON configuration files with validation and defaults.
"""

import yaml
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass
import os

logger = logging.getLogger(__name__)

@dataclass
class ConfigPaths:
    """Standard configuration file paths."""
    features: str = "configs/features.yaml"
    dataset: str = "configs/mdataset_config.yaml"
    
class M5ConfigLoader:
    """Configuration loader for M5 pipeline with validation and defaults."""
    
    def __init__(self, config_dir: str = "configs"):
        """
        Initialize config loader.
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = Path(config_dir)
        self.config_cache = {}
        
        # Ensure config directory exists
        self.config_dir.mkdir(exist_ok=True)
        
    def load_yaml(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load YAML configuration file with caching.
        
        Args:
            file_path: Path to YAML file
            
        Returns:
            Dictionary containing configuration
        """
        file_path = Path(file_path)
        
        # Check cache first
        cache_key = str(file_path.absolute())
        if cache_key in self.config_cache:
            return self.config_cache[cache_key]
        
        # Resolve path relative to config directory if not absolute
        if not file_path.is_absolute():
            file_path = self.config_dir / file_path
        
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        
        try:
            with open(file_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Cache the configuration
            self.config_cache[cache_key] = config
            logger.info(f"Loaded configuration from: {file_path}")
            
            return config
            
        except yaml.YAMLError as e:
            logger.error(f"Failed to parse YAML file {file_path}: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to load configuration from {file_path}: {e}")
            raise
    
    def load_json(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load JSON configuration file.
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            Dictionary containing configuration
        """
        file_path = Path(file_path)
        
        if not file_path.is_absolute():
            file_path = self.config_dir / file_path
        
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        
        try:
            with open(file_path, 'r') as f:
                config = json.load(f)
            
            logger.info(f"Loaded JSON configuration from: {file_path}")
            return config
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON file {file_path}: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to load configuration from {file_path}: {e}")
            raise
    
    def load_features_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Load feature engineering configuration.
        
        Args:
            config_path: Custom path to features config file
            
        Returns:
            Features configuration dictionary
        """
        if config_path is None:
            config_path = "m5_features.yaml"
        
        config = self.load_yaml(config_path)
        
        # Validate required sections
        required_sections = ['dataset', 'date_features', 'lag_features', 'rolling_features']
        missing_sections = [section for section in required_sections if section not in config]
        
        if missing_sections:
            logger.warning(f"Missing configuration sections: {missing_sections}")
        
        # Apply defaults if missing
        config = self._apply_feature_defaults(config)
        
        logger.info("Features configuration loaded and validated")
        return config
    
    def load_dataset_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Load dataset configuration.
        
        Args:
            config_path: Custom path to dataset config file
            
        Returns:
            Dataset configuration dictionary
        """
        if config_path is None:
            config_path = "m5_dataset.yaml"
        
        config = self.load_yaml(config_path)
        
        # Validate data paths exist
        if 'data_paths' in config:
            self._validate_data_paths(config['data_paths'])
        
        logger.info("Dataset configuration loaded and validated")
        return config
    
    def _apply_feature_defaults(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply default values for missing feature configuration sections."""
        
        defaults = {
            'dataset': {
                'target_column': 'sales',
                'group_columns': ['store_id', 'item_id'],
                'date_column': 'date'
            },
            'processing': {
                'memory_efficient': True,
                'chunk_size': 100000,
                'n_jobs': -1
            },
            'date_features': {
                'enabled': True,
                'basic_features': ['year', 'month', 'day', 'dayofweek', 'quarter'],
                'holiday_features': {'enabled': True, 'country': 'US'}
            },
            'lag_features': {
                'enabled': True,
                'sales_lags': {'windows': [1, 2, 3, 7, 14, 21, 28]}
            },
            'rolling_features': {
                'enabled': True,
                'windows': [7, 14, 28],
                'statistics': [{'name': 'mean'}, {'name': 'std'}]
            },
            'walmart_features': {
                'snap_features': {'enabled': True},
                'price_features': {'enabled': True},
                'event_features': {'enabled': True}
            }
        }
        
        # Deep merge defaults with user config
        merged_config = self._deep_merge(defaults, config)
        return merged_config
    
    def _deep_merge(self, default: Dict, override: Dict) -> Dict:
        """Deep merge two dictionaries."""
        result = default.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _validate_data_paths(self, data_paths: Dict[str, str]):
        """Validate that data file paths exist."""
        input_paths = ['sales_train', 'prices', 'calendar']
        
        for path_key in input_paths:
            if path_key in data_paths:
                path = Path(data_paths[path_key])
                if not path.exists():
                    logger.warning(f"Data file not found: {path} (key: {path_key})")
                else:
                    file_size_mb = path.stat().st_size / (1024 * 1024)
                    logger.info(f"Found {path_key}: {path} ({file_size_mb:.1f} MB)")
    
    def _get_default_logging_config(self) -> Dict[str, Any]:
        """Get default logging configuration."""
        return {
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': {
                'default': {
                    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                }
            },
            'handlers': {
                'console': {
                    'class': 'logging.StreamHandler',
                    'level': 'INFO',
                    'formatter': 'default'
                }
            },
            'root': {
                'level': 'INFO',
                'handlers': ['console']
            }
        }
    
    def save_config(self, config: Dict[str, Any], file_path: Union[str, Path], 
                   format_type: str = 'yaml'):
        """
        Save configuration to file.
        
        Args:
            config: Configuration dictionary to save
            file_path: Output file path
            format_type: 'yaml' or 'json'
        """
        file_path = Path(file_path)
        
        if not file_path.is_absolute():
            file_path = self.config_dir / file_path
        
        # Ensure parent directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(file_path, 'w') as f:
                if format_type.lower() == 'yaml':
                    yaml.dump(config, f, default_flow_style=False, indent=2)
                elif format_type.lower() == 'json':
                    json.dump(config, f, indent=2)
                else:
                    raise ValueError(f"Unsupported format: {format_type}")
            
            logger.info(f"Configuration saved to: {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to save configuration to {file_path}: {e}")
            raise
    
    def create_default_configs(self):
        """Create default configuration files if they don't exist."""
        
        # Create default features config
        features_config_path = self.config_dir / "m5_features.yaml"
        if not features_config_path.exists():
            logger.info("Creating default features configuration...")
            default_features_config = self._get_default_features_config()
            self.save_config(default_features_config, features_config_path)
        
        # Create default dataset config
        dataset_config_path = self.config_dir / "m5_dataset.yaml"
        if not dataset_config_path.exists():
            logger.info("Creating default dataset configuration...")
            default_dataset_config = self._get_default_dataset_config()
            self.save_config(default_dataset_config, dataset_config_path)
    
    def _get_default_features_config(self) -> Dict[str, Any]:
        """Get default features configuration."""
        return {
            'metadata': {
                'name': 'M5_Walmart_Features',
                'version': '1.0.0',
                'description': 'Default M5 feature engineering configuration'
            },
            'dataset': {
                'target_column': 'sales',
                'group_columns': ['store_id', 'item_id'],
                'date_column': 'date'
            },
            'processing': {
                'memory_efficient': True,
                'chunk_size': 100000,
                'n_jobs': -1
            },
            'date_features': {
                'enabled': True,
                'basic_features': ['year', 'month', 'day', 'dayofweek', 'quarter', 'weekofyear'],
                'derived_features': ['is_weekend', 'is_month_start', 'is_month_end'],
                'holiday_features': {
                    'enabled': True,
                    'country': 'US'
                }
            },
            'lag_features': {
                'enabled': True,
                'sales_lags': {
                    'windows': [1, 2, 3, 7, 14, 21, 28],
                    'dtype': 'int16'
                }
            },
            'rolling_features': {
                'enabled': True,
                'windows': [7, 14, 28],
                'statistics': [
                    {'name': 'mean', 'dtype': 'float32'},
                    {'name': 'std', 'dtype': 'float32', 'fill_na': 0}
                ]
            }
        }
    
    def _get_default_dataset_config(self) -> Dict[str, Any]:
        """Get default dataset configuration."""
        return {
            'metadata': {
                'name': 'M5_Walmart_Dataset',
                'version': '1.0.0'
            },
            'data_paths': {
                'sales_train': 'data/raw/sales_train_evaluation.csv',
                'prices': 'data/raw/sell_prices.csv',
                'calendar': 'data/raw/calendar.csv',
                'processed_data': 'data/processed/m5/m5_master.parquet',
                'features_data': 'data/features/m5/m5_features.parquet'
            },
            'processing': {
                'chunk_size': 5000,
                'max_memory_gb': 4.0
            },
            'quality': {
                'min_sales_per_item': 100,
                'max_zero_days_pct': 0.95
            }
        }
    
    def get_config_summary(self, config: Dict[str, Any]) -> str:
        """Get a summary of configuration settings."""
        summary_lines = []
        
        if 'metadata' in config:
            meta = config['metadata']
            summary_lines.append(f"Config: {meta.get('name', 'Unknown')} v{meta.get('version', '1.0')}")
        
        if 'dataset' in config:
            dataset = config['dataset']
            summary_lines.append(f"Target: {dataset.get('target_column', 'sales')}")
            summary_lines.append(f"Groups: {dataset.get('group_columns', [])}")
        
        # Count enabled features
        enabled_features = []
        feature_sections = ['date_features', 'lag_features', 'rolling_features', 
                          'walmart_features', 'advanced_features']
        
        for section in feature_sections:
            if config.get(section, {}).get('enabled', False):
                enabled_features.append(section.replace('_', ' ').title())
        
        if enabled_features:
            summary_lines.append(f"Enabled: {', '.join(enabled_features)}")
        
        return '\n'.join(summary_lines)

# Global config loader instance
config_loader = M5ConfigLoader()

# Convenience functions
def load_features_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load features configuration."""
    return config_loader.load_features_config(config_path)

def load_dataset_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load dataset configuration."""
    return config_loader.load_dataset_config(config_path)

def load_config(config_type: str, config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration by type.
    
    Args:
        config_type: 'features', 'dataset', or 'logging'
        config_path: Optional custom path
        
    Returns:
        Configuration dictionary
    """
    if config_type == 'features':
        return load_features_config(config_path)
    elif config_type == 'dataset':
        return load_dataset_config(config_path)
    elif config_type == 'logging':
        return config_loader.load_logging_config(config_path)
    else:
        raise ValueError(f"Unknown config type: {config_type}")

def create_default_configs():
    """Create default configuration files."""
    config_loader.create_default_configs()

if __name__ == "__main__":
    # Test configuration loading
    import argparse
    
    parser = argparse.ArgumentParser(description="M5 Configuration Manager")
    parser.add_argument("--create-defaults", action="store_true", 
                       help="Create default configuration files")
    parser.add_argument("--test-load", action="store_true",
                       help="Test loading configurations")
    parser.add_argument("--config-dir", default="configs",
                       help="Configuration directory")
    
    args = parser.parse_args()
    
    # Setup basic logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize config loader
    loader = M5ConfigLoader(args.config_dir)
    
    if args.create_defaults:
        loader.create_default_configs()
        print("Default configuration files created.")
    
    if args.test_load:
        try:
            # Test loading each config type
            features_config = loader.load_features_config()
            print("Features config loaded successfully")
            print(loader.get_config_summary(features_config))
            
            dataset_config = loader.load_dataset_config()
            print("\nDataset config loaded successfully")
            
        except Exception as e:
            print(f"Error loading configurations: {e}")