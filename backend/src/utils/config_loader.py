"""
Configuration Loader for Pipeline
====================================

Centralized configuration management for feature engineering pipeline.
Supports YAML and JSON configuration files with validation and defaults.
"""

import yaml
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
import sys

from src.logger import logger
from src.exception import CustomException

logger = logging.getLogger(__name__)

def find_project_root(marker_files=("pyproject.toml", ".git")) -> Path:
    """
    Walk up the directory tree until we find a known project marker like pyproject.toml or .git
    """
    current = Path(__file__).resolve()
    for parent in current.parents:
        if any((parent / marker).exists() for marker in marker_files):
            return parent
    raise RuntimeError("Project root not found. Make sure you're inside a valid project structure.")

PROJECT_ROOT = find_project_root()
DEFAULT_CONFIG_DIR = PROJECT_ROOT / "configs"

class ConfigLoader:
    def __init__(self, config_dir: Union[str, Path] = DEFAULT_CONFIG_DIR):
        self.config_dir = Path(config_dir)
        self.config_cache = {}

    def _get_full_path(self, file_path: Union[str, Path]) -> Path:
        p = Path(file_path)
        if not p.is_absolute():
            p = self.config_dir / p
        if not p.exists():
            raise FileNotFoundError(f"Config file not found: {p}")
        return p

    def load_yaml(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        path = self._get_full_path(file_path)
        cache_key = str(path.resolve())

        if cache_key in self.config_cache:
            logger.info(f"Using cached config for {path}")
            return self.config_cache[cache_key]

        try:
            with open(path, 'r') as f:
                config = yaml.safe_load(f)

            self.config_cache[cache_key] = config
            logger.info(f"Loaded configuration from: {path}")

            return config

        except yaml.YAMLError as e:
            logger.error(f"Failed to parse YAML file {path}: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to load configuration from {path}: {e}")
            raise CustomException(e, sys) from e

    def load_json(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        path = self._get_full_path(file_path)

        try:
            with open(path, 'r') as f:
                config = json.load(f)

            logger.info(f"Loaded JSON configuration from: {path}")
            return config

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON file {path}: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to load configuration from {path}: {e}")
            raise CustomException(e, sys) from e

    def load_features_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        if config_path is None:
            config_path = "ml_features.yaml"

        config = self.load_yaml(config_path)

        # Validate required sections
        required_sections = ['dataset', 'date_features', 'lag_features', 'rolling_features']
        missing_sections = [section for section in required_sections if section not in config]

        if missing_sections:
            logger.warning(f"Missing configuration sections: {missing_sections}")

        config = self._apply_feature_defaults(config)

        logger.info("Features configuration loaded and validated")
        return config

    def load_dataset_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        if config_path is None:
            config_path = "dataset_config.yaml"

        config = self.load_yaml(config_path)

        if 'data_paths' in config:
            self._validate_data_paths(config['data_paths'])

        logger.info("Dataset configuration loaded and validated")
        return config

    def _validate_data_paths(self, data_paths: Dict[str, str]):
        input_paths = ['sales_train', 'prices', 'calendar']

        for path_key in input_paths:
            if path_key in data_paths:
                path = Path(data_paths[path_key])
                if not path.exists():
                    logger.warning(f"Data file not found: {path} (key: {path_key})")
                else:
                    file_size_mb = path.stat().st_size / (1024 * 1024)
                    logger.info(f"Found {path_key}: {path} ({file_size_mb:.1f} MB)")

    def _apply_feature_defaults(self, config: Dict[str, Any]) -> Dict[str, Any]:
        # Example default application (customize as needed)
        if 'lag_features' not in config:
            config['lag_features'] = {}
        if 'rolling_features' not in config:
            config['rolling_features'] = {}
        # Add more defaults here as needed
        return config

    def save_config(self, config: Dict[str, Any], file_path: Union[str, Path], 
                    format_type: str = 'yaml'):
        file_path = Path(file_path)

        if not file_path.is_absolute():
            file_path = self.config_dir / file_path

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
            raise CustomException(e, sys) from e

# Global config loader instance
config_loader = ConfigLoader()

def load_features_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    return config_loader.load_features_config(config_path)

def load_dataset_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    return config_loader.load_dataset_config(config_path)

def load_config(config_type: str = "features", config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration by type.
    Args:
        config_type: 'features', 'dataset', or 'logging' (logging not implemented here)
        config_path: Optional custom path
    Returns:
        Configuration dictionary
    """
    if config_type == 'features':
        return load_features_config(config_path)
    elif config_type == 'dataset':
        return load_dataset_config(config_path)
    else:
        raise ValueError(f"Unknown config type: {config_type}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Configuration Loader")
    parser.add_argument("--create-defaults", action="store_true",
                        help="Create default configuration files (not implemented)")
    parser.add_argument("--test-load", action="store_true",
                        help="Test loading configurations")
    parser.add_argument("--config-dir", default="configs",
                        help="Configuration directory")

    args = parser.parse_args()

    loader = ConfigLoader(args.config_dir)

    if args.test_load:
        try:
            features_config = loader.load_features_config()
            print("Features config loaded successfully")

            dataset_config = loader.load_dataset_config()
            print("Dataset config loaded successfully")

        except Exception as e:
            print(f"Error loading configurations: {e}")
