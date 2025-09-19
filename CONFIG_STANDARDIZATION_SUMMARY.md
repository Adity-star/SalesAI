# Configuration Standardization Summary

## Overview
Successfully implemented centralized configuration management for the SalesAI project. All modules now import from `src.config` instead of using hardcoded strings, providing a single source of truth for paths, URIs, and hyperparameters.

## Configuration Module (`src/config.py`)

### Core Constants
- **DATA_DIR**: Main data directory
- **PROCESSED_DIR**: Processed data directory  
- **RAW_DATA_DIR**: Raw data directory
- **MODEL_DIR**: Model artifacts directory
- **ARTIFACTS_DIR**: General artifacts directory
- **FEATURE_STORE_PATH**: Feature store location
- **MLFLOW_TRACKING_URI**: MLflow tracking server URI
- **LOG_LEVEL**: Logging level configuration
- **LOG_FORMAT**: Log message format

### Parameter Groups
- **MODEL_HYPERPARAMS**: Machine learning model parameters
- **DATA_PARAMS**: Data processing parameters
- **FEATURE_PARAMS**: Feature engineering parameters
- **EVALUATION_PARAMS**: Model evaluation parameters
- **DEPLOYMENT_PARAMS**: Model deployment parameters
- **AIRFLOW_PARAMS**: Airflow DAG configuration
- **S3_PARAMS**: S3/MinIO configuration

### Environment Variable Support
All configuration values can be overridden using environment variables:
- `DATA_DIR` → `DATA_DIR` env var
- `MLFLOW_TRACKING_URI` → `MLFLOW_TRACKING_URI` env var
- `LOG_LEVEL` → `LOG_LEVEL` env var
- And many more...

## Updated Modules

### Source Modules (`src/`)
All source modules now import from `src.config`:

- **data_extraction.py**: Uses `DATA_DIR`, `RAW_DATA_DIR`
- **data_validation.py**: Uses `DATA_PARAMS`
- **model_training.py**: Uses `MODEL_HYPERPARAMS`, `DATA_PARAMS`, `MLFLOW_TRACKING_URI`, `MODEL_DIR`
- **model_evaluation.py**: Uses `EVALUATION_PARAMS`
- **model_deployment.py**: Uses `DEPLOYMENT_PARAMS`, `MLFLOW_TRACKING_URI`
- **reporting.py**: Uses `LOG_LEVEL`, `LOG_FORMAT`
- **cleanup.py**: Uses `DATA_DIR`, `ARTIFACTS_DIR`

### Task Modules (`tasks/`)
All task modules now import from `src.config`:

- **data_tasks.py**: Uses `DATA_DIR`, `RAW_DATA_DIR`
- **feature_tasks.py**: Uses `FEATURE_STORE_PATH`, `FEATURE_PARAMS`
- **training_tasks.py**: Uses `MODEL_DIR`, `MODEL_HYPERPARAMS`, `MLFLOW_TRACKING_URI`
- **model_tasks.py**: Uses `DEPLOYMENT_PARAMS`, `MLFLOW_TRACKING_URI`
- **deployment_tasks.py**: Uses `LOG_LEVEL`, `LOG_FORMAT`

### DAG (`dags/sales_forecasting_dag.py`)
The DAG now uses `AIRFLOW_PARAMS` for configuration:

- Owner, retry settings, email configuration
- All values come from centralized config

## Benefits Achieved

### 1. Single Source of Truth
- All configuration values are defined in one place
- No more scattered hardcoded strings across modules
- Easy to maintain and update

### 2. Environment Flexibility
- Configuration can be overridden via environment variables
- Different environments (dev, staging, prod) can use different values
- No code changes needed for environment-specific settings

### 3. Type Safety
- All configuration values are properly typed
- IDE autocomplete and type checking support
- Reduced runtime errors from incorrect configuration

### 4. Documentation
- All configuration options are documented in one place
- Clear parameter descriptions and default values
- Easy to understand what each setting controls

### 5. Maintainability
- Changes to configuration only require updates in one file
- No need to search through multiple files for hardcoded values
- Easier to audit and review configuration changes

## Usage Examples

### Basic Import
```python
from src.config import DATA_DIR, MLFLOW_TRACKING_URI, MODEL_HYPERPARAMS
```

### Using Configuration Values
```python
# Instead of hardcoded paths
output_dir = "/tmp/data"  # OLD

# Use configuration
output_dir = DATA_DIR  # NEW
```

### Environment Variable Override
```bash
# Override configuration via environment
export DATA_DIR="/custom/data/path"
export MLFLOW_TRACKING_URI="http://custom-mlflow:5000"
export LOG_LEVEL="DEBUG"
```

### Parameter Groups
```python
# Access grouped parameters
n_estimators = MODEL_HYPERPARAMS.get('n_estimators', 100)
lag_features = FEATURE_PARAMS.get('lag_features', [1, 7, 14])
retry_delay = AIRFLOW_PARAMS.get('retry_delay_minutes', 5)
```

## Migration Impact

### Before
- Hardcoded strings scattered across 15+ files
- Difficult to change configuration values
- No environment-specific configuration
- Inconsistent parameter usage

### After
- All configuration centralized in `src/config.py`
- Easy to update any configuration value
- Environment variable support
- Consistent parameter usage across all modules

## Next Steps

1. **Environment Configuration**: Set up environment-specific configuration files
2. **Validation**: Add configuration validation to catch invalid values
3. **Documentation**: Create configuration documentation for users
4. **Testing**: Add configuration tests to ensure all values are valid
5. **Monitoring**: Add configuration monitoring to track usage

The configuration standardization is now complete and all modules follow the single source of truth principle!
