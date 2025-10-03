
from datetime import date
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta, date as Date
from pydantic import ConfigDict, BaseModel, Field, validator



class PredictionRequest(BaseModel):
    """Single prediction request model."""
    store_id: str = Field(..., description="Store identifier")
    item_id: str = Field(..., description="Item/product identifier")
    date: Date = Field(..., description="Prediction date")
    horizon: int = Field(7, ge=1, le=90, description="Forecast horizon in days")
    model_name: Optional[str] = Field("lightgbm", description="Model to use for prediction")

    @validator('date')
    def validate_date(cls, v: Date) -> Date:
        if v > date.today() + timedelta(days=365):
            raise ValueError('Date cannot be more than 1 year in the future')
        return v

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "store_id": "CA_1",
                "item_id": "HOBBIES_1_001",
                "date": "2024-01-15",
                "horizon": 7,
                "model_name": "lightgbm"
            }
        }
    )


class BatchPredictionRequest(BaseModel):
    """Batch prediction request model."""
    items: List[Dict[str, Any]] = Field(..., description="List of items to predict")
    horizon: int = Field(7, ge=1, le=90, description="Forecast horizon in days")
    model_name: Optional[str] = Field("lightgbm", description="Model to use")
    output_format: str = Field("json", pattern="^(json|csv|parquet)$")

    @validator('items')
    def validate_items(cls, v: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if len(v) > 10000:
            raise ValueError('Maximum 10,000 items per batch request')
        return v

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "items": [
                    {"store_id": "CA_1", "item_id": "HOBBIES_1_001", "date": "2024-01-15"},
                    {"store_id": "CA_1", "item_id": "HOBBIES_1_002", "date": "2024-01-15"}
                ],
                "horizon": 7,
                "model_name": "lightgbm",
                "output_format": "json"
            }
        }
    )


class HierarchicalPredictionRequest(BaseModel):
    """Hierarchical aggregation prediction request."""
    aggregation_level: str = Field(..., pattern="^(item|category|department|store|state|total)$")
    filters: Optional[Dict[str, List[str]]] = Field(None, description="Filters for aggregation")
    start_date: date = Field(..., description="Start date for predictions")
    end_date: date = Field(..., description="End date for predictions")
    model_name: Optional[str] = Field("lightgbm", description="Model to use")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "aggregation_level": "category",
                "filters": {"cat_id": ["HOBBIES_1"], "state_id": ["CA"]},
                "start_date": "2024-01-15",
                "end_date": "2024-01-21",
                "model_name": "lightgbm"
            }
        }
    )


class PredictionResponse(BaseModel):
    """Prediction response model."""
    store_id: str
    item_id: str
    date: str
    prediction: float
    confidence_interval: Optional[Dict[str, float]] = None
    model_name: str
    model_version: str
    prediction_timestamp: datetime
    features_used: Optional[List[str]] = None
    features: Optional[Dict[str, Any]] = None


class BatchPredictionResponse(BaseModel):
    """Batch prediction response model."""
    request_id: str
    total_predictions: int
    successful_predictions: int
    failed_predictions: int
    predictions: List[PredictionResponse]
    processing_time_seconds: float
    model_name: str
    model_version: str


class ModelInfo(BaseModel):
    """Model information response."""
    model_name: str
    model_version: str
    model_type: str
    training_date: datetime
    feature_count: int
    performance_metrics: Dict[str, float]
    last_loaded: datetime
