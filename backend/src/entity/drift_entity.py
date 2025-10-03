
import logging
from typing import List, Optional
from dataclasses import dataclass
from datetime import datetime

# Configuration

logger = logging.getLogger(__name__)


@dataclass
class MonitoringConfig:
    """Configuration for model monitoring system."""
    # Database settings
    db_path: str = "monitoring.db"
    retention_days: int = 90
    
    # Drift detection settings
    drift_detection_window: int = 7  # days
    drift_threshold: float = 0.1     # KS test p-value threshold
    psi_threshold: float = 0.25      # Population Stability Index threshold
    
    # Performance monitoring
    accuracy_window: int = 14        # days to calculate rolling accuracy
    performance_degradation_threshold: float = 0.15  # 15% increase in error
    min_samples_for_monitoring: int = 100
    
    # Alerting settings
    email_alerts: bool = True
    slack_webhook_url: Optional[str] = None
    alert_cooldown_hours: int = 4    # Minimum hours between similar alerts
    
    # Business metrics
    business_metrics_enabled: bool = True
    forecast_bias_threshold: float = 0.1  # 10% bias threshold
    inventory_impact_tracking: bool = True

@dataclass
class DriftAlert:
    """Drift detection alert."""
    timestamp: datetime
    alert_type: str  # 'feature_drift', 'target_drift', 'performance_degradation'
    severity: str    # 'warning', 'critical'
    metric_name: str
    metric_value: float
    threshold: float
    description: str
    affected_features: Optional[List[str]] = None

@dataclass
class PerformanceMetrics:
    """Model performance metrics tracking."""
    model_name: str
    timestamp: datetime
    period_start: datetime
    period_end: datetime
    
    # Accuracy metrics
    rmse: float
    mae: float
    mase: float
    smape: float
    
    # Business metrics
    forecast_bias: float
    forecast_accuracy_pct: float
    inventory_waste_estimate: float
    
    # Volume metrics
    predictions_count: int
    zero_predictions_pct: float
    
    # Comparison to baseline
    rmse_change_pct: float
    mae_change_pct: float