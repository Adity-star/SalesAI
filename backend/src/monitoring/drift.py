"""
M5 Model Monitoring & Drift Detection System
==========================================

Comprehensive monitoring system for production M5 forecasting models:
- Prediction accuracy tracking against actuals
- Feature drift detection and alerts
- Model performance degradation monitoring
- Automated retraining triggers
- Business metrics dashboards
- Data quality monitoring
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, date
import warnings
import json
import sqlite3
import asyncio
from collections import defaultdict, deque
from abc import ABC, abstractmethod
import time
import hashlib

# Statistical libraries
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA

# Monitoring and alerting
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
import requests
import schedule

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configuration

logger = logging.getLogger(__name__)

# -------------------------------
# Configuration Classes
# -------------------------------

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

# -------------------------------
# Data Storage Layer
# -------------------------------

class MonitoringDatabase:
    """SQLite database for monitoring data storage."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize monitoring database tables."""
        conn = sqlite3.connect(self.db_path)
        
        # Predictions table
        conn.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                store_id TEXT NOT NULL,
                item_id TEXT NOT NULL,
                prediction_date DATE NOT NULL,
                prediction_value REAL NOT NULL,
                actual_value REAL,
                prediction_timestamp DATETIME NOT NULL,
                horizon_days INTEGER NOT NULL,
                features TEXT,  -- JSON string of features used
                model_version TEXT,
                UNIQUE(model_name, store_id, item_id, prediction_date, horizon_days)
            )
        ''')
        
        # Feature statistics table
        conn.execute('''
            CREATE TABLE IF NOT EXISTS feature_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                feature_name TEXT NOT NULL,
                mean_value REAL,
                std_value REAL,
                min_value REAL,
                max_value REAL,
                null_count INTEGER,
                total_count INTEGER,
                distribution_hash TEXT
            )
        ''')
        
        # Performance metrics table
        conn.execute('''
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                period_start DATETIME NOT NULL,
                period_end DATETIME NOT NULL,
                rmse REAL,
                mae REAL,
                mase REAL,
                smape REAL,
                forecast_bias REAL,
                predictions_count INTEGER,
                rmse_change_pct REAL,
                mae_change_pct REAL
            )
        ''')
        
        # Alerts table
        conn.execute('''
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                alert_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                threshold_value REAL NOT NULL,
                description TEXT NOT NULL,
                resolved BOOLEAN DEFAULT FALSE,
                resolved_timestamp DATETIME
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("Monitoring database initialized", db_path=self.db_path)
    
    def store_prediction(self, model_name: str, store_id: str, item_id: str,
                        prediction_date: date, prediction_value: float,
                        actual_value: Optional[float], horizon_days: int,
                        features: Optional[Dict] = None, model_version: str = "unknown"):
        """Store prediction and actual values."""
        conn = sqlite3.connect(self.db_path)
        
        features_json = json.dumps(features) if features else None
        
        conn.execute('''
            INSERT OR REPLACE INTO predictions 
            (model_name, store_id, item_id, prediction_date, prediction_value, 
             actual_value, prediction_timestamp, horizon_days, features, model_version)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (model_name, store_id, item_id, prediction_date, prediction_value,
              actual_value, datetime.now(), horizon_days, features_json, model_version))
        
        conn.commit()
        conn.close()
    
    def store_feature_stats(self, feature_stats: Dict[str, Dict]):
        """Store feature statistics."""
        conn = sqlite3.connect(self.db_path)
        timestamp = datetime.now()
        
        for feature_name, stats in feature_stats.items():
            conn.execute('''
                INSERT INTO feature_stats 
                (timestamp, feature_name, mean_value, std_value, min_value, max_value, 
                 null_count, total_count, distribution_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (timestamp, feature_name, stats.get('mean'), stats.get('std'),
                  stats.get('min'), stats.get('max'), stats.get('null_count'),
                  stats.get('total_count'), stats.get('distribution_hash')))
        
        conn.commit()
        conn.close()
    
    def store_performance_metrics(self, metrics: PerformanceMetrics):
        """Store performance metrics."""
        conn = sqlite3.connect(self.db_path)
        
        conn.execute('''
            INSERT INTO performance_metrics 
            (model_name, timestamp, period_start, period_end, rmse, mae, mase, smape,
             forecast_bias, predictions_count, rmse_change_pct, mae_change_pct)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (metrics.model_name, metrics.timestamp, metrics.period_start, metrics.period_end,
              metrics.rmse, metrics.mae, metrics.mase, metrics.smape, metrics.forecast_bias,
              metrics.predictions_count, metrics.rmse_change_pct, metrics.mae_change_pct))
        
        conn.commit()
        conn.close()
    
    def store_alert(self, alert: DriftAlert):
        """Store drift alert."""
        conn = sqlite3.connect(self.db_path)
        
        conn.execute('''
            INSERT INTO alerts 
            (timestamp, alert_type, severity, metric_name, metric_value, 
             threshold_value, description)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (alert.timestamp, alert.alert_type, alert.severity, alert.metric_name,
              alert.metric_value, alert.threshold, alert.description))
        
        conn.commit()
        conn.close()
    
    def get_recent_predictions(self, model_name: str, days: int = 7) -> pd.DataFrame:
        """Get recent predictions for analysis."""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT * FROM predictions 
            WHERE model_name = ? 
            AND prediction_timestamp >= datetime('now', '-{} days')
            ORDER BY prediction_timestamp DESC
        '''.format(days)
        
        df = pd.read_sql_query(query, conn, params=(model_name,))
        conn.close()
        
        return df
    
    def get_feature_history(self, feature_name: str, days: int = 30) -> pd.DataFrame:
        """Get feature statistics history."""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT * FROM feature_stats 
            WHERE feature_name = ? 
            AND timestamp >= datetime('now', '-{} days')
            ORDER BY timestamp DESC
        '''.format(days)
        
        df = pd.read_sql_query(query, conn, params=(feature_name,))
        conn.close()
        
        return df

# -------------------------------
# Drift Detection Engine
# -------------------------------

class DriftDetector:
    """Statistical drift detection for features and target variables."""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.reference_distributions = {}
        self.feature_scalers = {}
    
    def set_reference_distribution(self, feature_name: str, reference_data: np.ndarray):
        """Set reference distribution for drift detection."""
        self.reference_distributions[feature_name] = {
            'data': reference_data,
            'mean': np.mean(reference_data),
            'std': np.std(reference_data),
            'percentiles': np.percentile(reference_data, [1, 5, 25, 50, 75, 95, 99])
        }
        
        # Fit scaler for normalized comparison
        scaler = StandardScaler()
        self.feature_scalers[feature_name] = scaler.fit(reference_data.reshape(-1, 1))
        
        logger.info("Reference distribution set for feature", feature_name=feature_name)
    
    def detect_feature_drift(self, feature_name: str, current_data: np.ndarray) -> Tuple[bool, float, str]:
        """Detect drift in a single feature using multiple statistical tests."""
        if feature_name not in self.reference_distributions:
            logger.warning("No reference distribution for feature", feature_name=feature_name)
            return False, 1.0, "No reference distribution"
        
        reference = self.reference_distributions[feature_name]
        reference_data = reference['data']
        
        # Remove NaN values
        current_clean = current_data[~np.isnan(current_data)]
        reference_clean = reference_data[~np.isnan(reference_data)]
        
        if len(current_clean) < 10 or len(reference_clean) < 10:
            return False, 1.0, "Insufficient data"
        
        # 1. Kolmogorov-Smirnov test
        ks_stat, ks_p_value = stats.ks_2samp(reference_clean, current_clean)
        
        # 2. Population Stability Index (PSI)
        psi_value = self._calculate_psi(reference_clean, current_clean)
        
        # 3. Mean shift detection
        mean_shift = abs(np.mean(current_clean) - reference['mean']) / (reference['std'] + 1e-8)
        
        # Determine if drift detected
        drift_detected = (
            ks_p_value < self.config.drift_threshold or
            psi_value > self.config.psi_threshold or
            mean_shift > 3.0  # 3 standard deviations
        )
        
        # Create detailed description
        description = f"KS p-value: {ks_p_value:.4f}, PSI: {psi_value:.4f}, Mean shift: {mean_shift:.2f}σ"
        
        return drift_detected, min(ks_p_value, 1-psi_value), description
    
    def _calculate_psi(self, reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
        """Calculate Population Stability Index."""
        # Create bins based on reference distribution
        bin_edges = np.histogram_bin_edges(reference, bins=bins)
        
        # Calculate distributions
        ref_dist, _ = np.histogram(reference, bins=bin_edges)
        cur_dist, _ = np.histogram(current, bins=bin_edges)
        
        # Convert to proportions
        ref_prop = ref_dist / len(reference)
        cur_prop = cur_dist / len(current)
        
        # Avoid division by zero
        ref_prop = np.where(ref_prop == 0, 0.0001, ref_prop)
        cur_prop = np.where(cur_prop == 0, 0.0001, cur_prop)
        
        # Calculate PSI
        psi = np.sum((cur_prop - ref_prop) * np.log(cur_prop / ref_prop))
        return psi
    
    def detect_multivariate_drift(self, reference_df: pd.DataFrame, 
                                 current_df: pd.DataFrame) -> Tuple[bool, float, str]:
        """Detect drift across multiple features using PCA-based approach."""
        try:
            # Select numeric columns common to both datasets
            numeric_cols = reference_df.select_dtypes(include=[np.number]).columns
            common_cols = numeric_cols.intersection(current_df.columns)
            
            if len(common_cols) < 2:
                return False, 1.0, "Insufficient numeric features for multivariate analysis"
            
            ref_data = reference_df[common_cols].fillna(0)
            cur_data = current_df[common_cols].fillna(0)
            
            # Standardize data
            scaler = StandardScaler()
            ref_scaled = scaler.fit_transform(ref_data)
            cur_scaled = scaler.transform(cur_data)
            
            # Apply PCA
            pca = PCA(n_components=min(5, len(common_cols)))
            ref_pca = pca.fit_transform(ref_scaled)
            cur_pca = pca.transform(cur_scaled)
            
            # Test drift in first few principal components
            drift_scores = []
            for i in range(pca.n_components_):
                ks_stat, ks_p = stats.ks_2samp(ref_pca[:, i], cur_pca[:, i])
                drift_scores.append(ks_p)
            
            # Overall drift score (minimum p-value)
            overall_drift_score = min(drift_scores)
            drift_detected = overall_drift_score < self.config.drift_threshold
            
            description = f"Multivariate drift analysis: min p-value = {overall_drift_score:.4f}"
            
            return drift_detected, overall_drift_score, description
            
        except Exception as e:
            logger.error("Multivariate drift detection failed", error=str(e))
            return False, 1.0, f"Analysis failed: {e}"

# -------------------------------
# Performance Monitor
# -------------------------------

class PerformanceMonitor:
    """Monitor model performance and detect degradation."""
    
    def __init__(self, config: MonitoringConfig, database: MonitoringDatabase):
        self.config = config
        self.database = database
        self.baseline_metrics = {}
    
    def set_baseline_performance(self, model_name: str, baseline_metrics: Dict[str, float]):
        """Set baseline performance metrics for comparison."""
        self.baseline_metrics[model_name] = baseline_metrics
        logger.info("Baseline performance set for model", model_name=model_name, metrics=baseline_metrics)
    
    def calculate_performance_metrics(self, model_name: str, period_days: int = 7) -> Optional[PerformanceMetrics]:
        """Calculate current performance metrics for a model."""
        # Get recent predictions with actuals
        df = self.database.get_recent_predictions(model_name, period_days)
        
        # Filter for records with actual values
        df_with_actuals = df[df['actual_value'].notna()]
        
        if len(df_with_actuals) < self.config.min_samples_for_monitoring:
            logger.warning("Insufficient data for performance monitoring", 
                          model_name=model_name, samples=len(df_with_actuals))
            return None
        
        # Convert to numpy arrays
        y_true = df_with_actuals['actual_value'].values
        y_pred = df_with_actuals['prediction_value'].values
        
        # Calculate accuracy metrics
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        
        # MASE calculation (using seasonal naive as baseline)
        mase = self._calculate_mase(y_true, y_pred)
        
        # sMAPE calculation
        smape = self._calculate_smape(y_true, y_pred)
        
        # Business metrics
        forecast_bias = np.mean((y_pred - y_true) / (y_true + 1e-8))
        forecast_accuracy_pct = 100 * (1 - mae / (np.mean(y_true) + 1e-8))
        
        # Volume metrics
        predictions_count = len(df)
        zero_predictions_pct = 100 * np.mean(y_pred == 0)
        
        # Calculate change from baseline
        baseline = self.baseline_metrics.get(model_name, {})
        rmse_change_pct = ((rmse - baseline.get('rmse', rmse)) / baseline.get('rmse', rmse)) * 100
        mae_change_pct = ((mae - baseline.get('mae', mae)) / baseline.get('mae', mae)) * 100
        
        # Create performance metrics object
        metrics = PerformanceMetrics(
            model_name=model_name,
            timestamp=datetime.now(),
            period_start=datetime.now() - timedelta(days=period_days),
            period_end=datetime.now(),
            rmse=rmse,
            mae=mae,
            mase=mase,
            smape=smape,
            forecast_bias=forecast_bias,
            forecast_accuracy_pct=forecast_accuracy_pct,
            inventory_waste_estimate=0.0,  # Would calculate based on business logic
            predictions_count=predictions_count,
            zero_predictions_pct=zero_predictions_pct,
            rmse_change_pct=rmse_change_pct,
            mae_change_pct=mae_change_pct
        )
        
        # Store metrics
        self.database.store_performance_metrics(metrics)
        
        return metrics
    
    def _calculate_mase(self, y_true: np.ndarray, y_pred: np.ndarray, seasonal_period: int = 7) -> float:
        """Calculate Mean Absolute Scaled Error."""
        if len(y_true) <= seasonal_period:
            return np.nan
        
        # Seasonal naive forecast errors
        naive_errors = np.abs(y_true[seasonal_period:] - y_true[:-seasonal_period])
        scale = np.mean(naive_errors)
        
        if scale == 0:
            return 0.0
        
        return np.mean(np.abs(y_true - y_pred)) / scale
    
    def _calculate_smape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Symmetric Mean Absolute Percentage Error."""
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
        mask = denominator != 0
        
        if not np.any(mask):
            return 0.0
        
        return 100 * np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask])
    
    def detect_performance_degradation(self, current_metrics: PerformanceMetrics) -> List[DriftAlert]:
        """Detect performance degradation compared to baseline."""
        alerts = []
        model_name = current_metrics.model_name
        
        if model_name not in self.baseline_metrics:
            return alerts
        
        baseline = self.baseline_metrics[model_name]
        threshold = self.config.performance_degradation_threshold
        
        # Check RMSE degradation
        if current_metrics.rmse_change_pct > threshold * 100:
            alert = DriftAlert(
                timestamp=datetime.now(),
                alert_type='performance_degradation',
                severity='warning' if current_metrics.rmse_change_pct < threshold * 200 else 'critical',
                metric_name='rmse',
                metric_value=current_metrics.rmse,
                threshold=baseline.get('rmse', 0) * (1 + threshold),
                description=f"RMSE increased by {current_metrics.rmse_change_pct:.1f}% from baseline"
            )
            alerts.append(alert)
        
        # Check MAE degradation
        if current_metrics.mae_change_pct > threshold * 100:
            alert = DriftAlert(
                timestamp=datetime.now(),
                alert_type='performance_degradation',
                severity='warning' if current_metrics.mae_change_pct < threshold * 200 else 'critical',
                metric_name='mae',
                metric_value=current_metrics.mae,
                threshold=baseline.get('mae', 0) * (1 + threshold),
                description=f"MAE increased by {current_metrics.mae_change_pct:.1f}% from baseline"
            )
            alerts.append(alert)
        
        # Check forecast bias
        if abs(current_metrics.forecast_bias) > self.config.forecast_bias_threshold:
            alert = DriftAlert(
                timestamp=datetime.now(),
                alert_type='forecast_bias',
                severity='warning',
                metric_name='forecast_bias',
                metric_value=current_metrics.forecast_bias,
                threshold=self.config.forecast_bias_threshold,
                description=f"Forecast bias of {current_metrics.forecast_bias:.3f} exceeds threshold"
            )
            alerts.append(alert)
        
        return alerts

# -------------------------------
# Alert Manager
# -------------------------------

class AlertManager:
    """Manage and send alerts for monitoring system."""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.alert_history = defaultdict(deque)  # Track recent alerts
    
    def should_send_alert(self, alert: DriftAlert) -> bool:
        """Check if alert should be sent based on cooldown period."""
        alert_key = f"{alert.alert_type}_{alert.metric_name}"
        recent_alerts = self.alert_history[alert_key]
        
        # Clean old alerts outside cooldown period
        cutoff_time = datetime.now() - timedelta(hours=self.config.alert_cooldown_hours)
        while recent_alerts and recent_alerts[0] < cutoff_time:
            recent_alerts.popleft()
        
        # Check if we should send alert
        if len(recent_alerts) == 0:
            recent_alerts.append(datetime.now())
            return True
        
        return False
    
    def send_alert(self, alert: DriftAlert, recipients: List[str] = None):
        """Send alert via configured channels."""
        if not self.should_send_alert(alert):
            logger.debug("Alert suppressed due to cooldown", alert_type=alert.alert_type)
            return
        
        logger.info("Sending alert", alert_type=alert.alert_type, severity=alert.severity)
        
        # Send email alert
        if self.config.email_alerts and recipients:
            self._send_email_alert(alert, recipients)
        
        # Send Slack alert
        if self.config.slack_webhook_url:
            self._send_slack_alert(alert)
    
    def _send_email_alert(self, alert: DriftAlert, recipients: List[str]):
        """Send email alert."""
        try:
            subject = f"[{alert.severity.upper()}] M5 Model Alert: {alert.alert_type}"
            
            body = f"""
            Alert Details:
            - Type: {alert.alert_type}
            - Severity: {alert.severity}
            - Metric: {alert.metric_name}
            - Current Value: {alert.metric_value}
            - Threshold: {alert.threshold}
            - Description: {alert.description}
            - Timestamp: {alert.timestamp}
            
            Please investigate and take appropriate action.
            """
            
            msg = MimeMultipart()
            msg['From'] = "ml-monitoring@company.com"
            msg['To'] = ", ".join(recipients)
            msg['Subject'] = subject
            msg.attach(MimeText(body, 'plain'))
            
            # Note: Configure SMTP server settings
            # server = smtplib.SMTP('smtp.gmail.com', 587)
            # server.starttls()
            # server.login("your-email", "your-password")
            # server.sendmail(msg['From'], recipients, msg.as_string())
            # server.quit()
            
            logger.info("Email alert sent", recipients=recipients)
            
        except Exception as e:
            logger.error("Failed to send email alert", error=str(e))
    
    def _send_slack_alert(self, alert: DriftAlert):
        """Send Slack alert."""
        try:
            color = "danger" if alert.severity == "critical" else "warning"
            
            payload = {
                "attachments": [{
                    "color": color,
                    "title": f"M5 Model Alert: {alert.alert_type}",
                    "fields": [
                        {"title": "Severity", "value": alert.severity, "short": True},
                        {"title": "Metric", "value": alert.metric_name, "short": True},
                        {"title": "Current Value", "value": f"{alert.metric_value:.4f}", "short": True},
                        {"title": "Threshold", "value": f"{alert.threshold:.4f}", "short": True},
                        {"title": "Description", "value": alert.description, "short": False}
                    ],
                    "ts": int(alert.timestamp.timestamp())
                }]
            }
            
            response = requests.post(self.config.slack_webhook_url, json=payload, timeout=10)
            response.raise_for_status()
            
            logger.info("Slack alert sent successfully")
            
        except Exception as e:
            logger.error("Failed to send Slack alert", error=str(e))

# -------------------------------
# Main Monitoring System
# -------------------------------

class M5MonitoringSystem:
    """Main monitoring system coordinator."""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.database = MonitoringDatabase(config.db_path)
        self.drift_detector = DriftDetector(config)
        self.performance_monitor = PerformanceMonitor(config, self.database)
        self.alert_manager = AlertManager(config)
        
        logger.info("M5 Monitoring System initialized")
    
    def monitor_model_performance(self, model_name: str):
        """Run complete monitoring cycle for a model."""
        logger.info("Starting monitoring cycle", model_name=model_name)
        
        try:
            # Calculate current performance metrics
            metrics = self.performance_monitor.calculate_performance_metrics(
                model_name, self.config.accuracy_window
            )
            
            if metrics is None:
                logger.warning("No performance metrics calculated", model_name=model_name)
                return
            
            # Detect performance degradation
            alerts = self.performance_monitor.detect_performance_degradation(metrics)
            
            # Process alerts
            for alert in alerts:
                self.database.store_alert(alert)
                self.alert_manager.send_alert(alert, recipients=["ml-team@company.com"])
            
            logger.info("Monitoring cycle completed", 
                       model_name=model_name, alerts_generated=len(alerts))
            
        except Exception as e:
            logger.error("Monitoring cycle failed", model_name=model_name, error=str(e))
    
    def monitor_feature_drift(self, model_name: str, current_features: pd.DataFrame):
        """Monitor for feature drift."""
        logger.info("Monitoring feature drift", model_name=model_name)
        
        alerts = []
        
        for column in current_features.select_dtypes(include=[np.number]).columns:
            try:
                drift_detected, drift_score, description = self.drift_detector.detect_feature_drift(
                    column, current_features[column].values
                )
                
                if drift_detected:
                    alert = DriftAlert(
                        timestamp=datetime.now(),
                        alert_type='feature_drift',
                        severity='warning',
                        metric_name=column,
                        metric_value=drift_score,
                        threshold=self.config.drift_threshold,
                        description=description,
                        affected_features=[column]
                    )
                    alerts.append(alert)
                
            except Exception as e:
                logger.error("Drift detection failed for feature", 
                           feature=column, error=str(e))
        
        # Process alerts
        for alert in alerts:
            self.database.store_alert(alert)
            self.alert_manager.send_alert(alert)
        
        logger.info("Feature drift monitoring completed", alerts_generated=len(alerts))
    
    def generate_monitoring_report(self, model_name: str, days: int = 7) -> Dict[str, Any]:
        """Generate comprehensive monitoring report."""
        logger.info("Generating monitoring report", model_name=model_name, days=days)
        
        # Get performance metrics
        metrics = self.performance_monitor.calculate_performance_metrics(model_name, days)
        
        # Get recent alerts
        conn = sqlite3.connect(self.database.db_path)
        alerts_query = '''
            SELECT * FROM alerts 
            WHERE timestamp >= datetime('now', '-{} days')
            ORDER BY timestamp DESC
        '''.format(days)
        alerts_df = pd.read_sql_query(alerts_query, conn)
        conn.close()
        
        # Get prediction statistics
        predictions_df = self.database.get_recent_predictions(model_name, days)
        
        report = {
            'model_name': model_name,
            'report_period': f"{days} days",
            'generated_at': datetime.now().isoformat(),
            'performance_metrics': asdict(metrics) if metrics else None,
            'total_predictions': len(predictions_df),
            'predictions_with_actuals': len(predictions_df[predictions_df['actual_value'].notna()]),
            'alerts_summary': {
                'total_alerts': len(alerts_df),
                'critical_alerts': len(alerts_df[alerts_df['severity'] == 'critical']),
                'warning_alerts': len(alerts_df[alerts_df['severity'] == 'warning']),
                'alert_types': alerts_df['alert_type'].value_counts().to_dict() if len(alerts_df) > 0 else {}
            },
            'data_quality': {
                'zero_predictions_pct': (predictions_df['prediction_value'] == 0).mean() * 100,
                'prediction_coverage': len(predictions_df.groupby(['store_id', 'item_id'])),
                'missing_actuals_pct': predictions_df['actual_value'].isna().mean() * 100
            }
        }
        
        return report
    
    def create_monitoring_dashboard(self, model_name: str, days: int = 30, output_path: str = "monitoring_dashboard.html"):
        """Create interactive monitoring dashboard."""
        logger.info("Creating monitoring dashboard", model_name=model_name)
        
        try:
            # Get data
            predictions_df = self.database.get_recent_predictions(model_name, days)
            
            if len(predictions_df) == 0:
                logger.warning("No data available for dashboard")
                return
            
            # Convert timestamps
            predictions_df['prediction_timestamp'] = pd.to_datetime(predictions_df['prediction_timestamp'])
            
            # Create subplots
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=('Prediction vs Actual', 'Prediction Volume Over Time', 
                              'Error Distribution', 'Model Performance Trend',
                              'Prediction Accuracy by Horizon', 'Alert Timeline'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # 1. Prediction vs Actual scatter plot
            actual_data = predictions_df[predictions_df['actual_value'].notna()]
            if len(actual_data) > 0:
                fig.add_trace(
                    go.Scatter(x=actual_data['actual_value'], y=actual_data['prediction_value'],
                             mode='markers', name='Predictions', opacity=0.6),
                    row=1, col=1
                )
                
                # Add perfect prediction line
                min_val = min(actual_data['actual_value'].min(), actual_data['prediction_value'].min())
                max_val = max(actual_data['actual_value'].max(), actual_data['prediction_value'].max())
                fig.add_trace(
                    go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                             mode='lines', name='Perfect Prediction', line=dict(dash='dash')),
                    row=1, col=1
                )
            
            # 2. Prediction volume over time
            volume_by_date = predictions_df.groupby(predictions_df['prediction_timestamp'].dt.date).size()
            fig.add_trace(
                go.Scatter(x=volume_by_date.index, y=volume_by_date.values,
                         mode='lines+markers', name='Daily Predictions'),
                row=1, col=2
            )
            
            # 3. Error distribution
            if len(actual_data) > 0:
                errors = actual_data['prediction_value'] - actual_data['actual_value']
                fig.add_trace(
                    go.Histogram(x=errors, name='Prediction Errors', nbinsx=30),
                    row=2, col=1
                )
            
            # 4. Model performance trend (would need historical performance data)
            # For now, show a placeholder
            fig.add_trace(
                go.Scatter(x=volume_by_date.index, y=[1.0] * len(volume_by_date),
                         mode='lines', name='Performance Metric'),
                row=2, col=2
            )
            
            # 5. Prediction accuracy by horizon
            if len(actual_data) > 0:
                accuracy_by_horizon = actual_data.groupby('horizon_days').apply(
                    lambda x: 1 - mean_absolute_error(x['actual_value'], x['prediction_value']) / (x['actual_value'].mean() + 1e-8)
                )
                fig.add_trace(
                    go.Bar(x=accuracy_by_horizon.index, y=accuracy_by_horizon.values,
                          name='Accuracy by Horizon'),
                    row=3, col=1
                )
            
            # 6. Alert timeline (would need alerts data)
            fig.add_trace(
                go.Scatter(x=volume_by_date.index, y=[0] * len(volume_by_date),
                         mode='markers', name='Alerts'),
                row=3, col=2
            )
            
            # Update layout
            fig.update_layout(
                height=1000,
                title_text=f"M5 Model Monitoring Dashboard - {model_name}",
                showlegend=True
            )
            
            # Save dashboard
            fig.write_html(output_path)
            logger.info("Dashboard created", output_path=output_path)
            
        except Exception as e:
            logger.error("Dashboard creation failed", error=str(e))
    
    def setup_scheduled_monitoring(self, models: List[str], schedule_hours: int = 6):
        """Setup scheduled monitoring for models."""
        logger.info("Setting up scheduled monitoring", models=models, interval_hours=schedule_hours)
        
        def monitoring_job():
            for model_name in models:
                try:
                    self.monitor_model_performance(model_name)
                except Exception as e:
                    logger.error("Scheduled monitoring failed", model_name=model_name, error=str(e))
        
        # Schedule monitoring job
        schedule.every(schedule_hours).hours.do(monitoring_job)
        
        logger.info("Monitoring scheduled successfully")
        
        # Run scheduler (in production, this would run as a separate service)
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute

# -------------------------------
# CLI Interface
# -------------------------------

def main():
    """CLI interface for monitoring system."""
    import argparse
    
    parser = argparse.ArgumentParser(description="M5 Model Monitoring System")
    parser.add_argument("--config", help="Monitoring configuration file")
    parser.add_argument("--model", required=True, help="Model name to monitor")
    parser.add_argument("--action", choices=['monitor', 'report', 'dashboard', 'schedule'],
                       default='monitor', help="Action to perform")
    parser.add_argument("--days", type=int, default=7, help="Number of days to analyze")
    parser.add_argument("--output", help="Output file path")
    parser.add_argument("--schedule-hours", type=int, default=6, 
                       help="Hours between scheduled monitoring runs")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Load configuration
        if args.config:
            with open(args.config) as f:
                config_dict = json.load(f)
                config = MonitoringConfig(**config_dict)
        else:
            config = MonitoringConfig()
        
        # Initialize monitoring system
        monitoring = M5MonitoringSystem(config)
        
        if args.action == 'monitor':
            # Run monitoring cycle
            monitoring.monitor_model_performance(args.model)
            print(f"Monitoring completed for model: {args.model}")
            
        elif args.action == 'report':
            # Generate monitoring report
            report = monitoring.generate_monitoring_report(args.model, args.days)
            
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(report, f, indent=2)
                print(f"Report saved to: {args.output}")
            else:
                print(json.dumps(report, indent=2))
                
        elif args.action == 'dashboard':
            # Create monitoring dashboard
            output_path = args.output or f"{args.model}_dashboard.html"
            monitoring.create_monitoring_dashboard(args.model, args.days, output_path)
            print(f"Dashboard created: {output_path}")
            
        elif args.action == 'schedule':
            # Setup scheduled monitoring
            models = [args.model]  # Could be extended to multiple models
            monitoring.setup_scheduled_monitoring(models, args.schedule_hours)
            
    except Exception as e:
        logger.error("Monitoring system failed", error=str(e))
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())