"""
Quality Monitor

This module provides real-time quality monitoring capabilities for the PBF-LB/M data pipeline.
"""

import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import json

from src.data_pipeline.config.pipeline_config import get_pipeline_config
from src.data_pipeline.quality.validation.data_quality_service import DataQualityService, QualityProfile

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QualityStatus(Enum):
    """Quality status enumeration."""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    CRITICAL = "critical"

class QualityMetric(Enum):
    """Quality metric enumeration."""
    COMPLETENESS = "completeness"
    ACCURACY = "accuracy"
    CONSISTENCY = "consistency"
    TIMELINESS = "timeliness"
    VALIDITY = "validity"
    UNIQUENESS = "uniqueness"

@dataclass
class QualityMetrics:
    """Quality metrics data class."""
    source_name: str
    metric_name: QualityMetric
    value: float
    threshold: float
    status: QualityStatus
    timestamp: datetime = field(default_factory=datetime.now)
    details: Dict[str, Any] = field(default_factory=dict)

@dataclass
class QualityAlert:
    """Quality alert data class."""
    id: str
    source_name: str
    metric_name: QualityMetric
    current_value: float
    threshold: float
    severity: str
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    resolved: bool = False

@dataclass
class QualityDashboard:
    """Quality dashboard data class."""
    source_name: str
    overall_score: float
    status: QualityStatus
    metrics: List[QualityMetrics] = field(default_factory=list)
    alerts: List[QualityAlert] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.now)

@dataclass
class QualityReport:
    """Quality report data class."""
    report_id: str
    source_name: str
    report_type: str
    period_start: datetime
    period_end: datetime
    overall_score: float
    status: QualityStatus
    summary: Dict[str, Any] = field(default_factory=dict)
    details: Dict[str, Any] = field(default_factory=dict)
    generated_at: datetime = field(default_factory=datetime.now)

class QualityMonitor:
    """
    Real-time quality monitoring service for PBF-LB/M data pipeline.
    """
    
    def __init__(self):
        self.config = get_pipeline_config()
        self.data_quality_service = DataQualityService()
        self.monitoring_thread: Optional[threading.Thread] = None
        self.is_monitoring = False
        self.monitoring_interval = 60  # seconds
        self.quality_metrics: Dict[str, List[QualityMetrics]] = {}
        self.quality_alerts: List[QualityAlert] = []
        self.quality_dashboards: Dict[str, QualityDashboard] = {}
        self.alert_callbacks: List[Callable[[QualityAlert], None]] = []
        
        # Quality thresholds
        self.quality_thresholds = {
            QualityMetric.COMPLETENESS: {"excellent": 0.95, "good": 0.90, "fair": 0.80, "poor": 0.70},
            QualityMetric.ACCURACY: {"excellent": 0.98, "good": 0.95, "fair": 0.90, "poor": 0.80},
            QualityMetric.CONSISTENCY: {"excellent": 0.95, "good": 0.90, "fair": 0.80, "poor": 0.70},
            QualityMetric.TIMELINESS: {"excellent": 0.95, "good": 0.90, "fair": 0.80, "poor": 0.70},
            QualityMetric.VALIDITY: {"excellent": 0.98, "good": 0.95, "fair": 0.90, "poor": 0.80},
            QualityMetric.UNIQUENESS: {"excellent": 0.99, "good": 0.95, "fair": 0.90, "poor": 0.80}
        }
        
    def start_monitoring(self) -> bool:
        """
        Start quality monitoring.
        
        Returns:
            bool: True if monitoring started successfully, False otherwise
        """
        try:
            if self.is_monitoring:
                logger.warning("Quality monitoring is already running")
                return False
            
            self.is_monitoring = True
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitoring_thread.start()
            
            logger.info("Quality monitoring started")
            return True
            
        except Exception as e:
            logger.error(f"Error starting quality monitoring: {e}")
            return False
    
    def stop_monitoring(self) -> bool:
        """
        Stop quality monitoring.
        
        Returns:
            bool: True if monitoring stopped successfully, False otherwise
        """
        try:
            if not self.is_monitoring:
                logger.warning("Quality monitoring is not running")
                return False
            
            self.is_monitoring = False
            
            if self.monitoring_thread and self.monitoring_thread.is_alive():
                self.monitoring_thread.join(timeout=10)
            
            logger.info("Quality monitoring stopped")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping quality monitoring: {e}")
            return False
    
    def monitor_data_quality(self, source_name: str, data: List[Dict[str, Any]]) -> QualityDashboard:
        """
        Monitor data quality for a specific source.
        
        Args:
            source_name: The data source name
            data: The data to monitor
            
        Returns:
            QualityDashboard: Quality monitoring results
        """
        try:
            logger.info(f"Monitoring data quality for {source_name}: {len(data)} records")
            
            # Validate data quality
            if source_name == "pbf_process":
                quality_profile = self.data_quality_service.validate_pbf_process_data(data)
            elif source_name == "ispm_monitoring":
                quality_profile = self.data_quality_service.validate_ispm_monitoring_data(data)
            elif source_name == "ct_scan":
                quality_profile = self.data_quality_service.validate_ct_scan_data(data)
            elif source_name == "powder_bed":
                quality_profile = self.data_quality_service.validate_powder_bed_data(data)
            else:
                raise ValueError(f"Unknown source: {source_name}")
            
            # Calculate quality metrics
            metrics = self._calculate_quality_metrics(source_name, quality_profile, data)
            
            # Determine overall status
            overall_score = quality_profile.overall_score
            status = self._determine_quality_status(overall_score)
            
            # Check for alerts
            alerts = self._check_quality_alerts(source_name, metrics)
            
            # Create dashboard
            dashboard = QualityDashboard(
                source_name=source_name,
                overall_score=overall_score,
                status=status,
                metrics=metrics,
                alerts=alerts
            )
            
            # Store dashboard
            self.quality_dashboards[source_name] = dashboard
            
            # Store metrics
            if source_name not in self.quality_metrics:
                self.quality_metrics[source_name] = []
            self.quality_metrics[source_name].extend(metrics)
            
            # Store alerts
            self.quality_alerts.extend(alerts)
            
            # Send alerts
            for alert in alerts:
                self._send_quality_alert(alert)
            
            logger.info(f"Quality monitoring completed for {source_name}. Overall score: {overall_score:.2f}")
            return dashboard
            
        except Exception as e:
            logger.error(f"Error monitoring data quality for {source_name}: {e}")
            raise
    
    def get_quality_dashboard(self, source_name: str) -> Optional[QualityDashboard]:
        """
        Get quality dashboard for a specific source.
        
        Args:
            source_name: The data source name
            
        Returns:
            QualityDashboard: The quality dashboard, or None if not found
        """
        return self.quality_dashboards.get(source_name)
    
    def get_all_quality_dashboards(self) -> Dict[str, QualityDashboard]:
        """
        Get all quality dashboards.
        
        Returns:
            Dict[str, QualityDashboard]: All quality dashboards
        """
        return self.quality_dashboards.copy()
    
    def get_quality_metrics(self, source_name: str, hours: int = 24) -> List[QualityMetrics]:
        """
        Get quality metrics for a specific source and time period.
        
        Args:
            source_name: The data source name
            hours: Number of hours to look back
            
        Returns:
            List[QualityMetrics]: List of quality metrics
        """
        try:
            if source_name not in self.quality_metrics:
                return []
            
            cutoff_time = datetime.now() - timedelta(hours=hours)
            metrics = self.quality_metrics[source_name]
            
            return [metric for metric in metrics if metric.timestamp >= cutoff_time]
            
        except Exception as e:
            logger.error(f"Error getting quality metrics for {source_name}: {e}")
            return []
    
    def get_active_alerts(self) -> List[QualityAlert]:
        """
        Get all active (unresolved) quality alerts.
        
        Returns:
            List[QualityAlert]: List of active alerts
        """
        return [alert for alert in self.quality_alerts if not alert.resolved]
    
    def resolve_alert(self, alert_id: str) -> bool:
        """
        Resolve a quality alert.
        
        Args:
            alert_id: The alert ID
            
        Returns:
            bool: True if alert was resolved successfully, False otherwise
        """
        try:
            for alert in self.quality_alerts:
                if alert.id == alert_id:
                    alert.resolved = True
                    logger.info(f"Resolved quality alert {alert_id}")
                    return True
            
            logger.warning(f"Quality alert {alert_id} not found")
            return False
            
        except Exception as e:
            logger.error(f"Error resolving quality alert {alert_id}: {e}")
            return False
    
    def generate_quality_report(self, source_name: str, report_type: str, 
                              period_hours: int = 24) -> QualityReport:
        """
        Generate a quality report for a specific source.
        
        Args:
            source_name: The data source name
            report_type: The type of report (daily, weekly, monthly)
            period_hours: Number of hours to include in the report
            
        Returns:
            QualityReport: The generated quality report
        """
        try:
            logger.info(f"Generating {report_type} quality report for {source_name}")
            
            period_end = datetime.now()
            period_start = period_end - timedelta(hours=period_hours)
            
            # Get metrics for the period
            metrics = self.get_quality_metrics(source_name, period_hours)
            
            if not metrics:
                return QualityReport(
                    report_id=f"{source_name}_{report_type}_{int(time.time())}",
                    source_name=source_name,
                    report_type=report_type,
                    period_start=period_start,
                    period_end=period_end,
                    overall_score=0.0,
                    status=QualityStatus.CRITICAL,
                    summary={"message": "No quality metrics available for the period"}
                )
            
            # Calculate summary statistics
            summary = self._calculate_report_summary(metrics)
            
            # Determine overall status
            overall_score = summary.get("average_score", 0.0)
            status = self._determine_quality_status(overall_score)
            
            # Create report
            report = QualityReport(
                report_id=f"{source_name}_{report_type}_{int(time.time())}",
                source_name=source_name,
                report_type=report_type,
                period_start=period_start,
                period_end=period_end,
                overall_score=overall_score,
                status=status,
                summary=summary,
                details={
                    "total_metrics": len(metrics),
                    "metrics_by_type": self._group_metrics_by_type(metrics),
                    "alerts_count": len([alert for alert in self.quality_alerts 
                                       if not alert.resolved and alert.source_name == source_name])
                }
            )
            
            logger.info(f"Generated quality report for {source_name}. Overall score: {overall_score:.2f}")
            return report
            
        except Exception as e:
            logger.error(f"Error generating quality report for {source_name}: {e}")
            raise
    
    def add_alert_callback(self, callback: Callable[[QualityAlert], None]) -> bool:
        """
        Add an alert callback function.
        
        Args:
            callback: The callback function to call when alerts are generated
            
        Returns:
            bool: True if callback was added successfully, False otherwise
        """
        try:
            self.alert_callbacks.append(callback)
            logger.info("Added quality alert callback")
            return True
            
        except Exception as e:
            logger.error(f"Error adding quality alert callback: {e}")
            return False
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """
        Get the current monitoring status.
        
        Returns:
            Dict[str, Any]: Monitoring status information
        """
        try:
            total_sources = len(self.quality_dashboards)
            active_alerts = len(self.get_active_alerts())
            total_alerts = len(self.quality_alerts)
            
            # Calculate overall quality score
            if self.quality_dashboards:
                overall_score = sum(dashboard.overall_score for dashboard in self.quality_dashboards.values()) / total_sources
            else:
                overall_score = 0.0
            
            # Get status distribution
            status_distribution = {}
            for dashboard in self.quality_dashboards.values():
                status = dashboard.status.value
                status_distribution[status] = status_distribution.get(status, 0) + 1
            
            return {
                "is_monitoring": self.is_monitoring,
                "monitoring_interval": self.monitoring_interval,
                "total_sources": total_sources,
                "overall_quality_score": overall_score,
                "active_alerts": active_alerts,
                "total_alerts": total_alerts,
                "status_distribution": status_distribution,
                "last_updated": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting monitoring status: {e}")
            return {}
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                # This is a placeholder for continuous monitoring
                # In a real system, you would continuously monitor data quality
                
                # Sleep for monitoring interval
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in quality monitoring loop: {e}")
                time.sleep(self.monitoring_interval)
    
    def _calculate_quality_metrics(self, source_name: str, quality_profile: QualityProfile, 
                                 data: List[Dict[str, Any]]) -> List[QualityMetrics]:
        """Calculate quality metrics from quality profile."""
        try:
            metrics = []
            
            # Completeness metric
            completeness = quality_profile.valid_records / quality_profile.total_records if quality_profile.total_records > 0 else 0.0
            metrics.append(QualityMetrics(
                source_name=source_name,
                metric_name=QualityMetric.COMPLETENESS,
                value=completeness,
                threshold=self.quality_thresholds[QualityMetric.COMPLETENESS]["good"],
                status=self._determine_metric_status(completeness, QualityMetric.COMPLETENESS),
                details={
                    "valid_records": quality_profile.valid_records,
                    "total_records": quality_profile.total_records,
                    "invalid_records": quality_profile.invalid_records
                }
            ))
            
            # Accuracy metric (based on overall score)
            accuracy = quality_profile.overall_score
            metrics.append(QualityMetrics(
                source_name=source_name,
                metric_name=QualityMetric.ACCURACY,
                value=accuracy,
                threshold=self.quality_thresholds[QualityMetric.ACCURACY]["good"],
                status=self._determine_metric_status(accuracy, QualityMetric.ACCURACY),
                details={
                    "overall_score": accuracy,
                    "validation_level": quality_profile.validation_level.value
                }
            ))
            
            # Consistency metric (based on validation results)
            consistency = self._calculate_consistency_metric(quality_profile)
            metrics.append(QualityMetrics(
                source_name=source_name,
                metric_name=QualityMetric.CONSISTENCY,
                value=consistency,
                threshold=self.quality_thresholds[QualityMetric.CONSISTENCY]["good"],
                status=self._determine_metric_status(consistency, QualityMetric.CONSISTENCY),
                details={
                    "passed_rules": len([r for r in quality_profile.results if r.passed]),
                    "total_rules": len(quality_profile.results)
                }
            ))
            
            # Timeliness metric (placeholder - would be based on data freshness)
            timeliness = 0.95  # Placeholder value
            metrics.append(QualityMetrics(
                source_name=source_name,
                metric_name=QualityMetric.TIMELINESS,
                value=timeliness,
                threshold=self.quality_thresholds[QualityMetric.TIMELINESS]["good"],
                status=self._determine_metric_status(timeliness, QualityMetric.TIMELINESS),
                details={"data_freshness_minutes": 5}  # Placeholder
            ))
            
            # Validity metric (based on validation results)
            validity = self._calculate_validity_metric(quality_profile)
            metrics.append(QualityMetrics(
                source_name=source_name,
                metric_name=QualityMetric.VALIDITY,
                value=validity,
                threshold=self.quality_thresholds[QualityMetric.VALIDITY]["good"],
                status=self._determine_metric_status(validity, QualityMetric.VALIDITY),
                details={
                    "validation_results": len(quality_profile.results)
                }
            ))
            
            # Uniqueness metric (placeholder - would be based on duplicate detection)
            uniqueness = 0.98  # Placeholder value
            metrics.append(QualityMetrics(
                source_name=source_name,
                metric_name=QualityMetric.UNIQUENESS,
                value=uniqueness,
                threshold=self.quality_thresholds[QualityMetric.UNIQUENESS]["good"],
                status=self._determine_metric_status(uniqueness, QualityMetric.UNIQUENESS),
                details={"duplicate_count": 0}  # Placeholder
            ))
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating quality metrics: {e}")
            return []
    
    def _calculate_consistency_metric(self, quality_profile: QualityProfile) -> float:
        """Calculate consistency metric from quality profile."""
        try:
            if not quality_profile.results:
                return 0.0
            
            # Calculate consistency based on rule compliance
            passed_rules = len([r for r in quality_profile.results if r.passed])
            total_rules = len(quality_profile.results)
            
            return passed_rules / total_rules if total_rules > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating consistency metric: {e}")
            return 0.0
    
    def _calculate_validity_metric(self, quality_profile: QualityProfile) -> float:
        """Calculate validity metric from quality profile."""
        try:
            if not quality_profile.results:
                return 0.0
            
            # Calculate validity based on validation results
            total_score = sum(r.score for r in quality_profile.results)
            total_rules = len(quality_profile.results)
            
            return total_score / total_rules if total_rules > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating validity metric: {e}")
            return 0.0
    
    def _determine_metric_status(self, value: float, metric: QualityMetric) -> QualityStatus:
        """Determine quality status based on metric value and thresholds."""
        try:
            thresholds = self.quality_thresholds[metric]
            
            if value >= thresholds["excellent"]:
                return QualityStatus.EXCELLENT
            elif value >= thresholds["good"]:
                return QualityStatus.GOOD
            elif value >= thresholds["fair"]:
                return QualityStatus.FAIR
            elif value >= thresholds["poor"]:
                return QualityStatus.POOR
            else:
                return QualityStatus.CRITICAL
                
        except Exception as e:
            logger.error(f"Error determining metric status: {e}")
            return QualityStatus.CRITICAL
    
    def _determine_quality_status(self, overall_score: float) -> QualityStatus:
        """Determine overall quality status based on score."""
        try:
            if overall_score >= 0.95:
                return QualityStatus.EXCELLENT
            elif overall_score >= 0.90:
                return QualityStatus.GOOD
            elif overall_score >= 0.80:
                return QualityStatus.FAIR
            elif overall_score >= 0.70:
                return QualityStatus.POOR
            else:
                return QualityStatus.CRITICAL
                
        except Exception as e:
            logger.error(f"Error determining quality status: {e}")
            return QualityStatus.CRITICAL
    
    def _check_quality_alerts(self, source_name: str, metrics: List[QualityMetrics]) -> List[QualityAlert]:
        """Check for quality alerts based on metrics."""
        try:
            alerts = []
            
            for metric in metrics:
                if metric.status in [QualityStatus.POOR, QualityStatus.CRITICAL]:
                    alert = QualityAlert(
                        id=f"{source_name}_{metric.metric_name.value}_{int(time.time())}",
                        source_name=source_name,
                        metric_name=metric.metric_name,
                        current_value=metric.value,
                        threshold=metric.threshold,
                        severity=metric.status.value,
                        message=f"Quality alert: {metric.metric_name.value} is {metric.value:.3f} (threshold: {metric.threshold:.3f})"
                    )
                    alerts.append(alert)
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error checking quality alerts: {e}")
            return []
    
    def _send_quality_alert(self, alert: QualityAlert):
        """Send quality alert to callbacks."""
        try:
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    logger.error(f"Error in quality alert callback: {e}")
                    
        except Exception as e:
            logger.error(f"Error sending quality alert: {e}")
    
    def _calculate_report_summary(self, metrics: List[QualityMetrics]) -> Dict[str, Any]:
        """Calculate summary statistics for quality report."""
        try:
            if not metrics:
                return {}
            
            # Group metrics by type
            metrics_by_type = {}
            for metric in metrics:
                metric_type = metric.metric_name.value
                if metric_type not in metrics_by_type:
                    metrics_by_type[metric_type] = []
                metrics_by_type[metric_type].append(metric.value)
            
            # Calculate summary statistics
            summary = {
                "total_metrics": len(metrics),
                "average_score": sum(metric.value for metric in metrics) / len(metrics),
                "metrics_by_type": {}
            }
            
            for metric_type, values in metrics_by_type.items():
                summary["metrics_by_type"][metric_type] = {
                    "count": len(values),
                    "average": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values)
                }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error calculating report summary: {e}")
            return {}
    
    def _group_metrics_by_type(self, metrics: List[QualityMetrics]) -> Dict[str, int]:
        """Group metrics by type for reporting."""
        try:
            grouped = {}
            for metric in metrics:
                metric_type = metric.metric_name.value
                grouped[metric_type] = grouped.get(metric_type, 0) + 1
            
            return grouped
            
        except Exception as e:
            logger.error(f"Error grouping metrics by type: {e}")
            return {}
