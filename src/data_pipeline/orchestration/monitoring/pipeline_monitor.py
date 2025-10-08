"""
Pipeline Monitor

This module provides monitoring capabilities for the PBF-LB/M data pipeline.
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PipelineStatus(Enum):
    """Pipeline status enumeration."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"

class MetricType(Enum):
    """Metric type enumeration."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"

@dataclass
class Metric:
    """Metric data class."""
    name: str
    value: float
    metric_type: MetricType
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    unit: str = ""

@dataclass
class PipelineHealth:
    """Pipeline health data class."""
    pipeline_name: str
    status: PipelineStatus
    overall_score: float
    metrics: Dict[str, float] = field(default_factory=dict)
    issues: List[str] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.now)

@dataclass
class Alert:
    """Alert data class."""
    id: str
    pipeline_name: str
    alert_type: str
    severity: str
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    resolved: bool = False
    resolved_at: Optional[datetime] = None

class PipelineMonitor:
    """
    Monitors pipeline health and performance.
    """
    
    def __init__(self):
        self.config = get_pipeline_config()
        self.metrics: Dict[str, List[Metric]] = {}
        self.health_status: Dict[str, PipelineHealth] = {}
        self.alerts: List[Alert] = []
        self.monitoring_thread: Optional[threading.Thread] = None
        self.is_monitoring = False
        self.monitoring_interval = 60  # seconds
        self.alert_callbacks: List[Callable[[Alert], None]] = []
        
    def start_monitoring(self) -> bool:
        """
        Start pipeline monitoring.
        
        Returns:
            bool: True if monitoring started successfully, False otherwise
        """
        try:
            if self.is_monitoring:
                logger.warning("Pipeline monitoring is already running")
                return False
            
            self.is_monitoring = True
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitoring_thread.start()
            
            logger.info("Pipeline monitoring started")
            return True
            
        except Exception as e:
            logger.error(f"Error starting pipeline monitoring: {e}")
            return False
    
    def stop_monitoring(self) -> bool:
        """
        Stop pipeline monitoring.
        
        Returns:
            bool: True if monitoring stopped successfully, False otherwise
        """
        try:
            if not self.is_monitoring:
                logger.warning("Pipeline monitoring is not running")
                return False
            
            self.is_monitoring = False
            
            if self.monitoring_thread and self.monitoring_thread.is_alive():
                self.monitoring_thread.join(timeout=10)
            
            logger.info("Pipeline monitoring stopped")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping pipeline monitoring: {e}")
            return False
    
    def add_metric(self, metric: Metric) -> bool:
        """
        Add a metric to the monitor.
        
        Args:
            metric: The metric to add
            
        Returns:
            bool: True if metric was added successfully, False otherwise
        """
        try:
            if metric.name not in self.metrics:
                self.metrics[metric.name] = []
            
            self.metrics[metric.name].append(metric)
            
            # Keep only last 1000 metrics per name
            if len(self.metrics[metric.name]) > 1000:
                self.metrics[metric.name] = self.metrics[metric.name][-1000:]
            
            logger.debug(f"Added metric {metric.name}: {metric.value}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding metric {metric.name}: {e}")
            return False
    
    def get_metric(self, metric_name: str, labels: Optional[Dict[str, str]] = None) -> Optional[Metric]:
        """
        Get the latest metric by name and optional labels.
        
        Args:
            metric_name: The metric name
            labels: Optional labels to filter by
            
        Returns:
            Metric: The latest matching metric, or None if not found
        """
        try:
            if metric_name not in self.metrics:
                return None
            
            metrics = self.metrics[metric_name]
            
            if labels:
                # Filter by labels
                for metric in reversed(metrics):  # Start from latest
                    if all(metric.labels.get(k) == v for k, v in labels.items()):
                        return metric
            else:
                # Return latest metric
                return metrics[-1] if metrics else None
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting metric {metric_name}: {e}")
            return None
    
    def get_metric_history(self, metric_name: str, hours: int = 24) -> List[Metric]:
        """
        Get metric history for a specific time period.
        
        Args:
            metric_name: The metric name
            hours: Number of hours to look back
            
        Returns:
            List[Metric]: List of metrics within the time period
        """
        try:
            if metric_name not in self.metrics:
                return []
            
            cutoff_time = datetime.now() - timedelta(hours=hours)
            metrics = self.metrics[metric_name]
            
            return [metric for metric in metrics if metric.timestamp >= cutoff_time]
            
        except Exception as e:
            logger.error(f"Error getting metric history for {metric_name}: {e}")
            return []
    
    def get_pipeline_health(self, pipeline_name: str) -> Optional[PipelineHealth]:
        """
        Get the health status of a pipeline.
        
        Args:
            pipeline_name: The pipeline name
            
        Returns:
            PipelineHealth: The pipeline health status, or None if not found
        """
        return self.health_status.get(pipeline_name)
    
    def get_all_pipeline_health(self) -> Dict[str, PipelineHealth]:
        """
        Get health status for all pipelines.
        
        Returns:
            Dict[str, PipelineHealth]: Dictionary of pipeline health statuses
        """
        return self.health_status.copy()
    
    def add_alert_callback(self, callback: Callable[[Alert], None]) -> bool:
        """
        Add an alert callback function.
        
        Args:
            callback: The callback function to call when alerts are generated
            
        Returns:
            bool: True if callback was added successfully, False otherwise
        """
        try:
            self.alert_callbacks.append(callback)
            logger.info("Added alert callback")
            return True
            
        except Exception as e:
            logger.error(f"Error adding alert callback: {e}")
            return False
    
    def create_alert(self, pipeline_name: str, alert_type: str, severity: str, message: str) -> Alert:
        """
        Create a new alert.
        
        Args:
            pipeline_name: The pipeline name
            alert_type: The alert type
            severity: The alert severity
            message: The alert message
            
        Returns:
            Alert: The created alert
        """
        try:
            alert_id = f"{pipeline_name}_{alert_type}_{int(time.time())}"
            
            alert = Alert(
                id=alert_id,
                pipeline_name=pipeline_name,
                alert_type=alert_type,
                severity=severity,
                message=message
            )
            
            self.alerts.append(alert)
            
            # Call alert callbacks
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    logger.error(f"Error in alert callback: {e}")
            
            logger.info(f"Created alert {alert_id}: {message}")
            return alert
            
        except Exception as e:
            logger.error(f"Error creating alert: {e}")
            raise
    
    def resolve_alert(self, alert_id: str) -> bool:
        """
        Resolve an alert.
        
        Args:
            alert_id: The alert ID
            
        Returns:
            bool: True if alert was resolved successfully, False otherwise
        """
        try:
            for alert in self.alerts:
                if alert.id == alert_id:
                    alert.resolved = True
                    alert.resolved_at = datetime.now()
                    logger.info(f"Resolved alert {alert_id}")
                    return True
            
            logger.warning(f"Alert {alert_id} not found")
            return False
            
        except Exception as e:
            logger.error(f"Error resolving alert {alert_id}: {e}")
            return False
    
    def get_active_alerts(self) -> List[Alert]:
        """
        Get all active (unresolved) alerts.
        
        Returns:
            List[Alert]: List of active alerts
        """
        return [alert for alert in self.alerts if not alert.resolved]
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """
        Get the current monitoring status.
        
        Returns:
            Dict[str, Any]: Monitoring status information
        """
        try:
            total_metrics = sum(len(metrics) for metrics in self.metrics.values())
            active_alerts = len(self.get_active_alerts())
            total_alerts = len(self.alerts)
            
            pipeline_health_summary = {}
            for pipeline_name, health in self.health_status.items():
                pipeline_health_summary[pipeline_name] = {
                    "status": health.status.value,
                    "overall_score": health.overall_score,
                    "issues_count": len(health.issues)
                }
            
            return {
                "is_monitoring": self.is_monitoring,
                "monitoring_interval": self.monitoring_interval,
                "total_metrics": total_metrics,
                "active_alerts": active_alerts,
                "total_alerts": total_alerts,
                "pipeline_health": pipeline_health_summary
            }
            
        except Exception as e:
            logger.error(f"Error getting monitoring status: {e}")
            return {}
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                # Collect metrics from all pipelines
                self._collect_pipeline_metrics()
                
                # Update pipeline health
                self._update_pipeline_health()
                
                # Check for alerts
                self._check_alerts()
                
                # Sleep for monitoring interval
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.monitoring_interval)
    
    def _collect_pipeline_metrics(self):
        """Collect metrics from all pipelines."""
        try:
            # This is a placeholder implementation
            # In a real system, you would collect metrics from actual pipeline components
            
            # Example: Collect throughput metrics
            throughput_metric = Metric(
                name="pipeline_throughput",
                value=1000.0,  # records per minute
                metric_type=MetricType.GAUGE,
                labels={"pipeline": "pbf_process"},
                unit="records/min"
            )
            self.add_metric(throughput_metric)
            
            # Example: Collect latency metrics
            latency_metric = Metric(
                name="pipeline_latency",
                value=5.2,  # seconds
                metric_type=MetricType.HISTOGRAM,
                labels={"pipeline": "pbf_process"},
                unit="seconds"
            )
            self.add_metric(latency_metric)
            
            # Example: Collect error rate metrics
            error_rate_metric = Metric(
                name="pipeline_error_rate",
                value=0.02,  # 2%
                metric_type=MetricType.GAUGE,
                labels={"pipeline": "pbf_process"},
                unit="percent"
            )
            self.add_metric(error_rate_metric)
            
        except Exception as e:
            logger.error(f"Error collecting pipeline metrics: {e}")
    
    def _update_pipeline_health(self):
        """Update pipeline health status."""
        try:
            # This is a placeholder implementation
            # In a real system, you would calculate health based on actual metrics
            
            pipeline_names = ["pbf_process", "ispm_monitoring", "ct_scan", "powder_bed"]
            
            for pipeline_name in pipeline_names:
                # Get relevant metrics
                throughput_metric = self.get_metric("pipeline_throughput", {"pipeline": pipeline_name})
                latency_metric = self.get_metric("pipeline_latency", {"pipeline": pipeline_name})
                error_rate_metric = self.get_metric("pipeline_error_rate", {"pipeline": pipeline_name})
                
                # Calculate health score
                health_score = 1.0
                issues = []
                
                if throughput_metric and throughput_metric.value < 100:
                    health_score -= 0.3
                    issues.append("Low throughput")
                
                if latency_metric and latency_metric.value > 10:
                    health_score -= 0.2
                    issues.append("High latency")
                
                if error_rate_metric and error_rate_metric.value > 0.05:
                    health_score -= 0.4
                    issues.append("High error rate")
                
                # Determine status
                if health_score >= 0.8:
                    status = PipelineStatus.HEALTHY
                elif health_score >= 0.6:
                    status = PipelineStatus.DEGRADED
                else:
                    status = PipelineStatus.UNHEALTHY
                
                # Update health status
                self.health_status[pipeline_name] = PipelineHealth(
                    pipeline_name=pipeline_name,
                    status=status,
                    overall_score=health_score,
                    metrics={
                        "throughput": throughput_metric.value if throughput_metric else 0,
                        "latency": latency_metric.value if latency_metric else 0,
                        "error_rate": error_rate_metric.value if error_rate_metric else 0
                    },
                    issues=issues
                )
                
        except Exception as e:
            logger.error(f"Error updating pipeline health: {e}")
    
    def _check_alerts(self):
        """Check for conditions that should trigger alerts."""
        try:
            for pipeline_name, health in self.health_status.items():
                # Check for unhealthy pipelines
                if health.status == PipelineStatus.UNHEALTHY:
                    self.create_alert(
                        pipeline_name=pipeline_name,
                        alert_type="pipeline_unhealthy",
                        severity="critical",
                        message=f"Pipeline {pipeline_name} is unhealthy. Issues: {', '.join(health.issues)}"
                    )
                
                # Check for degraded pipelines
                elif health.status == PipelineStatus.DEGRADED:
                    self.create_alert(
                        pipeline_name=pipeline_name,
                        alert_type="pipeline_degraded",
                        severity="warning",
                        message=f"Pipeline {pipeline_name} is degraded. Issues: {', '.join(health.issues)}"
                    )
                
                # Check for high error rates
                if health.metrics.get("error_rate", 0) > 0.1:
                    self.create_alert(
                        pipeline_name=pipeline_name,
                        alert_type="high_error_rate",
                        severity="high",
                        message=f"Pipeline {pipeline_name} has high error rate: {health.metrics['error_rate']:.2%}"
                    )
                
                # Check for high latency
                if health.metrics.get("latency", 0) > 30:
                    self.create_alert(
                        pipeline_name=pipeline_name,
                        alert_type="high_latency",
                        severity="medium",
                        message=f"Pipeline {pipeline_name} has high latency: {health.metrics['latency']:.2f} seconds"
                    )
                
        except Exception as e:
            logger.error(f"Error checking alerts: {e}")
