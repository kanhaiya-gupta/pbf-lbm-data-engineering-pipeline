"""
Performance Monitor

This module provides performance monitoring capabilities for the PBF-LB/M data pipeline.
"""

import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import json
import statistics

from src.data_pipeline.config.pipeline_config import get_pipeline_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PerformanceMetric(Enum):
    """Performance metric enumeration."""
    THROUGHPUT = "throughput"
    LATENCY = "latency"
    CPU_USAGE = "cpu_usage"
    MEMORY_USAGE = "memory_usage"
    DISK_IO = "disk_io"
    NETWORK_IO = "network_io"
    ERROR_RATE = "error_rate"
    QUEUE_SIZE = "queue_size"
    PROCESSING_TIME = "processing_time"

class PerformanceLevel(Enum):
    """Performance level enumeration."""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    CRITICAL = "critical"

@dataclass
class PerformanceData:
    """Performance data class."""
    metric_name: PerformanceMetric
    value: float
    unit: str
    timestamp: datetime = field(default_factory=datetime.now)
    labels: Dict[str, str] = field(default_factory=dict)

@dataclass
class PerformanceThreshold:
    """Performance threshold data class."""
    metric_name: PerformanceMetric
    warning_threshold: float
    critical_threshold: float
    unit: str
    is_higher_better: bool = True  # True if higher values are better

@dataclass
class PerformanceAlert:
    """Performance alert data class."""
    id: str
    metric_name: PerformanceMetric
    current_value: float
    threshold_value: float
    alert_level: PerformanceLevel
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    resolved: bool = False

class PerformanceMonitor:
    """
    Monitors system and pipeline performance.
    """
    
    def __init__(self):
        self.config = get_pipeline_config()
        self.performance_data: Dict[PerformanceMetric, List[PerformanceData]] = {}
        self.performance_thresholds: Dict[PerformanceMetric, PerformanceThreshold] = {}
        self.performance_alerts: List[PerformanceAlert] = []
        self.monitoring_thread: Optional[threading.Thread] = None
        self.is_monitoring = False
        self.monitoring_interval = 30  # seconds
        self.alert_callbacks: List[Callable[[PerformanceAlert], None]] = []
        
        # Initialize performance thresholds
        self._initialize_thresholds()
        
    def start_monitoring(self) -> bool:
        """
        Start performance monitoring.
        
        Returns:
            bool: True if monitoring started successfully, False otherwise
        """
        try:
            if self.is_monitoring:
                logger.warning("Performance monitoring is already running")
                return False
            
            self.is_monitoring = True
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitoring_thread.start()
            
            logger.info("Performance monitoring started")
            return True
            
        except Exception as e:
            logger.error(f"Error starting performance monitoring: {e}")
            return False
    
    def stop_monitoring(self) -> bool:
        """
        Stop performance monitoring.
        
        Returns:
            bool: True if monitoring stopped successfully, False otherwise
        """
        try:
            if not self.is_monitoring:
                logger.warning("Performance monitoring is not running")
                return False
            
            self.is_monitoring = False
            
            if self.monitoring_thread and self.monitoring_thread.is_alive():
                self.monitoring_thread.join(timeout=10)
            
            logger.info("Performance monitoring stopped")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping performance monitoring: {e}")
            return False
    
    def add_performance_data(self, data: PerformanceData) -> bool:
        """
        Add performance data.
        
        Args:
            data: The performance data to add
            
        Returns:
            bool: True if data was added successfully, False otherwise
        """
        try:
            if data.metric_name not in self.performance_data:
                self.performance_data[data.metric_name] = []
            
            self.performance_data[data.metric_name].append(data)
            
            # Keep only last 1000 data points per metric
            if len(self.performance_data[data.metric_name]) > 1000:
                self.performance_data[data.metric_name] = self.performance_data[data.metric_name][-1000:]
            
            # Check for performance alerts
            self._check_performance_alerts(data)
            
            logger.debug(f"Added performance data for {data.metric_name}: {data.value}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding performance data: {e}")
            return False
    
    def get_performance_data(self, metric_name: PerformanceMetric, minutes: int = 60) -> List[PerformanceData]:
        """
        Get performance data for a specific metric and time period.
        
        Args:
            metric_name: The metric name
            minutes: Number of minutes to look back
            
        Returns:
            List[PerformanceData]: List of performance data within the time period
        """
        try:
            if metric_name not in self.performance_data:
                return []
            
            cutoff_time = datetime.now() - timedelta(minutes=minutes)
            data = self.performance_data[metric_name]
            
            return [d for d in data if d.timestamp >= cutoff_time]
            
        except Exception as e:
            logger.error(f"Error getting performance data for {metric_name}: {e}")
            return []
    
    def get_performance_summary(self, metric_name: PerformanceMetric, minutes: int = 60) -> Dict[str, Any]:
        """
        Get a performance summary for a specific metric.
        
        Args:
            metric_name: The metric name
            minutes: Number of minutes to look back
            
        Returns:
            Dict[str, Any]: Performance summary
        """
        try:
            data = self.get_performance_data(metric_name, minutes)
            
            if not data:
                return {
                    "metric_name": metric_name.value,
                    "data_points": 0,
                    "time_period_minutes": minutes
                }
            
            values = [d.value for d in data]
            
            summary = {
                "metric_name": metric_name.value,
                "data_points": len(data),
                "time_period_minutes": minutes,
                "current_value": values[-1] if values else 0,
                "average": statistics.mean(values),
                "median": statistics.median(values),
                "min": min(values),
                "max": max(values),
                "std_dev": statistics.stdev(values) if len(values) > 1 else 0,
                "unit": data[0].unit if data else ""
            }
            
            # Add performance level
            summary["performance_level"] = self._calculate_performance_level(metric_name, summary["current_value"])
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting performance summary for {metric_name}: {e}")
            return {}
    
    def get_all_performance_summaries(self, minutes: int = 60) -> Dict[str, Dict[str, Any]]:
        """
        Get performance summaries for all metrics.
        
        Args:
            minutes: Number of minutes to look back
            
        Returns:
            Dict[str, Dict[str, Any]]: Performance summaries for all metrics
        """
        try:
            summaries = {}
            
            for metric_name in PerformanceMetric:
                summary = self.get_performance_summary(metric_name, minutes)
                if summary:
                    summaries[metric_name.value] = summary
            
            return summaries
            
        except Exception as e:
            logger.error(f"Error getting all performance summaries: {e}")
            return {}
    
    def set_performance_threshold(self, threshold: PerformanceThreshold) -> bool:
        """
        Set a performance threshold.
        
        Args:
            threshold: The performance threshold
            
        Returns:
            bool: True if threshold was set successfully, False otherwise
        """
        try:
            self.performance_thresholds[threshold.metric_name] = threshold
            logger.info(f"Set performance threshold for {threshold.metric_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error setting performance threshold: {e}")
            return False
    
    def get_performance_threshold(self, metric_name: PerformanceMetric) -> Optional[PerformanceThreshold]:
        """
        Get a performance threshold.
        
        Args:
            metric_name: The metric name
            
        Returns:
            PerformanceThreshold: The performance threshold, or None if not found
        """
        return self.performance_thresholds.get(metric_name)
    
    def get_active_alerts(self) -> List[PerformanceAlert]:
        """
        Get all active (unresolved) performance alerts.
        
        Returns:
            List[PerformanceAlert]: List of active alerts
        """
        return [alert for alert in self.performance_alerts if not alert.resolved]
    
    def resolve_alert(self, alert_id: str) -> bool:
        """
        Resolve a performance alert.
        
        Args:
            alert_id: The alert ID
            
        Returns:
            bool: True if alert was resolved successfully, False otherwise
        """
        try:
            for alert in self.performance_alerts:
                if alert.id == alert_id:
                    alert.resolved = True
                    logger.info(f"Resolved performance alert {alert_id}")
                    return True
            
            logger.warning(f"Performance alert {alert_id} not found")
            return False
            
        except Exception as e:
            logger.error(f"Error resolving performance alert {alert_id}: {e}")
            return False
    
    def add_alert_callback(self, callback: Callable[[PerformanceAlert], None]) -> bool:
        """
        Add an alert callback function.
        
        Args:
            callback: The callback function to call when alerts are generated
            
        Returns:
            bool: True if callback was added successfully, False otherwise
        """
        try:
            self.alert_callbacks.append(callback)
            logger.info("Added performance alert callback")
            return True
            
        except Exception as e:
            logger.error(f"Error adding performance alert callback: {e}")
            return False
    
    def get_performance_monitor_status(self) -> Dict[str, Any]:
        """
        Get the current status of the performance monitor.
        
        Returns:
            Dict[str, Any]: Performance monitor status information
        """
        try:
            total_metrics = len(self.performance_data)
            total_data_points = sum(len(data) for data in self.performance_data.values())
            active_alerts = len(self.get_active_alerts())
            total_alerts = len(self.performance_alerts)
            
            # Get current performance levels
            current_levels = {}
            for metric_name in PerformanceMetric:
                summary = self.get_performance_summary(metric_name, 5)  # Last 5 minutes
                if summary:
                    current_levels[metric_name.value] = summary.get("performance_level", "unknown")
            
            return {
                "is_monitoring": self.is_monitoring,
                "monitoring_interval": self.monitoring_interval,
                "total_metrics": total_metrics,
                "total_data_points": total_data_points,
                "active_alerts": active_alerts,
                "total_alerts": total_alerts,
                "current_performance_levels": current_levels
            }
            
        except Exception as e:
            logger.error(f"Error getting performance monitor status: {e}")
            return {}
    
    def _initialize_thresholds(self):
        """Initialize default performance thresholds."""
        try:
            # Throughput thresholds (higher is better)
            self.set_performance_threshold(PerformanceThreshold(
                metric_name=PerformanceMetric.THROUGHPUT,
                warning_threshold=1000.0,  # records per minute
                critical_threshold=500.0,
                unit="records/min",
                is_higher_better=True
            ))
            
            # Latency thresholds (lower is better)
            self.set_performance_threshold(PerformanceThreshold(
                metric_name=PerformanceMetric.LATENCY,
                warning_threshold=5.0,  # seconds
                critical_threshold=10.0,
                unit="seconds",
                is_higher_better=False
            ))
            
            # CPU usage thresholds (lower is better)
            self.set_performance_threshold(PerformanceThreshold(
                metric_name=PerformanceMetric.CPU_USAGE,
                warning_threshold=80.0,  # percentage
                critical_threshold=95.0,
                unit="percent",
                is_higher_better=False
            ))
            
            # Memory usage thresholds (lower is better)
            self.set_performance_threshold(PerformanceThreshold(
                metric_name=PerformanceMetric.MEMORY_USAGE,
                warning_threshold=85.0,  # percentage
                critical_threshold=95.0,
                unit="percent",
                is_higher_better=False
            ))
            
            # Error rate thresholds (lower is better)
            self.set_performance_threshold(PerformanceThreshold(
                metric_name=PerformanceMetric.ERROR_RATE,
                warning_threshold=0.05,  # 5%
                critical_threshold=0.10,  # 10%
                unit="percent",
                is_higher_better=False
            ))
            
        except Exception as e:
            logger.error(f"Error initializing performance thresholds: {e}")
    
    def _check_performance_alerts(self, data: PerformanceData):
        """Check for performance alerts based on new data."""
        try:
            threshold = self.get_performance_threshold(data.metric_name)
            if not threshold:
                return
            
            # Check if current value exceeds thresholds
            alert_level = None
            threshold_value = None
            
            if threshold.is_higher_better:
                # For metrics where higher is better, alert if value is too low
                if data.value < threshold.critical_threshold:
                    alert_level = PerformanceLevel.CRITICAL
                    threshold_value = threshold.critical_threshold
                elif data.value < threshold.warning_threshold:
                    alert_level = PerformanceLevel.POOR
                    threshold_value = threshold.warning_threshold
            else:
                # For metrics where lower is better, alert if value is too high
                if data.value > threshold.critical_threshold:
                    alert_level = PerformanceLevel.CRITICAL
                    threshold_value = threshold.critical_threshold
                elif data.value > threshold.warning_threshold:
                    alert_level = PerformanceLevel.POOR
                    threshold_value = threshold.warning_threshold
            
            # Create alert if threshold exceeded
            if alert_level:
                alert_id = f"{data.metric_name.value}_{alert_level.value}_{int(time.time())}"
                
                alert = PerformanceAlert(
                    id=alert_id,
                    metric_name=data.metric_name,
                    current_value=data.value,
                    threshold_value=threshold_value,
                    alert_level=alert_level,
                    message=f"Performance alert: {data.metric_name.value} is {data.value} {data.unit}, threshold: {threshold_value} {data.unit}"
                )
                
                self.performance_alerts.append(alert)
                
                # Call alert callbacks
                for callback in self.alert_callbacks:
                    try:
                        callback(alert)
                    except Exception as e:
                        logger.error(f"Error in performance alert callback: {e}")
                
                logger.warning(f"Performance alert: {alert.message}")
                
        except Exception as e:
            logger.error(f"Error checking performance alerts: {e}")
    
    def _calculate_performance_level(self, metric_name: PerformanceMetric, value: float) -> PerformanceLevel:
        """
        Calculate performance level based on value and thresholds.
        
        Args:
            metric_name: The metric name
            value: The current value
            
        Returns:
            PerformanceLevel: The performance level
        """
        try:
            threshold = self.get_performance_threshold(metric_name)
            if not threshold:
                return PerformanceLevel.FAIR
            
            if threshold.is_higher_better:
                # For metrics where higher is better
                if value >= threshold.warning_threshold:
                    return PerformanceLevel.EXCELLENT
                elif value >= threshold.critical_threshold:
                    return PerformanceLevel.GOOD
                else:
                    return PerformanceLevel.POOR
            else:
                # For metrics where lower is better
                if value <= threshold.warning_threshold:
                    return PerformanceLevel.EXCELLENT
                elif value <= threshold.critical_threshold:
                    return PerformanceLevel.GOOD
                else:
                    return PerformanceLevel.POOR
                    
        except Exception as e:
            logger.error(f"Error calculating performance level: {e}")
            return PerformanceLevel.FAIR
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                # Collect system performance metrics
                self._collect_system_metrics()
                
                # Collect pipeline performance metrics
                self._collect_pipeline_metrics()
                
                # Sleep for monitoring interval
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in performance monitoring loop: {e}")
                time.sleep(self.monitoring_interval)
    
    def _collect_system_metrics(self):
        """Collect system performance metrics."""
        try:
            # This is a placeholder implementation
            # In a real system, you would collect actual system metrics
            
            # Simulate CPU usage
            cpu_data = PerformanceData(
                metric_name=PerformanceMetric.CPU_USAGE,
                value=50.0 + (time.time() % 30),  # Simulate CPU usage
                unit="percent"
            )
            self.add_performance_data(cpu_data)
            
            # Simulate memory usage
            memory_data = PerformanceData(
                metric_name=PerformanceMetric.MEMORY_USAGE,
                value=60.0 + (time.time() % 20),  # Simulate memory usage
                unit="percent"
            )
            self.add_performance_data(memory_data)
            
            # Simulate disk I/O
            disk_io_data = PerformanceData(
                metric_name=PerformanceMetric.DISK_IO,
                value=100.0 + (time.time() % 50),  # Simulate disk I/O
                unit="MB/s"
            )
            self.add_performance_data(disk_io_data)
            
            # Simulate network I/O
            network_io_data = PerformanceData(
                metric_name=PerformanceMetric.NETWORK_IO,
                value=50.0 + (time.time() % 25),  # Simulate network I/O
                unit="MB/s"
            )
            self.add_performance_data(network_io_data)
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
    
    def _collect_pipeline_metrics(self):
        """Collect pipeline performance metrics."""
        try:
            # This is a placeholder implementation
            # In a real system, you would collect actual pipeline metrics
            
            # Simulate throughput
            throughput_data = PerformanceData(
                metric_name=PerformanceMetric.THROUGHPUT,
                value=1200.0 + (time.time() % 200),  # Simulate throughput
                unit="records/min"
            )
            self.add_performance_data(throughput_data)
            
            # Simulate latency
            latency_data = PerformanceData(
                metric_name=PerformanceMetric.LATENCY,
                value=3.0 + (time.time() % 2),  # Simulate latency
                unit="seconds"
            )
            self.add_performance_data(latency_data)
            
            # Simulate error rate
            error_rate_data = PerformanceData(
                metric_name=PerformanceMetric.ERROR_RATE,
                value=0.02 + (time.time() % 0.03),  # Simulate error rate
                unit="percent"
            )
            self.add_performance_data(error_rate_data)
            
            # Simulate processing time
            processing_time_data = PerformanceData(
                metric_name=PerformanceMetric.PROCESSING_TIME,
                value=2.5 + (time.time() % 1.5),  # Simulate processing time
                unit="seconds"
            )
            self.add_performance_data(processing_time_data)
            
        except Exception as e:
            logger.error(f"Error collecting pipeline metrics: {e}")
