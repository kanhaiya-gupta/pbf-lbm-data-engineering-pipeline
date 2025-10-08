"""
ISPM monitoring domain entity.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass, field

from .base_entity import BaseEntity
from ..enums import MonitoringType


@dataclass
class ISPMMonitoring(BaseEntity):
    """
    Domain entity representing ISPM (In-Situ Process Monitoring) session.
    
    This entity encapsulates all information about ISPM monitoring sessions,
    including sensor data, monitoring parameters, and analysis results.
    """
    
    # Monitoring identification
    monitoring_name: str = field(default="")
    monitoring_session_id: Optional[str] = None
    
    # Process information
    process_id: Optional[str] = None
    build_id: Optional[str] = None
    
    # Monitoring configuration
    monitoring_type: MonitoringType = MonitoringType.ISPM_THERMAL
    sensor_configuration: Optional[Dict[str, Any]] = None
    
    # Monitoring state
    is_active: bool = False
    is_paused: bool = False
    
    # Timing information
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    last_data_received: Optional[datetime] = None
    
    # Monitoring data
    data_points_count: int = 0
    anomaly_count: int = 0
    threshold_violations: int = 0
    
    # Sensor information
    sensor_ids: Optional[List[str]] = None
    sensor_status: Optional[Dict[str, str]] = None
    
    # Analysis results
    anomaly_detections: Optional[List[Dict[str, Any]]] = None
    quality_assessments: Optional[List[Dict[str, Any]]] = None
    
    # Monitoring parameters
    sampling_rate: Optional[float] = None  # Hz
    data_retention_hours: Optional[int] = None
    
    def __post_init__(self):
        """Initialize default values."""
        if self.sensor_ids is None:
            object.__setattr__(self, 'sensor_ids', [])
        if self.sensor_status is None:
            object.__setattr__(self, 'sensor_status', {})
        if self.anomaly_detections is None:
            object.__setattr__(self, 'anomaly_detections', [])
        if self.quality_assessments is None:
            object.__setattr__(self, 'quality_assessments', [])
        super().__post_init__()
    
    def validate(self) -> None:
        """Validate ISPM monitoring entity."""
        if not self.monitoring_name:
            raise ValueError("Monitoring name cannot be empty")
        
        if self.start_time and self.end_time and self.start_time > self.end_time:
            raise ValueError("Start time cannot be after end time")
        
        if self.sampling_rate is not None and self.sampling_rate <= 0:
            raise ValueError("Sampling rate must be positive")
        
        if self.data_retention_hours is not None and self.data_retention_hours < 0:
            raise ValueError("Data retention hours cannot be negative")
        
        if self.data_points_count < 0:
            raise ValueError("Data points count cannot be negative")
        
        if self.anomaly_count < 0:
            raise ValueError("Anomaly count cannot be negative")
        
        if self.threshold_violations < 0:
            raise ValueError("Threshold violations cannot be negative")
    
    def start_monitoring(self, start_time: Optional[datetime] = None) -> 'ISPMMonitoring':
        """Start ISPM monitoring."""
        if start_time is None:
            start_time = datetime.utcnow()
        
        return self.update(
            is_active=True,
            is_paused=False,
            start_time=start_time,
            updated_by="ISPMMonitoringService"
        )
    
    def stop_monitoring(self, end_time: Optional[datetime] = None) -> 'ISPMMonitoring':
        """Stop ISPM monitoring."""
        if end_time is None:
            end_time = datetime.utcnow()
        
        return self.update(
            is_active=False,
            is_paused=False,
            end_time=end_time,
            updated_by="ISPMMonitoringService"
        )
    
    def pause_monitoring(self) -> 'ISPMMonitoring':
        """Pause ISPM monitoring."""
        return self.update(
            is_paused=True,
            updated_by="ISPMMonitoringService"
        )
    
    def resume_monitoring(self) -> 'ISPMMonitoring':
        """Resume ISPM monitoring."""
        return self.update(
            is_paused=False,
            updated_by="ISPMMonitoringService"
        )
    
    def add_sensor(self, sensor_id: str, sensor_type: str) -> 'ISPMMonitoring':
        """Add a sensor to the monitoring session."""
        new_sensor_ids = self.sensor_ids + [sensor_id]
        new_sensor_status = {**self.sensor_status, sensor_id: "active"}
        
        return self.update(
            sensor_ids=new_sensor_ids,
            sensor_status=new_sensor_status,
            updated_by="ISPMMonitoringService"
        )
    
    def remove_sensor(self, sensor_id: str) -> 'ISPMMonitoring':
        """Remove a sensor from the monitoring session."""
        new_sensor_ids = [sid for sid in self.sensor_ids if sid != sensor_id]
        new_sensor_status = {k: v for k, v in self.sensor_status.items() if k != sensor_id}
        
        return self.update(
            sensor_ids=new_sensor_ids,
            sensor_status=new_sensor_status,
            updated_by="ISPMMonitoringService"
        )
    
    def update_sensor_status(self, sensor_id: str, status: str) -> 'ISPMMonitoring':
        """Update sensor status."""
        new_sensor_status = {**self.sensor_status, sensor_id: status}
        
        return self.update(
            sensor_status=new_sensor_status,
            updated_by="ISPMMonitoringService"
        )
    
    def add_data_point(self, data_point: Dict[str, Any]) -> 'ISPMMonitoring':
        """Add a data point to the monitoring session."""
        return self.update(
            data_points_count=self.data_points_count + 1,
            last_data_received=datetime.utcnow(),
            updated_by="ISPMMonitoringService"
        )
    
    def add_anomaly_detection(self, anomaly: Dict[str, Any]) -> 'ISPMMonitoring':
        """Add an anomaly detection to the monitoring session."""
        new_anomaly_detections = self.anomaly_detections + [anomaly]
        
        return self.update(
            anomaly_count=self.anomaly_count + 1,
            anomaly_detections=new_anomaly_detections,
            updated_by="ISPMMonitoringService"
        )
    
    def add_threshold_violation(self) -> 'ISPMMonitoring':
        """Add a threshold violation to the monitoring session."""
        return self.update(
            threshold_violations=self.threshold_violations + 1,
            updated_by="ISPMMonitoringService"
        )
    
    def add_quality_assessment(self, assessment: Dict[str, Any]) -> 'ISPMMonitoring':
        """Add a quality assessment to the monitoring session."""
        new_quality_assessments = self.quality_assessments + [assessment]
        
        return self.update(
            quality_assessments=new_quality_assessments,
            updated_by="ISPMMonitoringService"
        )
    
    def get_duration(self) -> Optional[float]:
        """Get monitoring duration in seconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        elif self.start_time:
            return (datetime.utcnow() - self.start_time).total_seconds()
        return None
    
    def get_data_rate(self) -> Optional[float]:
        """Get data collection rate (points per second)."""
        duration = self.get_duration()
        if duration and duration > 0:
            return self.data_points_count / duration
        return None
    
    def get_anomaly_rate(self) -> Optional[float]:
        """Get anomaly detection rate (anomalies per hour)."""
        duration = self.get_duration()
        if duration and duration > 0:
            return (self.anomaly_count / duration) * 3600  # Convert to per hour
        return None
    
    def get_threshold_violation_rate(self) -> Optional[float]:
        """Get threshold violation rate (violations per hour)."""
        duration = self.get_duration()
        if duration and duration > 0:
            return (self.threshold_violations / duration) * 3600  # Convert to per hour
        return None
    
    def get_active_sensor_count(self) -> int:
        """Get count of active sensors."""
        return sum(1 for status in self.sensor_status.values() if status == "active")
    
    def get_failed_sensor_count(self) -> int:
        """Get count of failed sensors."""
        return sum(1 for status in self.sensor_status.values() if status == "failed")
    
    def is_data_stale(self, max_age_minutes: float = 5) -> bool:
        """Check if monitoring data is stale."""
        if self.last_data_received:
            age_minutes = (datetime.utcnow() - self.last_data_received).total_seconds() / 60
            return age_minutes > max_age_minutes
        return True
    
    def get_monitoring_summary(self) -> Dict[str, Any]:
        """Get comprehensive monitoring summary."""
        return {
            "monitoring_id": self.id,
            "monitoring_name": self.monitoring_name,
            "monitoring_session_id": self.monitoring_session_id,
            "monitoring_type": self.monitoring_type.value,
            "process_id": self.process_id,
            "build_id": self.build_id,
            "is_active": self.is_active,
            "is_paused": self.is_paused,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration": self.get_duration(),
            "last_data_received": self.last_data_received.isoformat() if self.last_data_received else None,
            "data_points_count": self.data_points_count,
            "anomaly_count": self.anomaly_count,
            "threshold_violations": self.threshold_violations,
            "sensor_count": len(self.sensor_ids),
            "active_sensor_count": self.get_active_sensor_count(),
            "failed_sensor_count": self.get_failed_sensor_count(),
            "data_rate": self.get_data_rate(),
            "anomaly_rate": self.get_anomaly_rate(),
            "threshold_violation_rate": self.get_threshold_violation_rate(),
            "is_data_stale": self.is_data_stale(),
            "sampling_rate": self.sampling_rate,
            "data_retention_hours": self.data_retention_hours
        }
    
    def get_monitoring_analytics(self) -> Dict[str, Any]:
        """Get monitoring analytics and insights."""
        analytics = {
            "performance_metrics": {
                "data_rate": self.get_data_rate(),
                "anomaly_rate": self.get_anomaly_rate(),
                "threshold_violation_rate": self.get_threshold_violation_rate(),
                "sensor_health": {
                    "total_sensors": len(self.sensor_ids),
                    "active_sensors": self.get_active_sensor_count(),
                    "failed_sensors": self.get_failed_sensor_count(),
                    "health_percentage": (self.get_active_sensor_count() / len(self.sensor_ids) * 100) if self.sensor_ids else 0
                }
            },
            "data_quality": {
                "is_data_stale": self.is_data_stale(),
                "data_points_count": self.data_points_count,
                "last_data_received": self.last_data_received.isoformat() if self.last_data_received else None
            },
            "anomaly_analysis": {
                "total_anomalies": self.anomaly_count,
                "anomaly_rate": self.get_anomaly_rate(),
                "recent_anomalies": self.anomaly_detections[-5:] if self.anomaly_detections else []
            },
            "threshold_analysis": {
                "total_violations": self.threshold_violations,
                "violation_rate": self.get_threshold_violation_rate(),
                "violation_severity": "high" if self.get_threshold_violation_rate() and self.get_threshold_violation_rate() > 10 else "normal"
            },
            "recommendations": []
        }
        
        # Add recommendations based on monitoring state
        if self.is_data_stale():
            analytics["recommendations"].append("Check sensor connectivity and data collection")
        
        if self.get_failed_sensor_count() > 0:
            analytics["recommendations"].append("Investigate and repair failed sensors")
        
        if self.get_anomaly_rate() and self.get_anomaly_rate() > 5:  # High anomaly rate
            analytics["recommendations"].append("Investigate high anomaly rate and review process parameters")
        
        if self.get_threshold_violation_rate() and self.get_threshold_violation_rate() > 10:  # High violation rate
            analytics["recommendations"].append("Review and adjust monitoring thresholds")
        
        if not self.is_active and not self.end_time:
            analytics["recommendations"].append("Consider restarting monitoring session")
        
        return analytics