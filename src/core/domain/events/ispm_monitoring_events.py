"""
ISPM monitoring domain events.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass, field

from .base_event import BaseEvent, EventType, EventSeverity
from ..enums import MonitoringType


@dataclass(frozen=True)
class ISPMMonitoringEvent(BaseEvent):
    """Base class for ISPM monitoring events."""
    
    # Monitoring identification
    monitoring_session_id: str = field(default="")
    sensor_id: Optional[str] = None
    monitoring_type: Optional[MonitoringType] = None
    
    # Process information
    process_id: Optional[str] = None
    build_id: Optional[str] = None
    
    def validate(self) -> None:
        """Validate ISPM monitoring event."""
        if not self.monitoring_session_id:
            raise ValueError("Monitoring session ID cannot be empty")


@dataclass(frozen=True)
class MonitoringStartedEvent(ISPMMonitoringEvent):
    """Event raised when ISPM monitoring starts."""
    
    def __post_init__(self):
        """Set default values for monitoring started event."""
        object.__setattr__(self, 'event_name', 'MonitoringStarted')
        object.__setattr__(self, 'event_type', EventType.DOMAIN)
        object.__setattr__(self, 'severity', EventSeverity.INFO)
        super().__post_init__()
    
    def validate(self) -> None:
        """Validate monitoring started event."""
        super().validate()
        if not self.data.get('monitoring_config'):
            raise ValueError("Monitoring configuration is required for monitoring started event")


@dataclass(frozen=True)
class MonitoringStoppedEvent(ISPMMonitoringEvent):
    """Event raised when ISPM monitoring stops."""
    
    def __post_init__(self):
        """Set default values for monitoring stopped event."""
        object.__setattr__(self, 'event_name', 'MonitoringStopped')
        object.__setattr__(self, 'event_type', EventType.DOMAIN)
        object.__setattr__(self, 'severity', EventSeverity.INFO)
        super().__post_init__()
    
    def validate(self) -> None:
        """Validate monitoring stopped event."""
        super().validate()
        if not self.data.get('stop_reason'):
            raise ValueError("Stop reason is required for monitoring stopped event")


@dataclass(frozen=True)
class AnomalyDetectedEvent(ISPMMonitoringEvent):
    """Event raised when an anomaly is detected during monitoring."""
    
    def __post_init__(self):
        """Set default values for anomaly detected event."""
        object.__setattr__(self, 'event_name', 'AnomalyDetected')
        object.__setattr__(self, 'event_type', EventType.DOMAIN)
        object.__setattr__(self, 'severity', EventSeverity.WARNING)
        super().__post_init__()
    
    def validate(self) -> None:
        """Validate anomaly detected event."""
        super().validate()
        if not self.data.get('anomaly_type'):
            raise ValueError("Anomaly type is required for anomaly detected event")
        if not self.data.get('anomaly_severity'):
            raise ValueError("Anomaly severity is required for anomaly detected event")


@dataclass(frozen=True)
class ThresholdExceededEvent(ISPMMonitoringEvent):
    """Event raised when a monitoring threshold is exceeded."""
    
    def __post_init__(self):
        """Set default values for threshold exceeded event."""
        object.__setattr__(self, 'event_name', 'ThresholdExceeded')
        object.__setattr__(self, 'event_type', EventType.DOMAIN)
        object.__setattr__(self, 'severity', EventSeverity.WARNING)
        super().__post_init__()
    
    def validate(self) -> None:
        """Validate threshold exceeded event."""
        super().validate()
        if not self.data.get('threshold_name'):
            raise ValueError("Threshold name is required for threshold exceeded event")
        if not self.data.get('threshold_value'):
            raise ValueError("Threshold value is required for threshold exceeded event")
        if not self.data.get('actual_value'):
            raise ValueError("Actual value is required for threshold exceeded event")


@dataclass(frozen=True)
class SensorFailureEvent(ISPMMonitoringEvent):
    """Event raised when a sensor fails."""
    
    def __post_init__(self):
        """Set default values for sensor failure event."""
        object.__setattr__(self, 'event_name', 'SensorFailure')
        object.__setattr__(self, 'event_type', EventType.DOMAIN)
        object.__setattr__(self, 'severity', EventSeverity.ERROR)
        super().__post_init__()
    
    def validate(self) -> None:
        """Validate sensor failure event."""
        super().validate()
        if not self.sensor_id:
            raise ValueError("Sensor ID is required for sensor failure event")
        if not self.data.get('failure_type'):
            raise ValueError("Failure type is required for sensor failure event")


@dataclass(frozen=True)
class DataQualityAlertEvent(ISPMMonitoringEvent):
    """Event raised when data quality issues are detected."""
    
    def __post_init__(self):
        """Set default values for data quality alert event."""
        object.__setattr__(self, 'event_name', 'DataQualityAlert')
        object.__setattr__(self, 'event_type', EventType.DOMAIN)
        object.__setattr__(self, 'severity', EventSeverity.WARNING)
        super().__post_init__()
    
    def validate(self) -> None:
        """Validate data quality alert event."""
        super().validate()
        if not self.data.get('quality_issue'):
            raise ValueError("Quality issue is required for data quality alert event")
        if not self.data.get('quality_score'):
            raise ValueError("Quality score is required for data quality alert event")


# Factory functions for creating events
def create_monitoring_started_event(
    monitoring_session_id: str,
    monitoring_config: Dict[str, Any],
    sensor_id: Optional[str] = None,
    monitoring_type: Optional[MonitoringType] = None,
    process_id: Optional[str] = None,
    build_id: Optional[str] = None,
    source: str = "ISPMMonitoringService",
    correlation_id: Optional[str] = None
) -> MonitoringStartedEvent:
    """Create a monitoring started event."""
    return MonitoringStartedEvent(
        event_id=BaseEvent.generate_event_id(),
        event_name="MonitoringStarted",
        event_type=EventType.DOMAIN,
        severity=EventSeverity.INFO,
        occurred_at=datetime.utcnow(),
        source=source,
        monitoring_session_id=monitoring_session_id,
        sensor_id=sensor_id,
        monitoring_type=monitoring_type,
        process_id=process_id,
        build_id=build_id,
        data={
            "monitoring_config": monitoring_config,
            "start_time": datetime.utcnow().isoformat()
        },
        correlation_id=correlation_id or BaseEvent.generate_correlation_id()
    )


def create_anomaly_detected_event(
    monitoring_session_id: str,
    anomaly_type: str,
    anomaly_severity: str,
    anomaly_data: Dict[str, Any],
    sensor_id: Optional[str] = None,
    monitoring_type: Optional[MonitoringType] = None,
    process_id: Optional[str] = None,
    build_id: Optional[str] = None,
    source: str = "ISPMMonitoringService",
    correlation_id: Optional[str] = None
) -> AnomalyDetectedEvent:
    """Create an anomaly detected event."""
    return AnomalyDetectedEvent(
        event_id=BaseEvent.generate_event_id(),
        event_name="AnomalyDetected",
        event_type=EventType.DOMAIN,
        severity=EventSeverity.WARNING,
        occurred_at=datetime.utcnow(),
        source=source,
        monitoring_session_id=monitoring_session_id,
        sensor_id=sensor_id,
        monitoring_type=monitoring_type,
        process_id=process_id,
        build_id=build_id,
        data={
            "anomaly_type": anomaly_type,
            "anomaly_severity": anomaly_severity,
            "anomaly_data": anomaly_data,
            "detection_time": datetime.utcnow().isoformat()
        },
        correlation_id=correlation_id or BaseEvent.generate_correlation_id()
    )


def create_threshold_exceeded_event(
    monitoring_session_id: str,
    threshold_name: str,
    threshold_value: float,
    actual_value: float,
    sensor_id: Optional[str] = None,
    monitoring_type: Optional[MonitoringType] = None,
    process_id: Optional[str] = None,
    build_id: Optional[str] = None,
    source: str = "ISPMMonitoringService",
    correlation_id: Optional[str] = None
) -> ThresholdExceededEvent:
    """Create a threshold exceeded event."""
    return ThresholdExceededEvent(
        event_id=BaseEvent.generate_event_id(),
        event_name="ThresholdExceeded",
        event_type=EventType.DOMAIN,
        severity=EventSeverity.WARNING,
        occurred_at=datetime.utcnow(),
        source=source,
        monitoring_session_id=monitoring_session_id,
        sensor_id=sensor_id,
        monitoring_type=monitoring_type,
        process_id=process_id,
        build_id=build_id,
        data={
            "threshold_name": threshold_name,
            "threshold_value": threshold_value,
            "actual_value": actual_value,
            "exceedance_amount": actual_value - threshold_value,
            "exceedance_time": datetime.utcnow().isoformat()
        },
        correlation_id=correlation_id or BaseEvent.generate_correlation_id()
    )


def create_sensor_failure_event(
    monitoring_session_id: str,
    sensor_id: str,
    failure_type: str,
    failure_details: Dict[str, Any],
    monitoring_type: Optional[MonitoringType] = None,
    process_id: Optional[str] = None,
    build_id: Optional[str] = None,
    source: str = "ISPMMonitoringService",
    correlation_id: Optional[str] = None
) -> SensorFailureEvent:
    """Create a sensor failure event."""
    return SensorFailureEvent(
        event_id=BaseEvent.generate_event_id(),
        event_name="SensorFailure",
        event_type=EventType.DOMAIN,
        severity=EventSeverity.ERROR,
        occurred_at=datetime.utcnow(),
        source=source,
        monitoring_session_id=monitoring_session_id,
        sensor_id=sensor_id,
        monitoring_type=monitoring_type,
        process_id=process_id,
        build_id=build_id,
        data={
            "failure_type": failure_type,
            "failure_details": failure_details,
            "failure_time": datetime.utcnow().isoformat()
        },
        correlation_id=correlation_id or BaseEvent.generate_correlation_id()
    )