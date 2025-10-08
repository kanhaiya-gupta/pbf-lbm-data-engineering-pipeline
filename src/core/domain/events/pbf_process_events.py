"""
PBF process domain events.
"""

from typing import Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass, field

from .base_event import BaseEvent, EventType, EventSeverity
from ..enums import ProcessStatus, QualityTier


@dataclass(frozen=True)
class PBFProcessEvent(BaseEvent):
    """Base class for PBF process events."""
    
    # Process identification
    process_id: str = field(default="")
    build_id: Optional[str] = None
    part_id: Optional[str] = None
    
    # Process information
    process_status: Optional[ProcessStatus] = None
    quality_tier: Optional[QualityTier] = None
    
    def validate(self) -> None:
        """Validate PBF process event."""
        if not self.process_id:
            raise ValueError("Process ID cannot be empty")


@dataclass(frozen=True)
class ProcessStartedEvent(PBFProcessEvent):
    """Event raised when a PBF process starts."""
    
    def __post_init__(self):
        """Set default values for process started event."""
        object.__setattr__(self, 'event_name', 'ProcessStarted')
        object.__setattr__(self, 'event_type', EventType.DOMAIN)
        object.__setattr__(self, 'severity', EventSeverity.INFO)
        super().__post_init__()
    
    def validate(self) -> None:
        """Validate process started event."""
        super().validate()
        if not self.data.get('process_parameters'):
            raise ValueError("Process parameters are required for process started event")


@dataclass(frozen=True)
class ProcessCompletedEvent(PBFProcessEvent):
    """Event raised when a PBF process completes successfully."""
    
    def __post_init__(self):
        """Set default values for process completed event."""
        object.__setattr__(self, 'event_name', 'ProcessCompleted')
        object.__setattr__(self, 'event_type', EventType.DOMAIN)
        object.__setattr__(self, 'severity', EventSeverity.INFO)
        super().__post_init__()
    
    def validate(self) -> None:
        """Validate process completed event."""
        super().validate()
        if not self.data.get('completion_time'):
            raise ValueError("Completion time is required for process completed event")
        if not self.data.get('build_duration'):
            raise ValueError("Build duration is required for process completed event")


@dataclass(frozen=True)
class ProcessFailedEvent(PBFProcessEvent):
    """Event raised when a PBF process fails."""
    
    def __post_init__(self):
        """Set default values for process failed event."""
        object.__setattr__(self, 'event_name', 'ProcessFailed')
        object.__setattr__(self, 'event_type', EventType.DOMAIN)
        object.__setattr__(self, 'severity', EventSeverity.ERROR)
        super().__post_init__()
    
    def validate(self) -> None:
        """Validate process failed event."""
        super().validate()
        if not self.data.get('failure_reason'):
            raise ValueError("Failure reason is required for process failed event")
        if not self.data.get('error_code'):
            raise ValueError("Error code is required for process failed event")


@dataclass(frozen=True)
class ProcessPausedEvent(PBFProcessEvent):
    """Event raised when a PBF process is paused."""
    
    def __post_init__(self):
        """Set default values for process paused event."""
        object.__setattr__(self, 'event_name', 'ProcessPaused')
        object.__setattr__(self, 'event_type', EventType.DOMAIN)
        object.__setattr__(self, 'severity', EventSeverity.WARNING)
        super().__post_init__()
    
    def validate(self) -> None:
        """Validate process paused event."""
        super().validate()
        if not self.data.get('pause_reason'):
            raise ValueError("Pause reason is required for process paused event")


@dataclass(frozen=True)
class ProcessResumedEvent(PBFProcessEvent):
    """Event raised when a PBF process is resumed."""
    
    def __post_init__(self):
        """Set default values for process resumed event."""
        object.__setattr__(self, 'event_name', 'ProcessResumed')
        object.__setattr__(self, 'event_type', EventType.DOMAIN)
        object.__setattr__(self, 'severity', EventSeverity.INFO)
        super().__post_init__()
    
    def validate(self) -> None:
        """Validate process resumed event."""
        super().validate()
        if not self.data.get('resume_reason'):
            raise ValueError("Resume reason is required for process resumed event")


@dataclass(frozen=True)
class ProcessCancelledEvent(PBFProcessEvent):
    """Event raised when a PBF process is cancelled."""
    
    def __post_init__(self):
        """Set default values for process cancelled event."""
        object.__setattr__(self, 'event_name', 'ProcessCancelled')
        object.__setattr__(self, 'event_type', EventType.DOMAIN)
        object.__setattr__(self, 'severity', EventSeverity.WARNING)
        super().__post_init__()
    
    def validate(self) -> None:
        """Validate process cancelled event."""
        super().validate()
        if not self.data.get('cancellation_reason'):
            raise ValueError("Cancellation reason is required for process cancelled event")


@dataclass(frozen=True)
class ProcessQualityCheckedEvent(PBFProcessEvent):
    """Event raised when a PBF process quality check is completed."""
    
    def __post_init__(self):
        """Set default values for process quality checked event."""
        object.__setattr__(self, 'event_name', 'ProcessQualityChecked')
        object.__setattr__(self, 'event_type', EventType.DOMAIN)
        object.__setattr__(self, 'severity', EventSeverity.INFO)
        super().__post_init__()
    
    def validate(self) -> None:
        """Validate process quality checked event."""
        super().validate()
        if not self.data.get('quality_metrics'):
            raise ValueError("Quality metrics are required for process quality checked event")
        if not self.data.get('quality_score'):
            raise ValueError("Quality score is required for process quality checked event")


# Factory functions for creating events
def create_process_started_event(
    process_id: str,
    process_parameters: Dict[str, Any],
    build_id: Optional[str] = None,
    part_id: Optional[str] = None,
    source: str = "PBFProcessService",
    correlation_id: Optional[str] = None
) -> ProcessStartedEvent:
    """Create a process started event."""
    return ProcessStartedEvent(
        event_id=BaseEvent.generate_event_id(),
        event_name="ProcessStarted",
        event_type=EventType.DOMAIN,
        severity=EventSeverity.INFO,
        occurred_at=datetime.utcnow(),
        source=source,
        process_id=process_id,
        build_id=build_id,
        part_id=part_id,
        data={
            "process_parameters": process_parameters,
            "start_time": datetime.utcnow().isoformat()
        },
        correlation_id=correlation_id or BaseEvent.generate_correlation_id()
    )


def create_process_completed_event(
    process_id: str,
    completion_time: datetime,
    build_duration: float,
    quality_metrics: Dict[str, Any],
    build_id: Optional[str] = None,
    part_id: Optional[str] = None,
    source: str = "PBFProcessService",
    correlation_id: Optional[str] = None
) -> ProcessCompletedEvent:
    """Create a process completed event."""
    return ProcessCompletedEvent(
        event_id=BaseEvent.generate_event_id(),
        event_name="ProcessCompleted",
        event_type=EventType.DOMAIN,
        severity=EventSeverity.INFO,
        occurred_at=datetime.utcnow(),
        source=source,
        process_id=process_id,
        build_id=build_id,
        part_id=part_id,
        data={
            "completion_time": completion_time.isoformat(),
            "build_duration": build_duration,
            "quality_metrics": quality_metrics,
            "success": True
        },
        correlation_id=correlation_id or BaseEvent.generate_correlation_id()
    )


def create_process_failed_event(
    process_id: str,
    failure_reason: str,
    error_code: str,
    error_details: Dict[str, Any],
    build_id: Optional[str] = None,
    part_id: Optional[str] = None,
    source: str = "PBFProcessService",
    correlation_id: Optional[str] = None
) -> ProcessFailedEvent:
    """Create a process failed event."""
    return ProcessFailedEvent(
        event_id=BaseEvent.generate_event_id(),
        event_name="ProcessFailed",
        event_type=EventType.DOMAIN,
        severity=EventSeverity.ERROR,
        occurred_at=datetime.utcnow(),
        source=source,
        process_id=process_id,
        build_id=build_id,
        part_id=part_id,
        data={
            "failure_reason": failure_reason,
            "error_code": error_code,
            "error_details": error_details,
            "failure_time": datetime.utcnow().isoformat()
        },
        correlation_id=correlation_id or BaseEvent.generate_correlation_id()
    )


def create_process_quality_checked_event(
    process_id: str,
    quality_metrics: Dict[str, Any],
    quality_score: float,
    quality_tier: QualityTier,
    build_id: Optional[str] = None,
    part_id: Optional[str] = None,
    source: str = "QualityService",
    correlation_id: Optional[str] = None
) -> ProcessQualityCheckedEvent:
    """Create a process quality checked event."""
    return ProcessQualityCheckedEvent(
        event_id=BaseEvent.generate_event_id(),
        event_name="ProcessQualityChecked",
        event_type=EventType.DOMAIN,
        severity=EventSeverity.INFO,
        occurred_at=datetime.utcnow(),
        source=source,
        process_id=process_id,
        build_id=build_id,
        part_id=part_id,
        quality_tier=quality_tier,
        data={
            "quality_metrics": quality_metrics,
            "quality_score": quality_score,
            "quality_tier": quality_tier.value,
            "check_time": datetime.utcnow().isoformat()
        },
        correlation_id=correlation_id or BaseEvent.generate_correlation_id()
    )