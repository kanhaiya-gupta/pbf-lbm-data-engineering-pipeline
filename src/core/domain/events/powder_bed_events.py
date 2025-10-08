"""
Powder bed domain events.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass, field

from .base_event import BaseEvent, EventType, EventSeverity


@dataclass(frozen=True)
class PowderBedEvent(BaseEvent):
    """Base class for powder bed events."""
    
    # Bed identification
    bed_id: str = field(default="")
    bed_session_id: Optional[str] = None
    
    # Process information
    process_id: Optional[str] = None
    build_id: Optional[str] = None
    
    def validate(self) -> None:
        """Validate powder bed event."""
        if not self.bed_id:
            raise ValueError("Bed ID cannot be empty")


@dataclass(frozen=True)
class BedPreparedEvent(PowderBedEvent):
    """Event raised when a powder bed is prepared."""
    
    def __post_init__(self):
        """Set default values for bed prepared event."""
        object.__setattr__(self, 'event_name', 'BedPrepared')
        object.__setattr__(self, 'event_type', EventType.DOMAIN)
        object.__setattr__(self, 'severity', EventSeverity.INFO)
        super().__post_init__()
    
    def validate(self) -> None:
        """Validate bed prepared event."""
        super().validate()
        if not self.data.get('bed_parameters'):
            raise ValueError("Bed parameters are required for bed prepared event")
        if not self.data.get('powder_properties'):
            raise ValueError("Powder properties are required for bed prepared event")


@dataclass(frozen=True)
class BedDisturbedEvent(PowderBedEvent):
    """Event raised when a powder bed is disturbed."""
    
    def __post_init__(self):
        """Set default values for bed disturbed event."""
        object.__setattr__(self, 'event_name', 'BedDisturbed')
        object.__setattr__(self, 'event_type', EventType.DOMAIN)
        object.__setattr__(self, 'severity', EventSeverity.WARNING)
        super().__post_init__()
    
    def validate(self) -> None:
        """Validate bed disturbed event."""
        super().validate()
        if not self.data.get('disturbance_type'):
            raise ValueError("Disturbance type is required for bed disturbed event")
        if not self.data.get('disturbance_severity'):
            raise ValueError("Disturbance severity is required for bed disturbed event")


@dataclass(frozen=True)
class PowderDepletedEvent(PowderBedEvent):
    """Event raised when powder is depleted."""
    
    def __post_init__(self):
        """Set default values for powder depleted event."""
        object.__setattr__(self, 'event_name', 'PowderDepleted')
        object.__setattr__(self, 'event_type', EventType.DOMAIN)
        object.__setattr__(self, 'severity', EventSeverity.WARNING)
        super().__post_init__()
    
    def validate(self) -> None:
        """Validate powder depleted event."""
        super().validate()
        if not self.data.get('remaining_powder_percentage'):
            raise ValueError("Remaining powder percentage is required for powder depleted event")
        if not self.data.get('powder_type'):
            raise ValueError("Powder type is required for powder depleted event")


@dataclass(frozen=True)
class BedQualityCheckedEvent(PowderBedEvent):
    """Event raised when powder bed quality is checked."""
    
    def __post_init__(self):
        """Set default values for bed quality checked event."""
        object.__setattr__(self, 'event_name', 'BedQualityChecked')
        object.__setattr__(self, 'event_type', EventType.DOMAIN)
        object.__setattr__(self, 'severity', EventSeverity.INFO)
        super().__post_init__()
    
    def validate(self) -> None:
        """Validate bed quality checked event."""
        super().validate()
        if not self.data.get('quality_metrics'):
            raise ValueError("Quality metrics are required for bed quality checked event")
        if not self.data.get('quality_score'):
            raise ValueError("Quality score is required for bed quality checked event")


@dataclass(frozen=True)
class BedCleanedEvent(PowderBedEvent):
    """Event raised when a powder bed is cleaned."""
    
    def __post_init__(self):
        """Set default values for bed cleaned event."""
        object.__setattr__(self, 'event_name', 'BedCleaned')
        object.__setattr__(self, 'event_type', EventType.DOMAIN)
        object.__setattr__(self, 'severity', EventSeverity.INFO)
        super().__post_init__()
    
    def validate(self) -> None:
        """Validate bed cleaned event."""
        super().validate()
        if not self.data.get('cleaning_method'):
            raise ValueError("Cleaning method is required for bed cleaned event")
        if not self.data.get('cleaning_duration'):
            raise ValueError("Cleaning duration is required for bed cleaned event")


# Factory functions for creating events
def create_bed_prepared_event(
    bed_id: str,
    bed_parameters: Dict[str, Any],
    powder_properties: Dict[str, Any],
    bed_session_id: Optional[str] = None,
    process_id: Optional[str] = None,
    build_id: Optional[str] = None,
    source: str = "PowderBedService",
    correlation_id: Optional[str] = None
) -> BedPreparedEvent:
    """Create a bed prepared event."""
    return BedPreparedEvent(
        event_id=BaseEvent.generate_event_id(),
        event_name="BedPrepared",
        event_type=EventType.DOMAIN,
        severity=EventSeverity.INFO,
        occurred_at=datetime.utcnow(),
        source=source,
        bed_id=bed_id,
        bed_session_id=bed_session_id,
        process_id=process_id,
        build_id=build_id,
        data={
            "bed_parameters": bed_parameters,
            "powder_properties": powder_properties,
            "preparation_time": datetime.utcnow().isoformat()
        },
        correlation_id=correlation_id or BaseEvent.generate_correlation_id()
    )


def create_bed_disturbed_event(
    bed_id: str,
    disturbance_type: str,
    disturbance_severity: str,
    disturbance_details: Dict[str, Any],
    bed_session_id: Optional[str] = None,
    process_id: Optional[str] = None,
    build_id: Optional[str] = None,
    source: str = "PowderBedService",
    correlation_id: Optional[str] = None
) -> BedDisturbedEvent:
    """Create a bed disturbed event."""
    return BedDisturbedEvent(
        event_id=BaseEvent.generate_event_id(),
        event_name="BedDisturbed",
        event_type=EventType.DOMAIN,
        severity=EventSeverity.WARNING,
        occurred_at=datetime.utcnow(),
        source=source,
        bed_id=bed_id,
        bed_session_id=bed_session_id,
        process_id=process_id,
        build_id=build_id,
        data={
            "disturbance_type": disturbance_type,
            "disturbance_severity": disturbance_severity,
            "disturbance_details": disturbance_details,
            "disturbance_time": datetime.utcnow().isoformat()
        },
        correlation_id=correlation_id or BaseEvent.generate_correlation_id()
    )


def create_powder_depleted_event(
    bed_id: str,
    remaining_powder_percentage: float,
    powder_type: str,
    powder_usage_stats: Dict[str, Any],
    bed_session_id: Optional[str] = None,
    process_id: Optional[str] = None,
    build_id: Optional[str] = None,
    source: str = "PowderBedService",
    correlation_id: Optional[str] = None
) -> PowderDepletedEvent:
    """Create a powder depleted event."""
    return PowderDepletedEvent(
        event_id=BaseEvent.generate_event_id(),
        event_name="PowderDepleted",
        event_type=EventType.DOMAIN,
        severity=EventSeverity.WARNING,
        occurred_at=datetime.utcnow(),
        source=source,
        bed_id=bed_id,
        bed_session_id=bed_session_id,
        process_id=process_id,
        build_id=build_id,
        data={
            "remaining_powder_percentage": remaining_powder_percentage,
            "powder_type": powder_type,
            "powder_usage_stats": powder_usage_stats,
            "depletion_time": datetime.utcnow().isoformat()
        },
        correlation_id=correlation_id or BaseEvent.generate_correlation_id()
    )


def create_bed_quality_checked_event(
    bed_id: str,
    quality_metrics: Dict[str, Any],
    quality_score: float,
    quality_assessment: Dict[str, Any],
    bed_session_id: Optional[str] = None,
    process_id: Optional[str] = None,
    build_id: Optional[str] = None,
    source: str = "PowderBedService",
    correlation_id: Optional[str] = None
) -> BedQualityCheckedEvent:
    """Create a bed quality checked event."""
    return BedQualityCheckedEvent(
        event_id=BaseEvent.generate_event_id(),
        event_name="BedQualityChecked",
        event_type=EventType.DOMAIN,
        severity=EventSeverity.INFO,
        occurred_at=datetime.utcnow(),
        source=source,
        bed_id=bed_id,
        bed_session_id=bed_session_id,
        process_id=process_id,
        build_id=build_id,
        data={
            "quality_metrics": quality_metrics,
            "quality_score": quality_score,
            "quality_assessment": quality_assessment,
            "check_time": datetime.utcnow().isoformat()
        },
        correlation_id=correlation_id or BaseEvent.generate_correlation_id()
    )


def create_bed_cleaned_event(
    bed_id: str,
    cleaning_method: str,
    cleaning_duration: float,
    cleaning_effectiveness: Dict[str, Any],
    bed_session_id: Optional[str] = None,
    process_id: Optional[str] = None,
    build_id: Optional[str] = None,
    source: str = "PowderBedService",
    correlation_id: Optional[str] = None
) -> BedCleanedEvent:
    """Create a bed cleaned event."""
    return BedCleanedEvent(
        event_id=BaseEvent.generate_event_id(),
        event_name="BedCleaned",
        event_type=EventType.DOMAIN,
        severity=EventSeverity.INFO,
        occurred_at=datetime.utcnow(),
        source=source,
        bed_id=bed_id,
        bed_session_id=bed_session_id,
        process_id=process_id,
        build_id=build_id,
        data={
            "cleaning_method": cleaning_method,
            "cleaning_duration": cleaning_duration,
            "cleaning_effectiveness": cleaning_effectiveness,
            "cleaning_time": datetime.utcnow().isoformat()
        },
        correlation_id=correlation_id or BaseEvent.generate_correlation_id()
    )