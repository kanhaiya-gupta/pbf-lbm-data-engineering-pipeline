"""
CT scan domain events.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass, field

from .base_event import BaseEvent, EventType, EventSeverity
from ..enums import DefectType, QualityTier


@dataclass(frozen=True)
class CTScanEvent(BaseEvent):
    """Base class for CT scan events."""
    
    # Scan identification
    scan_id: str = field(default="")
    scan_session_id: Optional[str] = None
    
    # Process information
    process_id: Optional[str] = None
    build_id: Optional[str] = None
    part_id: Optional[str] = None
    
    def validate(self) -> None:
        """Validate CT scan event."""
        if not self.scan_id:
            raise ValueError("Scan ID cannot be empty")


@dataclass(frozen=True)
class ScanStartedEvent(CTScanEvent):
    """Event raised when a CT scan starts."""
    
    def __post_init__(self):
        """Set default values for scan started event."""
        object.__setattr__(self, 'event_name', 'ScanStarted')
        object.__setattr__(self, 'event_type', EventType.DOMAIN)
        object.__setattr__(self, 'severity', EventSeverity.INFO)
        super().__post_init__()
    
    def validate(self) -> None:
        """Validate scan started event."""
        super().validate()
        if not self.data.get('scan_parameters'):
            raise ValueError("Scan parameters are required for scan started event")


@dataclass(frozen=True)
class ScanCompletedEvent(CTScanEvent):
    """Event raised when a CT scan completes successfully."""
    
    def __post_init__(self):
        """Set default values for scan completed event."""
        object.__setattr__(self, 'event_name', 'ScanCompleted')
        object.__setattr__(self, 'event_type', EventType.DOMAIN)
        object.__setattr__(self, 'severity', EventSeverity.INFO)
        super().__post_init__()
    
    def validate(self) -> None:
        """Validate scan completed event."""
        super().validate()
        if not self.data.get('scan_duration'):
            raise ValueError("Scan duration is required for scan completed event")
        if not self.data.get('image_count'):
            raise ValueError("Image count is required for scan completed event")


@dataclass(frozen=True)
class ScanFailedEvent(CTScanEvent):
    """Event raised when a CT scan fails."""
    
    def __post_init__(self):
        """Set default values for scan failed event."""
        object.__setattr__(self, 'event_name', 'ScanFailed')
        object.__setattr__(self, 'event_type', EventType.DOMAIN)
        object.__setattr__(self, 'severity', EventSeverity.ERROR)
        super().__post_init__()
    
    def validate(self) -> None:
        """Validate scan failed event."""
        super().validate()
        if not self.data.get('failure_reason'):
            raise ValueError("Failure reason is required for scan failed event")
        if not self.data.get('error_code'):
            raise ValueError("Error code is required for scan failed event")


@dataclass(frozen=True)
class DefectDetectedEvent(CTScanEvent):
    """Event raised when a defect is detected in CT scan analysis."""
    
    def __post_init__(self):
        """Set default values for defect detected event."""
        object.__setattr__(self, 'event_name', 'DefectDetected')
        object.__setattr__(self, 'event_type', EventType.DOMAIN)
        object.__setattr__(self, 'severity', EventSeverity.WARNING)
        super().__post_init__()
    
    def validate(self) -> None:
        """Validate defect detected event."""
        super().validate()
        if not self.data.get('defect_type'):
            raise ValueError("Defect type is required for defect detected event")
        if not self.data.get('defect_severity'):
            raise ValueError("Defect severity is required for defect detected event")
        if not self.data.get('defect_location'):
            raise ValueError("Defect location is required for defect detected event")


@dataclass(frozen=True)
class QualityAnalysisCompletedEvent(CTScanEvent):
    """Event raised when quality analysis of CT scan is completed."""
    
    def __post_init__(self):
        """Set default values for quality analysis completed event."""
        object.__setattr__(self, 'event_name', 'QualityAnalysisCompleted')
        object.__setattr__(self, 'event_type', EventType.DOMAIN)
        object.__setattr__(self, 'severity', EventSeverity.INFO)
        super().__post_init__()
    
    def validate(self) -> None:
        """Validate quality analysis completed event."""
        super().validate()
        if not self.data.get('quality_metrics'):
            raise ValueError("Quality metrics are required for quality analysis completed event")
        if not self.data.get('quality_score'):
            raise ValueError("Quality score is required for quality analysis completed event")


@dataclass(frozen=True)
class ReportGeneratedEvent(CTScanEvent):
    """Event raised when a CT scan report is generated."""
    
    def __post_init__(self):
        """Set default values for report generated event."""
        object.__setattr__(self, 'event_name', 'ReportGenerated')
        object.__setattr__(self, 'event_type', EventType.DOMAIN)
        object.__setattr__(self, 'severity', EventSeverity.INFO)
        super().__post_init__()
    
    def validate(self) -> None:
        """Validate report generated event."""
        super().validate()
        if not self.data.get('report_type'):
            raise ValueError("Report type is required for report generated event")
        if not self.data.get('report_path'):
            raise ValueError("Report path is required for report generated event")


# Factory functions for creating events
def create_scan_started_event(
    scan_id: str,
    scan_parameters: Dict[str, Any],
    scan_session_id: Optional[str] = None,
    process_id: Optional[str] = None,
    build_id: Optional[str] = None,
    part_id: Optional[str] = None,
    source: str = "CTScannerService",
    correlation_id: Optional[str] = None
) -> ScanStartedEvent:
    """Create a scan started event."""
    return ScanStartedEvent(
        event_id=BaseEvent.generate_event_id(),
        event_name="ScanStarted",
        event_type=EventType.DOMAIN,
        severity=EventSeverity.INFO,
        occurred_at=datetime.utcnow(),
        source=source,
        scan_id=scan_id,
        scan_session_id=scan_session_id,
        process_id=process_id,
        build_id=build_id,
        part_id=part_id,
        data={
            "scan_parameters": scan_parameters,
            "start_time": datetime.utcnow().isoformat()
        },
        correlation_id=correlation_id or BaseEvent.generate_correlation_id()
    )


def create_scan_completed_event(
    scan_id: str,
    scan_duration: float,
    image_count: int,
    scan_quality: Dict[str, Any],
    scan_session_id: Optional[str] = None,
    process_id: Optional[str] = None,
    build_id: Optional[str] = None,
    part_id: Optional[str] = None,
    source: str = "CTScannerService",
    correlation_id: Optional[str] = None
) -> ScanCompletedEvent:
    """Create a scan completed event."""
    return ScanCompletedEvent(
        event_id=BaseEvent.generate_event_id(),
        event_name="ScanCompleted",
        event_type=EventType.DOMAIN,
        severity=EventSeverity.INFO,
        occurred_at=datetime.utcnow(),
        source=source,
        scan_id=scan_id,
        scan_session_id=scan_session_id,
        process_id=process_id,
        build_id=build_id,
        part_id=part_id,
        data={
            "scan_duration": scan_duration,
            "image_count": image_count,
            "scan_quality": scan_quality,
            "completion_time": datetime.utcnow().isoformat()
        },
        correlation_id=correlation_id or BaseEvent.generate_correlation_id()
    )


def create_defect_detected_event(
    scan_id: str,
    defect_type: DefectType,
    defect_severity: str,
    defect_location: Dict[str, float],
    defect_properties: Dict[str, Any],
    scan_session_id: Optional[str] = None,
    process_id: Optional[str] = None,
    build_id: Optional[str] = None,
    part_id: Optional[str] = None,
    source: str = "CTAnalysisService",
    correlation_id: Optional[str] = None
) -> DefectDetectedEvent:
    """Create a defect detected event."""
    return DefectDetectedEvent(
        event_id=BaseEvent.generate_event_id(),
        event_name="DefectDetected",
        event_type=EventType.DOMAIN,
        severity=EventSeverity.WARNING,
        occurred_at=datetime.utcnow(),
        source=source,
        scan_id=scan_id,
        scan_session_id=scan_session_id,
        process_id=process_id,
        build_id=build_id,
        part_id=part_id,
        data={
            "defect_type": defect_type.value,
            "defect_severity": defect_severity,
            "defect_location": defect_location,
            "defect_properties": defect_properties,
            "detection_time": datetime.utcnow().isoformat()
        },
        correlation_id=correlation_id or BaseEvent.generate_correlation_id()
    )


def create_quality_analysis_completed_event(
    scan_id: str,
    quality_metrics: Dict[str, Any],
    quality_score: float,
    quality_tier: QualityTier,
    defect_summary: Dict[str, Any],
    scan_session_id: Optional[str] = None,
    process_id: Optional[str] = None,
    build_id: Optional[str] = None,
    part_id: Optional[str] = None,
    source: str = "CTAnalysisService",
    correlation_id: Optional[str] = None
) -> QualityAnalysisCompletedEvent:
    """Create a quality analysis completed event."""
    return QualityAnalysisCompletedEvent(
        event_id=BaseEvent.generate_event_id(),
        event_name="QualityAnalysisCompleted",
        event_type=EventType.DOMAIN,
        severity=EventSeverity.INFO,
        occurred_at=datetime.utcnow(),
        source=source,
        scan_id=scan_id,
        scan_session_id=scan_session_id,
        process_id=process_id,
        build_id=build_id,
        part_id=part_id,
        data={
            "quality_metrics": quality_metrics,
            "quality_score": quality_score,
            "quality_tier": quality_tier.value,
            "defect_summary": defect_summary,
            "analysis_time": datetime.utcnow().isoformat()
        },
        correlation_id=correlation_id or BaseEvent.generate_correlation_id()
    )


def create_report_generated_event(
    scan_id: str,
    report_type: str,
    report_path: str,
    report_metadata: Dict[str, Any],
    scan_session_id: Optional[str] = None,
    process_id: Optional[str] = None,
    build_id: Optional[str] = None,
    part_id: Optional[str] = None,
    source: str = "ReportService",
    correlation_id: Optional[str] = None
) -> ReportGeneratedEvent:
    """Create a report generated event."""
    return ReportGeneratedEvent(
        event_id=BaseEvent.generate_event_id(),
        event_name="ReportGenerated",
        event_type=EventType.DOMAIN,
        severity=EventSeverity.INFO,
        occurred_at=datetime.utcnow(),
        source=source,
        scan_id=scan_id,
        scan_session_id=scan_session_id,
        process_id=process_id,
        build_id=build_id,
        part_id=part_id,
        data={
            "report_type": report_type,
            "report_path": report_path,
            "report_metadata": report_metadata,
            "generation_time": datetime.utcnow().isoformat()
        },
        correlation_id=correlation_id or BaseEvent.generate_correlation_id()
    )