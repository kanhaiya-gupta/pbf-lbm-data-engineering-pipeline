"""
Neo4j Relationship Models

This module contains Pydantic models for Neo4j knowledge graph relationships.
"""

from typing import Optional, Dict, Any
from datetime import datetime, timezone
from pydantic import BaseModel, Field, validator


# =============================================================================
# RELATIONSHIP MODELS
# =============================================================================

class ProcessMachineRelationship(BaseModel):
    """Process-Machine relationship model."""
    from_id: str = Field(..., description="Source process ID")
    to_id: str = Field(..., description="Target machine ID")
    relationship_type: str = Field(default="USES_MACHINE", description="Relationship type")
    duration: Optional[int] = Field(None, ge=0, description="Duration in seconds")
    start_time: Optional[datetime] = Field(None, description="Start time")
    end_time: Optional[datetime] = Field(None, description="End time")
    utilization: Optional[float] = Field(None, ge=0, le=1, description="Machine utilization")
    
    # Graph metadata
    graph_id: str = Field(..., description="Graph relationship ID")
    source: str = Field(default="neo4j", description="Data source")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Creation timestamp")

    @validator('end_time')
    def end_time_after_start(cls, v, values):
        if v is not None and 'start_time' in values and values['start_time'] is not None:
            if v <= values['start_time']:
                raise ValueError('End time must be after start time')
        return v


class ProcessPartRelationship(BaseModel):
    """Process-Part relationship model."""
    from_id: str = Field(..., description="Source process ID")
    to_id: str = Field(..., description="Target part ID")
    relationship_type: str = Field(default="CREATES_PART", description="Relationship type")
    quantity: Optional[int] = Field(None, ge=0, description="Quantity created")
    success_rate: Optional[float] = Field(None, ge=0, le=1, description="Success rate")
    creation_time: Optional[datetime] = Field(None, description="Creation time")
    quality_grade: Optional[str] = Field(None, description="Quality grade")
    
    # Graph metadata
    graph_id: str = Field(..., description="Graph relationship ID")
    source: str = Field(default="neo4j", description="Data source")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Creation timestamp")


class ProcessBuildRelationship(BaseModel):
    """Process-Build relationship model."""
    from_id: str = Field(..., description="Source process ID")
    to_id: str = Field(..., description="Target build ID")
    relationship_type: str = Field(default="PART_OF_BUILD", description="Relationship type")
    sequence: Optional[int] = Field(None, ge=0, description="Process sequence")
    priority: Optional[str] = Field(None, description="Process priority")
    status: Optional[str] = Field(None, description="Process status in build")
    
    # Graph metadata
    graph_id: str = Field(..., description="Graph relationship ID")
    source: str = Field(default="neo4j", description="Data source")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Creation timestamp")


class ProcessMaterialRelationship(BaseModel):
    """Process-Material relationship model."""
    from_id: str = Field(..., description="Source process ID")
    to_id: str = Field(..., description="Target material ID")
    relationship_type: str = Field(default="USES_MATERIAL", description="Relationship type")
    quantity: Optional[float] = Field(None, ge=0, description="Material quantity")
    unit: Optional[str] = Field(None, description="Quantity unit")
    consumption_rate: Optional[float] = Field(None, ge=0, description="Consumption rate")
    
    # Graph metadata
    graph_id: str = Field(..., description="Graph relationship ID")
    source: str = Field(default="neo4j", description="Data source")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Creation timestamp")


class ProcessQualityRelationship(BaseModel):
    """Process-Quality relationship model."""
    from_id: str = Field(..., description="Source process ID")
    to_id: str = Field(..., description="Target quality ID")
    relationship_type: str = Field(default="HAS_QUALITY", description="Relationship type")
    measured_at: Optional[datetime] = Field(None, description="Measurement timestamp")
    confidence: Optional[float] = Field(None, ge=0, le=1, description="Measurement confidence")
    correlation: Optional[float] = Field(None, ge=-1, le=1, description="Quality correlation")
    
    # Graph metadata
    graph_id: str = Field(..., description="Graph relationship ID")
    source: str = Field(default="neo4j", description="Data source")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Creation timestamp")


class ProcessSensorRelationship(BaseModel):
    """Process-Sensor relationship model."""
    from_id: str = Field(..., description="Source process ID")
    to_id: str = Field(..., description="Target sensor ID")
    relationship_type: str = Field(default="MONITORED_BY", description="Relationship type")
    sampling_rate: Optional[float] = Field(None, ge=0, description="Sampling rate in Hz")
    active: Optional[bool] = Field(None, description="Monitoring active status")
    coverage: Optional[float] = Field(None, ge=0, le=1, description="Monitoring coverage")
    accuracy: Optional[float] = Field(None, ge=0, le=1, description="Monitoring accuracy")
    
    # Graph metadata
    graph_id: str = Field(..., description="Graph relationship ID")
    source: str = Field(default="neo4j", description="Data source")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Creation timestamp")


class ProcessOperatorRelationship(BaseModel):
    """Process-Operator relationship model."""
    from_id: str = Field(..., description="Source process ID")
    to_id: str = Field(..., description="Target operator ID")
    relationship_type: str = Field(default="OPERATED_BY", description="Relationship type")
    shift: Optional[str] = Field(None, description="Work shift")
    experience_level: Optional[str] = Field(None, description="Experience level")
    supervision_level: Optional[str] = Field(None, description="Supervision level")
    
    # Graph metadata
    graph_id: str = Field(..., description="Graph relationship ID")
    source: str = Field(default="neo4j", description="Data source")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Creation timestamp")


class ProcessAlertRelationship(BaseModel):
    """Process-Alert relationship model."""
    from_id: str = Field(..., description="Source process ID")
    to_id: str = Field(..., description="Target alert ID")
    relationship_type: str = Field(default="GENERATES_ALERT", description="Relationship type")
    triggered_at: Optional[datetime] = Field(None, description="Alert trigger timestamp")
    severity: Optional[str] = Field(None, description="Alert severity")
    resolved: Optional[bool] = Field(None, description="Alert resolved status")
    
    # Graph metadata
    graph_id: str = Field(..., description="Graph relationship ID")
    source: str = Field(default="neo4j", description="Data source")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Creation timestamp")


class ProcessDefectRelationship(BaseModel):
    """Process-Defect relationship model."""
    from_id: str = Field(..., description="Source process ID")
    to_id: str = Field(..., description="Target defect ID")
    relationship_type: str = Field(default="HAS_DEFECT", description="Relationship type")
    detected_at: Optional[datetime] = Field(None, description="Defect detection timestamp")
    impact: Optional[str] = Field(None, description="Defect impact")
    resolution: Optional[str] = Field(None, description="Defect resolution")
    
    # Graph metadata
    graph_id: str = Field(..., description="Graph relationship ID")
    source: str = Field(default="neo4j", description="Data source")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Creation timestamp")


class ProcessImageRelationship(BaseModel):
    """Process-Image relationship model."""
    from_id: str = Field(..., description="Source process ID")
    to_id: str = Field(..., description="Target image ID")
    relationship_type: str = Field(default="CAPTURED_BY", description="Relationship type")
    timestamp: Optional[datetime] = Field(None, description="Image capture timestamp")
    purpose: Optional[str] = Field(None, description="Image purpose")
    quality: Optional[float] = Field(None, ge=0, le=1, description="Image quality")
    
    # Graph metadata
    graph_id: str = Field(..., description="Graph relationship ID")
    source: str = Field(default="neo4j", description="Data source")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Creation timestamp")


class ProcessLogRelationship(BaseModel):
    """Process-Log relationship model."""
    from_id: str = Field(..., description="Source process ID")
    to_id: str = Field(..., description="Target log ID")
    relationship_type: str = Field(default="LOGGED_IN", description="Relationship type")
    level: Optional[str] = Field(None, description="Log level")
    timestamp: Optional[datetime] = Field(None, description="Log timestamp")
    detail_level: Optional[str] = Field(None, description="Log detail level")
    
    # Graph metadata
    graph_id: str = Field(..., description="Graph relationship ID")
    source: str = Field(default="neo4j", description="Data source")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Creation timestamp")


# =============================================================================
# MACHINE RELATIONSHIPS
# =============================================================================

class MachineProcessRelationship(BaseModel):
    """Machine-Process relationship model."""
    from_id: str = Field(..., description="Source machine ID")
    to_id: str = Field(..., description="Target process ID")
    relationship_type: str = Field(default="HOSTS_PROCESS", description="Relationship type")
    capacity: Optional[int] = Field(None, ge=0, description="Machine capacity")
    utilization: Optional[float] = Field(None, ge=0, le=1, description="Utilization rate")
    
    # Graph metadata
    graph_id: str = Field(..., description="Graph relationship ID")
    source: str = Field(default="neo4j", description="Data source")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Creation timestamp")


class MachineSensorRelationship(BaseModel):
    """Machine-Sensor relationship model."""
    from_id: str = Field(..., description="Source machine ID")
    to_id: str = Field(..., description="Target sensor ID")
    relationship_type: str = Field(default="HAS_SENSOR", description="Relationship type")
    installation_date: Optional[datetime] = Field(None, description="Installation date")
    position: Optional[str] = Field(None, description="Sensor position")
    
    # Graph metadata
    graph_id: str = Field(..., description="Graph relationship ID")
    source: str = Field(default="neo4j", description="Data source")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Creation timestamp")


class MachineOperatorRelationship(BaseModel):
    """Machine-Operator relationship model."""
    from_id: str = Field(..., description="Source machine ID")
    to_id: str = Field(..., description="Target operator ID")
    relationship_type: str = Field(default="OPERATED_BY", description="Relationship type")
    authorization_level: Optional[str] = Field(None, description="Authorization level")
    training_completed: Optional[bool] = Field(None, description="Training completed")
    
    # Graph metadata
    graph_id: str = Field(..., description="Graph relationship ID")
    source: str = Field(default="neo4j", description="Data source")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Creation timestamp")


# =============================================================================
# PART RELATIONSHIPS
# =============================================================================

class PartBuildRelationship(BaseModel):
    """Part-Build relationship model."""
    from_id: str = Field(..., description="Source part ID")
    to_id: str = Field(..., description="Target build ID")
    relationship_type: str = Field(default="BELONGS_TO_BUILD", description="Relationship type")
    sequence: Optional[int] = Field(None, ge=0, description="Part sequence")
    priority: Optional[str] = Field(None, description="Part priority")
    
    # Graph metadata
    graph_id: str = Field(..., description="Graph relationship ID")
    source: str = Field(default="neo4j", description="Data source")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Creation timestamp")


class PartMaterialRelationship(BaseModel):
    """Part-Material relationship model."""
    from_id: str = Field(..., description="Source part ID")
    to_id: str = Field(..., description="Target material ID")
    relationship_type: str = Field(default="MADE_OF_MATERIAL", description="Relationship type")
    quantity: Optional[float] = Field(None, ge=0, description="Material quantity")
    unit: Optional[str] = Field(None, description="Quantity unit")
    
    # Graph metadata
    graph_id: str = Field(..., description="Graph relationship ID")
    source: str = Field(default="neo4j", description="Data source")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Creation timestamp")


class PartQualityRelationship(BaseModel):
    """Part-Quality relationship model."""
    from_id: str = Field(..., description="Source part ID")
    to_id: str = Field(..., description="Target quality ID")
    relationship_type: str = Field(default="HAS_QUALITY", description="Relationship type")
    measured_at: Optional[datetime] = Field(None, description="Measurement timestamp")
    acceptance_criteria: Optional[str] = Field(None, description="Acceptance criteria")
    
    # Graph metadata
    graph_id: str = Field(..., description="Graph relationship ID")
    source: str = Field(default="neo4j", description="Data source")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Creation timestamp")


class PartDefectRelationship(BaseModel):
    """Part-Defect relationship model."""
    from_id: str = Field(..., description="Source part ID")
    to_id: str = Field(..., description="Target defect ID")
    relationship_type: str = Field(default="HAS_DEFECT", description="Relationship type")
    detected_at: Optional[datetime] = Field(None, description="Defect detection timestamp")
    severity_impact: Optional[str] = Field(None, description="Severity impact")
    
    # Graph metadata
    graph_id: str = Field(..., description="Graph relationship ID")
    source: str = Field(default="neo4j", description="Data source")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Creation timestamp")


# =============================================================================
# SENSOR RELATIONSHIPS
# =============================================================================

class SensorAlertRelationship(BaseModel):
    """Sensor-Alert relationship model."""
    from_id: str = Field(..., description="Source sensor ID")
    to_id: str = Field(..., description="Target alert ID")
    relationship_type: str = Field(default="TRIGGERS_ALERT", description="Relationship type")
    threshold_exceeded: Optional[bool] = Field(None, description="Threshold exceeded")
    trigger_value: Optional[float] = Field(None, description="Trigger value")
    
    # Graph metadata
    graph_id: str = Field(..., description="Graph relationship ID")
    source: str = Field(default="neo4j", description="Data source")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Creation timestamp")


class SensorDefectRelationship(BaseModel):
    """Sensor-Defect relationship model."""
    from_id: str = Field(..., description="Source sensor ID")
    to_id: str = Field(..., description="Target defect ID")
    relationship_type: str = Field(default="DETECTED_BY_SENSOR", description="Relationship type")
    detection_confidence: Optional[float] = Field(None, ge=0, le=1, description="Detection confidence")
    detection_method: Optional[str] = Field(None, description="Detection method")
    
    # Graph metadata
    graph_id: str = Field(..., description="Graph relationship ID")
    source: str = Field(default="neo4j", description="Data source")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Creation timestamp")


# =============================================================================
# USER RELATIONSHIPS
# =============================================================================

class UserAlertRelationship(BaseModel):
    """User-Alert relationship model."""
    from_id: str = Field(..., description="Source user ID")
    to_id: str = Field(..., description="Target alert ID")
    relationship_type: str = Field(default="RESOLVES_ALERT", description="Relationship type")
    response_time: Optional[int] = Field(None, ge=0, description="Response time in seconds")
    resolution_notes: Optional[str] = Field(None, description="Resolution notes")
    
    # Graph metadata
    graph_id: str = Field(..., description="Graph relationship ID")
    source: str = Field(default="neo4j", description="Data source")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Creation timestamp")


class UserBuildRelationship(BaseModel):
    """User-Build relationship model."""
    from_id: str = Field(..., description="Source user ID")
    to_id: str = Field(..., description="Target build ID")
    relationship_type: str = Field(default="CREATES_BUILD", description="Relationship type")
    created_at: Optional[datetime] = Field(None, description="Creation timestamp")
    design_authority: Optional[bool] = Field(None, description="Design authority")
    
    # Graph metadata
    graph_id: str = Field(..., description="Graph relationship ID")
    source: str = Field(default="neo4j", description="Data source")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Creation timestamp")
