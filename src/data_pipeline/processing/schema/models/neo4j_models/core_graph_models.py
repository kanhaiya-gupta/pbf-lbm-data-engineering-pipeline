"""
Neo4j Core Graph Models

This module contains Pydantic models for core Neo4j knowledge graph nodes.
"""

from typing import Optional, List, Dict, Any, Union
from datetime import datetime, date, timezone
from enum import Enum
from pydantic import BaseModel, Field, validator, root_validator
import uuid


# =============================================================================
# ENUMS
# =============================================================================

class ProcessStatus(str, Enum):
    """Process status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"
    IN_PROGRESS = "in_progress"
    UNKNOWN = "unknown"

class MachineStatus(str, Enum):
    """Machine status enumeration."""
    OPERATIONAL = "operational"
    MAINTENANCE = "maintenance"
    IDLE = "idle"
    ERROR = "error"
    OFFLINE = "offline"
    CALIBRATING = "calibrating"
    ACTIVE = "active"

class PartStatus(str, Enum):
    """Part status enumeration."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    INSPECTED = "inspected"
    REJECTED = "rejected"
    UNKNOWN = "unknown"

class BuildStatus(str, Enum):
    """Build status enumeration."""
    PLANNING = "planning"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"
    UNKNOWN = "unknown"

class QualityGrade(str, Enum):
    """Quality grade enumeration."""
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    F = "F"

class SensorStatus(str, Enum):
    """Sensor status enumeration."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    CALIBRATING = "calibrating"
    ERROR = "error"
    MAINTENANCE = "maintenance"
    UNKNOWN = "unknown"

class AlertSeverity(str, Enum):
    """Alert severity enumeration."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class AlertStatus(str, Enum):
    """Alert status enumeration."""
    ACTIVE = "active"
    RESOLVED = "resolved"
    ACKNOWLEDGED = "acknowledged"
    ESCALATED = "escalated"

class DefectSeverity(str, Enum):
    """Defect severity enumeration."""
    MINOR = "minor"
    MODERATE = "moderate"
    MAJOR = "major"
    CRITICAL = "critical"

class DefectStatus(str, Enum):
    """Defect status enumeration."""
    DETECTED = "detected"
    CONFIRMED = "confirmed"
    RESOLVED = "resolved"
    IGNORED = "ignored"

class UserRole(str, Enum):
    """User role enumeration."""
    ADMIN = "admin"
    ENGINEER = "engineer"
    OPERATOR = "operator"
    SUPERVISOR = "supervisor"
    QUALITY = "quality"
    MAINTENANCE = "maintenance"

class LogLevel(str, Enum):
    """Log level enumeration."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class ImageType(str, Enum):
    """Image type enumeration."""
    PROCESS_MONITORING = "process_monitoring"
    QUALITY_INSPECTION = "quality_inspection"
    DEFECT_ANALYSIS = "defect_analysis"
    MAINTENANCE = "maintenance"
    DOCUMENTATION = "documentation"


# =============================================================================
# COMPONENT MODELS
# =============================================================================

class Dimensions(BaseModel):
    """Dimensions model."""
    x: float = Field(..., ge=0, description="X dimension in mm")
    y: float = Field(..., ge=0, description="Y dimension in mm")
    z: float = Field(..., ge=0, description="Z dimension in mm")

class Location(BaseModel):
    """Location model."""
    x: float = Field(..., description="X coordinate")
    y: float = Field(..., description="Y coordinate")
    z: float = Field(..., description="Z coordinate")

class QualityMetrics(BaseModel):
    """Quality metrics model."""
    density: float = Field(..., ge=0, le=1, description="Density (0-1)")
    surface_roughness: float = Field(..., ge=0, description="Surface roughness in μm")
    dimensional_accuracy: float = Field(..., ge=0, description="Dimensional accuracy in μm")
    tensile_strength: Optional[float] = Field(None, ge=0, description="Tensile strength in MPa")
    yield_strength: Optional[float] = Field(None, ge=0, description="Yield strength in MPa")
    hardness: Optional[float] = Field(None, ge=0, description="Hardness in HV")

class MaterialProperties(BaseModel):
    """Material properties model."""
    density: float = Field(..., ge=0, description="Material density in g/cm³")
    melting_point: float = Field(..., ge=0, description="Melting point in °C")
    thermal_conductivity: float = Field(..., ge=0, description="Thermal conductivity in W/m·K")
    yield_strength: Optional[float] = Field(None, ge=0, description="Yield strength in MPa")
    tensile_strength: Optional[float] = Field(None, ge=0, description="Tensile strength in MPa")

class SensorRange(BaseModel):
    """Sensor range model."""
    min: float = Field(..., description="Minimum value")
    max: float = Field(..., description="Maximum value")

    @validator('max')
    def max_greater_than_min(cls, v, values):
        if 'min' in values and v <= values['min']:
            raise ValueError('max must be greater than min')
        return v


# =============================================================================
# CORE NODE MODELS
# =============================================================================

class ProcessModel(BaseModel):
    """Process node model."""
    process_id: str = Field(..., min_length=1, max_length=50, description="Unique process identifier")
    timestamp: datetime = Field(..., description="Process timestamp")
    material_type: str = Field(..., min_length=1, max_length=50, description="Material type")
    quality_grade: Optional[QualityGrade] = Field(None, description="Quality grade")
    laser_power: float = Field(..., ge=0, le=1000, description="Laser power in watts")
    scan_speed: float = Field(..., ge=0, le=10000, description="Scan speed in mm/s")
    layer_thickness: float = Field(..., ge=0.01, le=1.0, description="Layer thickness in mm")
    density: Optional[float] = Field(None, ge=0, le=1, description="Process density")
    surface_roughness: Optional[float] = Field(None, ge=0, description="Surface roughness in μm")
    status: ProcessStatus = Field(..., description="Process status")
    duration: Optional[int] = Field(None, ge=0, description="Process duration in seconds")
    energy_consumption: Optional[float] = Field(None, ge=0, description="Energy consumption in kWh")
    powder_usage: Optional[float] = Field(None, ge=0, description="Powder usage in kg")
    build_temperature: Optional[float] = Field(None, ge=0, le=2000, description="Build temperature in °C")
    chamber_pressure: Optional[float] = Field(None, ge=0, le=1000, description="Chamber pressure in bar")
    hatch_spacing: Optional[float] = Field(None, ge=0.01, le=1.0, description="Hatch spacing in mm")
    exposure_time: Optional[float] = Field(None, ge=0, le=3600, description="Exposure time in seconds")
    
    # Relationship fields for graph connections
    machine_id: Optional[str] = Field(None, description="Associated machine ID")
    build_id: Optional[str] = Field(None, description="Associated build ID")
    part_id: Optional[str] = Field(None, description="Associated part ID")
    
    # Graph metadata
    graph_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Graph node ID")
    source: str = Field(default="neo4j", description="Data source")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")

    @validator('timestamp')
    def timestamp_not_future(cls, v):
        if v > datetime.now(timezone.utc):
            raise ValueError('Process timestamp cannot be in the future')
        return v

    @validator('density')
    def density_valid_range(cls, v):
        if v is not None and (v < 0 or v > 1):
            raise ValueError('Density must be between 0 and 1')
        return v


class MachineModel(BaseModel):
    """Machine node model."""
    machine_id: str = Field(..., min_length=1, max_length=50, description="Unique machine identifier")
    machine_type: str = Field(..., min_length=1, max_length=50, description="Machine type")
    model: str = Field(..., min_length=1, max_length=100, description="Machine model")
    status: MachineStatus = Field(..., description="Machine status")
    location: str = Field(..., min_length=1, max_length=200, description="Machine location")
    installation_date: date = Field(..., description="Installation date")
    max_build_volume: Optional[Dimensions] = Field(None, description="Maximum build volume")
    laser_power_max: Optional[float] = Field(None, ge=0, le=10000, description="Maximum laser power in watts")
    layer_thickness_range: Optional[Dict[str, float]] = Field(None, description="Layer thickness range")
    accuracy: Optional[float] = Field(None, ge=0, le=1, description="Machine accuracy")
    maintenance_date: Optional[date] = Field(None, description="Last maintenance date")
    utilization_rate: Optional[float] = Field(None, ge=0, le=1, description="Utilization rate")
    
    # Graph metadata
    graph_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Graph node ID")
    source: str = Field(default="neo4j", description="Data source")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")

    @validator('installation_date')
    def installation_date_not_future(cls, v):
        if v > date.today():
            raise ValueError('Installation date cannot be in the future')
        return v

    @validator('maintenance_date')
    def maintenance_date_not_future(cls, v):
        if v is not None and v > date.today():
            raise ValueError('Maintenance date cannot be in the future')
        return v


class PartModel(BaseModel):
    """Part node model."""
    part_id: str = Field(..., min_length=1, max_length=50, description="Unique part identifier")
    part_type: str = Field(..., min_length=1, max_length=100, description="Part type")
    material_type: str = Field(..., min_length=1, max_length=50, description="Material type")
    dimensions: Optional[Dimensions] = Field(None, description="Part dimensions")
    volume: Optional[float] = Field(None, ge=0, description="Part volume in cm³")
    surface_area: Optional[float] = Field(None, ge=0, description="Surface area in cm²")
    weight: Optional[float] = Field(None, ge=0, description="Part weight in kg")
    status: PartStatus = Field(..., description="Part status")
    quality_grade: Optional[QualityGrade] = Field(None, description="Quality grade")
    tolerance: Optional[float] = Field(None, ge=0, description="Tolerance in mm")
    finish_quality: Optional[str] = Field(None, max_length=50, description="Finish quality")
    
    # Graph metadata
    graph_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Graph node ID")
    source: str = Field(default="neo4j", description="Data source")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")


class BuildModel(BaseModel):
    """Build node model."""
    build_id: str = Field(..., min_length=1, max_length=50, description="Unique build identifier")
    build_name: str = Field(..., min_length=1, max_length=200, description="Build name")
    status: BuildStatus = Field(..., description="Build status")
    created_date: date = Field(..., description="Build creation date")
    completed_date: Optional[date] = Field(None, description="Build completion date")
    total_parts: Optional[int] = Field(None, ge=0, description="Total number of parts")
    success_rate: Optional[float] = Field(None, ge=0, le=1, description="Success rate")
    total_duration: Optional[int] = Field(None, ge=0, description="Total duration in seconds")
    material_usage: Optional[float] = Field(None, ge=0, description="Material usage in kg")
    energy_consumption: Optional[float] = Field(None, ge=0, description="Energy consumption in kWh")
    quality_grade: Optional[QualityGrade] = Field(None, description="Overall quality grade")
    
    # Graph metadata
    graph_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Graph node ID")
    source: str = Field(default="neo4j", description="Data source")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")

    @validator('completed_date')
    def completed_date_after_created(cls, v, values):
        if v is not None and 'created_date' in values and v < values['created_date']:
            raise ValueError('Completed date cannot be before created date')
        return v


class MaterialModel(BaseModel):
    """Material node model."""
    material_type: str = Field(..., min_length=1, max_length=50, description="Material type")
    properties: Optional[MaterialProperties] = Field(None, description="Material properties")
    supplier: Optional[str] = Field(None, max_length=100, description="Material supplier")
    certification: Optional[str] = Field(None, max_length=100, description="Material certification")
    batch_number: Optional[str] = Field(None, max_length=50, description="Batch number")
    condition: Optional[str] = Field(None, max_length=50, description="Material condition")
    storage_temperature: Optional[float] = Field(None, ge=-50, le=100, description="Storage temperature in °C")
    humidity: Optional[float] = Field(None, ge=0, le=100, description="Humidity percentage")
    shelf_life: Optional[int] = Field(None, ge=0, description="Shelf life in days")
    
    # Graph metadata
    graph_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Graph node ID")
    source: str = Field(default="neo4j", description="Data source")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")


class QualityModel(BaseModel):
    """Quality node model."""
    grade: QualityGrade = Field(..., description="Quality grade")
    metrics: Optional[QualityMetrics] = Field(None, description="Quality metrics")
    standards: Optional[List[str]] = Field(None, description="Quality standards")
    inspector: Optional[str] = Field(None, max_length=100, description="Inspector name")
    inspection_date: Optional[date] = Field(None, description="Inspection date")
    test_method: Optional[str] = Field(None, max_length=100, description="Test method")
    confidence_level: Optional[float] = Field(None, ge=0, le=1, description="Confidence level")
    
    # Graph metadata
    graph_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Graph node ID")
    source: str = Field(default="neo4j", description="Data source")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")


class SensorModel(BaseModel):
    """Sensor node model."""
    sensor_id: str = Field(..., min_length=1, max_length=50, description="Unique sensor identifier")
    sensor_type: str = Field(..., min_length=1, max_length=50, description="Sensor type")
    location: str = Field(..., min_length=1, max_length=100, description="Sensor location")
    model: Optional[str] = Field(None, max_length=100, description="Sensor model")
    calibration_date: Optional[date] = Field(None, description="Calibration date")
    accuracy: Optional[float] = Field(None, ge=0, le=1, description="Sensor accuracy")
    range: Optional[SensorRange] = Field(None, description="Sensor range")
    sampling_rate: Optional[float] = Field(None, ge=0, description="Sampling rate in Hz")
    status: SensorStatus = Field(..., description="Sensor status")
    last_reading: Optional[float] = Field(None, description="Last sensor reading")
    
    # Graph metadata
    graph_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Graph node ID")
    source: str = Field(default="neo4j", description="Data source")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")


class UserModel(BaseModel):
    """User node model."""
    user_id: str = Field(..., min_length=1, max_length=50, description="Unique user identifier")
    username: str = Field(..., min_length=1, max_length=100, description="Username")
    name: str = Field(..., min_length=1, max_length=100, description="Full name")
    role: UserRole = Field(..., description="User role")
    department: Optional[str] = Field(None, max_length=100, description="Department")
    email: Optional[str] = Field(None, max_length=200, description="Email address")
    phone: Optional[str] = Field(None, max_length=20, description="Phone number")
    active: bool = Field(True, description="User active status")
    last_login: Optional[datetime] = Field(None, description="Last login timestamp")
    hire_date: Optional[date] = Field(None, description="Hire date")
    
    # Graph metadata
    graph_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Graph node ID")
    source: str = Field(default="neo4j", description="Data source")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")

    @validator('email')
    def email_format(cls, v):
        if v is not None and '@' not in v:
            raise ValueError('Invalid email format')
        return v


class OperatorModel(BaseModel):
    """Operator node model."""
    operator_id: str = Field(..., min_length=1, max_length=50, description="Unique operator identifier")
    name: str = Field(..., min_length=1, max_length=100, description="Operator name")
    certification: str = Field(..., min_length=1, max_length=100, description="Certification level")
    experience_years: int = Field(..., ge=0, le=50, description="Years of experience")
    shift: str = Field(..., min_length=1, max_length=20, description="Work shift")
    machine_authorization: Optional[List[str]] = Field(None, description="Authorized machines")
    training_completed: Optional[List[str]] = Field(None, description="Completed training")
    performance_rating: Optional[float] = Field(None, ge=0, le=5, description="Performance rating")
    
    # Graph metadata
    graph_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Graph node ID")
    source: str = Field(default="neo4j", description="Data source")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")


class AlertModel(BaseModel):
    """Alert node model."""
    alert_id: str = Field(..., min_length=1, max_length=50, description="Unique alert identifier")
    severity: AlertSeverity = Field(..., description="Alert severity")
    status: AlertStatus = Field(..., description="Alert status")
    timestamp: datetime = Field(..., description="Alert timestamp")
    message: str = Field(..., min_length=1, max_length=500, description="Alert message")
    threshold: Optional[float] = Field(None, description="Alert threshold")
    actual_value: Optional[float] = Field(None, description="Actual value")
    resolution_time: Optional[int] = Field(None, ge=0, description="Resolution time in seconds")
    resolved_by: Optional[str] = Field(None, max_length=100, description="Resolved by user")
    resolution_notes: Optional[str] = Field(None, max_length=500, description="Resolution notes")
    
    # Graph metadata
    graph_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Graph node ID")
    source: str = Field(default="neo4j", description="Data source")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")


class DefectModel(BaseModel):
    """Defect node model."""
    defect_id: str = Field(..., min_length=1, max_length=50, description="Unique defect identifier")
    defect_type: str = Field(..., min_length=1, max_length=100, description="Defect type")
    severity: DefectSeverity = Field(..., description="Defect severity")
    location: Optional[Location] = Field(None, description="Defect location")
    size: Optional[float] = Field(None, ge=0, description="Defect size")
    detection_method: Optional[str] = Field(None, max_length=100, description="Detection method")
    confidence: Optional[float] = Field(None, ge=0, le=1, description="Detection confidence")
    status: DefectStatus = Field(..., description="Defect status")
    timestamp: Optional[datetime] = Field(None, description="Defect detection timestamp")
    impact: Optional[str] = Field(None, max_length=100, description="Defect impact")
    
    # Graph metadata
    graph_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Graph node ID")
    source: str = Field(default="neo4j", description="Data source")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")


class ImageModel(BaseModel):
    """Image node model."""
    image_id: str = Field(..., min_length=1, max_length=50, description="Unique image identifier")
    image_type: ImageType = Field(..., description="Image type")
    format: str = Field(..., min_length=1, max_length=10, description="Image format")
    resolution: Optional[Dict[str, int]] = Field(None, description="Image resolution")
    file_size: Optional[int] = Field(None, ge=0, description="File size in bytes")
    timestamp: Optional[datetime] = Field(None, description="Image timestamp")
    camera_position: Optional[Location] = Field(None, description="Camera position")
    lighting_conditions: Optional[str] = Field(None, max_length=50, description="Lighting conditions")
    quality_score: Optional[float] = Field(None, ge=0, le=1, description="Image quality score")
    file_path: Optional[str] = Field(None, max_length=500, description="File path")
    
    # Graph metadata
    graph_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Graph node ID")
    source: str = Field(default="neo4j", description="Data source")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")


class LogModel(BaseModel):
    """Log node model."""
    log_id: str = Field(..., min_length=1, max_length=50, description="Unique log identifier")
    level: LogLevel = Field(..., description="Log level")
    source: str = Field(..., min_length=1, max_length=100, description="Log source")
    message: str = Field(..., min_length=1, max_length=1000, description="Log message")
    timestamp: datetime = Field(..., description="Log timestamp")
    component: Optional[str] = Field(None, max_length=100, description="Component")
    session_id: Optional[str] = Field(None, max_length=100, description="Session ID")
    user_id: Optional[str] = Field(None, max_length=50, description="User ID")
    
    # Graph metadata
    graph_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Graph node ID")
    source: str = Field(default="neo4j", description="Data source")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")


class InspectionModel(BaseModel):
    """Inspection node model."""
    inspection_id: str = Field(..., min_length=1, max_length=50, description="Unique inspection identifier")
    inspector: str = Field(..., min_length=1, max_length=100, description="Inspector name")
    inspection_date: date = Field(..., description="Inspection date")
    inspection_type: str = Field(..., min_length=1, max_length=100, description="Inspection type")
    result: str = Field(..., min_length=1, max_length=100, description="Inspection result")
    notes: Optional[str] = Field(None, max_length=500, description="Inspection notes")
    standards: Optional[List[str]] = Field(None, description="Applied standards")
    
    # Graph metadata
    graph_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Graph node ID")
    source: str = Field(default="neo4j", description="Data source")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")


# =============================================================================
# IMAGE NODE MODELS
# =============================================================================

class ThermalImageModel(BaseModel):
    """Thermal image node model."""
    # Core fields
    thermal_id: str = Field(..., min_length=1, max_length=100, description="Unique thermal image identifier")
    process_id: str = Field(..., min_length=1, max_length=100, description="Associated process ID")
    thermal_type: str = Field(..., min_length=1, max_length=50, description="Type of thermal image")
    file_path: str = Field(..., min_length=1, max_length=500, description="Thermal image file path")
    file_size: int = Field(..., ge=0, description="File size in bytes")
    dimensions: Dict[str, Union[int, float]] = Field(..., description="Image dimensions (width, height, resolution)")
    format: str = Field(..., min_length=1, max_length=10, description="Image format (TIFF, PNG, etc.)")
    temperature_range: Dict[str, float] = Field(..., description="Temperature range (min, max) in Celsius")
    emissivity: float = Field(..., ge=0, le=1, description="Surface emissivity coefficient")
    ambient_temperature: float = Field(..., description="Ambient temperature in Celsius")
    timestamp: datetime = Field(..., description="Thermal image capture timestamp")
    camera_position: Optional[Dict[str, float]] = Field(None, description="Thermal camera position (x, y, z)")
    distance_to_target: Optional[float] = Field(None, gt=0, description="Distance to target in meters")
    field_of_view: Optional[Dict[str, float]] = Field(None, description="Field of view (horizontal, vertical) in degrees")
    thermal_resolution: Optional[float] = Field(None, gt=0, description="Thermal resolution in mK")
    quality_score: Optional[float] = Field(None, ge=0, le=1, description="Thermal image quality score")
    hot_spots_detected: Optional[List[Dict[str, Any]]] = Field(None, description="Detected hot spots")
    cold_spots_detected: Optional[List[Dict[str, Any]]] = Field(None, description="Detected cold spots")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional thermal image metadata")
    
    # Graph metadata
    graph_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Graph node ID")
    source: str = Field(default="neo4j", description="Data source")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")

class ProcessImageModel(BaseModel):
    """Process image node model."""
    # Core fields
    image_id: str = Field(..., min_length=1, max_length=100, description="Unique image identifier")
    process_id: str = Field(..., min_length=1, max_length=100, description="Associated process ID")
    image_type: str = Field(..., min_length=1, max_length=50, description="Type of process image")
    file_path: str = Field(..., min_length=1, max_length=500, description="Image file path")
    file_size: int = Field(..., ge=0, description="File size in bytes")
    dimensions: Dict[str, Union[int, float]] = Field(..., description="Image dimensions (width, height, resolution)")
    format: str = Field(..., min_length=1, max_length=10, description="Image format (PNG, JPEG, etc.)")
    resolution: float = Field(..., gt=0, description="Image resolution in DPI")
    timestamp: datetime = Field(..., description="Image capture timestamp")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional image metadata")
    
    # Graph metadata
    graph_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Graph node ID")
    source: str = Field(default="neo4j", description="Data source")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")

class CTScanImageModel(BaseModel):
    """CT scan image node model."""
    # Core fields
    scan_id: str = Field(..., min_length=1, max_length=100, description="Unique scan identifier")
    part_id: str = Field(..., min_length=1, max_length=100, description="Associated part ID")
    scan_type: str = Field(..., min_length=1, max_length=50, description="Type of CT scan")
    file_path: str = Field(..., min_length=1, max_length=500, description="Scan file path")
    file_size: int = Field(..., ge=0, description="File size in bytes")
    voxel_size: Dict[str, float] = Field(..., description="Voxel dimensions (x, y, z)")
    scan_resolution: Dict[str, int] = Field(..., description="Scan resolution (x, y, z)")
    timestamp: datetime = Field(..., description="Scan timestamp")
    quality_score: Optional[float] = Field(None, ge=0, le=1, description="Scan quality score")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional scan metadata")
    
    # Graph metadata
    graph_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Graph node ID")
    source: str = Field(default="neo4j", description="Data source")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")

class PowderBedImageModel(BaseModel):
    """Powder bed image node model."""
    # Core fields
    image_id: str = Field(..., min_length=1, max_length=100, description="Unique image identifier")
    build_id: str = Field(..., min_length=1, max_length=100, description="Associated build ID")
    layer_number: int = Field(..., ge=0, description="Layer number in build")
    image_type: str = Field(..., min_length=1, max_length=50, description="Type of powder bed image")
    file_path: str = Field(..., min_length=1, max_length=500, description="Image file path")
    file_size: int = Field(..., ge=0, description="File size in bytes")
    dimensions: Dict[str, Union[int, float]] = Field(..., description="Image dimensions (width, height, resolution)")
    timestamp: datetime = Field(..., description="Image capture timestamp")
    powder_density: Optional[float] = Field(None, ge=0, le=1, description="Powder density in image")
    defects_detected: Optional[List[str]] = Field(None, description="Detected defects")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional image metadata")
    
    # Graph metadata
    graph_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Graph node ID")
    source: str = Field(default="neo4j", description="Data source")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")


# =============================================================================
# FILE NODE MODELS
# =============================================================================

class BuildFileModel(BaseModel):
    """Build file node model."""
    # Core fields
    file_id: str = Field(..., min_length=1, max_length=100, description="Unique file identifier")
    build_id: str = Field(..., min_length=1, max_length=100, description="Associated build ID")
    file_type: str = Field(..., min_length=1, max_length=50, description="Type of build file")
    file_path: str = Field(..., min_length=1, max_length=500, description="File path")
    file_size: int = Field(..., ge=0, description="File size in bytes")
    format: str = Field(..., min_length=1, max_length=20, description="File format")
    version: str = Field(..., min_length=1, max_length=20, description="File version")
    checksum: str = Field(..., min_length=32, max_length=64, description="File checksum")
    file_created_at: datetime = Field(..., description="File creation timestamp")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional file metadata")
    
    # Graph metadata
    graph_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Graph node ID")
    source: str = Field(default="neo4j", description="Data source")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")

class ModelFileModel(BaseModel):
    """3D model file node model."""
    # Core fields
    model_id: str = Field(..., min_length=1, max_length=100, description="Unique model identifier")
    part_id: str = Field(..., min_length=1, max_length=100, description="Associated part ID")
    file_type: str = Field(..., min_length=1, max_length=50, description="Type of model file")
    file_path: str = Field(..., min_length=1, max_length=500, description="File path")
    file_size: int = Field(..., ge=0, description="File size in bytes")
    format: str = Field(..., min_length=1, max_length=20, description="Model format (STL, OBJ, etc.)")
    version: str = Field(..., min_length=1, max_length=20, description="Model version")
    dimensions: Dict[str, float] = Field(..., description="Model dimensions (x, y, z)")
    volume: float = Field(..., ge=0, description="Model volume")
    surface_area: float = Field(..., ge=0, description="Model surface area")
    complexity_score: Optional[float] = Field(None, ge=0, description="Model complexity score")
    model_created_at: datetime = Field(..., description="Model creation timestamp")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional model metadata")
    
    # Graph metadata
    graph_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Graph node ID")
    source: str = Field(default="neo4j", description="Data source")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")

class LogFileModel(BaseModel):
    """Process log file node model."""
    # Core fields
    log_id: str = Field(..., min_length=1, max_length=100, description="Unique log identifier")
    process_id: str = Field(..., min_length=1, max_length=100, description="Associated process ID")
    log_type: str = Field(..., min_length=1, max_length=50, description="Type of log file")
    file_path: str = Field(..., min_length=1, max_length=500, description="Log file path")
    file_size: int = Field(..., ge=0, description="File size in bytes")
    format: str = Field(..., min_length=1, max_length=20, description="Log format")
    level: str = Field(..., min_length=1, max_length=20, description="Log level")
    entries_count: int = Field(..., ge=0, description="Number of log entries")
    start_time: datetime = Field(..., description="Log start timestamp")
    end_time: datetime = Field(..., description="Log end timestamp")
    duration: float = Field(..., ge=0, description="Log duration in seconds")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional log metadata")
    
    # Graph metadata
    graph_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Graph node ID")
    source: str = Field(default="neo4j", description="Data source")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")


# =============================================================================
# CACHE NODE MODELS
# =============================================================================

class ProcessCacheModel(BaseModel):
    """Process cache node model."""
    # Core fields
    cache_id: str = Field(..., min_length=1, max_length=100, description="Unique cache identifier")
    process_id: str = Field(..., min_length=1, max_length=100, description="Associated process ID")
    cache_type: str = Field(..., min_length=1, max_length=50, description="Type of cache")
    key: str = Field(..., min_length=1, max_length=200, description="Cache key")
    value: str = Field(..., min_length=1, max_length=10000, description="Cached value")
    size: int = Field(..., ge=0, description="Cache entry size in bytes")
    ttl: int = Field(..., ge=0, description="Time to live in seconds")
    created_at: datetime = Field(..., description="Cache creation timestamp")
    expires_at: datetime = Field(..., description="Cache expiration timestamp")
    access_count: int = Field(default=0, ge=0, description="Number of times accessed")
    last_accessed: Optional[datetime] = Field(None, description="Last access timestamp")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional cache metadata")
    
    # Graph metadata
    graph_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Graph node ID")
    source: str = Field(default="neo4j", description="Data source")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")

class AnalyticsCacheModel(BaseModel):
    """Analytics cache node model."""
    # Core fields
    cache_id: str = Field(..., min_length=1, max_length=100, description="Unique cache identifier")
    analysis_type: str = Field(..., min_length=1, max_length=50, description="Type of analysis")
    cache_key: str = Field(..., min_length=1, max_length=200, description="Cache key")
    result_data: str = Field(..., min_length=1, max_length=50000, description="Cached analysis result")
    size: int = Field(..., ge=0, description="Cache entry size in bytes")
    ttl: int = Field(..., ge=0, description="Time to live in seconds")
    created_at: datetime = Field(..., description="Cache creation timestamp")
    expires_at: datetime = Field(..., description="Cache expiration timestamp")
    computation_time: float = Field(..., ge=0, description="Original computation time in seconds")
    accuracy: Optional[float] = Field(None, ge=0, le=1, description="Analysis accuracy")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional cache metadata")
    
    # Graph metadata
    graph_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Graph node ID")
    source: str = Field(default="neo4j", description="Data source")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")


# =============================================================================
# QUEUE NODE MODELS
# =============================================================================

class JobQueueModel(BaseModel):
    """Job queue node model."""
    # Core fields
    job_id: str = Field(..., min_length=1, max_length=100, description="Unique job identifier")
    queue_name: str = Field(..., min_length=1, max_length=100, description="Queue name")
    job_type: str = Field(..., min_length=1, max_length=50, description="Type of job")
    priority: int = Field(..., ge=0, le=10, description="Job priority (0=highest, 10=lowest)")
    status: str = Field(..., min_length=1, max_length=20, description="Job status")
    payload: str = Field(..., min_length=1, max_length=10000, description="Job payload")
    created_at: datetime = Field(..., description="Job creation timestamp")
    started_at: Optional[datetime] = Field(None, description="Job start timestamp")
    completed_at: Optional[datetime] = Field(None, description="Job completion timestamp")
    retry_count: int = Field(default=0, ge=0, description="Number of retries")
    max_retries: int = Field(default=3, ge=0, description="Maximum retries allowed")
    timeout: int = Field(default=3600, ge=0, description="Job timeout in seconds")
    # Relationship fields for relationship extraction
    process_id: Optional[str] = Field(None, min_length=1, max_length=100, description="Associated process ID")
    machine_id: Optional[str] = Field(None, min_length=1, max_length=100, description="Associated machine ID")
    user_id: Optional[str] = Field(None, min_length=1, max_length=100, description="Associated user ID")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional job metadata")
    
    # Graph metadata
    graph_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Graph node ID")
    source: str = Field(default="neo4j", description="Data source")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")


# =============================================================================
# SESSION NODE MODELS
# =============================================================================

class UserSessionModel(BaseModel):
    """User session node model."""
    # Core fields
    session_id: str = Field(..., min_length=1, max_length=100, description="Unique session identifier")
    user_id: str = Field(..., min_length=1, max_length=100, description="Associated user ID")
    session_type: str = Field(..., min_length=1, max_length=50, description="Type of session")
    status: str = Field(..., min_length=1, max_length=20, description="Session status")
    created_at: datetime = Field(..., description="Session creation timestamp")
    last_activity: datetime = Field(..., description="Last activity timestamp")
    expires_at: datetime = Field(..., description="Session expiration timestamp")
    ip_address: Optional[str] = Field(None, min_length=7, max_length=45, description="Client IP address")
    user_agent: Optional[str] = Field(None, max_length=500, description="Client user agent")
    permissions: Optional[List[str]] = Field(None, description="Session permissions")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional session metadata")
    
    # Graph metadata
    graph_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Graph node ID")
    source: str = Field(default="neo4j", description="Data source")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")


# =============================================================================
# READING NODE MODELS
# =============================================================================

class SensorReadingModel(BaseModel):
    """Sensor reading node model."""
    # Core fields
    reading_id: str = Field(..., min_length=1, max_length=100, description="Unique reading identifier")
    sensor_id: str = Field(..., min_length=1, max_length=100, description="Associated sensor ID")
    reading_type: str = Field(..., min_length=1, max_length=50, description="Type of reading")
    value: float = Field(..., description="Reading value")
    unit: str = Field(..., min_length=1, max_length=20, description="Reading unit")
    timestamp: datetime = Field(..., description="Reading timestamp")
    quality: Optional[float] = Field(None, ge=0, le=1, description="Reading quality score")
    status: str = Field(..., min_length=1, max_length=20, description="Reading status")
    location: Optional[Dict[str, float]] = Field(None, description="Reading location coordinates")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional reading metadata")
    
    # Graph metadata
    graph_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Graph node ID")
    source: str = Field(default="neo4j", description="Data source")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")


# =============================================================================
# EVENT NODE MODELS
# =============================================================================

class ProcessMonitoringModel(BaseModel):
    """Process monitoring event node model."""
    # Core fields
    event_id: str = Field(..., min_length=1, max_length=100, description="Unique event identifier")
    process_id: str = Field(..., min_length=1, max_length=100, description="Associated process ID")
    event_type: str = Field(..., min_length=1, max_length=50, description="Type of monitoring event")
    severity: str = Field(..., min_length=1, max_length=20, description="Event severity")
    message: str = Field(..., min_length=1, max_length=1000, description="Event message")
    timestamp: datetime = Field(..., description="Event timestamp")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Event parameters")
    status: str = Field(..., min_length=1, max_length=20, description="Event status")
    resolved: bool = Field(default=False, description="Whether event is resolved")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional event metadata")
    
    # Graph metadata
    graph_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Graph node ID")
    source: str = Field(default="neo4j", description="Data source")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")

class MachineStatusModel(BaseModel):
    """Machine status update node model."""
    # Core fields
    status_id: str = Field(..., min_length=1, max_length=100, description="Unique status identifier")
    machine_id: str = Field(..., min_length=1, max_length=100, description="Associated machine ID")
    status_type: str = Field(..., min_length=1, max_length=50, description="Type of status update")
    status_value: str = Field(..., min_length=1, max_length=50, description="Status value")
    timestamp: datetime = Field(..., description="Status timestamp")
    duration: Optional[float] = Field(None, ge=0, description="Status duration in seconds")
    reason: Optional[str] = Field(None, max_length=500, description="Status change reason")
    operator_id: Optional[str] = Field(None, min_length=1, max_length=100, description="Operator who changed status")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional status metadata")
    
    # Graph metadata
    graph_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Graph node ID")
    source: str = Field(default="neo4j", description="Data source")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")

class AlertEventModel(BaseModel):
    """Alert event node model."""
    # Core fields
    alert_id: str = Field(..., min_length=1, max_length=100, description="Unique alert identifier")
    source_id: str = Field(..., min_length=1, max_length=100, description="Source entity ID")
    alert_type: str = Field(..., min_length=1, max_length=50, description="Type of alert")
    severity: str = Field(..., min_length=1, max_length=20, description="Alert severity")
    message: str = Field(..., min_length=1, max_length=1000, description="Alert message")
    timestamp: datetime = Field(..., description="Alert timestamp")
    status: str = Field(..., min_length=1, max_length=20, description="Alert status")
    acknowledged: bool = Field(default=False, description="Whether alert is acknowledged")
    acknowledged_by: Optional[str] = Field(None, min_length=1, max_length=100, description="Who acknowledged the alert")
    acknowledged_at: Optional[datetime] = Field(None, description="Acknowledgment timestamp")
    resolved: bool = Field(default=False, description="Whether alert is resolved")
    resolved_by: Optional[str] = Field(None, min_length=1, max_length=100, description="Who resolved the alert")
    resolved_at: Optional[datetime] = Field(None, description="Resolution timestamp")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional alert metadata")
    
    # Graph metadata
    graph_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Graph node ID")
    source: str = Field(default="neo4j", description="Data source")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")


class BatchModel(BaseModel):
    """Material batch node model."""
    # Core fields
    batch_id: str = Field(..., min_length=1, max_length=100, description="Unique batch identifier")
    material_id: str = Field(..., min_length=1, max_length=100, description="Associated material ID")
    batch_number: str = Field(..., min_length=1, max_length=50, description="Batch number")
    supplier: str = Field(..., min_length=1, max_length=100, description="Material supplier")
    condition: str = Field(..., min_length=1, max_length=50, description="Batch condition")
    quantity: float = Field(..., ge=0, description="Batch quantity")
    unit: str = Field(..., min_length=1, max_length=20, description="Quantity unit")
    expiry_date: Optional[datetime] = Field(None, description="Batch expiry date")
    storage_location: Optional[str] = Field(None, min_length=1, max_length=100, description="Storage location")
    quality_certificate: Optional[str] = Field(None, min_length=1, max_length=200, description="Quality certificate")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional batch metadata")
    
    # Graph metadata
    graph_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Graph node ID")
    source: str = Field(default="neo4j", description="Data source")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")


class MeasurementModel(BaseModel):
    """Sensor measurement node model."""
    # Core fields
    measurement_id: str = Field(..., min_length=1, max_length=100, description="Unique measurement identifier")
    sensor_id: str = Field(..., min_length=1, max_length=100, description="Associated sensor ID")
    measurement_type: str = Field(..., min_length=1, max_length=50, description="Type of measurement")
    value: float = Field(..., description="Measurement value")
    unit: str = Field(..., min_length=1, max_length=20, description="Measurement unit")
    timestamp: datetime = Field(..., description="Measurement timestamp")
    quality: Optional[str] = Field(None, min_length=1, max_length=20, description="Measurement quality")
    status: str = Field(..., min_length=1, max_length=20, description="Measurement status")
    location: Optional[Dict[str, float]] = Field(None, description="Measurement location coordinates")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional measurement metadata")
    
    # Graph metadata
    graph_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Graph node ID")
    source: str = Field(default="neo4j", description="Data source")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")


class MachineConfigModel(BaseModel):
    """Machine configuration node model."""
    # Core fields
    config_id: str = Field(..., min_length=1, max_length=100, description="Unique configuration identifier")
    machine_id: str = Field(..., min_length=1, max_length=100, description="Associated machine ID")
    process_id: Optional[str] = Field(None, min_length=1, max_length=100, description="Associated process ID")
    build_id: Optional[str] = Field(None, min_length=1, max_length=100, description="Associated build ID")
    config_type: str = Field(..., min_length=1, max_length=50, description="Type of configuration")
    config_data: Dict[str, Any] = Field(..., description="Configuration parameters and settings")
    file_size: Optional[int] = Field(None, ge=0, description="Configuration file size in bytes")
    config_created_at: datetime = Field(..., description="Configuration creation timestamp")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional configuration metadata")
    
    # Graph metadata
    graph_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Graph node ID")
    source: str = Field(default="neo4j", description="Data source")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")


class SensorTypeModel(BaseModel):
    """Sensor type node model."""
    # Core fields
    sensor_type_id: str = Field(..., min_length=1, max_length=100, description="Unique sensor type identifier")
    sensor_type: str = Field(..., min_length=1, max_length=50, description="Type of sensor")
    description: str = Field(..., min_length=1, max_length=500, description="Sensor type description")
    unit: str = Field(..., min_length=1, max_length=20, description="Measurement unit")
    range_min: Optional[float] = Field(None, description="Minimum measurement range")
    range_max: Optional[float] = Field(None, description="Maximum measurement range")
    accuracy: Optional[float] = Field(None, ge=0, le=1, description="Sensor accuracy")
    sampling_rate: Optional[float] = Field(None, ge=0, description="Sampling rate in Hz")
    calibration_required: bool = Field(default=True, description="Whether calibration is required")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional sensor type metadata")
    
    # Graph metadata
    graph_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Graph node ID")
    source: str = Field(default="neo4j", description="Data source")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")
