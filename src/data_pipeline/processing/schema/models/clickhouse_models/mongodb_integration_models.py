"""
ClickHouse MongoDB Integration Models

This module contains Pydantic models for MongoDB data integrated into ClickHouse data warehouse.
"""

from typing import Optional, List, Dict, Any, Union
from datetime import datetime, date
from enum import Enum
from pydantic import BaseModel, Field, validator


# ============================================================================
# ENUMS
# ============================================================================

class LogLevel(str, Enum):
    """Log level enumeration."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class EventType(str, Enum):
    """Event type enumeration."""
    PROCESS_START = "process_start"
    PROCESS_END = "process_end"
    LAYER_COMPLETE = "layer_complete"
    ERROR_OCCURRED = "error_occurred"
    MAINTENANCE = "maintenance"
    CALIBRATION = "calibration"
    QUALITY_CHECK = "quality_check"

class AnnotationType(str, Enum):
    """Annotation type enumeration."""
    COMMENT = "comment"
    WARNING = "warning"
    ERROR = "error"
    SUCCESS = "success"
    INFO = "info"

class ConfigurationType(str, Enum):
    """Configuration type enumeration."""
    MACHINE = "machine"
    PROCESS = "process"
    LASER = "laser"
    PLATFORM = "platform"
    POWDER = "powder"
    ENVIRONMENT = "environment"

class ConfigurationFormat(str, Enum):
    """Configuration format enumeration."""
    JSON = "json"
    XML = "xml"
    YAML = "yaml"
    INI = "ini"
    BINARY = "binary"

class DataFormat(str, Enum):
    """Data format enumeration."""
    CSV = "csv"
    JSON = "json"
    BINARY = "binary"
    HDF5 = "hdf5"
    PARQUET = "parquet"

class ProcessingStatus(str, Enum):
    """Processing status enumeration."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class DataQuality(str, Enum):
    """Data quality enumeration."""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    UNKNOWN = "unknown"

class ModelType(str, Enum):
    """Model type enumeration."""
    CAD = "cad"
    STL = "stl"
    OBJ = "obj"
    PLY = "ply"
    STEP = "step"
    IGES = "iges"

class ModelFormat(str, Enum):
    """Model format enumeration."""
    ASCII = "ascii"
    BINARY = "binary"
    COMPRESSED = "compressed"

class InstructionType(str, Enum):
    """Instruction type enumeration."""
    LAYER = "layer"
    CONTOUR = "contour"
    HATCH = "hatch"
    SUPPORT = "support"
    CALIBRATION = "calibration"

class MaterialType(str, Enum):
    """Material type enumeration."""
    METAL = "metal"
    POLYMER = "polymer"
    CERAMIC = "ceramic"
    COMPOSITE = "composite"

class SupportType(str, Enum):
    """Support type enumeration."""
    TREE = "tree"
    LATTICE = "lattice"
    SOLID = "solid"
    MINIMAL = "minimal"

class ScanType(str, Enum):
    """Scan type enumeration."""
    QUALITY = "quality"
    DEFECT = "defect"
    DIMENSIONAL = "dimensional"
    MATERIAL = "material"

class ImageFormat(str, Enum):
    """Image format enumeration."""
    PNG = "png"
    JPEG = "jpeg"
    TIFF = "tiff"
    BMP = "bmp"
    RAW = "raw"

class DefectType(str, Enum):
    """Defect type enumeration."""
    POROSITY = "porosity"
    CRACK = "crack"
    WARPAGE = "warpage"
    SURFACE_ROUGHNESS = "surface_roughness"
    DIMENSIONAL_ERROR = "dimensional_error"

class DefectSeverity(str, Enum):
    """Defect severity enumeration."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class PowderBedType(str, Enum):
    """Powder bed type enumeration."""
    FRESH = "fresh"
    RECYCLED = "recycled"
    MIXED = "mixed"
    CONTAMINATED = "contaminated"

class PowderBedDefectType(str, Enum):
    """Powder bed defect type enumeration."""
    INHOMOGENEITY = "inhomogeneity"
    CONTAMINATION = "contamination"
    MOISTURE = "moisture"
    PARTICLE_SIZE = "particle_size"

class PowderBedDefectSeverity(str, Enum):
    """Powder bed defect severity enumeration."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ProcessStage(str, Enum):
    """Process stage enumeration."""
    PREPARATION = "preparation"
    LAYER_DEPOSITION = "layer_deposition"
    LASER_MELTING = "laser_melting"
    COOLING = "cooling"
    POST_PROCESSING = "post_processing"

class ProcessDefectType(str, Enum):
    """Process defect type enumeration."""
    LAYER_SHIFT = "layer_shift"
    INCOMPLETE_MELTING = "incomplete_melting"
    OVERHEATING = "overheating"
    UNDERHEATING = "underheating"

class ProcessDefectSeverity(str, Enum):
    """Process defect severity enumeration."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class FileType(str, Enum):
    """File type enumeration."""
    CONFIGURATION = "configuration"
    PROGRAM = "program"
    CALIBRATION = "calibration"
    MAINTENANCE = "maintenance"
    LOG = "log"

class FileCategory(str, Enum):
    """File category enumeration."""
    SYSTEM = "system"
    USER = "user"
    AUTOMATED = "automated"
    MANUAL = "manual"


# ============================================================================
# COMPONENT MODELS
# ============================================================================

class LogMetadata(BaseModel):
    """Log metadata model."""
    source_module: str = Field(..., description="Source module")
    function_name: Optional[str] = Field(None, description="Function name")
    line_number: Optional[int] = Field(None, description="Line number")
    thread_id: Optional[str] = Field(None, description="Thread ID")
    process_id: Optional[str] = Field(None, description="Process ID")

class AnnotationInfo(BaseModel):
    """Annotation information model."""
    annotation_type: AnnotationType = Field(..., description="Annotation type")
    annotation_text: str = Field(..., description="Annotation text")
    confidence: Optional[float] = Field(None, ge=0, le=1, description="Annotation confidence")
    tags: List[str] = Field(default_factory=list, description="Annotation tags")

class RelatedDocument(BaseModel):
    """Related document model."""
    document_id: str = Field(..., description="Related document ID")
    document_type: str = Field(..., description="Related document type")
    relationship_type: str = Field(..., description="Relationship type")

class LogRelationship(BaseModel):
    """Log relationship model."""
    related_logs: List[str] = Field(default_factory=list, description="Related log IDs")
    parent_log: Optional[str] = Field(None, description="Parent log ID")
    child_logs: List[str] = Field(default_factory=list, description="Child log IDs")

class CalibrationData(BaseModel):
    """Calibration data model."""
    calibration_type: str = Field(..., description="Calibration type")
    calibration_date: datetime = Field(..., description="Calibration date")
    calibration_parameters: Dict[str, Any] = Field(..., description="Calibration parameters")
    accuracy: float = Field(..., ge=0, le=100, description="Calibration accuracy")
    technician: Optional[str] = Field(None, description="Calibration technician")

class CalibrationInfo(BaseModel):
    """Calibration information model."""
    last_calibrated: Optional[datetime] = Field(None, description="Last calibration date")
    calibration_accuracy: float = Field(..., ge=0, le=100, description="Calibration accuracy")
    calibration_uncertainty: float = Field(..., ge=0, description="Calibration uncertainty")
    calibration_method: str = Field(..., description="Calibration method")

class ConfigurationMetadata(BaseModel):
    """Configuration metadata model."""
    version: str = Field(..., description="Configuration version")
    created_by: str = Field(..., description="Created by")
    created_at: datetime = Field(..., description="Creation timestamp")
    modified_by: Optional[str] = Field(None, description="Modified by")
    modified_at: Optional[datetime] = Field(None, description="Modification timestamp")

class MeasurementRange(BaseModel):
    """Measurement range model."""
    min_value: float = Field(..., description="Minimum value")
    max_value: float = Field(..., description="Maximum value")
    unit: str = Field(..., description="Measurement unit")
    resolution: float = Field(..., description="Measurement resolution")

class GeometryInfo(BaseModel):
    """Geometry information model."""
    vertex_count: int = Field(..., ge=0, description="Vertex count")
    face_count: int = Field(..., ge=0, description="Face count")
    edge_count: int = Field(..., ge=0, description="Edge count")
    bounding_box: Dict[str, float] = Field(..., description="Bounding box coordinates")
    volume: float = Field(..., ge=0, description="Model volume")
    surface_area: float = Field(..., ge=0, description="Model surface area")

class QualityAssessment(BaseModel):
    """Quality assessment model."""
    model_quality_score: float = Field(..., ge=0, le=100, description="Model quality score")
    mesh_quality: str = Field(..., description="Mesh quality")
    watertight: bool = Field(..., description="Watertight status")
    manifold: bool = Field(..., description="Manifold status")
    self_intersecting: bool = Field(..., description="Self-intersecting status")

class ProcessingInfo(BaseModel):
    """Processing information model."""
    processing_status: ProcessingStatus = Field(..., description="Processing status")
    processing_algorithm: str = Field(..., description="Processing algorithm")
    processing_parameters: Dict[str, Any] = Field(..., description="Processing parameters")
    processing_duration: float = Field(..., ge=0, description="Processing duration in seconds")

class ModelMetadata(BaseModel):
    """Model metadata model."""
    model_version: str = Field(..., description="Model version")
    model_units: str = Field(..., description="Model units")
    model_scale: float = Field(..., ge=0, description="Model scale")
    model_rotation: Dict[str, float] = Field(..., description="Model rotation")
    model_translation: Dict[str, float] = Field(..., description="Model translation")

class InstructionMetadata(BaseModel):
    """Instruction metadata model."""
    instruction_version: str = Field(..., description="Instruction version")
    instruction_priority: int = Field(..., ge=0, description="Instruction priority")
    instruction_status: str = Field(..., description="Instruction status")
    created_by: str = Field(..., description="Created by")
    created_at: datetime = Field(..., description="Creation timestamp")

class CTQualityMetrics(BaseModel):
    """CT scan quality metrics model."""
    image_quality_score: float = Field(..., ge=0, le=100, description="Image quality score")
    noise_level: float = Field(..., ge=0, le=1, description="Noise level")
    contrast_ratio: float = Field(..., ge=0, description="Contrast ratio")
    sharpness_score: float = Field(..., ge=0, le=1, description="Sharpness score")
    artifact_count: int = Field(..., ge=0, description="Artifact count")

class DefectAnalysis(BaseModel):
    """Defect analysis model."""
    defect_count: int = Field(..., ge=0, description="Defect count")
    defect_types: List[DefectType] = Field(..., description="Defect types")
    defect_severity: DefectSeverity = Field(..., description="Defect severity")
    defect_locations: List[str] = Field(..., description="Defect locations")
    defect_volumes: List[float] = Field(..., description="Defect volumes")

class DimensionalAnalysis(BaseModel):
    """Dimensional analysis model."""
    dimensional_accuracy: float = Field(..., ge=0, le=100, description="Dimensional accuracy")
    measurement_uncertainty: float = Field(..., ge=0, description="Measurement uncertainty")
    measurement_method: str = Field(..., description="Measurement method")
    measurement_equipment: str = Field(..., description="Measurement equipment")

class PowderAnalysis(BaseModel):
    """Powder analysis model."""
    powder_coverage: float = Field(..., ge=0, le=100, description="Powder coverage")
    powder_distribution: str = Field(..., description="Powder distribution")
    powder_particle_size: float = Field(..., ge=0, description="Particle size in micrometers")
    powder_flowability: float = Field(..., ge=0, le=100, description="Powder flowability")

class PowderCharacteristics(BaseModel):
    """Powder characteristics model."""
    particle_size_distribution: Dict[str, float] = Field(..., description="Particle size distribution")
    shape_factor: float = Field(..., ge=0, le=1, description="Shape factor")
    density: float = Field(..., ge=0, description="Powder density")
    flowability: float = Field(..., ge=0, le=100, description="Flowability percentage")

class BedQualityMetrics(BaseModel):
    """Bed quality metrics model."""
    bed_quality_score: float = Field(..., ge=0, le=100, description="Bed quality score")
    uniformity: float = Field(..., ge=0, le=100, description="Bed uniformity")
    density: float = Field(..., ge=0, le=100, description="Bed density")
    surface_roughness: float = Field(..., ge=0, description="Surface roughness")

class ImageAnalysis(BaseModel):
    """Image analysis model."""
    brightness: float = Field(..., ge=0, le=255, description="Image brightness")
    contrast: float = Field(..., ge=0, le=1, description="Image contrast")
    sharpness: float = Field(..., ge=0, le=1, description="Image sharpness")
    color_balance: Dict[str, float] = Field(..., description="Color balance")

class DefectDetection(BaseModel):
    """Defect detection model."""
    defect_count: int = Field(..., ge=0, description="Defect count")
    defect_types: List[PowderBedDefectType] = Field(..., description="Defect types")
    defect_severity: PowderBedDefectSeverity = Field(..., description="Defect severity")
    defect_locations: List[str] = Field(..., description="Defect locations")
    defect_areas: List[float] = Field(..., description="Defect areas")

class PowderBedDefect(BaseModel):
    """Powder bed defect model."""
    defect_type: PowderBedDefectType = Field(..., description="Defect type")
    severity: PowderBedDefectSeverity = Field(..., description="Defect severity")
    location: str = Field(..., description="Defect location")
    area: float = Field(..., ge=0, description="Defect area")
    description: str = Field(..., description="Defect description")

class DefectLocation(BaseModel):
    """Defect location model."""
    x_coordinate: float = Field(..., description="X coordinate")
    y_coordinate: float = Field(..., description="Y coordinate")
    z_coordinate: Optional[float] = Field(None, description="Z coordinate")
    region: str = Field(..., description="Defect region")

class ProcessAnalysis(BaseModel):
    """Process analysis model."""
    process_quality_score: float = Field(..., ge=0, le=100, description="Process quality score")
    layer_completeness: float = Field(..., ge=0, le=100, description="Layer completeness")
    dimensional_accuracy: float = Field(..., ge=0, le=100, description="Dimensional accuracy")
    surface_quality: str = Field(..., description="Surface quality")

class ProcessQualityMetrics(BaseModel):
    """Process quality metrics model."""
    overall_quality: float = Field(..., ge=0, le=100, description="Overall quality")
    dimensional_accuracy: float = Field(..., ge=0, le=100, description="Dimensional accuracy")
    surface_finish: float = Field(..., ge=0, le=100, description="Surface finish")
    material_density: float = Field(..., ge=0, le=100, description="Material density")

class LayerAnalysis(BaseModel):
    """Layer analysis model."""
    layer_number: int = Field(..., ge=0, description="Layer number")
    layer_thickness: float = Field(..., ge=0, description="Layer thickness")
    layer_quality: float = Field(..., ge=0, le=100, description="Layer quality")
    layer_completeness: float = Field(..., ge=0, le=100, description="Layer completeness")

class ProcessDefect(BaseModel):
    """Process defect model."""
    defect_type: ProcessDefectType = Field(..., description="Defect type")
    severity: ProcessDefectSeverity = Field(..., description="Defect severity")
    location: str = Field(..., description="Defect location")
    description: str = Field(..., description="Defect description")

class FileMetadata(BaseModel):
    """File metadata model."""
    file_size: int = Field(..., ge=0, description="File size in bytes")
    file_hash: str = Field(..., description="File hash")
    file_format: str = Field(..., description="File format")
    file_status: str = Field(..., description="File status")

class UsageAnalytics(BaseModel):
    """Usage analytics model."""
    usage_count: int = Field(..., ge=0, description="Usage count")
    last_accessed: Optional[datetime] = Field(None, description="Last accessed timestamp")
    access_frequency: float = Field(..., ge=0, description="Access frequency")
    user_rating: Optional[float] = Field(None, ge=0, le=5, description="User rating")

class FileQualityMetrics(BaseModel):
    """File quality metrics model."""
    file_quality_score: float = Field(..., ge=0, le=100, description="File quality score")
    file_integrity_score: float = Field(..., ge=0, le=100, description="File integrity score")
    file_completeness: float = Field(..., ge=0, le=100, description="File completeness")
    file_validation_status: str = Field(..., description="File validation status")


# ============================================================================
# MAIN MODELS
# ============================================================================

class ProcessLogModel(BaseModel):
    """Process Log model for ClickHouse - matches process_logs.sql schema with optional fields."""
    # Primary identifiers
    id: int = Field(..., description="Primary key ID")
    log_id: str = Field(..., description="Log ID")
    process_id: Optional[str] = Field(None, description="Process ID")
    build_id: Optional[str] = Field(None, description="Build ID")
    part_id: Optional[str] = Field(None, description="Part ID")
    machine_id: Optional[str] = Field(None, description="Machine ID")
    timestamp: datetime = Field(..., description="Log timestamp")
    
    # Log information (flattened) - optional
    log_level: Optional[str] = Field(None, description="Log level")
    log_message: Optional[str] = Field(None, description="Log message")
    source_module: Optional[str] = Field(None, description="Source module")
    event_type: Optional[str] = Field(None, description="Event type")
    user_id: Optional[str] = Field(None, description="User ID")
    
    # Annotation data (flattened) - optional
    annotation_type: Optional[str] = Field(None, description="Annotation type")
    annotation_text: Optional[str] = Field(None, description="Annotation text")
    related_documents: Optional[List[str]] = Field(default_factory=list, description="Related documents")
    
    # Session metadata (flattened) - optional
    session_id: Optional[str] = Field(None, description="Session ID")
    ip_address: Optional[str] = Field(None, description="IP address")
    user_agent: Optional[str] = Field(None, description="User agent")
    
    # Data quality (flattened) - optional
    data_quality: Optional[str] = Field(None, description="Data quality")
    validation_status: Optional[str] = Field(None, description="Validation status")
    generated_by: Optional[str] = Field(None, description="Generated by")
    generation_timestamp: Optional[datetime] = Field(None, description="Generation timestamp")
    
    # Tags and relationships (flattened) - optional
    tags: Optional[List[str]] = Field(default_factory=list, description="Tags")
    related_processes: Optional[List[str]] = Field(default_factory=list, description="Related processes")
    related_builds: Optional[List[str]] = Field(default_factory=list, description="Related builds")
    related_parts: Optional[List[str]] = Field(default_factory=list, description="Related parts")
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class MachineConfigurationModel(BaseModel):
    """Machine Configuration model for ClickHouse - matches machine_configurations.sql schema with optional fields."""
    # Primary identifiers
    id: int = Field(..., description="Primary key ID")
    config_id: str = Field(..., description="Configuration ID")
    process_id: Optional[str] = Field(None, description="Process ID")
    build_id: Optional[str] = Field(None, description="Build ID")
    machine_id: Optional[str] = Field(None, description="Machine ID")
    timestamp: datetime = Field(..., description="Configuration timestamp")
    
    # Configuration metadata (flattened)
    config_name: Optional[str] = Field(None, description="Configuration name")
    config_type: Optional[str] = Field(None, description="Configuration type")
    config_version: Optional[str] = Field(None, description="Configuration version")
    config_format: Optional[str] = Field(None, description="Configuration format")
    
    # File information (flattened)
    config_data_path: Optional[str] = Field(None, description="Configuration data path")
    config_file_size: Optional[int] = Field(None, ge=0, description="Configuration file size")
    config_data_hash: Optional[str] = Field(None, description="Configuration data hash")
    
    # Laser settings (flattened)
    laser_power: Optional[int] = Field(None, ge=0, description="Laser power in watts")
    laser_speed: Optional[int] = Field(None, ge=0, description="Laser speed in mm/s")
    laser_frequency: Optional[int] = Field(None, ge=0, description="Laser frequency in Hz")
    laser_power_calibrated: Optional[float] = Field(None, ge=0, description="Calibrated laser power")
    laser_calibration_date: Optional[datetime] = Field(None, description="Laser calibration date")
    laser_calibration_accuracy: Optional[float] = Field(None, ge=0, le=1, description="Laser calibration accuracy")
    
    # Temperature settings (flattened)
    bed_temperature: Optional[float] = Field(None, description="Bed temperature in Celsius")
    chamber_temperature: Optional[float] = Field(None, description="Chamber temperature in Celsius")
    max_temperature: Optional[float] = Field(None, ge=0, description="Maximum temperature in Celsius")
    max_pressure: Optional[float] = Field(None, ge=0, description="Maximum pressure in bar")
    
    # Powder settings (flattened)
    layer_thickness: Optional[float] = Field(None, ge=0, description="Layer thickness in mm")
    powder_density: Optional[float] = Field(None, ge=0, description="Powder density in g/cm³")
    
    # Safety and limits (flattened)
    safety_limits_max_temp: Optional[float] = Field(None, ge=0, description="Safety limit max temperature")
    safety_limits_max_pressure: Optional[float] = Field(None, ge=0, description="Safety limit max pressure")
    
    # Calibration data (flattened)
    calibration_date: Optional[datetime] = Field(None, description="Calibration date")
    calibration_accuracy: Optional[float] = Field(None, ge=0, le=1, description="Calibration accuracy")
    calibration_uncertainty: Optional[float] = Field(None, ge=0, description="Calibration uncertainty")
    calibration_factor: Optional[float] = Field(None, ge=0, description="Calibration factor")
    
    # Configuration status (flattened)
    config_status: Optional[str] = Field(None, description="Configuration status")
    is_active: Optional[int] = Field(None, ge=0, le=1, description="Is active flag (0/1)")
    config_priority: Optional[int] = Field(None, ge=0, description="Configuration priority")
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class RawSensorDataModel(BaseModel):
    """Raw Sensor Data model for ClickHouse - matches raw_sensor_data.sql schema with optional fields."""
    # Primary identifiers
    id: int = Field(..., description="Primary key ID")
    sensor_id: str = Field(..., description="Sensor ID")
    process_id: Optional[str] = Field(None, description="Process ID")
    build_id: Optional[str] = Field(None, description="Build ID")
    machine_id: Optional[str] = Field(None, description="Machine ID")
    timestamp: datetime = Field(..., description="Data timestamp")
    
    # File information (flattened)
    file_path: Optional[str] = Field(None, description="File path")
    file_size: Optional[int] = Field(None, ge=0, description="File size in bytes")
    file_hash: Optional[str] = Field(None, description="File hash")
    file_status: Optional[str] = Field(None, description="File status")
    
    # Sensor metadata (flattened)
    sensor_type: Optional[str] = Field(None, description="Sensor type")
    data_format: Optional[str] = Field(None, description="Data format")
    sampling_rate: Optional[int] = Field(None, ge=0, description="Sampling rate in Hz")
    data_duration: Optional[float] = Field(None, ge=0, description="Data duration in seconds")
    data_points: Optional[int] = Field(None, ge=0, description="Number of data points")
    processing_status: Optional[str] = Field(None, description="Processing status")
    data_quality: Optional[str] = Field(None, description="Data quality")
    
    # Calibration information (flattened)
    last_calibrated: Optional[date] = Field(None, description="Last calibration date")
    calibration_factor: Optional[float] = Field(None, ge=0, description="Calibration factor")
    calibration_accuracy: Optional[float] = Field(None, ge=0, le=1, description="Calibration accuracy")
    calibration_uncertainty: Optional[float] = Field(None, ge=0, description="Calibration uncertainty")
    
    # Measurement range (flattened)
    min_value: Optional[float] = Field(None, description="Minimum measurement value")
    max_value: Optional[float] = Field(None, description="Maximum measurement value")
    unit: Optional[str] = Field(None, description="Measurement unit")
    
    # Processing metadata (flattened)
    processing_algorithm: Optional[str] = Field(None, description="Processing algorithm")
    processing_parameters: Optional[str] = Field(None, description="Processing parameters")
    processing_version: Optional[str] = Field(None, description="Processing version")
    processing_timestamp: Optional[datetime] = Field(None, description="Processing timestamp")
    
    # Quality metrics (flattened)
    signal_to_noise_ratio: Optional[float] = Field(None, ge=0, description="Signal to noise ratio")
    data_completeness: Optional[float] = Field(None, ge=0, le=1, description="Data completeness percentage")
    outlier_percentage: Optional[float] = Field(None, ge=0, le=1, description="Outlier percentage")
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            date: lambda v: v.isoformat()
        }

class ModelFileModel(BaseModel):
    """3D Model File model for ClickHouse - matches 3d_model_files.sql schema with optional fields."""
    # Primary identifiers
    id: int = Field(..., description="Primary key ID")
    model_id: str = Field(..., description="Model ID")
    process_id: Optional[str] = Field(None, description="Process ID")
    build_id: Optional[str] = Field(None, description="Build ID")
    part_id: Optional[str] = Field(None, description="Part ID")
    machine_id: Optional[str] = Field(None, description="Machine ID")
    timestamp: datetime = Field(..., description="Model timestamp")
    
    # File information (flattened)
    file_path: Optional[str] = Field(None, description="File path")
    file_name: Optional[str] = Field(None, description="File name")
    file_size: Optional[int] = Field(None, ge=0, description="File size in bytes")
    file_hash: Optional[str] = Field(None, description="File hash")
    file_format: Optional[str] = Field(None, description="File format")
    file_status: Optional[str] = Field(None, description="File status")
    
    # 3D model metadata (flattened)
    model_type: Optional[str] = Field(None, description="Model type")
    model_version: Optional[str] = Field(None, description="Model version")
    model_units: Optional[str] = Field(None, description="Model units")
    model_scale: Optional[float] = Field(None, ge=0, description="Model scale")
    model_rotation_x: Optional[float] = Field(None, description="Model rotation X")
    model_rotation_y: Optional[float] = Field(None, description="Model rotation Y")
    model_rotation_z: Optional[float] = Field(None, description="Model rotation Z")
    model_translation_x: Optional[float] = Field(None, description="Model translation X")
    model_translation_y: Optional[float] = Field(None, description="Model translation Y")
    model_translation_z: Optional[float] = Field(None, description="Model translation Z")
    
    # Geometry information (flattened)
    vertex_count: Optional[int] = Field(None, ge=0, description="Vertex count")
    face_count: Optional[int] = Field(None, ge=0, description="Face count")
    edge_count: Optional[int] = Field(None, ge=0, description="Edge count")
    bounding_box_min_x: Optional[float] = Field(None, description="Bounding box min X")
    bounding_box_min_y: Optional[float] = Field(None, description="Bounding box min Y")
    bounding_box_min_z: Optional[float] = Field(None, description="Bounding box min Z")
    bounding_box_max_x: Optional[float] = Field(None, description="Bounding box max X")
    bounding_box_max_y: Optional[float] = Field(None, description="Bounding box max Y")
    bounding_box_max_z: Optional[float] = Field(None, description="Bounding box max Z")
    volume: Optional[float] = Field(None, ge=0, description="Volume")
    surface_area: Optional[float] = Field(None, ge=0, description="Surface area")
    
    # Quality metrics (flattened)
    model_quality_score: Optional[float] = Field(None, ge=0, le=1, description="Model quality score")
    mesh_quality: Optional[str] = Field(None, description="Mesh quality")
    watertight: Optional[int] = Field(None, ge=0, le=1, description="Watertight flag (0/1)")
    manifold: Optional[int] = Field(None, ge=0, le=1, description="Manifold flag (0/1)")
    self_intersecting: Optional[int] = Field(None, ge=0, le=1, description="Self-intersecting flag (0/1)")
    
    # Processing information (flattened)
    processing_status: Optional[str] = Field(None, description="Processing status")
    processing_algorithm: Optional[str] = Field(None, description="Processing algorithm")
    processing_parameters: Optional[str] = Field(None, description="Processing parameters")
    processing_duration: Optional[float] = Field(None, ge=0, description="Processing duration in seconds")
    processing_timestamp: Optional[datetime] = Field(None, description="Processing timestamp")
    
    # User and session data (flattened)
    user_id: Optional[str] = Field(None, description="User ID")
    session_id: Optional[str] = Field(None, description="Session ID")
    upload_source: Optional[str] = Field(None, description="Upload source")
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class BuildInstructionModel(BaseModel):
    """Build Instruction model for ClickHouse - matches build_instructions.sql schema with optional fields."""
    # Primary identifiers
    id: int = Field(..., description="Primary key ID")
    instruction_id: str = Field(..., description="Instruction ID")
    build_id: Optional[str] = Field(None, description="Build ID")
    process_id: Optional[str] = Field(None, description="Process ID")
    part_id: Optional[str] = Field(None, description="Part ID")
    machine_id: Optional[str] = Field(None, description="Machine ID")
    timestamp: datetime = Field(..., description="Instruction timestamp")
    
    # Instruction metadata (flattened)
    instruction_type: Optional[str] = Field(None, description="Instruction type")
    instruction_content: Optional[str] = Field(None, description="Instruction content")
    instruction_version: Optional[str] = Field(None, description="Instruction version")
    instruction_status: Optional[str] = Field(None, description="Instruction status")
    instruction_priority: Optional[int] = Field(None, ge=0, description="Instruction priority")
    
    # Layer information (flattened)
    layer_number: Optional[int] = Field(None, ge=0, description="Layer number")
    layer_thickness: Optional[float] = Field(None, ge=0, description="Layer thickness in mm")
    laser_power: Optional[float] = Field(None, ge=0, description="Laser power in watts")
    scan_speed: Optional[float] = Field(None, ge=0, description="Scan speed in mm/s")
    hatch_spacing: Optional[float] = Field(None, ge=0, description="Hatch spacing in mm")
    exposure_time: Optional[float] = Field(None, ge=0, description="Exposure time in seconds")
    
    # Geometry data (flattened)
    contour_paths: Optional[str] = Field(None, description="Contour paths")
    hatch_patterns: Optional[str] = Field(None, description="Hatch patterns")
    support_structures: Optional[str] = Field(None, description="Support structures")
    support_type: Optional[str] = Field(None, description="Support type")
    support_density: Optional[float] = Field(None, ge=0, le=1, description="Support density")
    
    # Material requirements (flattened)
    material_type: Optional[str] = Field(None, description="Material type")
    powder_type: Optional[str] = Field(None, description="Powder type")
    powder_amount: Optional[float] = Field(None, ge=0, description="Powder amount in grams")
    powder_condition: Optional[str] = Field(None, description="Powder condition")
    plate_material: Optional[str] = Field(None, description="Plate material")
    plate_temperature: Optional[float] = Field(None, description="Plate temperature in Celsius")
    plate_preparation: Optional[str] = Field(None, description="Plate preparation")
    
    # Quality requirements (flattened)
    dimensional_tolerance: Optional[float] = Field(None, ge=0, description="Dimensional tolerance in mm")
    surface_roughness: Optional[float] = Field(None, ge=0, description="Surface roughness in micrometers")
    density_requirement: Optional[float] = Field(None, ge=0, le=100, description="Density requirement percentage")
    tensile_strength: Optional[float] = Field(None, ge=0, description="Tensile strength in MPa")
    yield_strength: Optional[float] = Field(None, ge=0, description="Yield strength in MPa")
    hardness: Optional[float] = Field(None, ge=0, description="Hardness in HV")
    
    # Process parameters (flattened)
    build_temperature: Optional[float] = Field(None, description="Build temperature in Celsius")
    chamber_atmosphere: Optional[str] = Field(None, description="Chamber atmosphere")
    oxygen_level: Optional[float] = Field(None, ge=0, le=100, description="Oxygen level percentage")
    build_speed: Optional[float] = Field(None, ge=0, description="Build speed in mm/s")
    cooling_rate: Optional[float] = Field(None, ge=0, description="Cooling rate in °C/min")
    
    # User and session data (flattened)
    user_id: Optional[str] = Field(None, description="User ID")
    session_id: Optional[str] = Field(None, description="Session ID")
    created_by: Optional[str] = Field(None, description="Created by")
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class CTScanImageModel(BaseModel):
    """CT Scan Image model for ClickHouse - matches ct_scan_images.sql schema with optional fields."""
    # Primary identifiers
    id: int = Field(..., description="Primary key ID")
    scan_id: str = Field(..., description="Scan ID")
    process_id: Optional[str] = Field(None, description="Process ID")
    build_id: Optional[str] = Field(None, description="Build ID")
    part_id: Optional[str] = Field(None, description="Part ID")
    machine_id: Optional[str] = Field(None, description="Machine ID")
    timestamp: datetime = Field(..., description="Scan timestamp")
    
    # File information (flattened)
    file_path: Optional[str] = Field(None, description="File path")
    file_name: Optional[str] = Field(None, description="File name")
    file_size: Optional[int] = Field(None, ge=0, description="File size in bytes")
    file_hash: Optional[str] = Field(None, description="File hash")
    file_format: Optional[str] = Field(None, description="File format")
    file_status: Optional[str] = Field(None, description="File status")
    
    # CT scan metadata (flattened)
    scan_type: Optional[str] = Field(None, description="Scan type")
    scan_resolution_x: Optional[int] = Field(None, ge=0, description="Scan resolution X")
    scan_resolution_y: Optional[int] = Field(None, ge=0, description="Scan resolution Y")
    scan_resolution_z: Optional[int] = Field(None, ge=0, description="Scan resolution Z")
    voxel_size_x: Optional[float] = Field(None, ge=0, description="Voxel size X")
    voxel_size_y: Optional[float] = Field(None, ge=0, description="Voxel size Y")
    voxel_size_z: Optional[float] = Field(None, ge=0, description="Voxel size Z")
    scan_duration: Optional[float] = Field(None, ge=0, description="Scan duration in seconds")
    scan_parameters: Optional[str] = Field(None, description="Scan parameters")
    
    # Image processing (flattened)
    processing_status: Optional[str] = Field(None, description="Processing status")
    processing_algorithm: Optional[str] = Field(None, description="Processing algorithm")
    processing_parameters: Optional[str] = Field(None, description="Processing parameters")
    processing_duration: Optional[float] = Field(None, ge=0, description="Processing duration in seconds")
    processing_timestamp: Optional[datetime] = Field(None, description="Processing timestamp")
    
    # Quality metrics (flattened)
    image_quality_score: Optional[float] = Field(None, ge=0, le=1, description="Image quality score")
    noise_level: Optional[float] = Field(None, ge=0, description="Noise level")
    contrast_ratio: Optional[float] = Field(None, ge=0, description="Contrast ratio")
    sharpness_score: Optional[float] = Field(None, ge=0, le=1, description="Sharpness score")
    artifact_count: Optional[int] = Field(None, ge=0, description="Artifact count")
    
    # Defect analysis (flattened)
    defect_count: Optional[int] = Field(None, ge=0, description="Defect count")
    defect_types: Optional[List[str]] = Field(None, description="Defect types")
    defect_severity: Optional[str] = Field(None, description="Defect severity")
    defect_locations: Optional[List[str]] = Field(None, description="Defect locations")
    defect_volumes: Optional[List[float]] = Field(None, description="Defect volumes")
    
    # Dimensional measurements (flattened)
    dimensional_accuracy: Optional[float] = Field(None, ge=0, description="Dimensional accuracy")
    measurement_uncertainty: Optional[float] = Field(None, ge=0, description="Measurement uncertainty")
    measurement_method: Optional[str] = Field(None, description="Measurement method")
    measurement_equipment: Optional[str] = Field(None, description="Measurement equipment")
    
    # User and session data (flattened)
    user_id: Optional[str] = Field(None, description="User ID")
    session_id: Optional[str] = Field(None, description="Session ID")
    operator_id: Optional[str] = Field(None, description="Operator ID")
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class PowderBedImageModel(BaseModel):
    """Powder Bed Image model for ClickHouse - matches powder_bed_images.sql schema with optional fields."""
    # Primary identifiers
    id: int = Field(..., description="Primary key ID")
    image_id: str = Field(..., description="Image ID")
    process_id: Optional[str] = Field(None, description="Process ID")
    build_id: Optional[str] = Field(None, description="Build ID")
    part_id: Optional[str] = Field(None, description="Part ID")
    machine_id: Optional[str] = Field(None, description="Machine ID")
    timestamp: datetime = Field(..., description="Image timestamp")
    
    # File information (flattened)
    file_path: Optional[str] = Field(None, description="File path")
    file_name: Optional[str] = Field(None, description="File name")
    file_size: Optional[int] = Field(None, ge=0, description="File size in bytes")
    file_hash: Optional[str] = Field(None, description="File hash")
    file_format: Optional[str] = Field(None, description="File format")
    file_status: Optional[str] = Field(None, description="File status")
    
    # Image metadata (flattened)
    image_type: Optional[str] = Field(None, description="Image type")
    image_width: Optional[int] = Field(None, ge=0, description="Image width in pixels")
    image_height: Optional[int] = Field(None, ge=0, description="Image height in pixels")
    image_depth: Optional[int] = Field(None, ge=0, le=255, description="Image depth in bits")
    image_channels: Optional[int] = Field(None, ge=0, le=4, description="Number of image channels")
    image_resolution: Optional[float] = Field(None, ge=0, description="Image resolution in DPI")
    image_compression: Optional[str] = Field(None, description="Image compression")
    
    # Powder bed information (flattened)
    powder_bed_temperature: Optional[float] = Field(None, description="Powder bed temperature in Celsius")
    powder_bed_pressure: Optional[float] = Field(None, ge=0, description="Powder bed pressure in bar")
    powder_bed_humidity: Optional[float] = Field(None, ge=0, le=100, description="Powder bed humidity percentage")
    powder_bed_level: Optional[float] = Field(None, ge=0, le=100, description="Powder bed level percentage")
    powder_bed_quality: Optional[str] = Field(None, description="Powder bed quality")
    
    # Image processing (flattened)
    processing_status: Optional[str] = Field(None, description="Processing status")
    processing_algorithm: Optional[str] = Field(None, description="Processing algorithm")
    processing_parameters: Optional[str] = Field(None, description="Processing parameters")
    processing_duration: Optional[float] = Field(None, ge=0, description="Processing duration in seconds")
    processing_timestamp: Optional[datetime] = Field(None, description="Processing timestamp")
    
    # Quality metrics (flattened)
    image_quality_score: Optional[float] = Field(None, ge=0, le=1, description="Image quality score")
    noise_level: Optional[float] = Field(None, ge=0, le=1, description="Noise level")
    contrast_ratio: Optional[float] = Field(None, ge=0, description="Contrast ratio")
    sharpness_score: Optional[float] = Field(None, ge=0, le=1, description="Sharpness score")
    brightness: Optional[float] = Field(None, ge=0, le=255, description="Image brightness")
    exposure_level: Optional[float] = Field(None, ge=0, le=1, description="Exposure level")
    
    # Defect analysis (flattened)
    defect_count: Optional[int] = Field(None, ge=0, description="Defect count")
    defect_types: Optional[List[str]] = Field(None, description="Defect types")
    defect_severity: Optional[str] = Field(None, description="Defect severity")
    defect_locations: Optional[List[str]] = Field(None, description="Defect locations")
    defect_areas: Optional[List[float]] = Field(None, description="Defect areas")
    
    # Powder analysis (flattened)
    powder_coverage: Optional[float] = Field(None, ge=0, le=1, description="Powder coverage percentage")
    powder_distribution: Optional[str] = Field(None, description="Powder distribution")
    powder_particle_size: Optional[float] = Field(None, ge=0, description="Powder particle size in micrometers")
    powder_flowability: Optional[float] = Field(None, ge=0, le=1, description="Powder flowability")
    
    # User and session data (flattened)
    user_id: Optional[str] = Field(None, description="User ID")
    session_id: Optional[str] = Field(None, description="Session ID")
    operator_id: Optional[str] = Field(None, description="Operator ID")
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class ProcessImageModel(BaseModel):
    """Process Image model for ClickHouse - matches process_images.sql schema with optional fields."""
    # Primary identifiers
    id: int = Field(..., description="Primary key ID")
    image_id: str = Field(..., description="Image ID")
    process_id: Optional[str] = Field(None, description="Process ID")
    build_id: Optional[str] = Field(None, description="Build ID")
    part_id: Optional[str] = Field(None, description="Part ID")
    machine_id: Optional[str] = Field(None, description="Machine ID")
    timestamp: datetime = Field(..., description="Image timestamp")
    
    # File information (flattened)
    file_path: Optional[str] = Field(None, description="File path")
    file_name: Optional[str] = Field(None, description="File name")
    file_size: Optional[int] = Field(None, ge=0, description="File size in bytes")
    file_hash: Optional[str] = Field(None, description="File hash")
    file_format: Optional[str] = Field(None, description="File format")
    file_status: Optional[str] = Field(None, description="File status")
    
    # Image metadata (flattened)
    image_type: Optional[str] = Field(None, description="Image type")
    image_width: Optional[int] = Field(None, ge=0, description="Image width in pixels")
    image_height: Optional[int] = Field(None, ge=0, description="Image height in pixels")
    image_depth: Optional[int] = Field(None, ge=0, le=255, description="Image depth in bits")
    image_channels: Optional[int] = Field(None, ge=0, le=4, description="Number of image channels")
    image_resolution: Optional[float] = Field(None, ge=0, description="Image resolution in DPI")
    image_compression: Optional[str] = Field(None, description="Image compression")
    
    # Process information (flattened)
    process_stage: Optional[str] = Field(None, description="Process stage")
    layer_number: Optional[int] = Field(None, ge=0, description="Layer number")
    process_temperature: Optional[float] = Field(None, description="Process temperature in Celsius")
    process_pressure: Optional[float] = Field(None, ge=0, description="Process pressure in bar")
    process_speed: Optional[float] = Field(None, ge=0, description="Process speed in mm/s")
    laser_power: Optional[float] = Field(None, ge=0, description="Laser power in watts")
    
    # Image processing (flattened)
    processing_status: Optional[str] = Field(None, description="Processing status")
    processing_algorithm: Optional[str] = Field(None, description="Processing algorithm")
    processing_parameters: Optional[str] = Field(None, description="Processing parameters")
    processing_duration: Optional[float] = Field(None, ge=0, description="Processing duration in seconds")
    processing_timestamp: Optional[datetime] = Field(None, description="Processing timestamp")
    
    # Quality metrics (flattened)
    image_quality_score: Optional[float] = Field(None, ge=0, le=1, description="Image quality score")
    noise_level: Optional[float] = Field(None, ge=0, le=1, description="Noise level")
    contrast_ratio: Optional[float] = Field(None, ge=0, description="Contrast ratio")
    sharpness_score: Optional[float] = Field(None, ge=0, le=1, description="Sharpness score")
    brightness: Optional[float] = Field(None, ge=0, le=255, description="Image brightness")
    exposure_level: Optional[float] = Field(None, ge=0, le=1, description="Exposure level")
    
    # Defect analysis (flattened)
    defect_count: Optional[int] = Field(None, ge=0, description="Defect count")
    defect_types: Optional[List[str]] = Field(None, description="Defect types")
    defect_severity: Optional[str] = Field(None, description="Defect severity")
    defect_locations: Optional[List[str]] = Field(None, description="Defect locations")
    defect_areas: Optional[List[float]] = Field(None, description="Defect areas")
    
    # Process analysis (flattened)
    process_quality_score: Optional[float] = Field(None, ge=0, le=1, description="Process quality score")
    layer_completeness: Optional[float] = Field(None, ge=0, le=1, description="Layer completeness percentage")
    dimensional_accuracy: Optional[float] = Field(None, ge=0, description="Dimensional accuracy")
    surface_quality: Optional[str] = Field(None, description="Surface quality")
    
    # User and session data (flattened)
    user_id: Optional[str] = Field(None, description="User ID")
    session_id: Optional[str] = Field(None, description="Session ID")
    operator_id: Optional[str] = Field(None, description="Operator ID")
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class MachineBuildFileModel(BaseModel):
    """Machine Build File model for ClickHouse - matches machine_build_files.sql schema with optional fields."""
    # Primary identifiers
    id: int = Field(..., description="Primary key ID")
    file_id: str = Field(..., description="File ID")
    machine_id: Optional[str] = Field(None, description="Machine ID")
    build_id: Optional[str] = Field(None, description="Build ID")
    process_id: Optional[str] = Field(None, description="Process ID")
    timestamp: datetime = Field(..., description="File timestamp")
    
    # File information (flattened)
    file_path: Optional[str] = Field(None, description="File path")
    file_name: Optional[str] = Field(None, description="File name")
    file_size: Optional[int] = Field(None, ge=0, description="File size in bytes")
    file_hash: Optional[str] = Field(None, description="File hash")
    file_format: Optional[str] = Field(None, description="File format")
    file_status: Optional[str] = Field(None, description="File status")
    
    # Build file metadata (flattened)
    file_type: Optional[str] = Field(None, description="File type")
    file_category: Optional[str] = Field(None, description="File category")
    file_version: Optional[str] = Field(None, description="File version")
    file_priority: Optional[int] = Field(None, ge=0, description="File priority")
    file_dependencies: Optional[List[str]] = Field(None, description="File dependencies")
    
    # Machine information (flattened)
    machine_type: Optional[str] = Field(None, description="Machine type")
    machine_model: Optional[str] = Field(None, description="Machine model")
    machine_serial_number: Optional[str] = Field(None, description="Machine serial number")
    machine_firmware_version: Optional[str] = Field(None, description="Machine firmware version")
    machine_software_version: Optional[str] = Field(None, description="Machine software version")
    
    # Build information (flattened)
    build_type: Optional[str] = Field(None, description="Build type")
    build_parameters: Optional[str] = Field(None, description="Build parameters")
    build_configuration: Optional[str] = Field(None, description="Build configuration")
    build_environment: Optional[str] = Field(None, description="Build environment")
    
    # File processing (flattened)
    processing_status: Optional[str] = Field(None, description="Processing status")
    processing_algorithm: Optional[str] = Field(None, description="Processing algorithm")
    processing_parameters: Optional[str] = Field(None, description="Processing parameters")
    processing_duration: Optional[float] = Field(None, ge=0, description="Processing duration in seconds")
    processing_timestamp: Optional[datetime] = Field(None, description="Processing timestamp")
    
    # Quality metrics (flattened)
    file_quality_score: Optional[float] = Field(None, ge=0, le=1, description="File quality score")
    file_integrity_score: Optional[float] = Field(None, ge=0, le=1, description="File integrity score")
    file_completeness: Optional[float] = Field(None, ge=0, le=1, description="File completeness")
    file_validation_status: Optional[str] = Field(None, description="File validation status")
    
    # Usage analytics (flattened)
    usage_count: Optional[int] = Field(None, ge=0, description="Usage count")
    last_accessed: Optional[datetime] = Field(None, description="Last accessed timestamp")
    access_frequency: Optional[float] = Field(None, ge=0, description="Access frequency")
    user_rating: Optional[float] = Field(None, ge=0, le=5, description="User rating")
    
    # User and session data (flattened)
    user_id: Optional[str] = Field(None, description="User ID")
    session_id: Optional[str] = Field(None, description="Session ID")
    operator_id: Optional[str] = Field(None, description="Operator ID")
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
