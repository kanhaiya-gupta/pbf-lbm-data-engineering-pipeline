"""
MongoDB Document Models for PBF-LB/M Data Pipeline

This module provides Pydantic models that EXACTLY match the MongoDB JSON schemas,
ensuring perfect validation and consistency between models and database schemas.
"""

from typing import Any, Dict, Optional, List, Union
from datetime import datetime
from pydantic import BaseModel, Field, field_validator, model_validator
from enum import Enum


class FileStatus(str, Enum):
    """Enum for file processing status."""
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    PROCESSED = "processed"
    FAILED = "failed"
    ARCHIVED = "archived"


class BaseMongoDBDocument(BaseModel):
    """
    Base MongoDB document model that matches JSON schemas exactly.
    """
    
    # MongoDB-specific fields
    id: Optional[str] = Field(None, alias="_id", description="MongoDB ObjectId")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Document creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Document last update timestamp")
    version: int = Field(default=1, ge=1, description="Document version for optimistic locking")
    
    # Document identification
    document_type: str = Field(..., description="Type of document")
    document_id: str = Field(..., pattern="^[A-Z0-9_]+$", description="Unique document identifier")
    
    # PostgreSQL relationships
    process_id: Optional[str] = Field(None, pattern="^[A-Z0-9_]+$", description="Associated PostgreSQL process ID")
    build_id: Optional[str] = Field(None, pattern="^[A-Z0-9_]+$", description="Associated PostgreSQL build ID")
    part_id: Optional[str] = Field(None, pattern="^[A-Z0-9_]+$", description="Associated PostgreSQL part ID")
    machine_id: Optional[str] = Field(None, pattern="^[A-Z0-9_]+$", description="Associated PostgreSQL machine ID")
    
    # File storage information
    file_path: Optional[str] = Field(None, description="Path to file in GridFS")
    file_size: Optional[int] = Field(None, description="File size in bytes")
    file_hash: Optional[str] = Field(None, description="File hash for integrity verification")
    file_status: FileStatus = Field(default=FileStatus.UPLOADED, description="File processing status")
    
    # Metadata and relationships
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Flexible document metadata")
    tags: List[str] = Field(default_factory=list, description="Searchable tags")
    relationships: Dict[str, List[str]] = Field(default_factory=dict, description="Document relationships")
    
    class Config:
        """Pydantic configuration for MongoDB documents."""
        use_enum_values = True
        validate_assignment = True
        allow_population_by_field_name = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
    
    @field_validator('document_id', 'process_id', 'build_id', 'part_id', 'machine_id')
    @classmethod
    def validate_id_format(cls, v):
        """Validate ID formats."""
        if v is not None and not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError("ID must contain only alphanumeric characters, underscores, and hyphens")
        return v
    
    @model_validator(mode='after')
    def validate_relationships(self):
        """Validate that at least one PostgreSQL relationship exists."""
        if not any([self.process_id, self.build_id, self.part_id, self.machine_id]):
            raise ValueError("At least one PostgreSQL relationship (process_id, build_id, part_id, machine_id) must be provided")
        return self


class ProcessImageDocument(BaseMongoDBDocument):
    """
    MongoDB document model for process images - matches process_images_collection.json exactly.
    """
    
    document_type: str = Field(default="image", description="Document type")
    
    # Image-specific fields - EXACTLY matching JSON schema
    image_width: Optional[int] = Field(None, description="Image width in pixels")
    image_height: Optional[int] = Field(None, description="Image height in pixels")
    image_format: Optional[str] = Field(None, description="Image format")
    image_quality: Optional[str] = Field(None, description="Image quality assessment")
    camera_settings: Optional[Dict[str, Any]] = Field(None, description="Camera settings when image was taken")
    
    # Image processing metadata
    processing_status: Optional[str] = Field(None, description="Image processing status")
    analysis_results: Optional[Dict[str, Any]] = Field(None, description="Image analysis results")
    defects_detected: Optional[List[Dict[str, Any]]] = Field(None, description="Defects detected in image")
    
    @field_validator('image_format')
    @classmethod
    def validate_image_format(cls, v):
        """Validate image format matches JSON schema enum."""
        if v is not None and v not in ["JPEG", "PNG", "TIFF", "BMP", "WEBP", "DICOM", "RAW"]:
            raise ValueError("Image format must be one of: JPEG, PNG, TIFF, BMP, WEBP, DICOM, RAW")
        return v
    
    @field_validator('image_quality')
    @classmethod
    def validate_image_quality(cls, v):
        """Validate image quality matches JSON schema enum."""
        if v is not None and v not in ["low", "medium", "high", "ultra"]:
            raise ValueError("Image quality must be one of: low, medium, high, ultra")
        return v
    
    @field_validator('processing_status')
    @classmethod
    def validate_processing_status(cls, v):
        """Validate processing status matches JSON schema enum."""
        if v is not None and v not in ["pending", "processing", "analyzed", "failed"]:
            raise ValueError("Processing status must be one of: pending, processing, analyzed, failed")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "document_type": "image",
                "document_id": "IMG_2024_001",
                "process_id": "PROC_2024_001",
                "image_width": 1920,
                "image_height": 1080,
                "image_format": "JPEG",
                "image_quality": "high",
                "camera_settings": {"exposure": "1/60", "iso": "100", "aperture": "f/2.8"},
                "processing_status": "analyzed",
                "analysis_results": {"defect_count": 0, "quality_score": 95.5},
                "defects_detected": []
            }
        }


class CTScanImageDocument(BaseMongoDBDocument):
    """
    MongoDB document model for CT scan images - matches ct_scan_images_collection.json exactly.
    """
    
    document_type: str = Field(default="ct_scan_image", description="Document type")
    
    # CT scan specific fields - EXACTLY matching JSON schema
    scan_parameters: Dict[str, Any] = Field(..., description="CT scan parameters")
    image_dimensions: Dict[str, Any] = Field(..., description="Image dimensions")
    image_format: Optional[str] = Field(None, description="Image format")
    image_quality: Optional[Dict[str, Any]] = Field(None, description="Image quality metrics")
    processing_status: Optional[str] = Field(None, description="Image processing status")
    analysis_results: Optional[Dict[str, Any]] = Field(None, description="CT scan analysis results")
    scan_metadata: Optional[Dict[str, Any]] = Field(None, description="Additional scan metadata")
    
    @field_validator('image_format')
    @classmethod
    def validate_image_format(cls, v):
        """Validate image format matches JSON schema enum."""
        if v is not None and v not in ["DICOM", "TIFF", "RAW", "TXM", "NIfTI", "Analyze", "MHA", "MHD"]:
            raise ValueError("Image format must be one of: DICOM, TIFF, RAW, TXM, NIfTI, Analyze, MHA, MHD")
        return v
    
    @field_validator('processing_status')
    @classmethod
    def validate_processing_status(cls, v):
        """Validate processing status matches JSON schema enum."""
        if v is not None and v not in ["pending", "processing", "processed", "failed"]:
            raise ValueError("Processing status must be one of: pending, processing, processed, failed")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "document_type": "ct_scan_image",
                "document_id": "CT_IMG_2024_001",
                "process_id": "PROC_2024_001",
                "scan_parameters": {
                    "voltage": 200.0,
                    "current": 10.0,
                    "exposure_time": 0.5,
                    "number_of_projections": 1000
                },
                "image_dimensions": {
                    "width": 1024,
                    "height": 1024,
                    "depth": 512
                },
                "image_format": "DICOM"
            }
        }


class PowderBedImageDocument(BaseMongoDBDocument):
    """
    MongoDB document model for powder bed images - matches powder_bed_images_collection.json exactly.
    """
    
    document_type: str = Field(default="powder_bed_image", description="Document type")
    
    # Powder bed specific fields
    layer_number: int = Field(..., ge=0, description="Layer number corresponding to the image")
    image_dimensions: Dict[str, Any] = Field(..., description="Image dimensions")
    image_format: Optional[str] = Field(None, description="Image format")
    camera_settings: Optional[Dict[str, Any]] = Field(None, description="Camera settings when image was taken")
    powder_bed_conditions: Optional[Dict[str, Any]] = Field(None, description="Powder bed conditions during imaging")
    processing_status: Optional[str] = Field(None, description="Image processing status")
    analysis_results: Optional[Dict[str, Any]] = Field(None, description="Powder bed analysis results")
    defect_analysis: Optional[List[Dict[str, Any]]] = Field(None, description="Detailed defect analysis")
    image_metadata: Optional[Dict[str, Any]] = Field(None, description="Additional image metadata")
    
    class Config:
        schema_extra = {
            "example": {
                "document_type": "powder_bed_image",
                "document_id": "POWDER_IMG_2024_001",
                "process_id": "PROC_2024_001",
                "layer_number": 150,
                "image_dimensions": {
                    "width": 1920,
                    "height": 1080,
                    "resolution": 0.1
                },
                "image_format": "JPEG"
            }
        }


class MachineBuildFileDocument(BaseMongoDBDocument):
    """
    MongoDB document model for machine build files - matches machine_build_files_collection.json exactly.
    """
    
    document_type: str = Field(default="build_file", description="Document type")
    
    # Build file specific fields - EXACTLY matching JSON schema
    file_type: str = Field(..., description="Build file type")
    machine_type: Optional[str] = Field(None, description="Target machine type")
    build_parameters: Optional[Dict[str, Any]] = Field(None, description="Build parameters extracted from file")
    layer_count: Optional[int] = Field(None, description="Number of layers in build file")
    estimated_build_time: Optional[float] = Field(None, description="Estimated build time in hours")
    validation_status: Optional[str] = Field(None, description="File validation status")
    validation_errors: Optional[List[str]] = Field(None, description="Validation errors if any")
    processing_notes: Optional[str] = Field(None, description="Processing notes and comments")
    file_metadata: Optional[Dict[str, Any]] = Field(None, description="Additional file metadata")
    
    @field_validator('file_type')
    @classmethod
    def validate_file_type(cls, v):
        """Validate file type matches JSON schema enum."""
        if v not in ["slm", "cli", "sli", "gcode", "nc"]:
            raise ValueError("File type must be one of: slm, cli, sli, gcode, nc")
        return v
    
    @field_validator('validation_status')
    @classmethod
    def validate_validation_status(cls, v):
        """Validate validation status matches JSON schema enum."""
        if v is not None and v not in ["pending", "validating", "valid", "invalid", "warning"]:
            raise ValueError("Validation status must be one of: pending, validating, valid, invalid, warning")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "document_type": "build_file",
                "document_id": "BUILD_FILE_2024_001",
                "process_id": "PROC_2024_001",
                "file_type": "slm",
                "machine_type": "SLM280HL",
                "build_parameters": {
                    "layer_height": 0.05,
                    "laser_power": 200,
                    "scan_speed": 1000
                },
                "layer_count": 150,
                "estimated_build_time": 8.5
            }
        }


class Model3DFileDocument(BaseMongoDBDocument):
    """
    MongoDB document model for 3D model files - matches 3d_model_files_collection.json exactly.
    """
    
    document_type: str = Field(default="3d_model", description="Document type")
    
    # 3D model specific fields - EXACTLY matching JSON schema
    model_format: str = Field(..., description="3D model file format")
    model_type: Optional[str] = Field(None, description="Type of 3D model")
    model_dimensions: Optional[Dict[str, Any]] = Field(None, description="3D model dimensions")
    model_quality: Optional[Dict[str, Any]] = Field(None, description="3D model quality metrics")
    processing_status: Optional[str] = Field(None, description="Model processing status")
    processing_results: Optional[Dict[str, Any]] = Field(None, description="Model processing results")
    model_metadata: Optional[Dict[str, Any]] = Field(None, description="Additional model metadata")
    build_requirements: Optional[Dict[str, Any]] = Field(None, description="Build requirements for the model")
    
    @field_validator('model_format')
    @classmethod
    def validate_model_format(cls, v):
        """Validate model format matches JSON schema enum."""
        if v not in ["STL", "STEP", "PLY", "OBJ", "3MF", "AMF"]:
            raise ValueError("Model format must be one of: STL, STEP, PLY, OBJ, 3MF, AMF")
        return v
    
    @field_validator('model_type')
    @classmethod
    def validate_model_type(cls, v):
        """Validate model type matches JSON schema enum."""
        if v is not None and v not in ["CAD", "SCAN", "SIMULATION", "REFERENCE", "TEMPLATE"]:
            raise ValueError("Model type must be one of: CAD, SCAN, SIMULATION, REFERENCE, TEMPLATE")
        return v
    
    @field_validator('processing_status')
    @classmethod
    def validate_processing_status(cls, v):
        """Validate processing status matches JSON schema enum."""
        if v is not None and v not in ["pending", "processing", "processed", "failed"]:
            raise ValueError("Processing status must be one of: pending, processing, processed, failed")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "document_type": "3d_model",
                "document_id": "MODEL_3D_2024_001",
                "process_id": "PROC_2024_001",
                "model_format": "STL",
                "model_type": "CAD",
                "model_dimensions": {
                    "length": 100.0,
                    "width": 50.0,
                    "height": 25.0,
                    "volume": 125000.0
                }
            }
        }


class RawSensorDataDocument(BaseMongoDBDocument):
    """
    MongoDB document model for raw sensor data - matches raw_sensor_data_collection.json exactly.
    """
    
    document_type: str = Field(default="raw_sensor_data", description="Document type")
    
    # Sensor data specific fields
    sensor_type: str = Field(..., description="Type of sensor")
    sensor_id: str = Field(..., description="Sensor identifier")
    data_format: str = Field(..., description="Data format")
    sampling_rate: Optional[float] = Field(None, description="Sampling rate in Hz")
    data_duration: Optional[float] = Field(None, description="Data duration in seconds")
    data_points: Optional[int] = Field(None, description="Number of data points")
    processing_status: Optional[str] = Field(None, description="Data processing status")
    data_quality: Optional[str] = Field(None, description="Data quality assessment")
    calibration_info: Optional[Dict[str, Any]] = Field(None, description="Sensor calibration information")
    sensor_location: Optional[Dict[str, Any]] = Field(None, description="3D coordinates of the sensor")
    measurement_range: Optional[Dict[str, Any]] = Field(None, description="Expected min/max range of measurement")
    data_statistics: Optional[Dict[str, Any]] = Field(None, description="Statistical analysis of the data")
    anomaly_detection: Optional[Dict[str, Any]] = Field(None, description="Anomaly detection results")
    
    class Config:
        schema_extra = {
            "example": {
                "document_type": "raw_sensor_data",
                "document_id": "SENSOR_DATA_2024_001",
                "process_id": "PROC_2024_001",
                "sensor_type": "THERMAL",
                "sensor_id": "THERMAL_001",
                "data_format": "binary",
                "sampling_rate": 100.0,
                "data_duration": 3600.0
            }
        }


class ProcessLogDocument(BaseMongoDBDocument):
    """
    MongoDB document model for process logs - matches process_logs_collection.json exactly.
    """
    
    document_type: str = Field(..., description="Document type (process_log or user_annotation)")
    
    # Log specific fields
    log_level: Optional[str] = Field(None, description="Log level")
    log_source: Optional[str] = Field(None, description="Source of the log")
    log_message: Optional[str] = Field(None, description="Log message content")
    error_code: Optional[str] = Field(None, description="Error code if applicable")
    stack_trace: Optional[str] = Field(None, description="Stack trace for errors")
    context_data: Optional[Dict[str, Any]] = Field(None, description="Additional context data")
    related_errors: Optional[List[str]] = Field(None, description="Related error IDs")
    resolution_status: Optional[str] = Field(None, description="Error resolution status")
    
    # User annotation specific fields
    user_id: Optional[str] = Field(None, description="User who created the annotation")
    annotation_type: Optional[str] = Field(None, description="Type of annotation")
    content: Optional[str] = Field(None, description="Annotation content")
    annotation_position: Optional[Dict[str, Any]] = Field(None, description="Position/coordinates of annotation")
    is_public: Optional[bool] = Field(None, description="Whether annotation is public")
    is_resolved: Optional[bool] = Field(None, description="Whether annotation is resolved")
    parent_annotation: Optional[str] = Field(None, description="Parent annotation ID for replies")
    attachments: Optional[List[str]] = Field(None, description="Attached file IDs")
    
    class Config:
        schema_extra = {
            "example": {
                "document_type": "process_log",
                "document_id": "LOG_2024_001",
                "process_id": "PROC_2024_001",
                "log_level": "ERROR",
                "log_source": "laser_system",
                "log_message": "Laser power dropped below threshold"
            }
        }


class BuildInstructionDocument(BaseMongoDBDocument):
    """
    MongoDB document model for build instructions - matches build_instructions_collection.json exactly.
    """
    
    document_type: str = Field(default="build_instruction", description="Document type")
    
    # Build instruction specific fields
    instruction_type: str = Field(..., description="Type of build instruction")
    instruction_title: str = Field(..., description="Title of the instruction")
    instruction_content: str = Field(..., description="Main instruction content")
    instruction_steps: Optional[List[Dict[str, Any]]] = Field(None, description="Step-by-step instructions")
    safety_requirements: Optional[List[Dict[str, Any]]] = Field(None, description="Safety requirements and protocols")
    quality_checkpoints: Optional[List[Dict[str, Any]]] = Field(None, description="Quality control checkpoints")
    required_tools: Optional[List[Dict[str, Any]]] = Field(None, description="Required tools and equipment")
    material_requirements: Optional[List[Dict[str, Any]]] = Field(None, description="Material requirements")
    approval_workflow: Optional[Dict[str, Any]] = Field(None, description="Approval workflow information")
    version_control: Optional[Dict[str, Any]] = Field(None, description="Version control information")
    
    class Config:
        schema_extra = {
            "example": {
                "document_type": "build_instruction",
                "document_id": "INSTRUCTION_2024_001",
                "process_id": "PROC_2024_001",
                "instruction_type": "setup",
                "instruction_title": "Machine Setup Procedure",
                "instruction_content": "Complete machine setup for PBF process"
            }
        }


class MachineConfigDocument(BaseMongoDBDocument):
    """
    MongoDB document model for machine configurations - matches machine_configurations_collection.json exactly.
    """
    
    document_type: str = Field(default="machine_config", description="Document type")
    
    # Machine config specific fields
    config_type: str = Field(..., description="Type of configuration")
    config_version: str = Field(..., description="Configuration version")
    config_data: Dict[str, Any] = Field(..., description="Configuration data")
    is_active: bool = Field(default=True, description="Whether configuration is active")
    applied_by: Optional[str] = Field(None, description="User who applied the configuration")
    applied_at: Optional[datetime] = Field(None, description="When configuration was applied")
    validation_status: Optional[str] = Field(None, description="Configuration validation status")
    validation_errors: Optional[List[str]] = Field(None, description="Validation errors if any")
    config_file_path: Optional[str] = Field(None, description="Path to configuration file in GridFS")
    config_file_size: Optional[int] = Field(None, description="Configuration file size in bytes")
    config_file_hash: Optional[str] = Field(None, description="Configuration file hash for integrity verification")
    dependencies: Optional[List[str]] = Field(None, description="Dependent configuration IDs")
    compatibility: Optional[Dict[str, Any]] = Field(None, description="Machine compatibility information")
    rollback_info: Optional[Dict[str, Any]] = Field(None, description="Rollback information")
    
    class Config:
        schema_extra = {
            "example": {
                "document_type": "machine_config",
                "document_id": "CONFIG_2024_001",
                "machine_id": "MACHINE_001",
                "config_type": "calibration",
                "config_version": "1.2.0",
                "config_data": {
                    "laser_power_calibration": 1.05,
                    "temperature_offset": 2.0
                }
            }
        }
