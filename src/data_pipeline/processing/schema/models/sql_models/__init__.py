"""
Schema Models Module

This module contains Pydantic models for all PBF-LB/M data types in the data pipeline.
"""

from .base_model import BaseDataModel
from .pbf_process_model import PBFProcessModel, QualityMetrics, AtmosphereType, QualityGrade
from .ispm_monitoring_model import (
    ISPMMonitoringModel, 
    SensorLocation, 
    MeasurementRange, 
    EnvironmentalConditions,
    SensorType, 
    SignalQuality, 
    AnomalySeverity
)
from .ct_scan_model import (
    CTScanModel,
    ScanParameters,
    FileMetadata,
    ImageDimensions,
    CTQualityMetrics,
    DefectTypeInfo,
    DefectAnalysis,
    DimensionalAnalysis,
    ScanType,
    FileFormat,
    ProcessingStatus,
    ArtifactSeverity,
    DefectType,
    DefectSeverity,
    AcceptanceStatus
)
from .powder_bed_model import (
    PowderBedModel,
    ImageMetadata,
    CaptureSettings,
    PowderCharacteristics,
    ParticleSizeDistribution,
    BedQualityMetrics,
    ImageAnalysis,
    ColorBalance,
    TextureAnalysis,
    DefectDetection,
    PowderBedDefect,
    DefectLocation,
    EnvironmentalConditions as PowderBedEnvironmentalConditions,
    ImageFormat,
    ProcessingStatus as PowderBedProcessingStatus,
    PowderBedDefectType,
    DefectSeverity as PowderBedDefectSeverity,
    QualityAssessment
)
from .model_factory import (
    ModelFactory,
    ModelType,
    create_model,
    validate_data,
    get_model_schema,
    get_available_models
)

__all__ = [
    # Base model
    "BaseDataModel",
    
    # PBF Process models
    "PBFProcessModel",
    "QualityMetrics",
    "AtmosphereType",
    "QualityGrade",
    
    # ISPM Monitoring models
    "ISPMMonitoringModel",
    "SensorLocation",
    "MeasurementRange",
    "EnvironmentalConditions",
    "SensorType",
    "SignalQuality",
    "AnomalySeverity",
    
    # CT Scan models
    "CTScanModel",
    "ScanParameters",
    "FileMetadata",
    "ImageDimensions",
    "CTQualityMetrics",
    "DefectTypeInfo",
    "DefectAnalysis",
    "DimensionalAnalysis",
    "ScanType",
    "FileFormat",
    "ProcessingStatus",
    "ArtifactSeverity",
    "DefectType",
    "DefectSeverity",
    "AcceptanceStatus",
    
    # Powder Bed models
    "PowderBedModel",
    "ImageMetadata",
    "CaptureSettings",
    "PowderCharacteristics",
    "ParticleSizeDistribution",
    "BedQualityMetrics",
    "ImageAnalysis",
    "ColorBalance",
    "TextureAnalysis",
    "DefectDetection",
    "PowderBedDefect",
    "DefectLocation",
    "PowderBedEnvironmentalConditions",
    "ImageFormat",
    "PowderBedProcessingStatus",
    "PowderBedDefectType",
    "PowderBedDefectSeverity",
    "QualityAssessment",
    
    # Model factory
    "ModelFactory",
    "ModelType",
    "create_model",
    "validate_data",
    "get_model_schema",
    "get_available_models"
]

# Model registry for easy access
MODEL_REGISTRY = {
    "pbf_process": PBFProcessModel,
    "ispm_monitoring": ISPMMonitoringModel,
    "ct_scan": CTScanModel,
    "powder_bed": PowderBedModel
}

# Model type mappings
MODEL_TYPE_MAPPINGS = {
    "PBF_PROCESS": "pbf_process",
    "ISPM_MONITORING": "ispm_monitoring",
    "CT_SCAN": "ct_scan",
    "POWDER_BED": "powder_bed"
}

def get_model_class(model_type: str):
    """
    Get model class by type string.
    
    Args:
        model_type: Model type string
        
    Returns:
        Model class
    """
    return MODEL_REGISTRY.get(model_type)

def get_model_type_enum(model_type: str):
    """
    Get model type enum by string.
    
    Args:
        model_type: Model type string
        
    Returns:
        ModelType enum value
    """
    return ModelType(model_type)

def create_model_instance(model_type: str, data: dict):
    """
    Create a model instance from data.
    
    Args:
        model_type: Model type string
        data: Model data dictionary
        
    Returns:
        Model instance
    """
    model_class = get_model_class(model_type)
    if not model_class:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model_class(**data)

def validate_model_data(model_type: str, data: dict):
    """
    Validate data against a model schema.
    
    Args:
        model_type: Model type string
        data: Data dictionary to validate
        
    Returns:
        Validation result dictionary
    """
    return ModelFactory.validate_data(model_type, data)

def get_model_schema_info(model_type: str):
    """
    Get schema information for a model type.
    
    Args:
        model_type: Model type string
        
    Returns:
        Schema information dictionary
    """
    return ModelFactory.get_model_schema(model_type)

def get_all_model_types():
    """
    Get all available model types.
    
    Returns:
        List of model type strings
    """
    return list(MODEL_REGISTRY.keys())

def get_model_metadata(model_type: str):
    """
    Get metadata for a model type.
    
    Args:
        model_type: Model type string
        
    Returns:
        Model metadata dictionary
    """
    return ModelFactory.get_model_metadata(model_type)
