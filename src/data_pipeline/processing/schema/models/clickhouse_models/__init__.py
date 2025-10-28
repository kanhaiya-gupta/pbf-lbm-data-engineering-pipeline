"""
ClickHouse Models Module

This module contains Pydantic models for all ClickHouse data warehouse schemas.
ClickHouse is used for analytics, reporting, and business intelligence.
"""

from .core_operational_models import (
    # PBF Process models
    PBFProcessModel,
    ProcessParameters,
    MaterialInfo,
    QualityMetrics,
    ProcessStatus,
    AtmosphereType,
    QualityGrade,
    
    # Machine Status models
    MachineStatusModel,
    OperationalState,
    SystemHealth,
    LaserSystem,
    BuildPlatform,
    PowderSystem,
    EnvironmentalConditions,
    AlertInfo,
    PerformanceMetrics,
    MaintenanceInfo,
    MachineStatusType,
    HealthStatus,
    LaserStatus,
    PlatformStatus,
    PowderStatus,
    AlertSeverity,
    MaintenanceType,
    
    # Sensor Readings models
    SensorReadingModel,
    SensorType,
    UnitType,
    QualityScore,
    
    # Analytics models
    AnalyticsModel,
    PerformanceAnalysis,
    QualityAnalysis,
    CostAnalysis,
    TrendAnalysis,
    PredictiveAnalysis,
    AnomalyDetection,
    ComparativeAnalysis,
    AnalysisType,
    TrendDirection,
    AnomalyType,
    ComparisonType
)

from .mongodb_integration_models import (
    # Process Logs models
    ProcessLogModel,
    LogLevel,
    EventType,
    AnnotationType,
    LogMetadata,
    AnnotationInfo,
    RelatedDocument,
    LogRelationship,
    
    # Machine Configurations models
    MachineConfigurationModel,
    ConfigurationType,
    ConfigurationFormat,
    CalibrationData,
    CalibrationInfo,
    ConfigurationMetadata,
    
    # Raw Sensor Data models
    RawSensorDataModel,
    DataFormat,
    ProcessingStatus,
    DataQuality,
    CalibrationInfo as RawSensorCalibrationInfo,
    MeasurementRange,
    
    # 3D Model Files models
    ModelFileModel,
    ModelType,
    ModelFormat,
    GeometryInfo,
    QualityAssessment,
    ProcessingInfo,
    ModelMetadata,
    
    # Build Instructions models
    BuildInstructionModel,
    InstructionType,
    MaterialType,
    SupportType,
    InstructionMetadata,
    
    # CT Scan Images models
    CTScanImageModel,
    ScanType,
    ImageFormat,
    CTQualityMetrics,
    DefectAnalysis,
    DimensionalAnalysis,
    DefectType,
    DefectSeverity,
    
    # Powder Bed Images models
    PowderBedImageModel,
    PowderBedType,
    PowderAnalysis,
    PowderCharacteristics,
    BedQualityMetrics,
    ImageAnalysis,
    DefectDetection,
    PowderBedDefect,
    DefectLocation,
    PowderBedDefectType,
    PowderBedDefectSeverity,
    
    # Process Images models
    ProcessImageModel,
    ProcessStage,
    ProcessAnalysis,
    ProcessQualityMetrics,
    LayerAnalysis,
    ProcessDefect,
    ProcessDefectType,
    ProcessDefectSeverity,
    
    # Machine Build Files models
    MachineBuildFileModel,
    FileType,
    FileCategory,
    FileMetadata,
    UsageAnalytics,
    FileQualityMetrics
)

from .multi_database_models import (
    # Redis Cache Data models
    RedisCacheDataModel,
    CacheOperation,
    CacheType,
    CompressionType,
    CachePerformanceMetrics,
    DataType,
    CacheAnalytics,
    
    # Job Queue Data models
    JobQueueDataModel,
    JobType,
    JobStatus,
    ExecutionStatus,
    QueueAnalytics,
    ErrorInfo,
    
    # User Session Data models
    UserSessionDataModel,
    SessionStatus,
    SessionType,
    LoginMethod,
    SecurityLevel,
    SessionAnalytics,
    SecurityInfo,
    
    # Cassandra Time Series models
    CassandraTimeSeriesModel,
    AggregationType,
    AggregationWindow,
    CassandraCalibrationInfo,
    TimeSeriesAnalytics,
    
    # ISPM Monitoring models
    ISPMMonitoringModel,
    MonitoringType,
    ISPMAnomalyType,
    AnomalySeverity,
    ISPMEnvironmentalConditions,
    ISPMProcessParameters,
    ISPMAnomalyDetection
)

from .model_factory import (
    ClickHouseModelFactory,
    ClickHouseModelType,
    create_clickhouse_model,
    validate_clickhouse_data,
    get_clickhouse_model_schema,
    get_available_clickhouse_models
)

__all__ = [
    # Core Operational Models
    "PBFProcessModel",
    "ProcessParameters",
    "MaterialInfo", 
    "QualityMetrics",
    "ProcessStatus",
    "AtmosphereType",
    "QualityGrade",
    
    "MachineStatusModel",
    "OperationalState",
    "SystemHealth",
    "LaserSystem",
    "BuildPlatform",
    "PowderSystem",
    "EnvironmentalConditions",
    "AlertInfo",
    "PerformanceMetrics",
    "MaintenanceInfo",
    "MachineStatusType",
    "HealthStatus",
    "LaserStatus",
    "PlatformStatus",
    "PowderStatus",
    "AlertSeverity",
    "MaintenanceType",
    
    "SensorReadingModel",
    "SensorType",
    "UnitType",
    "QualityScore",
    
    "AnalyticsModel",
    "PerformanceAnalysis",
    "QualityAnalysis",
    "CostAnalysis",
    "TrendAnalysis",
    "PredictiveAnalysis",
    "AnomalyDetection",
    "ComparativeAnalysis",
    "AnalysisType",
    "TrendDirection",
    "AnomalyType",
    "ComparisonType",
    
    # MongoDB Integration Models
    "ProcessLogModel",
    "LogLevel",
    "EventType",
    "AnnotationType",
    "LogMetadata",
    "AnnotationInfo",
    "RelatedDocument",
    "LogRelationship",
    
    "MachineConfigurationModel",
    "ConfigurationType",
    "ConfigurationFormat",
    "CalibrationData",
    "CalibrationInfo",
    "ConfigurationMetadata",
    
    "RawSensorDataModel",
    "DataFormat",
    "ProcessingStatus",
    "DataQuality",
    "RawSensorCalibrationInfo",
    "MeasurementRange",
    
    "ModelFileModel",
    "ModelType",
    "ModelFormat",
    "GeometryInfo",
    "QualityAssessment",
    "ProcessingInfo",
    "ModelMetadata",
    
    "BuildInstructionModel",
    "InstructionType",
    "MaterialType",
    "SupportType",
    "InstructionMetadata",
    
    "CTScanImageModel",
    "ScanType",
    "ImageFormat",
    "CTQualityMetrics",
    "DefectAnalysis",
    "DimensionalAnalysis",
    "DefectType",
    "DefectSeverity",
    
    "PowderBedImageModel",
    "PowderBedType",
    "PowderAnalysis",
    "PowderCharacteristics",
    "BedQualityMetrics",
    "ImageAnalysis",
    "DefectDetection",
    "PowderBedDefect",
    "DefectLocation",
    "PowderBedDefectType",
    "PowderBedDefectSeverity",
    
    "ProcessImageModel",
    "ProcessStage",
    "ProcessAnalysis",
    "ProcessQualityMetrics",
    "LayerAnalysis",
    "ProcessDefect",
    "ProcessDefectType",
    "ProcessDefectSeverity",
    
    "MachineBuildFileModel",
    "FileType",
    "FileCategory",
    "FileMetadata",
    "UsageAnalytics",
    "FileQualityMetrics",
    
    # Multi-Database Models
    "RedisCacheDataModel",
    "CacheOperation",
    "CacheType",
    "CompressionType",
    "CachePerformanceMetrics",
    "DataType",
    "CacheAnalytics",
    
    "JobQueueDataModel",
    "JobType",
    "JobStatus",
    "ExecutionStatus",
    "QueueAnalytics",
    "ErrorInfo",
    
    "UserSessionDataModel",
    "SessionStatus",
    "SessionType",
    "LoginMethod",
    "SecurityLevel",
    "SessionAnalytics",
    "SecurityInfo",
    
    "CassandraTimeSeriesModel",
    "AggregationType",
    "AggregationWindow",
    "CassandraCalibrationInfo",
    "TimeSeriesAnalytics",
    
    "ISPMMonitoringModel",
    "MonitoringType",
    "ISPMAnomalyType",
    "AnomalySeverity",
    "ISPMEnvironmentalConditions",
    "ISPMProcessParameters",
    "ISPMAnomalyDetection",
    
    # Model Factory
    "ClickHouseModelFactory",
    "ClickHouseModelType",
    "create_clickhouse_model",
    "validate_clickhouse_data",
    "get_clickhouse_model_schema",
    "get_available_clickhouse_models"
]

# ClickHouse Model Registry
CLICKHOUSE_MODEL_REGISTRY = {
    # Core Operational Models
    "pbf_processes": PBFProcessModel,
    "machine_status": MachineStatusModel,
    "sensor_readings": SensorReadingModel,
    "analytics": AnalyticsModel,
    
    # MongoDB Integration Models
    "process_logs": ProcessLogModel,
    "machine_configurations": MachineConfigurationModel,
    "raw_sensor_data": RawSensorDataModel,
    "3d_model_files": ModelFileModel,
    "build_instructions": BuildInstructionModel,
    "ct_scan_images": CTScanImageModel,
    "powder_bed_images": PowderBedImageModel,
    "process_images": ProcessImageModel,
    "machine_build_files": MachineBuildFileModel,
    
    # Multi-Database Models
    "redis_cache_data": RedisCacheDataModel,
    "job_queue_data": JobQueueDataModel,
    "user_session_data": UserSessionDataModel,
    "cassandra_time_series": CassandraTimeSeriesModel,
    "ispm_monitoring": ISPMMonitoringModel
}

# ClickHouse Model Type Mappings
CLICKHOUSE_MODEL_TYPE_MAPPINGS = {
    "PBF_PROCESSES": "pbf_processes",
    "MACHINE_STATUS": "machine_status",
    "SENSOR_READINGS": "sensor_readings",
    "ANALYTICS": "analytics",
    "PROCESS_LOGS": "process_logs",
    "MACHINE_CONFIGURATIONS": "machine_configurations",
    "RAW_SENSOR_DATA": "raw_sensor_data",
    "3D_MODEL_FILES": "3d_model_files",
    "BUILD_INSTRUCTIONS": "build_instructions",
    "CT_SCAN_IMAGES": "ct_scan_images",
    "POWDER_BED_IMAGES": "powder_bed_images",
    "PROCESS_IMAGES": "process_images",
    "MACHINE_BUILD_FILES": "machine_build_files",
    "REDIS_CACHE_DATA": "redis_cache_data",
    "JOB_QUEUE_DATA": "job_queue_data",
    "USER_SESSION_DATA": "user_session_data",
    "CASSANDRA_TIME_SERIES": "cassandra_time_series",
    "ISPM_MONITORING": "ispm_monitoring"
}

def get_clickhouse_model_class(model_type: str):
    """
    Get ClickHouse model class by type string.
    
    Args:
        model_type: Model type string
        
    Returns:
        Model class
    """
    return CLICKHOUSE_MODEL_REGISTRY.get(model_type)

def get_clickhouse_model_type_enum(model_type: str):
    """
    Get ClickHouse model type enum by string.
    
    Args:
        model_type: Model type string
        
    Returns:
        ClickHouseModelType enum value
    """
    return ClickHouseModelType(model_type)

def create_clickhouse_model_instance(model_type: str, data: dict):
    """
    Create a ClickHouse model instance from data.
    
    Args:
        model_type: Model type string
        data: Model data dictionary
        
    Returns:
        Model instance
    """
    model_class = get_clickhouse_model_class(model_type)
    if not model_class:
        raise ValueError(f"Unknown ClickHouse model type: {model_type}")
    
    return model_class(**data)

def validate_clickhouse_model_data(model_type: str, data: dict):
    """
    Validate data against a ClickHouse model schema.
    
    Args:
        model_type: Model type string
        data: Data dictionary to validate
        
    Returns:
        Validation result dictionary
    """
    return ClickHouseModelFactory.validate_data(model_type, data)

def get_clickhouse_model_schema_info(model_type: str):
    """
    Get schema information for a ClickHouse model type.
    
    Args:
        model_type: Model type string
        
    Returns:
        Schema information dictionary
    """
    return ClickHouseModelFactory.get_model_schema(model_type)

def get_all_clickhouse_model_types():
    """
    Get all available ClickHouse model types.
    
    Returns:
        List of model type strings
    """
    return list(CLICKHOUSE_MODEL_REGISTRY.keys())

def get_clickhouse_model_metadata(model_type: str):
    """
    Get metadata for a ClickHouse model type.
    
    Args:
        model_type: Model type string
        
    Returns:
        Model metadata dictionary
    """
    return ClickHouseModelFactory.get_model_metadata(model_type)
