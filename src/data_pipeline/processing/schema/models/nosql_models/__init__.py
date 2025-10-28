"""
NoSQL Models Module

This module contains NoSQL-specific Pydantic models for the PBF-LB/M data pipeline,
supporting MongoDB, Cassandra, Redis, and Elasticsearch with PostgreSQL relationships.

Key Features:
- Multi-database support (MongoDB, Cassandra, Redis, Elasticsearch)
- PostgreSQL relationship maintenance
- Flexible document schemas
- Advanced indexing and validation
- Time-series optimization
- Cache management
- Search and analytics
"""

from .mongodb_document_models import (
    # Base models
    BaseMongoDBDocument,
    FileStatus,
    
    # Specific document models that match JSON schemas exactly
    ProcessImageDocument,
    CTScanImageDocument,
    PowderBedImageDocument,
    MachineBuildFileDocument,
    Model3DFileDocument,
    RawSensorDataDocument,
    ProcessLogDocument,
    BuildInstructionDocument,
    MachineConfigDocument
)

from .cassandra_time_series_models import (
    # Base models
    BaseTimeSeriesModel,
    SensorType,
    ProcessStatus,
    MachineStatus,
    AlertSeverity,
    
    # Time-series models
    SensorReading,
    ProcessMonitoring,
    MachineStatusUpdate,
    AlertEvent,
    AnalyticsAggregation
)

from .redis_cache_models import (
    # Base models
    BaseRedisCache,
    CacheStatus,
    
    # Cache models
    ProcessDataCache,
    SensorReadingCache,
    MachineStatusCache,
    AnalyticsCache,
    JobQueueCache,
    UserSessionCache
)

from .elasticsearch_document_models import (
    # Base models
    BaseElasticsearchDocument,
    ProcessStatus,
    MachineStatus,
    SensorType,
    MaterialType,
    QualityStatus,
    AlertSeverity,
    
    # Document models that match Elasticsearch schemas exactly
    PBFProcessDocument,
    SensorReadingsDocument,
    QualityMetricsDocument,
    MachineStatusDocument,
    BuildInstructionsDocument,
    AnalyticsDocument,
    SearchLogsDocument,
    ElasticsearchDocumentFactory
)

__all__ = [
    # MongoDB models
    "BaseMongoDBDocument",
    "FileStatus",
    "ProcessImageDocument",
    "CTScanImageDocument", 
    "PowderBedImageDocument",
    "MachineBuildFileDocument",
    "Model3DFileDocument",
    "RawSensorDataDocument",
    "ProcessLogDocument",
    "BuildInstructionDocument",
    "MachineConfigDocument",
    
    # Cassandra models
    "BaseTimeSeriesModel",
    "SensorType",
    "ProcessStatus",
    "MachineStatus", 
    "AlertSeverity",
    "SensorReading",
    "ProcessMonitoring",
    "MachineStatusUpdate",
    "AlertEvent",
    "AnalyticsAggregation",
    
    # Redis models
    "BaseRedisCache",
    "CacheStatus",
    "ProcessDataCache",
    "SensorReadingCache",
    "MachineStatusCache",
    "AnalyticsCache",
    "JobQueueCache",
    "UserSessionCache",
    
    # Elasticsearch models
    "BaseElasticsearchDocument",
    "PBFProcessDocument",
    "SensorReadingsDocument",
    "QualityMetricsDocument",
    "MachineStatusDocument",
    "BuildInstructionsDocument",
    "AnalyticsDocument",
    "SearchLogsDocument",
    "ElasticsearchDocumentFactory"
]