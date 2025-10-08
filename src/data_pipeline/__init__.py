"""
PBF-LB/M Data Pipeline Module

This module provides a comprehensive data pipeline solution for Powder Bed Fusion - Laser Beam/Metal
additive manufacturing research. It includes ingestion, processing, storage, quality, and orchestration
components for handling PBF-LB/M data from various sources including ISPM monitoring, CT scans,
powder bed analysis, and process parameters.

Key Features:
- Multi-source data ingestion (streaming, batch, CDC)
- Real-time and batch processing with Apache Spark
- Multi-model storage (PostgreSQL, Snowflake, S3, Delta Lake)
- Comprehensive data quality management
- Workflow orchestration with Apache Airflow
- Schema management and evolution
- Incremental processing and backfill capabilities
"""

from .config import (
    config_manager,
    get_pipeline_settings,
    get_etl_config,
    get_spark_config,
    get_kafka_config,
    get_s3_config,
    get_quality_rules,
    PipelineSettings
)
from .ingestion import (
    # Streaming
    KafkaProducer,
    KafkaConsumer,
    ISPMStreamProcessor,
    PowderBedStreamProcessor,
    MessageSerializer,
    # Batch
    CTDataIngester,
    ISPMDataIngester,
    MachineDataIngester,
    S3Ingester,
    DatabaseIngester,
    # CDC
    PostgresCDC,
    KafkaCDCConnector,
    ChangeEventProcessor,
    ConflictResolver
)
from .processing import (
    # ETL
    ETLOrchestrator,
    DatabaseIntegration,
    # NoSQL ETL functions
    extract_from_mongodb,
    extract_from_redis,
    extract_from_cassandra,
    extract_from_elasticsearch,
    extract_from_nosql_source,
    transform_document_data,
    transform_key_value_data,
    transform_columnar_data,
    transform_graph_data,
    transform_multi_model_data,
    load_to_mongodb,
    load_to_redis,
    load_to_cassandra,
    load_to_elasticsearch,
    load_to_neo4j,
    load_to_nosql_destination,
    # Streaming
    KafkaStreamsProcessor,
    FlinkProcessor,
    RealTimeTransformer,
    ISPMStreamJoins,
    PowderBedStreamJoins,
    StreamSinkManager,
    SinkConfig,
    SinkType,
    # Incremental
    CDCProcessor,
    WatermarkManager,
    DeltaProcessor,
    BackfillProcessor,
    # Schema
    SchemaRegistry,
    SchemaValidator,
    SchemaEvolver,
    MultiModelManager,
    DataModelType,
    SchemaFormat
)
from .storage import (
    # Data Lake
    S3Client,
    DataArchiver,
    DeltaLakeManager,
    ParquetManager,
    MongoDBClient,
    # Data Warehouse
    SnowflakeClient,
    QueryExecutor,
    TableManager,
    WarehouseOptimizer,
    ElasticsearchClient,
    # Operational
    PostgresClient,
    ConnectionPool,
    TransactionManager,
    RedisClient,
    CassandraClient,
    Neo4jClient
)
from .quality import (
    # Validation
    DataQualityService,
    SchemaValidator as QualitySchemaValidator,
    BusinessRuleValidator,
    DataTypeValidator,
    QualityValidator,
    AnomalyDetector,
    DefectAnalyzer,
    SurfaceQualityAnalyzer,
    # Remediation
    RemediationService,
    RemediationConfig,
    create_remediation_service,
    auto_remediate_data,
    # Monitoring (commented out - may develop in future)
    # QualityMonitor,
    # QualityScorer,
    # TrendAnalyzer,
    # QualityDashboardGenerator
)
from .orchestration import (
    # Airflow
    PBFProcessDAG,
    ISPMMonitoringDAG,
    CTScanDAG,
    PowderBedDAG,
    DataQualityDAG,
    DBTDAG,
    AirflowClient,
    SparkAirflowIntegration,
    # Scheduling
    JobScheduler,
    DependencyManager,
    ResourceAllocator,
    PriorityManager,
    # Monitoring
    PipelineMonitor,
    JobMonitor,
    PerformanceMonitor,
    AlertManager
)

__version__ = "1.0.0"
__author__ = "PBF-LB/M Data Pipeline Team"
__description__ = "Comprehensive data pipeline for PBF-LB/M additive manufacturing research"

__all__ = [
    # Configuration
    "config_manager",
    "get_pipeline_settings",
    "get_etl_config",
    "get_spark_config",
    "get_kafka_config",
    "get_s3_config",
    "get_quality_rules",
    "PipelineSettings",
    
    # Ingestion
    "KafkaProducer",
    "KafkaConsumer",
    "ISPMStreamProcessor",
    "PowderBedStreamProcessor",
    "MessageSerializer",
    "CTDataIngester",
    "ISPMDataIngester",
    "MachineDataIngester",
    "S3Ingester",
    "DatabaseIngester",
    "PostgresCDC",
    "KafkaCDCConnector",
    "ChangeEventProcessor",
    "ConflictResolver",
    
    # Processing
    "ETLOrchestrator",
    "DatabaseIntegration",
    # NoSQL ETL functions
    "extract_from_mongodb",
    "extract_from_redis",
    "extract_from_cassandra",
    "extract_from_elasticsearch",
    "extract_from_nosql_source",
    "transform_document_data",
    "transform_key_value_data",
    "transform_columnar_data",
    "transform_graph_data",
    "transform_multi_model_data",
    "load_to_mongodb",
    "load_to_redis",
    "load_to_cassandra",
    "load_to_elasticsearch",
    "load_to_neo4j",
    "load_to_nosql_destination",
    "KafkaStreamsProcessor",
    "FlinkProcessor",
    "RealTimeTransformer",
    "ISPMStreamJoins",
    "PowderBedStreamJoins",
    "StreamSinkManager",
    "SinkConfig",
    "SinkType",
    "CDCProcessor",
    "WatermarkManager",
    "DeltaProcessor",
    "BackfillProcessor",
    "SchemaRegistry",
    "SchemaValidator",
    "SchemaEvolver",
    "MultiModelManager",
    "DataModelType",
    "SchemaFormat",
    
    # Storage
    "S3Client",
    "DataArchiver",
    "DeltaLakeManager",
    "ParquetManager",
    "MongoDBClient",
    "SnowflakeClient",
    "QueryExecutor",
    "TableManager",
    "WarehouseOptimizer",
    "ElasticsearchClient",
    "PostgresClient",
    "ConnectionPool",
    "TransactionManager",
    "RedisClient",
    "CassandraClient",
    "Neo4jClient",
    
    # Quality
    "DataQualityService",
    "QualitySchemaValidator",
    "BusinessRuleValidator",
    "DataTypeValidator",
    "QualityValidator",
    "AnomalyDetector",
    "DefectAnalyzer",
    "SurfaceQualityAnalyzer",
    "RemediationService",
    "RemediationConfig",
    "create_remediation_service",
    "auto_remediate_data",
    "RemediationEngine",
    "RemediationAction",
    "RemediationResult",
    "DataCleanser",
    "QualityRouter",
    "DeadLetterQueue",
    # Quality monitoring
    "QualityMonitor",
    "QualityScorer",
    "TrendAnalyzer",
    "QualityDashboardGenerator",
    
    # Orchestration
    "PBFProcessDAG",
    "ISPMMonitoringDAG",
    "CTScanDAG",
    "PowderBedDAG",
    "DataQualityDAG",
    "DBTDAG",
    "AirflowClient",
    "SparkAirflowIntegration",
    "JobScheduler",
    "DependencyManager",
    "ResourceAllocator",
    "PriorityManager",
    "PipelineMonitor",
    "JobMonitor",
    "PerformanceMonitor",
    "AlertManager"
]

# External Software Dependencies
from . import external

# Optional Virtual Environment Module - Lazy import to prevent memory issues
try:
    from . import virtual_environment
    # Virtual environment module is available
    _VE_AVAILABLE = True
except ImportError as e:
    import warnings
    warnings.warn(f"Virtual environment module not available: {e}")
    _VE_AVAILABLE = False
    virtual_environment = None
