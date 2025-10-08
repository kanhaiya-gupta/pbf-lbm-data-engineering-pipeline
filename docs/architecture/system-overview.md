# System Architecture Overview

## 🎯 **PBF-LB/M NoSQL Data Warehouse Architecture**

The PBF-LB/M (Powder Bed Fusion - Laser Beam/Metal) NoSQL Data Warehouse is a comprehensive data engineering platform designed for additive manufacturing research. It provides end-to-end data processing capabilities from raw sensor data to advanced analytics and visualization.

## 🏗️ **Core Architecture Principles**

### **1. Multi-Model Data Architecture**
The system employs a multi-model NoSQL approach, using different database technologies optimized for specific data types:

- **PostgreSQL**: Primary operational database for structured data
- **MongoDB**: Document store for flexible PBF process data
- **Redis**: In-memory cache for real-time data access
- **Cassandra**: Time-series data for ISPM monitoring
- **Elasticsearch**: Full-text search and analytics
- **Neo4j**: Graph database for process relationships

### **2. Domain-Driven Design**
The architecture follows Domain-Driven Design principles with clear separation of concerns:

```mermaid
graph TB
    subgraph "Core Domain"
        A[Domain Entities<br/>PBFProcess, CTScan, ISPMData]
        B[Value Objects<br/>ProcessParameters, QualityMetrics]
        C[Domain Events<br/>ProcessStarted, QualityAssessed]
        D[Enums<br/>ProcessStatus, QualityTier]
    end
    
    subgraph "Application Layer"
        E[Services<br/>ProcessService, QualityService]
        F[Repositories<br/>ProcessRepository, QualityRepository]
        G[Interfaces<br/>External Services]
    end
    
    subgraph "Infrastructure"
        H[Data Pipeline<br/>Ingestion, Processing, Storage]
        I[External Systems<br/>ISPM, CT Scanners, Machines]
    end
    
    A --> E
    B --> E
    C --> E
    D --> E
    E --> F
    E --> G
    F --> H
    G --> I
```

### **3. Event-Driven Architecture**
The system uses event-driven patterns for loose coupling and scalability:

```mermaid
sequenceDiagram
    participant ISPM as ISPM System
    participant Kafka as Kafka Stream
    participant Processor as Stream Processor
    participant Storage as Multi-Model Storage
    participant Analytics as Analytics Engine
    
    ISPM->>Kafka: Process Data Event
    Kafka->>Processor: Stream Processing
    Processor->>Storage: Store Processed Data
    Storage->>Analytics: Trigger Analysis
    Analytics->>Storage: Store Results
```

## 📊 **Data Flow Architecture**

### **1. Data Ingestion Layer**

```mermaid
graph LR
    subgraph "Data Sources"
        A[ISPM Monitoring<br/>Real-time Sensors]
        B[CT Scans<br/>3D Imaging Data]
        C[Build Files<br/>.mtt, .sli, .cli]
        D[Process Logs<br/>Machine Data]
        E[CAD Models<br/>STL, STEP Files]
    end
    
    subgraph "Ingestion Methods"
        F[Streaming<br/>Kafka, Flink]
        G[Batch<br/>Scheduled ETL]
        H[CDC<br/>Change Data Capture]
        I[File Upload<br/>Direct Processing]
    end
    
    A --> F
    B --> G
    C --> I
    D --> H
    E --> I
    
    F --> J[Data Pipeline]
    G --> J
    H --> J
    I --> J
```

### **2. Data Processing Pipeline**

```mermaid
graph TB
    subgraph "Processing Stages"
        A[Raw Data Ingestion]
        B[Data Validation & Quality]
        C[Data Transformation]
        D[Multi-Model Storage]
        E[Analytics & ML]
        F[Visualization & Export]
    end
    
    A --> B
    B --> C
    C --> D
    D --> E
    E --> F
    
    subgraph "Processing Types"
        G[Real-time Streaming<br/>Apache Flink]
        H[Batch Processing<br/>Apache Spark]
        I[Build File Parsing<br/>libSLM/PySLM]
        J[Voxel Processing<br/>3D Spatial Analysis]
    end
    
    B --> G
    B --> H
    B --> I
    B --> J
```

## 🔧 **Core Components**

### **1. Data Pipeline Module**

The main data pipeline orchestrates all data processing activities:

```python
# Core data pipeline components
from src.data_pipeline import (
    # Configuration
    config_manager,
    get_pipeline_settings,
    get_etl_config,
    
    # Ingestion
    KafkaProducer,
    KafkaConsumer,
    ISPMStreamProcessor,
    CTDataIngester,
    
    # Processing
    ETLOrchestrator,
    extract_from_mongodb,
    transform_document_data,
    load_to_cassandra,
    
    # Storage
    MongoDBClient,
    RedisClient,
    CassandraClient,
    ElasticsearchClient,
    Neo4jClient,
    
    # Quality
    DataQualityService,
    AnomalyDetector,
    RemediationService,
    
    # Orchestration
    PBFProcessDAG,
    AirflowClient,
    JobScheduler
)
```

### **2. Build File Processing**

Advanced build file parsing leveraging libSLM and PySLM:

```python
# Build file processing components
from src.data_pipeline.processing.build_parsing import (
    # Core parsing
    BuildFileParser,
    FormatDetector,
    MetadataExtractor,
    
    # Format-specific parsers
    EOSParser,      # EOS machines (.sli, .cli)
    MTTParser,      # MTT format
    RealizerParser, # Realizer format
    SLMParser,      # SLM Solutions format
    
    # Data extractors (10 specialized extractors)
    PowerExtractor,           # Laser power analysis
    VelocityExtractor,        # Scan velocity analysis
    PathExtractor,           # Scan path geometry
    EnergyExtractor,         # Energy consumption
    LayerExtractor,          # Layer-specific data
    TimestampExtractor,      # Timing data
    LaserFocusExtractor,     # Laser focus analysis
    JumpParametersExtractor, # Jump speed/delay
    BuildStyleExtractor,     # Build style metadata
    GeometryTypeExtractor    # Geometry distribution
)
```

### **3. Analytics Engine**

Comprehensive analytics and sensitivity analysis:

```python
# Analytics components
from src.data_pipeline.processing.analytics import (
    # Sensitivity Analysis
    GlobalSensitivityAnalyzer,
    SobolAnalyzer,
    MorrisAnalyzer,
    UncertaintyQuantifier,
    
    # Statistical Analysis
    MultivariateAnalyzer,
    TimeSeriesAnalyzer,
    RegressionAnalyzer,
    
    # Process Analysis
    ParameterAnalyzer,
    QualityAnalyzer,
    ProcessOptimizer,
    
    # Reporting
    AnalysisReportGenerator,
    SensitivityVisualizer
)
```

### **4. Voxel Visualization**

3D voxel-based visualization and analysis:

```python
# Voxel visualization components
from src.data_pipeline.visualization.voxel_clients import (
    # Core voxel processing
    CADVoxelizer,
    VoxelGrid,
    MultiModalFusion,
    VoxelRenderer,
    
    # Analysis tools
    SpatialQualityAnalyzer,
    DefectDetector3D,
    PorosityAnalyzer,
    
    # Interaction
    VoxelController,
    VoxelExporter
)
```

### **5. Virtual Environment**

Virtual testing and simulation capabilities:

```python
# Virtual environment components
from src.data_pipeline.virtual_environment import (
    # VM Management
    VMOrchestrator,
    VMProvisioner,
    
    # Simulation Engines
    ThermalSimulator,
    FluidDynamicsSimulator,
    MultiPhysicsSimulator,
    
    # Digital Twin
    DigitalTwinModel,
    TwinSynchronizer,
    TwinPredictor,
    
    # Testing Frameworks
    VirtualExperimentDesigner,
    AutomatedTestRunner,
    
    # Cloud Integration
    CloudProviderManager,
    DistributedComputingManager
)
```

## 🗄️ **Storage Architecture**

### **Multi-Model Database Strategy**

```mermaid
graph TB
    subgraph "Data Types"
        A[Process Parameters<br/>Structured Data]
        B[ISPM Monitoring<br/>Time-Series Data]
        C[CT Scan Data<br/>Large Binary Data]
        D[Research Metadata<br/>Document Data]
        E[Process Relationships<br/>Graph Data]
        F[Search & Analytics<br/>Full-Text Search]
    end
    
    subgraph "Database Selection"
        G[PostgreSQL<br/>Primary Operational]
        H[MongoDB<br/>Document Store]
        I[Redis<br/>Real-time Cache]
        J[Cassandra<br/>Time-Series]
        K[Neo4j<br/>Graph Database]
        L[Elasticsearch<br/>Search Engine]
    end
    
    A --> G
    B --> J
    C --> H
    D --> H
    E --> K
    F --> L
    
    G --> M[Data Pipeline]
    H --> M
    I --> M
    J --> M
    K --> M
    L --> M
```

### **Data Routing Strategy**

The system automatically routes data to the most appropriate database based on data characteristics:

- **Structured Process Data** → PostgreSQL
- **Time-Series Sensor Data** → Cassandra
- **Large Binary Data (CT Scans)** → MongoDB
- **Real-time Cache Data** → Redis
- **Relationship Data** → Neo4j
- **Searchable Content** → Elasticsearch

## 🔄 **Processing Workflows**

### **1. Real-Time Processing Workflow**

```mermaid
sequenceDiagram
    participant ISPM as ISPM Sensors
    participant Kafka as Kafka Stream
    participant Flink as Flink Processor
    participant Redis as Redis Cache
    participant Cassandra as Cassandra
    participant Monitor as Quality Monitor
    
    ISPM->>Kafka: Real-time Data
    Kafka->>Flink: Stream Processing
    Flink->>Redis: Cache Results
    Flink->>Cassandra: Store Time-Series
    Flink->>Monitor: Trigger Quality Check
    Monitor->>Redis: Update Quality Status
```

### **2. Build File Processing Workflow**

```mermaid
sequenceDiagram
    participant Upload as File Upload
    participant Parser as Build Parser
    participant Extractors as Data Extractors
    participant Storage as Multi-Model Storage
    participant Voxel as Voxel Processor
    
    Upload->>Parser: Build File (.mtt, .sli, .cli)
    Parser->>Extractors: Parse with libSLM
    Extractors->>Storage: Extract Parameters
    Storage->>Voxel: Create Voxel Grid
    Voxel->>Storage: Store Voxel Data
```

### **3. Analytics Workflow**

```mermaid
sequenceDiagram
    participant Data as Stored Data
    participant Analytics as Analytics Engine
    participant Sensitivity as Sensitivity Analysis
    participant ML as ML Models
    participant Reports as Report Generator
    
    Data->>Analytics: Process Data
    Analytics->>Sensitivity: Sensitivity Analysis
    Sensitivity->>ML: Train Models
    ML->>Reports: Generate Reports
    Reports->>Data: Store Results
```

## 🚀 **Scalability & Performance**

### **Horizontal Scaling**

- **Kafka Clusters**: Multiple brokers for high-throughput streaming
- **Spark Clusters**: Distributed processing for large datasets
- **Database Sharding**: Horizontal partitioning across multiple nodes
- **Container Orchestration**: Kubernetes for dynamic scaling

### **Performance Optimization**

- **Multi-Level Caching**: Redis for hot data, application-level caching
- **Parallel Processing**: Concurrent data processing pipelines
- **Index Optimization**: Database-specific indexing strategies
- **Query Optimization**: Optimized queries for each database type

## 🔒 **Security & Compliance**

### **Data Security**

- **Encryption**: End-to-end encryption for sensitive data
- **Access Control**: Role-based access control (RBAC)
- **Audit Logging**: Comprehensive audit trails
- **Data Privacy**: GDPR compliance and data anonymization

### **System Security**

- **Network Security**: Secure network architecture
- **Container Security**: Secure container deployment
- **API Security**: OAuth 2.0 and JWT authentication
- **Monitoring**: Security monitoring and alerting

## 📈 **Monitoring & Observability**

### **System Monitoring**

- **Application Performance**: APM with DataDog
- **Infrastructure Monitoring**: Prometheus and Grafana
- **Log Aggregation**: Centralized logging system
- **Distributed Tracing**: Jaeger for request tracing

### **Data Quality Monitoring**

- **Real-time Quality Checks**: Automated data validation
- **Quality Dashboards**: Visual quality metrics
- **Anomaly Detection**: ML-based anomaly detection
- **Alerting**: Proactive quality alerts

## 🎯 **Key Benefits**

1. **Comprehensive Data Integration**: Unified platform for all PBF-LB/M data types
2. **Advanced Analytics**: Sophisticated sensitivity analysis and ML capabilities
3. **Real-Time Processing**: Low-latency data processing and analysis
4. **Scalable Architecture**: Horizontal scaling for growing data volumes
5. **Research-Ready**: Built specifically for additive manufacturing research
6. **Quality Assurance**: Comprehensive data quality management
7. **Visualization**: Advanced 3D voxel visualization capabilities
8. **Virtual Testing**: Virtual environment for controlled experiments

This architecture provides a robust, scalable, and comprehensive platform for PBF-LB/M additive manufacturing research, enabling advanced data analysis, process optimization, and quality assurance.
