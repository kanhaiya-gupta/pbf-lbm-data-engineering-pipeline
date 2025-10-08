# PBF-LB/M NoSQL Data Warehouse

> **Comprehensive Data Engineering Platform for Powder Bed Fusion - Laser Beam/Metal (PBF-LB/M) Additive Manufacturing Research**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Spark 3.4+](https://img.shields.io/badge/Spark-3.4+-orange.svg)](https://spark.apache.org/)
[![Airflow 3.1+](https://img.shields.io/badge/Airflow-3.1+-green.svg)](https://airflow.apache.org/)
[![MongoDB](https://img.shields.io/badge/MongoDB-7.0+-green.svg)](https://www.mongodb.com/)
[![Redis](https://img.shields.io/badge/Redis-7.0+-red.svg)](https://redis.io/)
[![Cassandra](https://img.shields.io/badge/Cassandra-4.0+-blue.svg)](https://cassandra.apache.org/)
[![Elasticsearch](https://img.shields.io/badge/Elasticsearch-8.0+-yellow.svg)](https://www.elastic.co/elasticsearch/)
[![Neo4j](https://img.shields.io/badge/Neo4j-5.0+-purple.svg)](https://neo4j.com/)

## 🎯 **Overview**

This project provides a complete data pipeline solution for PBF-LB/M research, enabling advanced data analysis, process optimization, and quality assurance through:

- **Multi-Model NoSQL Architecture**: PostgreSQL, MongoDB, Redis, Cassandra, Elasticsearch, Neo4j
- **Advanced Build File Processing**: libSLM/PySLM integration for world-class parsing
- **3D Voxel Visualization**: Spatially-resolved process control and quality analysis
- **Comprehensive Analytics**: Sensitivity analysis, statistical modeling, and ML
- **Virtual Environment**: Virtual testing and simulation capabilities
- **Real-Time Processing**: Streaming data ingestion and processing

## 🏗️ **Architecture**

```mermaid
graph TB
    subgraph "Data Sources"
        A[ISPM Monitoring<br/>Real-time Sensors]
        B[CT Scans<br/>3D Imaging Data]
        C[Build Files<br/>.mtt, .sli, .cli]
        D[Process Logs<br/>Machine Data]
        E[CAD Models<br/>STL, STEP Files]
    end
    
    subgraph "Data Pipeline"
        F[Ingestion<br/>Streaming, Batch, CDC]
        G[Processing<br/>ETL, Analytics, Voxel]
        H[Storage<br/>Multi-Model NoSQL]
        I[Quality<br/>Validation, Monitoring]
        J[Orchestration<br/>Airflow, Scheduling]
    end
    
    subgraph "Advanced Features"
        K[Build File Parsing<br/>libSLM/PySLM Integration]
        L[Voxel Visualization<br/>3D Spatial Analysis]
        M[Analytics Engine<br/>Sensitivity Analysis]
        N[Virtual Environment<br/>Testing & Simulation]
    end
    
    A --> F
    B --> F
    C --> F
    D --> F
    E --> F
    
    F --> G
    G --> H
    H --> I
    I --> J
    
    G --> K
    G --> L
    G --> M
    G --> N
```

## 🔧 **Core Components**

### **1. Data Pipeline**
- **Multi-Source Ingestion**: Streaming (Kafka), Batch (ETL), CDC (Change Data Capture)
- **Real-Time Processing**: Apache Flink for streaming, Apache Spark for batch
- **Multi-Model Storage**: Optimized database selection based on data characteristics
- **Quality Management**: Comprehensive validation, monitoring, and remediation
- **Workflow Orchestration**: Apache Airflow DAGs for complex workflows

### **2. Build File Processing**
- **libSLM Integration**: C++ library for parsing .mtt, .sli, .cli, .rea, .slm files
- **PySLM Integration**: Python library for advanced analysis and visualization
- **10 Specialized Extractors**: Power, velocity, path, energy, layer, timestamp, focus, jump, style, geometry
- **Per-Geometry Parameters**: Laser parameters defined for individual scan paths
- **CT-Build Correlation**: Temporal correlation for defect analysis

### **3. Voxel Visualization**
- **3D Voxel Grid**: Spatially-resolved representation of PBF-LB/M components
- **Multi-Modal Fusion**: Integration of CAD, process, ISPM, and CT data
- **Interactive 3D Rendering**: Real-time visualization and navigation
- **Defect Detection**: AI-powered 3D defect detection and classification
- **Porosity Analysis**: Comprehensive porosity characterization

### **4. Analytics Engine**
- **Sensitivity Analysis**: Sobol indices, Morris screening, design of experiments
- **Statistical Analysis**: Multivariate, time series, regression, nonparametric methods
- **Process Analysis**: Parameter optimization, quality prediction, sensor analysis
- **ML Integration**: Random forest, neural networks, Bayesian analysis

### **5. Virtual Environment**
- **VM Management**: Virtual machine orchestration and provisioning
- **Simulation Engines**: Thermal, fluid, mechanical, multi-physics simulation
- **Digital Twin**: Real-time synchronization and prediction
- **Testing Frameworks**: Experimental design, automated testing, validation
- **Cloud Integration**: AWS, Azure, GCP with distributed computing

## 🚀 **Key Features**

### **Advanced Build File Processing**
- **libSLM/PySLM Integration**: World-class parsing of .mtt, .sli, .cli, .rea, .slm files
- **10 Specialized Extractors**: Power, velocity, path, energy, layer, timestamp, focus, jump, style, geometry
- **Per-Geometry Parameters**: Laser parameters defined for individual scan paths
- **CT-Build Correlation**: Temporal correlation for defect analysis

### **3D Voxel Visualization**
- **Spatial Resolution**: Voxel-level analysis and process control
- **Multi-Modal Fusion**: Integration of CAD, process, ISPM, and CT data
- **Interactive 3D Rendering**: Real-time visualization and navigation
- **Defect Detection**: AI-powered 3D defect detection and classification

### **Comprehensive Analytics**
- **Sensitivity Analysis**: Sobol indices, Morris screening, design of experiments
- **Statistical Analysis**: Multivariate, time series, regression, nonparametric methods
- **Process Analysis**: Parameter optimization, quality prediction, sensor analysis
- **ML Integration**: Random forest, neural networks, Bayesian analysis

### **Virtual Environment**
- **VM Management**: Virtual machine orchestration and provisioning
- **Simulation Engines**: Thermal, fluid, mechanical, multi-physics simulation
- **Digital Twin**: Real-time synchronization and prediction
- **Testing Frameworks**: Experimental design, automated testing, validation

## 🛠️ Technology Stack

### **Data Processing**
- **Apache Spark**: Distributed data processing and ETL
- **Apache Kafka**: Real-time data streaming
- **Apache Airflow**: Workflow orchestration
- **DBT**: Data transformation and modeling

### **🗄️ Multi-Model Data Storage**

#### **🏠 Local Storage (On-Premises)**
- **PostgreSQL**: Primary operational database, real-time data
- **MongoDB**: Document storage for unstructured data, metadata
- **Redis**: High-performance caching layer, session management
- **MinIO**: Local object storage (S3-compatible), raw data backup, development datasets

#### **☁️ Cloud Storage (AWS/Azure/GCP)**
- **Snowflake**: Large-scale analytics, data warehousing, business intelligence
- **AWS S3**: Scalable data lake, long-term storage, data archiving
- **BigQuery**: Ad-hoc queries, data exploration, research analytics
- **MongoDB Atlas**: Managed document storage, global distribution

#### **🤖 ML Research & Advanced Analytics**
- **Training Data**: Stored in both local (fast access) and cloud (scalability)
- **Research Data**: Cloud storage for collaboration and sharing
- **Analytics**: Data warehouse for complex queries and business intelligence
- **Data Lake**: Raw data storage for exploration and experimentation

### **Infrastructure**
- **Docker**: Containerization
- **Kubernetes**: Container orchestration
- **Terraform**: Infrastructure as code
- **Prometheus**: Monitoring and alerting
- **Grafana**: Visualization and dashboards

### **Development**
- **Python 3.9+**: Primary programming language
- **PySpark**: Spark Python API
- **FastAPI**: API development
- **Pydantic**: Data validation and serialization
- **Pytest**: Testing framework

## 📁 **Project Structure**

```
pbf-lbm-nosql-data-warehouse/
├── src/                        # Source code
│   ├── data_pipeline/          # Main data pipeline
│   ├── core/                   # Core domain entities
│   └── ml/                     # Machine learning models
├── config/                     # Configuration files
├── docs/                       # Documentation
├── docker/                     # Docker configurations
├── requirements/               # Python dependencies
└── roadmaps/                   # Project roadmap
```

*Detailed project structure available in [docs/project-structure.md](docs/project-structure.md)*

## 📊 **Data Flow & Storage Strategy**

### **Where Your Data Goes**

```mermaid
graph TB
    subgraph "📊 Data Sources"
        ISPM[ISPM Sensors<br/>📡 Real-time]
        CT[CT Scans<br/>🔬 Batch]
        BUILD[Build Files<br/>🏗️ Batch]
        CAD[CAD Models<br/>📐 Batch]
    end

    subgraph "⚙️ Processing"
        KAFKA[Kafka Streaming]
        SPARK[Spark Processing]
        AIRFLOW[Airflow Orchestration]
    end

    subgraph "🏠 Local Storage"
        POSTGRES[(PostgreSQL<br/>🗄️ Operational)]
        MONGODB[(MongoDB<br/>🍃 Documents)]
        REDIS[(Redis<br/>⚡ Cache)]
        MINIO[(MinIO<br/>📦 Object Storage)]
    end

    subgraph "☁️ Cloud Storage"
        SNOWFLAKE[(Snowflake<br/>❄️ Analytics)]
        AWS_S3[(AWS S3<br/>☁️ Data Lake)]
        BIGQUERY[(BigQuery<br/>🔍 Research)]
    end

    subgraph "🤖 ML & Research"
        ML_TRAINING[ML Model Training]
        ADVANCED_ANALYTICS[Advanced Analytics]
        RESEARCH[Research Queries]
    end

    %% Data Flow
    ISPM --> KAFKA --> POSTGRES
    CT --> SPARK --> MONGODB
    BUILD --> SPARK --> SNOWFLAKE
    CAD --> SPARK --> MINIO

    %% ML and Analytics Usage
    POSTGRES --> ML_TRAINING
    SNOWFLAKE --> ADVANCED_ANALYTICS
    AWS_S3 --> RESEARCH
    BIGQUERY --> RESEARCH
```

### **Data Storage Strategy**

- **Real-time Data** → **Local Storage** (PostgreSQL, Redis) for immediate access
- **Batch Data** → **Cloud Storage** (Snowflake, AWS S3) for analytics and research
- **ML Training** → **Both Local & Cloud** for optimal performance and scalability
- **Research Data** → **Cloud Storage** for collaboration and sharing

## 🔧 **Installation & Setup**

### **Prerequisites**
- Python 3.8+
- Apache Spark 3.4+
- Apache Airflow 3.1+
- PostgreSQL 13+
- Docker and Docker Compose

### **Quick Start**

1. **Clone the repository:**
```bash
git clone <repository-url>
cd pbf-lbm-nosql-data-warehouse
```

2. **Install core dependencies:**
```bash
pip install -r requirements/requirements_core.txt
pip install -r requirements/requirements_airflow.txt
pip install -r requirements/requirements_ml.txt
```

3. **Set up external libraries:**
```bash
# Install libSLM (C++ library with Python bindings)
cd src/data_pipeline/external/libSLM
mkdir build && cd build
cmake ..
make -j4
make install

# Install PySLM (Python library)
cd src/data_pipeline/external/pyslm
pip install -e .
```

4. **Start the system:**
```bash
docker-compose -f docker/docker-compose.dev.yml up -d
python scripts/init_database.py
python scripts/start_pipeline.py
```

## 📊 **Data Flow**

```mermaid
sequenceDiagram
    participant ISPM as ISPM Sensors
    participant Kafka as Kafka Stream
    participant Parser as Build Parser
    participant Voxel as Voxel Processor
    participant Analytics as Analytics Engine
    participant Storage as Multi-Model Storage
    
    ISPM->>Kafka: Real-time Data
    Kafka->>Parser: Stream Processing
    Parser->>Voxel: Process Parameters
    Voxel->>Analytics: Voxel Data
    Analytics->>Storage: Analysis Results
    Storage->>Voxel: Historical Data
    Voxel->>Analytics: Enhanced Analysis
```

## 🔬 **Research Applications**

### **Process Optimization**
- Parameter sensitivity analysis and optimization
- Quality prediction modeling
- Defect root cause analysis

### **Quality Assurance**
- Real-time quality monitoring
- Automated defect detection
- Porosity analysis and characterization

### **Virtual Testing**
- Controlled parameter experiments
- Multi-physics simulation
- Digital twin validation

## 🎯 **Key Benefits**

1. **World-Class Build File Processing**: Leverages libSLM/PySLM for maximum reliability
2. **Spatial Resolution**: Voxel-level analysis and process control
3. **Multi-Modal Integration**: Unified representation of diverse data sources
4. **Advanced Analytics**: Sophisticated sensitivity analysis and ML capabilities
5. **Virtual Testing**: Controlled experiments without physical resources
6. **Real-Time Processing**: Low-latency data processing and analysis
7. **Scalable Architecture**: Horizontal scaling for growing data volumes
8. **Research-Ready**: Built specifically for additive manufacturing research

## 🤝 **Contributing**

We welcome contributions from the research community! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📚 **Documentation**

Comprehensive documentation is available in the `docs/` directory:

- **[System Architecture](docs/architecture/system-overview.md)**: Complete system architecture and design principles
- **[Build File Parser](docs/build-parsing/build-file-parser.md)**: Advanced build file processing with libSLM/PySLM
- **[Sensitivity Analysis](docs/analytics/sensitivity-analysis.md)**: Comprehensive analytics and statistical analysis
- **[Voxel Visualization](docs/visualization/voxel-visualization.md)**: 3D voxel-based visualization and analysis
- **[Virtual Environment](docs/virtual-environment/virtual-environment.md)**: Virtual testing and simulation capabilities

## 🗺️ **Roadmap**

See our [Project Roadmap](roadmaps/README.md) for planned features and development phases.

- **Phase 1**: PBF Data Pipeline Optimization
- **Phase 2**: NoSQL Database Integration  
- **Phase 3**: ML/AI Integration

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 **Contact**

- **Issues**: [GitHub Issues](https://github.com/your-username/pbf-lbm-nosql-data-warehouse/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/pbf-lbm-nosql-data-warehouse/discussions)

---

**Built for PBF-LB/M Research Excellence** 🚀
