# Project Structure - PBF-LB/M NoSQL Data Warehouse

This document provides a comprehensive overview of the project structure, explaining the organization of directories, files, and their purposes within the PBF-LB/M NoSQL Data Warehouse.

## 📁 **Root Directory Structure**

```
pbf-lbm-nosql-data-warehouse/
├── src/                        # Source code
├── config/                     # Configuration files
├── docs/                       # Documentation
├── docker/                     # Docker configurations
├── requirements/               # Python dependencies
├── roadmaps/                   # Project roadmap and phases
├── help_project/               # Reference implementation
├── README.md                   # Project overview
└── .gitignore                  # Git ignore rules
```

## 🔧 **Source Code (`src/`)**

### **Core Domain (`src/core/`)**
```
src/core/
├── __init__.py
├── domain/                     # Domain-driven design entities
│   ├── entities/               # Core business entities
│   │   ├── base_entity.py      # Base entity class
│   │   ├── pbf_process.py      # PBF process entity
│   │   ├── ispm_monitoring.py  # ISPM monitoring entity
│   │   ├── ct_scan.py          # CT scan entity
│   │   └── powder_bed.py       # Powder bed entity
│   ├── value_objects/          # Domain value objects
│   │   ├── process_parameters.py
│   │   ├── quality_metrics.py
│   │   ├── voxel_coordinates.py
│   │   └── defect_classification.py
│   ├── events/                 # Domain events
│   │   ├── pbf_process_events.py
│   │   ├── ispm_monitoring_events.py
│   │   ├── ct_scan_events.py
│   │   └── powder_bed_events.py
│   └── enums/                  # Domain enumerations
│       ├── process_status.py
│       ├── quality_tier.py
│       └── defect_type.py
├── interfaces/                 # Repository and service interfaces
│   ├── repositories/           # Repository interfaces
│   └── external/               # External service interfaces
├── monitoring/                 # Shared monitoring utilities
│   ├── metrics/                # Metrics collection
│   ├── tracing/                # Distributed tracing
│   ├── dashboards/             # Monitoring dashboards
│   └── apm/                    # Application Performance Monitoring
└── exceptions/                 # Shared exceptions
    └── domain_exceptions.py
```

### **Data Pipeline (`src/data_pipeline/`)**
```
src/data_pipeline/
├── __init__.py
├── config/                     # Configuration management
│   ├── config_manager.py       # Central configuration manager
│   ├── pipeline_config.py      # Pipeline settings
│   ├── etl_config.py           # ETL configuration
│   ├── streaming_config.py     # Streaming configuration
│   ├── storage_config.py       # Storage configuration
│   ├── quality_config.py       # Quality configuration
│   └── orchestration_config.py # Orchestration configuration
├── ingestion/                  # Data ingestion layer
│   ├── streaming/              # Real-time data ingestion
│   │   ├── kafka_producer.py   # Kafka message producer
│   │   ├── kafka_consumer.py   # Kafka message consumer
│   │   ├── kafka_ingester.py   # Kafka data ingester
│   │   ├── ispm_stream_processor.py
│   │   ├── powder_bed_stream_processor.py
│   │   └── message_serializer.py
│   ├── batch/                  # Batch data ingestion
│   │   ├── ct_data_ingester.py
│   │   ├── ispm_data_ingester.py
│   │   ├── machine_data_ingester.py
│   │   ├── s3_ingester.py
│   │   ├── database_ingester.py
│   │   └── file_ingester.py
│   └── cdc/                    # Change Data Capture
│       ├── postgres_cdc.py
│       ├── kafka_cdc_connector.py
│       ├── change_event_processor.py
│       └── conflict_resolver.py
├── processing/                 # Data processing layer
│   ├── etl/                    # ETL operations
│   │   ├── extract.py          # Data extraction
│   │   ├── transform.py        # Data transformation
│   │   ├── load.py             # Data loading
│   │   └── database_integration.py
│   ├── streaming/              # Stream processing
│   │   ├── kafka_streams_processor.py
│   │   ├── flink_processor.py
│   │   ├── streaming_processor.py
│   │   ├── real_time_transformer.py
│   │   ├── ispm_stream_joins.py
│   │   └── powder_bed_stream_joins.py
│   ├── incremental/            # Incremental processing
│   │   ├── cdc_processor.py
│   │   ├── watermark_manager.py
│   │   ├── delta_processor.py
│   │   └── backfill_processor.py
│   ├── schema/                 # Schema management
│   │   ├── schema_registry.py
│   │   ├── schema_validator.py
│   │   ├── schema_evolver.py
│   │   └── multi_model_manager.py
│   ├── dbt/                    # DBT transformations
│   │   └── dbt_orchestrator.py
│   ├── analytics/              # Analytics and ML
│   │   ├── sensitivity_analysis/
│   │   │   ├── global_analysis.py
│   │   │   ├── local_analysis.py
│   │   │   ├── doe.py
│   │   │   └── uncertainty.py
│   │   ├── statistical_analysis/
│   │   │   ├── multivariate.py
│   │   │   ├── time_series.py
│   │   │   ├── regression.py
│   │   │   └── nonparametric.py
│   │   ├── process_analysis/
│   │   │   ├── parameter_analysis.py
│   │   │   ├── quality_analysis.py
│   │   │   ├── sensor_analysis.py
│   │   │   └── optimization.py
│   │   └── reporting/
│   │       ├── report_generators.py
│   │       ├── visualization.py
│   │       └── documentation.py
│   ├── build_parsing/          # Build file processing
│   │   ├── base_parser.py      # Abstract base parser
│   │   ├── core/               # Core parsing components
│   │   │   ├── build_file_parser.py
│   │   │   ├── format_detector.py
│   │   │   └── metadata_extractor.py
│   │   ├── format_parsers/     # Format-specific parsers
│   │   │   ├── eos_parser.py   # EOS format (.sli, .cli)
│   │   │   ├── mtt_parser.py   # MTT format (.mtt)
│   │   │   ├── realizer_parser.py # Realizer format (.rea)
│   │   │   ├── slm_parser.py   # SLM format (.slm)
│   │   │   └── generic_parser.py # Generic fallback
│   │   ├── data_extractors/    # Data extraction components
│   │   │   ├── power_extractor.py
│   │   │   ├── velocity_extractor.py
│   │   │   ├── path_extractor.py
│   │   │   ├── energy_extractor.py
│   │   │   ├── layer_extractor.py
│   │   │   ├── timestamp_extractor.py
│   │   │   ├── laser_focus_extractor.py
│   │   │   ├── jump_parameters_extractor.py
│   │   │   ├── build_style_extractor.py
│   │   │   └── geometry_type_extractor.py
│   │   └── utils/              # Utility functions
│   │       ├── file_utils.py
│   │       └── validation_utils.py
│   └── external/               # External library integration
│       ├── libSLM/             # libSLM C++ library
│       └── pyslm/              # PySLM Python library
├── storage/                    # Data storage layer
│   ├── data_lake/              # Data lake storage
│   │   ├── s3_client.py
│   │   ├── data_archiver.py
│   │   ├── delta_lake_manager.py
│   │   ├── parquet_manager.py
│   │   └── mongodb_client.py
│   ├── data_warehouse/         # Data warehouse storage
│   │   ├── snowflake_client.py
│   │   ├── query_executor.py
│   │   ├── table_manager.py
│   │   ├── warehouse_optimizer.py
│   │   └── elasticsearch_client.py
│   └── operational/            # Operational storage
│       ├── postgres_client.py
│       ├── connection_pool.py
│       ├── transaction_manager.py
│       ├── redis_client.py
│       ├── cassandra_client.py
│       └── neo4j_client.py
├── quality/                    # Data quality layer
│   ├── validation/             # Data validation
│   │   ├── data_quality_service.py
│   │   ├── schema_validator.py
│   │   ├── business_rule_validator.py
│   │   ├── data_type_validator.py
│   │   ├── quality_validator.py
│   │   ├── anomaly_detector.py
│   │   ├── defect_analyzer.py
│   │   └── surface_quality_analyzer.py
│   ├── monitoring/             # Quality monitoring
│   │   ├── quality_monitor.py
│   │   ├── quality_scorer.py
│   │   ├── trend_analyzer.py
│   │   └── quality_dashboard_generator.py
│   └── remediation/            # Quality remediation
│       ├── remediation_service.py
│       ├── remediation_engine.py
│       ├── data_cleanser.py
│       ├── quality_router.py
│       └── dead_letter_queue.py
├── orchestration/              # Workflow orchestration
│   ├── airflow/                # Apache Airflow DAGs
│   │   ├── pbf_process_dag.py
│   │   ├── ispm_monitoring_dag.py
│   │   ├── ct_scan_dag.py
│   │   ├── powder_bed_dag.py
│   │   ├── data_quality_dag.py
│   │   ├── dbt_dag.py
│   │   ├── airflow_client.py
│   │   ├── spark_airflow_integration.py
│   │   └── email_notifications.py
│   ├── scheduling/             # Job scheduling
│   │   ├── job_scheduler.py
│   │   ├── dependency_manager.py
│   │   ├── resource_allocator.py
│   │   └── priority_manager.py
│   └── monitoring/             # Pipeline monitoring
│       ├── pipeline_monitor.py
│       ├── job_monitor.py
│       ├── performance_monitor.py
│       └── alert_manager.py
├── visualization/              # Visualization components
│   └── voxel_clients/          # Voxel visualization
│       ├── core/               # Core voxel components
│       │   ├── cad_voxelizer.py
│       │   ├── multi_modal_fusion.py
│       │   ├── voxel_process_controller.py
│       │   ├── voxel_renderer.py
│       │   └── voxel_loader.py
│       ├── analysis/           # Analysis components
│       │   ├── spatial_quality_analyzer.py
│       │   ├── defect_detector_3d.py
│       │   └── porosity_analyzer.py
│       ├── interaction/        # User interaction
│       │   └── voxel_controller.py
│       └── export/             # Data export
│           └── voxel_exporter.py
└── virtual_environment/        # Virtual environment
    ├── vm_management/          # VM management
    │   ├── orchestration.py
    │   ├── provisioning.py
    │   ├── storage.py
    │   └── security.py
    ├── simulation_engines/     # Simulation engines
    │   ├── thermal_simulation.py
    │   ├── fluid_dynamics.py
    │   ├── mechanical_simulation.py
    │   ├── material_physics.py
    │   └── multi_physics.py
    ├── digital_twin/           # Digital twin
    │   ├── twin_models.py
    │   ├── synchronization.py
    │   ├── prediction.py
    │   └── validation.py
    ├── testing_frameworks/     # Testing frameworks
    │   ├── experiment_design.py
    │   ├── automated_testing.py
    │   ├── validation.py
    │   └── reporting.py
    └── cloud_integration/      # Cloud integration
        ├── cloud_providers.py
        ├── distributed_computing.py
        ├── containerization.py
        └── serverless.py
```

## ⚙️ **Configuration (`config/`)**

```
config/
├── data_pipeline/              # Data pipeline configurations
│   ├── etl/                    # ETL configurations
│   │   ├── data_sources/       # Data source configurations
│   │   ├── etl_jobs.yaml
│   │   ├── nosql_etl_config.yaml
│   │   └── spark_config.yaml
│   ├── orchestration/          # Orchestration configurations
│   │   ├── airflow_config.yaml
│   │   ├── monitoring_config.yaml
│   │   ├── nosql_orchestration.yaml
│   │   └── scheduling_config.yaml
│   ├── pipeline/               # Pipeline configurations
│   │   ├── environments/       # Environment-specific configs
│   │   ├── feature_flags.yaml
│   │   ├── multi_model_config.yaml
│   │   └── pipeline_settings.yaml
│   ├── quality/                # Quality configurations
│   │   ├── nosql_quality_config.yaml
│   │   ├── quality_rules.yaml
│   │   ├── remediation_config.yaml
│   │   └── sla_settings.yaml
│   ├── schemas/                # Schema configurations
│   │   └── nosql_schemas.yaml
│   ├── storage/                # Storage configurations
│   │   ├── delta_lake_config.yaml
│   │   ├── nosql_storage_config.yaml
│   │   ├── postgres_config.yaml
│   │   ├── s3_config.yaml
│   │   └── snowflake_config.yaml
│   └── streaming/              # Streaming configurations
│       ├── flink_config.yaml
│       ├── kafka_config.yaml
│       ├── nosql_streaming_config.yaml
│       └── streaming_jobs.yaml
└── ml/                         # Machine learning configurations
    ├── environments/            # ML environment configs
    ├── evidently/              # Evidently AI configurations
    ├── feast/                  # Feast feature store configs
    ├── features/               # Feature definitions
    ├── global/                 # Global ML configurations
    ├── mlflow/                 # MLflow configurations
    ├── models/                 # Model configurations
    ├── monitoring/             # ML monitoring configurations
    ├── pipelines/              # ML pipeline configurations
    └── serving/                # Model serving configurations
```

## 📚 **Documentation (`docs/`)**

```
docs/
├── README.md                   # Documentation overview
├── project-structure.md        # This file - detailed project structure
├── architecture/               # Architecture documentation
│   └── system-overview.md      # System architecture overview
├── build-parsing/              # Build file parsing documentation
│   └── build-file-parser.md    # Build file parser details
├── analytics/                  # Analytics documentation
│   └── sensitivity-analysis.md # Sensitivity analysis details
├── visualization/              # Visualization documentation
│   └── voxel-visualization.md  # Voxel visualization details
└── virtual-environment/        # Virtual environment documentation
    └── virtual-environment.md  # Virtual environment details
```

## 🐳 **Docker (`docker/`)**

```
docker/
├── docker-compose.dev.yml      # Development environment
├── docker-compose.prod.yml     # Production environment
├── Dockerfile.airflow          # Airflow container
├── Dockerfile.api              # API container
├── Dockerfile.dbt              # DBT container
├── Dockerfile.ml               # ML container
├── Dockerfile.spark            # Spark container
└── Dockerfile.worker           # Worker container
```

## 📦 **Dependencies (`requirements/`)**

```
requirements/
├── requirements_airflow_client.txt
├── requirements_airflow.txt
├── requirements_cloud.txt
├── requirements_core.txt
├── requirements_dbt.txt
├── requirements_ml.txt
├── requirements_monitoring.txt
├── requirements_quality.txt
├── requirements_spark.txt
└── requirements_streaming.txt
```

## 🗺️ **Roadmaps (`roadmaps/`)**

```
roadmaps/
└── phases/                     # Development phases
    ├── phase-1-pbf-data-pipeline-optimization/
    ├── phase-2-nosql-database-integration/
    └── phase-3-ml-ai-integration/
```

## 🔧 **External Libraries (`src/data_pipeline/external/`)**

```
src/data_pipeline/external/
├── libSLM/                     # libSLM C++ library
│   ├── CMakeLists.txt
│   ├── README.md
│   ├── src/                    # C++ source code
│   ├── python/                 # Python bindings
│   └── build/                  # Build artifacts
└── pyslm/                      # PySLM Python library
    ├── setup.py
    ├── README.md
    ├── pyslm/                  # Python source code
    └── examples/               # Usage examples
```

## 📋 **Key Design Principles**

### **1. Domain-Driven Design (DDD)**
- **Core Domain**: Business logic in `src/core/domain/`
- **Entities**: Core business objects
- **Value Objects**: Immutable domain concepts
- **Events**: Domain events for loose coupling

### **2. Clean Architecture**
- **Separation of Concerns**: Clear boundaries between layers
- **Dependency Inversion**: High-level modules don't depend on low-level modules
- **Interface Segregation**: Small, focused interfaces

### **3. Multi-Model Data Architecture**
- **Right Tool for Right Job**: Different databases for different data types
- **Data Routing**: Automatic routing based on data characteristics
- **Consistency**: Eventual consistency where appropriate

### **4. Modular Design**
- **Loose Coupling**: Modules can be developed independently
- **High Cohesion**: Related functionality grouped together
- **Extensibility**: Easy to add new features and components

## 🚀 **Getting Started with the Codebase**

### **1. Start with Core Domain**
```bash
# Understand the business domain
src/core/domain/entities/
src/core/domain/value_objects/
src/core/domain/events/
```

### **2. Explore Data Pipeline**
```bash
# Understand data flow
src/data_pipeline/ingestion/
src/data_pipeline/processing/
src/data_pipeline/storage/
```

### **3. Check Configuration**
```bash
# Understand system configuration
config/data_pipeline/
config/ml/
```

### **4. Review Documentation**
```bash
# Read comprehensive documentation
docs/architecture/system-overview.md
docs/build-parsing/build-file-parser.md
```

## 🔍 **Finding Specific Functionality**

### **Build File Processing**
- **Main Parser**: `src/data_pipeline/processing/build_parsing/core/build_file_parser.py`
- **Format Parsers**: `src/data_pipeline/processing/build_parsing/format_parsers/`
- **Data Extractors**: `src/data_pipeline/processing/build_parsing/data_extractors/`

### **Analytics and ML**
- **Sensitivity Analysis**: `src/data_pipeline/processing/analytics/sensitivity_analysis/`
- **Statistical Analysis**: `src/data_pipeline/processing/analytics/statistical_analysis/`
- **Process Analysis**: `src/data_pipeline/processing/analytics/process_analysis/`

### **Voxel Visualization**
- **Core Components**: `src/data_pipeline/visualization/voxel_clients/core/`
- **Analysis Tools**: `src/data_pipeline/visualization/voxel_clients/analysis/`
- **User Interface**: `src/data_pipeline/visualization/voxel_clients/interaction/`

### **Virtual Environment**
- **VM Management**: `src/data_pipeline/virtual_environment/vm_management/`
- **Simulation Engines**: `src/data_pipeline/virtual_environment/simulation_engines/`
- **Digital Twin**: `src/data_pipeline/virtual_environment/digital_twin/`

This project structure follows industry best practices for large-scale data engineering projects, ensuring maintainability, scalability, and extensibility for PBF-LB/M additive manufacturing research.
