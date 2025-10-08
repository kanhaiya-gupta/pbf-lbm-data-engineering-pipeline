"""
ETL Configuration

This module provides ETL-specific configuration for PBF-LB/M data processing.
It handles Spark configuration, ETL job settings, and data source configurations.
"""

import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass


@dataclass
class SparkConfig:
    """Spark configuration for PBF-LB/M ETL processing"""
    
    # Spark application settings
    app_name: str = "PBF-LB/M-ETL"
    master: str = "local[*]"
    deploy_mode: str = "client"
    
    # Memory settings
    driver_memory: str = "2g"
    executor_memory: str = "2g"
    driver_max_result_size: str = "1g"
    
    # Core settings
    executor_cores: int = 2
    max_result_size: str = "1g"
    
    # Serialization settings
    serializer: str = "org.apache.spark.serializer.KryoSerializer"
    kryo_registration_required: bool = True
    
    # SQL settings
    sql_adaptive_enabled: bool = True
    sql_adaptive_coalesce_partitions_enabled: bool = True
    sql_adaptive_skew_join_enabled: bool = True
    
    # Dynamic allocation
    dynamic_allocation_enabled: bool = True
    dynamic_allocation_min_executors: int = 1
    dynamic_allocation_max_executors: int = 10
    
    def to_dict(self) -> Dict[str, str]:
        """Convert Spark configuration to dictionary"""
        return {
            "spark.app.name": self.app_name,
            "spark.master": self.master,
            "spark.submit.deployMode": self.deploy_mode,
            "spark.driver.memory": self.driver_memory,
            "spark.executor.memory": self.executor_memory,
            "spark.driver.maxResultSize": self.driver_max_result_size,
            "spark.executor.cores": str(self.executor_cores),
            "spark.sql.adaptive.enabled": str(self.sql_adaptive_enabled).lower(),
            "spark.sql.adaptive.coalescePartitions.enabled": str(self.sql_adaptive_coalesce_partitions_enabled).lower(),
            "spark.sql.adaptive.skewJoin.enabled": str(self.sql_adaptive_skew_join_enabled).lower(),
            "spark.dynamicAllocation.enabled": str(self.dynamic_allocation_enabled).lower(),
            "spark.dynamicAllocation.minExecutors": str(self.dynamic_allocation_min_executors),
            "spark.dynamicAllocation.maxExecutors": str(self.dynamic_allocation_max_executors),
            "spark.serializer": self.serializer,
            "spark.kryo.registrationRequired": str(self.kryo_registration_required).lower()
        }


@dataclass
class ETLJobConfig:
    """ETL job configuration for PBF-LB/M processing"""
    
    # Job settings
    job_name: str
    job_type: str  # "extract", "transform", "load", "full_etl"
    enabled: bool = True
    
    # Processing settings
    batch_size: int = 1000
    max_retries: int = 3
    retry_delay: int = 5
    timeout: int = 300
    
    # Data source settings
    source_type: str = "kafka"  # "kafka", "s3", "database", "file"
    source_config: Dict[str, Any] = None
    
    # Data sink settings
    sink_type: str = "database"  # "database", "s3", "kafka", "file"
    sink_config: Dict[str, Any] = None
    
    # Transformation settings
    transformation_type: str = "standard"  # "standard", "custom", "ml"
    transformation_config: Dict[str, Any] = None
    
    def __post_init__(self):
        """Initialize default configurations if not provided"""
        if self.source_config is None:
            self.source_config = {}
        if self.sink_config is None:
            self.sink_config = {}
        if self.transformation_config is None:
            self.transformation_config = {}


@dataclass
class DataSourceConfig:
    """Data source configuration for PBF-LB/M data"""
    
    # Source identification
    source_name: str
    source_type: str  # "pbf_process", "ispm_monitoring", "ct_scan", "powder_bed"
    
    # Connection settings
    connection_string: str = ""
    username: str = ""
    password: str = ""
    
    # Data settings
    table_name: str = ""
    schema_name: str = ""
    partition_columns: List[str] = None
    
    # Processing settings
    batch_size: int = 1000
    parallel_reads: int = 4
    read_timeout: int = 300
    
    # Filtering settings
    filter_conditions: Dict[str, Any] = None
    date_range: Dict[str, str] = None  # {"start": "2024-01-01", "end": "2024-12-31"}
    
    def __post_init__(self):
        """Initialize default values if not provided"""
        if self.partition_columns is None:
            self.partition_columns = []
        if self.filter_conditions is None:
            self.filter_conditions = {}
        if self.date_range is None:
            self.date_range = {}


class ETLConfig:
    """ETL configuration manager for PBF-LB/M data processing"""
    
    def __init__(self):
        self.spark_config = SparkConfig()
        self.etl_jobs: Dict[str, ETLJobConfig] = {}
        self.data_sources: Dict[str, DataSourceConfig] = {}
        self._load_default_configurations()
    
    def _load_default_configurations(self):
        """Load default ETL configurations"""
        # Default Spark configuration
        self.spark_config = SparkConfig()
        
        # Default ETL jobs
        self.etl_jobs = {
            "pbf_process_etl": ETLJobConfig(
                job_name="PBF Process ETL",
                job_type="full_etl",
                source_type="kafka",
                sink_type="database",
                transformation_type="standard"
            ),
            "ispm_monitoring_etl": ETLJobConfig(
                job_name="ISPM Monitoring ETL",
                job_type="full_etl",
                source_type="kafka",
                sink_type="database",
                transformation_type="standard"
            ),
            "ct_scan_etl": ETLJobConfig(
                job_name="CT Scan ETL",
                job_type="full_etl",
                source_type="s3",
                sink_type="database",
                transformation_type="custom"
            ),
            "powder_bed_etl": ETLJobConfig(
                job_name="Powder Bed ETL",
                job_type="full_etl",
                source_type="kafka",
                sink_type="database",
                transformation_type="standard"
            )
        }
        
        # Default data sources
        self.data_sources = {
            "pbf_process": DataSourceConfig(
                source_name="PBF Process Data",
                source_type="pbf_process",
                table_name="pbf_process_data",
                schema_name="lpbf_research"
            ),
            "ispm_monitoring": DataSourceConfig(
                source_name="ISPM Monitoring Data",
                source_type="ispm_monitoring",
                table_name="ispm_monitoring_data",
                schema_name="lpbf_research"
            ),
            "ct_scan": DataSourceConfig(
                source_name="CT Scan Data",
                source_type="ct_scan",
                table_name="ct_scan_data",
                schema_name="lpbf_research"
            ),
            "powder_bed": DataSourceConfig(
                source_name="Powder Bed Data",
                source_type="powder_bed",
                table_name="powder_bed_data",
                schema_name="lpbf_research"
            )
        }
    
    @classmethod
    def from_environment(cls) -> 'ETLConfig':
        """Create ETL configuration from environment variables"""
        config = cls()
        
        # Update Spark configuration from environment
        config.spark_config.app_name = os.getenv("SPARK_APP_NAME", "PBF-LB/M-ETL")
        config.spark_config.master = os.getenv("SPARK_MASTER", "local[*]")
        config.spark_config.driver_memory = os.getenv("SPARK_DRIVER_MEMORY", "2g")
        config.spark_config.executor_memory = os.getenv("SPARK_EXECUTOR_MEMORY", "2g")
        config.spark_config.executor_cores = int(os.getenv("SPARK_EXECUTOR_CORES", "2"))
        
        return config
    
    def get_spark_config(self) -> SparkConfig:
        """Get Spark configuration"""
        return self.spark_config
    
    def get_etl_settings(self) -> Dict[str, Any]:
        """Get ETL job settings"""
        return {
            "jobs": {name: {
                "job_name": job.job_name,
                "job_type": job.job_type,
                "enabled": job.enabled,
                "batch_size": job.batch_size,
                "max_retries": job.max_retries,
                "retry_delay": job.retry_delay,
                "timeout": job.timeout,
                "source_type": job.source_type,
                "sink_type": job.sink_type,
                "transformation_type": job.transformation_type
            } for name, job in self.etl_jobs.items()}
        }
    
    def get_data_source_config(self, source_name: str) -> Optional[DataSourceConfig]:
        """Get data source configuration by name"""
        return self.data_sources.get(source_name)
    
    def get_all_data_sources(self) -> Dict[str, DataSourceConfig]:
        """Get all data source configurations"""
        return self.data_sources.copy()
    
    def add_etl_job(self, job_name: str, job_config: ETLJobConfig) -> None:
        """Add a new ETL job configuration"""
        self.etl_jobs[job_name] = job_config
    
    def add_data_source(self, source_name: str, source_config: DataSourceConfig) -> None:
        """Add a new data source configuration"""
        self.data_sources[source_name] = source_config
    
    def update_spark_config(self, **kwargs) -> None:
        """Update Spark configuration"""
        for key, value in kwargs.items():
            if hasattr(self.spark_config, key):
                setattr(self.spark_config, key, value)


# Global configuration instance
_etl_config: Optional[ETLConfig] = None


def get_etl_config() -> ETLConfig:
    """
    Get the global ETL configuration instance.
    
    Returns:
        ETLConfig: The global ETL configuration
    """
    global _etl_config
    if _etl_config is None:
        _etl_config = ETLConfig.from_environment()
    return _etl_config


def set_etl_config(config: ETLConfig) -> None:
    """
    Set the global ETL configuration instance.
    
    Args:
        config: The ETL configuration to set
    """
    global _etl_config
    _etl_config = config


def reset_etl_config() -> None:
    """Reset the global ETL configuration to None."""
    global _etl_config
    _etl_config = None


def load_etl_config(config_path: Optional[str] = None) -> ETLConfig:
    """
    Load ETL configuration from file or environment.
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        ETLConfig: Loaded ETL configuration
    """
    # For now, just return from environment
    # TODO: Add file-based configuration loading
    return ETLConfig.from_environment()