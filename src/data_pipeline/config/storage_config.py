"""
Storage Configuration

This module provides storage configuration for PBF-LB/M data storage.
It handles S3, Snowflake, PostgreSQL, and Delta Lake configurations.
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class S3Config:
    """S3 configuration for PBF-LB/M data lake storage"""
    
    # Connection settings
    endpoint_url: str = "http://localhost:9000"
    aws_access_key_id: str = "minioadmin"
    aws_secret_access_key: str = "minioadmin"
    region_name: str = "us-east-1"
    
    # Bucket settings
    bucket_name: str = "pbf-lbm-data-lake"
    prefix: str = "lpbf-research/"
    
    # Performance settings
    multipart_threshold: int = 64 * 1024 * 1024  # 64MB
    multipart_chunksize: int = 16 * 1024 * 1024  # 16MB
    max_concurrency: int = 10
    
    # Retention settings
    retention_days: int = 365
    lifecycle_rules: Dict[str, Any] = None
    
    def __post_init__(self):
        """Initialize default lifecycle rules if not provided"""
        if self.lifecycle_rules is None:
            self.lifecycle_rules = {
                "raw_data": {
                    "prefix": "raw/",
                    "retention_days": 365,
                    "transition_to_ia_days": 30,
                    "transition_to_glacier_days": 90
                },
                "processed_data": {
                    "prefix": "processed/",
                    "retention_days": 180,
                    "transition_to_ia_days": 7,
                    "transition_to_glacier_days": 30
                },
                "archived_data": {
                    "prefix": "archived/",
                    "retention_days": 2555,  # 7 years
                    "transition_to_glacier_days": 0
                }
            }


@dataclass
class SnowflakeConfig:
    """Snowflake configuration for PBF-LB/M data warehouse"""
    
    # Connection settings
    account: str = "your-account.snowflakecomputing.com"
    user: str = "your-username"
    password: str = "your-password"
    warehouse: str = "COMPUTE_WH"
    database: str = "LPBF_RESEARCH"
    schema: str = "PUBLIC"
    role: str = "ACCOUNTADMIN"
    
    # Performance settings
    warehouse_size: str = "SMALL"
    auto_suspend: int = 60
    auto_resume: bool = True
    
    # Query settings
    query_timeout: int = 300
    max_retries: int = 3
    retry_delay: int = 5
    
    # Data loading settings
    copy_options: Dict[str, Any] = None
    
    def __post_init__(self):
        """Initialize default copy options if not provided"""
        if self.copy_options is None:
            self.copy_options = {
                "FILE_FORMAT": "PARQUET",
                "ON_ERROR": "CONTINUE",
                "PURGE": True,
                "RETURN_FAILED_ONLY": True
            }


@dataclass
class PostgresConfig:
    """PostgreSQL configuration for PBF-LB/M operational storage"""
    
    # Connection settings
    host: str = "localhost"
    port: int = 5432
    database: str = "lpbf_research"
    username: str = "postgres"
    password: str = "password"
    
    # Connection pool settings
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    pool_recycle: int = 3600
    
    # Performance settings
    statement_timeout: int = 300
    query_timeout: int = 300
    max_retries: int = 3
    retry_delay: int = 5
    
    # Schema settings
    default_schema: str = "public"
    search_path: str = "public"
    
    def get_connection_string(self) -> str:
        """Get PostgreSQL connection string"""
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"


@dataclass
class DeltaLakeConfig:
    """Delta Lake configuration for PBF-LB/M data lake"""
    
    # Storage settings
    storage_path: str = "s3a://pbf-lbm-data-lake/delta/"
    checkpoint_path: str = "s3a://pbf-lbm-data-lake/checkpoints/"
    
    # Performance settings
    auto_optimize: bool = True
    optimize_interval: int = 24  # hours
    z_order_columns: Dict[str, list] = None
    
    # Retention settings
    retention_period: int = 30  # days
    vacuum_interval: int = 7  # days
    
    # Schema evolution
    allow_schema_evolution: bool = True
    merge_schema: bool = True
    
    def __post_init__(self):
        """Initialize default z-order columns if not provided"""
        if self.z_order_columns is None:
            self.z_order_columns = {
                "pbf_process_data": ["process_id", "timestamp"],
                "ispm_monitoring_data": ["sensor_id", "timestamp"],
                "ct_scan_data": ["scan_id", "timestamp"],
                "powder_bed_data": ["camera_id", "timestamp"]
            }


class StorageConfig:
    """Storage configuration manager for PBF-LB/M data storage"""
    
    def __init__(self):
        self.s3_config = S3Config()
        self.snowflake_config = SnowflakeConfig()
        self.postgres_config = PostgresConfig()
        self.delta_lake_config = DeltaLakeConfig()
        self._load_default_configurations()
    
    def _load_default_configurations(self):
        """Load default storage configurations"""
        # Default S3 configuration
        self.s3_config = S3Config()
        
        # Default Snowflake configuration
        self.snowflake_config = SnowflakeConfig()
        
        # Default PostgreSQL configuration
        self.postgres_config = PostgresConfig()
        
        # Default Delta Lake configuration
        self.delta_lake_config = DeltaLakeConfig()
    
    @classmethod
    def from_environment(cls) -> 'StorageConfig':
        """Create storage configuration from environment variables"""
        config = cls()
        
        # Update S3 configuration from environment
        config.s3_config.endpoint_url = os.getenv("S3_ENDPOINT_URL", "http://localhost:9000")
        config.s3_config.aws_access_key_id = os.getenv("S3_ACCESS_KEY_ID", "minioadmin")
        config.s3_config.aws_secret_access_key = os.getenv("S3_SECRET_ACCESS_KEY", "minioadmin")
        config.s3_config.bucket_name = os.getenv("S3_BUCKET_NAME", "pbf-lbm-data-lake")
        
        # Update Snowflake configuration from environment
        config.snowflake_config.account = os.getenv("SNOWFLAKE_ACCOUNT", "your-account.snowflakecomputing.com")
        config.snowflake_config.user = os.getenv("SNOWFLAKE_USER", "your-username")
        config.snowflake_config.password = os.getenv("SNOWFLAKE_PASSWORD", "your-password")
        config.snowflake_config.warehouse = os.getenv("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH")
        config.snowflake_config.database = os.getenv("SNOWFLAKE_DATABASE", "LPBF_RESEARCH")
        
        # Update PostgreSQL configuration from environment
        config.postgres_config.host = os.getenv("POSTGRES_HOST", "localhost")
        config.postgres_config.port = int(os.getenv("POSTGRES_PORT", "5432"))
        config.postgres_config.database = os.getenv("POSTGRES_DATABASE", "lpbf_research")
        config.postgres_config.username = os.getenv("POSTGRES_USERNAME", "postgres")
        config.postgres_config.password = os.getenv("POSTGRES_PASSWORD", "password")
        
        return config
    
    def get_s3_config(self) -> S3Config:
        """Get S3 configuration"""
        return self.s3_config
    
    def get_snowflake_config(self) -> SnowflakeConfig:
        """Get Snowflake configuration"""
        return self.snowflake_config
    
    def get_postgres_config(self) -> PostgresConfig:
        """Get PostgreSQL configuration"""
        return self.postgres_config
    
    def get_delta_lake_config(self) -> DeltaLakeConfig:
        """Get Delta Lake configuration"""
        return self.delta_lake_config
    
    def get_storage_settings(self) -> Dict[str, Any]:
        """Get all storage settings"""
        return {
            "s3": {
                "endpoint_url": self.s3_config.endpoint_url,
                "bucket_name": self.s3_config.bucket_name,
                "prefix": self.s3_config.prefix,
                "retention_days": self.s3_config.retention_days
            },
            "snowflake": {
                "account": self.snowflake_config.account,
                "warehouse": self.snowflake_config.warehouse,
                "database": self.snowflake_config.database,
                "schema": self.snowflake_config.schema,
                "warehouse_size": self.snowflake_config.warehouse_size
            },
            "postgres": {
                "host": self.postgres_config.host,
                "port": self.postgres_config.port,
                "database": self.postgres_config.database,
                "pool_size": self.postgres_config.pool_size,
                "max_overflow": self.postgres_config.max_overflow
            },
            "delta_lake": {
                "storage_path": self.delta_lake_config.storage_path,
                "checkpoint_path": self.delta_lake_config.checkpoint_path,
                "auto_optimize": self.delta_lake_config.auto_optimize,
                "retention_period": self.delta_lake_config.retention_period
            }
        }
    
    def update_s3_config(self, **kwargs) -> None:
        """Update S3 configuration"""
        for key, value in kwargs.items():
            if hasattr(self.s3_config, key):
                setattr(self.s3_config, key, value)
    
    def update_snowflake_config(self, **kwargs) -> None:
        """Update Snowflake configuration"""
        for key, value in kwargs.items():
            if hasattr(self.snowflake_config, key):
                setattr(self.snowflake_config, key, value)
    
    def update_postgres_config(self, **kwargs) -> None:
        """Update PostgreSQL configuration"""
        for key, value in kwargs.items():
            if hasattr(self.postgres_config, key):
                setattr(self.postgres_config, key, value)
    
    def update_delta_lake_config(self, **kwargs) -> None:
        """Update Delta Lake configuration"""
        for key, value in kwargs.items():
            if hasattr(self.delta_lake_config, key):
                setattr(self.delta_lake_config, key, value)


# Global configuration instance
_storage_config: Optional[StorageConfig] = None


def get_storage_config() -> StorageConfig:
    """
    Get the global storage configuration instance.
    
    Returns:
        StorageConfig: The global storage configuration
    """
    global _storage_config
    if _storage_config is None:
        _storage_config = StorageConfig.from_environment()
    return _storage_config


def get_s3_config() -> S3Config:
    """
    Get S3 configuration from the global storage config.
    
    Returns:
        S3Config: The S3 configuration
    """
    return get_storage_config().get_s3_config()


def get_postgres_config() -> PostgresConfig:
    """
    Get PostgreSQL configuration from the global storage config.
    
    Returns:
        PostgresConfig: The PostgreSQL configuration
    """
    return get_storage_config().get_postgres_config()


def get_delta_lake_config() -> DeltaLakeConfig:
    """
    Get Delta Lake configuration from the global storage config.
    
    Returns:
        DeltaLakeConfig: The Delta Lake configuration
    """
    return get_storage_config().get_delta_lake_config()


def get_snowflake_config() -> SnowflakeConfig:
    """
    Get Snowflake configuration from the global storage config.
    
    Returns:
        SnowflakeConfig: The Snowflake configuration
    """
    return get_storage_config().get_snowflake_config()


def set_storage_config(config: StorageConfig) -> None:
    """
    Set the global storage configuration instance.
    
    Args:
        config: The storage configuration to set
    """
    global _storage_config
    _storage_config = config


def reset_storage_config() -> None:
    """Reset the global storage configuration to None."""
    global _storage_config
    _storage_config = None


def load_storage_config(config_path: Optional[str] = None) -> StorageConfig:
    """
    Load storage configuration from file or environment.
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        StorageConfig: Loaded storage configuration
    """
    # For now, just return from environment
    # TODO: Add file-based configuration loading
    return StorageConfig.from_environment()