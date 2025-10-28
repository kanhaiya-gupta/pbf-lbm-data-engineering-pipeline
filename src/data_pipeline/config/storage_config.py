"""
Storage Configuration for PBF-LB/M Data Pipeline

This module provides storage configuration management with environment variable support,
connection pooling, and production-ready settings for S3, Snowflake, and Delta Lake.

Features:
- Environment variable configuration
- Connection pooling and timeout settings
- SSL/TLS support for production
- Health check configuration
- Performance optimization settings
- Lifecycle management for data retention
"""

import os
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class S3Config(BaseModel):
    """S3 configuration model for PBF-LB/M data lake storage."""
    
    # Connection settings
    endpoint_url: str = Field(default="http://localhost:9000", description="S3 endpoint URL")
    aws_access_key_id: str = Field(default="minioadmin", description="AWS access key ID")
    aws_secret_access_key: str = Field(default="minioadmin", description="AWS secret access key")
    region_name: str = Field(default="us-east-1", description="AWS region name")
    
    # Security settings
    use_ssl: bool = Field(default=True, description="Use SSL/TLS for connections")
    verify_ssl: bool = Field(default=True, description="Verify SSL certificates")
    
    # Bucket settings
    bucket_name: str = Field(default="pbf-lbm-data-lake", description="S3 bucket name")
    prefix: str = Field(default="lpbf-research/", description="S3 object prefix")
    
    # Performance settings
    multipart_threshold: int = Field(default=67108864, description="Multipart upload threshold (64MB)")
    multipart_chunksize: int = Field(default=16777216, description="Multipart chunk size (16MB)")
    max_concurrency: int = Field(default=10, description="Maximum concurrent uploads")
    max_connections: int = Field(default=50, description="Maximum connections to S3")
    timeout: int = Field(default=30, description="Connection timeout in seconds")
    retry_attempts: int = Field(default=3, description="Number of retry attempts")
    
    # Retention settings
    retention_days: int = Field(default=365, description="Default retention period in days")
    lifecycle_rules: Optional[Dict[str, Any]] = Field(default=None, description="S3 lifecycle rules")
    
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


class SnowflakeConfig(BaseModel):
    """Snowflake configuration model for PBF-LB/M data warehouse."""
    
    # Connection settings
    account: str = Field(default="your-account.snowflakecomputing.com", description="Snowflake account identifier")
    user: str = Field(default="your-username", description="Snowflake username")
    password: str = Field(default="your-password", description="Snowflake password")
    warehouse: str = Field(default="COMPUTE_WH", description="Snowflake warehouse name")
    database: str = Field(default="LPBF_RESEARCH", description="Snowflake database name")
    schema_name: str = Field(default="PUBLIC", description="Snowflake schema name")
    role: str = Field(default="ACCOUNTADMIN", description="Snowflake role")
    
    # Security settings
    use_ssl: bool = Field(default=True, description="Use SSL/TLS for connections")
    verify_ssl: bool = Field(default=True, description="Verify SSL certificates")
    
    # Performance settings
    warehouse_size: str = Field(default="SMALL", description="Warehouse size (X-SMALL, SMALL, MEDIUM, LARGE, X-LARGE)")
    auto_suspend: int = Field(default=60, description="Auto-suspend timeout in seconds")
    auto_resume: bool = Field(default=True, description="Auto-resume warehouse when needed")
    max_connections: int = Field(default=50, description="Maximum connections to Snowflake")
    timeout: int = Field(default=30, description="Connection timeout in seconds")
    
    # Query settings
    query_timeout: int = Field(default=300, description="Query timeout in seconds")
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    retry_delay: int = Field(default=5, description="Retry delay in seconds")
    
    # Data loading settings
    copy_options: Optional[Dict[str, Any]] = Field(default=None, description="Snowflake COPY options")
    
    def __post_init__(self):
        """Initialize default copy options if not provided"""
        if self.copy_options is None:
            self.copy_options = {
                "FILE_FORMAT": "PARQUET",
                "ON_ERROR": "CONTINUE",
                "PURGE": True,
                "RETURN_FAILED_ONLY": True
            }


class DeltaLakeConfig(BaseModel):
    """Delta Lake configuration model for PBF-LB/M data lake."""
    
    # Storage settings
    storage_path: str = Field(default="s3a://pbf-lbm-data-lake/delta/", description="Delta Lake storage path")
    checkpoint_path: str = Field(default="s3a://pbf-lbm-data-lake/checkpoints/", description="Delta Lake checkpoint path")
    
    # Performance settings
    auto_optimize: bool = Field(default=True, description="Enable automatic optimization")
    optimize_interval: int = Field(default=24, description="Optimization interval in hours")
    z_order_columns: Optional[Dict[str, List[str]]] = Field(default=None, description="Z-order columns for optimization")
    
    # Retention settings
    retention_period: int = Field(default=30, description="Data retention period in days")
    vacuum_interval: int = Field(default=7, description="Vacuum interval in days")
    
    # Schema evolution
    allow_schema_evolution: bool = Field(default=True, description="Allow schema evolution")
    merge_schema: bool = Field(default=True, description="Merge schema on evolution")
    
    # Performance optimization
    max_file_size: int = Field(default=134217728, description="Maximum file size (128MB)")
    target_file_size: int = Field(default=67108864, description="Target file size (64MB)")
    max_records_per_file: int = Field(default=1000000, description="Maximum records per file")
    
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
        self.delta_lake_config = DeltaLakeConfig()
        self._load_default_configurations()
    
    def _load_default_configurations(self):
        """Load default storage configurations"""
        # Default S3 configuration
        self.s3_config = S3Config()
        
        # Default Snowflake configuration
        self.snowflake_config = SnowflakeConfig()
        
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
        config.snowflake_config.warehouse = os.getenv("SNOWFLAKE_WAREHOUSE", "PBF_WAREHOUSE")
        config.snowflake_config.database = os.getenv("SNOWFLAKE_DATABASE", "PBF_ANALYTICS")
        config.snowflake_config.schema_name = os.getenv("SNOWFLAKE_SCHEMA", "RAW")
        config.snowflake_config.role = os.getenv("SNOWFLAKE_ROLE", "ACCOUNTADMIN")
        config.snowflake_config.timeout = int(os.getenv("SNOWFLAKE_CONNECTION_TIMEOUT", "60"))
        config.snowflake_config.query_timeout = int(os.getenv("SNOWFLAKE_QUERY_TIMEOUT", "300"))
        
        
        return config
    
    def get_s3_config(self) -> S3Config:
        """Get S3 configuration"""
        return self.s3_config
    
    def get_snowflake_config(self) -> SnowflakeConfig:
        """Get Snowflake configuration"""
        return self.snowflake_config
    
    
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
                "schema": self.snowflake_config.schema_name,
                "warehouse_size": self.snowflake_config.warehouse_size
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




def get_s3_config() -> S3Config:
    """
    Get S3 configuration from environment variables.
    
    Returns:
        S3Config: S3 configuration with environment variable overrides
    """
    return S3Config(
        endpoint_url=os.getenv("S3_ENDPOINT_URL", "http://localhost:9000"),
        aws_access_key_id=os.getenv("S3_ACCESS_KEY_ID", "minioadmin"),
        aws_secret_access_key=os.getenv("S3_SECRET_ACCESS_KEY", "minioadmin"),
        region_name=os.getenv("S3_REGION_NAME", "us-east-1"),
        use_ssl=os.getenv("S3_USE_SSL", "true").lower() == "true",
        verify_ssl=os.getenv("S3_VERIFY_SSL", "true").lower() == "true",
        bucket_name=os.getenv("S3_BUCKET_NAME", "pbf-lbm-data-lake"),
        prefix=os.getenv("S3_PREFIX", "lpbf-research/"),
        multipart_threshold=int(os.getenv("S3_MULTIPART_THRESHOLD", "67108864")),
        multipart_chunksize=int(os.getenv("S3_MULTIPART_CHUNKSIZE", "16777216")),
        max_concurrency=int(os.getenv("S3_MAX_CONCURRENCY", "10")),
        max_connections=int(os.getenv("S3_MAX_CONNECTIONS", "50")),
        timeout=int(os.getenv("S3_TIMEOUT", "30")),
        retry_attempts=int(os.getenv("S3_RETRY_ATTEMPTS", "3")),
        retention_days=int(os.getenv("S3_RETENTION_DAYS", "365"))
    )


def get_snowflake_config() -> SnowflakeConfig:
    """
    Get Snowflake configuration from environment variables.
    
    Returns:
        SnowflakeConfig: Snowflake configuration with environment variable overrides
    """
    return SnowflakeConfig(
        account=os.getenv("SNOWFLAKE_ACCOUNT", "your-account.snowflakecomputing.com"),
        user=os.getenv("SNOWFLAKE_USER", "your-username"),
        password=os.getenv("SNOWFLAKE_PASSWORD", "your-password"),
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE", "PBF_WAREHOUSE"),
        database=os.getenv("SNOWFLAKE_DATABASE", "PBF_ANALYTICS"),
        schema_name=os.getenv("SNOWFLAKE_SCHEMA", "RAW"),
        role=os.getenv("SNOWFLAKE_ROLE", "ACCOUNTADMIN"),
        use_ssl=os.getenv("SNOWFLAKE_USE_SSL", "true").lower() == "true",
        verify_ssl=os.getenv("SNOWFLAKE_VERIFY_SSL", "true").lower() == "true",
        warehouse_size=os.getenv("SNOWFLAKE_WAREHOUSE_SIZE", "SMALL"),
        auto_suspend=int(os.getenv("SNOWFLAKE_AUTO_SUSPEND", "60")),
        auto_resume=os.getenv("SNOWFLAKE_AUTO_RESUME", "true").lower() == "true",
        max_connections=int(os.getenv("SNOWFLAKE_MAX_CONNECTIONS", "50")),
        timeout=int(os.getenv("SNOWFLAKE_CONNECTION_TIMEOUT", "60")),
        query_timeout=int(os.getenv("SNOWFLAKE_QUERY_TIMEOUT", "300")),
        max_retries=int(os.getenv("SNOWFLAKE_MAX_RETRIES", "3")),
        retry_delay=int(os.getenv("SNOWFLAKE_RETRY_DELAY", "5"))
    )


def get_delta_lake_config() -> DeltaLakeConfig:
    """
    Get Delta Lake configuration from environment variables.
    
    Returns:
        DeltaLakeConfig: Delta Lake configuration with environment variable overrides
    """
    return DeltaLakeConfig(
        storage_path=os.getenv("DELTA_LAKE_STORAGE_PATH", "s3a://pbf-lbm-data-lake/delta/"),
        checkpoint_path=os.getenv("DELTA_LAKE_CHECKPOINT_PATH", "s3a://pbf-lbm-data-lake/checkpoints/"),
        auto_optimize=os.getenv("DELTA_LAKE_AUTO_OPTIMIZE", "true").lower() == "true",
        optimize_interval=int(os.getenv("DELTA_LAKE_OPTIMIZE_INTERVAL", "24")),
        retention_period=int(os.getenv("DELTA_LAKE_RETENTION_PERIOD", "30")),
        vacuum_interval=int(os.getenv("DELTA_LAKE_VACUUM_INTERVAL", "7")),
        allow_schema_evolution=os.getenv("DELTA_LAKE_ALLOW_SCHEMA_EVOLUTION", "true").lower() == "true",
        merge_schema=os.getenv("DELTA_LAKE_MERGE_SCHEMA", "true").lower() == "true",
        max_file_size=int(os.getenv("DELTA_LAKE_MAX_FILE_SIZE", "134217728")),
        target_file_size=int(os.getenv("DELTA_LAKE_TARGET_FILE_SIZE", "67108864")),
        max_records_per_file=int(os.getenv("DELTA_LAKE_MAX_RECORDS_PER_FILE", "1000000"))
    )


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