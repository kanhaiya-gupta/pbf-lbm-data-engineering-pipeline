"""
ClickHouse Configuration for PBF-LB/M Data Pipeline

This module provides ClickHouse configuration management for data warehouse
analytics in the PBF-LB/M data pipeline system.

Features:
- Environment variable configuration
- Connection pooling and timeout settings
- SSL/TLS support for production
- Performance optimization settings
- Analytics and reporting configuration
"""

import os
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, validator
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class ClickHouseConfig(BaseModel):
    """ClickHouse configuration model for PBF-LB/M data pipeline."""
    
    # Connection settings
    host: str = Field(default="localhost", description="ClickHouse server host")
    port: int = Field(default=8123, description="ClickHouse HTTP port")
    native_port: int = Field(default=9090, description="ClickHouse native port")
    mysql_port: int = Field(default=9009, description="ClickHouse MySQL port")
    database: str = Field(default="pbf_analytics", description="Default database name")
    username: str = Field(default="analytics_user", description="ClickHouse username")
    password: str = Field(default="analytics_password", description="ClickHouse password")
    
    # Connection pool settings
    max_connections: int = Field(default=20, description="Maximum connections in pool")
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    timeout: int = Field(default=30, description="Request timeout in seconds")
    connect_timeout: int = Field(default=10, description="Connection timeout in seconds")
    retry_delay: float = Field(default=1.0, description="Delay between retries in seconds")
    
    # SSL/TLS settings
    secure: bool = Field(default=False, description="Use HTTPS connection")
    verify_ssl: bool = Field(default=True, description="Verify SSL certificates")
    ca_cert: Optional[str] = Field(default=None, description="Path to CA certificate")
    client_cert: Optional[str] = Field(default=None, description="Path to client certificate")
    client_key: Optional[str] = Field(default=None, description="Path to client key")
    
    # Performance settings
    max_memory_usage: int = Field(default=10000000000, description="Maximum memory usage per query (10GB)")
    max_execution_time: int = Field(default=300, description="Maximum query execution time in seconds")
    max_threads: int = Field(default=8, description="Maximum number of threads for query execution")
    max_concurrent_queries: int = Field(default=100, description="Maximum concurrent queries")
    
    # Compression settings
    enable_compression: bool = Field(default=True, description="Enable response compression")
    compression_level: int = Field(default=6, description="Compression level (1-9)")
    compression_method: str = Field(default="gzip", description="Compression method")
    
    # Cache settings
    use_uncompressed_cache: bool = Field(default=True, description="Use uncompressed cache")
    mark_cache_size: int = Field(default=5368709120, description="Mark cache size (5GB)")
    uncompressed_cache_size: int = Field(default=8589934592, description="Uncompressed cache size (8GB)")
    
    # Query settings
    query_log_enabled: bool = Field(default=True, description="Enable query logging")
    query_log_level: str = Field(default="information", description="Query log level")
    slow_query_threshold: int = Field(default=1000, description="Slow query threshold in milliseconds")
    
    # Analytics settings
    default_aggregation_level: str = Field(default="day", description="Default aggregation level")
    enable_approximate_functions: bool = Field(default=True, description="Enable approximate functions")
    sampling_ratio: float = Field(default=0.1, description="Default sampling ratio for large datasets")
    
    # Time-series settings
    time_series_retention_days: int = Field(default=365, description="Time-series data retention in days")
    time_series_compression: str = Field(default="LZ4", description="Time-series compression algorithm")
    time_series_partition_by: str = Field(default="toYYYYMM(timestamp)", description="Time-series partition function")
    
    # Data warehouse settings
    warehouse_retention_days: int = Field(default=2555, description="Data warehouse retention in days (7 years)")
    warehouse_compression: str = Field(default="ZSTD", description="Data warehouse compression algorithm")
    warehouse_partition_by: str = Field(default="toYYYYMM(timestamp)", description="Data warehouse partition function")
    
    # Monitoring settings
    health_check_interval: int = Field(default=30, description="Health check interval in seconds")
    metrics_enabled: bool = Field(default=True, description="Enable metrics collection")
    profiling_enabled: bool = Field(default=True, description="Enable query profiling")
    
    # Security settings
    access_management: bool = Field(default=True, description="Enable access management")
    quota_enabled: bool = Field(default=True, description="Enable quota management")
    quota_max_queries: int = Field(default=1000, description="Maximum queries per hour")
    quota_max_errors: int = Field(default=100, description="Maximum errors per hour")
    
    # Backup settings
    backup_enabled: bool = Field(default=True, description="Enable automated backups")
    backup_interval_hours: int = Field(default=24, description="Backup interval in hours")
    backup_retention_days: int = Field(default=30, description="Backup retention in days")
    backup_compression: str = Field(default="gzip", description="Backup compression method")
    
    # Replication settings
    replication_enabled: bool = Field(default=False, description="Enable replication")
    replica_hosts: List[str] = Field(default_factory=list, description="Replica host addresses")
    replica_priority: int = Field(default=1, description="Replica priority")
    
    # Advanced settings
    distributed_ddl_timeout: int = Field(default=60, description="Distributed DDL timeout in seconds")
    insert_distributed_timeout: int = Field(default=300, description="Distributed insert timeout in seconds")
    max_insert_block_size: int = Field(default=1048576, description="Maximum insert block size")
    min_insert_block_size_rows: int = Field(default=65536, description="Minimum insert block size in rows")
    
    # Materialized views settings
    materialized_views_enabled: bool = Field(default=True, description="Enable materialized views")
    materialized_views_refresh_interval: int = Field(default=3600, description="Materialized views refresh interval in seconds")
    
    # Dictionary settings
    dictionaries_enabled: bool = Field(default=True, description="Enable dictionaries")
    dictionary_reload_interval: int = Field(default=3600, description="Dictionary reload interval in seconds")
    
    # Logging settings
    log_level: str = Field(default="information", description="Log level")
    log_queries: bool = Field(default=True, description="Log queries")
    log_queries_min_type: str = Field(default="QUERY_START", description="Minimum query type to log")
    log_queries_min_duration_ms: int = Field(default=0, description="Minimum query duration to log in milliseconds")
    
    @validator('host')
    def validate_host(cls, v):
        """Validate host format."""
        if not v or not isinstance(v, str):
            raise ValueError("Host must be a non-empty string")
        return v
    
    @validator('port', 'native_port', 'mysql_port')
    def validate_ports(cls, v):
        """Validate port numbers."""
        if not isinstance(v, int) or v < 1 or v > 65535:
            raise ValueError("Port must be an integer between 1 and 65535")
        return v
    
    @validator('max_connections', 'max_retries', 'timeout', 'connect_timeout')
    def validate_positive_integers(cls, v):
        """Validate positive integer values."""
        if not isinstance(v, int) or v <= 0:
            raise ValueError("Value must be a positive integer")
        return v
    
    @validator('compression_level')
    def validate_compression_level(cls, v):
        """Validate compression level."""
        if not isinstance(v, int) or v < 1 or v > 9:
            raise ValueError("Compression level must be between 1 and 9")
        return v
    
    @validator('sampling_ratio')
    def validate_sampling_ratio(cls, v):
        """Validate sampling ratio."""
        if not isinstance(v, (int, float)) or v <= 0 or v > 1:
            raise ValueError("Sampling ratio must be between 0 and 1")
        return v
    
    @validator('compression_method')
    def validate_compression_method(cls, v):
        """Validate compression method."""
        valid_methods = ['gzip', 'deflate', 'lz4', 'zstd', 'brotli']
        if v not in valid_methods:
            raise ValueError(f"Compression method must be one of: {', '.join(valid_methods)}")
        return v
    
    @validator('time_series_compression', 'warehouse_compression')
    def validate_compression_algorithm(cls, v):
        """Validate compression algorithm."""
        valid_algorithms = ['LZ4', 'ZSTD', 'LZ4HC', 'SNAPPY', 'NONE']
        if v not in valid_algorithms:
            raise ValueError(f"Compression algorithm must be one of: {', '.join(valid_algorithms)}")
        return v
    
    @validator('log_level')
    def validate_log_level(cls, v):
        """Validate log level."""
        valid_levels = ['trace', 'debug', 'information', 'warning', 'error', 'fatal']
        if v not in valid_levels:
            raise ValueError(f"Log level must be one of: {', '.join(valid_levels)}")
        return v
    
    @validator('default_aggregation_level')
    def validate_aggregation_level(cls, v):
        """Validate aggregation level."""
        valid_levels = ['second', 'minute', 'hour', 'day', 'week', 'month', 'quarter', 'year']
        if v not in valid_levels:
            raise ValueError(f"Aggregation level must be one of: {', '.join(valid_levels)}")
        return v
    
    class Config:
        """Pydantic configuration."""
        env_prefix = "CLICKHOUSE_"
        case_sensitive = False
        validate_assignment = True
        extra = "forbid"


def get_clickhouse_config() -> ClickHouseConfig:
    """
    Get ClickHouse configuration from environment variables.
    
    Returns:
        ClickHouseConfig: Configured ClickHouse settings
    """
    return ClickHouseConfig()


def get_clickhouse_config_dict() -> Dict[str, Any]:
    """
    Get ClickHouse configuration as dictionary.
    
    Returns:
        Dict[str, Any]: ClickHouse configuration dictionary
    """
    config = get_clickhouse_config()
    return config.dict()


def get_clickhouse_connection_string() -> str:
    """
    Get ClickHouse connection string.
    
    Returns:
        str: ClickHouse connection string
    """
    config = get_clickhouse_config()
    protocol = "https" if config.secure else "http"
    return f"{protocol}://{config.host}:{config.port}"


def get_clickhouse_native_connection_string() -> str:
    """
    Get ClickHouse native connection string.
    
    Returns:
        str: ClickHouse native connection string
    """
    config = get_clickhouse_config()
    return f"clickhouse://{config.username}:{config.password}@{config.host}:{config.native_port}/{config.database}"


# Environment variable mappings
ENV_MAPPINGS = {
    'CLICKHOUSE_HOST': 'host',
    'CLICKHOUSE_PORT': 'port',
    'CLICKHOUSE_NATIVE_PORT': 'native_port',
    'CLICKHOUSE_MYSQL_PORT': 'mysql_port',
    'CLICKHOUSE_DATABASE': 'database',
    'CLICKHOUSE_USERNAME': 'username',
    'CLICKHOUSE_PASSWORD': 'password',
    'CLICKHOUSE_SECURE': 'secure',
    'CLICKHOUSE_MAX_CONNECTIONS': 'max_connections',
    'CLICKHOUSE_TIMEOUT': 'timeout',
    'CLICKHOUSE_MAX_MEMORY_USAGE': 'max_memory_usage',
    'CLICKHOUSE_MAX_THREADS': 'max_threads',
    'CLICKHOUSE_ENABLE_COMPRESSION': 'enable_compression',
    'CLICKHOUSE_COMPRESSION_LEVEL': 'compression_level',
    'CLICKHOUSE_QUERY_LOG_ENABLED': 'query_log_enabled',
    'CLICKHOUSE_HEALTH_CHECK_INTERVAL': 'health_check_interval',
    'CLICKHOUSE_BACKUP_ENABLED': 'backup_enabled',
    'CLICKHOUSE_REPLICATION_ENABLED': 'replication_enabled',
}


if __name__ == "__main__":
    """Test the configuration."""
    config = get_clickhouse_config()
    print("ClickHouse Configuration:")
    print(f"Host: {config.host}:{config.port}")
    print(f"Database: {config.database}")
    print(f"Username: {config.username}")
    print(f"Secure: {config.secure}")
    print(f"Max Connections: {config.max_connections}")
    print(f"Timeout: {config.timeout}")
    print(f"Compression: {config.enable_compression}")
    print(f"Query Log: {config.query_log_enabled}")
    print(f"Backup: {config.backup_enabled}")
    print(f"Replication: {config.replication_enabled}")
