"""
Snowflake Configuration for PBF-LB/M Data Pipeline

This module provides Snowflake configuration management for data warehouse
operations in the PBF-LB/M data pipeline system.

Features:
- Environment variable configuration
- Connection pooling and timeout settings
- SSL/TLS support for production
- Performance optimization settings
- Data warehouse operations configuration
"""

import os
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, validator
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class SnowflakeConfig(BaseModel):
    """Snowflake configuration model for PBF-LB/M data pipeline."""
    
    # Connection settings
    account: str = Field(default="your-account.snowflakecomputing.com", description="Snowflake account identifier")
    user: str = Field(default="your-username", description="Snowflake username")
    password: str = Field(default="your-password", description="Snowflake password")
    warehouse: str = Field(default="PBF_WAREHOUSE", description="Snowflake warehouse name")
    database: str = Field(default="PBF_ANALYTICS", description="Snowflake database name")
    schema_name: str = Field(default="RAW", description="Snowflake schema name")
    role: str = Field(default="ACCOUNTADMIN", description="Snowflake role")
    region: str = Field(default="us-west-2", description="Snowflake region")
    
    # Security settings
    use_ssl: bool = Field(default=True, description="Use SSL/TLS for connections")
    verify_ssl: bool = Field(default=True, description="Verify SSL certificates")
    
    # Performance settings
    warehouse_size: str = Field(default="SMALL", description="Warehouse size (X-SMALL, SMALL, MEDIUM, LARGE, X-LARGE)")
    auto_suspend: int = Field(default=60, description="Auto-suspend timeout in seconds")
    auto_resume: bool = Field(default=True, description="Auto-resume warehouse when needed")
    max_connections: int = Field(default=50, description="Maximum connections to Snowflake")
    
    # Timeout settings
    connection_timeout: int = Field(default=60, description="Connection timeout in seconds")
    query_timeout: int = Field(default=300, description="Query timeout in seconds")
    network_timeout: int = Field(default=600, description="Network timeout in seconds")
    
    # Retry settings
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    retry_delay: int = Field(default=5, description="Retry delay in seconds")
    retry_backoff_factor: float = Field(default=2.0, description="Retry backoff factor")
    
    # Data loading settings
    batch_size: int = Field(default=1000, description="Batch size for data loading")
    max_batch_size: int = Field(default=10000, description="Maximum batch size")
    copy_options: Dict[str, Any] = Field(default_factory=dict, description="Snowflake COPY options")
    
    # Session settings
    autocommit: bool = Field(default=True, description="Auto-commit transactions")
    client_session_keep_alive: bool = Field(default=True, description="Keep client session alive")
    client_session_keep_alive_heartbeat_frequency: int = Field(default=3600, description="Heartbeat frequency in seconds")
    
    # Development settings
    debug: bool = Field(default=False, description="Enable debug mode")
    log_level: str = Field(default="INFO", description="Log level (DEBUG, INFO, WARNING, ERROR)")
    query_tag: str = Field(default="pbf-dev", description="Query tag for tracking")
    
    # Data retention settings
    data_retention_days: int = Field(default=365, description="Data retention period in days")
    time_travel_enabled: bool = Field(default=True, description="Enable time travel")
    failover_enabled: bool = Field(default=False, description="Enable failover")
    
    # Performance optimization
    query_result_format: str = Field(default="JSON", description="Query result format (JSON, ARROW, PARQUET)")
    enable_arrow_result_batches: bool = Field(default=True, description="Enable Arrow result batches")
    arrow_number_to_decimal: bool = Field(default=True, description="Convert Arrow numbers to decimal")
    
    @validator('warehouse_size')
    def validate_warehouse_size(cls, v):
        """Validate warehouse size."""
        valid_sizes = ['X-SMALL', 'SMALL', 'MEDIUM', 'LARGE', 'X-LARGE', 'XX-LARGE', 'XXX-LARGE']
        if v.upper() not in valid_sizes:
            raise ValueError(f"Invalid warehouse size: {v}. Must be one of {valid_sizes}")
        return v.upper()
    
    @validator('log_level')
    def validate_log_level(cls, v):
        """Validate log level."""
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f"Invalid log level: {v}. Must be one of {valid_levels}")
        return v.upper()
    
    @validator('query_result_format')
    def validate_query_result_format(cls, v):
        """Validate query result format."""
        valid_formats = ['JSON', 'ARROW', 'PARQUET']
        if v.upper() not in valid_formats:
            raise ValueError(f"Invalid query result format: {v}. Must be one of {valid_formats}")
        return v.upper()
    
    def get_connection_params(self) -> Dict[str, Any]:
        """Get connection parameters for Snowflake connector."""
        return {
            "account": self.account,
            "user": self.user,
            "password": self.password,
            "warehouse": self.warehouse,
            "database": self.database,
            "schema": self.schema_name,
            "role": self.role,
            "region": self.region,
            "use_ssl": self.use_ssl,
            "verify_ssl": self.verify_ssl,
            "autocommit": self.autocommit,
            "client_session_keep_alive": self.client_session_keep_alive,
            "client_session_keep_alive_heartbeat_frequency": self.client_session_keep_alive_heartbeat_frequency,
            "timezone": "UTC"
        }
    
    def get_session_params(self) -> Dict[str, Any]:
        """Get session parameters for Snowflake operations."""
        return {
            "QUERY_TAG": self.query_tag,
            "WAREHOUSE_SIZE": self.warehouse_size,
            "AUTO_SUSPEND": self.auto_suspend,
            "AUTO_RESUME": self.auto_resume
        }
    
    def get_copy_options(self) -> Dict[str, Any]:
        """Get COPY options for data loading."""
        default_options = {
            "FILE_FORMAT": "PARQUET",
            "ON_ERROR": "CONTINUE",
            "PURGE": True,
            "RETURN_FAILED_ONLY": True,
            "SIZE_LIMIT": 100000000,  # 100MB
            "VALIDATION_MODE": "RETURN_ERRORS"
        }
        
        # Merge with custom options
        copy_options = default_options.copy()
        copy_options.update(self.copy_options)
        return copy_options


class SnowflakeWarehouseConfig(BaseModel):
    """Snowflake warehouse configuration for PBF-LB/M operations."""
    
    warehouse_name: str = Field(default="PBF_WAREHOUSE", description="Warehouse name")
    warehouse_size: str = Field(default="SMALL", description="Warehouse size")
    auto_suspend: int = Field(default=60, description="Auto-suspend timeout in seconds")
    auto_resume: bool = Field(default=True, description="Auto-resume warehouse")
    initially_suspended: bool = Field(default=True, description="Initially suspended")
    resource_monitor: str = Field(default="", description="Resource monitor name")
    comment: str = Field(default="PBF-LB/M Manufacturing Data Warehouse", description="Warehouse comment")
    
    # Scaling settings
    scaling_policy: str = Field(default="STANDARD", description="Scaling policy (STANDARD, ECONOMY)")
    min_cluster_count: int = Field(default=1, description="Minimum cluster count")
    max_cluster_count: int = Field(default=1, description="Maximum cluster count")
    
    # Performance settings
    statement_timeout_in_seconds: int = Field(default=3600, description="Statement timeout")
    statement_queued_timeout_in_seconds: int = Field(default=0, description="Statement queued timeout")
    
    def get_create_warehouse_sql(self) -> str:
        """Get SQL for creating the warehouse."""
        return f"""
        CREATE WAREHOUSE IF NOT EXISTS {self.warehouse_name}
        WITH WAREHOUSE_SIZE = {self.warehouse_size}
        AUTO_SUSPEND = {self.auto_suspend}
        AUTO_RESUME = {self.auto_resume}
        INITIALLY_SUSPENDED = {self.initially_suspended}
        SCALING_POLICY = {self.scaling_policy}
        MIN_CLUSTER_COUNT = {self.min_cluster_count}
        MAX_CLUSTER_COUNT = {self.max_cluster_count}
        STATEMENT_TIMEOUT_IN_SECONDS = {self.statement_timeout_in_seconds}
        STATEMENT_QUEUED_TIMEOUT_IN_SECONDS = {self.statement_queued_timeout_in_seconds}
        COMMENT = '{self.comment}'
        """


class SnowflakeDatabaseConfig(BaseModel):
    """Snowflake database configuration for PBF-LB/M operations."""
    
    database_name: str = Field(default="PBF_ANALYTICS", description="Database name")
    comment: str = Field(default="PBF-LB/M Manufacturing Analytics Database", description="Database comment")
    data_retention_time_in_days: int = Field(default=365, description="Data retention time in days")
    
    # Schemas to create
    schemas: List[str] = Field(
        default_factory=lambda: ["RAW", "STAGING", "ANALYTICS", "REPORTS"],
        description="List of schemas to create"
    )
    
    def get_create_database_sql(self) -> str:
        """Get SQL for creating the database."""
        return f"""
        CREATE DATABASE IF NOT EXISTS {self.database_name}
        COMMENT = '{self.comment}'
        DATA_RETENTION_TIME_IN_DAYS = {self.data_retention_time_in_days}
        """
    
    def get_create_schemas_sql(self) -> List[str]:
        """Get SQL for creating schemas."""
        sql_statements = []
        for schema in self.schemas:
            sql_statements.append(f"CREATE SCHEMA IF NOT EXISTS {self.database_name}.{schema}")
        return sql_statements


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
        region=os.getenv("SNOWFLAKE_REGION", "us-west-2"),
        use_ssl=os.getenv("SNOWFLAKE_USE_SSL", "true").lower() == "true",
        verify_ssl=os.getenv("SNOWFLAKE_VERIFY_SSL", "true").lower() == "true",
        warehouse_size=os.getenv("SNOWFLAKE_WAREHOUSE_SIZE", "SMALL"),
        auto_suspend=int(os.getenv("SNOWFLAKE_AUTO_SUSPEND", "60")),
        auto_resume=os.getenv("SNOWFLAKE_AUTO_RESUME", "true").lower() == "true",
        max_connections=int(os.getenv("SNOWFLAKE_MAX_CONNECTIONS", "50")),
        connection_timeout=int(os.getenv("SNOWFLAKE_CONNECTION_TIMEOUT", "60")),
        query_timeout=int(os.getenv("SNOWFLAKE_QUERY_TIMEOUT", "300")),
        network_timeout=int(os.getenv("SNOWFLAKE_NETWORK_TIMEOUT", "600")),
        max_retries=int(os.getenv("SNOWFLAKE_MAX_RETRIES", "3")),
        retry_delay=int(os.getenv("SNOWFLAKE_RETRY_DELAY", "5")),
        retry_backoff_factor=float(os.getenv("SNOWFLAKE_RETRY_BACKOFF_FACTOR", "2.0")),
        batch_size=int(os.getenv("SNOWFLAKE_BATCH_SIZE", "1000")),
        max_batch_size=int(os.getenv("SNOWFLAKE_MAX_BATCH_SIZE", "10000")),
        autocommit=os.getenv("SNOWFLAKE_AUTOCOMMIT", "true").lower() == "true",
        client_session_keep_alive=os.getenv("SNOWFLAKE_CLIENT_SESSION_KEEP_ALIVE", "true").lower() == "true",
        client_session_keep_alive_heartbeat_frequency=int(os.getenv("SNOWFLAKE_CLIENT_SESSION_KEEP_ALIVE_HEARTBEAT_FREQUENCY", "3600")),
        debug=os.getenv("SNOWFLAKE_DEBUG", "false").lower() == "true",
        log_level=os.getenv("SNOWFLAKE_LOG_LEVEL", "INFO"),
        query_tag=os.getenv("SNOWFLAKE_QUERY_TAG", "pbf-dev"),
        data_retention_days=int(os.getenv("SNOWFLAKE_DATA_RETENTION_DAYS", "365")),
        time_travel_enabled=os.getenv("SNOWFLAKE_TIME_TRAVEL_ENABLED", "true").lower() == "true",
        failover_enabled=os.getenv("SNOWFLAKE_FAILOVER_ENABLED", "false").lower() == "true",
        query_result_format=os.getenv("SNOWFLAKE_QUERY_RESULT_FORMAT", "JSON"),
        enable_arrow_result_batches=os.getenv("SNOWFLAKE_ENABLE_ARROW_RESULT_BATCHES", "true").lower() == "true",
        arrow_number_to_decimal=os.getenv("SNOWFLAKE_ARROW_NUMBER_TO_DECIMAL", "true").lower() == "true"
    )


def get_snowflake_warehouse_config() -> SnowflakeWarehouseConfig:
    """
    Get Snowflake warehouse configuration.
    
    Returns:
        SnowflakeWarehouseConfig: Warehouse configuration
    """
    return SnowflakeWarehouseConfig(
        warehouse_name=os.getenv("SNOWFLAKE_WAREHOUSE", "PBF_WAREHOUSE"),
        warehouse_size=os.getenv("SNOWFLAKE_WAREHOUSE_SIZE", "SMALL"),
        auto_suspend=int(os.getenv("SNOWFLAKE_AUTO_SUSPEND", "60")),
        auto_resume=os.getenv("SNOWFLAKE_AUTO_RESUME", "true").lower() == "true",
        initially_suspended=os.getenv("SNOWFLAKE_INITIALLY_SUSPENDED", "true").lower() == "true",
        resource_monitor=os.getenv("SNOWFLAKE_RESOURCE_MONITOR", ""),
        comment=os.getenv("SNOWFLAKE_WAREHOUSE_COMMENT", "PBF-LB/M Manufacturing Data Warehouse"),
        scaling_policy=os.getenv("SNOWFLAKE_SCALING_POLICY", "STANDARD"),
        min_cluster_count=int(os.getenv("SNOWFLAKE_MIN_CLUSTER_COUNT", "1")),
        max_cluster_count=int(os.getenv("SNOWFLAKE_MAX_CLUSTER_COUNT", "1")),
        statement_timeout_in_seconds=int(os.getenv("SNOWFLAKE_STATEMENT_TIMEOUT", "3600")),
        statement_queued_timeout_in_seconds=int(os.getenv("SNOWFLAKE_STATEMENT_QUEUED_TIMEOUT", "0"))
    )


def get_snowflake_database_config() -> SnowflakeDatabaseConfig:
    """
    Get Snowflake database configuration.
    
    Returns:
        SnowflakeDatabaseConfig: Database configuration
    """
    return SnowflakeDatabaseConfig(
        database_name=os.getenv("SNOWFLAKE_DATABASE", "PBF_ANALYTICS"),
        comment=os.getenv("SNOWFLAKE_DATABASE_COMMENT", "PBF-LB/M Manufacturing Analytics Database"),
        data_retention_time_in_days=int(os.getenv("SNOWFLAKE_DATA_RETENTION_DAYS", "365")),
        schemas=os.getenv("SNOWFLAKE_SCHEMAS", "RAW,STAGING,ANALYTICS,REPORTS").split(",")
    )


# Global configuration instances
_snowflake_config: Optional[SnowflakeConfig] = None
_snowflake_warehouse_config: Optional[SnowflakeWarehouseConfig] = None
_snowflake_database_config: Optional[SnowflakeDatabaseConfig] = None


def get_global_snowflake_config() -> SnowflakeConfig:
    """
    Get the global Snowflake configuration instance.
    
    Returns:
        SnowflakeConfig: The global Snowflake configuration
    """
    global _snowflake_config
    if _snowflake_config is None:
        _snowflake_config = get_snowflake_config()
    return _snowflake_config


def get_global_snowflake_warehouse_config() -> SnowflakeWarehouseConfig:
    """
    Get the global Snowflake warehouse configuration instance.
    
    Returns:
        SnowflakeWarehouseConfig: The global warehouse configuration
    """
    global _snowflake_warehouse_config
    if _snowflake_warehouse_config is None:
        _snowflake_warehouse_config = get_snowflake_warehouse_config()
    return _snowflake_warehouse_config


def get_global_snowflake_database_config() -> SnowflakeDatabaseConfig:
    """
    Get the global Snowflake database configuration instance.
    
    Returns:
        SnowflakeDatabaseConfig: The global database configuration
    """
    global _snowflake_database_config
    if _snowflake_database_config is None:
        _snowflake_database_config = get_snowflake_database_config()
    return _snowflake_database_config


def reset_snowflake_configs() -> None:
    """Reset all global Snowflake configuration instances."""
    global _snowflake_config, _snowflake_warehouse_config, _snowflake_database_config
    _snowflake_config = None
    _snowflake_warehouse_config = None
    _snowflake_database_config = None
