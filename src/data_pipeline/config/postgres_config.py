"""
PostgreSQL Configuration for PBF-LB/M Data Pipeline

This module provides PostgreSQL configuration management with environment variable support,
connection pooling, and production-ready settings.

Features:
- Environment variable configuration
- Connection pooling and timeout settings
- SSL/TLS support for production
- Health check configuration
- Performance optimization settings
"""

import os
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class PostgreSQLConfig(BaseModel):
    """PostgreSQL configuration model."""
    
    # Connection settings
    host: str = Field(default="localhost", description="PostgreSQL host")
    port: int = Field(default=5432, description="PostgreSQL port")
    database: str = Field(default="lpbf_research", description="PostgreSQL database name")
    username: str = Field(default="postgres", description="PostgreSQL username")
    password: str = Field(default="password", description="PostgreSQL password")
    
    # Connection pool settings
    pool_size: int = Field(default=10, description="Connection pool size")
    max_overflow: int = Field(default=20, description="Maximum overflow connections")
    pool_timeout: int = Field(default=30, description="Pool timeout in seconds")
    pool_recycle: int = Field(default=3600, description="Pool recycle time in seconds")
    
    # Performance settings
    statement_timeout: int = Field(default=300, description="Statement timeout in seconds")
    query_timeout: int = Field(default=300, description="Query timeout in seconds")
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    retry_delay: int = Field(default=5, description="Retry delay in seconds")
    
    # Schema settings
    default_schema: str = Field(default="public", description="Default schema")
    search_path: str = Field(default="public", description="Search path")
    
    # SSL/TLS settings
    ssl_enabled: bool = Field(default=False, description="Enable SSL/TLS encryption")
    ssl_certfile: Optional[str] = Field(default=None, description="SSL certificate file")
    ssl_keyfile: Optional[str] = Field(default=None, description="SSL private key file")
    ssl_ca_certs: Optional[str] = Field(default=None, description="SSL CA certificates file")
    ssl_check_hostname: bool = Field(default=True, description="Check SSL hostname")
    
    # Health check settings
    health_check_interval: int = Field(default=30, description="Health check interval in seconds")
    ping_interval: int = Field(default=60, description="Ping interval in seconds")
    
    # Performance optimization
    enable_prepared_statements: bool = Field(default=True, description="Enable prepared statements")
    enable_connection_pooling: bool = Field(default=True, description="Enable connection pooling")
    enable_query_optimization: bool = Field(default=True, description="Enable query optimization")
    
    # Monitoring and metrics
    enable_metrics: bool = Field(default=True, description="Enable connection metrics")
    enable_query_logging: bool = Field(default=False, description="Enable query logging")
    log_slow_queries: bool = Field(default=True, description="Log slow queries")
    slow_query_threshold: int = Field(default=1000, description="Slow query threshold in milliseconds")
    
    # Backup and recovery
    enable_automatic_backups: bool = Field(default=False, description="Enable automatic backups")
    backup_interval: int = Field(default=24, description="Backup interval in hours")
    backup_retention_days: int = Field(default=7, description="Backup retention in days")
    
    class Config:
        env_prefix = "POSTGRES_"
        case_sensitive = False

    def get_connection_string(self) -> str:
        """Get PostgreSQL connection string."""
        ssl_mode = "require" if self.ssl_enabled else "disable"
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}?sslmode={ssl_mode}"

    def get_connection_dict(self) -> Dict[str, Any]:
        """Get PostgreSQL connection parameters as dictionary."""
        return {
            'host': self.host,
            'port': self.port,
            'database': self.database,
            'user': self.username,
            'password': self.password,
            'sslmode': 'require' if self.ssl_enabled else 'disable',
            'connect_timeout': self.pool_timeout,
            'application_name': 'pbf-lbm-pipeline'
        }


def get_postgres_config() -> PostgreSQLConfig:
    """
    Get PostgreSQL configuration from environment variables.
    
    Returns:
        PostgreSQLConfig: Configured PostgreSQL settings
    """
    return PostgreSQLConfig(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        database=os.getenv("POSTGRES_DATABASE", "lpbf_research"),
        username=os.getenv("POSTGRES_USERNAME", "postgres"),
        password=os.getenv("POSTGRES_PASSWORD", "password"),
        pool_size=int(os.getenv("POSTGRES_POOL_SIZE", "10")),
        max_overflow=int(os.getenv("POSTGRES_MAX_OVERFLOW", "20")),
        pool_timeout=int(os.getenv("POSTGRES_POOL_TIMEOUT", "30")),
        pool_recycle=int(os.getenv("POSTGRES_POOL_RECYCLE", "3600")),
        statement_timeout=int(os.getenv("POSTGRES_STATEMENT_TIMEOUT", "300")),
        query_timeout=int(os.getenv("POSTGRES_QUERY_TIMEOUT", "300")),
        max_retries=int(os.getenv("POSTGRES_MAX_RETRIES", "3")),
        retry_delay=int(os.getenv("POSTGRES_RETRY_DELAY", "5")),
        default_schema=os.getenv("POSTGRES_DEFAULT_SCHEMA", "public"),
        search_path=os.getenv("POSTGRES_SEARCH_PATH", "public"),
        ssl_enabled=os.getenv("POSTGRES_SSL_ENABLED", "false").lower() == "true",
        ssl_certfile=os.getenv("POSTGRES_SSL_CERTFILE"),
        ssl_keyfile=os.getenv("POSTGRES_SSL_KEYFILE"),
        ssl_ca_certs=os.getenv("POSTGRES_SSL_CA_CERTS"),
        ssl_check_hostname=os.getenv("POSTGRES_SSL_CHECK_HOSTNAME", "true").lower() == "true",
        health_check_interval=int(os.getenv("POSTGRES_HEALTH_CHECK_INTERVAL", "30")),
        ping_interval=int(os.getenv("POSTGRES_PING_INTERVAL", "60")),
        enable_prepared_statements=os.getenv("POSTGRES_ENABLE_PREPARED_STATEMENTS", "true").lower() == "true",
        enable_connection_pooling=os.getenv("POSTGRES_ENABLE_CONNECTION_POOLING", "true").lower() == "true",
        enable_query_optimization=os.getenv("POSTGRES_ENABLE_QUERY_OPTIMIZATION", "true").lower() == "true",
        enable_metrics=os.getenv("POSTGRES_ENABLE_METRICS", "true").lower() == "true",
        enable_query_logging=os.getenv("POSTGRES_ENABLE_QUERY_LOGGING", "false").lower() == "true",
        log_slow_queries=os.getenv("POSTGRES_LOG_SLOW_QUERIES", "true").lower() == "true",
        slow_query_threshold=int(os.getenv("POSTGRES_SLOW_QUERY_THRESHOLD", "1000")),
        enable_automatic_backups=os.getenv("POSTGRES_ENABLE_AUTOMATIC_BACKUPS", "false").lower() == "true",
        backup_interval=int(os.getenv("POSTGRES_BACKUP_INTERVAL", "24")),
        backup_retention_days=int(os.getenv("POSTGRES_BACKUP_RETENTION_DAYS", "7"))
    )


def get_postgres_connection_string() -> str:
    """
    Get PostgreSQL connection string for logging and debugging.
    
    Returns:
        str: Connection string (without password)
    """
    config = get_postgres_config()
    return f"postgresql://{config.username}:***@{config.host}:{config.port}/{config.database}"
