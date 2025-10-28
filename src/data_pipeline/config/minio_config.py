"""
MinIO Configuration for PBF-LB/M Data Pipeline

This module provides MinIO configuration management for object storage
in the PBF-LB/M data warehouse system.
"""

import os
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field, validator
from pathlib import Path


class MinIOConfig(BaseModel):
    """MinIO configuration for PBF-LB/M data pipeline."""
    
    # Connection settings
    host: str = Field(default="localhost", description="MinIO server host")
    port: int = Field(default=9000, description="MinIO server port")
    console_port: int = Field(default=9001, description="MinIO console port")
    access_key: str = Field(default="minioadmin", description="MinIO access key")
    secret_key: str = Field(default="minioadmin123", description="MinIO secret key")
    secure: bool = Field(default=False, description="Use HTTPS connection")
    ssl_verify: bool = Field(default=True, description="Verify SSL certificates")
    region: str = Field(default="us-east-1", description="MinIO region")
    
    # Connection pool settings
    timeout: int = Field(default=30, description="Connection timeout in seconds")
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    retry_delay: float = Field(default=1.0, description="Delay between retries in seconds")
    connection_pool_size: int = Field(default=10, description="Connection pool size")
    max_connections: int = Field(default=50, description="Maximum connections")
    
    # Performance settings
    multipart_threshold: int = Field(default=67108864, description="Multipart upload threshold (64MB)")
    multipart_chunksize: int = Field(default=16777216, description="Multipart chunk size (16MB)")
    max_concurrency: int = Field(default=10, description="Maximum concurrent operations")
    
    # Compression settings
    enable_compression: bool = Field(default=True, description="Enable compression")
    compression_level: int = Field(default=6, description="Compression level (1-9)")
    
    # Encryption settings
    enable_encryption: bool = Field(default=False, description="Enable encryption")
    encryption_key: Optional[str] = Field(default=None, description="Encryption key")
    
    # Lifecycle settings
    lifecycle_enabled: bool = Field(default=True, description="Enable lifecycle management")
    default_retention_days: int = Field(default=365, description="Default retention in days")
    raw_data_retention_days: int = Field(default=365, description="Raw data retention in days")
    processed_data_retention_days: int = Field(default=180, description="Processed data retention in days")
    archived_data_retention_days: int = Field(default=2555, description="Archived data retention in days (7 years)")
    
    # Versioning settings
    enable_versioning: bool = Field(default=True, description="Enable object versioning")
    
    # Monitoring settings
    enable_metrics: bool = Field(default=True, description="Enable metrics collection")
    metrics_interval: int = Field(default=60, description="Metrics collection interval in seconds")
    
    # Logging settings
    log_level: str = Field(default="INFO", description="Log level")
    log_requests: bool = Field(default=False, description="Log requests")
    log_responses: bool = Field(default=False, description="Log responses")
    
    # PBF-LB/M specific settings
    bucket_name: str = Field(default="pbf-lbm-data", description="Default bucket name")
    raw_data_prefix: str = Field(default="raw/", description="Raw data prefix")
    processed_data_prefix: str = Field(default="processed/", description="Processed data prefix")
    archived_data_prefix: str = Field(default="archived/", description="Archived data prefix")
    analytics_data_prefix: str = Field(default="analytics/", description="Analytics data prefix")
    models_prefix: str = Field(default="models/", description="ML models prefix")
    logs_prefix: str = Field(default="logs/", description="Logs prefix")
    
    # PBF-LB/M data types
    process_data_prefix: str = Field(default="process/", description="Process data prefix")
    monitoring_data_prefix: str = Field(default="monitoring/", description="Monitoring data prefix")
    ct_scan_data_prefix: str = Field(default="ct-scan/", description="CT scan data prefix")
    powder_bed_data_prefix: str = Field(default="powder-bed/", description="Powder bed data prefix")
    research_data_prefix: str = Field(default="research/", description="Research data prefix")
    
    @validator('compression_level')
    def validate_compression_level(cls, v):
        if not 1 <= v <= 9:
            raise ValueError('Compression level must be between 1 and 9')
        return v
    
    @validator('log_level')
    def validate_log_level(cls, v):
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f'Log level must be one of {valid_levels}')
        return v.upper()
    
    def get_endpoint_url(self) -> str:
        """Get MinIO endpoint URL."""
        protocol = "https" if self.secure else "http"
        return f"{protocol}://{self.host}:{self.port}"
    
    def get_console_url(self) -> str:
        """Get MinIO console URL."""
        protocol = "https" if self.secure else "http"
        return f"{protocol}://{self.host}:{self.console_port}"
    
    def get_lifecycle_rules(self) -> Dict[str, Any]:
        """Get lifecycle rules for different data types."""
        return {
            "raw_data": {
                "prefix": self.raw_data_prefix,
                "retention_days": self.raw_data_retention_days,
                "transition_to_ia_days": 30,
                "transition_to_glacier_days": 90
            },
            "processed_data": {
                "prefix": self.processed_data_prefix,
                "retention_days": self.processed_data_retention_days,
                "transition_to_ia_days": 7,
                "transition_to_glacier_days": 30
            },
            "archived_data": {
                "prefix": self.archived_data_prefix,
                "retention_days": self.archived_data_retention_days,
                "transition_to_ia_days": 0
            }
        }
    
    def get_bucket_policy(self) -> Dict[str, Any]:
        """Get bucket policy for PBF-LB/M data access."""
        return {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": "*",
                    "Action": [
                        "s3:GetObject",
                        "s3:PutObject",
                        "s3:DeleteObject"
                    ],
                    "Resource": f"arn:aws:s3:::{self.bucket_name}/*"
                }
            ]
        }


def get_minio_config() -> MinIOConfig:
    """Get MinIO configuration from environment variables."""
    
    return MinIOConfig(
        # Connection settings
        host=os.getenv("MINIO_HOST", "localhost"),
        port=int(os.getenv("MINIO_PORT", "9000")),
        console_port=int(os.getenv("MINIO_CONSOLE_PORT", "9001")),
        access_key=os.getenv("MINIO_ROOT_USER", "minioadmin"),
        secret_key=os.getenv("MINIO_ROOT_PASSWORD", "minioadmin123"),
        secure=os.getenv("MINIO_SECURE", "false").lower() == "true",
        ssl_verify=os.getenv("MINIO_SSL_VERIFY", "true").lower() == "true",
        region=os.getenv("MINIO_REGION", "us-east-1"),
        
        # Connection pool settings
        timeout=int(os.getenv("MINIO_TIMEOUT", "30")),
        max_retries=int(os.getenv("MINIO_MAX_RETRIES", "3")),
        retry_delay=float(os.getenv("MINIO_RETRY_DELAY", "1.0")),
        connection_pool_size=int(os.getenv("MINIO_CONNECTION_POOL_SIZE", "10")),
        max_connections=int(os.getenv("MINIO_MAX_CONNECTIONS", "50")),
        
        # Performance settings
        multipart_threshold=int(os.getenv("MINIO_MULTIPART_THRESHOLD", "67108864")),
        multipart_chunksize=int(os.getenv("MINIO_MULTIPART_CHUNKSIZE", "16777216")),
        max_concurrency=int(os.getenv("MINIO_MAX_CONCURRENCY", "10")),
        
        # Compression settings
        enable_compression=os.getenv("MINIO_ENABLE_COMPRESSION", "true").lower() == "true",
        compression_level=int(os.getenv("MINIO_COMPRESSION_LEVEL", "6")),
        
        # Encryption settings
        enable_encryption=os.getenv("MINIO_ENABLE_ENCRYPTION", "false").lower() == "true",
        encryption_key=os.getenv("MINIO_ENCRYPTION_KEY"),
        
        # Lifecycle settings
        lifecycle_enabled=os.getenv("MINIO_LIFECYCLE_ENABLED", "true").lower() == "true",
        default_retention_days=int(os.getenv("MINIO_DEFAULT_RETENTION_DAYS", "365")),
        raw_data_retention_days=int(os.getenv("MINIO_RAW_DATA_RETENTION_DAYS", "365")),
        processed_data_retention_days=int(os.getenv("MINIO_PROCESSED_DATA_RETENTION_DAYS", "180")),
        archived_data_retention_days=int(os.getenv("MINIO_ARCHIVED_DATA_RETENTION_DAYS", "2555")),
        
        # Versioning settings
        enable_versioning=os.getenv("MINIO_ENABLE_VERSIONING", "true").lower() == "true",
        
        # Monitoring settings
        enable_metrics=os.getenv("MINIO_ENABLE_METRICS", "true").lower() == "true",
        metrics_interval=int(os.getenv("MINIO_METRICS_INTERVAL", "60")),
        
        # Logging settings
        log_level=os.getenv("MINIO_LOG_LEVEL", "INFO"),
        log_requests=os.getenv("MINIO_LOG_REQUESTS", "false").lower() == "true",
        log_responses=os.getenv("MINIO_LOG_RESPONSES", "false").lower() == "true",
        
        # PBF-LB/M specific settings
        bucket_name=os.getenv("MINIO_BUCKET_NAME", "pbf-lbm-data")
    )


def create_minio_client_config() -> Dict[str, Any]:
    """Create MinIO client configuration dictionary."""
    config = get_minio_config()
    
    return {
        "endpoint": config.get_endpoint_url(),
        "access_key": config.access_key,
        "secret_key": config.secret_key,
        "secure": config.secure,
        "region": config.region,
        "timeout": config.timeout,
        "max_retries": config.max_retries,
        "retry_delay": config.retry_delay,
        "connection_pool_size": config.connection_pool_size,
        "max_connections": config.max_connections,
        "multipart_threshold": config.multipart_threshold,
        "multipart_chunksize": config.multipart_chunksize,
        "max_concurrency": config.max_concurrency,
        "enable_compression": config.enable_compression,
        "compression_level": config.compression_level,
        "enable_encryption": config.enable_encryption,
        "encryption_key": config.encryption_key,
        "bucket_name": config.bucket_name
    }
