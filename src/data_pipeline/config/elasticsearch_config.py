"""
Elasticsearch Configuration for PBF-LB/M Data Pipeline

This module provides Elasticsearch configuration management with environment variable support,
connection pooling, and production-ready settings.

Features:
- Environment variable configuration
- Connection pooling and timeout settings
- SSL/TLS support for production
- Health check configuration
- Search and analytics settings
"""

import os
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class ElasticsearchConfig(BaseModel):
    """Elasticsearch configuration model."""
    
    # Connection settings
    hosts: List[str] = Field(default_factory=lambda: ["localhost:9200"], description="Elasticsearch cluster hosts")
    username: Optional[str] = Field(default=None, description="Elasticsearch username")
    password: Optional[str] = Field(default=None, description="Elasticsearch password")
    
    # Connection pool settings
    max_connections: int = Field(default=20, description="Maximum connections in pool")
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    timeout: int = Field(default=30, description="Request timeout in seconds")
    retry_on_timeout: bool = Field(default=True, description="Retry on timeout")
    
    # SSL/TLS settings
    verify_certs: bool = Field(default=True, description="Verify SSL certificates")
    ssl_show_warn: bool = Field(default=True, description="Show SSL warnings")
    ca_certs: Optional[str] = Field(default=None, description="Path to CA certificates")
    client_cert: Optional[str] = Field(default=None, description="Path to client certificate")
    client_key: Optional[str] = Field(default=None, description="Path to client key")
    
    # Health check settings
    health_check_interval: int = Field(default=30, description="Health check interval in seconds")
    ping_interval: int = Field(default=60, description="Ping interval in seconds")
    
    # Search settings
    default_index: str = Field(default="pbf_process_data", description="Default index name")
    search_timeout: int = Field(default=30, description="Search timeout in seconds")
    max_result_window: int = Field(default=10000, description="Maximum result window size")
    
    # Index settings
    default_shards: int = Field(default=1, description="Default number of shards")
    default_replicas: int = Field(default=0, description="Default number of replicas")
    refresh_interval: str = Field(default="1s", description="Default refresh interval")
    
    # Bulk operations
    bulk_size: int = Field(default=1000, description="Bulk operation batch size")
    bulk_timeout: str = Field(default="30s", description="Bulk operation timeout")
    
    # Analytics settings
    enable_analytics: bool = Field(default=True, description="Enable analytics features")
    analytics_retention_days: int = Field(default=30, description="Analytics data retention in days")
    
    # Performance settings
    enable_compression: bool = Field(default=True, description="Enable response compression")
    compression_level: int = Field(default=6, description="Compression level (1-9)")
    
    # Monitoring settings
    enable_metrics: bool = Field(default=True, description="Enable metrics collection")
    metrics_interval: int = Field(default=60, description="Metrics collection interval in seconds")
    
    # Security settings
    api_key: Optional[str] = Field(default=None, description="Elasticsearch API key")
    bearer_token: Optional[str] = Field(default=None, description="Bearer token for authentication")
    
    # Advanced settings
    sniff_on_start: bool = Field(default=False, description="Sniff cluster nodes on start")
    sniff_on_connection_fail: bool = Field(default=True, description="Sniff on connection failure")
    sniffer_timeout: int = Field(default=10, description="Sniffer timeout in seconds")
    
    # PBF-LB/M specific settings
    process_index_prefix: str = Field(default="pbf_process", description="Process data index prefix")
    monitoring_index_prefix: str = Field(default="ispm_monitoring", description="Monitoring data index prefix")
    ct_scan_index_prefix: str = Field(default="ct_scan", description="CT scan data index prefix")
    powder_bed_index_prefix: str = Field(default="powder_bed", description="Powder bed data index prefix")
    research_index_prefix: str = Field(default="research", description="Research data index prefix")
    analytics_index_prefix: str = Field(default="analytics", description="Analytics data index prefix")
    
    # Data retention settings
    process_data_retention_days: int = Field(default=365, description="Process data retention in days")
    monitoring_data_retention_days: int = Field(default=90, description="Monitoring data retention in days")
    ct_scan_data_retention_days: int = Field(default=730, description="CT scan data retention in days")
    research_data_retention_days: int = Field(default=1825, description="Research data retention in days")
    
    # Search optimization
    enable_fuzzy_search: bool = Field(default=True, description="Enable fuzzy search")
    enable_autocomplete: bool = Field(default=True, description="Enable autocomplete")
    enable_highlighting: bool = Field(default=True, description="Enable search result highlighting")
    max_highlighted_fragments: int = Field(default=3, description="Maximum highlighted fragments")
    
    # Aggregation settings
    max_aggregation_buckets: int = Field(default=10000, description="Maximum aggregation buckets")
    aggregation_timeout: int = Field(default=60, description="Aggregation timeout in seconds")
    
    # Machine learning settings
    enable_ml: bool = Field(default=False, description="Enable machine learning features")
    ml_model_retention_days: int = Field(default=90, description="ML model retention in days")
    
    # Geospatial settings
    enable_geo_search: bool = Field(default=True, description="Enable geospatial search")
    geo_precision: int = Field(default=7, description="Geospatial precision level")
    
    # Text analysis settings
    custom_analyzers: Dict[str, Any] = Field(default_factory=dict, description="Custom text analyzers")
    synonym_files: List[str] = Field(default_factory=list, description="Synonym files for text analysis")
    
    # Backup and recovery
    enable_snapshots: bool = Field(default=True, description="Enable snapshot backups")
    snapshot_repository: str = Field(default="pbf_backups", description="Snapshot repository name")
    snapshot_schedule: str = Field(default="0 2 * * *", description="Snapshot schedule (cron format)")
    
    # Logging settings
    log_level: str = Field(default="INFO", description="Logging level")
    log_requests: bool = Field(default=False, description="Log all requests")
    log_responses: bool = Field(default=False, description="Log all responses")
    
    class Config:
        """Pydantic configuration."""
        env_prefix = "ELASTICSEARCH_"
        case_sensitive = False


def get_elasticsearch_config() -> ElasticsearchConfig:
    """
    Get Elasticsearch configuration from environment variables.
    
    Returns:
        ElasticsearchConfig: Configured Elasticsearch settings
    """
    return ElasticsearchConfig(
        hosts=[f"http://{os.getenv('ELASTICSEARCH_HOST', 'localhost')}:{os.getenv('ELASTICSEARCH_PORT', '9200')}"],
        username=os.getenv("ELASTICSEARCH_USERNAME"),
        password=os.getenv("ELASTICSEARCH_PASSWORD"),
        max_connections=int(os.getenv("ELASTICSEARCH_MAX_CONNECTIONS", "20")),
        max_retries=int(os.getenv("ELASTICSEARCH_MAX_RETRIES", "3")),
        timeout=int(os.getenv("ELASTICSEARCH_TIMEOUT", "30")),
        retry_on_timeout=os.getenv("ELASTICSEARCH_RETRY_ON_TIMEOUT", "true").lower() == "true",
        verify_certs=os.getenv("ELASTICSEARCH_VERIFY_CERTS", "true").lower() == "true",
        ssl_show_warn=os.getenv("ELASTICSEARCH_SSL_SHOW_WARN", "true").lower() == "true",
        ca_certs=os.getenv("ELASTICSEARCH_CA_CERTS"),
        client_cert=os.getenv("ELASTICSEARCH_CLIENT_CERT"),
        client_key=os.getenv("ELASTICSEARCH_CLIENT_KEY"),
        health_check_interval=int(os.getenv("ELASTICSEARCH_HEALTH_CHECK_INTERVAL", "30")),
        ping_interval=int(os.getenv("ELASTICSEARCH_PING_INTERVAL", "60")),
        default_index=os.getenv("ELASTICSEARCH_DEFAULT_INDEX", "pbf_process_data"),
        search_timeout=int(os.getenv("ELASTICSEARCH_SEARCH_TIMEOUT", "30")),
        max_result_window=int(os.getenv("ELASTICSEARCH_MAX_RESULT_WINDOW", "10000")),
        default_shards=int(os.getenv("ELASTICSEARCH_DEFAULT_SHARDS", "1")),
        default_replicas=int(os.getenv("ELASTICSEARCH_DEFAULT_REPLICAS", "0")),
        refresh_interval=os.getenv("ELASTICSEARCH_REFRESH_INTERVAL", "1s"),
        bulk_size=int(os.getenv("ELASTICSEARCH_BULK_SIZE", "1000")),
        bulk_timeout=os.getenv("ELASTICSEARCH_BULK_TIMEOUT", "30s"),
        enable_analytics=os.getenv("ELASTICSEARCH_ENABLE_ANALYTICS", "true").lower() == "true",
        analytics_retention_days=int(os.getenv("ELASTICSEARCH_ANALYTICS_RETENTION_DAYS", "30")),
        enable_compression=os.getenv("ELASTICSEARCH_ENABLE_COMPRESSION", "true").lower() == "true",
        compression_level=int(os.getenv("ELASTICSEARCH_COMPRESSION_LEVEL", "6")),
        enable_metrics=os.getenv("ELASTICSEARCH_ENABLE_METRICS", "true").lower() == "true",
        metrics_interval=int(os.getenv("ELASTICSEARCH_METRICS_INTERVAL", "60")),
        api_key=os.getenv("ELASTICSEARCH_API_KEY"),
        bearer_token=os.getenv("ELASTICSEARCH_BEARER_TOKEN"),
        sniff_on_start=os.getenv("ELASTICSEARCH_SNIFF_ON_START", "false").lower() == "true",
        sniff_on_connection_fail=os.getenv("ELASTICSEARCH_SNIFF_ON_CONNECTION_FAIL", "true").lower() == "true",
        sniffer_timeout=int(os.getenv("ELASTICSEARCH_SNIFFER_TIMEOUT", "10")),
        process_index_prefix=os.getenv("ELASTICSEARCH_PROCESS_INDEX_PREFIX", "pbf_process"),
        monitoring_index_prefix=os.getenv("ELASTICSEARCH_MONITORING_INDEX_PREFIX", "ispm_monitoring"),
        ct_scan_index_prefix=os.getenv("ELASTICSEARCH_CT_SCAN_INDEX_PREFIX", "ct_scan"),
        powder_bed_index_prefix=os.getenv("ELASTICSEARCH_POWDER_BED_INDEX_PREFIX", "powder_bed"),
        research_index_prefix=os.getenv("ELASTICSEARCH_RESEARCH_INDEX_PREFIX", "research"),
        analytics_index_prefix=os.getenv("ELASTICSEARCH_ANALYTICS_INDEX_PREFIX", "analytics"),
        process_data_retention_days=int(os.getenv("ELASTICSEARCH_PROCESS_DATA_RETENTION_DAYS", "365")),
        monitoring_data_retention_days=int(os.getenv("ELASTICSEARCH_MONITORING_DATA_RETENTION_DAYS", "90")),
        ct_scan_data_retention_days=int(os.getenv("ELASTICSEARCH_CT_SCAN_DATA_RETENTION_DAYS", "730")),
        research_data_retention_days=int(os.getenv("ELASTICSEARCH_RESEARCH_DATA_RETENTION_DAYS", "1825")),
        enable_fuzzy_search=os.getenv("ELASTICSEARCH_ENABLE_FUZZY_SEARCH", "true").lower() == "true",
        enable_autocomplete=os.getenv("ELASTICSEARCH_ENABLE_AUTOCOMPLETE", "true").lower() == "true",
        enable_highlighting=os.getenv("ELASTICSEARCH_ENABLE_HIGHLIGHTING", "true").lower() == "true",
        max_highlighted_fragments=int(os.getenv("ELASTICSEARCH_MAX_HIGHLIGHTED_FRAGMENTS", "3")),
        max_aggregation_buckets=int(os.getenv("ELASTICSEARCH_MAX_AGGREGATION_BUCKETS", "10000")),
        aggregation_timeout=int(os.getenv("ELASTICSEARCH_AGGREGATION_TIMEOUT", "60")),
        enable_ml=os.getenv("ELASTICSEARCH_ENABLE_ML", "false").lower() == "true",
        ml_model_retention_days=int(os.getenv("ELASTICSEARCH_ML_MODEL_RETENTION_DAYS", "90")),
        enable_geo_search=os.getenv("ELASTICSEARCH_ENABLE_GEO_SEARCH", "true").lower() == "true",
        geo_precision=int(os.getenv("ELASTICSEARCH_GEO_PRECISION", "7")),
        enable_snapshots=os.getenv("ELASTICSEARCH_ENABLE_SNAPSHOTS", "true").lower() == "true",
        snapshot_repository=os.getenv("ELASTICSEARCH_SNAPSHOT_REPOSITORY", "pbf_backups"),
        snapshot_schedule=os.getenv("ELASTICSEARCH_SNAPSHOT_SCHEDULE", "0 2 * * *"),
        log_level=os.getenv("ELASTICSEARCH_LOG_LEVEL", "INFO"),
        log_requests=os.getenv("ELASTICSEARCH_LOG_REQUESTS", "false").lower() == "true",
        log_responses=os.getenv("ELASTICSEARCH_LOG_RESPONSES", "false").lower() == "true"
    )


def get_elasticsearch_connection_string() -> str:
    """
    Get Elasticsearch connection string from environment variables.
    
    Returns:
        str: Elasticsearch connection string
    """
    url = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
    username = os.getenv("ELASTICSEARCH_USERNAME")
    password = os.getenv("ELASTICSEARCH_PASSWORD")
    
    if username and password:
        # Parse URL and add authentication
        if "://" in url:
            protocol, rest = url.split("://", 1)
            return f"{protocol}://{username}:{password}@{rest}"
        else:
            return f"http://{username}:{password}@{url}"
    
    return url


def get_elasticsearch_index_config(index_name: str) -> Dict[str, Any]:
    """
    Get Elasticsearch index configuration for a specific index.
    
    Args:
        index_name: Name of the index
        
    Returns:
        Dict[str, Any]: Index configuration
    """
    config = get_elasticsearch_config()
    
    # Base index settings
    index_config = {
        "settings": {
            "number_of_shards": config.default_shards,
            "number_of_replicas": config.default_replicas,
            "refresh_interval": config.refresh_interval,
            "max_result_window": config.max_result_window
        }
    }
    
    # Add index-specific settings
    if index_name.startswith(config.process_index_prefix):
        index_config["settings"]["number_of_shards"] = 3
        index_config["settings"]["number_of_replicas"] = 1
    elif index_name.startswith(config.monitoring_index_prefix):
        index_config["settings"]["number_of_shards"] = 2
        index_config["settings"]["number_of_replicas"] = 1
    elif index_name.startswith(config.ct_scan_index_prefix):
        index_config["settings"]["number_of_shards"] = 2
        index_config["settings"]["number_of_replicas"] = 1
    elif index_name.startswith(config.research_index_prefix):
        index_config["settings"]["number_of_shards"] = 1
        index_config["settings"]["number_of_replicas"] = 1
    
    return index_config


# Export the main configuration function
__all__ = ["ElasticsearchConfig", "get_elasticsearch_config", "get_elasticsearch_connection_string", "get_elasticsearch_index_config"]
