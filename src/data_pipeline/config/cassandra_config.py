"""
Cassandra configuration management for PBF-LB/M data pipeline.
"""

import os
from typing import Optional, List
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class CassandraConfig(BaseModel):
    """Cassandra configuration model."""
    
    # Connection settings
    hosts: List[str] = Field(default_factory=lambda: ["localhost"], description="Cassandra cluster hosts")
    port: int = Field(default=9042, description="Cassandra native protocol port")
    keyspace: str = Field(default="pbf_timeseries", description="Default keyspace name")
    
    # Authentication
    username: Optional[str] = Field(default=None, description="Cassandra username")
    password: Optional[str] = Field(default=None, description="Cassandra password")
    
    # Connection pool settings
    max_connections: int = Field(default=50, description="Maximum connections per host")
    max_requests_per_connection: int = Field(default=32768, description="Maximum requests per connection")
    connection_timeout: int = Field(default=10, description="Connection timeout in seconds")
    request_timeout: int = Field(default=10, description="Request timeout in seconds")
    
    # Compression and performance
    compression: str = Field(default="LZ4", description="Compression algorithm")
    protocol_version: int = Field(default=3, description="Native protocol version")
    
    # Consistency settings
    default_consistency: str = Field(default="LOCAL_QUORUM", description="Default consistency level")
    serial_consistency: str = Field(default="LOCAL_SERIAL", description="Serial consistency level")
    
    # Retry and reconnection
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    retry_delay: float = Field(default=1.0, description="Retry delay in seconds")
    reconnect_delay: float = Field(default=2.0, description="Reconnection delay in seconds")
    
    # SSL/TLS settings
    ssl_enabled: bool = Field(default=False, description="Enable SSL/TLS encryption")
    ssl_certfile: Optional[str] = Field(default=None, description="SSL certificate file")
    ssl_keyfile: Optional[str] = Field(default=None, description="SSL private key file")
    ssl_ca_certs: Optional[str] = Field(default=None, description="SSL CA certificates file")
    ssl_check_hostname: bool = Field(default=True, description="Check SSL hostname")
    
    # Monitoring and metrics
    enable_metrics: bool = Field(default=True, description="Enable connection metrics")
    enable_tracing: bool = Field(default=False, description="Enable query tracing")
    
    # Time-series specific settings
    default_ttl: int = Field(default=31536000, description="Default TTL in seconds (1 year)")
    batch_size: int = Field(default=100, description="Default batch size for inserts")
    max_batch_size: int = Field(default=1000, description="Maximum batch size")
    
    class Config:
        env_prefix = "CASSANDRA_"
        case_sensitive = False


def get_cassandra_config() -> CassandraConfig:
    """
    Get Cassandra configuration from environment variables.
    
    Returns:
        CassandraConfig: Configured Cassandra settings
    """
    return CassandraConfig(
        hosts=os.getenv("CASSANDRA_HOSTS", "localhost").split(","),
        port=int(os.getenv("CASSANDRA_PORT", "9042")),
        keyspace=os.getenv("CASSANDRA_KEYSPACE", "pbf_timeseries"),
        username=os.getenv("CASSANDRA_USERNAME"),
        password=os.getenv("CASSANDRA_PASSWORD"),
        max_connections=int(os.getenv("CASSANDRA_MAX_CONNECTIONS", "50")),
        max_requests_per_connection=int(os.getenv("CASSANDRA_MAX_REQUESTS_PER_CONNECTION", "32768")),
        connection_timeout=int(os.getenv("CASSANDRA_CONNECTION_TIMEOUT", "10")),
        request_timeout=int(os.getenv("CASSANDRA_REQUEST_TIMEOUT", "10")),
        compression=os.getenv("CASSANDRA_COMPRESSION", "LZ4"),
        protocol_version=int(os.getenv("CASSANDRA_PROTOCOL_VERSION", "3")),
        default_consistency=os.getenv("CASSANDRA_DEFAULT_CONSISTENCY", "LOCAL_QUORUM"),
        serial_consistency=os.getenv("CASSANDRA_SERIAL_CONSISTENCY", "LOCAL_SERIAL"),
        max_retries=int(os.getenv("CASSANDRA_MAX_RETRIES", "3")),
        retry_delay=float(os.getenv("CASSANDRA_RETRY_DELAY", "1.0")),
        reconnect_delay=float(os.getenv("CASSANDRA_RECONNECT_DELAY", "2.0")),
        ssl_enabled=os.getenv("CASSANDRA_SSL_ENABLED", "false").lower() == "true",
        ssl_certfile=os.getenv("CASSANDRA_SSL_CERTFILE"),
        ssl_keyfile=os.getenv("CASSANDRA_SSL_KEYFILE"),
        ssl_ca_certs=os.getenv("CASSANDRA_SSL_CA_CERTS"),
        ssl_check_hostname=os.getenv("CASSANDRA_SSL_CHECK_HOSTNAME", "true").lower() == "true",
        enable_metrics=os.getenv("CASSANDRA_ENABLE_METRICS", "true").lower() == "true",
        enable_tracing=os.getenv("CASSANDRA_ENABLE_TRACING", "false").lower() == "true",
        default_ttl=int(os.getenv("CASSANDRA_DEFAULT_TTL", "31536000")),
        batch_size=int(os.getenv("CASSANDRA_BATCH_SIZE", "100")),
        max_batch_size=int(os.getenv("CASSANDRA_MAX_BATCH_SIZE", "1000"))
    )


def get_cassandra_connection_string() -> str:
    """
    Get Cassandra connection string for logging and debugging.
    
    Returns:
        str: Connection string (without password)
    """
    config = get_cassandra_config()
    hosts_str = ",".join(config.hosts)
    return f"cassandra://{hosts_str}:{config.port}/{config.keyspace}"
