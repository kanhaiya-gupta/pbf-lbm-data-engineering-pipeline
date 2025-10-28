#!/usr/bin/env python3
"""
Redis Configuration for PBF-LB/M Data Pipeline

This module provides Redis configuration management with environment variable support,
connection pooling, and production-ready settings.

Features:
- Environment variable configuration
- Connection pooling and timeout settings
- SSL/TLS support for production
- Health check configuration
- Cache-specific settings
"""

import os
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class RedisConfig(BaseModel):
    """Redis configuration model."""
    
    # Connection settings
    host: str = Field(default="localhost", description="Redis host")
    port: int = Field(default=6379, description="Redis port")
    db: int = Field(default=0, description="Redis database number")
    password: Optional[str] = Field(default=None, description="Redis password")
    
    # Connection pool settings
    max_connections: int = Field(default=20, description="Maximum connections in pool")
    retry_on_timeout: bool = Field(default=True, description="Retry on timeout")
    socket_timeout: int = Field(default=5, description="Socket timeout in seconds")
    socket_connect_timeout: int = Field(default=5, description="Socket connect timeout in seconds")
    
    # SSL/TLS settings
    ssl: bool = Field(default=False, description="Enable SSL")
    ssl_cert_reqs: str = Field(default="required", description="SSL certificate requirements")
    ssl_ca_certs: Optional[str] = Field(default=None, description="SSL CA certificates path")
    ssl_certfile: Optional[str] = Field(default=None, description="SSL certificate file")
    ssl_keyfile: Optional[str] = Field(default=None, description="SSL key file")
    
    # Health check settings
    health_check_interval: int = Field(default=30, description="Health check interval in seconds")
    ping_interval: int = Field(default=60, description="Ping interval in seconds")
    
    # Cache settings
    default_ttl: int = Field(default=3600, description="Default TTL in seconds")
    key_prefix: str = Field(default="pbf:", description="Key prefix for namespacing")
    
    # Performance settings
    decode_responses: bool = Field(default=True, description="Decode responses to strings")
    encoding: str = Field(default="utf-8", description="String encoding")
    
    class Config:
        """Pydantic configuration."""
        env_prefix = "REDIS_"
        case_sensitive = False

def get_redis_config() -> RedisConfig:
    """
    Get Redis configuration from environment variables.
    
    Returns:
        RedisConfig: Redis configuration object
    """
    return RedisConfig(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", "6379")),
        db=int(os.getenv("REDIS_DB", "0")),
        password=os.getenv("REDIS_PASSWORD"),
        max_connections=int(os.getenv("REDIS_MAX_CONNECTIONS", "20")),
        retry_on_timeout=os.getenv("REDIS_RETRY_ON_TIMEOUT", "true").lower() == "true",
        socket_timeout=int(os.getenv("REDIS_SOCKET_TIMEOUT", "5")),
        socket_connect_timeout=int(os.getenv("REDIS_SOCKET_CONNECT_TIMEOUT", "5")),
        ssl=os.getenv("REDIS_SSL", "false").lower() == "true",
        ssl_cert_reqs=os.getenv("REDIS_SSL_CERT_REQS", "required"),
        ssl_ca_certs=os.getenv("REDIS_SSL_CA_CERTS"),
        ssl_certfile=os.getenv("REDIS_SSL_CERTFILE"),
        ssl_keyfile=os.getenv("REDIS_SSL_KEYFILE"),
        health_check_interval=int(os.getenv("REDIS_HEALTH_CHECK_INTERVAL", "30")),
        ping_interval=int(os.getenv("REDIS_PING_INTERVAL", "60")),
        default_ttl=int(os.getenv("REDIS_DEFAULT_TTL", "3600")),
        key_prefix=os.getenv("REDIS_KEY_PREFIX", "pbf:"),
        decode_responses=os.getenv("REDIS_DECODE_RESPONSES", "true").lower() == "true",
        encoding=os.getenv("REDIS_ENCODING", "utf-8")
    )

def get_redis_connection_params() -> Dict[str, Any]:
    """
    Get Redis connection parameters as dictionary.
    
    Returns:
        Dict[str, Any]: Redis connection parameters
    """
    config = get_redis_config()
    
    return {
        "host": config.host,
        "port": config.port,
        "db": config.db,
        "password": config.password,
        "max_connections": config.max_connections,
        "retry_on_timeout": config.retry_on_timeout,
        "socket_timeout": config.socket_timeout,
        "socket_connect_timeout": config.socket_connect_timeout,
        "ssl": config.ssl,
        "ssl_cert_reqs": config.ssl_cert_reqs,
        "ssl_ca_certs": config.ssl_ca_certs,
        "ssl_certfile": config.ssl_certfile,
        "ssl_keyfile": config.ssl_keyfile,
        "decode_responses": config.decode_responses,
        "encoding": config.encoding
    }

def get_cache_settings() -> Dict[str, Any]:
    """
    Get cache-specific settings.
    
    Returns:
        Dict[str, Any]: Cache settings
    """
    config = get_redis_config()
    
    return {
        "default_ttl": config.default_ttl,
        "key_prefix": config.key_prefix,
        "health_check_interval": config.health_check_interval,
        "ping_interval": config.ping_interval
    }

# Cache type specific TTL settings
CACHE_TTL_SETTINGS = {
    "process_cache": 3600,      # 1 hour
    "machine_status": 1800,      # 30 minutes
    "sensor_reading": 300,       # 5 minutes
    "analytics": 7200,          # 2 hours
    "job_queue": 86400,          # 1 day
    "user_session": 3600         # 1 hour
}

def get_cache_ttl(cache_type: str) -> int:
    """
    Get TTL for specific cache type.
    
    Args:
        cache_type: Type of cache
        
    Returns:
        int: TTL in seconds
    """
    return CACHE_TTL_SETTINGS.get(cache_type, 3600)

def get_namespaced_key(key: str, cache_type: str = None) -> str:
    """
    Get namespaced Redis key.
    
    Args:
        key: Base key
        cache_type: Type of cache for namespacing
        
    Returns:
        str: Namespaced key
    """
    config = get_redis_config()
    prefix = config.key_prefix
    
    if cache_type:
        return f"{prefix}{cache_type}:{key}"
    else:
        return f"{prefix}{key}"