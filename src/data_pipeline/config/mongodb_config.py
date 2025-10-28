"""
MongoDB Configuration

This module provides MongoDB configuration for the PBF-LB/M data pipeline.
"""

import os
from typing import List, Optional, Dict
from dataclasses import dataclass


@dataclass
class MongoDBConfig:
    """MongoDB configuration for PBF-LB/M unstructured data storage"""
    
    # Connection settings
    host: str = "localhost"
    port: int = 27017
    database: str = "pbf_data_lake"
    username: Optional[str] = None
    password: Optional[str] = None
    auth_source: str = "admin"
    
    # Connection pool settings
    max_pool_size: int = 100
    min_pool_size: int = 10
    max_idle_time_ms: int = 30000
    server_selection_timeout_ms: int = 5000
    connect_timeout_ms: int = 10000
    socket_timeout_ms: int = 0
    
    # Performance settings
    read_preference: str = "primary"
    write_concern: str = "majority"
    read_concern: str = "majority"
    
    # GridFS settings
    gridfs_bucket: str = "fs"
    chunk_size: int = 261120  # 255KB default
    
    # Index settings
    create_indexes: bool = True
    index_background: bool = True
    
    # Security settings
    ssl: bool = False
    ssl_cert_reqs: str = "CERT_REQUIRED"
    ssl_ca_certs: Optional[str] = None
    ssl_certfile: Optional[str] = None
    ssl_keyfile: Optional[str] = None
    
    # Replica set settings
    replica_set: Optional[str] = None
    read_preference_tags: Optional[List[Dict[str, str]]] = None
    
    def __post_init__(self):
        """Initialize configuration from environment variables."""
        # Load from environment variables if not explicitly set
        self.host = os.getenv('MONGO_HOST', self.host)
        self.port = int(os.getenv('MONGO_PORT', self.port))
        self.database = os.getenv('MONGO_DATABASE', self.database)
        self.username = os.getenv('MONGO_ROOT_USERNAME', self.username)
        self.password = os.getenv('MONGO_ROOT_PASSWORD', self.password)
        self.auth_source = os.getenv('MONGO_AUTH_SOURCE', self.auth_source)
        
        # Connection pool settings
        self.max_pool_size = int(os.getenv('MONGO_MAX_POOL_SIZE', self.max_pool_size))
        self.min_pool_size = int(os.getenv('MONGO_MIN_POOL_SIZE', self.min_pool_size))
        self.max_idle_time_ms = int(os.getenv('MONGO_MAX_IDLE_TIME_MS', self.max_idle_time_ms))
        self.server_selection_timeout_ms = int(os.getenv('MONGO_SERVER_SELECTION_TIMEOUT_MS', self.server_selection_timeout_ms))
        self.connect_timeout_ms = int(os.getenv('MONGO_CONNECT_TIMEOUT_MS', self.connect_timeout_ms))
        self.socket_timeout_ms = int(os.getenv('MONGO_SOCKET_TIMEOUT_MS', self.socket_timeout_ms))
        
        # Performance settings
        self.read_preference = os.getenv('MONGO_READ_PREFERENCE', self.read_preference)
        self.write_concern = os.getenv('MONGO_WRITE_CONCERN', self.write_concern)
        self.read_concern = os.getenv('MONGO_READ_CONCERN', self.read_concern)
        
        # GridFS settings
        self.gridfs_bucket = os.getenv('MONGO_GRIDFS_BUCKET', self.gridfs_bucket)
        self.chunk_size = int(os.getenv('MONGO_CHUNK_SIZE', self.chunk_size))
        
        # Index settings
        self.create_indexes = os.getenv('MONGO_CREATE_INDEXES', 'true').lower() == 'true'
        self.index_background = os.getenv('MONGO_INDEX_BACKGROUND', 'true').lower() == 'true'
        
        # Security settings
        self.ssl = os.getenv('MONGO_SSL', 'false').lower() == 'true'
        self.ssl_cert_reqs = os.getenv('MONGO_SSL_CERT_REQS', self.ssl_cert_reqs)
        self.ssl_ca_certs = os.getenv('MONGO_SSL_CA_CERTS', self.ssl_ca_certs)
        self.ssl_certfile = os.getenv('MONGO_SSL_CERTFILE', self.ssl_certfile)
        self.ssl_keyfile = os.getenv('MONGO_SSL_KEYFILE', self.ssl_keyfile)
        
        # Replica set settings
        self.replica_set = os.getenv('MONGO_REPLICA_SET', self.replica_set)
        
        # Parse read preference tags if provided
        read_pref_tags = os.getenv('MONGO_READ_PREFERENCE_TAGS')
        if read_pref_tags:
            try:
                import json
                self.read_preference_tags = json.loads(read_pref_tags)
            except (json.JSONDecodeError, ValueError):
                self.read_preference_tags = None


def get_mongodb_config() -> MongoDBConfig:
    """
    Get MongoDB configuration.
    
    Returns:
        MongoDBConfig: MongoDB configuration object
    """
    return MongoDBConfig()
