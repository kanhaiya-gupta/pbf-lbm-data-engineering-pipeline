"""
Neo4j Configuration Module for PBF-LB/M Data Warehouse

This module provides configuration management for Neo4j graph database connections,
including connection pooling, authentication, and performance settings.
"""

import os
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Neo4jConfig(BaseModel):
    """Neo4j configuration model with validation and defaults."""
    
    # Connection settings
    uri: str = Field(default="bolt://localhost:7687", description="Neo4j connection URI")
    username: str = Field(default="neo4j", description="Neo4j username")
    password: str = Field(default="password", description="Neo4j password")
    
    # Connection pool settings
    max_connection_lifetime: int = Field(default=3600, description="Max connection lifetime in seconds")
    max_connection_pool_size: int = Field(default=50, description="Max connections in pool")
    connection_acquisition_timeout: int = Field(default=60, description="Connection acquisition timeout in seconds")
    connection_timeout: int = Field(default=30, description="Connection timeout in seconds")
    
    # Transaction settings
    max_transaction_retry_time: int = Field(default=30, description="Max transaction retry time in seconds")
    max_retry_time: int = Field(default=30, description="Max retry time in seconds")
    
    # Performance settings
    max_execution_time: int = Field(default=300, description="Max query execution time in seconds")
    max_memory: int = Field(default=512, description="Max memory per query in MB")
    
    # Security settings
    encrypted: bool = Field(default=False, description="Enable encryption")
    trust: str = Field(default="TRUST_SYSTEM_CA_SIGNED_CERTIFICATES", description="Trust strategy")
    
    # Monitoring settings
    enable_metrics: bool = Field(default=True, description="Enable metrics collection")
    enable_tracing: bool = Field(default=False, description="Enable query tracing")
    
    # Graph settings
    default_database: str = Field(default="neo4j", description="Default database name")
    enable_apoc: bool = Field(default=True, description="Enable APOC procedures")
    enable_gds: bool = Field(default=True, description="Enable Graph Data Science library")
    
    # Knowledge graph specific settings
    node_batch_size: int = Field(default=1000, description="Batch size for node operations")
    relationship_batch_size: int = Field(default=1000, description="Batch size for relationship operations")
    index_batch_size: int = Field(default=1000, description="Batch size for index operations")
    
    # PBF-LB/M specific settings
    process_node_label: str = Field(default="Process", description="Label for process nodes")
    machine_node_label: str = Field(default="Machine", description="Label for machine nodes")
    part_node_label: str = Field(default="Part", description="Label for part nodes")
    sensor_node_label: str = Field(default="Sensor", description="Label for sensor nodes")
    build_node_label: str = Field(default="Build", description="Label for build nodes")
    
    # Relationship types
    process_uses_machine: str = Field(default="USES_MACHINE", description="Process uses machine relationship")
    process_creates_part: str = Field(default="CREATES_PART", description="Process creates part relationship")
    machine_has_sensor: str = Field(default="HAS_SENSOR", description="Machine has sensor relationship")
    part_belongs_to_build: str = Field(default="BELONGS_TO_BUILD", description="Part belongs to build relationship")
    sensor_monitors_process: str = Field(default="MONITORS_PROCESS", description="Sensor monitors process relationship")
    
    @validator('uri')
    def validate_uri(cls, v):
        """Validate Neo4j URI format."""
        if not v.startswith(('bolt://', 'neo4j://', 'bolt+s://', 'neo4j+s://')):
            raise ValueError("URI must start with bolt://, neo4j://, bolt+s://, or neo4j+s://")
        return v
    
    @validator('trust')
    def validate_trust(cls, v):
        """Validate trust strategy."""
        valid_trusts = [
            'TRUST_SYSTEM_CA_SIGNED_CERTIFICATES',
            'TRUST_ALL_CERTIFICATES',
            'TRUST_CUSTOM_CA_SIGNED_CERTIFICATES'
        ]
        if v not in valid_trusts:
            raise ValueError(f"Trust must be one of: {valid_trusts}")
        return v
    
    @validator('max_connection_pool_size')
    def validate_pool_size(cls, v):
        """Validate connection pool size."""
        if v < 1 or v > 1000:
            raise ValueError("Connection pool size must be between 1 and 1000")
        return v
    
    @validator('max_execution_time')
    def validate_execution_time(cls, v):
        """Validate max execution time."""
        if v < 1 or v > 3600:
            raise ValueError("Max execution time must be between 1 and 3600 seconds")
        return v
    
    class Config:
        """Pydantic configuration."""
        env_prefix = "NEO4J_"
        case_sensitive = False
        validate_assignment = True


def get_neo4j_config() -> Neo4jConfig:
    """
    Get Neo4j configuration from environment variables.
    
    Returns:
        Neo4jConfig: Configured Neo4j settings
    """
    return Neo4jConfig(
        uri=os.getenv('NEO4J_URI', 'bolt://localhost:7687'),
        username=os.getenv('NEO4J_USERNAME', 'neo4j'),
        password=os.getenv('NEO4J_PASSWORD', 'password'),
        max_connection_lifetime=int(os.getenv('NEO4J_MAX_CONNECTION_LIFETIME', '3600')),
        max_connection_pool_size=int(os.getenv('NEO4J_MAX_CONNECTION_POOL_SIZE', '50')),
        connection_acquisition_timeout=int(os.getenv('NEO4J_CONNECTION_ACQUISITION_TIMEOUT', '60')),
        connection_timeout=int(os.getenv('NEO4J_CONNECTION_TIMEOUT', '30')),
        max_transaction_retry_time=int(os.getenv('NEO4J_MAX_TRANSACTION_RETRY_TIME', '30')),
        max_retry_time=int(os.getenv('NEO4J_MAX_RETRY_TIME', '30')),
        max_execution_time=int(os.getenv('NEO4J_MAX_EXECUTION_TIME', '300')),
        max_memory=int(os.getenv('NEO4J_MAX_MEMORY', '512')),
        encrypted=os.getenv('NEO4J_ENCRYPTED', 'false').lower() == 'true',
        trust=os.getenv('NEO4J_TRUST', 'TRUST_SYSTEM_CA_SIGNED_CERTIFICATES'),
        enable_metrics=os.getenv('NEO4J_ENABLE_METRICS', 'true').lower() == 'true',
        enable_tracing=os.getenv('NEO4J_ENABLE_TRACING', 'false').lower() == 'true',
        default_database=os.getenv('NEO4J_DEFAULT_DATABASE', 'neo4j'),
        enable_apoc=os.getenv('NEO4J_ENABLE_APOC', 'true').lower() == 'true',
        enable_gds=os.getenv('NEO4J_ENABLE_GDS', 'true').lower() == 'true',
        node_batch_size=int(os.getenv('NEO4J_NODE_BATCH_SIZE', '1000')),
        relationship_batch_size=int(os.getenv('NEO4J_RELATIONSHIP_BATCH_SIZE', '1000')),
        index_batch_size=int(os.getenv('NEO4J_INDEX_BATCH_SIZE', '1000')),
        process_node_label=os.getenv('NEO4J_PROCESS_NODE_LABEL', 'Process'),
        machine_node_label=os.getenv('NEO4J_MACHINE_NODE_LABEL', 'Machine'),
        part_node_label=os.getenv('NEO4J_PART_NODE_LABEL', 'Part'),
        sensor_node_label=os.getenv('NEO4J_SENSOR_NODE_LABEL', 'Sensor'),
        build_node_label=os.getenv('NEO4J_BUILD_NODE_LABEL', 'Build'),
        process_uses_machine=os.getenv('NEO4J_PROCESS_USES_MACHINE', 'USES_MACHINE'),
        process_creates_part=os.getenv('NEO4J_PROCESS_CREATES_PART', 'CREATES_PART'),
        machine_has_sensor=os.getenv('NEO4J_MACHINE_HAS_SENSOR', 'HAS_SENSOR'),
        part_belongs_to_build=os.getenv('NEO4J_PART_BELONGS_TO_BUILD', 'BELONGS_TO_BUILD'),
        sensor_monitors_process=os.getenv('NEO4J_SENSOR_MONITORS_PROCESS', 'MONITORS_PROCESS')
    )


def get_neo4j_connection_config() -> Dict[str, Any]:
    """
    Get Neo4j connection configuration as dictionary.
    
    Returns:
        Dict[str, Any]: Connection configuration dictionary
    """
    config = get_neo4j_config()
    return {
        'uri': config.uri,
        'username': config.username,
        'password': config.password,
        'max_connection_lifetime': config.max_connection_lifetime,
        'max_connection_pool_size': config.max_connection_pool_size,
        'connection_acquisition_timeout': config.connection_acquisition_timeout,
        'connection_timeout': config.connection_timeout,
        'max_transaction_retry_time': config.max_transaction_retry_time,
        'max_retry_time': config.max_retry_time,
        'max_execution_time': config.max_execution_time,
        'max_memory': config.max_memory,
        'encrypted': config.encrypted,
        'trust': config.trust,
        'enable_metrics': config.enable_metrics,
        'enable_tracing': config.enable_tracing,
        'default_database': config.default_database
    }


def get_neo4j_graph_config() -> Dict[str, Any]:
    """
    Get Neo4j graph-specific configuration as dictionary.
    
    Returns:
        Dict[str, Any]: Graph configuration dictionary
    """
    config = get_neo4j_config()
    return {
        'node_batch_size': config.node_batch_size,
        'relationship_batch_size': config.relationship_batch_size,
        'index_batch_size': config.index_batch_size,
        'process_node_label': config.process_node_label,
        'machine_node_label': config.machine_node_label,
        'part_node_label': config.part_node_label,
        'sensor_node_label': config.sensor_node_label,
        'build_node_label': config.build_node_label,
        'process_uses_machine': config.process_uses_machine,
        'process_creates_part': config.process_creates_part,
        'machine_has_sensor': config.machine_has_sensor,
        'part_belongs_to_build': config.part_belongs_to_build,
        'sensor_monitors_process': config.sensor_monitors_process
    }


# Export the main configuration function
__all__ = ['Neo4jConfig', 'get_neo4j_config', 'get_neo4j_connection_config', 'get_neo4j_graph_config']
