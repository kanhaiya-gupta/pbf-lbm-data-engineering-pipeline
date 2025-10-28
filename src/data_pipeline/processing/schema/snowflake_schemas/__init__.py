"""
Snowflake Schema Definitions

This module contains all Snowflake schema definitions for the PBF-LB/M manufacturing data pipeline.
Schemas are organized by data source and table type for modularity and maintainability.
"""

from .postgresql_schemas import *
from .mongodb_schemas import *
from .cassandra_schemas import *
from .redis_schemas import *
from .elasticsearch_schemas import *
from .neo4j_schemas import *
from .schema_factory import SnowflakeSchemaFactory

__all__ = [
    'SnowflakeSchemaFactory',
    # PostgreSQL schemas
    'POSTGRESQL_PBF_PROCESS_DATA_SCHEMA',
    'POSTGRESQL_POWDER_BED_DATA_SCHEMA', 
    'POSTGRESQL_CT_SCAN_DATA_SCHEMA',
    'POSTGRESQL_ISPM_MONITORING_DATA_SCHEMA',
    'POSTGRESQL_POWDER_BED_DEFECTS_SCHEMA',
    'POSTGRESQL_CT_SCAN_DEFECT_TYPES_SCHEMA',
    # MongoDB schemas
    'MONGODB_PROCESS_IMAGES_SCHEMA',
    'MONGODB_CT_SCAN_IMAGES_SCHEMA',
    'MONGODB_POWDER_BED_IMAGES_SCHEMA',
    'MONGODB_MACHINE_BUILD_FILES_SCHEMA',
    'MONGODB_MODEL_3D_FILES_SCHEMA',
    'MONGODB_RAW_SENSOR_DATA_SCHEMA',
    'MONGODB_PROCESS_LOGS_SCHEMA',
    'MONGODB_MACHINE_CONFIGURATIONS_SCHEMA',
    # Cassandra schemas
    'CASSANDRA_SENSOR_READINGS_SCHEMA',
    'CASSANDRA_MACHINE_EVENTS_SCHEMA',
    'CASSANDRA_PROCESS_MONITORING_SCHEMA',
    'CASSANDRA_ALERT_EVENTS_SCHEMA',
    # Redis schemas
    'REDIS_PROCESS_CACHE_SCHEMA',
    'REDIS_MACHINE_STATUS_CACHE_SCHEMA',
    'REDIS_SENSOR_READINGS_CACHE_SCHEMA',
    'REDIS_ANALYTICS_CACHE_SCHEMA',
    'REDIS_JOB_QUEUE_ITEMS_SCHEMA',
    'REDIS_USER_SESSIONS_SCHEMA',
    # Elasticsearch schemas
    'ELASTICSEARCH_PBF_PROCESS_SCHEMA',
    'ELASTICSEARCH_SENSOR_READINGS_SCHEMA',
    'ELASTICSEARCH_QUALITY_METRICS_SCHEMA',
    'ELASTICSEARCH_MACHINE_STATUS_SCHEMA',
    'ELASTICSEARCH_BUILD_INSTRUCTIONS_SCHEMA',
    'ELASTICSEARCH_ANALYTICS_SCHEMA',
    'ELASTICSEARCH_SEARCH_LOGS_SCHEMA',
    # Neo4j schemas
    'NEO4J_NODES_SCHEMA',
    'NEO4J_RELATIONSHIPS_SCHEMA',
]
