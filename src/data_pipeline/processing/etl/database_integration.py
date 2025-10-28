"""
Database Integration

This module provides database integration capabilities for ETL operations in the PBF-LB/M data pipeline,
including both traditional SQL databases and NoSQL databases.
"""

import pandas as pd
import psycopg2
from sqlalchemy import create_engine, text
from typing import Dict, List, Optional, Any, Union
import logging
from contextlib import contextmanager
import json

from src.data_pipeline.config.pipeline_config import get_pipeline_config
from src.data_pipeline.storage.operational.mongodb_client import MongoDBClient
from src.data_pipeline.storage.operational.elasticsearch_client import ElasticsearchClient
from src.data_pipeline.storage.operational.redis_client import RedisClient
from src.data_pipeline.storage.operational.cassandra_client import CassandraClient
from src.data_pipeline.storage.operational.neo4j_client import Neo4jClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseIntegration:
    """
    Database integration service for ETL operations.
    """
    
    def __init__(self):
        self.config = get_pipeline_config()
        self.connections = {}
        self._initialize_connections()
    
    def _initialize_connections(self):
        """Initialize database connections."""
        try:
            # PostgreSQL connection
            postgres_config = self.config.get('postgres', {})
            if postgres_config:
                self.connections['postgres'] = create_engine(
                    f"postgresql://{postgres_config['user']}:{postgres_config['password']}@"
                    f"{postgres_config['host']}:{postgres_config['port']}/{postgres_config['database']}"
                )
            
            # NoSQL connections
            self._initialize_nosql_connections()
            
            logger.info("Database connections initialized")
            
        except Exception as e:
            logger.error(f"Error initializing database connections: {e}")
    
    def _initialize_nosql_connections(self):
        """Initialize NoSQL database connections."""
        try:
            # MongoDB connection
            mongodb_config = self.config.get('mongodb', {})
            if mongodb_config:
                self.connections['mongodb'] = MongoDBClient(
                    connection_string=mongodb_config.get('connection_string'),
                    database_name=mongodb_config.get('database_name')
                )
                self.connections['mongodb'].connect()
            
            # Redis connection
            redis_config = self.config.get('redis', {})
            if redis_config:
                self.connections['redis'] = RedisClient(
                    host=redis_config.get('host', 'localhost'),
                    port=redis_config.get('port', 6379),
                    password=redis_config.get('password'),
                    db=redis_config.get('db', 0)
                )
                self.connections['redis'].connect()
            
            # Cassandra connection
            cassandra_config = self.config.get('cassandra', {})
            if cassandra_config:
                self.connections['cassandra'] = CassandraClient(
                    hosts=cassandra_config.get('hosts', ['localhost']),
                    keyspace=cassandra_config.get('keyspace'),
                    username=cassandra_config.get('username'),
                    password=cassandra_config.get('password')
                )
                self.connections['cassandra'].connect()
            
            # Elasticsearch connection
            elasticsearch_config = self.config.get('elasticsearch', {})
            if elasticsearch_config:
                self.connections['elasticsearch'] = ElasticsearchClient(
                    hosts=elasticsearch_config.get('hosts', ['localhost']),
                    username=elasticsearch_config.get('username'),
                    password=elasticsearch_config.get('password')
                )
                self.connections['elasticsearch'].connect()
            
            # Neo4j connection
            neo4j_config = self.config.get('neo4j', {})
            if neo4j_config:
                self.connections['neo4j'] = Neo4jClient(
                    uri=neo4j_config.get('uri'),
                    username=neo4j_config.get('username'),
                    password=neo4j_config.get('password'),
                    database=neo4j_config.get('database', 'neo4j')
                )
                self.connections['neo4j'].connect()
            
            logger.info("NoSQL database connections initialized")
            
        except Exception as e:
            logger.error(f"Error initializing NoSQL connections: {e}")
    
    @contextmanager
    def get_connection(self, db_type: str = 'postgres'):
        """Get database connection with context manager."""
        try:
            connection = self.connections.get(db_type)
            if not connection:
                raise ValueError(f"Database connection not found: {db_type}")
            
            conn = connection.connect()
            try:
                yield conn
            finally:
                conn.close()
                
        except Exception as e:
            logger.error(f"Error getting database connection: {e}")
            raise
    
    def execute_query(self, query: str, db_type: str = 'postgres', params: Optional[Dict] = None) -> pd.DataFrame:
        """Execute a query and return results as DataFrame."""
        try:
            with self.get_connection(db_type) as conn:
                result = pd.read_sql_query(text(query), conn, params=params)
                logger.info(f"Query executed successfully, returned {len(result)} rows")
                return result
                
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            raise
    
    def insert_data(self, table_name: str, data: pd.DataFrame, db_type: str = 'postgres', 
                   if_exists: str = 'append') -> bool:
        """Insert data into database table."""
        try:
            with self.get_connection(db_type) as conn:
                data.to_sql(table_name, conn, if_exists=if_exists, index=False)
                logger.info(f"Data inserted successfully into {table_name}")
                return True
                
        except Exception as e:
            logger.error(f"Error inserting data: {e}")
            return False
    
    def update_data(self, table_name: str, data: pd.DataFrame, where_clause: str, 
                   db_type: str = 'postgres') -> bool:
        """Update data in database table."""
        try:
            with self.get_connection(db_type) as conn:
                for _, row in data.iterrows():
                    update_query = f"UPDATE {table_name} SET "
                    set_clauses = []
                    for col, value in row.items():
                        if col != 'id':  # Assuming id is the primary key
                            set_clauses.append(f"{col} = %s")
                    
                    update_query += ", ".join(set_clauses)
                    update_query += f" WHERE {where_clause}"
                    
                    conn.execute(text(update_query), list(row.values()))
                
                logger.info(f"Data updated successfully in {table_name}")
                return True
                
        except Exception as e:
            logger.error(f"Error updating data: {e}")
            return False
    
    def delete_data(self, table_name: str, where_clause: str, db_type: str = 'postgres') -> bool:
        """Delete data from database table."""
        try:
            with self.get_connection(db_type) as conn:
                delete_query = f"DELETE FROM {table_name} WHERE {where_clause}"
                result = conn.execute(text(delete_query))
                logger.info(f"Deleted {result.rowcount} rows from {table_name}")
                return True
                
        except Exception as e:
            logger.error(f"Error deleting data: {e}")
            return False
    
    # =============================================================================
    # NoSQL Database Operations
    # =============================================================================
    
    def insert_document(self, db_type: str, collection_name: str, document: Dict[str, Any]) -> bool:
        """
        Insert a document into NoSQL database.
        
        Args:
            db_type: Type of database (mongodb, elasticsearch)
            collection_name: Collection/index name
            document: Document to insert
            
        Returns:
            bool: True if successful
        """
        try:
            if db_type == 'mongodb':
                client = self.connections.get('mongodb')
                if client:
                    client.insert_document(collection_name, document)
                    return True
            
            elif db_type == 'elasticsearch':
                client = self.connections.get('elasticsearch')
                if client:
                    client.index_document(collection_name, document)
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error inserting document: {e}")
            return False
    
    def get_document(self, db_type: str, collection_name: str, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a document from NoSQL database.
        
        Args:
            db_type: Type of database (mongodb, elasticsearch)
            collection_name: Collection/index name
            doc_id: Document ID
            
        Returns:
            Optional[Dict]: Document if found
        """
        try:
            if db_type == 'mongodb':
                client = self.connections.get('mongodb')
                if client:
                    return client.find_document(collection_name, {"_id": doc_id})
            
            elif db_type == 'elasticsearch':
                client = self.connections.get('elasticsearch')
                if client:
                    return client.get_document(collection_name, doc_id)
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting document: {e}")
            return None
    
    def set_cache(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """
        Set a value in Redis cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
            
        Returns:
            bool: True if successful
        """
        try:
            client = self.connections.get('redis')
            if client:
                return client.cache_set(key, value, ttl)
            return False
            
        except Exception as e:
            logger.error(f"Error setting cache: {e}")
            return False
    
    def get_cache(self, key: str) -> Optional[Any]:
        """
        Get a value from Redis cache.
        
        Args:
            key: Cache key
            
        Returns:
            Optional[Any]: Cached value if found
        """
        try:
            client = self.connections.get('redis')
            if client:
                return client.cache_get(key)
            return None
            
        except Exception as e:
            logger.error(f"Error getting cache: {e}")
            return None
    
    def insert_time_series(self, sensor_id: str, timestamp: str, value: float, 
                          quality: int = 1, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Insert time-series data into Cassandra.
        
        Args:
            sensor_id: Sensor identifier
            timestamp: Timestamp
            value: Sensor value
            quality: Data quality score
            metadata: Optional metadata
            
        Returns:
            bool: True if successful
        """
        try:
            client = self.connections.get('cassandra')
            if client:
                data = {
                    'sensor_id': sensor_id,
                    'timestamp': timestamp,
                    'value': value,
                    'quality': quality,
                    'metadata': json.dumps(metadata) if metadata else None
                }
                return client.insert_row('sensor_data', data)
            return False
            
        except Exception as e:
            logger.error(f"Error inserting time-series data: {e}")
            return False
    
    def create_graph_node(self, label: str, properties: Dict[str, Any]) -> Optional[int]:
        """
        Create a node in Neo4j graph.
        
        Args:
            label: Node label
            properties: Node properties
            
        Returns:
            Optional[int]: Node ID if successful
        """
        try:
            client = self.connections.get('neo4j')
            if client:
                return client.create_node(label, properties)
            return None
            
        except Exception as e:
            logger.error(f"Error creating graph node: {e}")
            return None
    
    def create_graph_relationship(self, from_node_id: int, to_node_id: int,
                                 relationship_type: str, properties: Optional[Dict[str, Any]] = None) -> Optional[int]:
        """
        Create a relationship in Neo4j graph.
        
        Args:
            from_node_id: Source node ID
            to_node_id: Target node ID
            relationship_type: Type of relationship
            properties: Optional relationship properties
            
        Returns:
            Optional[int]: Relationship ID if successful
        """
        try:
            client = self.connections.get('neo4j')
            if client:
                return client.create_relationship(from_node_id, to_node_id, relationship_type, properties)
            return None
            
        except Exception as e:
            logger.error(f"Error creating graph relationship: {e}")
            return None
    
    def search_documents(self, db_type: str, collection_name: str, query: Dict[str, Any],
                        limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search documents in NoSQL database.
        
        Args:
            db_type: Type of database (mongodb, elasticsearch)
            collection_name: Collection/index name
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List[Dict]: Search results
        """
        try:
            if db_type == 'mongodb':
                client = self.connections.get('mongodb')
                if client:
                    return client.find_documents(collection_name, query, limit=limit)
            
            elif db_type == 'elasticsearch':
                client = self.connections.get('elasticsearch')
                if client:
                    result = client.search(collection_name, {"query": query}, size=limit)
                    return result.get('hits', [])
            
            return []
            
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []
    
    def get_time_series_data(self, sensor_id: str, start_time: Optional[str] = None,
                            end_time: Optional[str] = None, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get time-series data from Cassandra.
        
        Args:
            sensor_id: Sensor identifier
            start_time: Optional start time
            end_time: Optional end time
            limit: Optional limit on results
            
        Returns:
            List[Dict]: Time-series data
        """
        try:
            client = self.connections.get('cassandra')
            if client:
                return client.select_time_series('sensor_data', sensor_id, start_time, end_time, limit)
            return []
            
        except Exception as e:
            logger.error(f"Error getting time-series data: {e}")
            return []
    
    def get_graph_path(self, from_node_id: int, to_node_id: int,
                      relationship_types: Optional[List[str]] = None) -> Optional[List[Dict[str, Any]]]:
        """
        Get shortest path between nodes in Neo4j graph.
        
        Args:
            from_node_id: Source node ID
            to_node_id: Target node ID
            relationship_types: Optional relationship types to follow
            
        Returns:
            Optional[List[Dict]]: Path if found
        """
        try:
            client = self.connections.get('neo4j')
            if client:
                return client.get_shortest_path(from_node_id, to_node_id, relationship_types)
            return None
            
        except Exception as e:
            logger.error(f"Error getting graph path: {e}")
            return None
    
    def close_all_connections(self):
        """Close all database connections."""
        try:
            for db_type, connection in self.connections.items():
                if hasattr(connection, 'disconnect'):
                    connection.disconnect()
                elif hasattr(connection, 'close'):
                    connection.close()
            
            logger.info("All database connections closed")
            
        except Exception as e:
            logger.error(f"Error closing connections: {e}")
