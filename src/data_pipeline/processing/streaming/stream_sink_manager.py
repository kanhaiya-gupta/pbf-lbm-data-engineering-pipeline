"""
Stream Sink Manager for NoSQL Databases

This module provides a unified interface for writing streaming data to various
NoSQL databases in the PBF-LB/M data pipeline. It handles connection management,
error recovery, and provides consistent configuration across all streaming processors.
"""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import logging
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

from src.data_pipeline.storage.operational.mongodb_client import MongoDBClient
from src.data_pipeline.storage.operational.elasticsearch_client import ElasticsearchClient
from src.data_pipeline.storage.operational.redis_client import RedisClient
from src.data_pipeline.storage.operational.cassandra_client import CassandraClient
from src.data_pipeline.storage.operational.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)


class SinkType(Enum):
    """Enumeration of supported sink types."""
    MONGODB = "mongodb"
    REDIS = "redis"
    CASSANDRA = "cassandra"
    ELASTICSEARCH = "elasticsearch"
    NEO4J = "neo4j"


@dataclass
class SinkConfig:
    """Configuration for a stream sink."""
    sink_type: SinkType
    connection_config: Dict[str, Any]
    write_config: Dict[str, Any]
    retry_config: Dict[str, Any]
    batch_config: Dict[str, Any]


class StreamSink(ABC):
    """Abstract base class for stream sinks."""
    
    def __init__(self, config: SinkConfig):
        self.config = config
        self.client = None
        self.batch_buffer = []
        self.batch_size = config.batch_config.get('batch_size', 100)
        self.batch_timeout = config.batch_config.get('timeout_seconds', 5)
        self.last_batch_time = datetime.utcnow()
    
    @abstractmethod
    def connect(self) -> bool:
        """Establish connection to the sink."""
        pass
    
    @abstractmethod
    def disconnect(self):
        """Close connection to the sink."""
        pass
    
    @abstractmethod
    def write_record(self, record: Dict[str, Any]) -> bool:
        """Write a single record to the sink."""
        pass
    
    @abstractmethod
    def write_batch(self, records: List[Dict[str, Any]]) -> bool:
        """Write a batch of records to the sink."""
        pass
    
    def add_to_batch(self, record: Dict[str, Any]) -> bool:
        """Add record to batch buffer and flush if needed."""
        try:
            self.batch_buffer.append(record)
            
            # Check if batch should be flushed
            should_flush = (
                len(self.batch_buffer) >= self.batch_size or
                (datetime.utcnow() - self.last_batch_time).seconds >= self.batch_timeout
            )
            
            if should_flush:
                return self.flush_batch()
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding record to batch: {e}")
            return False
    
    def flush_batch(self) -> bool:
        """Flush the current batch buffer."""
        try:
            if not self.batch_buffer:
                return True
            
            success = self.write_batch(self.batch_buffer.copy())
            if success:
                self.batch_buffer.clear()
                self.last_batch_time = datetime.utcnow()
            
            return success
            
        except Exception as e:
            logger.error(f"Error flushing batch: {e}")
            return False


class MongoDBStreamSink(StreamSink):
    """MongoDB stream sink implementation."""
    
    def connect(self) -> bool:
        """Establish connection to MongoDB."""
        try:
            self.client = MongoDBClient(
                connection_string=self.config.connection_config['connection_string'],
                database_name=self.config.connection_config['database_name']
            )
            return self.client.connect()
            
        except Exception as e:
            logger.error(f"Error connecting to MongoDB: {e}")
            return False
    
    def disconnect(self):
        """Close MongoDB connection."""
        if self.client:
            self.client.disconnect()
    
    def write_record(self, record: Dict[str, Any]) -> bool:
        """Write a single record to MongoDB."""
        try:
            if not self.client:
                return False
            
            collection_name = self.config.write_config.get('collection_name', 'stream_data')
            doc_id = self.client.insert_document(collection_name, record)
            return doc_id is not None
            
        except Exception as e:
            logger.error(f"Error writing record to MongoDB: {e}")
            return False
    
    def write_batch(self, records: List[Dict[str, Any]]) -> bool:
        """Write a batch of records to MongoDB."""
        try:
            if not self.client:
                return False
            
            collection_name = self.config.write_config.get('collection_name', 'stream_data')
            doc_ids = self.client.insert_documents(collection_name, records)
            return len(doc_ids) == len(records)
            
        except Exception as e:
            logger.error(f"Error writing batch to MongoDB: {e}")
            return False


class RedisStreamSink(StreamSink):
    """Redis stream sink implementation."""
    
    def connect(self) -> bool:
        """Establish connection to Redis."""
        try:
            self.client = RedisClient(
                host=self.config.connection_config['host'],
                port=self.config.connection_config.get('port', 6379),
                password=self.config.connection_config.get('password'),
                db=self.config.connection_config.get('db', 0)
            )
            return self.client.connect()
            
        except Exception as e:
            logger.error(f"Error connecting to Redis: {e}")
            return False
    
    def disconnect(self):
        """Close Redis connection."""
        if self.client:
            self.client.disconnect()
    
    def write_record(self, record: Dict[str, Any]) -> bool:
        """Write a single record to Redis."""
        try:
            if not self.client:
                return False
            
            key_prefix = self.config.write_config.get('key_prefix', 'stream:')
            ttl = self.config.write_config.get('ttl', 3600)
            
            # Generate key from record
            key = self._generate_key(record, key_prefix)
            value = json.dumps(record)
            
            return self.client.cache_set(key, value, ttl)
            
        except Exception as e:
            logger.error(f"Error writing record to Redis: {e}")
            return False
    
    def write_batch(self, records: List[Dict[str, Any]]) -> bool:
        """Write a batch of records to Redis."""
        try:
            if not self.client:
                return False
            
            key_prefix = self.config.write_config.get('key_prefix', 'stream:')
            ttl = self.config.write_config.get('ttl', 3600)
            
            # Prepare batch data
            batch_data = {}
            for record in records:
                key = self._generate_key(record, key_prefix)
                value = json.dumps(record)
                batch_data[key] = value
            
            # Write batch
            success = self.client.mset(batch_data)
            
            # Set TTL for all keys
            if success:
                for key in batch_data.keys():
                    self.client.expire(key, ttl)
            
            return success
            
        except Exception as e:
            logger.error(f"Error writing batch to Redis: {e}")
            return False
    
    def _generate_key(self, record: Dict[str, Any], prefix: str) -> str:
        """Generate Redis key from record."""
        # Use timestamp and record ID if available
        timestamp = record.get('timestamp', datetime.utcnow().isoformat())
        record_id = record.get('id', record.get('_id', 'unknown'))
        return f"{prefix}{timestamp}:{record_id}"


class CassandraStreamSink(StreamSink):
    """Cassandra stream sink implementation."""
    
    def connect(self) -> bool:
        """Establish connection to Cassandra."""
        try:
            self.client = CassandraClient(
                hosts=self.config.connection_config['hosts'],
                keyspace=self.config.connection_config['keyspace'],
                username=self.config.connection_config.get('username'),
                password=self.config.connection_config.get('password')
            )
            return self.client.connect()
            
        except Exception as e:
            logger.error(f"Error connecting to Cassandra: {e}")
            return False
    
    def disconnect(self):
        """Close Cassandra connection."""
        if self.client:
            self.client.disconnect()
    
    def write_record(self, record: Dict[str, Any]) -> bool:
        """Write a single record to Cassandra."""
        try:
            if not self.client:
                return False
            
            table_name = self.config.write_config.get('table_name', 'stream_data')
            return self.client.insert_row(table_name, record)
            
        except Exception as e:
            logger.error(f"Error writing record to Cassandra: {e}")
            return False
    
    def write_batch(self, records: List[Dict[str, Any]]) -> bool:
        """Write a batch of records to Cassandra."""
        try:
            if not self.client:
                return False
            
            table_name = self.config.write_config.get('table_name', 'stream_data')
            inserted_count = self.client.insert_batch(table_name, records)
            return inserted_count == len(records)
            
        except Exception as e:
            logger.error(f"Error writing batch to Cassandra: {e}")
            return False


class ElasticsearchStreamSink(StreamSink):
    """Elasticsearch stream sink implementation."""
    
    def connect(self) -> bool:
        """Establish connection to Elasticsearch."""
        try:
            self.client = ElasticsearchClient(
                hosts=self.config.connection_config['hosts'],
                username=self.config.connection_config.get('username'),
                password=self.config.connection_config.get('password')
            )
            return self.client.connect()
            
        except Exception as e:
            logger.error(f"Error connecting to Elasticsearch: {e}")
            return False
    
    def disconnect(self):
        """Close Elasticsearch connection."""
        if self.client:
            self.client.disconnect()
    
    def write_record(self, record: Dict[str, Any]) -> bool:
        """Write a single record to Elasticsearch."""
        try:
            if not self.client:
                return False
            
            index_name = self.config.write_config.get('index_name', 'stream_data')
            doc_id = self.client.index_document(index_name, record)
            return doc_id is not None
            
        except Exception as e:
            logger.error(f"Error writing record to Elasticsearch: {e}")
            return False
    
    def write_batch(self, records: List[Dict[str, Any]]) -> bool:
        """Write a batch of records to Elasticsearch."""
        try:
            if not self.client:
                return False
            
            index_name = self.config.write_config.get('index_name', 'stream_data')
            indexed_count = self.client.bulk_index(index_name, records)
            return indexed_count == len(records)
            
        except Exception as e:
            logger.error(f"Error writing batch to Elasticsearch: {e}")
            return False


class Neo4jStreamSink(StreamSink):
    """Neo4j stream sink implementation."""
    
    def connect(self) -> bool:
        """Establish connection to Neo4j."""
        try:
            self.client = Neo4jClient(
                uri=self.config.connection_config['uri'],
                username=self.config.connection_config['username'],
                password=self.config.connection_config['password'],
                database=self.config.connection_config.get('database', 'neo4j')
            )
            return self.client.connect()
            
        except Exception as e:
            logger.error(f"Error connecting to Neo4j: {e}")
            return False
    
    def disconnect(self):
        """Close Neo4j connection."""
        if self.client:
            self.client.disconnect()
    
    def write_record(self, record: Dict[str, Any]) -> bool:
        """Write a single record to Neo4j."""
        try:
            if not self.client:
                return False
            
            node_label = self.config.write_config.get('node_label', 'StreamData')
            node_id = self.client.create_node(node_label, record)
            return node_id is not None
            
        except Exception as e:
            logger.error(f"Error writing record to Neo4j: {e}")
            return False
    
    def write_batch(self, records: List[Dict[str, Any]]) -> bool:
        """Write a batch of records to Neo4j."""
        try:
            if not self.client:
                return False
            
            node_label = self.config.write_config.get('node_label', 'StreamData')
            node_ids = self.client.create_nodes(node_label, records)
            return len(node_ids) == len(records)
            
        except Exception as e:
            logger.error(f"Error writing batch to Neo4j: {e}")
            return False


class StreamSinkManager:
    """
    Unified manager for stream sinks to various NoSQL databases.
    
    Provides a consistent interface for writing streaming data to different
    NoSQL databases with connection management, error recovery, and batching.
    """
    
    def __init__(self):
        self.sinks: Dict[str, StreamSink] = {}
        self.sink_configs: Dict[str, SinkConfig] = {}
    
    def add_sink(self, sink_name: str, config: SinkConfig) -> bool:
        """
        Add a new sink configuration.
        
        Args:
            sink_name: Name identifier for the sink
            config: Sink configuration
            
        Returns:
            bool: True if sink added successfully
        """
        try:
            self.sink_configs[sink_name] = config
            
            # Create sink instance
            sink = self._create_sink(config)
            if sink and sink.connect():
                self.sinks[sink_name] = sink
                logger.info(f"Added sink: {sink_name} ({config.sink_type.value})")
                return True
            else:
                logger.error(f"Failed to connect sink: {sink_name}")
                return False
                
        except Exception as e:
            logger.error(f"Error adding sink {sink_name}: {e}")
            return False
    
    def remove_sink(self, sink_name: str) -> bool:
        """
        Remove a sink.
        
        Args:
            sink_name: Name of the sink to remove
            
        Returns:
            bool: True if sink removed successfully
        """
        try:
            if sink_name in self.sinks:
                self.sinks[sink_name].disconnect()
                del self.sinks[sink_name]
            
            if sink_name in self.sink_configs:
                del self.sink_configs[sink_name]
            
            logger.info(f"Removed sink: {sink_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error removing sink {sink_name}: {e}")
            return False
    
    def write_to_sink(self, sink_name: str, record: Dict[str, Any], 
                     use_batch: bool = True) -> bool:
        """
        Write a record to a specific sink.
        
        Args:
            sink_name: Name of the sink
            record: Record to write
            use_batch: Whether to use batching
            
        Returns:
            bool: True if write successful
        """
        try:
            if sink_name not in self.sinks:
                logger.error(f"Sink not found: {sink_name}")
                return False
            
            sink = self.sinks[sink_name]
            
            if use_batch:
                return sink.add_to_batch(record)
            else:
                return sink.write_record(record)
                
        except Exception as e:
            logger.error(f"Error writing to sink {sink_name}: {e}")
            return False
    
    def write_to_all_sinks(self, record: Dict[str, Any], 
                          use_batch: bool = True) -> Dict[str, bool]:
        """
        Write a record to all configured sinks.
        
        Args:
            record: Record to write
            use_batch: Whether to use batching
            
        Returns:
            Dict[str, bool]: Results for each sink
        """
        results = {}
        
        for sink_name in self.sinks.keys():
            results[sink_name] = self.write_to_sink(sink_name, record, use_batch)
        
        return results
    
    def flush_all_sinks(self) -> Dict[str, bool]:
        """
        Flush all sink batch buffers.
        
        Returns:
            Dict[str, bool]: Results for each sink
        """
        results = {}
        
        for sink_name, sink in self.sinks.items():
            try:
                results[sink_name] = sink.flush_batch()
            except Exception as e:
                logger.error(f"Error flushing sink {sink_name}: {e}")
                results[sink_name] = False
        
        return results
    
    def get_sink_status(self) -> Dict[str, Dict[str, Any]]:
        """
        Get status of all sinks.
        
        Returns:
            Dict[str, Dict[str, Any]]: Status information for each sink
        """
        status = {}
        
        for sink_name, sink in self.sinks.items():
            config = self.sink_configs.get(sink_name)
            status[sink_name] = {
                'sink_type': config.sink_type.value if config else 'unknown',
                'connected': sink.client is not None,
                'batch_size': len(sink.batch_buffer),
                'last_batch_time': sink.last_batch_time.isoformat()
            }
        
        return status
    
    def close_all_sinks(self):
        """Close all sink connections."""
        for sink_name, sink in self.sinks.items():
            try:
                sink.disconnect()
                logger.info(f"Closed sink: {sink_name}")
            except Exception as e:
                logger.error(f"Error closing sink {sink_name}: {e}")
        
        self.sinks.clear()
        self.sink_configs.clear()
    
    def _create_sink(self, config: SinkConfig) -> Optional[StreamSink]:
        """Create a sink instance based on configuration."""
        try:
            if config.sink_type == SinkType.MONGODB:
                return MongoDBStreamSink(config)
            elif config.sink_type == SinkType.REDIS:
                return RedisStreamSink(config)
            elif config.sink_type == SinkType.CASSANDRA:
                return CassandraStreamSink(config)
            elif config.sink_type == SinkType.ELASTICSEARCH:
                return ElasticsearchStreamSink(config)
            elif config.sink_type == SinkType.NEO4J:
                return Neo4jStreamSink(config)
            else:
                logger.error(f"Unsupported sink type: {config.sink_type}")
                return None
                
        except Exception as e:
            logger.error(f"Error creating sink: {e}")
            return None
