"""
Kafka Streams Processor

This module provides Kafka Streams processing capabilities for the PBF-LB/M data pipeline,
including enhanced NoSQL sink support for real-time data streaming.
"""

import json
from typing import Dict, List, Optional, Any
import logging
from kafka import KafkaProducer, KafkaConsumer
from kafka.errors import KafkaError
from datetime import datetime

from src.data_pipeline.config.pipeline_config import get_pipeline_config
from .stream_sink_manager import StreamSinkManager, SinkConfig, SinkType

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KafkaStreamsProcessor:
    """
    Kafka Streams processor for real-time data processing.
    """
    
    def __init__(self):
        self.config = get_pipeline_config()
        self.producer = None
        self.consumer = None
        self.sink_manager = StreamSinkManager()
        self._initialize_kafka()
        self._initialize_nosql_sinks()
    
    def _initialize_kafka(self):
        """Initialize Kafka producer and consumer."""
        try:
            kafka_config = self.config.get('kafka', {})
            bootstrap_servers = kafka_config.get('bootstrap_servers', ['localhost:9092'])
            
            # Initialize producer
            self.producer = KafkaProducer(
                bootstrap_servers=bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                key_serializer=lambda k: k.encode('utf-8') if k else None
            )
            
            logger.info("Kafka Streams processor initialized")
            
        except Exception as e:
            logger.error(f"Error initializing Kafka Streams processor: {e}")
    
    def _initialize_nosql_sinks(self):
        """Initialize NoSQL sinks for streaming data."""
        try:
            # MongoDB sink for document storage
            mongodb_config = self.config.get('mongodb', {})
            if mongodb_config:
                sink_config = SinkConfig(
                    sink_type=SinkType.MONGODB,
                    connection_config={
                        'connection_string': mongodb_config.get('connection_string'),
                        'database_name': mongodb_config.get('database_name')
                    },
                    write_config={
                        'collection_name': 'stream_data'
                    },
                    retry_config={'max_retries': 3, 'retry_delay': 1},
                    batch_config={'batch_size': 100, 'timeout_seconds': 5}
                )
                self.sink_manager.add_sink('mongodb', sink_config)
            
            # Redis sink for caching
            redis_config = self.config.get('redis', {})
            if redis_config:
                sink_config = SinkConfig(
                    sink_type=SinkType.REDIS,
                    connection_config={
                        'host': redis_config.get('host', 'localhost'),
                        'port': redis_config.get('port', 6379),
                        'password': redis_config.get('password'),
                        'db': redis_config.get('db', 0)
                    },
                    write_config={
                        'key_prefix': 'stream:',
                        'ttl': 3600
                    },
                    retry_config={'max_retries': 3, 'retry_delay': 1},
                    batch_config={'batch_size': 50, 'timeout_seconds': 2}
                )
                self.sink_manager.add_sink('redis', sink_config)
            
            # Cassandra sink for time-series data
            cassandra_config = self.config.get('cassandra', {})
            if cassandra_config:
                sink_config = SinkConfig(
                    sink_type=SinkType.CASSANDRA,
                    connection_config={
                        'hosts': cassandra_config.get('hosts', ['localhost']),
                        'keyspace': cassandra_config.get('keyspace'),
                        'username': cassandra_config.get('username'),
                        'password': cassandra_config.get('password')
                    },
                    write_config={
                        'table_name': 'stream_data'
                    },
                    retry_config={'max_retries': 3, 'retry_delay': 1},
                    batch_config={'batch_size': 200, 'timeout_seconds': 10}
                )
                self.sink_manager.add_sink('cassandra', sink_config)
            
            logger.info("NoSQL sinks initialized for Kafka Streams processor")
            
        except Exception as e:
            logger.error(f"Error initializing NoSQL sinks: {e}")
    
    def process_pbf_process_stream(self, topic: str = 'pbf_process_stream') -> None:
        """Process PBF process data stream."""
        try:
            consumer = KafkaConsumer(
                topic,
                bootstrap_servers=self.config.get('kafka', {}).get('bootstrap_servers', ['localhost:9092']),
                value_deserializer=lambda m: json.loads(m.decode('utf-8'))
            )
            
            for message in consumer:
                try:
                    data = message.value
                    processed_data = self._process_pbf_data(data)
                    
                    # Send to Kafka topic
                    self._send_processed_data('pbf_process_processed', processed_data)
                    
                    # Write to NoSQL sinks
                    self._write_to_nosql_sinks('pbf_process', processed_data)
                    
                except Exception as e:
                    logger.error(f"Error processing PBF data: {e}")
                    
        except Exception as e:
            logger.error(f"Error processing PBF process stream: {e}")
    
    def process_ispm_monitoring_stream(self, topic: str = 'ispm_monitoring_stream') -> None:
        """Process ISPM monitoring data stream."""
        try:
            consumer = KafkaConsumer(
                topic,
                bootstrap_servers=self.config.get('kafka', {}).get('bootstrap_servers', ['localhost:9092']),
                value_deserializer=lambda m: json.loads(m.decode('utf-8'))
            )
            
            for message in consumer:
                try:
                    data = message.value
                    processed_data = self._process_ispm_data(data)
                    
                    # Send to Kafka topic
                    self._send_processed_data('ispm_monitoring_processed', processed_data)
                    
                    # Write to NoSQL sinks
                    self._write_to_nosql_sinks('ispm_monitoring', processed_data)
                    
                except Exception as e:
                    logger.error(f"Error processing ISPM data: {e}")
                    
        except Exception as e:
            logger.error(f"Error processing ISPM monitoring stream: {e}")
    
    def _process_pbf_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process PBF process data."""
        try:
            # Add processing logic here
            processed_data = data.copy()
            processed_data['processed_at'] = datetime.utcnow().isoformat()
            processed_data['processing_status'] = 'completed'
            
            return processed_data
            
        except Exception as e:
            logger.error(f"Error processing PBF data: {e}")
            return data
    
    def _process_ispm_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process ISPM monitoring data."""
        try:
            # Add processing logic here
            processed_data = data.copy()
            processed_data['processed_at'] = datetime.utcnow().isoformat()
            processed_data['processing_status'] = 'completed'
            
            return processed_data
            
        except Exception as e:
            logger.error(f"Error processing ISPM data: {e}")
            return data
    
    def _send_processed_data(self, topic: str, data: Dict[str, Any]) -> None:
        """Send processed data to output topic."""
        try:
            if self.producer:
                self.producer.send(topic, value=data)
                logger.info(f"Processed data sent to topic: {topic}")
                
        except Exception as e:
            logger.error(f"Error sending processed data: {e}")
    
    def _write_to_nosql_sinks(self, data_type: str, data: Dict[str, Any]) -> None:
        """Write processed data to NoSQL sinks."""
        try:
            # Add metadata for NoSQL storage
            nosql_data = data.copy()
            nosql_data['data_type'] = data_type
            nosql_data['stream_timestamp'] = datetime.utcnow().isoformat()
            
            # Write to all configured sinks
            results = self.sink_manager.write_to_all_sinks(nosql_data, use_batch=True)
            
            # Log results
            for sink_name, success in results.items():
                if success:
                    logger.debug(f"Data written to {sink_name} sink")
                else:
                    logger.warning(f"Failed to write data to {sink_name} sink")
                    
        except Exception as e:
            logger.error(f"Error writing to NoSQL sinks: {e}")
    
    def write_to_mongodb(self, collection_name: str, data: Dict[str, Any]) -> bool:
        """Write data directly to MongoDB."""
        try:
            return self.sink_manager.write_to_sink('mongodb', data, use_batch=False)
        except Exception as e:
            logger.error(f"Error writing to MongoDB: {e}")
            return False
    
    def write_to_redis(self, data: Dict[str, Any]) -> bool:
        """Write data directly to Redis."""
        try:
            return self.sink_manager.write_to_sink('redis', data, use_batch=False)
        except Exception as e:
            logger.error(f"Error writing to Redis: {e}")
            return False
    
    def write_to_cassandra(self, data: Dict[str, Any]) -> bool:
        """Write data directly to Cassandra."""
        try:
            return self.sink_manager.write_to_sink('cassandra', data, use_batch=False)
        except Exception as e:
            logger.error(f"Error writing to Cassandra: {e}")
            return False
    
    def flush_all_sinks(self) -> Dict[str, bool]:
        """Flush all NoSQL sink batch buffers."""
        try:
            return self.sink_manager.flush_all_sinks()
        except Exception as e:
            logger.error(f"Error flushing sinks: {e}")
            return {}
    
    def get_sink_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all NoSQL sinks."""
        try:
            return self.sink_manager.get_sink_status()
        except Exception as e:
            logger.error(f"Error getting sink status: {e}")
            return {}
    
    def close(self):
        """Close Kafka connections and NoSQL sinks."""
        try:
            # Flush all sink buffers before closing
            self.flush_all_sinks()
            
            # Close NoSQL sinks
            self.sink_manager.close_all_sinks()
            
            # Close Kafka connections
            if self.producer:
                self.producer.close()
            if self.consumer:
                self.consumer.close()
                
            logger.info("Kafka Streams processor closed")
            
        except Exception as e:
            logger.error(f"Error closing Kafka Streams processor: {e}")
