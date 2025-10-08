"""
Kafka Ingester for PBF-LB/M Data Pipeline

This module provides a unified Kafka ingestion interface that combines
producer and consumer functionality for streaming data ingestion.
"""

import json
import logging
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from dataclasses import dataclass
from kafka import KafkaProducer, KafkaConsumer
from kafka.errors import KafkaError

from src.data_pipeline.config.streaming_config import get_kafka_config
from src.data_pipeline.ingestion.streaming.message_serializer import MessageSerializer, SerializationFormat

logger = logging.getLogger(__name__)


@dataclass
class KafkaIngestionConfig:
    """Configuration for Kafka ingestion."""
    bootstrap_servers: List[str]
    topic: str
    group_id: Optional[str] = None
    auto_offset_reset: str = 'latest'
    enable_auto_commit: bool = True
    value_serializer: str = 'json'
    key_serializer: str = 'string'
    max_poll_records: int = 500
    session_timeout_ms: int = 30000
    request_timeout_ms: int = 40000


class KafkaIngester:
    """
    Unified Kafka ingestion interface for PBF-LB/M data pipeline.
    
    This class provides both producer and consumer functionality for
    streaming data ingestion from various PBF-LB/M data sources.
    """
    
    def __init__(self, config: Optional[KafkaIngestionConfig] = None):
        """
        Initialize the Kafka ingester.
        
        Args:
            config: Kafka ingestion configuration
        """
        self.config = config or self._load_default_config()
        self.producer = None
        self.consumer = None
        self.serializer = MessageSerializer()
        
        logger.info(f"Kafka Ingester initialized for topic: {self.config.topic}")
    
    def _load_default_config(self) -> KafkaIngestionConfig:
        """Load default configuration from environment."""
        kafka_config = get_kafka_config()
        
        return KafkaIngestionConfig(
            bootstrap_servers=kafka_config.bootstrap_servers,
            topic=kafka_config.default_topic,
            group_id=kafka_config.consumer_group_id,
            auto_offset_reset=kafka_config.auto_offset_reset,
            enable_auto_commit=kafka_config.enable_auto_commit,
            value_serializer=kafka_config.value_serializer,
            key_serializer=kafka_config.key_serializer,
            max_poll_records=kafka_config.max_poll_records,
            session_timeout_ms=kafka_config.session_timeout_ms,
            request_timeout_ms=kafka_config.request_timeout_ms
        )
    
    def _get_producer(self) -> KafkaProducer:
        """Get or create Kafka producer."""
        if self.producer is None:
            producer_config = {
                'bootstrap_servers': self.config.bootstrap_servers,
                'value_serializer': self._get_serializer(self.config.value_serializer),
                'key_serializer': self._get_serializer(self.config.key_serializer),
                'retries': 3,
                'retry_backoff_ms': 100,
                'request_timeout_ms': self.config.request_timeout_ms,
                'acks': 'all'  # Wait for all replicas to acknowledge
            }
            
            self.producer = KafkaProducer(**producer_config)
            logger.info("Kafka producer created")
        
        return self.producer
    
    def _get_consumer(self) -> KafkaConsumer:
        """Get or create Kafka consumer."""
        if self.consumer is None:
            consumer_config = {
                'bootstrap_servers': self.config.bootstrap_servers,
                'group_id': self.config.group_id,
                'auto_offset_reset': self.config.auto_offset_reset,
                'enable_auto_commit': self.config.enable_auto_commit,
                'max_poll_records': self.config.max_poll_records,
                'session_timeout_ms': self.config.session_timeout_ms,
                'request_timeout_ms': self.config.request_timeout_ms,
                'value_deserializer': self._get_deserializer(self.config.value_serializer),
                'key_deserializer': self._get_deserializer(self.config.key_serializer)
            }
            
            self.consumer = KafkaConsumer(self.config.topic, **consumer_config)
            logger.info(f"Kafka consumer created for topic: {self.config.topic}")
        
        return self.consumer
    
    def _get_serializer(self, serializer_type: str) -> Callable:
        """Get serializer function based on type."""
        if serializer_type == 'json':
            return lambda x: json.dumps(x).encode('utf-8')
        elif serializer_type == 'string':
            return lambda x: str(x).encode('utf-8')
        else:
            return lambda x: x
    
    def _get_deserializer(self, deserializer_type: str) -> Callable:
        """Get deserializer function based on type."""
        if deserializer_type == 'json':
            return lambda x: json.loads(x.decode('utf-8')) if x else None
        elif deserializer_type == 'string':
            return lambda x: x.decode('utf-8') if x else None
        else:
            return lambda x: x
    
    def produce_message(self, message: Dict[str, Any], key: Optional[str] = None) -> bool:
        """
        Produce a message to Kafka topic.
        
        Args:
            message: Message data to send
            key: Optional message key
            
        Returns:
            bool: True if message was sent successfully
        """
        try:
            producer = self._get_producer()
            
            # Add metadata to message
            enriched_message = {
                'data': message,
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'source': 'pbf_lbm_pipeline',
                    'version': '1.0'
                }
            }
            
            # Send message
            future = producer.send(
                self.config.topic,
                value=enriched_message,
                key=key
            )
            
            # Wait for confirmation
            record_metadata = future.get(timeout=10)
            
            logger.debug(f"Message sent to topic {record_metadata.topic}, "
                        f"partition {record_metadata.partition}, "
                        f"offset {record_metadata.offset}")
            
            return True
            
        except KafkaError as e:
            logger.error(f"Failed to send message to Kafka: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error sending message: {e}")
            return False
    
    def consume_messages(self, timeout_ms: int = 1000, max_records: int = 100) -> List[Dict[str, Any]]:
        """
        Consume messages from Kafka topic.
        
        Args:
            timeout_ms: Timeout in milliseconds
            max_records: Maximum number of records to consume
            
        Returns:
            List of consumed messages
        """
        try:
            consumer = self._get_consumer()
            messages = []
            
            # Poll for messages
            message_batch = consumer.poll(timeout_ms=timeout_ms, max_records=max_records)
            
            for topic_partition, records in message_batch.items():
                for record in records:
                    try:
                        message = {
                            'topic': record.topic,
                            'partition': record.partition,
                            'offset': record.offset,
                            'key': record.key,
                            'value': record.value,
                            'timestamp': record.timestamp,
                            'headers': dict(record.headers) if record.headers else {}
                        }
                        messages.append(message)
                        
                    except Exception as e:
                        logger.error(f"Error processing record: {e}")
                        continue
            
            if messages:
                logger.info(f"Consumed {len(messages)} messages from topic {self.config.topic}")
            
            return messages
            
        except KafkaError as e:
            logger.error(f"Failed to consume messages from Kafka: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error consuming messages: {e}")
            return []
    
    def ingest_from_kafka(self, timeout_ms: int = 1000, max_records: int = 100) -> List[Dict[str, Any]]:
        """
        Ingest data from Kafka topic.
        
        Args:
            timeout_ms: Timeout in milliseconds
            max_records: Maximum number of records to consume
            
        Returns:
            List of ingested data records
        """
        messages = self.consume_messages(timeout_ms, max_records)
        
        # Extract data from messages
        data_records = []
        for message in messages:
            try:
                if message['value'] and 'data' in message['value']:
                    data_records.append(message['value']['data'])
                else:
                    data_records.append(message['value'])
            except Exception as e:
                logger.error(f"Error extracting data from message: {e}")
                continue
        
        return data_records
    
    def check_topic_message_count(self, topic: Optional[str] = None) -> int:
        """
        Check the number of messages in a topic.
        
        Args:
            topic: Topic name (uses default if None)
            
        Returns:
            Number of messages in the topic
        """
        try:
            target_topic = topic or self.config.topic
            consumer = self._get_consumer()
            
            # Get topic partitions
            partitions = consumer.partitions_for_topic(target_topic)
            if not partitions:
                logger.warning(f"No partitions found for topic: {target_topic}")
                return 0
            
            # Get end offsets for all partitions
            end_offsets = consumer.end_offsets(partitions)
            total_messages = sum(end_offsets.values())
            
            logger.info(f"Topic {target_topic} has {total_messages} messages")
            return total_messages
            
        except Exception as e:
            logger.error(f"Error checking topic message count: {e}")
            return 0
    
    def get_topic_metadata(self, topic: Optional[str] = None) -> Dict[str, Any]:
        """
        Get metadata for a Kafka topic.
        
        Args:
            topic: Topic name (uses default if None)
            
        Returns:
            Topic metadata dictionary
        """
        try:
            target_topic = topic or self.config.topic
            consumer = self._get_consumer()
            
            # Get topic metadata
            metadata = consumer.list_consumer_group_offsets()
            partitions = consumer.partitions_for_topic(target_topic)
            
            topic_metadata = {
                'topic': target_topic,
                'partitions': list(partitions) if partitions else [],
                'partition_count': len(partitions) if partitions else 0,
                'consumer_groups': list(metadata.keys()) if metadata else []
            }
            
            return topic_metadata
            
        except Exception as e:
            logger.error(f"Error getting topic metadata: {e}")
            return {}
    
    def close(self):
        """Close producer and consumer connections."""
        try:
            if self.producer:
                self.producer.close()
                self.producer = None
                logger.info("Kafka producer closed")
            
            if self.consumer:
                self.consumer.close()
                self.consumer = None
                logger.info("Kafka consumer closed")
                
        except Exception as e:
            logger.error(f"Error closing Kafka connections: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


# Convenience functions for common operations
def create_kafka_ingester(topic: str, **kwargs) -> KafkaIngester:
    """
    Create a Kafka ingester with custom configuration.
    
    Args:
        topic: Kafka topic name
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured KafkaIngester instance
    """
    config = KafkaIngestionConfig(
        bootstrap_servers=kwargs.get('bootstrap_servers', ['localhost:9092']),
        topic=topic,
        group_id=kwargs.get('group_id'),
        auto_offset_reset=kwargs.get('auto_offset_reset', 'latest'),
        enable_auto_commit=kwargs.get('enable_auto_commit', True),
        value_serializer=kwargs.get('value_serializer', 'json'),
        key_serializer=kwargs.get('key_serializer', 'string'),
        max_poll_records=kwargs.get('max_poll_records', 500),
        session_timeout_ms=kwargs.get('session_timeout_ms', 30000),
        request_timeout_ms=kwargs.get('request_timeout_ms', 40000)
    )
    
    return KafkaIngester(config)


def ingest_ispm_data(topic: str = 'ispm_monitoring', **kwargs) -> List[Dict[str, Any]]:
    """
    Ingest ISPM monitoring data from Kafka.
    
    Args:
        topic: Kafka topic name
        **kwargs: Additional configuration parameters
        
    Returns:
        List of ISPM monitoring data records
    """
    with create_kafka_ingester(topic, **kwargs) as ingester:
        return ingester.ingest_from_kafka()


def ingest_powder_bed_data(topic: str = 'powder_bed_monitoring', **kwargs) -> List[Dict[str, Any]]:
    """
    Ingest powder bed monitoring data from Kafka.
    
    Args:
        topic: Kafka topic name
        **kwargs: Additional configuration parameters
        
    Returns:
        List of powder bed monitoring data records
    """
    with create_kafka_ingester(topic, **kwargs) as ingester:
        return ingester.ingest_from_kafka()
