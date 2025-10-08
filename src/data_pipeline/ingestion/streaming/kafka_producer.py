"""
Kafka Producer for PBF-LB/M Data Pipeline

This module provides Kafka message production capabilities for streaming data.
"""

import json
import logging
from typing import Dict, Any, Optional, List
from kafka import KafkaProducer
from kafka.errors import KafkaError
from src.data_pipeline.config.streaming_config import get_kafka_config

logger = logging.getLogger(__name__)


class KafkaProducer:
    """
    Kafka producer for sending PBF-LB/M data to Kafka topics.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Kafka producer.
        
        Args:
            config: Optional Kafka configuration. If None, uses default config.
        """
        self.config = config or get_kafka_config()
        self.producer = None
        self._initialize_producer()
    
    def _initialize_producer(self) -> None:
        """Initialize the Kafka producer with configuration."""
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=self.config["bootstrap_servers"],
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                key_serializer=lambda k: k.encode('utf-8') if k else None,
                acks='all',  # Wait for all replicas to acknowledge
                retries=3,
                retry_backoff_ms=100,
                request_timeout_ms=30000,
                max_block_ms=10000
            )
            logger.info("Kafka producer initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Kafka producer: {e}")
            raise
    
    def send_message(self, topic: str, message: Dict[str, Any], key: Optional[str] = None) -> bool:
        """
        Send a message to a Kafka topic.
        
        Args:
            topic: Target Kafka topic
            message: Message data to send
            key: Optional message key for partitioning
            
        Returns:
            bool: True if message was sent successfully, False otherwise
        """
        try:
            future = self.producer.send(topic, value=message, key=key)
            record_metadata = future.get(timeout=10)
            logger.debug(f"Message sent to topic {topic}, partition {record_metadata.partition}, offset {record_metadata.offset}")
            return True
        except KafkaError as e:
            logger.error(f"Failed to send message to topic {topic}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error sending message to topic {topic}: {e}")
            return False
    
    def send_batch(self, topic: str, messages: List[Dict[str, Any]], keys: Optional[List[str]] = None) -> int:
        """
        Send a batch of messages to a Kafka topic.
        
        Args:
            topic: Target Kafka topic
            messages: List of message data to send
            keys: Optional list of message keys for partitioning
            
        Returns:
            int: Number of messages sent successfully
        """
        if keys and len(keys) != len(messages):
            logger.error("Number of keys must match number of messages")
            return 0
        
        successful_sends = 0
        for i, message in enumerate(messages):
            key = keys[i] if keys else None
            if self.send_message(topic, message, key):
                successful_sends += 1
        
        logger.info(f"Sent {successful_sends}/{len(messages)} messages to topic {topic}")
        return successful_sends
    
    def send_ispm_monitoring_data(self, monitoring_data: Dict[str, Any]) -> bool:
        """
        Send ISPM monitoring data to the appropriate Kafka topic.
        
        Args:
            monitoring_data: ISPM monitoring data
            
        Returns:
            bool: True if sent successfully
        """
        topic = "ispm_monitoring_events"
        key = monitoring_data.get("sensor_id", "unknown")
        return self.send_message(topic, monitoring_data, key)
    
    def send_powder_bed_data(self, powder_bed_data: Dict[str, Any]) -> bool:
        """
        Send powder bed monitoring data to the appropriate Kafka topic.
        
        Args:
            powder_bed_data: Powder bed monitoring data
            
        Returns:
            bool: True if sent successfully
        """
        topic = "powder_bed_monitoring_events"
        key = powder_bed_data.get("camera_id", "unknown")
        return self.send_message(topic, powder_bed_data, key)
    
    def send_pbf_process_data(self, process_data: Dict[str, Any]) -> bool:
        """
        Send PBF process data to the appropriate Kafka topic.
        
        Args:
            process_data: PBF process data
            
        Returns:
            bool: True if sent successfully
        """
        topic = "pbf_process_events"
        key = process_data.get("process_id", "unknown")
        return self.send_message(topic, process_data, key)
    
    def close(self) -> None:
        """Close the Kafka producer."""
        if self.producer:
            self.producer.flush()
            self.producer.close()
            logger.info("Kafka producer closed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
