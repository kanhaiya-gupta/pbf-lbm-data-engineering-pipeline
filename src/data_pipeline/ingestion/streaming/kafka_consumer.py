"""
Kafka Consumer for PBF-LB/M Data Pipeline

This module provides Kafka message consumption capabilities for streaming data.
"""

import json
import logging
from typing import Dict, Any, Optional, Callable, List
from kafka import KafkaConsumer
from kafka.errors import KafkaError
from src.data_pipeline.config.streaming_config import get_kafka_config

logger = logging.getLogger(__name__)


class KafkaConsumer:
    """
    Kafka consumer for receiving PBF-LB/M data from Kafka topics.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Kafka consumer.
        
        Args:
            config: Optional Kafka configuration. If None, uses default config.
        """
        self.config = config or get_kafka_config()
        self.consumer = None
        self._initialize_consumer()
    
    def _initialize_consumer(self) -> None:
        """Initialize the Kafka consumer with configuration."""
        try:
            self.consumer = KafkaConsumer(
                bootstrap_servers=self.config["bootstrap_servers"],
                group_id=self.config["group_id"],
                auto_offset_reset=self.config["auto_offset_reset"],
                enable_auto_commit=self.config["enable_auto_commit"],
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                key_deserializer=lambda m: m.decode('utf-8') if m else None,
                consumer_timeout_ms=1000,  # 1 second timeout
                max_poll_records=500,
                session_timeout_ms=30000,
                heartbeat_interval_ms=10000
            )
            logger.info("Kafka consumer initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Kafka consumer: {e}")
            raise
    
    def subscribe_to_topic(self, topic: str) -> None:
        """
        Subscribe to a specific Kafka topic.
        
        Args:
            topic: Kafka topic to subscribe to
        """
        try:
            self.consumer.subscribe([topic])
            logger.info(f"Subscribed to topic: {topic}")
        except Exception as e:
            logger.error(f"Failed to subscribe to topic {topic}: {e}")
            raise
    
    def subscribe_to_topics(self, topics: List[str]) -> None:
        """
        Subscribe to multiple Kafka topics.
        
        Args:
            topics: List of Kafka topics to subscribe to
        """
        try:
            self.consumer.subscribe(topics)
            logger.info(f"Subscribed to topics: {topics}")
        except Exception as e:
            logger.error(f"Failed to subscribe to topics {topics}: {e}")
            raise
    
    def consume_messages(self, message_handler: Callable[[Dict[str, Any]], None], max_messages: Optional[int] = None) -> int:
        """
        Consume messages from subscribed topics.
        
        Args:
            message_handler: Function to handle received messages
            max_messages: Maximum number of messages to consume (None for unlimited)
            
        Returns:
            int: Number of messages consumed
        """
        if not self.consumer:
            logger.error("Consumer not initialized")
            return 0
        
        messages_consumed = 0
        try:
            for message in self.consumer:
                try:
                    message_data = {
                        "topic": message.topic,
                        "partition": message.partition,
                        "offset": message.offset,
                        "key": message.key,
                        "value": message.value,
                        "timestamp": message.timestamp
                    }
                    message_handler(message_data)
                    messages_consumed += 1
                    
                    if max_messages and messages_consumed >= max_messages:
                        break
                        
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    continue
                    
        except KafkaError as e:
            logger.error(f"Kafka error during message consumption: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during message consumption: {e}")
        
        logger.info(f"Consumed {messages_consumed} messages")
        return messages_consumed
    
    def consume_ispm_monitoring_data(self, message_handler: Callable[[Dict[str, Any]], None]) -> int:
        """
        Consume ISPM monitoring data from the appropriate Kafka topic.
        
        Args:
            message_handler: Function to handle ISPM monitoring messages
            
        Returns:
            int: Number of messages consumed
        """
        self.subscribe_to_topic("ispm_monitoring_events")
        return self.consume_messages(message_handler)
    
    def consume_powder_bed_data(self, message_handler: Callable[[Dict[str, Any]], None]) -> int:
        """
        Consume powder bed monitoring data from the appropriate Kafka topic.
        
        Args:
            message_handler: Function to handle powder bed monitoring messages
            
        Returns:
            int: Number of messages consumed
        """
        self.subscribe_to_topic("powder_bed_monitoring_events")
        return self.consume_messages(message_handler)
    
    def consume_pbf_process_data(self, message_handler: Callable[[Dict[str, Any]], None]) -> int:
        """
        Consume PBF process data from the appropriate Kafka topic.
        
        Args:
            message_handler: Function to handle PBF process messages
            
        Returns:
            int: Number of messages consumed
        """
        self.subscribe_to_topic("pbf_process_events")
        return self.consume_messages(message_handler)
    
    def get_topic_partitions(self, topic: str) -> List[int]:
        """
        Get partition information for a topic.
        
        Args:
            topic: Kafka topic name
            
        Returns:
            List[int]: List of partition numbers
        """
        try:
            partitions = self.consumer.partitions_for_topic(topic)
            return list(partitions) if partitions else []
        except Exception as e:
            logger.error(f"Failed to get partitions for topic {topic}: {e}")
            return []
    
    def close(self) -> None:
        """Close the Kafka consumer."""
        if self.consumer:
            self.consumer.close()
            logger.info("Kafka consumer closed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
