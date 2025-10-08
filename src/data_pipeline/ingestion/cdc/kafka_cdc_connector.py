"""
Kafka CDC Connector for PBF-LB/M Data Pipeline

This module provides Change Data Capture capabilities for Kafka topics.
"""

import logging
import json
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
from src.data_pipeline.config.streaming_config import get_kafka_config

logger = logging.getLogger(__name__)


class KafkaCDCConnector:
    """
    Kafka Change Data Capture connector.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Kafka CDC connector.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.kafka_config = get_kafka_config()
        self.consumer = None
        self.producer = None
        self.change_handlers = []
        self.topic_offsets = {}
        self._initialize_kafka_clients()
    
    def _initialize_kafka_clients(self) -> None:
        """Initialize Kafka consumer and producer."""
        try:
            from kafka import KafkaConsumer, KafkaProducer
            
            # Initialize consumer
            self.consumer = KafkaConsumer(
                bootstrap_servers=self.kafka_config["bootstrap_servers"],
                group_id=f"{self.kafka_config['group_id']}_cdc",
                auto_offset_reset='earliest',
                enable_auto_commit=False,
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                key_deserializer=lambda m: m.decode('utf-8') if m else None
            )
            
            # Initialize producer
            self.producer = KafkaProducer(
                bootstrap_servers=self.kafka_config["bootstrap_servers"],
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                key_serializer=lambda k: k.encode('utf-8') if k else None
            )
            
            logger.info("Kafka CDC clients initialized successfully")
            
        except ImportError:
            logger.error("kafka-python not available for Kafka operations")
        except Exception as e:
            logger.error(f"Failed to initialize Kafka CDC clients: {e}")
    
    def add_change_handler(self, handler: Callable[[Dict[str, Any]], None]) -> None:
        """
        Add a change handler for CDC events.
        
        Args:
            handler: Function to handle CDC events
        """
        self.change_handlers.append(handler)
    
    def _emit_change_event(self, event: Dict[str, Any]) -> None:
        """
        Emit a change event to all registered handlers.
        
        Args:
            event: Change event data
        """
        for handler in self.change_handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Error in change handler: {e}")
    
    def start_cdc_stream(self, topics: List[str]) -> None:
        """
        Start CDC stream for specified topics.
        
        Args:
            topics: List of Kafka topics to monitor
        """
        logger.info(f"Starting CDC stream for topics: {topics}")
        
        if not self.consumer:
            logger.error("Kafka consumer not initialized")
            return
        
        try:
            # Subscribe to topics
            self.consumer.subscribe(topics)
            
            # Start consuming messages
            for message in self.consumer:
                try:
                    # Process message as CDC event
                    change_event = self._process_kafka_message(message)
                    if change_event:
                        self._emit_change_event(change_event)
                        
                        # Commit offset
                        self.consumer.commit()
                        
                except Exception as e:
                    logger.error(f"Error processing Kafka message: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error in CDC stream: {e}")
    
    def _process_kafka_message(self, message) -> Optional[Dict[str, Any]]:
        """
        Process Kafka message into CDC event.
        
        Args:
            message: Kafka message object
            
        Returns:
            Optional[Dict[str, Any]]: CDC event or None if processing failed
        """
        try:
            # Extract message information
            change_event = {
                "topic": message.topic,
                "partition": message.partition,
                "offset": message.offset,
                "key": message.key,
                "value": message.value,
                "timestamp": message.timestamp,
                "event_type": "kafka_cdc",
                "processed_at": datetime.now().isoformat()
            }
            
            # Add change type based on message content
            if message.value:
                if "op" in message.value:
                    change_event["change_type"] = message.value["op"]
                elif "event_type" in message.value:
                    change_event["change_type"] = message.value["event_type"]
                else:
                    change_event["change_type"] = "insert"  # Default
            
            return change_event
            
        except Exception as e:
            logger.error(f"Error processing Kafka message: {e}")
            return None
    
    def send_cdc_event(self, topic: str, event: Dict[str, Any], key: Optional[str] = None) -> bool:
        """
        Send CDC event to Kafka topic.
        
        Args:
            topic: Target Kafka topic
            event: CDC event data
            key: Optional message key
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.producer:
            logger.error("Kafka producer not initialized")
            return False
        
        try:
            # Add CDC metadata
            event["cdc_timestamp"] = datetime.now().isoformat()
            event["cdc_source"] = "kafka_cdc_connector"
            
            # Send message
            future = self.producer.send(topic, value=event, key=key)
            record_metadata = future.get(timeout=10)
            
            logger.debug(f"Sent CDC event to topic {topic}, partition {record_metadata.partition}, offset {record_metadata.offset}")
            return True
            
        except Exception as e:
            logger.error(f"Error sending CDC event to topic {topic}: {e}")
            return False
    
    def get_topic_offsets(self, topics: List[str]) -> Dict[str, Dict[int, int]]:
        """
        Get current offsets for specified topics.
        
        Args:
            topics: List of Kafka topics
            
        Returns:
            Dict[str, Dict[int, int]]: Topic offsets by partition
        """
        if not self.consumer:
            logger.error("Kafka consumer not initialized")
            return {}
        
        offsets = {}
        
        try:
            for topic in topics:
                partitions = self.consumer.partitions_for_topic(topic)
                if partitions:
                    topic_offsets = {}
                    for partition in partitions:
                        # Get latest offset
                        latest_offset = self.consumer.end_offsets([(topic, partition)])[(topic, partition)]
                        topic_offsets[partition] = latest_offset
                    offsets[topic] = topic_offsets
            
            logger.info(f"Retrieved offsets for {len(topics)} topics")
            
        except Exception as e:
            logger.error(f"Error getting topic offsets: {e}")
        
        return offsets
    
    def get_consumer_group_offsets(self, group_id: str, topics: List[str]) -> Dict[str, Dict[int, int]]:
        """
        Get consumer group offsets for specified topics.
        
        Args:
            group_id: Consumer group ID
            topics: List of Kafka topics
            
        Returns:
            Dict[str, Dict[int, int]]: Consumer group offsets by topic and partition
        """
        if not self.consumer:
            logger.error("Kafka consumer not initialized")
            return {}
        
        offsets = {}
        
        try:
            for topic in topics:
                partitions = self.consumer.partitions_for_topic(topic)
                if partitions:
                    topic_offsets = {}
                    for partition in partitions:
                        # Get committed offset
                        committed_offset = self.consumer.committed((topic, partition))
                        topic_offsets[partition] = committed_offset or 0
                    offsets[topic] = topic_offsets
            
            logger.info(f"Retrieved consumer group offsets for {len(topics)} topics")
            
        except Exception as e:
            logger.error(f"Error getting consumer group offsets: {e}")
        
        return offsets
    
    def reset_offsets(self, topics: List[str], offset: str = 'earliest') -> bool:
        """
        Reset offsets for specified topics.
        
        Args:
            topics: List of Kafka topics
            offset: Offset to reset to ('earliest' or 'latest')
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.consumer:
            logger.error("Kafka consumer not initialized")
            return False
        
        try:
            # Create topic partitions
            topic_partitions = []
            for topic in topics:
                partitions = self.consumer.partitions_for_topic(topic)
                if partitions:
                    for partition in partitions:
                        topic_partitions.append((topic, partition))
            
            if offset == 'earliest':
                # Reset to beginning
                self.consumer.seek_to_beginning(*topic_partitions)
            elif offset == 'latest':
                # Reset to end
                self.consumer.seek_to_end(*topic_partitions)
            
            logger.info(f"Reset offsets for {len(topics)} topics to {offset}")
            return True
            
        except Exception as e:
            logger.error(f"Error resetting offsets: {e}")
            return False
    
    def get_kafka_metrics(self) -> Dict[str, Any]:
        """
        Get Kafka metrics and statistics.
        
        Returns:
            Dict[str, Any]: Kafka metrics
        """
        if not self.consumer:
            logger.error("Kafka consumer not initialized")
            return {}
        
        metrics = {
            "consumer_metrics": {},
            "producer_metrics": {},
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            # Get consumer metrics
            if hasattr(self.consumer, 'metrics'):
                metrics["consumer_metrics"] = self.consumer.metrics()
            
            # Get producer metrics
            if self.producer and hasattr(self.producer, 'metrics'):
                metrics["producer_metrics"] = self.producer.metrics()
            
            logger.info("Retrieved Kafka metrics")
            
        except Exception as e:
            logger.error(f"Error getting Kafka metrics: {e}")
        
        return metrics
    
    def close_connections(self) -> None:
        """Close Kafka connections."""
        if self.consumer:
            self.consumer.close()
            logger.info("Kafka consumer closed")
        
        if self.producer:
            self.producer.close()
            logger.info("Kafka producer closed")
    
    def get_ingestion_stats(self) -> Dict[str, Any]:
        """
        Get ingestion statistics.
        
        Returns:
            Dict[str, Any]: Ingestion statistics
        """
        return {
            "consumer_initialized": self.consumer is not None,
            "producer_initialized": self.producer is not None,
            "change_handlers_count": len(self.change_handlers),
            "topic_offsets": self.topic_offsets,
            "ingestion_timestamp": datetime.now().isoformat()
        }
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close_connections()
