"""
ISPM Stream Processor for PBF-LB/M Data Pipeline

This module provides real-time processing of ISPM (In-Situ Process Monitoring) data streams.
"""

import logging
import json
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime
from src.data_pipeline.ingestion.streaming.kafka_consumer import KafkaConsumer
from src.data_pipeline.ingestion.streaming.kafka_producer import KafkaProducer
from src.core.domain.entities.ispm_monitoring import ISPMMonitoring
from src.core.domain.events.ispm_monitoring_events import MonitoringStartedEvent, AnomalyDetectedEvent

logger = logging.getLogger(__name__)


class ISPMStreamProcessor:
    """
    Real-time processor for ISPM monitoring data streams.
    """
    
    def __init__(self, consumer_config: Optional[Dict[str, Any]] = None, producer_config: Optional[Dict[str, Any]] = None):
        """
        Initialize ISPM stream processor.
        
        Args:
            consumer_config: Optional Kafka consumer configuration
            producer_config: Optional Kafka producer configuration
        """
        self.consumer = KafkaConsumer(consumer_config)
        self.producer = KafkaProducer(producer_config)
        self.anomaly_detection_enabled = True
        self.quality_thresholds = {
            "temperature": {"min": 20, "max": 3000},
            "pressure": {"min": 0, "max": 1000},
            "laser_power": {"min": 0, "max": 1000},
            "scan_speed": {"min": 0, "max": 5000}
        }
        self.event_handlers = []
    
    def add_event_handler(self, handler: Callable[[Any], None]) -> None:
        """
        Add an event handler for domain events.
        
        Args:
            handler: Function to handle domain events
        """
        self.event_handlers.append(handler)
    
    def _emit_event(self, event: Any) -> None:
        """
        Emit a domain event to all registered handlers.
        
        Args:
            event: Domain event to emit
        """
        for handler in self.event_handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Error in event handler: {e}")
    
    def _validate_ispm_data(self, data: Dict[str, Any]) -> bool:
        """
        Validate ISPM monitoring data against quality thresholds.
        
        Args:
            data: ISPM monitoring data
            
        Returns:
            bool: True if data is valid, False otherwise
        """
        try:
            for metric, thresholds in self.quality_thresholds.items():
                if metric in data:
                    value = float(data[metric])
                    if value < thresholds["min"] or value > thresholds["max"]:
                        logger.warning(f"ISPM data validation failed for {metric}: {value} not in range [{thresholds['min']}, {thresholds['max']}]")
                        return False
            return True
        except (ValueError, TypeError) as e:
            logger.error(f"Error validating ISPM data: {e}")
            return False
    
    def _detect_anomalies(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Detect anomalies in ISPM monitoring data.
        
        Args:
            data: ISPM monitoring data
            
        Returns:
            List[Dict[str, Any]]: List of detected anomalies
        """
        anomalies = []
        
        # Simple anomaly detection based on thresholds
        for metric, thresholds in self.quality_thresholds.items():
            if metric in data:
                value = float(data[metric])
                if value < thresholds["min"] * 0.8 or value > thresholds["max"] * 1.2:
                    anomalies.append({
                        "metric": metric,
                        "value": value,
                        "threshold_min": thresholds["min"],
                        "threshold_max": thresholds["max"],
                        "severity": "high" if value < thresholds["min"] * 0.5 or value > thresholds["max"] * 1.5 else "medium"
                    })
        
        return anomalies
    
    def _process_ispm_message(self, message_data: Dict[str, Any]) -> None:
        """
        Process a single ISPM monitoring message.
        
        Args:
            message_data: Kafka message data containing ISPM monitoring information
        """
        try:
            ispm_data = message_data["value"]
            
            # Validate data
            if not self._validate_ispm_data(ispm_data):
                logger.warning("Invalid ISPM data received, skipping processing")
                return
            
            # Create domain entity
            ispm_monitoring = ISPMMonitoring.from_dict(ispm_data)
            
            # Emit data received event
            event = MonitoringStartedEvent(
                event_id=ispm_monitoring.monitoring_id,
                event_type="ISPMMonitoringDataReceived",
                monitoring_id=ispm_monitoring.monitoring_id,
                timestamp=datetime.now(),
                event_data=ispm_data
            )
            self._emit_event(event)
            
            # Detect anomalies if enabled
            if self.anomaly_detection_enabled:
                anomalies = self._detect_anomalies(ispm_data)
                if anomalies:
                    for anomaly in anomalies:
                        anomaly_event = AnomalyDetectedEvent(
                            event_id=ispm_monitoring.monitoring_id,
                            event_type="ISPMMonitoringAnomalyDetected",
                            monitoring_id=ispm_monitoring.monitoring_id,
                            timestamp=datetime.now(),
                            anomaly_type=anomaly["metric"],
                            severity=1.0 if anomaly["severity"] == "high" else 0.5,
                            event_data=anomaly
                        )
                        self._emit_event(anomaly_event)
                        
                        # Send anomaly to alert topic
                        self.producer.send_message("ispm_anomaly_alerts", {
                            "monitoring_id": str(ispm_monitoring.monitoring_id),
                            "anomaly": anomaly,
                            "timestamp": datetime.now().isoformat()
                        })
            
            # Send processed data to downstream topic
            self.producer.send_message("ispm_processed_data", {
                "monitoring_id": str(ispm_monitoring.monitoring_id),
                "data": ispm_data,
                "processed_at": datetime.now().isoformat(),
                "anomalies_detected": len(anomalies) if self.anomaly_detection_enabled else 0
            })
            
            logger.debug(f"Processed ISPM monitoring data for monitoring_id: {ispm_monitoring.monitoring_id}")
            
        except Exception as e:
            logger.error(f"Error processing ISPM message: {e}")
    
    def start_processing(self, max_messages: Optional[int] = None) -> None:
        """
        Start processing ISPM monitoring data streams.
        
        Args:
            max_messages: Maximum number of messages to process (None for unlimited)
        """
        logger.info("Starting ISPM stream processing")
        
        try:
            self.consumer.subscribe_to_topic("ispm_monitoring_events")
            self.consumer.consume_messages(self._process_ispm_message, max_messages)
        except Exception as e:
            logger.error(f"Error in ISPM stream processing: {e}")
        finally:
            self.cleanup()
    
    def set_quality_thresholds(self, thresholds: Dict[str, Dict[str, float]]) -> None:
        """
        Set quality thresholds for anomaly detection.
        
        Args:
            thresholds: Dictionary of metric thresholds
        """
        self.quality_thresholds = thresholds
        logger.info("Updated ISPM quality thresholds")
    
    def enable_anomaly_detection(self, enabled: bool = True) -> None:
        """
        Enable or disable anomaly detection.
        
        Args:
            enabled: Whether to enable anomaly detection
        """
        self.anomaly_detection_enabled = enabled
        logger.info(f"Anomaly detection {'enabled' if enabled else 'disabled'}")
    
    def cleanup(self) -> None:
        """Clean up resources."""
        if self.consumer:
            self.consumer.close()
        if self.producer:
            self.producer.close()
        logger.info("ISPM stream processor cleaned up")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()
