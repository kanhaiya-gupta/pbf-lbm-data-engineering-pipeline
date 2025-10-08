"""
Powder Bed Stream Processor for PBF-LB/M Data Pipeline

This module provides real-time processing of powder bed monitoring data streams.
"""

import logging
import json
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime
from src.data_pipeline.ingestion.streaming.kafka_consumer import KafkaConsumer
from src.data_pipeline.ingestion.streaming.kafka_producer import KafkaProducer
from src.core.domain.entities.powder_bed import PowderBed
from src.core.domain.events.powder_bed_events import BedPreparedEvent, BedQualityCheckedEvent

logger = logging.getLogger(__name__)


class PowderBedStreamProcessor:
    """
    Real-time processor for powder bed monitoring data streams.
    """
    
    def __init__(self, consumer_config: Optional[Dict[str, Any]] = None, producer_config: Optional[Dict[str, Any]] = None):
        """
        Initialize powder bed stream processor.
        
        Args:
            consumer_config: Optional Kafka consumer configuration
            producer_config: Optional Kafka producer configuration
        """
        self.consumer = KafkaConsumer(consumer_config)
        self.producer = KafkaProducer(producer_config)
        self.quality_assessment_enabled = True
        self.quality_thresholds = {
            "surface_roughness": {"min": 0, "max": 50},  # micrometers
            "surface_quality_score": {"min": 0, "max": 1.0}
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
    
    def _validate_powder_bed_data(self, data: Dict[str, Any]) -> bool:
        """
        Validate powder bed monitoring data.
        
        Args:
            data: Powder bed monitoring data
            
        Returns:
            bool: True if data is valid, False otherwise
        """
        try:
            required_fields = ["process_id", "camera_id", "layer_number", "image_path"]
            for field in required_fields:
                if field not in data:
                    logger.warning(f"Missing required field {field} in powder bed data")
                    return False
            
            # Validate layer number
            if not isinstance(data["layer_number"], int) or data["layer_number"] < 0:
                logger.warning(f"Invalid layer number: {data['layer_number']}")
                return False
            
            return True
        except Exception as e:
            logger.error(f"Error validating powder bed data: {e}")
            return False
    
    def _assess_surface_quality(self, data: Dict[str, Any]) -> float:
        """
        Assess surface quality based on available metrics.
        
        Args:
            data: Powder bed monitoring data
            
        Returns:
            float: Quality score between 0.0 and 1.0
        """
        try:
            quality_score = 1.0
            
            # Check surface roughness if available
            if "surface_roughness" in data and data["surface_roughness"] is not None:
                roughness = float(data["surface_roughness"])
                if roughness > 20:  # micrometers
                    quality_score -= 0.3
                elif roughness > 10:
                    quality_score -= 0.1
            
            # Check image quality indicators if available
            if "image_size" in data and data["image_size"] is not None:
                image_size = int(data["image_size"])
                if image_size < 1000000:  # Less than 1MB might indicate poor quality
                    quality_score -= 0.2
            
            return max(0.0, min(1.0, quality_score))
        except Exception as e:
            logger.error(f"Error assessing surface quality: {e}")
            return 0.5  # Default moderate quality
    
    def _process_powder_bed_message(self, message_data: Dict[str, Any]) -> None:
        """
        Process a single powder bed monitoring message.
        
        Args:
            message_data: Kafka message data containing powder bed monitoring information
        """
        try:
            powder_bed_data = message_data["value"]
            
            # Validate data
            if not self._validate_powder_bed_data(powder_bed_data):
                logger.warning("Invalid powder bed data received, skipping processing")
                return
            
            # Create domain entity
            powder_bed = PowderBed.from_dict(powder_bed_data)
            
            # Emit image captured event
            event = BedPreparedEvent(
                event_id=powder_bed.powder_bed_id,
                event_type="PowderBedImageCaptured",
                powder_bed_id=powder_bed.powder_bed_id,
                timestamp=datetime.now(),
                layer_number=powder_bed.layer_number,
                event_data=powder_bed_data
            )
            self._emit_event(event)
            
            # Assess quality if enabled
            if self.quality_assessment_enabled:
                quality_score = self._assess_surface_quality(powder_bed_data)
                
                # Update powder bed entity with quality score
                powder_bed.surface_quality_score = quality_score
                
                # Emit quality assessed event
                quality_event = BedQualityCheckedEvent(
                    event_id=powder_bed.powder_bed_id,
                    event_type="PowderBedQualityAssessed",
                    powder_bed_id=powder_bed.powder_bed_id,
                    timestamp=datetime.now(),
                    quality_score=quality_score,
                    event_data={"quality_score": quality_score, "layer_number": powder_bed.layer_number}
                )
                self._emit_event(quality_event)
                
                # Send quality assessment to downstream topic
                self.producer.send_message("powder_bed_quality_assessments", {
                    "powder_bed_id": str(powder_bed.powder_bed_id),
                    "process_id": str(powder_bed.process_id),
                    "layer_number": powder_bed.layer_number,
                    "quality_score": quality_score,
                    "assessed_at": datetime.now().isoformat()
                })
            
            # Send processed data to downstream topic
            self.producer.send_message("powder_bed_processed_data", {
                "powder_bed_id": str(powder_bed.powder_bed_id),
                "data": powder_bed_data,
                "processed_at": datetime.now().isoformat(),
                "quality_assessed": self.quality_assessment_enabled
            })
            
            logger.debug(f"Processed powder bed monitoring data for powder_bed_id: {powder_bed.powder_bed_id}")
            
        except Exception as e:
            logger.error(f"Error processing powder bed message: {e}")
    
    def start_processing(self, max_messages: Optional[int] = None) -> None:
        """
        Start processing powder bed monitoring data streams.
        
        Args:
            max_messages: Maximum number of messages to process (None for unlimited)
        """
        logger.info("Starting powder bed stream processing")
        
        try:
            self.consumer.subscribe_to_topic("powder_bed_monitoring_events")
            self.consumer.consume_messages(self._process_powder_bed_message, max_messages)
        except Exception as e:
            logger.error(f"Error in powder bed stream processing: {e}")
        finally:
            self.cleanup()
    
    def set_quality_thresholds(self, thresholds: Dict[str, Dict[str, float]]) -> None:
        """
        Set quality thresholds for quality assessment.
        
        Args:
            thresholds: Dictionary of quality thresholds
        """
        self.quality_thresholds = thresholds
        logger.info("Updated powder bed quality thresholds")
    
    def enable_quality_assessment(self, enabled: bool = True) -> None:
        """
        Enable or disable quality assessment.
        
        Args:
            enabled: Whether to enable quality assessment
        """
        self.quality_assessment_enabled = enabled
        logger.info(f"Quality assessment {'enabled' if enabled else 'disabled'}")
    
    def cleanup(self) -> None:
        """Clean up resources."""
        if self.consumer:
            self.consumer.close()
        if self.producer:
            self.producer.close()
        logger.info("Powder bed stream processor cleaned up")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()
