"""
Change Event Processor for PBF-LB/M Data Pipeline

This module provides processing capabilities for change data capture events.
"""

import logging
import json
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class ChangeEventType(Enum):
    """Types of change events."""
    INSERT = "insert"
    UPDATE = "update"
    DELETE = "delete"
    TRUNCATE = "truncate"
    UNKNOWN = "unknown"


class ChangeEventProcessor:
    """
    Processor for change data capture events.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize change event processor.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.event_handlers = {}
        self.event_filters = {}
        self.processing_stats = {
            "events_processed": 0,
            "events_filtered": 0,
            "events_failed": 0,
            "start_time": datetime.now()
        }
    
    def add_event_handler(self, event_type: str, handler: Callable[[Dict[str, Any]], None]) -> None:
        """
        Add an event handler for specific event type.
        
        Args:
            event_type: Type of event to handle
            handler: Function to handle the event
        """
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)
        logger.info(f"Added event handler for type: {event_type}")
    
    def add_event_filter(self, event_type: str, filter_func: Callable[[Dict[str, Any]], bool]) -> None:
        """
        Add an event filter for specific event type.
        
        Args:
            event_type: Type of event to filter
            filter_func: Function that returns True if event should be processed
        """
        if event_type not in self.event_filters:
            self.event_filters[event_type] = []
        self.event_filters[event_type].append(filter_func)
        logger.info(f"Added event filter for type: {event_type}")
    
    def process_change_event(self, event: Dict[str, Any]) -> bool:
        """
        Process a change event.
        
        Args:
            event: Change event data
            
        Returns:
            bool: True if event was processed successfully, False otherwise
        """
        try:
            # Extract event type
            event_type = self._extract_event_type(event)
            
            # Apply filters
            if not self._apply_filters(event_type, event):
                self.processing_stats["events_filtered"] += 1
                logger.debug(f"Event filtered out: {event_type}")
                return False
            
            # Process event
            success = self._process_event(event_type, event)
            
            if success:
                self.processing_stats["events_processed"] += 1
                logger.debug(f"Event processed successfully: {event_type}")
            else:
                self.processing_stats["events_failed"] += 1
                logger.warning(f"Event processing failed: {event_type}")
            
            return success
            
        except Exception as e:
            self.processing_stats["events_failed"] += 1
            logger.error(f"Error processing change event: {e}")
            return False
    
    def _extract_event_type(self, event: Dict[str, Any]) -> str:
        """
        Extract event type from change event.
        
        Args:
            event: Change event data
            
        Returns:
            str: Event type
        """
        # Try different possible event type fields
        event_type_fields = ['event_type', 'change_type', 'op', 'operation']
        
        for field in event_type_fields:
            if field in event:
                event_type = str(event[field]).lower()
                if event_type in [e.value for e in ChangeEventType]:
                    return event_type
        
        # Default to unknown if no valid event type found
        return ChangeEventType.UNKNOWN.value
    
    def _apply_filters(self, event_type: str, event: Dict[str, Any]) -> bool:
        """
        Apply filters for event type.
        
        Args:
            event_type: Type of event
            event: Change event data
            
        Returns:
            bool: True if event passes filters, False otherwise
        """
        if event_type not in self.event_filters:
            return True  # No filters, process event
        
        for filter_func in self.event_filters[event_type]:
            try:
                if not filter_func(event):
                    return False
            except Exception as e:
                logger.error(f"Error in event filter: {e}")
                return False
        
        return True
    
    def _process_event(self, event_type: str, event: Dict[str, Any]) -> bool:
        """
        Process event with appropriate handlers.
        
        Args:
            event_type: Type of event
            event: Change event data
            
        Returns:
            bool: True if processing was successful, False otherwise
        """
        if event_type not in self.event_handlers:
            logger.warning(f"No handlers found for event type: {event_type}")
            return False
        
        success = True
        for handler in self.event_handlers[event_type]:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Error in event handler: {e}")
                success = False
        
        return success
    
    def process_batch_events(self, events: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Process a batch of change events.
        
        Args:
            events: List of change events
            
        Returns:
            Dict[str, int]: Processing statistics
        """
        logger.info(f"Processing batch of {len(events)} change events")
        
        batch_stats = {
            "total": len(events),
            "processed": 0,
            "filtered": 0,
            "failed": 0
        }
        
        for event in events:
            try:
                if self.process_change_event(event):
                    batch_stats["processed"] += 1
                else:
                    batch_stats["filtered"] += 1
            except Exception as e:
                logger.error(f"Error processing batch event: {e}")
                batch_stats["failed"] += 1
        
        logger.info(f"Batch processing completed: {batch_stats}")
        return batch_stats
    
    def create_event_transformer(self, source_schema: Dict[str, Any], target_schema: Dict[str, Any]) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
        """
        Create an event transformer function.
        
        Args:
            source_schema: Source event schema
            target_schema: Target event schema
            
        Returns:
            Callable: Event transformer function
        """
        def transform_event(event: Dict[str, Any]) -> Dict[str, Any]:
            """
            Transform event from source schema to target schema.
            
            Args:
                event: Source event data
                
            Returns:
                Dict[str, Any]: Transformed event data
            """
            try:
                transformed_event = {}
                
                # Map fields from source to target
                for target_field, source_field in target_schema.items():
                    if source_field in event:
                        transformed_event[target_field] = event[source_field]
                    elif target_field in event:
                        transformed_event[target_field] = event[target_field]
                    else:
                        # Set default value if available
                        if "default" in target_schema[target_field]:
                            transformed_event[target_field] = target_schema[target_field]["default"]
                
                # Add transformation metadata
                transformed_event["_transformed_at"] = datetime.now().isoformat()
                transformed_event["_source_schema"] = source_schema
                transformed_event["_target_schema"] = target_schema
                
                return transformed_event
                
            except Exception as e:
                logger.error(f"Error transforming event: {e}")
                return event
        
        return transform_event
    
    def create_event_validator(self, schema: Dict[str, Any]) -> Callable[[Dict[str, Any]], bool]:
        """
        Create an event validator function.
        
        Args:
            schema: Event schema for validation
            
        Returns:
            Callable: Event validator function
        """
        def validate_event(event: Dict[str, Any]) -> bool:
            """
            Validate event against schema.
            
            Args:
                event: Event data to validate
                
            Returns:
                bool: True if event is valid, False otherwise
            """
            try:
                # Check required fields
                required_fields = schema.get("required", [])
                for field in required_fields:
                    if field not in event:
                        logger.warning(f"Missing required field: {field}")
                        return False
                
                # Validate field types
                properties = schema.get("properties", {})
                for field, value in event.items():
                    if field in properties:
                        field_schema = properties[field]
                        if "type" in field_schema:
                            expected_type = field_schema["type"]
                            if not self._validate_field_type(value, expected_type):
                                logger.warning(f"Invalid type for field {field}: expected {expected_type}, got {type(value)}")
                                return False
                
                return True
                
            except Exception as e:
                logger.error(f"Error validating event: {e}")
                return False
        
        return validate_event
    
    def _validate_field_type(self, value: Any, expected_type: str) -> bool:
        """
        Validate field type.
        
        Args:
            value: Field value
            expected_type: Expected type
            
        Returns:
            bool: True if type is valid, False otherwise
        """
        type_mapping = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "array": list,
            "object": dict
        }
        
        if expected_type in type_mapping:
            expected_python_type = type_mapping[expected_type]
            return isinstance(value, expected_python_type)
        
        return True  # Unknown type, assume valid
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """
        Get processing statistics.
        
        Returns:
            Dict[str, Any]: Processing statistics
        """
        current_time = datetime.now()
        processing_time = (current_time - self.processing_stats["start_time"]).total_seconds()
        
        stats = self.processing_stats.copy()
        stats["processing_time_seconds"] = processing_time
        stats["events_per_second"] = stats["events_processed"] / processing_time if processing_time > 0 else 0
        stats["current_time"] = current_time.isoformat()
        
        return stats
    
    def reset_stats(self) -> None:
        """Reset processing statistics."""
        self.processing_stats = {
            "events_processed": 0,
            "events_filtered": 0,
            "events_failed": 0,
            "start_time": datetime.now()
        }
        logger.info("Processing statistics reset")
    
    def get_ingestion_stats(self) -> Dict[str, Any]:
        """
        Get ingestion statistics.
        
        Returns:
            Dict[str, Any]: Ingestion statistics
        """
        return {
            "event_handlers_count": sum(len(handlers) for handlers in self.event_handlers.values()),
            "event_filters_count": sum(len(filters) for filters in self.event_filters.values()),
            "processing_stats": self.get_processing_stats(),
            "ingestion_timestamp": datetime.now().isoformat()
        }
