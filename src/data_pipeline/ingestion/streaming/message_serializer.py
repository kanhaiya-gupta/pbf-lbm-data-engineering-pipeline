"""
Message Serializer for PBF-LB/M Data Pipeline

This module provides message serialization and deserialization capabilities for streaming data.
"""

import json
import logging
from typing import Dict, Any, Optional, Union, List
from datetime import datetime
from enum import Enum
import avro.schema
import avro.io
import io

logger = logging.getLogger(__name__)


class SerializationFormat(Enum):
    """Supported serialization formats."""
    JSON = "json"
    AVRO = "avro"
    PROTOBUF = "protobuf"


class MessageSerializer:
    """
    Message serializer for PBF-LB/M data pipeline messages.
    """
    
    def __init__(self, default_format: SerializationFormat = SerializationFormat.JSON):
        """
        Initialize message serializer.
        
        Args:
            default_format: Default serialization format
        """
        self.default_format = default_format
        self.avro_schemas = {}
        self._load_avro_schemas()
    
    def _load_avro_schemas(self) -> None:
        """Load Avro schemas for different message types."""
        # In a real implementation, these would be loaded from schema registry or files
        self.avro_schemas = {
            "ispm_monitoring": {
                "type": "record",
                "name": "ISPMMonitoring",
                "fields": [
                    {"name": "monitoring_id", "type": "string"},
                    {"name": "process_id", "type": "string"},
                    {"name": "sensor_id", "type": "string"},
                    {"name": "timestamp", "type": "long"},
                    {"name": "sensor_type", "type": "string"},
                    {"name": "sensor_value", "type": "double"},
                    {"name": "unit", "type": "string"}
                ]
            },
            "powder_bed": {
                "type": "record",
                "name": "PowderBed",
                "fields": [
                    {"name": "powder_bed_id", "type": "string"},
                    {"name": "process_id", "type": "string"},
                    {"name": "camera_id", "type": "string"},
                    {"name": "timestamp", "type": "long"},
                    {"name": "layer_number", "type": "int"},
                    {"name": "image_path", "type": "string"}
                ]
            },
            "pbf_process": {
                "type": "record",
                "name": "PBFProcess",
                "fields": [
                    {"name": "process_id", "type": "string"},
                    {"name": "start_time", "type": "long"},
                    {"name": "machine_id", "type": "string"},
                    {"name": "material_type", "type": "string"},
                    {"name": "process_status", "type": "string"}
                ]
            }
        }
    
    def serialize(self, data: Dict[str, Any], format_type: Optional[SerializationFormat] = None, message_type: Optional[str] = None) -> bytes:
        """
        Serialize data to bytes.
        
        Args:
            data: Data to serialize
            format_type: Serialization format (uses default if None)
            message_type: Message type for Avro serialization
            
        Returns:
            bytes: Serialized data
        """
        format_type = format_type or self.default_format
        
        try:
            if format_type == SerializationFormat.JSON:
                return self._serialize_json(data)
            elif format_type == SerializationFormat.AVRO:
                return self._serialize_avro(data, message_type)
            elif format_type == SerializationFormat.PROTOBUF:
                return self._serialize_protobuf(data)
            else:
                raise ValueError(f"Unsupported serialization format: {format_type}")
        except Exception as e:
            logger.error(f"Error serializing data: {e}")
            raise
    
    def deserialize(self, data: bytes, format_type: Optional[SerializationFormat] = None, message_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Deserialize bytes to data.
        
        Args:
            data: Serialized data
            format_type: Serialization format (uses default if None)
            message_type: Message type for Avro deserialization
            
        Returns:
            Dict[str, Any]: Deserialized data
        """
        format_type = format_type or self.default_format
        
        try:
            if format_type == SerializationFormat.JSON:
                return self._deserialize_json(data)
            elif format_type == SerializationFormat.AVRO:
                return self._deserialize_avro(data, message_type)
            elif format_type == SerializationFormat.PROTOBUF:
                return self._deserialize_protobuf(data)
            else:
                raise ValueError(f"Unsupported serialization format: {format_type}")
        except Exception as e:
            logger.error(f"Error deserializing data: {e}")
            raise
    
    def _serialize_json(self, data: Dict[str, Any]) -> bytes:
        """Serialize data to JSON bytes."""
        return json.dumps(data, default=self._json_serializer).encode('utf-8')
    
    def _deserialize_json(self, data: bytes) -> Dict[str, Any]:
        """Deserialize JSON bytes to data."""
        return json.loads(data.decode('utf-8'))
    
    def _serialize_avro(self, data: Dict[str, Any], message_type: Optional[str] = None) -> bytes:
        """Serialize data to Avro bytes."""
        if not message_type or message_type not in self.avro_schemas:
            raise ValueError(f"Avro schema not found for message type: {message_type}")
        
        schema = avro.schema.parse(json.dumps(self.avro_schemas[message_type]))
        writer = avro.io.DatumWriter(schema)
        bytes_writer = io.BytesIO()
        encoder = avro.io.BinaryEncoder(bytes_writer)
        writer.write(data, encoder)
        return bytes_writer.getvalue()
    
    def _deserialize_avro(self, data: bytes, message_type: Optional[str] = None) -> Dict[str, Any]:
        """Deserialize Avro bytes to data."""
        if not message_type or message_type not in self.avro_schemas:
            raise ValueError(f"Avro schema not found for message type: {message_type}")
        
        schema = avro.schema.parse(json.dumps(self.avro_schemas[message_type]))
        reader = avro.io.DatumReader(schema)
        bytes_reader = io.BytesIO(data)
        decoder = avro.io.BinaryDecoder(bytes_reader)
        return reader.read(decoder)
    
    def _serialize_protobuf(self, data: Dict[str, Any]) -> bytes:
        """Serialize data to Protocol Buffers bytes."""
        # Placeholder for Protocol Buffers serialization
        # In a real implementation, this would use protobuf library
        logger.warning("Protocol Buffers serialization not implemented")
        return self._serialize_json(data)
    
    def _deserialize_protobuf(self, data: bytes) -> Dict[str, Any]:
        """Deserialize Protocol Buffers bytes to data."""
        # Placeholder for Protocol Buffers deserialization
        # In a real implementation, this would use protobuf library
        logger.warning("Protocol Buffers deserialization not implemented")
        return self._deserialize_json(data)
    
    def _json_serializer(self, obj: Any) -> Any:
        """Custom JSON serializer for special types."""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, Enum):
            return obj.value
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    
    def add_avro_schema(self, message_type: str, schema: Dict[str, Any]) -> None:
        """
        Add an Avro schema for a message type.
        
        Args:
            message_type: Message type identifier
            schema: Avro schema definition
        """
        self.avro_schemas[message_type] = schema
        logger.info(f"Added Avro schema for message type: {message_type}")
    
    def get_supported_formats(self) -> List[SerializationFormat]:
        """
        Get list of supported serialization formats.
        
        Returns:
            List[SerializationFormat]: List of supported formats
        """
        return list(SerializationFormat)
    
    def validate_schema(self, data: Dict[str, Any], message_type: str) -> bool:
        """
        Validate data against Avro schema.
        
        Args:
            data: Data to validate
            message_type: Message type for schema validation
            
        Returns:
            bool: True if data is valid, False otherwise
        """
        if message_type not in self.avro_schemas:
            logger.warning(f"No schema found for message type: {message_type}")
            return False
        
        try:
            schema = avro.schema.parse(json.dumps(self.avro_schemas[message_type]))
            # In a real implementation, this would validate the data against the schema
            # For now, we'll do basic validation
            required_fields = [field["name"] for field in schema.fields if field.get("default") is None]
            for field in required_fields:
                if field not in data:
                    logger.warning(f"Missing required field: {field}")
                    return False
            return True
        except Exception as e:
            logger.error(f"Error validating schema: {e}")
            return False
