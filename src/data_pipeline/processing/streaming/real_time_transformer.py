"""
Real-time Transformer

This module provides real-time data transformation capabilities for the PBF-LB/M data pipeline,
including multi-model support for NoSQL data transformation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
import logging
from datetime import datetime, timedelta
import json

from src.data_pipeline.config.pipeline_config import get_pipeline_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealTimeTransformer:
    """
    Real-time data transformer for PBF-LB/M data pipeline.
    """
    
    def __init__(self):
        self.config = get_pipeline_config()
        self.transformation_rules = self._load_transformation_rules()
    
    def _load_transformation_rules(self) -> Dict[str, Any]:
        """Load transformation rules from configuration."""
        try:
            return self.config.get('transformation_rules', {})
        except Exception as e:
            logger.error(f"Error loading transformation rules: {e}")
            return {}
    
    def transform_pbf_process_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform PBF process data in real-time."""
        try:
            transformed_data = data.copy()
            
            # Apply temperature normalization
            if 'temperature' in transformed_data:
                transformed_data['temperature_normalized'] = self._normalize_temperature(
                    transformed_data['temperature']
                )
            
            # Apply pressure normalization
            if 'pressure' in transformed_data:
                transformed_data['pressure_normalized'] = self._normalize_pressure(
                    transformed_data['pressure']
                )
            
            # Calculate process efficiency
            if all(key in transformed_data for key in ['laser_power', 'scan_speed', 'layer_height']):
                transformed_data['process_efficiency'] = self._calculate_process_efficiency(
                    transformed_data['laser_power'],
                    transformed_data['scan_speed'],
                    transformed_data['layer_height']
                )
            
            # Add quality score
            transformed_data['quality_score'] = self._calculate_quality_score(transformed_data)
            
            # Add transformation timestamp
            transformed_data['transformed_at'] = datetime.now().isoformat()
            
            logger.info("PBF process data transformed successfully")
            return transformed_data
            
        except Exception as e:
            logger.error(f"Error transforming PBF process data: {e}")
            return data
    
    def transform_ispm_monitoring_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform ISPM monitoring data in real-time."""
        try:
            transformed_data = data.copy()
            
            # Apply sensor value normalization
            if 'value' in transformed_data and 'sensor_type' in transformed_data:
                transformed_data['value_normalized'] = self._normalize_sensor_value(
                    transformed_data['value'],
                    transformed_data['sensor_type']
                )
            
            # Calculate sensor health score
            if 'sensor_id' in transformed_data:
                transformed_data['sensor_health_score'] = self._calculate_sensor_health_score(
                    transformed_data['sensor_id'],
                    transformed_data.get('value', 0)
                )
            
            # Add anomaly detection
            transformed_data['is_anomaly'] = self._detect_anomaly(transformed_data)
            
            # Add transformation timestamp
            transformed_data['transformed_at'] = datetime.now().isoformat()
            
            logger.info("ISPM monitoring data transformed successfully")
            return transformed_data
            
        except Exception as e:
            logger.error(f"Error transforming ISPM monitoring data: {e}")
            return data
    
    def _normalize_temperature(self, temperature: float) -> float:
        """Normalize temperature value."""
        try:
            # Normalize to 0-1 range based on typical PBF temperature range
            min_temp = 20.0  # Room temperature
            max_temp = 2000.0  # Maximum PBF temperature
            
            normalized = (temperature - min_temp) / (max_temp - min_temp)
            return max(0.0, min(1.0, normalized))
            
        except Exception as e:
            logger.error(f"Error normalizing temperature: {e}")
            return 0.0
    
    def _normalize_pressure(self, pressure: float) -> float:
        """Normalize pressure value."""
        try:
            # Normalize to 0-1 range based on typical PBF pressure range
            min_pressure = 0.0  # Vacuum
            max_pressure = 1000.0  # Maximum PBF pressure (mbar)
            
            normalized = (pressure - min_pressure) / (max_pressure - min_pressure)
            return max(0.0, min(1.0, normalized))
            
        except Exception as e:
            logger.error(f"Error normalizing pressure: {e}")
            return 0.0
    
    def _calculate_process_efficiency(self, laser_power: float, scan_speed: float, 
                                    layer_height: float) -> float:
        """Calculate process efficiency score."""
        try:
            # Simple efficiency calculation based on process parameters
            efficiency = (laser_power * scan_speed) / (layer_height * 1000)
            return min(1.0, max(0.0, efficiency))
            
        except Exception as e:
            logger.error(f"Error calculating process efficiency: {e}")
            return 0.0
    
    def _calculate_quality_score(self, data: Dict[str, Any]) -> float:
        """Calculate overall quality score."""
        try:
            scores = []
            
            # Temperature quality score
            if 'temperature_normalized' in data:
                temp_score = 1.0 - abs(data['temperature_normalized'] - 0.5) * 2
                scores.append(temp_score)
            
            # Pressure quality score
            if 'pressure_normalized' in data:
                pressure_score = 1.0 - abs(data['pressure_normalized'] - 0.5) * 2
                scores.append(pressure_score)
            
            # Process efficiency score
            if 'process_efficiency' in data:
                scores.append(data['process_efficiency'])
            
            return np.mean(scores) if scores else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating quality score: {e}")
            return 0.0
    
    def _normalize_sensor_value(self, value: float, sensor_type: str) -> float:
        """Normalize sensor value based on sensor type."""
        try:
            # Define normalization ranges for different sensor types
            ranges = {
                'temperature': (0, 2000),
                'pressure': (0, 1000),
                'humidity': (0, 100),
                'vibration': (0, 1000)
            }
            
            min_val, max_val = ranges.get(sensor_type, (0, 100))
            normalized = (value - min_val) / (max_val - min_val)
            return max(0.0, min(1.0, normalized))
            
        except Exception as e:
            logger.error(f"Error normalizing sensor value: {e}")
            return 0.0
    
    def _calculate_sensor_health_score(self, sensor_id: str, value: float) -> float:
        """Calculate sensor health score."""
        try:
            # Simple health score calculation
            # In a real implementation, this would consider historical data
            health_score = 1.0 - abs(value - 50.0) / 100.0  # Assuming 50 is optimal
            return max(0.0, min(1.0, health_score))
            
        except Exception as e:
            logger.error(f"Error calculating sensor health score: {e}")
            return 0.0
    
    def _detect_anomaly(self, data: Dict[str, Any]) -> bool:
        """Detect anomalies in sensor data."""
        try:
            # Simple anomaly detection based on value thresholds
            if 'value' in data:
                value = data['value']
                sensor_type = data.get('sensor_type', 'unknown')
                
                # Define anomaly thresholds for different sensor types
                thresholds = {
                    'temperature': (0, 2000),
                    'pressure': (0, 1000),
                    'humidity': (0, 100),
                    'vibration': (0, 1000)
                }
                
                min_threshold, max_threshold = thresholds.get(sensor_type, (0, 100))
                
                if value < min_threshold or value > max_threshold:
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error detecting anomaly: {e}")
            return False
    
    # =============================================================================
    # Multi-Model NoSQL Transformations
    # =============================================================================
    
    def transform_document_data(self, data: Dict[str, Any], 
                               transformation_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Transform document-based data (MongoDB) for real-time processing.
        
        Args:
            data: Input document data
            transformation_config: Optional transformation configuration
            
        Returns:
            Dict: Transformed document data
        """
        try:
            logger.debug("Transforming document data for real-time processing")
            
            if transformation_config is None:
                transformation_config = {}
            
            transformed_data = data.copy()
            
            # Flatten nested structures if needed
            if transformation_config.get('flatten_nested', False):
                transformed_data = self._flatten_document_structure(transformed_data)
            
            # Apply field mappings
            field_mappings = transformation_config.get('field_mappings', {})
            for old_field, new_field in field_mappings.items():
                if old_field in transformed_data:
                    transformed_data[new_field] = transformed_data.pop(old_field)
            
            # Apply data type conversions
            type_conversions = transformation_config.get('type_conversions', {})
            for field, target_type in type_conversions.items():
                if field in transformed_data:
                    transformed_data[field] = self._convert_data_type(
                        transformed_data[field], target_type
                    )
            
            # Add real-time metadata
            transformed_data['transformed_at'] = datetime.utcnow().isoformat()
            transformed_data['transformation_type'] = 'document'
            
            return transformed_data
            
        except Exception as e:
            logger.error(f"Error transforming document data: {e}")
            return data
    
    def transform_key_value_data(self, data: Dict[str, Any],
                                transformation_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Transform key-value data (Redis) for real-time processing.
        
        Args:
            data: Input key-value data
            transformation_config: Optional transformation configuration
            
        Returns:
            Dict: Transformed key-value data
        """
        try:
            logger.debug("Transforming key-value data for real-time processing")
            
            if transformation_config is None:
                transformation_config = {}
            
            transformed_data = data.copy()
            
            # Parse JSON values if they exist
            json_fields = transformation_config.get('json_fields', [])
            for field in json_fields:
                if field in transformed_data and isinstance(transformed_data[field], str):
                    try:
                        transformed_data[field] = json.loads(transformed_data[field])
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse JSON field: {field}")
            
            # Apply key transformations
            key_transformations = transformation_config.get('key_transformations', {})
            for key_pattern, transformation in key_transformations.items():
                if key_pattern in transformed_data:
                    if transformation == 'uppercase':
                        transformed_data[key_pattern] = str(transformed_data[key_pattern]).upper()
                    elif transformation == 'lowercase':
                        transformed_data[key_pattern] = str(transformed_data[key_pattern]).lower()
            
            # Add real-time metadata
            transformed_data['transformed_at'] = datetime.utcnow().isoformat()
            transformed_data['transformation_type'] = 'key_value'
            
            return transformed_data
            
        except Exception as e:
            logger.error(f"Error transforming key-value data: {e}")
            return data
    
    def transform_columnar_data(self, data: Dict[str, Any],
                               transformation_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Transform columnar data (Cassandra) for real-time processing.
        
        Args:
            data: Input columnar data
            transformation_config: Optional transformation configuration
            
        Returns:
            Dict: Transformed columnar data
        """
        try:
            logger.debug("Transforming columnar data for real-time processing")
            
            if transformation_config is None:
                transformation_config = {}
            
            transformed_data = data.copy()
            
            # Ensure partition key is properly formatted
            partition_key = transformation_config.get('partition_key')
            if partition_key and partition_key in transformed_data:
                transformed_data[partition_key] = str(transformed_data[partition_key])
            
            # Ensure clustering keys are properly typed
            clustering_keys = transformation_config.get('clustering_keys', [])
            for key in clustering_keys:
                if key in transformed_data:
                    # Convert to appropriate type for clustering
                    if key == 'timestamp':
                        transformed_data[key] = self._convert_to_timestamp(transformed_data[key])
                    else:
                        transformed_data[key] = str(transformed_data[key])
            
            # Apply columnar-specific transformations
            column_transformations = transformation_config.get('column_transformations', {})
            for column, transformation in column_transformations.items():
                if column in transformed_data:
                    transformed_data[column] = self._apply_column_transformation(
                        transformed_data[column], transformation
                    )
            
            # Add real-time metadata
            transformed_data['transformed_at'] = datetime.utcnow().isoformat()
            transformed_data['transformation_type'] = 'columnar'
            
            return transformed_data
            
        except Exception as e:
            logger.error(f"Error transforming columnar data: {e}")
            return data
    
    def transform_graph_data(self, data: Dict[str, Any],
                            transformation_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Transform graph data (Neo4j) for real-time processing.
        
        Args:
            data: Input graph data
            transformation_config: Optional transformation configuration
            
        Returns:
            Dict: Transformed graph data
        """
        try:
            logger.debug("Transforming graph data for real-time processing")
            
            if transformation_config is None:
                transformation_config = {}
            
            transformed_data = data.copy()
            
            # Extract node properties
            node_properties = transformation_config.get('node_properties', {})
            for prop_name, prop_config in node_properties.items():
                if prop_name in transformed_data:
                    transformed_data[prop_name] = self._transform_node_property(
                        transformed_data[prop_name], prop_config
                    )
            
            # Extract relationship properties
            relationship_properties = transformation_config.get('relationship_properties', {})
            for rel_name, rel_config in relationship_properties.items():
                if rel_name in transformed_data:
                    transformed_data[rel_name] = self._transform_relationship_property(
                        transformed_data[rel_name], rel_config
                    )
            
            # Apply graph-specific transformations
            graph_transformations = transformation_config.get('graph_transformations', {})
            for transformation_type, config in graph_transformations.items():
                if transformation_type == 'node_label_mapping':
                    transformed_data = self._apply_node_label_mapping(transformed_data, config)
                elif transformation_type == 'relationship_type_mapping':
                    transformed_data = self._apply_relationship_type_mapping(transformed_data, config)
            
            # Add real-time metadata
            transformed_data['transformed_at'] = datetime.utcnow().isoformat()
            transformed_data['transformation_type'] = 'graph'
            
            return transformed_data
            
        except Exception as e:
            logger.error(f"Error transforming graph data: {e}")
            return data
    
    def transform_multi_model_data(self, data: Dict[str, Any], target_model: str,
                                  transformation_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Transform data for a specific NoSQL model.
        
        Args:
            data: Input data
            target_model: Target NoSQL model (mongodb, redis, cassandra, neo4j)
            transformation_config: Optional transformation configuration
            
        Returns:
            Dict: Transformed data for the target model
        """
        try:
            logger.debug(f"Transforming data for {target_model} model")
            
            if target_model.lower() == 'mongodb':
                return self.transform_document_data(data, transformation_config)
            elif target_model.lower() == 'redis':
                return self.transform_key_value_data(data, transformation_config)
            elif target_model.lower() == 'cassandra':
                return self.transform_columnar_data(data, transformation_config)
            elif target_model.lower() == 'neo4j':
                return self.transform_graph_data(data, transformation_config)
            else:
                logger.warning(f"Unknown target model: {target_model}")
                return data
                
        except Exception as e:
            logger.error(f"Error transforming multi-model data: {e}")
            return data
    
    # =============================================================================
    # Helper Methods for Multi-Model Transformations
    # =============================================================================
    
    def _flatten_document_structure(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Flatten nested document structures."""
        try:
            flattened = {}
            
            for key, value in data.items():
                if isinstance(value, dict):
                    # Recursively flatten nested dictionaries
                    nested_flattened = self._flatten_document_structure(value)
                    for nested_key, nested_value in nested_flattened.items():
                        flattened[f"{key}_{nested_key}"] = nested_value
                elif isinstance(value, list):
                    # Handle arrays
                    flattened[f"{key}_array"] = value
                    flattened[f"{key}_count"] = len(value)
                else:
                    flattened[key] = value
            
            return flattened
            
        except Exception as e:
            logger.error(f"Error flattening document structure: {e}")
            return data
    
    def _convert_data_type(self, value: Any, target_type: str) -> Any:
        """Convert value to target data type."""
        try:
            if target_type == 'string':
                return str(value)
            elif target_type == 'integer':
                return int(value)
            elif target_type == 'float':
                return float(value)
            elif target_type == 'boolean':
                return bool(value)
            elif target_type == 'timestamp':
                return self._convert_to_timestamp(value)
            else:
                return value
                
        except Exception as e:
            logger.error(f"Error converting data type: {e}")
            return value
    
    def _convert_to_timestamp(self, value: Any) -> str:
        """Convert value to timestamp string."""
        try:
            if isinstance(value, datetime):
                return value.isoformat()
            elif isinstance(value, str):
                # Try to parse and convert
                dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
                return dt.isoformat()
            else:
                return datetime.utcnow().isoformat()
                
        except Exception as e:
            logger.error(f"Error converting to timestamp: {e}")
            return datetime.utcnow().isoformat()
    
    def _apply_column_transformation(self, value: Any, transformation: str) -> Any:
        """Apply column-specific transformation."""
        try:
            if transformation == 'normalize':
                # Normalize numeric values
                if isinstance(value, (int, float)):
                    return (value - 0) / (100 - 0) if value != 0 else 0
            elif transformation == 'round':
                # Round numeric values
                if isinstance(value, float):
                    return round(value, 2)
            elif transformation == 'hash':
                # Hash string values
                if isinstance(value, str):
                    return hash(value) % 1000000
            
            return value
            
        except Exception as e:
            logger.error(f"Error applying column transformation: {e}")
            return value
    
    def _transform_node_property(self, value: Any, config: Dict[str, Any]) -> Any:
        """Transform node property according to configuration."""
        try:
            if config.get('type') == 'string' and not isinstance(value, str):
                return str(value)
            elif config.get('type') == 'numeric' and isinstance(value, str):
                try:
                    return float(value)
                except ValueError:
                    return 0
            elif config.get('required') and value is None:
                return config.get('default', '')
            
            return value
            
        except Exception as e:
            logger.error(f"Error transforming node property: {e}")
            return value
    
    def _transform_relationship_property(self, value: Any, config: Dict[str, Any]) -> Any:
        """Transform relationship property according to configuration."""
        try:
            if config.get('type') == 'weight' and isinstance(value, (int, float)):
                # Normalize weight values
                return max(0, min(1, value))
            elif config.get('type') == 'timestamp':
                return self._convert_to_timestamp(value)
            
            return value
            
        except Exception as e:
            logger.error(f"Error transforming relationship property: {e}")
            return value
    
    def _apply_node_label_mapping(self, data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply node label mapping."""
        try:
            label_field = config.get('label_field', 'type')
            label_mappings = config.get('mappings', {})
            
            if label_field in data:
                current_label = data[label_field]
                if current_label in label_mappings:
                    data[label_field] = label_mappings[current_label]
            
            return data
            
        except Exception as e:
            logger.error(f"Error applying node label mapping: {e}")
            return data
    
    def _apply_relationship_type_mapping(self, data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply relationship type mapping."""
        try:
            rel_field = config.get('relationship_field', 'relationship_type')
            rel_mappings = config.get('mappings', {})
            
            if rel_field in data:
                current_type = data[rel_field]
                if current_type in rel_mappings:
                    data[rel_field] = rel_mappings[current_type]
            
            return data
            
        except Exception as e:
            logger.error(f"Error applying relationship type mapping: {e}")
            return data
