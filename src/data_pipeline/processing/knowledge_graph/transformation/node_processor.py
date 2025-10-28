"""
Schema-Aware Node Processor for Knowledge Graph Transformation

This module processes and normalizes nodes from different data sources
into schema-compliant format using Pydantic validation models.
"""

import logging
from typing import Dict, List, Any, Optional, Set
from datetime import datetime, date
import uuid
import json
import os
from pathlib import Path

from src.data_pipeline.processing.knowledge_graph.utils.json_parser import safe_json_loads_with_fallback

# Import our Neo4j models and validation
from src.data_pipeline.processing.schema.models.neo4j_models import (
    Neo4jModelFactory, GraphValidationEngine,
    ProcessModel, MachineModel, PartModel, BuildModel, MaterialModel,
    QualityModel, SensorModel, UserModel, OperatorModel, AlertModel,
    DefectModel, ImageModel, LogModel, InspectionModel
)

logger = logging.getLogger(__name__)


class NodeProcessor:
    """
    Schema-aware node processor that transforms raw data into Neo4j schema-compliant nodes.
    
    Handles:
    - Schema field mapping from raw data to Neo4j schema
    - Pydantic model validation for data quality
    - Manufacturing-specific business logic validation
    - Export to kg_neo4j data lake structure
    """
    
    def __init__(self, output_dir: str = "data_lake/kg_neo4j"):
        """Initialize the schema-aware node processor."""
        self.factory = Neo4jModelFactory()
        self.validation_engine = GraphValidationEngine()
        
        # Ensure all models are loaded immediately
        self.factory.ensure_models_loaded()
        
        # Debug: Check factory initialization
        logger.debug(f"Factory models after initialization: {list(self.factory._models.keys())}")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create output subdirectories
        (self.output_dir / "nodes").mkdir(exist_ok=True)
        (self.output_dir / "relationships").mkdir(exist_ok=True)
        (self.output_dir / "graph_metadata").mkdir(exist_ok=True)
        
        self.processed_nodes: Dict[str, Dict[str, Any]] = {}
        self.node_id_mapping: Dict[str, str] = {}
        self.duplicate_nodes: List[Dict[str, Any]] = []
        self.validation_results: List[Dict[str, Any]] = []
        
    def process_postgresql_nodes(self, nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process PostgreSQL nodes using schema mapping and Pydantic validation.
        
        Args:
            nodes: List of PostgreSQL node data
            
        Returns:
            List[Dict[str, Any]]: Schema-compliant processed nodes
        """
        processed = []
        
        for node in nodes:
            try:
                # First normalize the node to get basic structure
                normalized_node = self._normalize_node(node, source="postgresql")
                if not normalized_node:
                    continue
                
                # Map to schema fields based on node type
                schema_node = self._map_to_schema_fields(node, source="postgresql")
                if schema_node:
                    # Merge schema fields into normalized node
                    normalized_node.update(schema_node)
                    
                    # Validate with Pydantic model
                    validated_node = self._validate_with_pydantic(normalized_node)
                    if validated_node:
                        processed.append(validated_node)
                        self._save_to_kg_neo4j(validated_node, "nodes")
            except Exception as e:
                # Check if this is a JSON parsing error that's already handled by safe parser
                if any(json_error in str(e) for json_error in [
                    "Expecting property name enclosed in double quotes",
                    "Extra data: line",
                    "Expecting value: line",
                    "Unterminated string",
                    "Expecting ',' delimiter",
                    "Expecting ':' delimiter"
                ]):
                    # These are JSON parsing errors that are already handled by safe_json_loads_with_fallback
                    # Don't log them as warnings since they're handled gracefully
                    continue
                else:
                    # This is a real error that needs to be logged
                    logger.warning(f"Failed to process PostgreSQL node: {e}")
                    continue
                
        logger.info(f"✅ Processed {len(processed)} PostgreSQL nodes with schema validation")
        return processed
    
    def export_kg_neo4j_data(self) -> Dict[str, Any]:
        """
        Export complete kg_neo4j data lake structure with metadata.
        
        Returns:
            Dict[str, Any]: Complete data lake structure
        """
        # Create metadata
        metadata = {
            'export_timestamp': datetime.utcnow().isoformat(),
            'total_nodes': len(self.processed_nodes),
            'validation_results': self.validation_results,
            'schema_version': '1.0.0',
            'data_sources': list(set([r.get('source', 'unknown') for r in self.validation_results])),
            'node_types': list(set([r.get('node_type', 'unknown') for r in self.validation_results]))
        }
        
        # Save metadata
        metadata_file = self.output_dir / "graph_metadata" / "export_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        # Create summary
        summary = {
            'metadata': metadata,
            'nodes_directory': str(self.output_dir / "nodes"),
            'relationships_directory': str(self.output_dir / "relationships"),
            'graph_metadata_directory': str(self.output_dir / "graph_metadata")
        }
        
        logger.info(f"✅ Exported kg_neo4j data lake to {self.output_dir}")
        logger.info(f"📊 Total nodes: {metadata['total_nodes']}")
        logger.info(f"📁 Data sources: {metadata['data_sources']}")
        logger.info(f"🏷️ Node types: {metadata['node_types']}")
        
        return summary
    
    def process_mongodb_nodes(self, nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process MongoDB nodes using schema mapping and Pydantic validation.
        
        Args:
            nodes: List of MongoDB node data
            
        Returns:
            List[Dict[str, Any]]: Schema-compliant processed nodes
        """
        processed = []
        
        for node in nodes:
            try:
                # First normalize the node to get basic structure
                normalized_node = self._normalize_node(node, source="mongodb")
                if not normalized_node:
                    continue
                
                # Map to schema fields based on node type
                schema_node = self._map_to_schema_fields(node, source="mongodb")
                if schema_node:
                    # Merge schema fields into normalized node
                    normalized_node.update(schema_node)
                    
                    # Validate with Pydantic model
                    validated_node = self._validate_with_pydantic(normalized_node)
                    if validated_node:
                        processed.append(validated_node)
                        self._save_to_kg_neo4j(validated_node, "nodes")
            except Exception as e:
                # Check if this is a JSON parsing error that's already handled by safe parser
                if any(json_error in str(e) for json_error in [
                    "Expecting property name enclosed in double quotes",
                    "Extra data: line",
                    "Expecting value: line",
                    "Unterminated string",
                    "Expecting ',' delimiter",
                    "Expecting ':' delimiter"
                ]):
                    # These are JSON parsing errors that are already handled by safe_json_loads_with_fallback
                    # Don't log them as warnings since they're handled gracefully
                    continue
                else:
                    # This is a real error that needs to be logged
                    logger.warning(f"Failed to process MongoDB node: {e}")
                    continue
                
        logger.info(f"✅ Processed {len(processed)} MongoDB nodes with schema validation")
        return processed
    
    def process_cassandra_nodes(self, nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process Cassandra nodes using schema mapping and Pydantic validation.
        
        Args:
            nodes: List of Cassandra node data
            
        Returns:
            List[Dict[str, Any]]: Schema-compliant processed nodes
        """
        processed = []
        
        for node in nodes:
            try:
                # First normalize the node to get basic structure
                normalized_node = self._normalize_node(node, source="cassandra")
                if not normalized_node:
                    continue
                
                # Map to schema fields based on node type
                schema_node = self._map_to_schema_fields(node, source="cassandra")
                if schema_node:
                    # Merge schema fields into normalized node
                    normalized_node.update(schema_node)
                    
                    # Validate with Pydantic model
                    validated_node = self._validate_with_pydantic(normalized_node)
                    if validated_node:
                        processed.append(validated_node)
                        self._save_to_kg_neo4j(validated_node, "nodes")
            except Exception as e:
                # Check if this is a JSON parsing error that's already handled by safe parser
                if any(json_error in str(e) for json_error in [
                    "Expecting property name enclosed in double quotes",
                    "Extra data: line",
                    "Expecting value: line",
                    "Unterminated string",
                    "Expecting ',' delimiter",
                    "Expecting ':' delimiter"
                ]):
                    # These are JSON parsing errors that are already handled by safe_json_loads_with_fallback
                    # Don't log them as warnings since they're handled gracefully
                    continue
                else:
                    # This is a real error that needs to be logged
                    logger.warning(f"Failed to process Cassandra node: {e}")
                    continue
                
        logger.info(f"✅ Processed {len(processed)} Cassandra nodes with schema validation")
        return processed
    
    def process_redis_nodes(self, nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process Redis nodes using schema mapping and Pydantic validation.
        
        Args:
            nodes: List of Redis node data
            
        Returns:
            List[Dict[str, Any]]: Schema-compliant processed nodes
        """
        processed = []
        
        for node in nodes:
            try:
                # First normalize the node to get basic structure
                normalized_node = self._normalize_node(node, source="redis")
                if not normalized_node:
                    continue
                
                # Map to schema fields based on node type
                schema_node = self._map_to_schema_fields(node, source="redis")
                if schema_node:
                    # Merge schema fields into normalized node
                    normalized_node.update(schema_node)
                    
                    # Validate with Pydantic model
                    validated_node = self._validate_with_pydantic(normalized_node)
                    if validated_node:
                        processed.append(validated_node)
                        self._save_to_kg_neo4j(validated_node, "nodes")
            except Exception as e:
                # Check if this is a JSON parsing error that's already handled by safe parser
                if any(json_error in str(e) for json_error in [
                    "Expecting property name enclosed in double quotes",
                    "Extra data: line",
                    "Expecting value: line",
                    "Unterminated string",
                    "Expecting ',' delimiter",
                    "Expecting ':' delimiter"
                ]):
                    # These are JSON parsing errors that are already handled by safe_json_loads_with_fallback
                    # Don't log them as warnings since they're handled gracefully
                    continue
                else:
                    # This is a real error that needs to be logged
                    logger.warning(f"Failed to process Redis node: {e}")
                    continue
                
        logger.info(f"✅ Processed {len(processed)} Redis nodes with schema validation")
        return processed
    
    def _normalize_node(self, node: Dict[str, Any], source: str) -> Optional[Dict[str, Any]]:
        """
        Normalize a node from any source into a standard format.
        
        Args:
            node: Raw node data
            source: Source database name
            
        Returns:
            Optional[Dict[str, Any]]: Normalized node or None if invalid
        """
        if not node or not isinstance(node, dict):
            return None
        
        # Extract node type and ID
        node_type = node.get('node_type', 'Unknown')
        node_id = (node.get('id') or node.get('_id') or node.get('node_id') or 
                  node.get('document_id') or node.get('session_id') or 
                  node.get('user_id') or node.get('cache_key') or
                  node.get('part_id') or node.get('process_id') or 
                  node.get('machine_id') or node.get('build_id') or
                  node.get('sensor_id') or node.get('quality_id') or
                  node.get('aggregation_id') or node.get('alert_id') or
                  node.get('reading_id') or node.get('material_type') or
                  node.get('sensor_type'))
        
        if not node_id:
            logger.warning(f"No ID found for node: {node}")
            return None
        
        # Generate unique graph ID
        graph_id = self._generate_graph_id(node_type, node_id, source)
        
        # Normalize properties
        properties = self._normalize_properties(node)
        
        # Create normalized node
        normalized_node = {
            'graph_id': graph_id,
            'node_type': node_type,
            'source_id': str(node_id),
            'source': source,
            'properties': properties,
            'labels': self._generate_labels(node_type, properties),
            'created_at': datetime.utcnow().isoformat(),
            'updated_at': datetime.utcnow().isoformat()
        }
        
        # Preserve relationship fields at the top level for relationship extraction
        relationship_fields = ['machine_id', 'build_id', 'part_id', 'process_id', 'sensor_id', 'quality_id', 'material_id', 'document_id', 'job_id', 'session_id', 'user_id']
        for field in relationship_fields:
            if field in node:
                normalized_node[field] = node[field]
        
        # Check for duplicates
        if self._is_duplicate(normalized_node):
            self.duplicate_nodes.append(normalized_node)
            return None
        
        # Store processed node
        self.processed_nodes[graph_id] = normalized_node
        self.node_id_mapping[f"{source}:{node_id}"] = graph_id
        
        return normalized_node
    
    def _map_to_schema_fields(self, node: Dict[str, Any], source: str) -> Optional[Dict[str, Any]]:
        """
        Map raw data to Neo4j schema fields based on node type.
        
        Args:
            node: Raw node data
            source: Source database name
            
        Returns:
            Optional[Dict[str, Any]]: Schema-mapped node data
        """
        node_type = node.get('node_type', 'Unknown')
        
        # Map based on node type to schema fields
        if node_type.lower() == 'process':
            schema_node = self._map_process_fields(node, source)
        elif node_type.lower() == 'machine':
            schema_node = self._map_machine_fields(node, source)
        elif node_type.lower() == 'part':
            schema_node = self._map_part_fields(node, source)
        elif node_type.lower() == 'build':
            schema_node = self._map_build_fields(node, source)
        elif node_type.lower() == 'material':
            schema_node = self._map_material_fields(node, source)
        elif node_type.lower() == 'quality':
            schema_node = self._map_quality_fields(node, source)
        elif node_type.lower() == 'sensor':
            schema_node = self._map_sensor_fields(node, source)
        elif node_type.lower() == 'user':
            schema_node = self._map_user_fields(node, source)
        elif node_type.lower() == 'operator':
            schema_node = self._map_operator_fields(node, source)
        elif node_type.lower() == 'alert':
            schema_node = self._map_alert_fields(node, source)
        elif node_type.lower() == 'defect':
            schema_node = self._map_defect_fields(node, source)
        elif node_type.lower() == 'image':
            schema_node = self._map_image_fields(node, source)
        elif node_type.lower() == 'log':
            schema_node = self._map_log_fields(node, source)
        elif node_type.lower() == 'inspection':
            schema_node = self._map_inspection_fields(node, source)
        # New node types - handle both lowercase and proper case
        elif node_type.lower() == 'processimage' or node_type == 'ProcessImage' or node_type == 'process_image':
            schema_node = self._map_process_image_fields(node, source)
        elif node_type.lower() == 'ctscanimage' or node_type == 'CTScanImage' or node_type == 'ct_scan_image':
            schema_node = self._map_ct_scan_image_fields(node, source)
        elif node_type.lower() == 'powderbedimage' or node_type == 'PowderBedImage' or node_type == 'powder_bed_image':
            schema_node = self._map_powder_bed_image_fields(node, source)
        elif node_type.lower() == 'machinebuildfile' or node_type == 'MachineBuildFile' or node_type == 'machine_build_file':
            schema_node = self._map_build_file_fields(node, source)
        elif node_type.lower() == 'model3dfile' or node_type == 'Model3DFile' or node_type == 'model_3d_file':
            schema_node = self._map_model_file_fields(node, source)
        elif node_type.lower() == 'rawsensordata' or node_type == 'RawSensorData' or node_type == 'raw_sensor_data':
            schema_node = self._map_sensor_reading_fields(node, source)
        elif node_type.lower() == 'processlog' or node_type == 'ProcessLog' or node_type == 'process_log':
            schema_node = self._map_log_file_fields(node, source)
        elif node_type.lower() == 'machineconfig' or node_type == 'MachineConfig' or node_type == 'machine_configuration':
            schema_node = self._map_machine_config_fields(node, source)
        elif node_type.lower() == 'buildfile' or node_type == 'BuildFile':
            schema_node = self._map_build_file_fields(node, source)
        elif node_type.lower() == 'modelfile' or node_type == 'ModelFile':
            schema_node = self._map_model_file_fields(node, source)
        elif node_type.lower() == 'logfile' or node_type == 'LogFile':
            schema_node = self._map_log_file_fields(node, source)
        elif node_type.lower() == 'processcache' or node_type == 'ProcessCache':
            schema_node = self._map_process_cache_fields(node, source)
        elif node_type.lower() == 'analyticscache' or node_type == 'AnalyticsCache':
            schema_node = self._map_analytics_cache_fields(node, source)
        elif node_type.lower() == 'jobqueue' or node_type == 'JobQueue':
            schema_node = self._map_job_queue_fields(node, source)
        elif node_type.lower() == 'usersession' or node_type == 'UserSession':
            schema_node = self._map_user_session_fields(node, source)
        elif node_type.lower() == 'sensorreading' or node_type == 'SensorReading':
            schema_node = self._map_sensor_reading_fields(node, source)
        elif node_type.lower() == 'processmonitoring' or node_type == 'ProcessMonitoring':
            schema_node = self._map_process_monitoring_fields(node, source)
        elif node_type.lower() == 'machinestatus' or node_type == 'MachineStatus':
            schema_node = self._map_machine_status_fields(node, source)
        elif node_type.lower() == 'alertevent' or node_type == 'AlertEvent':
            schema_node = self._map_alert_event_fields(node, source)
        elif node_type.lower() == 'sensortype' or node_type == 'SensorType' or node_type.lower() == 'sensor_types':
            # Handle both regular SensorType nodes and PostgreSQL sensor_types relationship data
            if node_type.lower() == 'sensor_types':
                node['node_type'] = 'SensorType'
            schema_node = self._map_sensor_type_fields(node, source)
        # Handle PostgreSQL relationship data types
        elif node_type.lower() == 'quality_metrics':
            # Map quality_metrics to Quality nodes (not QualityMetric)
            node['node_type'] = 'Quality'
            schema_node = self._map_quality_fields(node, source)
        elif node_type.lower() == 'material_properties':
            # Map material_properties to Material nodes
            node['node_type'] = 'Material'
            schema_node = self._map_material_fields(node, source)
        else:
            logger.warning(f"Unknown node type: {node_type}")
            return None
        
        # Add node_type to the schema node for validation
        if schema_node:
            # Use the updated node_type if it was changed (for relationship data mapping)
            schema_node['node_type'] = node.get('node_type', node_type)
        
        return schema_node
    
    def _convert_to_neo4j_model_type(self, node_type: str) -> str:
        """Convert MongoDB node types to Neo4j model types."""
        conversion_map = {
            'process_image': 'ProcessImage',
            'ct_scan_image': 'CTScanImage', 
            'powder_bed_image': 'PowderBedImage',
            'machine_build_file': 'BuildFile',
            'model_3d_file': 'ModelFile',
            'raw_sensor_data': 'SensorReading',
            'process_log': 'LogFile',
            'machine_configuration': 'MachineConfig'
        }
        
        # Check for exact match first
        if node_type in conversion_map:
            return conversion_map[node_type]
        
        # Check for case-insensitive match
        for mongo_type, neo4j_type in conversion_map.items():
            if node_type.lower() == mongo_type.lower():
                return neo4j_type
        
        # If no conversion found, return the original type (might be already correct)
        return node_type
    
    def _map_process_fields(self, node: Dict[str, Any], source: str) -> Dict[str, Any]:
        """Map raw data to Process schema fields."""
        # Map status to valid enum values
        status = node.get('status') or node.get('process_status') or 'unknown'
        if status.lower() in ['active', 'running', 'operational']:
            mapped_status = 'running'
        elif status.lower() in ['inactive', 'idle', 'stopped']:
            mapped_status = 'pending'
        elif status.lower() in ['maintenance', 'calibrating']:
            mapped_status = 'paused'
        elif status.lower() in ['error', 'failed']:
            mapped_status = 'failed'
        elif status.lower() in ['completed', 'finished']:
            mapped_status = 'completed'
        elif status.lower() in ['cancelled', 'aborted']:
            mapped_status = 'cancelled'
        else:
            mapped_status = 'unknown'
        
        # Handle quality_score to density conversion (0-100 to 0-1)
        quality_score = node.get('quality_score')
        if quality_score is not None:
            # Convert from 0-100 range to 0-1 range for density
            density = float(quality_score) / 100.0 if quality_score else None
        else:
            density = None
        
        # Map quality_grade from quality_score
        quality_grade = None
        if quality_score is not None:
            score = float(quality_score)
            if score >= 90:
                quality_grade = 'A'
            elif score >= 80:
                quality_grade = 'B'
            elif score >= 70:
                quality_grade = 'C'
            elif score >= 60:
                quality_grade = 'D'
            else:
                quality_grade = 'F'
        
        return {
            'process_id': node.get('id') or node.get('process_id') or f"PROC_{uuid.uuid4().hex[:8]}",
            'timestamp': self._parse_timestamp(node.get('timestamp') or node.get('created_at')),
            'material_type': node.get('material_type') or node.get('material') or 'Unknown',
            'quality_grade': quality_grade,
            'laser_power': float(node.get('laser_power', 0)) if node.get('laser_power') else None,
            'scan_speed': float(node.get('scan_speed', 0)) if node.get('scan_speed') else None,
            'layer_thickness': float(node.get('layer_thickness', 0)) if node.get('layer_thickness') else None,
            'density': density,
            'surface_roughness': float(node.get('surface_roughness', 0)) if node.get('surface_roughness') else None,
            'status': mapped_status,
            'duration': int(node.get('duration', 0)) if node.get('duration') else None,
            'energy_consumption': float(node.get('energy_consumption', 0)) if node.get('energy_consumption') else None,
            'powder_usage': float(node.get('powder_usage', 0)) if node.get('powder_usage') else None,
            'build_temperature': float(node.get('build_temperature', 0)) if node.get('build_temperature') else None,
            'chamber_pressure': float(node.get('chamber_pressure', 0)) if node.get('chamber_pressure') else None,
            'hatch_spacing': float(node.get('hatch_spacing', 0)) if node.get('hatch_spacing') else None,
            'exposure_time': float(node.get('exposure_time', 0)) if node.get('exposure_time') else None,
            # Preserve relationship fields for relationship extraction
            'machine_id': node.get('machine_id'),
            'build_id': node.get('build_id'),
            'part_id': node.get('part_id'),
            'source': source
        }
    
    def _map_machine_fields(self, node: Dict[str, Any], source: str) -> Dict[str, Any]:
        """Map raw data to Machine schema fields."""
        # Map status to valid enum values
        status = node.get('status', '').lower()
        if status in ['operational', 'maintenance', 'idle', 'error', 'offline', 'calibrating', 'active']:
            mapped_status = status
        else:
            mapped_status = 'operational'  # Default to operational status
        
        # Parse installation date with fallback
        installation_date = self._parse_date(node.get('installation_date') or node.get('created_at'))
        if not installation_date:
            from datetime import date
            installation_date = date.today()  # Fallback to today's date
        
        return {
            'machine_id': node.get('id') or node.get('machine_id') or f"MACHINE_{uuid.uuid4().hex[:8]}",
            'machine_type': node.get('machine_type') or node.get('type') or 'Unknown',
            'model': node.get('model') or node.get('machine_model') or 'Unknown',
            'status': mapped_status,
            'location': node.get('location') or node.get('machine_location') or 'Unknown',
            'installation_date': installation_date,
            'max_build_volume': self._parse_dimensions(node.get('max_build_volume')),
            'laser_power_max': float(node.get('laser_power_max', 0)) if node.get('laser_power_max') else None,
            'layer_thickness_range': node.get('layer_thickness_range'),
            'accuracy': float(node.get('accuracy', 0)) if node.get('accuracy') else None,
            'maintenance_date': self._parse_date(node.get('maintenance_date')),
            'utilization_rate': float(node.get('utilization_rate', 0)) if node.get('utilization_rate') else None,
            'source': source
        }
    
    def _validate_with_pydantic(self, schema_node: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Validate schema-mapped node with Pydantic model.
        
        Args:
            schema_node: Schema-mapped node data
            
        Returns:
            Optional[Dict[str, Any]]: Validated node or None if validation fails
        """
        node_type = schema_node.get('node_type', 'Unknown')
        
        try:
            # Convert MongoDB node types to Neo4j model types
            model_type = self._convert_to_neo4j_model_type(node_type)
            
            
            # Get the appropriate Pydantic model
            model_class = self.factory.get_model_class(model_type)
            if not model_class:
                logger.warning(f"No Pydantic model found for node type: {model_type}")
                logger.warning(f"Available model types: {list(self.factory._models.keys())}")
                return None
            
            # Create and validate the model
            model_instance = model_class(**schema_node)
            
            # Store validation results
            self.validation_results.append({
                'node_id': schema_node.get('process_id') or schema_node.get('machine_id') or schema_node.get('part_id') or schema_node.get('sensor_id'),
                'node_type': model_type,
                'valid': True,
                'warnings': []
            })
            
            # Preserve the node_type in the validated node
            validated_node = model_instance.model_dump()
            validated_node['node_type'] = model_type
            return validated_node
                
        except Exception as e:
            logger.error(f"Pydantic validation error for {model_type}: {e}")
            return None
    
    def _save_to_kg_neo4j(self, validated_node: Dict[str, Any], category: str):
        """
        Save validated node to kg_neo4j data lake structure.
        
        Args:
            validated_node: Validated node data
            category: Category (nodes, relationships)
        """
        node_type = validated_node.get('node_type', 'Unknown')
        node_id = (validated_node.get('process_id') or 
                  validated_node.get('machine_id') or 
                  validated_node.get('part_id') or 
                  validated_node.get('build_id') or 
                  f"{node_type}_{uuid.uuid4().hex[:8]}")
        
        # Save to appropriate JSON file
        output_file = self.output_dir / category / f"{node_type.lower()}.json"
        
        # Load existing data or create new list
        if output_file.exists():
            with open(output_file, 'r') as f:
                data = json.load(f)
        else:
            data = []
        
        # Add new node
        data.append(validated_node)
        
        # Save back to file
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def _parse_timestamp(self, timestamp_str: Any) -> datetime:
        """Parse timestamp string to datetime object."""
        if isinstance(timestamp_str, datetime):
            return timestamp_str
        if isinstance(timestamp_str, str):
            try:
                return datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            except:
                return datetime.utcnow()
        return datetime.utcnow()
    
    def _parse_date(self, date_str: Any) -> Optional[date]:
        """Parse date string to date object."""
        if isinstance(date_str, date):
            return date_str
        if isinstance(date_str, str):
            try:
                return datetime.fromisoformat(date_str).date()
            except:
                return None
        return None
    
    def _parse_dimensions(self, dimensions: Any) -> Optional[Dict[str, float]]:
        """Parse dimensions data."""
        if isinstance(dimensions, dict):
            # Map common dimension field names to x, y, z format
            if 'length' in dimensions and 'width' in dimensions and 'height' in dimensions:
                return {
                    'x': float(dimensions.get('length', 0)),
                    'y': float(dimensions.get('width', 0)),
                    'z': float(dimensions.get('height', 0))
                }
            elif 'x' in dimensions and 'y' in dimensions and 'z' in dimensions:
                return dimensions
            else:
                # Try to map other common patterns
                return {
                    'x': float(dimensions.get('x', dimensions.get('length', 0))),
                    'y': float(dimensions.get('y', dimensions.get('width', 0))),
                    'z': float(dimensions.get('z', dimensions.get('height', 0)))
                }
        if isinstance(dimensions, str):
            try:
                parsed = safe_json_loads_with_fallback(dimensions, "dimensions", 5000, {})
                
                if isinstance(parsed, dict):
                    # Apply the same mapping logic
                    if 'length' in parsed and 'width' in parsed and 'height' in parsed:
                        return {
                            'x': float(parsed.get('length', 0)),
                            'y': float(parsed.get('width', 0)),
                            'z': float(parsed.get('height', 0))
                        }
                    return parsed
                return None
            except Exception as e:
                logger.warning(f"Unexpected error parsing dimensions: {e}")
                return None
        return None
    
    def _map_part_fields(self, node: Dict[str, Any], source: str) -> Dict[str, Any]:
        """Map raw data to Part schema fields."""
        # Map status to valid enum values
        status = node.get('status', '').lower()
        if status in ['pending', 'in_progress', 'completed', 'failed', 'inspected', 'rejected', 'unknown']:
            mapped_status = status
        else:
            mapped_status = 'unknown'  # Default to unknown status
        
        return {
            'part_id': node.get('id') or node.get('part_id') or f"PART_{uuid.uuid4().hex[:8]}",
            'part_type': node.get('part_type') or node.get('type') or 'Unknown',
            'material_type': node.get('material_type') or node.get('material') or 'Unknown',
            'dimensions': self._parse_dimensions(node.get('dimensions')),
            'volume': float(node.get('volume', 0)) if node.get('volume') else None,
            'surface_area': float(node.get('surface_area', 0)) if node.get('surface_area') else None,
            'weight': float(node.get('weight', 0)) if node.get('weight') else None,
            'status': mapped_status,
            'quality_grade': node.get('quality_grade') or node.get('grade'),
            'tolerance': float(node.get('tolerance', 0)) if node.get('tolerance') else None,
            'finish_quality': node.get('finish_quality'),
            'source': source
        }
    
    def _map_build_fields(self, node: Dict[str, Any], source: str) -> Dict[str, Any]:
        """Map raw data to Build schema fields."""
        # Map status to valid enum values - handle both 'status' and 'build_status' fields
        status = node.get('status') or node.get('build_status') or 'unknown'
        if status.lower() in ['planning', 'in_progress', 'completed', 'failed', 'cancelled', 'paused', 'unknown']:
            mapped_status = status.lower()
        else:
            mapped_status = 'unknown'  # Default to unknown status
        
        # Parse dates with fallback
        created_date = self._parse_date(node.get('created_date') or node.get('created_at'))
        if not created_date:
            from datetime import date
            created_date = date.today()  # Fallback to today's date
        
        return {
            'build_id': node.get('id') or node.get('build_id') or f"BUILD_{uuid.uuid4().hex[:8]}",
            'build_name': node.get('build_name') or node.get('name') or 'Unknown',
            'status': mapped_status,
            'created_date': created_date,
            'completed_date': self._parse_date(node.get('completed_date')),
            'total_parts': int(node.get('total_parts', 0)) if node.get('total_parts') else None,
            'success_rate': float(node.get('success_rate', 0)) if node.get('success_rate') else None,
            'total_duration': int(node.get('total_duration', 0)) if node.get('total_duration') else None,
            'material_usage': float(node.get('material_usage', 0)) if node.get('material_usage') else None,
            'energy_consumption': float(node.get('energy_consumption', 0)) if node.get('energy_consumption') else None,
            'quality_grade': node.get('quality_grade') or node.get('grade'),
            'source': source
        }
    
    def _map_material_fields(self, node: Dict[str, Any], source: str) -> Dict[str, Any]:
        """Map raw data to Material schema fields."""
        return {
            'material_type': node.get('material_type') or node.get('type') or 'Unknown',
            'properties': self._parse_material_properties(node.get('properties')),
            'supplier': node.get('supplier'),
            'certification': node.get('certification'),
            'batch_number': node.get('batch_number'),
            'condition': node.get('condition'),
            'storage_temperature': float(node.get('storage_temperature', 0)) if node.get('storage_temperature') else None,
            'humidity': float(node.get('humidity', 0)) if node.get('humidity') else None,
            'shelf_life': int(node.get('shelf_life', 0)) if node.get('shelf_life') else None,
            'source': source
        }
    
    def _map_quality_fields(self, node: Dict[str, Any], source: str) -> Dict[str, Any]:
        """Map raw data to Quality schema fields."""
        # Map quality grades to valid enum values (A-F)
        grade = node.get('grade') or node.get('quality_grade') or 'Unknown'
        if grade.upper() == 'POOR':
            grade = 'F'  # Map POOR to F grade
        elif grade.upper() not in ['A', 'B', 'C', 'D', 'F']:
            grade = 'F'  # Default to F for invalid grades
        
        return {
            'grade': grade,
            'metrics': self._parse_quality_metrics(node.get('metrics')),
            'standards': node.get('standards', []),
            'inspector': node.get('inspector'),
            'inspection_date': self._parse_date(node.get('inspection_date')),
            'test_method': node.get('test_method'),
            'confidence_level': float(node.get('confidence_level', 0)) if node.get('confidence_level') else None,
            'source': source
        }
    
    def _map_sensor_fields(self, node: Dict[str, Any], source: str) -> Dict[str, Any]:
        """Map raw data to Sensor schema fields."""
        # Map status to valid enum values - handle both 'status' and 'sensor_status' fields
        status = node.get('status') or node.get('sensor_status') or 'unknown'
        if status.lower() in ['active', 'inactive', 'calibrating', 'error', 'maintenance', 'unknown']:
            mapped_status = status.lower()
        else:
            mapped_status = 'unknown'  # Default to unknown status
        
        return {
            'sensor_id': node.get('id') or node.get('sensor_id') or f"SENSOR_{uuid.uuid4().hex[:8]}",
            'sensor_type': node.get('sensor_type') or node.get('type') or 'Unknown',
            'location': node.get('location') or node.get('sensor_location') or 'Unknown',
            'model': node.get('model'),
            'calibration_date': self._parse_date(node.get('calibration_date')),
            'accuracy': float(node.get('accuracy', 0)) if node.get('accuracy') else None,
            'range': self._parse_sensor_range(node.get('range') or node.get('measurement_range')),
            'sampling_rate': float(node.get('sampling_rate', 0)) if node.get('sampling_rate') else None,
            'status': mapped_status,
            'last_reading': float(node.get('last_reading', 0)) if node.get('last_reading') else None,
            'source': source
        }
    
    def _map_user_fields(self, node: Dict[str, Any], source: str) -> Dict[str, Any]:
        """Map raw data to User schema fields."""
        return {
            'user_id': node.get('id') or node.get('user_id') or f"USER_{uuid.uuid4().hex[:8]}",
            'username': node.get('username') or node.get('name') or 'Unknown',
            'name': node.get('name') or node.get('full_name') or 'Unknown',
            'role': node.get('role') or 'operator',
            'department': node.get('department'),
            'email': node.get('email'),
            'phone': node.get('phone'),
            'active': bool(node.get('active', True)),
            'last_login': self._parse_timestamp(node.get('last_login')),
            'hire_date': self._parse_date(node.get('hire_date')),
            'source': source
        }
    
    def _map_operator_fields(self, node: Dict[str, Any], source: str) -> Dict[str, Any]:
        """Map raw data to Operator schema fields."""
        return {
            'operator_id': node.get('id') or node.get('operator_id') or f"OPERATOR_{uuid.uuid4().hex[:8]}",
            'name': node.get('name') or 'Unknown',
            'certification': node.get('certification') or 'Unknown',
            'experience_years': int(node.get('experience_years', 0)) if node.get('experience_years') else 0,
            'shift': node.get('shift') or 'Unknown',
            'machine_authorization': node.get('machine_authorization', []),
            'training_completed': node.get('training_completed', []),
            'performance_rating': float(node.get('performance_rating', 0)) if node.get('performance_rating') else None,
            'source': source
        }
    
    def _map_alert_fields(self, node: Dict[str, Any], source: str) -> Dict[str, Any]:
        """Map raw data to Alert schema fields."""
        return {
            'alert_id': node.get('id') or node.get('alert_id') or f"ALERT_{uuid.uuid4().hex[:8]}",
            'severity': node.get('severity') or 'info',
            'status': node.get('status') or 'active',
            'timestamp': self._parse_timestamp(node.get('timestamp') or node.get('created_at')),
            'message': node.get('message') or 'Unknown',
            'threshold': float(node.get('threshold', 0)) if node.get('threshold') else None,
            'actual_value': float(node.get('actual_value', 0)) if node.get('actual_value') else None,
            'resolution_time': int(node.get('resolution_time', 0)) if node.get('resolution_time') else None,
            'resolved_by': node.get('resolved_by'),
            'resolution_notes': node.get('resolution_notes'),
            'source': source
        }
    
    def _map_defect_fields(self, node: Dict[str, Any], source: str) -> Dict[str, Any]:
        """Map raw data to Defect schema fields."""
        return {
            'defect_id': node.get('id') or node.get('defect_id') or f"DEFECT_{uuid.uuid4().hex[:8]}",
            'defect_type': node.get('defect_type') or node.get('type') or 'Unknown',
            'severity': node.get('severity') or 'minor',
            'location': self._parse_location(node.get('location')),
            'size': float(node.get('size', 0)) if node.get('size') else None,
            'detection_method': node.get('detection_method'),
            'confidence': float(node.get('confidence', 0)) if node.get('confidence') else None,
            'status': node.get('status') or 'detected',
            'timestamp': self._parse_timestamp(node.get('timestamp')),
            'impact': node.get('impact'),
            'source': source
        }
    
    def _map_image_fields(self, node: Dict[str, Any], source: str) -> Dict[str, Any]:
        """Map raw data to Image schema fields."""
        return {
            'image_id': node.get('id') or node.get('image_id') or f"IMAGE_{uuid.uuid4().hex[:8]}",
            'image_type': node.get('image_type') or node.get('type') or 'process_monitoring',
            'format': node.get('format') or 'jpg',
            'resolution': self._parse_resolution(node.get('resolution')),
            'file_size': int(node.get('file_size', 0)) if node.get('file_size') else None,
            'timestamp': self._parse_timestamp(node.get('timestamp')),
            'camera_position': self._parse_location(node.get('camera_position')),
            'lighting_conditions': node.get('lighting_conditions'),
            'quality_score': float(node.get('quality_score', 0)) if node.get('quality_score') else None,
            'file_path': node.get('file_path'),
            'source': source
        }
    
    def _map_log_fields(self, node: Dict[str, Any], source: str) -> Dict[str, Any]:
        """Map raw data to Log schema fields."""
        return {
            'log_id': node.get('id') or node.get('log_id') or f"LOG_{uuid.uuid4().hex[:8]}",
            'level': node.get('level') or 'info',
            'source': node.get('source') or 'Unknown',
            'message': node.get('message') or 'Unknown',
            'timestamp': self._parse_timestamp(node.get('timestamp') or node.get('created_at')),
            'component': node.get('component'),
            'session_id': node.get('session_id'),
            'user_id': node.get('user_id'),
            'source': source
        }
    
    def _map_inspection_fields(self, node: Dict[str, Any], source: str) -> Dict[str, Any]:
        """Map raw data to Inspection schema fields."""
        return {
            'inspection_id': node.get('id') or node.get('inspection_id') or f"INSPECTION_{uuid.uuid4().hex[:8]}",
            'inspector': node.get('inspector') or 'Unknown',
            'inspection_date': self._parse_date(node.get('inspection_date')),
            'inspection_type': node.get('inspection_type') or node.get('type') or 'Unknown',
            'result': node.get('result') or 'Unknown',
            'notes': node.get('notes'),
            'standards': node.get('standards', []),
            'source': source
        }
    
    def _parse_material_properties(self, properties: Any) -> Optional[Dict[str, Any]]:
        """Parse material properties."""
        if isinstance(properties, dict):
            return properties
        if isinstance(properties, str):
            try:
                return safe_json_loads_with_fallback(properties, "material_properties", 5000, {})
            except Exception as e:
                logger.warning(f"Unexpected error parsing material properties: {e}")
                return None
        return None
    
    def _parse_quality_metrics(self, metrics: Any) -> Optional[Dict[str, Any]]:
        """Parse quality metrics."""
        if isinstance(metrics, dict):
            return metrics
        if isinstance(metrics, str):
            try:
                return safe_json_loads_with_fallback(metrics, "quality_metrics", 5000, {})
            except Exception as e:
                logger.warning(f"Unexpected error parsing quality metrics: {e}")
                return None
        return None
    
    def _parse_sensor_range(self, range_data: Any) -> Optional[Dict[str, float]]:
        """Parse sensor range data."""
        if isinstance(range_data, dict):
            return range_data
        if isinstance(range_data, str):
            # Handle simple range strings like "0-1000"
            if '-' in range_data and len(range_data) < 100:  # Simple range format
                try:
                    parts = range_data.split('-')
                    if len(parts) == 2:
                        min_val = float(parts[0].strip())
                        max_val = float(parts[1].strip())
                        return {"min": min_val, "max": max_val}
                except (ValueError, IndexError):
                    pass
            
            # Handle JSON strings
            try:
                return safe_json_loads_with_fallback(range_data, "sensor_range", 5000, {})
            except Exception as e:
                logger.warning(f"Unexpected error parsing sensor range: {e}")
                return None
        return None
    
    def _parse_location(self, location: Any) -> Optional[Dict[str, float]]:
        """Parse location data."""
        if isinstance(location, dict):
            return location
        if isinstance(location, str):
            try:
                return safe_json_loads_with_fallback(location, "location", 5000, {})
            except Exception as e:
                logger.warning(f"Unexpected error parsing location: {e}")
                return None
        return None
    
    def _parse_resolution(self, resolution: Any) -> Optional[Dict[str, int]]:
        """Parse image resolution data."""
        if isinstance(resolution, dict):
            return resolution
        if isinstance(resolution, str):
            try:
                return safe_json_loads_with_fallback(resolution, "resolution", 5000, {})
            except Exception as e:
                logger.warning(f"Unexpected error parsing resolution: {e}")
                return None
        return None
    
    def _generate_graph_id(self, node_type: str, node_id: str, source: str) -> str:
        """Generate a unique graph ID for the node."""
        # Use a combination of type, ID, and source for uniqueness
        base_id = f"{node_type}_{node_id}_{source}"
        return f"KG_{hash(base_id) % 100000000:08d}"
    
    def _normalize_properties(self, node: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize node properties to standard format."""
        properties = {}
        
        for key, value in node.items():
            if key in ['node_type', 'id', '_id', 'node_id']:
                continue
                
            # Normalize key names
            normalized_key = self._normalize_key(key)
            
            # Normalize values
            normalized_value = self._normalize_value(value)
            
            if normalized_value is not None:
                properties[normalized_key] = normalized_value
        
        return properties
    
    def _normalize_key(self, key: str) -> str:
        """Normalize property key names."""
        # Convert to lowercase and replace special characters
        normalized = key.lower().replace(' ', '_').replace('-', '_')
        
        # Remove common prefixes
        if normalized.startswith('node_'):
            normalized = normalized[5:]
        elif normalized.startswith('data_'):
            normalized = normalized[5:]
            
        return normalized
    
    def _normalize_value(self, value: Any) -> Any:
        """Normalize property values."""
        if value is None:
            return None
        elif isinstance(value, (str, int, float, bool)):
            return value
        elif isinstance(value, datetime):
            return value.isoformat()
        elif isinstance(value, (list, dict)):
            return json.dumps(value) if value else None
        else:
            return str(value)
    
    def _generate_labels(self, node_type: str, properties: Dict[str, Any]) -> List[str]:
        """Generate Neo4j labels for the node."""
        labels = [node_type]
        
        # Add source-specific labels
        if 'process' in node_type.lower():
            labels.append('Process')
        elif 'machine' in node_type.lower():
            labels.append('Machine')
        elif 'sensor' in node_type.lower():
            labels.append('Sensor')
        elif 'part' in node_type.lower():
            labels.append('Part')
        elif 'build' in node_type.lower():
            labels.append('Build')
        elif 'image' in node_type.lower():
            labels.append('Image')
        elif 'log' in node_type.lower():
            labels.append('Log')
        
        # Add status-based labels
        status = properties.get('status', '').lower()
        if status in ['active', 'running', 'operational']:
            labels.append('Active')
        elif status in ['inactive', 'stopped', 'offline']:
            labels.append('Inactive')
        elif status in ['error', 'failed', 'fault']:
            labels.append('Error')
        
        return list(set(labels))  # Remove duplicates
    
    def _is_duplicate(self, node: Dict[str, Any]) -> bool:
        """Check if a node is a duplicate of an existing one."""
        node_type = node['node_type']
        source_id = node['source_id']
        
        # Check if we already have a node with the same type and source ID
        for existing_node in self.processed_nodes.values():
            if (existing_node['node_type'] == node_type and 
                existing_node['source_id'] == source_id):
                return True
        
        return False
    
    def get_processed_nodes(self) -> Dict[str, Dict[str, Any]]:
        """Get all processed nodes."""
        return self.processed_nodes
    
    def get_node_id_mapping(self) -> Dict[str, str]:
        """Get the mapping from source IDs to graph IDs."""
        return self.node_id_mapping
    
    def get_duplicate_nodes(self) -> List[Dict[str, Any]]:
        """Get all duplicate nodes that were filtered out."""
        return self.duplicate_nodes
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics."""
        node_types = {}
        sources = {}
        
        for node in self.processed_nodes.values():
            node_type = node['node_type']
            source = node['source']
            
            node_types[node_type] = node_types.get(node_type, 0) + 1
            sources[source] = sources.get(source, 0) + 1
        
        return {
            'total_nodes': len(self.processed_nodes),
            'duplicate_nodes': len(self.duplicate_nodes),
            'node_types': node_types,
            'sources': sources,
            'unique_graph_ids': len(set(self.processed_nodes.keys()))
        }
    
    # New node type mapping methods
    def _map_process_image_fields(self, node: Dict[str, Any], source: str) -> Dict[str, Any]:
        """Map raw data to ProcessImage schema fields."""
        return {
            'image_id': node.get('document_id') or node.get('id') or f"IMG_{uuid.uuid4().hex[:8]}",
            'process_id': node.get('process_id') or 'Unknown',
            'image_type': node.get('image_type') or 'process',
            'file_path': f"/data/images/{node.get('document_id', uuid.uuid4().hex[:8])}.{node.get('image_format', 'jpg')}",
            'file_size': int(node.get('file_size', 0)) if node.get('file_size') else 0,
            'dimensions': node.get('image_dimensions') or {'width': 1920, 'height': 1080},
            'format': node.get('image_format') or 'jpg',
            'resolution': 72.0,  # Default resolution since not in MongoDB
            'timestamp': self._parse_timestamp(node.get('created_at')),
            'metadata': node.get('metadata') or {},
            'source': source
        }
    
    def _map_ct_scan_image_fields(self, node: Dict[str, Any], source: str) -> Dict[str, Any]:
        """Map raw data to CTScanImage schema fields."""
        return {
            'scan_id': node.get('document_id') or node.get('id') or f"CT_{uuid.uuid4().hex[:8]}",
            'part_id': node.get('part_id') or 'Unknown',
            'scan_type': node.get('scan_type') or 'ct_scan',
            'file_path': f"/data/scans/{node.get('document_id', uuid.uuid4().hex[:8])}.{node.get('image_format', 'dcm')}",
            'file_size': int(node.get('file_size', 0)) if node.get('file_size') else 0,
            'voxel_size': {'x': 0.5, 'y': 0.5, 'z': 0.5},  # Default voxel size
            'scan_resolution': {'x': 512, 'y': 512, 'z': 100},  # Default resolution
            'timestamp': self._parse_timestamp(node.get('created_at')),
            'quality_score': None,  # Not available in MongoDB
            'metadata': node.get('metadata') or {},
            'source': source
        }
    
    def _map_powder_bed_image_fields(self, node: Dict[str, Any], source: str) -> Dict[str, Any]:
        """Map raw data to PowderBedImage schema fields."""
        return {
            'image_id': node.get('document_id') or node.get('id') or f"PBI_{uuid.uuid4().hex[:8]}",
            'build_id': node.get('process_id') or 'Unknown',  # MongoDB uses process_id, not build_id
            'layer_number': int(node.get('layer_number', 0)) if node.get('layer_number') else 0,
            'image_type': node.get('image_type') or 'powder_bed',
            'file_path': f"/data/powder_bed/{node.get('document_id', uuid.uuid4().hex[:8])}.{node.get('image_format', 'jpg')}",
            'file_size': int(node.get('file_size', 0)) if node.get('file_size') else 0,
            'dimensions': node.get('image_dimensions') or {'width': 1920, 'height': 1080},
            'timestamp': self._parse_timestamp(node.get('created_at')),
            'powder_density': None,  # Not available in MongoDB
            'defects_detected': [],  # Not available in MongoDB
            'metadata': node.get('metadata') or {},
            'source': source
        }
    
    def _map_build_file_fields(self, node: Dict[str, Any], source: str) -> Dict[str, Any]:
        """Map raw data to BuildFile schema fields."""
        return {
            'file_id': node.get('id') or node.get('file_id') or f"BF_{uuid.uuid4().hex[:8]}",
            'build_id': node.get('build_id') or node.get('build') or 'Unknown',
            'file_type': node.get('file_type') or node.get('type') or 'build_file',
            'file_path': node.get('file_path') or node.get('path') or f"/data/builds/{uuid.uuid4().hex[:8]}.slm",
            'file_size': int(node.get('file_size', 0)) if node.get('file_size') else 0,
            'format': node.get('format') or node.get('file_format') or 'slm',
            'version': node.get('version') or '1.0',
            'checksum': node.get('checksum') or node.get('hash') or f"hash_{uuid.uuid4().hex}",
            'file_created_at': self._parse_timestamp(node.get('timestamp') or node.get('created_at')),
            'metadata': node.get('metadata') or {},
            'source': source
        }
    
    def _map_model_file_fields(self, node: Dict[str, Any], source: str) -> Dict[str, Any]:
        """Map raw data to ModelFile schema fields."""
        return {
            'model_id': node.get('id') or node.get('model_id') or f"MF_{uuid.uuid4().hex[:8]}",
            'part_id': node.get('part_id') or node.get('part') or 'Unknown',
            'file_type': node.get('file_type') or node.get('type') or 'model_file',
            'file_path': node.get('file_path') or node.get('path') or f"/data/models/{uuid.uuid4().hex[:8]}.stl",
            'file_size': int(node.get('file_size', 0)) if node.get('file_size') else 0,
            'format': node.get('format') or node.get('file_format') or 'stl',
            'version': node.get('version') or '1.0',
            'dimensions': node.get('dimensions') or {'x': 100.0, 'y': 100.0, 'z': 50.0},
            'volume': node.get('volume') or 1000000.0,
            'surface_area': node.get('surface_area') or 60000.0,
            'complexity_score': node.get('complexity_score'),
            'model_created_at': self._parse_timestamp(node.get('timestamp') or node.get('created_at')),
            'metadata': node.get('metadata') or {},
            'source': source
        }
    
    def _map_log_file_fields(self, node: Dict[str, Any], source: str) -> Dict[str, Any]:
        """Map raw data to LogFile schema fields."""
        start_time = self._parse_timestamp(node.get('start_time') or node.get('timestamp') or node.get('created_at'))
        end_time = self._parse_timestamp(node.get('end_time') or node.get('timestamp') or node.get('created_at'))
        
        # Calculate duration if both times are available
        duration = 0.0
        if start_time and end_time:
            duration = (end_time - start_time).total_seconds()
        
        return {
            'log_id': node.get('id') or node.get('log_id') or f"LF_{uuid.uuid4().hex[:8]}",
            'process_id': node.get('process_id') or node.get('process') or 'Unknown',
            'log_type': node.get('log_type') or node.get('type') or 'process_log',
            'file_path': node.get('file_path') or node.get('path') or f"/data/logs/{uuid.uuid4().hex[:8]}.log",
            'file_size': int(node.get('file_size', 0)) if node.get('file_size') else 0,
            'format': node.get('format') or node.get('file_format') or 'log',
            'level': node.get('log_level') or node.get('level') or 'info',
            'entries_count': int(node.get('entries_count', 1)) if node.get('entries_count') else 1,
            'start_time': start_time,
            'end_time': end_time,
            'duration': duration,
            'metadata': node.get('metadata') or {},
            'source': source
        }
    
    # Cassandra and Redis mapping methods
    def _map_sensor_reading_fields(self, node: Dict[str, Any], source: str) -> Dict[str, Any]:
        """Map raw data to SensorReading schema fields."""
        return {
            'reading_id': node.get('id') or node.get('reading_id') or f"SR_{uuid.uuid4().hex[:8]}",
            'sensor_id': node.get('sensor_id') or node.get('sensor') or 'Unknown',
            'reading_type': node.get('reading_type') or node.get('type') or 'sensor_reading',
            'value': float(node.get('value', 0)) if node.get('value') else 0.0,
            'unit': node.get('unit') or node.get('measurement_unit') or 'unknown',
            'timestamp': self._parse_timestamp(node.get('timestamp') or node.get('created_at')),
            'quality': float(node.get('quality_score', 0)) / 100.0 if node.get('quality_score') else None,
            'status': node.get('status') or node.get('state') or 'active',
            'location': node.get('location') or node.get('coordinates'),
            'metadata': node.get('metadata') or {},
            'source': source
        }
    
    def _map_process_monitoring_fields(self, node: Dict[str, Any], source: str) -> Dict[str, Any]:
        """Map raw data to ProcessMonitoring schema fields."""
        return {
            'event_id': node.get('id') or node.get('event_id') or f"PM_{uuid.uuid4().hex[:8]}",
            'process_id': node.get('process_id') or node.get('process') or 'Unknown',
            'event_type': node.get('event_type') or node.get('type') or 'monitoring',
            'severity': node.get('severity') or node.get('level') or 'info',
            'message': node.get('message') or node.get('description') or 'Process monitoring event',
            'timestamp': self._parse_timestamp(node.get('timestamp') or node.get('created_at')),
            'parameters': node.get('parameters') or node.get('data'),
            'status': node.get('status') or node.get('state') or 'active',
            'resolved': bool(node.get('resolved', False)),
            'metadata': node.get('metadata') or {},
            'source': source
        }
    
    def _map_machine_status_fields(self, node: Dict[str, Any], source: str) -> Dict[str, Any]:
        """Map raw data to MachineStatus schema fields."""
        return {
            'status_id': node.get('id') or node.get('status_id') or f"MS_{uuid.uuid4().hex[:8]}",
            'machine_id': node.get('machine_id') or node.get('machine') or 'Unknown',
            'status_type': node.get('status_type') or node.get('type') or 'status_update',
            'status_value': node.get('status') or node.get('state') or 'unknown',
            'timestamp': self._parse_timestamp(node.get('timestamp') or node.get('created_at')),
            'duration': float(node.get('duration', 0)) if node.get('duration') else None,
            'reason': node.get('reason') or node.get('description'),
            'operator_id': node.get('operator_id') or node.get('operator'),
            'metadata': node.get('metadata') or {},
            'source': source
        }
    
    def _map_alert_event_fields(self, node: Dict[str, Any], source: str) -> Dict[str, Any]:
        """Map raw data to AlertEvent schema fields."""
        return {
            'alert_id': node.get('id') or node.get('alert_id') or f"AE_{uuid.uuid4().hex[:8]}",
            'source_id': node.get('process_id') or node.get('process') or 'Unknown',
            'alert_type': node.get('alert_type') or node.get('type') or 'warning',
            'severity': node.get('severity') or node.get('level') or 'medium',
            'message': node.get('message') or node.get('description') or 'Alert event',
            'timestamp': self._parse_timestamp(node.get('timestamp') or node.get('created_at')),
            'status': node.get('status') or node.get('state') or 'active',
            'acknowledged': bool(node.get('acknowledged', False)),
            'acknowledged_by': node.get('acknowledged_by') or node.get('acknowledged_by_user'),
            'acknowledged_at': self._parse_timestamp(node.get('acknowledged_at')),
            'resolved': bool(node.get('resolved', False)),
            'resolved_by': node.get('resolved_by') or node.get('resolved_by_user'),
            'resolved_at': self._parse_timestamp(node.get('resolved_at')),
            'metadata': node.get('metadata') or {},
            'source': source
        }
    
    def _map_process_cache_fields(self, node: Dict[str, Any], source: str) -> Dict[str, Any]:
        """Map raw data to ProcessCache schema fields."""
        return {
            'cache_id': node.get('id') or node.get('cache_id') or f"PC_{uuid.uuid4().hex[:8]}",
            'process_id': node.get('process_id') or node.get('process') or 'Unknown',
            'cache_type': node.get('cache_type') or node.get('type') or 'process_cache',
            'key': node.get('key') or node.get('cache_key') or f"cache_{uuid.uuid4().hex[:8]}",
            'value': node.get('value') or node.get('cache_value') or 'cached_data',
            'size': int(node.get('size', 100)) if node.get('size') else 100,
            'ttl': int(node.get('ttl', 3600)) if node.get('ttl') else 3600,
            'created_at': self._parse_timestamp(node.get('created_at') or node.get('timestamp')),
            'expires_at': self._parse_timestamp(node.get('expires_at') or node.get('expiry')),
            'access_count': int(node.get('hit_count', 0)) if node.get('hit_count') else 0,
            'last_accessed': self._parse_timestamp(node.get('last_accessed')),
            'metadata': node.get('metadata') or {},
            'source': source
        }
    
    def _map_analytics_cache_fields(self, node: Dict[str, Any], source: str) -> Dict[str, Any]:
        """Map raw data to AnalyticsCache schema fields."""
        return {
            'cache_id': node.get('id') or node.get('cache_id') or f"AC_{uuid.uuid4().hex[:8]}",
            'analysis_type': node.get('analysis_type') or node.get('type') or 'analytics',
            'cache_key': node.get('key') or node.get('cache_key') or f"analytics_{uuid.uuid4().hex[:8]}",
            'result_data': node.get('value') or node.get('cache_value') or 'analytics_result',
            'size': int(node.get('size', 1000)) if node.get('size') else 1000,
            'ttl': int(node.get('ttl', 3600)) if node.get('ttl') else 3600,
            'created_at': self._parse_timestamp(node.get('created_at') or node.get('timestamp')),
            'expires_at': self._parse_timestamp(node.get('expires_at') or node.get('expiry')),
            'computation_time': float(node.get('computation_time', 1.0)) if node.get('computation_time') else 1.0,
            'accuracy': float(node.get('accuracy', 0.95)) if node.get('accuracy') else None,
            'metadata': node.get('metadata') or {},
            'source': source
        }
    
    def _map_job_queue_fields(self, node: Dict[str, Any], source: str) -> Dict[str, Any]:
        """Map raw data to JobQueue schema fields."""
        return {
            'job_id': node.get('id') or node.get('job_id') or f"JQ_{uuid.uuid4().hex[:8]}",
            'queue_name': node.get('queue_name') or node.get('queue') or 'default',
            'job_type': node.get('job_type') or node.get('type') or 'processing',
            'priority': int(node.get('priority', 5)) if node.get('priority') else 5,
            'status': node.get('status') or node.get('state') or 'pending',
            'payload': node.get('payload') or node.get('data') or 'job_data',
            'created_at': self._parse_timestamp(node.get('created_at') or node.get('timestamp')),
            'started_at': self._parse_timestamp(node.get('started_at')),
            'completed_at': self._parse_timestamp(node.get('completed_at')),
            'retry_count': int(node.get('retry_count', 0)) if node.get('retry_count') else 0,
            'max_retries': int(node.get('max_retries', 3)) if node.get('max_retries') else 3,
            'timeout': int(node.get('timeout', 3600)) if node.get('timeout') else 3600,
            # Preserve relationship fields for relationship extraction
            'process_id': node.get('process_id'),
            'machine_id': node.get('machine_id'),
            'user_id': node.get('user_id'),
            'metadata': node.get('metadata') or {},
            'source': source
        }
    
    def _map_user_session_fields(self, node: Dict[str, Any], source: str) -> Dict[str, Any]:
        """Map raw data to UserSession schema fields."""
        return {
            'session_id': node.get('id') or node.get('session_id') or f"US_{uuid.uuid4().hex[:8]}",
            'user_id': node.get('user_id') or node.get('user') or 'Unknown',
            'session_type': node.get('session_type') or node.get('type') or 'web_session',
            'status': node.get('status') or node.get('state') or 'active',
            'created_at': self._parse_timestamp(node.get('created_at') or node.get('timestamp')),
            'last_activity': self._parse_timestamp(node.get('last_activity') or node.get('last_accessed')),
            'expires_at': self._parse_timestamp(node.get('expires_at') or node.get('expiry')),
            'ip_address': node.get('ip_address') or node.get('ip') or '0.0.0.0',
            'user_agent': node.get('user_agent') or node.get('agent') or '',
            'permissions': node.get('permissions') or node.get('roles'),
            'metadata': node.get('metadata') or {},
            'source': source
        }
    
    def _map_machine_config_fields(self, node: Dict[str, Any], source: str) -> Dict[str, Any]:
        """Map raw data to MachineConfig schema fields."""
        return {
            'config_id': node.get('document_id') or node.get('id') or f"MC_{uuid.uuid4().hex[:8]}",
            'machine_id': node.get('machine_id') or 'Unknown',
            'process_id': node.get('process_id'),
            'build_id': node.get('build_id'),
            'config_type': node.get('config_type') or 'machine_config',
            'config_data': node.get('config_data') or {},
            'file_size': node.get('file_size'),
            'config_created_at': self._parse_timestamp(node.get('created_at')),
            'metadata': node.get('metadata') or {},
            'source': source
        }
    
    def _map_sensor_type_fields(self, node: Dict[str, Any], source: str) -> Dict[str, Any]:
        """Map raw data to SensorType schema fields."""
        return {
            'sensor_type_id': node.get('id') or node.get('sensor_type_id') or f"ST_{uuid.uuid4().hex[:8]}",
            'sensor_type': node.get('sensor_type') or node.get('type') or 'Unknown',
            'description': node.get('description') or 'Sensor type description',
            'unit': node.get('unit') or node.get('measurement_unit') or 'unknown',
            'range_min': float(node.get('range_min', 0)) if node.get('range_min') else None,
            'range_max': float(node.get('range_max', 100)) if node.get('range_max') else None,
            'accuracy': float(node.get('accuracy', 0.1)) if node.get('accuracy') else None,
            'sampling_rate': float(node.get('sampling_rate', 1.0)) if node.get('sampling_rate') else None,
            'calibration_required': bool(node.get('calibration_required', True)),
            'metadata': node.get('metadata') or {},
            'source': source
        }
