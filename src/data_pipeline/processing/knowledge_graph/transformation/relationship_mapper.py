"""
Schema-Aware Relationship Mapper for Knowledge Graph Transformation

This module maps relationships between entities from different data sources
into schema-compliant relationship structure using Pydantic validation.
"""

import logging
from typing import Dict, List, Any, Optional, Set, Tuple
from datetime import datetime
import json
import uuid
from pathlib import Path

from src.data_pipeline.processing.knowledge_graph.utils.json_parser import safe_json_loads_with_fallback

# Import our Neo4j models and validation
from src.data_pipeline.processing.schema.models.neo4j_models import (
    Neo4jModelFactory, GraphValidationEngine,
    ProcessMachineRelationship, ProcessPartRelationship, ProcessBuildRelationship,
    ProcessMaterialRelationship, ProcessQualityRelationship, ProcessSensorRelationship,
    ProcessOperatorRelationship, ProcessAlertRelationship, ProcessDefectRelationship,
    ProcessImageRelationship, ProcessLogRelationship
)

logger = logging.getLogger(__name__)


class RelationshipMapper:
    """
    Schema-aware relationship mapper that transforms raw relationships into Neo4j schema-compliant relationships.
    
    Handles:
    - Schema field mapping from raw relationships to Neo4j schema
    - Pydantic model validation for relationship data quality
    - Manufacturing-specific business logic validation
    - Export to kg_neo4j data lake structure
    """
    
    def __init__(self, processed_nodes: Dict[str, Dict[str, Any]], output_dir: str = "data_lake/kg_neo4j"):
        """
        Initialize the schema-aware relationship mapper.
        
        Args:
            processed_nodes: Dictionary of processed nodes with graph_id as key
            output_dir: Output directory for kg_neo4j data lake
        """
        self.processed_nodes = processed_nodes
        self.factory = Neo4jModelFactory()
        self.validation_engine = GraphValidationEngine()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create output subdirectories
        (self.output_dir / "relationships").mkdir(exist_ok=True)
        
        self.relationships: List[Dict[str, Any]] = []
        self.relationship_types: Set[str] = set()
        self.validation_results: List[Dict[str, Any]] = []
        
    def map_postgresql_relationships(self, nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Map relationships from PostgreSQL nodes using schema mapping and Pydantic validation.
        
        Args:
            nodes: List of PostgreSQL nodes
            
        Returns:
            List[Dict[str, Any]]: Schema-compliant mapped relationships
        """
        relationships = []
        
        for node in nodes:
            try:
                node_relationships = self._extract_relationships_from_node(node, "postgresql")
                for rel in node_relationships:
                    # Map to schema fields
                    schema_rel = self._map_relationship_to_schema(rel)
                    if schema_rel:
                        # Validate with Pydantic model
                        validated_rel = self._validate_relationship_with_pydantic(schema_rel)
                        if validated_rel:
                            relationships.append(validated_rel)
                            self._save_relationship_to_kg_neo4j(validated_rel)
            except Exception as e:
                logger.warning(f"Failed to map PostgreSQL relationships: {e}")
                continue
        
        logger.info(f"✅ Mapped {len(relationships)} PostgreSQL relationships with schema validation")
        return relationships
    
    def map_mongodb_relationships(self, nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Map relationships from MongoDB nodes.
        
        Args:
            nodes: List of MongoDB nodes
            
        Returns:
            List[Dict[str, Any]]: Mapped relationships
        """
        relationships = []
        
        for node in nodes:
            try:
                node_relationships = self._extract_relationships_from_node(node, "mongodb")
                relationships.extend(node_relationships)
            except Exception as e:
                logger.warning(f"Failed to map MongoDB relationships: {e}")
                continue
        
        logger.info(f"✅ Mapped {len(relationships)} MongoDB relationships")
        return relationships
    
    def map_cassandra_relationships(self, nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Map relationships from Cassandra nodes.
        
        Args:
            nodes: List of Cassandra nodes
            
        Returns:
            List[Dict[str, Any]]: Mapped relationships
        """
        relationships = []
        
        for node in nodes:
            try:
                node_relationships = self._extract_relationships_from_node(node, "cassandra")
                relationships.extend(node_relationships)
            except Exception as e:
                logger.warning(f"Failed to map Cassandra relationships: {e}")
                continue
        
        logger.info(f"✅ Mapped {len(relationships)} Cassandra relationships")
        return relationships
    
    def map_redis_relationships(self, nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Map relationships from Redis nodes.
        
        Args:
            nodes: List of Redis nodes
            
        Returns:
            List[Dict[str, Any]]: Mapped relationships
        """
        relationships = []
        
        for node in nodes:
            try:
                node_relationships = self._extract_relationships_from_node(node, "redis")
                relationships.extend(node_relationships)
            except Exception as e:
                logger.warning(f"Failed to map Redis relationships: {e}")
                continue
        
        logger.info(f"✅ Mapped {len(relationships)} Redis relationships")
        return relationships
    
    def _map_relationship_to_schema(self, relationship: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Map raw relationship data to Neo4j schema fields.
        
        Args:
            relationship: Raw relationship data
            
        Returns:
            Optional[Dict[str, Any]]: Schema-mapped relationship data
        """
        rel_type = relationship.get('relationship_type', 'Unknown')
        
        # Map based on relationship type to schema fields
        if rel_type.upper() == 'USES_MACHINE':
            schema_rel = self._map_uses_machine_fields(relationship)
        elif rel_type.upper() == 'CREATES_PART':
            schema_rel = self._map_creates_part_fields(relationship)
        elif rel_type.upper() == 'USES_MATERIAL':
            schema_rel = self._map_uses_material_fields(relationship)
        elif rel_type.upper() == 'PART_OF_BUILD':
            schema_rel = self._map_part_of_build_fields(relationship)
        elif rel_type.upper() == 'HAS_QUALITY':
            schema_rel = self._map_has_quality_fields(relationship)
        elif rel_type.upper() == 'OPERATED_BY':
            schema_rel = self._map_operated_by_fields(relationship)
        elif rel_type.upper() == 'MONITORED_BY':
            schema_rel = self._map_monitored_by_fields(relationship)
        elif rel_type.upper() == 'GENERATES_ALERT':
            schema_rel = self._map_generates_alert_fields(relationship)
        elif rel_type.upper() == 'HAS_DEFECT':
            schema_rel = self._map_has_defect_fields(relationship)
        elif rel_type.upper() == 'CAPTURED_BY':
            schema_rel = self._map_captured_by_fields(relationship)
        elif rel_type.upper() == 'LOGGED_BY':
            schema_rel = self._map_logged_by_fields(relationship)
        else:
            logger.warning(f"Unknown relationship type: {rel_type}")
            return None
        
        # Add relationship_type to the schema relationship for validation
        if schema_rel:
            schema_rel['relationship_type'] = rel_type
        
        return schema_rel
    
    def _map_uses_machine_fields(self, relationship: Dict[str, Any]) -> Dict[str, Any]:
        """Map raw data to USES_MACHINE schema fields."""
        return {
            'relationship_type': 'USES_MACHINE',
            'from_id': relationship.get('from_node_id'),
            'to_id': relationship.get('to_node_id'),
            'start_time': self._parse_timestamp(relationship.get('start_time')),
            'end_time': self._parse_timestamp(relationship.get('end_time')),
            'duration': int(relationship.get('duration', 0)) if relationship.get('duration') else None,
            'utilization': float(relationship.get('efficiency', 0)) if relationship.get('efficiency') else None,
            'graph_id': f"REL_{uuid.uuid4().hex[:8]}",
            'source': relationship.get('source', 'unknown')
        }
    
    def _map_creates_part_fields(self, relationship: Dict[str, Any]) -> Dict[str, Any]:
        """Map raw data to CREATES_PART schema fields."""
        return {
            'relationship_type': 'CREATES_PART',
            'from_id': relationship.get('from_node_id'),
            'to_id': relationship.get('to_node_id'),
            'quantity': int(relationship.get('quantity', 1)) if relationship.get('quantity') else None,
            'success_rate': float(relationship.get('success_rate', 0)) if relationship.get('success_rate') else None,
            'creation_time': self._parse_timestamp(relationship.get('creation_time')),
            'quality_grade': relationship.get('quality_grade'),
            'graph_id': f"REL_{uuid.uuid4().hex[:8]}",
            'source': relationship.get('source', 'unknown')
        }
    
    def _validate_relationship_with_pydantic(self, schema_relationship: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Validate schema-mapped relationship with Pydantic model.
        
        Args:
            schema_relationship: Schema-mapped relationship data
            
        Returns:
            Optional[Dict[str, Any]]: Validated relationship or None if validation fails
        """
        rel_type = schema_relationship.get('relationship_type', 'Unknown')
        
        try:
            # Get the appropriate Pydantic model
            model_class = self.factory.get_model_class(rel_type)
            if not model_class:
                logger.warning(f"No Pydantic model found for relationship type: {rel_type}")
                return None
            
            # Create and validate the model
            model_instance = model_class(**schema_relationship)
            
            # Additional business logic validation
            validation_result = self.validation_engine.validate_relationship(
                model_instance.dict(), rel_type
            )
            
            if validation_result.valid:
                # Store validation results
                self.validation_results.append({
                    'relationship_id': f"{rel_type}_{uuid.uuid4().hex[:8]}",
                    'relationship_type': rel_type,
                    'valid': True,
                    'warnings': [w.dict() for w in validation_result.warnings]
                })
                
                return model_instance.dict()
            else:
                logger.warning(f"Validation failed for {rel_type}: {validation_result.errors}")
                return None
                
        except Exception as e:
            logger.error(f"Pydantic validation error for {rel_type}: {e}")
            return None
    
    def _save_relationship_to_kg_neo4j(self, validated_relationship: Dict[str, Any]):
        """
        Save validated relationship to kg_neo4j data lake structure.
        
        Args:
            validated_relationship: Validated relationship data
        """
        rel_type = validated_relationship.get('relationship_type', 'Unknown')
        
        # Save to appropriate JSON file
        output_file = self.output_dir / "relationships" / f"{rel_type.lower()}.json"
        
        # Load existing data or create new list
        if output_file.exists():
            with open(output_file, 'r') as f:
                data = json.load(f)
        else:
            data = []
        
        # Add new relationship
        data.append(validated_relationship)
        
        # Save back to file
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def _parse_timestamp(self, timestamp_str: Any) -> Optional[datetime]:
        """Parse timestamp string to datetime object."""
        if isinstance(timestamp_str, datetime):
            return timestamp_str
        if isinstance(timestamp_str, str):
            try:
                return datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            except:
                return None
        return None
    
    def _parse_parameters(self, parameters: Any) -> Optional[Dict[str, Any]]:
        """Parse parameters data."""
        if isinstance(parameters, dict):
            return parameters
        if isinstance(parameters, str):
            try:
                # For extremely large JSON strings, bypass parsing entirely
                if len(parameters) > 50000:  # 50KB limit - bypass entirely for massive strings
                    logger.warning(f"Massive JSON string detected ({len(parameters)} chars), bypassing parsing entirely")
                    return {"raw_data": parameters[:1000], "size": len(parameters), "bypassed": True}
                
                # For large JSON strings, be very aggressive with truncation
                if len(parameters) > 5000:  # 5KB limit - very aggressive
                    logger.warning(f"Large JSON string detected ({len(parameters)} chars), truncating to 5KB for parsing")
                    # Truncate to 5KB and try to find a reasonable end point
                    truncated = parameters[:5000]
                    
                    # Try to find a safe truncation point
                    last_brace = truncated.rfind('}')
                    last_bracket = truncated.rfind(']')
                    last_complete = max(last_brace, last_bracket)
                    
                    if last_complete > 0:
                        # Check if we're in the middle of a string
                        remaining = parameters[last_complete:]
                        if remaining.strip().startswith('"') and not remaining.strip().endswith('"'):
                            # We're in the middle of a string, find the last complete key-value pair
                            truncated = truncated[:last_complete]
                        else:
                            truncated = truncated[:last_complete + 1]
                    else:
                        # If no complete object found, just truncate and add closing brace
                        truncated = truncated.rstrip(',') + '}'
                    
                    return safe_json_loads_with_fallback(truncated, "parameters", 5000, {})
                else:
                    return safe_json_loads_with_fallback(parameters, "parameters", 5000, {})
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse parameters JSON: {e}")
                return {"raw_data": parameters[:500] if len(parameters) > 500 else parameters}
            except Exception as e:
                logger.warning(f"Unexpected error parsing parameters: {e}")
                return None
        return None
    
    def _map_uses_material_fields(self, relationship: Dict[str, Any]) -> Dict[str, Any]:
        """Map raw data to USES_MATERIAL schema fields."""
        return {
            'relationship_type': 'USES_MATERIAL',
            'from_id': relationship.get('from_node_id'),
            'to_id': relationship.get('to_node_id'),
            'quantity': float(relationship.get('quantity', 0)) if relationship.get('quantity') else None,
            'usage_time': self._parse_timestamp(relationship.get('usage_time')),
            'graph_id': f"REL_{uuid.uuid4().hex[:8]}",
            'source': relationship.get('source', 'unknown')
        }
    
    def _map_part_of_build_fields(self, relationship: Dict[str, Any]) -> Dict[str, Any]:
        """Map raw data to PART_OF_BUILD schema fields."""
        return {
            'relationship_type': 'PART_OF_BUILD',
            'from_id': relationship.get('from_node_id'),
            'to_id': relationship.get('to_node_id'),
            'position': relationship.get('position'),
            'layer_number': int(relationship.get('layer_number', 0)) if relationship.get('layer_number') else None,
            'graph_id': f"REL_{uuid.uuid4().hex[:8]}",
            'source': relationship.get('source', 'unknown')
        }
    
    def _map_has_quality_fields(self, relationship: Dict[str, Any]) -> Dict[str, Any]:
        """Map raw data to HAS_QUALITY schema fields."""
        return {
            'relationship_type': 'HAS_QUALITY',
            'from_id': relationship.get('from_node_id'),
            'to_id': relationship.get('to_node_id'),
            'grade': relationship.get('grade'),
            'inspection_date': self._parse_timestamp(relationship.get('inspection_date')),
            'graph_id': f"REL_{uuid.uuid4().hex[:8]}",
            'source': relationship.get('source', 'unknown')
        }
    
    def _map_operated_by_fields(self, relationship: Dict[str, Any]) -> Dict[str, Any]:
        """Map raw data to OPERATED_BY schema fields."""
        return {
            'relationship_type': 'OPERATED_BY',
            'from_id': relationship.get('from_node_id'),
            'to_id': relationship.get('to_node_id'),
            'shift': relationship.get('shift'),
            'start_time': self._parse_timestamp(relationship.get('start_time')),
            'end_time': self._parse_timestamp(relationship.get('end_time')),
            'graph_id': f"REL_{uuid.uuid4().hex[:8]}",
            'source': relationship.get('source', 'unknown')
        }
    
    def _map_monitored_by_fields(self, relationship: Dict[str, Any]) -> Dict[str, Any]:
        """Map raw data to MONITORED_BY schema fields."""
        return {
            'relationship_type': 'MONITORED_BY',
            'from_id': relationship.get('from_node_id'),
            'to_id': relationship.get('to_node_id'),
            'start_time': self._parse_timestamp(relationship.get('start_time')),
            'end_time': self._parse_timestamp(relationship.get('end_time')),
            'monitoring_data': self._parse_parameters(relationship.get('monitoring_data')),
            'alert_threshold': float(relationship.get('alert_threshold', 0)) if relationship.get('alert_threshold') else None,
            'graph_id': f"REL_{uuid.uuid4().hex[:8]}",
            'source': relationship.get('source', 'unknown')
        }
    
    def _map_generates_alert_fields(self, relationship: Dict[str, Any]) -> Dict[str, Any]:
        """Map raw data to GENERATES_ALERT schema fields."""
        return {
            'relationship_type': 'GENERATES_ALERT',
            'from_id': relationship.get('from_node_id'),
            'to_id': relationship.get('to_node_id'),
            'severity': relationship.get('severity'),
            'timestamp': self._parse_timestamp(relationship.get('timestamp')),
            'message': relationship.get('message'),
            'graph_id': f"REL_{uuid.uuid4().hex[:8]}",
            'source': relationship.get('source', 'unknown')
        }
    
    def _map_has_defect_fields(self, relationship: Dict[str, Any]) -> Dict[str, Any]:
        """Map raw data to HAS_DEFECT schema fields."""
        return {
            'relationship_type': 'HAS_DEFECT',
            'from_id': relationship.get('from_node_id'),
            'to_id': relationship.get('to_node_id'),
            'defect_type': relationship.get('defect_type'),
            'severity': relationship.get('severity'),
            'detection_time': self._parse_timestamp(relationship.get('detection_time')),
            'graph_id': f"REL_{uuid.uuid4().hex[:8]}",
            'source': relationship.get('source', 'unknown')
        }
    
    def _map_captured_by_fields(self, relationship: Dict[str, Any]) -> Dict[str, Any]:
        """Map raw data to CAPTURED_BY schema fields."""
        return {
            'relationship_type': 'CAPTURED_BY',
            'from_id': relationship.get('from_node_id'),
            'to_id': relationship.get('to_node_id'),
            'timestamp': self._parse_timestamp(relationship.get('timestamp')),
            'image_type': relationship.get('image_type'),
            'file_path': relationship.get('file_path'),
            'graph_id': f"REL_{uuid.uuid4().hex[:8]}",
            'source': relationship.get('source', 'unknown')
        }
    
    def _map_logged_by_fields(self, relationship: Dict[str, Any]) -> Dict[str, Any]:
        """Map raw data to LOGGED_BY schema fields."""
        return {
            'relationship_type': 'LOGGED_BY',
            'from_id': relationship.get('from_node_id'),
            'to_id': relationship.get('to_node_id'),
            'timestamp': self._parse_timestamp(relationship.get('timestamp')),
            'level': relationship.get('level'),
            'message': relationship.get('message'),
            'graph_id': f"REL_{uuid.uuid4().hex[:8]}",
            'source': relationship.get('source', 'unknown')
        }
    
    def _extract_relationships_from_node(self, node: Dict[str, Any], source: str) -> List[Dict[str, Any]]:
        """
        Extract relationships from a single node.
        
        Args:
            node: Node data
            source: Source database name
            
        Returns:
            List[Dict[str, Any]]: Extracted relationships
        """
        relationships = []
        
        # For raw data, use 'id' field; for processed data, use 'graph_id' field
        # Handle different ID field names for different node types
        from_graph_id = node.get('graph_id') or node.get('id') or node.get('process_id') or node.get('machine_id') or node.get('part_id') or node.get('build_id')
        node_type = node.get('node_type', 'Unknown')
        
        if not from_graph_id:
            return relationships
        
        
        # Extract relationships based on node type and properties
        if node_type == 'Process':
            relationships.extend(self._extract_process_relationships(node, from_graph_id, source))
        elif node_type == 'Machine':
            relationships.extend(self._extract_machine_relationships(node, from_graph_id, source))
        elif node_type == 'Part':
            relationships.extend(self._extract_part_relationships(node, from_graph_id, source))
        elif node_type == 'Sensor':
            relationships.extend(self._extract_sensor_relationships(node, from_graph_id, source))
        elif node_type == 'Build':
            relationships.extend(self._extract_build_relationships(node, from_graph_id, source))
        elif 'Image' in node_type:
            relationships.extend(self._extract_image_relationships(node, from_graph_id, source))
        elif 'Log' in node_type:
            relationships.extend(self._extract_log_relationships(node, from_graph_id, source))
        # New node types from MongoDB, Cassandra, Redis
        elif node_type in ['ProcessImage', 'CTScanImage', 'PowderBedImage', 'ThermalImage']:
            relationships.extend(self._extract_image_relationships(node, from_graph_id, source))
        elif node_type in ['BuildFile', 'ModelFile', 'LogFile']:
            relationships.extend(self._extract_file_relationships(node, from_graph_id, source))
        elif node_type in ['SensorReading', 'ProcessMonitoring', 'MachineStatus', 'AlertEvent']:
            relationships.extend(self._extract_event_relationships(node, from_graph_id, source))
        elif node_type in ['ProcessCache', 'AnalyticsCache']:
            relationships.extend(self._extract_cache_relationships(node, from_graph_id, source))
        elif node_type in ['JobQueue', 'UserSession']:
            relationships.extend(self._extract_session_relationships(node, from_graph_id, source))
        elif node_type in ['MachineConfig']:
            relationships.extend(self._extract_machine_config_relationships(node, from_graph_id, source))
        
        return relationships
    
    def _extract_process_relationships(self, node: Dict[str, Any], from_graph_id: str, source: str) -> List[Dict[str, Any]]:
        """Extract relationships for Process nodes."""
        relationships = []
        
        # For raw data, fields are directly in the node; for processed data, they're in properties
        properties = node.get('properties', node)
        
        # Process -> Machine
        machine_id = properties.get('machine_id')
        if machine_id:
            to_graph_id = self._find_target_node(machine_id)
            if to_graph_id:
                relationships.append(self._create_relationship(
                    from_graph_id, to_graph_id, "USES_MACHINE", 
                    {"source": source, "relationship_type": "operational"}
                ))
        
        # Process -> Build
        build_id = properties.get('build_id')
        if build_id:
            to_graph_id = self._find_target_node(build_id)
            if to_graph_id:
                relationships.append(self._create_relationship(
                    from_graph_id, to_graph_id, "PART_OF_BUILD",
                    {"source": source, "relationship_type": "structural"}
                ))
        
        # Process -> Part
        part_id = properties.get('part_id')
        if part_id:
            to_graph_id = self._find_target_node(part_id)
            if to_graph_id:
                relationships.append(self._create_relationship(
                    from_graph_id, to_graph_id, "CREATES_PART",
                    {"source": source, "relationship_type": "manufacturing"}
                ))
        return relationships
    
    def _extract_machine_relationships(self, node: Dict[str, Any], from_graph_id: str, source: str) -> List[Dict[str, Any]]:
        """Extract relationships for Machine nodes."""
        relationships = []
        
        # Get properties from the node
        properties = node.get('properties', {})
        
        # Machine -> Process (reverse of USES_MACHINE)
        process_id = properties.get('process_id')
        if process_id:
            to_graph_id = self._find_target_node(process_id)
            if to_graph_id:
                relationships.append(self._create_relationship(
                    from_graph_id, to_graph_id, "HOSTS_PROCESS",
                    {"source": source, "relationship_type": "operational"}
                ))
        
        return relationships
    
    def _extract_part_relationships(self, node: Dict[str, Any], from_graph_id: str, source: str) -> List[Dict[str, Any]]:
        """Extract relationships for Part nodes."""
        relationships = []
        
        # Get properties from the node
        properties = node.get('properties', {})
        
        # Part -> Build
        build_id = properties.get('build_id')
        if build_id:
            to_graph_id = self._find_target_node(build_id)
            if to_graph_id:
                relationships.append(self._create_relationship(
                    from_graph_id, to_graph_id, "BELONGS_TO_BUILD",
                    {"source": source, "relationship_type": "structural"}
                ))
        
        # Part -> Process (reverse of CREATES_PART)
        process_id = properties.get('process_id')
        if process_id:
            to_graph_id = self._find_target_node(process_id)
            if to_graph_id:
                relationships.append(self._create_relationship(
                    from_graph_id, to_graph_id, "CREATED_BY_PROCESS",
                    {"source": source, "relationship_type": "manufacturing"}
                ))
        
        return relationships
    
    def _extract_sensor_relationships(self, node: Dict[str, Any], from_graph_id: str, source: str) -> List[Dict[str, Any]]:
        """Extract relationships for Sensor nodes."""
        relationships = []
        
        # Get properties from the node
        properties = node.get('properties', {})
        
        # Sensor -> Process
        process_id = properties.get('process_id')
        if process_id:
            to_graph_id = self._find_target_node(process_id)
            if to_graph_id:
                relationships.append(self._create_relationship(
                    from_graph_id, to_graph_id, "MONITORS_PROCESS",
                    {"source": source, "relationship_type": "monitoring"}
                ))
        
        # Sensor -> Machine
        machine_id = properties.get('machine_id')
        if machine_id:
            to_graph_id = self._find_target_node(machine_id)
            if to_graph_id:
                relationships.append(self._create_relationship(
                    from_graph_id, to_graph_id, "ATTACHED_TO_MACHINE",
                    {"source": source, "relationship_type": "physical"}
                ))
        
        return relationships
    
    def _extract_build_relationships(self, node: Dict[str, Any], from_graph_id: str, source: str) -> List[Dict[str, Any]]:
        """Extract relationships for Build nodes."""
        relationships = []
        
        # Get properties from the node
        properties = node.get('properties', {})
        
        # Build -> Process (reverse of PART_OF_BUILD)
        process_id = properties.get('process_id')
        if process_id:
            to_graph_id = self._find_target_node(process_id)
            if to_graph_id:
                relationships.append(self._create_relationship(
                    from_graph_id, to_graph_id, "CONTAINS_PROCESS",
                    {"source": source, "relationship_type": "structural"}
                ))
        
        return relationships
    
    def _extract_image_relationships(self, node: Dict[str, Any], from_graph_id: str, source: str) -> List[Dict[str, Any]]:
        """Extract relationships for Image nodes."""
        relationships = []
        
        # Get properties from the node (for processed nodes) or directly from node (for raw nodes)
        properties = node.get('properties', {})
        
        # Image -> Process (check both properties and direct node fields)
        process_id = properties.get('process_id') or node.get('process_id')
        if process_id:
            to_graph_id = self._find_target_node(process_id)
            if to_graph_id:
                relationships.append(self._create_relationship(
                    from_graph_id, to_graph_id, "DOCUMENTS_PROCESS",
                    {"source": source, "relationship_type": "documentation"}
                ))
        
        # Image -> Part (for CT scan images)
        part_id = properties.get('part_id') or node.get('part_id')
        if part_id:
            to_graph_id = self._find_target_node(part_id)
            if to_graph_id:
                relationships.append(self._create_relationship(
                    from_graph_id, to_graph_id, "DOCUMENTS_PART",
                    {"source": source, "relationship_type": "documentation"}
                ))
        
        return relationships
    
    def _extract_log_relationships(self, node: Dict[str, Any], from_graph_id: str, source: str) -> List[Dict[str, Any]]:
        """Extract relationships for Log nodes."""
        relationships = []
        
        # Get properties from the node (for processed nodes) or directly from node (for raw nodes)
        properties = node.get('properties', {})
        
        # Log -> Process (check both properties and direct node fields)
        process_id = properties.get('process_id') or node.get('process_id')
        if process_id:
            to_graph_id = self._find_target_node(process_id)
            if to_graph_id:
                relationships.append(self._create_relationship(
                    from_graph_id, to_graph_id, "RECORDS_PROCESS",
                    {"source": source, "relationship_type": "logging"}
                ))
        
        return relationships
    
    def _extract_file_relationships(self, node: Dict[str, Any], from_graph_id: str, source: str) -> List[Dict[str, Any]]:
        """Extract relationships for File nodes (BuildFile, ModelFile, LogFile)."""
        relationships = []
        
        # Get properties from the node (fields are directly in the node for these sources)
        properties = node.get('properties', node)
        
        # File -> Process
        process_id = properties.get('process_id')
        if process_id:
            to_graph_id = self._find_target_node(process_id)
            if to_graph_id:
                relationships.append(self._create_relationship(
                    from_graph_id, to_graph_id, "BELONGS_TO_PROCESS",
                    {"source": source, "relationship_type": "ownership"}
                ))
        
        # File -> Machine
        machine_id = properties.get('machine_id')
        if machine_id:
            to_graph_id = self._find_target_node(machine_id)
            if to_graph_id:
                relationships.append(self._create_relationship(
                    from_graph_id, to_graph_id, "STORED_ON_MACHINE",
                    {"source": source, "relationship_type": "storage"}
                ))
        
        return relationships
    
    def _extract_event_relationships(self, node: Dict[str, Any], from_graph_id: str, source: str) -> List[Dict[str, Any]]:
        """Extract relationships for Event nodes (SensorReading, ProcessMonitoring, MachineStatus, AlertEvent)."""
        relationships = []
        
        # Get properties from the node (fields are directly in the node for these sources)
        properties = node.get('properties', node)
        
        # Event -> Process
        process_id = properties.get('process_id')
        if process_id:
            to_graph_id = self._find_target_node(process_id)
            if to_graph_id:
                relationships.append(self._create_relationship(
                    from_graph_id, to_graph_id, "RELATES_TO_PROCESS",
                    {"source": source, "relationship_type": "temporal"}
                ))
        
        # Event -> Machine
        machine_id = properties.get('machine_id')
        if machine_id:
            to_graph_id = self._find_target_node(machine_id)
            if to_graph_id:
                relationships.append(self._create_relationship(
                    from_graph_id, to_graph_id, "OCCURS_ON_MACHINE",
                    {"source": source, "relationship_type": "location"}
                ))
        
        # Event -> Sensor (for SensorReading)
        sensor_id = properties.get('sensor_id')
        if sensor_id:
            to_graph_id = self._find_target_node(sensor_id)
            if to_graph_id:
                relationships.append(self._create_relationship(
                    from_graph_id, to_graph_id, "GENERATED_BY_SENSOR",
                    {"source": source, "relationship_type": "measurement"}
                ))
        
        return relationships
    
    def _extract_cache_relationships(self, node: Dict[str, Any], from_graph_id: str, source: str) -> List[Dict[str, Any]]:
        """Extract relationships for Cache nodes (ProcessCache, AnalyticsCache)."""
        relationships = []
        
        # Get properties from the node (fields are directly in the node for these sources)
        properties = node.get('properties', node)
        
        # Cache -> Process
        process_id = properties.get('process_id')
        if process_id:
            to_graph_id = self._find_target_node(process_id)
            if to_graph_id:
                relationships.append(self._create_relationship(
                    from_graph_id, to_graph_id, "CACHES_PROCESS",
                    {"source": source, "relationship_type": "performance"}
                ))
        
        return relationships
    
    def _extract_session_relationships(self, node: Dict[str, Any], from_graph_id: str, source: str) -> List[Dict[str, Any]]:
        """Extract relationships for Session nodes (JobQueue, UserSession)."""
        relationships = []
        
        # Get properties from the node (fields are directly in the node for these sources)
        properties = node.get('properties', node)
        
        # Session -> User
        user_id = properties.get('user_id')
        if user_id:
            to_graph_id = self._find_target_node(user_id)
            if to_graph_id:
                relationships.append(self._create_relationship(
                    from_graph_id, to_graph_id, "OWNED_BY_USER",
                    {"source": source, "relationship_type": "ownership"}
                ))
        
        # JobQueue -> Process
        process_id = properties.get('process_id')
        if process_id:
            to_graph_id = self._find_target_node(process_id)
            if to_graph_id:
                relationships.append(self._create_relationship(
                    from_graph_id, to_graph_id, "QUEUES_PROCESS",
                    {"source": source, "relationship_type": "workflow"}
                ))
        
        return relationships
    
    def _extract_machine_config_relationships(self, node: Dict[str, Any], from_graph_id: str, source: str) -> List[Dict[str, Any]]:
        """Extract relationships for MachineConfig nodes."""
        relationships = []
        
        # Get properties from the node (fields are directly in the node for these sources)
        properties = node.get('properties', node)
        
        # MachineConfig -> Machine
        machine_id = properties.get('machine_id')
        if machine_id:
            to_graph_id = self._find_target_node(machine_id)
            if to_graph_id:
                relationships.append(self._create_relationship(
                    from_graph_id, to_graph_id, "CONFIGURES_MACHINE",
                    {"source": source, "relationship_type": "configuration"}
                ))
        
        # MachineConfig -> Process
        process_id = properties.get('process_id')
        if process_id:
            to_graph_id = self._find_target_node(process_id)
            if to_graph_id:
                relationships.append(self._create_relationship(
                    from_graph_id, to_graph_id, "CONFIGURES_PROCESS",
                    {"source": source, "relationship_type": "configuration"}
                ))
        
        # MachineConfig -> Build
        build_id = properties.get('build_id')
        if build_id:
            to_graph_id = self._find_target_node(build_id)
            if to_graph_id:
                relationships.append(self._create_relationship(
                    from_graph_id, to_graph_id, "CONFIGURES_BUILD",
                    {"source": source, "relationship_type": "configuration"}
                ))
        
        return relationships
    
    def _find_target_node(self, target_id: str) -> Optional[str]:
        """
        Find the graph ID for a target node.
        
        Args:
            target_id: The target node ID (could be process_id, machine_id, etc.)
            
        Returns:
            Optional[str]: The graph_id of the target node, or None if not found
        """
        # Look through all processed nodes to find one with matching ID fields
        for node_id, node in self.processed_nodes.items():
            # Check various ID fields that might match the target_id
            if (node.get('process_id') == str(target_id) or 
                node.get('machine_id') == str(target_id) or
                node.get('part_id') == str(target_id) or
                node.get('build_id') == str(target_id) or
                node.get('sensor_id') == str(target_id) or
                node.get('quality_id') == str(target_id) or
                node.get('material_id') == str(target_id) or
                node.get('document_id') == str(target_id) or
                node.get('job_id') == str(target_id) or
                node.get('session_id') == str(target_id) or
                node.get('user_id') == str(target_id)):
                return node.get('graph_id')
        
        # Also check if the target_id matches the node_id directly (for cases where node_id is the same as the ID)
        if target_id in self.processed_nodes:
            return self.processed_nodes[target_id].get('graph_id')
        
        return None
    
    def _create_relationship(self, from_id: str, to_id: str, relationship_type: str, properties: Dict[str, Any]) -> Dict[str, Any]:
        """Create a relationship object."""
        self.relationship_types.add(relationship_type)
        
        return {
            'from_id': from_id,
            'to_id': to_id,
            'relationship_type': relationship_type,
            'properties': properties,
            'created_at': datetime.utcnow().isoformat()
        }
    
    def get_relationships(self) -> List[Dict[str, Any]]:
        """Get all mapped relationships."""
        return self.relationships
    
    def get_relationship_types(self) -> Set[str]:
        """Get all relationship types."""
        return self.relationship_types
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get relationship mapping statistics."""
        type_counts = {}
        source_counts = {}
        
        for rel in self.relationships:
            rel_type = rel['relationship_type']
            source = rel['properties'].get('source', 'unknown')
            
            type_counts[rel_type] = type_counts.get(rel_type, 0) + 1
            source_counts[source] = source_counts.get(source, 0) + 1
        
        return {
            'total_relationships': len(self.relationships),
            'relationship_types': len(self.relationship_types),
            'type_counts': type_counts,
            'source_counts': source_counts
        }
