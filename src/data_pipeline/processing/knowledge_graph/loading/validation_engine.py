"""
Validation Engine for Knowledge Graph Loading

This module validates data before loading into Neo4j to ensure data quality.
"""

import logging
from typing import Dict, List, Any, Optional, Set, Callable
from datetime import datetime
import re

logger = logging.getLogger(__name__)


class ValidationEngine:
    """
    Validates knowledge graph data before loading.
    
    Validates:
    - Node structure and required fields
    - Relationship structure and references
    - Data types and constraints
    - Business logic rules
    """
    
    def __init__(self):
        """Initialize the validation engine."""
        self.validation_rules: Dict[str, List[Callable]] = {}
        self.validation_results: Dict[str, List[Dict[str, Any]]] = {}
        
    def validate_nodes(self, nodes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate nodes before loading.
        
        Args:
            nodes: List of nodes to validate
            
        Returns:
            Dict[str, Any]: Validation results
        """
        logger.info(f"🔍 Validating {len(nodes)} nodes...")
        
        results = {
            'total_nodes': len(nodes),
            'valid_nodes': 0,
            'invalid_nodes': 0,
            'warnings': 0,
            'errors': []
        }
        
        for i, node in enumerate(nodes):
            node_validation = self._validate_single_node(node, i)
            
            if node_validation['valid']:
                results['valid_nodes'] += 1
            else:
                results['invalid_nodes'] += 1
                results['errors'].extend(node_validation['errors'])
            
            if node_validation['warnings']:
                results['warnings'] += len(node_validation['warnings'])
        
        logger.info(f"✅ Node validation completed:")
        logger.info(f"   ✅ Valid: {results['valid_nodes']}")
        logger.info(f"   ❌ Invalid: {results['invalid_nodes']}")
        logger.info(f"   ⚠️ Warnings: {results['warnings']}")
        
        return results
    
    def validate_relationships(self, relationships: List[Dict[str, Any]], 
                             node_ids: Set[str]) -> Dict[str, Any]:
        """
        Validate relationships before loading.
        
        Args:
            relationships: List of relationships to validate
            node_ids: Set of valid node IDs
            
        Returns:
            Dict[str, Any]: Validation results
        """
        logger.info(f"🔍 Validating {len(relationships)} relationships...")
        
        results = {
            'total_relationships': len(relationships),
            'valid_relationships': 0,
            'invalid_relationships': 0,
            'warnings': 0,
            'errors': []
        }
        
        for i, rel in enumerate(relationships):
            rel_validation = self._validate_single_relationship(rel, node_ids, i)
            
            if rel_validation['valid']:
                results['valid_relationships'] += 1
            else:
                results['invalid_relationships'] += 1
                results['errors'].extend(rel_validation['errors'])
            
            if rel_validation['warnings']:
                results['warnings'] += len(rel_validation['warnings'])
        
        logger.info(f"✅ Relationship validation completed:")
        logger.info(f"   ✅ Valid: {results['valid_relationships']}")
        logger.info(f"   ❌ Invalid: {results['invalid_relationships']}")
        logger.info(f"   ⚠️ Warnings: {results['warnings']}")
        
        return results
    
    def _validate_single_node(self, node: Dict[str, Any], index: int) -> Dict[str, Any]:
        """Validate a single node."""
        validation = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Check required fields
        required_fields = ['graph_id', 'node_type', 'labels', 'properties']
        for field in required_fields:
            if field not in node:
                validation['valid'] = False
                validation['errors'].append({
                    'field': field,
                    'message': f'Required field missing: {field}',
                    'index': index
                })
        
        # Validate graph_id format
        if 'graph_id' in node:
            if not self._is_valid_graph_id(node['graph_id']):
                validation['valid'] = False
                validation['errors'].append({
                    'field': 'graph_id',
                    'message': f'Invalid graph_id format: {node["graph_id"]}',
                    'index': index
                })
        
        # Validate node_type
        if 'node_type' in node:
            if not self._is_valid_node_type(node['node_type']):
                validation['warnings'].append({
                    'field': 'node_type',
                    'message': f'Unknown node_type: {node["node_type"]}',
                    'index': index
                })
        
        # Validate labels
        if 'labels' in node:
            if not isinstance(node['labels'], list) or not node['labels']:
                validation['valid'] = False
                validation['errors'].append({
                    'field': 'labels',
                    'message': 'Labels must be a non-empty list',
                    'index': index
                })
            else:
                for label in node['labels']:
                    if not self._is_valid_label(label):
                        validation['warnings'].append({
                            'field': 'labels',
                            'message': f'Invalid label format: {label}',
                            'index': index
                        })
        
        # Validate properties
        if 'properties' in node:
            if not isinstance(node['properties'], dict):
                validation['valid'] = False
                validation['errors'].append({
                    'field': 'properties',
                    'message': 'Properties must be a dictionary',
                    'index': index
                })
            else:
                for key, value in node['properties'].items():
                    if not self._is_valid_property_key(key):
                        validation['warnings'].append({
                            'field': f'properties.{key}',
                            'message': f'Invalid property key format: {key}',
                            'index': index
                        })
        
        return validation
    
    def _validate_single_relationship(self, rel: Dict[str, Any], 
                                    node_ids: Set[str], index: int) -> Dict[str, Any]:
        """Validate a single relationship."""
        validation = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Check required fields
        required_fields = ['from_id', 'to_id', 'relationship_type']
        for field in required_fields:
            if field not in rel:
                validation['valid'] = False
                validation['errors'].append({
                    'field': field,
                    'message': f'Required field missing: {field}',
                    'index': index
                })
        
        # Validate node references
        if 'from_id' in rel and 'to_id' in rel:
            if rel['from_id'] not in node_ids:
                validation['valid'] = False
                validation['errors'].append({
                    'field': 'from_id',
                    'message': f'Source node not found: {rel["from_id"]}',
                    'index': index
                })
            
            if rel['to_id'] not in node_ids:
                validation['valid'] = False
                validation['errors'].append({
                    'field': 'to_id',
                    'message': f'Target node not found: {rel["to_id"]}',
                    'index': index
                })
        
        # Validate relationship type
        if 'relationship_type' in rel:
            if not self._is_valid_relationship_type(rel['relationship_type']):
                validation['warnings'].append({
                    'field': 'relationship_type',
                    'message': f'Unknown relationship type: {rel["relationship_type"]}',
                    'index': index
                })
        
        # Validate properties
        if 'properties' in rel:
            if not isinstance(rel['properties'], dict):
                validation['valid'] = False
                validation['errors'].append({
                    'field': 'properties',
                    'message': 'Properties must be a dictionary',
                    'index': index
                })
        
        return validation
    
    def _is_valid_graph_id(self, graph_id: str) -> bool:
        """Validate graph ID format."""
        # Graph ID should be alphanumeric with underscores
        return bool(re.match(r'^[A-Za-z0-9_]+$', graph_id))
    
    def _is_valid_node_type(self, node_type: str) -> bool:
        """Validate node type."""
        valid_types = {
            # Core manufacturing nodes
            'Process', 'Machine', 'Sensor', 'Part', 'Build', 
            'Quality', 'Material', 'SensorType',
            
            # Image nodes
            'ProcessImage', 'CTScanImage', 'PowderBedImage', 'ThermalImage',
            
            # File nodes
            'BuildFile', 'ModelFile', 'LogFile',
            
            # Cache nodes
            'ProcessCache', 'AnalyticsCache',
            
            # Queue nodes
            'JobQueue',
            
            # Session nodes
            'UserSession',
            
            # Reading nodes
            'SensorReading',
            
            # Event nodes
            'ProcessMonitoring', 'MachineStatus', 'AlertEvent',
            
            # Configuration nodes
            'MachineConfig', 'Batch', 'Measurement',
            
            # Legacy nodes (for backward compatibility)
            'MachineBuildFile', 'Model3DFile', 'RawSensorData',
            'ProcessLog', 'QualityMetric', 'MaterialProperty', 
            'AnalyticsAggregation'
        }
        return node_type in valid_types
    
    def _is_valid_label(self, label: str) -> bool:
        """Validate Neo4j label format."""
        # Labels should be alphanumeric with underscores, start with letter
        return bool(re.match(r'^[A-Za-z][A-Za-z0-9_]*$', label))
    
    def _is_valid_property_key(self, key: str) -> bool:
        """Validate property key format."""
        # Property keys should be alphanumeric with underscores
        return bool(re.match(r'^[A-Za-z][A-Za-z0-9_]*$', key))
    
    def _is_valid_relationship_type(self, rel_type: str) -> bool:
        """Validate relationship type."""
        valid_types = {
            # Core manufacturing relationships
            'USES_MACHINE', 'HOSTS_PROCESS', 'CREATES_PART', 'CREATED_BY_PROCESS',
            'BELONGS_TO_BUILD', 'CONTAINS_PROCESS', 'PART_OF_BUILD',
            'MONITORS_PROCESS', 'ATTACHED_TO_MACHINE', 'DOCUMENTS_PROCESS',
            'RECORDS_PROCESS', 'HAS_QUALITY', 'HAS_MATERIAL', 'GENERATES_ALERT',
            'AGGREGATES_DATA', 'RELATED_TO', 'CONNECTED_TO',
            
            # Configuration relationships
            'CONFIGURES_MACHINE', 'CONFIGURES_PROCESS', 'CONFIGURES_BUILD',
            
            # Sensor relationships
            'GENERATED_BY_SENSOR', 'MEASURES_PROCESS', 'MONITORS_MACHINE',
            
            # Queue relationships
            'QUEUES_PROCESS', 'ASSIGNED_TO_USER', 'PROCESSES_JOB',
            
            # Cache relationships
            'CACHES_DATA', 'STORES_RESULTS', 'ACCELERATES_QUERY',
            
            # Session relationships
            'INITIATED_BY_USER', 'MANAGES_SESSION', 'TRACKS_ACTIVITY',
            
            # File relationships
            'CONTAINS_FILE', 'STORES_MODEL', 'LOGS_EVENTS',
            
            # Image relationships
            'CAPTURES_IMAGE', 'SCANS_OBJECT', 'MONITORS_BED'
        }
        return rel_type in valid_types
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get validation summary."""
        return {
            'validation_rules': len(self.validation_rules),
            'validation_results': self.validation_results,
            'timestamp': datetime.utcnow().isoformat()
        }
