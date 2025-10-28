"""
Schema-Aware Graph Builder for Knowledge Graph Transformation

This module builds the overall graph structure by combining schema-compliant nodes
and relationships into a unified knowledge graph with kg_neo4j data lake export.
"""

import logging
from typing import Dict, List, Any, Optional, Set
from datetime import datetime
import json
import uuid
from pathlib import Path

# Import our Neo4j models and validation
from src.data_pipeline.processing.schema.models.neo4j_models import (
    Neo4jModelFactory, GraphValidationEngine
)

logger = logging.getLogger(__name__)


class GraphBuilder:
    """
    Schema-aware graph builder that assembles the complete knowledge graph
    from schema-compliant nodes and relationships with kg_neo4j data lake export.
    
    Handles:
    - Graph structure assembly from schema-compliant data
    - Node and relationship integration with validation
    - Graph-level business logic validation
    - Export to kg_neo4j data lake structure
    """
    
    def __init__(self, output_dir: str = "data_lake/kg_neo4j"):
        """Initialize the schema-aware graph builder."""
        self.factory = Neo4jModelFactory()
        self.validation_engine = GraphValidationEngine()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create output subdirectories
        (self.output_dir / "complete_graph").mkdir(exist_ok=True)
        (self.output_dir / "graph_metadata").mkdir(exist_ok=True)
        
        self.graph_nodes: Dict[str, Dict[str, Any]] = {}
        self.graph_relationships: List[Dict[str, Any]] = []
        self.graph_metadata: Dict[str, Any] = {}
        self.validation_results: List[Dict[str, Any]] = []
        
    def build_graph(self, 
                   processed_nodes: Dict[str, Dict[str, Any]], 
                   relationships: List[Dict[str, Any]],
                   metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Build the complete schema-compliant knowledge graph with validation.
        
        Args:
            processed_nodes: Schema-compliant processed nodes from NodeProcessor
            relationships: Schema-compliant relationships from RelationshipMapper
            metadata: Optional graph metadata
            
        Returns:
            Dict[str, Any]: Complete graph structure with kg_neo4j export
        """
        logger.info("🚀 Building schema-compliant knowledge graph...")
        
        # Transform nodes to Neo4j format (nested structure for loading components)
        self.graph_nodes = self._transform_nodes_to_neo4j_format(processed_nodes)
        self.graph_relationships = relationships.copy()
        
        # Build graph metadata
        self.graph_metadata = self._build_metadata(metadata)
        
        # Validate complete graph structure with schema compliance
        self._validate_complete_graph()
        
        # Optimize graph structure
        self._optimize_graph()
        
        # Build final graph
        graph = {
            'metadata': self.graph_metadata,
            'nodes': self.graph_nodes,
            'relationships': self.graph_relationships,
            'statistics': self._build_statistics()
        }
        
        # Export to kg_neo4j data lake
        self._export_complete_graph_to_kg_neo4j(graph)
        
        logger.info(f"✅ Schema-compliant knowledge graph built successfully")
        logger.info(f"   📊 Nodes: {len(self.graph_nodes)}")
        logger.info(f"   🔗 Relationships: {len(self.graph_relationships)}")
        logger.info(f"📁 Exported to kg_neo4j data lake: {self.output_dir}")
        
        return graph
    
    def _validate_complete_graph(self):
        """
        Validate the complete graph structure with schema compliance and business logic.
        """
        logger.info("🔍 Validating complete graph structure...")
        
        # Validate all nodes with schema compliance
        node_validation_results = []
        for node_id, node_data in self.graph_nodes.items():
            try:
                node_type = node_data.get('node_type', 'Unknown')
                validation_result = self.validation_engine.validate_node(node_data, node_type)
                node_validation_results.append({
                    'node_id': node_id,
                    'node_type': node_type,
                    'valid': validation_result.valid,
                    'errors': [e.dict() for e in validation_result.errors],
                    'warnings': [w.dict() for w in validation_result.warnings]
                })
            except Exception as e:
                logger.warning(f"Failed to validate node {node_id}: {e}")
                node_validation_results.append({
                    'node_id': node_id,
                    'valid': False,
                    'errors': [{'message': str(e)}]
                })
        
        # Validate all relationships with schema compliance
        relationship_validation_results = []
        for rel in self.graph_relationships:
            try:
                rel_type = rel.get('relationship_type', 'Unknown')
                validation_result = self.validation_engine.validate_relationship(rel, rel_type)
                relationship_validation_results.append({
                    'relationship_type': rel_type,
                    'valid': validation_result.valid,
                    'errors': [e.dict() for e in validation_result.errors],
                    'warnings': [w.dict() for w in validation_result.warnings]
                })
            except Exception as e:
                logger.warning(f"Failed to validate relationship: {e}")
                relationship_validation_results.append({
                    'valid': False,
                    'errors': [{'message': str(e)}]
                })
        
        # Store validation results
        self.validation_results = {
            'nodes': node_validation_results,
            'relationships': relationship_validation_results,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Log validation summary
        valid_nodes = sum(1 for r in node_validation_results if r.get('valid', False))
        valid_relationships = sum(1 for r in relationship_validation_results if r.get('valid', False))
        
        logger.info(f"✅ Graph validation completed:")
        logger.info(f"   📊 Nodes: {valid_nodes}/{len(node_validation_results)} valid")
        logger.info(f"   🔗 Relationships: {valid_relationships}/{len(relationship_validation_results)} valid")
    
    def _export_complete_graph_to_kg_neo4j(self, graph: Dict[str, Any]):
        """
        Export the complete graph to kg_neo4j data lake structure.
        
        Args:
            graph: Complete graph structure
        """
        logger.info("💾 Exporting complete graph to kg_neo4j data lake...")
        
        # Export complete graph
        complete_graph_file = self.output_dir / "complete_graph" / "complete_graph.json"
        with open(complete_graph_file, 'w') as f:
            json.dump(graph, f, indent=2, default=str)
        
        # Export metadata
        metadata_file = self.output_dir / "graph_metadata" / "complete_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(self.graph_metadata, f, indent=2, default=str)
        
        # Export validation results
        validation_file = self.output_dir / "graph_metadata" / "validation_results.json"
        with open(validation_file, 'w') as f:
            json.dump(self.validation_results, f, indent=2, default=str)
        
        # Create summary
        summary = {
            'export_timestamp': datetime.utcnow().isoformat(),
            'complete_graph_file': str(complete_graph_file),
            'metadata_file': str(metadata_file),
            'validation_file': str(validation_file),
            'total_nodes': len(self.graph_nodes),
            'total_relationships': len(self.graph_relationships),
            'schema_version': '1.0.0'
        }
        
        summary_file = self.output_dir / "graph_metadata" / "export_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"✅ Complete graph exported to kg_neo4j data lake")
        logger.info(f"📁 Complete graph: {complete_graph_file}")
        logger.info(f"📁 Metadata: {metadata_file}")
        logger.info(f"📁 Validation: {validation_file}")
        logger.info(f"📁 Summary: {summary_file}")
    
    def _build_metadata(self, metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Build graph metadata."""
        base_metadata = {
            'graph_name': 'PBF-LB/M Knowledge Graph',
            'description': 'Knowledge graph for Powder Bed Fusion - Laser Beam Melting processes',
            'version': '1.0.0',
            'created_at': datetime.utcnow().isoformat(),
            'updated_at': datetime.utcnow().isoformat(),
            'node_count': len(self.graph_nodes),
            'relationship_count': len(self.graph_relationships)
        }
        
        if metadata:
            base_metadata.update(metadata)
        
        return base_metadata
    
    def _transform_nodes_to_neo4j_format(self, processed_nodes: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Transform flat node structure to Neo4j format expected by loading components.
        
        Args:
            processed_nodes: Flat node structure from transformation
            
        Returns:
            Dict[str, Dict[str, Any]]: Neo4j format nodes with nested structure
        """
        logger.info("🔄 Transforming nodes to Neo4j format...")
        
        neo4j_nodes = {}
        
        for node_id, node_data in processed_nodes.items():
            # Extract graph metadata
            graph_id = node_data.get('graph_id', node_id)
            node_type = node_data.get('node_type', 'Unknown')
            source = node_data.get('source', 'unknown')
            created_at = node_data.get('created_at', datetime.utcnow().isoformat())
            updated_at = node_data.get('updated_at', created_at)
            
            # Create labels from node_type
            labels = [node_type]
            
            # Separate properties from metadata
            properties = {}
            for key, value in node_data.items():
                # Skip metadata fields
                if key not in ['graph_id', 'node_type', 'source', 'created_at', 'updated_at']:
                    properties[key] = value
            
            # Create Neo4j format node
            neo4j_node = {
                'graph_id': graph_id,
                'node_type': node_type,
                'labels': labels,
                'properties': properties,
                'source': source,
                'created_at': created_at,
                'updated_at': updated_at
            }
            
            neo4j_nodes[node_id] = neo4j_node
        
        logger.info(f"✅ Transformed {len(neo4j_nodes)} nodes to Neo4j format")
        return neo4j_nodes
    
    def _validate_graph(self) -> None:
        """Validate the graph structure."""
        logger.info("🔍 Validating graph structure...")
        
        # Check for orphaned relationships
        orphaned_rels = []
        for rel in self.graph_relationships:
            from_id = rel['from_id']
            to_id = rel['to_id']
            
            if from_id not in self.graph_nodes:
                orphaned_rels.append(f"Missing from node: {from_id}")
            if to_id not in self.graph_nodes:
                orphaned_rels.append(f"Missing to node: {to_id}")
        
        if orphaned_rels:
            logger.warning(f"Found {len(orphaned_rels)} orphaned relationships")
            for orphan in orphaned_rels[:5]:  # Log first 5
                logger.warning(f"  - {orphan}")
        
        # Check for duplicate relationships
        seen_rels = set()
        duplicates = []
        for rel in self.graph_relationships:
            rel_key = (rel['from_id'], rel['to_id'], rel['relationship_type'])
            if rel_key in seen_rels:
                duplicates.append(rel_key)
            else:
                seen_rels.add(rel_key)
        
        if duplicates:
            logger.warning(f"Found {len(duplicates)} duplicate relationships")
        
        logger.info("✅ Graph validation completed")
    
    def _optimize_graph(self) -> None:
        """Optimize the graph structure."""
        logger.info("⚡ Optimizing graph structure...")
        
        # Remove orphaned relationships
        original_count = len(self.graph_relationships)
        self.graph_relationships = [
            rel for rel in self.graph_relationships
            if rel['from_id'] in self.graph_nodes and rel['to_id'] in self.graph_nodes
        ]
        removed_count = original_count - len(self.graph_relationships)
        
        if removed_count > 0:
            logger.info(f"   🧹 Removed {removed_count} orphaned relationships")
        
        # Deduplicate relationships
        seen_rels = set()
        unique_rels = []
        duplicates_removed = 0
        
        for rel in self.graph_relationships:
            rel_key = (rel['from_id'], rel['to_id'], rel['relationship_type'])
            if rel_key not in seen_rels:
                seen_rels.add(rel_key)
                unique_rels.append(rel)
            else:
                duplicates_removed += 1
        
        if duplicates_removed > 0:
            logger.info(f"   🧹 Removed {duplicates_removed} duplicate relationships")
        
        self.graph_relationships = unique_rels
        
        logger.info("✅ Graph optimization completed")
    
    def _build_statistics(self) -> Dict[str, Any]:
        """Build graph statistics."""
        # Node statistics
        node_types = {}
        node_sources = {}
        node_labels = {}
        
        for node in self.graph_nodes.values():
            node_type = node['node_type']
            source = node['source']
            labels = node.get('labels', [])
            
            node_types[node_type] = node_types.get(node_type, 0) + 1
            node_sources[source] = node_sources.get(source, 0) + 1
            
            for label in labels:
                node_labels[label] = node_labels.get(label, 0) + 1
        
        # Relationship statistics
        rel_types = {}
        rel_sources = {}
        
        for rel in self.graph_relationships:
            rel_type = rel['relationship_type']
            source = rel['properties'].get('source', 'unknown')
            
            rel_types[rel_type] = rel_types.get(rel_type, 0) + 1
            rel_sources[source] = rel_sources.get(source, 0) + 1
        
        # Calculate connectivity metrics
        node_degrees = self._calculate_node_degrees()
        connectivity_stats = self._calculate_connectivity_stats(node_degrees)
        
        return {
            'nodes': {
                'total': len(self.graph_nodes),
                'by_type': node_types,
                'by_source': node_sources,
                'by_label': node_labels
            },
            'relationships': {
                'total': len(self.graph_relationships),
                'by_type': rel_types,
                'by_source': rel_sources
            },
            'connectivity': connectivity_stats
        }
    
    def _calculate_node_degrees(self) -> Dict[str, int]:
        """Calculate node degrees (number of connections)."""
        degrees = {node_id: 0 for node_id in self.graph_nodes.keys()}
        
        for rel in self.graph_relationships:
            from_id = rel['from_id']
            to_id = rel['to_id']
            
            if from_id in degrees:
                degrees[from_id] += 1
            if to_id in degrees:
                degrees[to_id] += 1
        
        return degrees
    
    def _calculate_connectivity_stats(self, node_degrees: Dict[str, int]) -> Dict[str, Any]:
        """Calculate connectivity statistics."""
        degrees = list(node_degrees.values())
        
        if not degrees:
            return {
                'average_degree': 0,
                'max_degree': 0,
                'min_degree': 0,
                'isolated_nodes': 0,
                'highly_connected_nodes': 0
            }
        
        avg_degree = sum(degrees) / len(degrees)
        max_degree = max(degrees)
        min_degree = min(degrees)
        isolated_nodes = sum(1 for d in degrees if d == 0)
        highly_connected = sum(1 for d in degrees if d >= 10)
        
        return {
            'average_degree': round(avg_degree, 2),
            'max_degree': max_degree,
            'min_degree': min_degree,
            'isolated_nodes': isolated_nodes,
            'highly_connected_nodes': highly_connected
        }
    
    def get_graph_nodes(self) -> Dict[str, Dict[str, Any]]:
        """Get all graph nodes."""
        return self.graph_nodes
    
    def get_graph_relationships(self) -> List[Dict[str, Any]]:
        """Get all graph relationships."""
        return self.graph_relationships
    
    def get_graph_metadata(self) -> Dict[str, Any]:
        """Get graph metadata."""
        return self.graph_metadata
    
    def export_to_neo4j_format(self) -> Dict[str, Any]:
        """Export graph to Neo4j format."""
        return {
            'nodes': [
                {
                    'id': node_id,
                    'labels': node['labels'],
                    'properties': node['properties']
                }
                for node_id, node in self.graph_nodes.items()
            ],
            'relationships': [
                {
                    'from_id': rel['from_id'],
                    'to_id': rel['to_id'],
                    'type': rel['relationship_type'],
                    'properties': rel['properties']
                }
                for rel in self.graph_relationships
            ]
        }
    
    def export_to_cypher(self) -> List[str]:
        """Export graph to Cypher queries."""
        cypher_queries = []
        
        # Create nodes
        for node_id, node in self.graph_nodes.items():
            labels = ':'.join(node['labels'])
            properties = self._format_properties_for_cypher(node['properties'])
            
            query = f"CREATE (n:{labels} {properties})"
            cypher_queries.append(query)
        
        # Create relationships
        for rel in self.graph_relationships:
            from_id = rel['from_id']
            to_id = rel['to_id']
            rel_type = rel['relationship_type']
            properties = self._format_properties_for_cypher(rel['properties'])
            
            query = f"MATCH (a), (b) WHERE a.graph_id = '{from_id}' AND b.graph_id = '{to_id}' CREATE (a)-[r:{rel_type} {properties}]->(b)"
            cypher_queries.append(query)
        
        return cypher_queries
    
    def _format_properties_for_cypher(self, properties: Dict[str, Any]) -> str:
        """Format properties for Cypher queries."""
        if not properties:
            return ""
        
        formatted_props = []
        for key, value in properties.items():
            if isinstance(value, str):
                formatted_props.append(f"{key}: '{value}'")
            elif isinstance(value, (int, float, bool)):
                formatted_props.append(f"{key}: {value}")
            else:
                formatted_props.append(f"{key}: '{json.dumps(value)}'")
        
        return "{" + ", ".join(formatted_props) + "}"
