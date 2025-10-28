"""
Neo4j Loader for Knowledge Graph

This module handles the loading of knowledge graph data into Neo4j database.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import json

from src.data_pipeline.storage.operational.neo4j_client import Neo4jClient
from src.data_pipeline.config.neo4j_config import get_neo4j_config

logger = logging.getLogger(__name__)


class Neo4jLoader:
    """
    Loads knowledge graph data into Neo4j database.
    
    Handles:
    - Node creation and updates
    - Relationship creation and updates
    - Batch operations for performance
    - Data validation and error handling
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Neo4j loader.
        
        Args:
            config: Optional Neo4j configuration
        """
        if config is None:
            config = get_neo4j_config()
        
        self.client = Neo4jClient(config)
        self.loaded_nodes: Dict[str, str] = {}  # graph_id -> neo4j_id mapping
        self.loaded_relationships: List[str] = []
        self.errors: List[Dict[str, Any]] = []
        
    def connect(self) -> bool:
        """Connect to Neo4j database."""
        try:
            self.client.connect()
            logger.info("✅ Connected to Neo4j for graph loading")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to connect to Neo4j: {e}")
            return False
    
    def disconnect(self) -> None:
        """Disconnect from Neo4j database."""
        try:
            self.client.disconnect()
            logger.info("✅ Disconnected from Neo4j")
        except Exception as e:
            logger.warning(f"⚠️ Error during disconnect: {e}")
    
    def load_nodes(self, nodes: List[Dict[str, Any]], batch_size: int = 100) -> Dict[str, Any]:
        """
        Load nodes into Neo4j.
        
        Args:
            nodes: List of nodes to load
            batch_size: Batch size for loading
            
        Returns:
            Dict[str, Any]: Loading results
        """
        logger.info(f"🚀 Loading {len(nodes)} nodes into Neo4j...")
        
        results = {
            'total_nodes': len(nodes),
            'loaded_nodes': 0,
            'failed_nodes': 0,
            'errors': []
        }
        
        # Process nodes in batches
        for i in range(0, len(nodes), batch_size):
            batch = nodes[i:i + batch_size]
            batch_results = self._load_node_batch(batch)
            
            results['loaded_nodes'] += batch_results['loaded']
            results['failed_nodes'] += batch_results['failed']
            results['errors'].extend(batch_results['errors'])
        
        logger.info(f"✅ Node loading completed:")
        logger.info(f"   📊 Loaded: {results['loaded_nodes']}")
        logger.info(f"   ❌ Failed: {results['failed_nodes']}")
        logger.info(f"   🔑 Node ID mapping size: {len(self.loaded_nodes)}")
        
        return results
    
    def _load_node_batch(self, nodes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Load a batch of nodes."""
        batch_results = {
            'loaded': 0,
            'failed': 0,
            'errors': []
        }
        
        for node in nodes:
            try:
                # Create node in Neo4j
                neo4j_id = self._create_node(node)
                if neo4j_id:
                    self.loaded_nodes[node['graph_id']] = neo4j_id
                    batch_results['loaded'] += 1
                else:
                    batch_results['failed'] += 1
                    batch_results['errors'].append({
                        'node_id': node['graph_id'],
                        'error': 'Failed to create node',
                        'timestamp': datetime.utcnow().isoformat()
                    })
            except Exception as e:
                batch_results['failed'] += 1
                batch_results['errors'].append({
                    'node_id': node.get('graph_id', 'unknown'),
                    'error': str(e),
                    'timestamp': datetime.utcnow().isoformat()
                })
                logger.warning(f"Failed to load node {node.get('graph_id', 'unknown')}: {e}")
        
        return batch_results
    
    def _create_node(self, node: Dict[str, Any]) -> Optional[str]:
        """Create a single node in Neo4j."""
        try:
            # Prepare node data
            labels = node.get('labels', [])
            properties = node.get('properties', {})
            
            # Flatten nested objects in properties for Neo4j compatibility
            flattened_properties = self._flatten_properties(properties)
            
            # Add graph metadata
            flattened_properties['graph_id'] = node['graph_id']
            flattened_properties['source'] = node.get('source', 'unknown')
            flattened_properties['node_type'] = node.get('node_type', 'Unknown')
            flattened_properties['created_at'] = node.get('created_at', datetime.utcnow().isoformat())
            
            # Create node using Neo4j client (use first label as primary)
            primary_label = labels[0] if labels else 'Node'
            result = self.client.create_node(
                label=primary_label,
                properties=flattened_properties
            )
            
            if result:
                # Neo4jClient.create_node returns an integer ID, not a dictionary
                return str(result)
            
            return None
            
        except Exception as e:
            logger.debug(f"Error creating node {node.get('graph_id', 'unknown')}: {e}")
            return None
    
    def _flatten_properties(self, properties: Dict[str, Any], prefix: str = '') -> Dict[str, Any]:
        """
        Flatten nested objects in properties for Neo4j compatibility.
        
        Neo4j only supports primitive types and arrays of primitives.
        Nested objects are flattened with dot notation.
        """
        flattened = {}
        
        for key, value in properties.items():
            full_key = f"{prefix}{key}" if prefix else key
            
            if isinstance(value, dict):
                # Recursively flatten nested dictionaries
                nested = self._flatten_properties(value, f"{full_key}.")
                flattened.update(nested)
            elif isinstance(value, list):
                # Check if list contains only primitives
                if all(isinstance(item, (str, int, float, bool)) or item is None for item in value):
                    flattened[full_key] = value
                else:
                    # Convert complex list to JSON string
                    flattened[full_key] = json.dumps(value, default=str)
            elif isinstance(value, (str, int, float, bool)) or value is None:
                # Primitive types are fine
                flattened[full_key] = value
            else:
                # Convert complex objects to JSON string
                flattened[full_key] = json.dumps(value, default=str)
        
        return flattened
    
    def load_relationships(self, relationships: List[Dict[str, Any]], batch_size: int = 100) -> Dict[str, Any]:
        """
        Load relationships into Neo4j.
        
        Args:
            relationships: List of relationships to load
            batch_size: Batch size for loading
            
        Returns:
            Dict[str, Any]: Loading results
        """
        logger.info(f"🚀 Loading {len(relationships)} relationships into Neo4j...")
        
        results = {
            'total_relationships': len(relationships),
            'loaded_relationships': 0,
            'failed_relationships': 0,
            'errors': []
        }
        
        # Process relationships in batches
        for i in range(0, len(relationships), batch_size):
            batch = relationships[i:i + batch_size]
            batch_results = self._load_relationship_batch(batch)
            
            results['loaded_relationships'] += batch_results['loaded']
            results['failed_relationships'] += batch_results['failed']
            results['errors'].extend(batch_results['errors'])
        
        logger.info(f"✅ Relationship loading completed:")
        logger.info(f"   🔗 Loaded: {results['loaded_relationships']}")
        logger.info(f"   ❌ Failed: {results['failed_relationships']}")
        
        return results
    
    def _load_relationship_batch(self, relationships: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Load a batch of relationships."""
        batch_results = {
            'loaded': 0,
            'failed': 0,
            'errors': []
        }
        
        
        for rel in relationships:
            try:
                # Use direct Cypher statements like the simple loader
                from_id = rel.get('from_id')
                to_id = rel.get('to_id')
                rel_type = rel.get('relationship_type', 'RELATES_TO')
                properties = rel.get('properties', {})
                
                # Clean properties for Neo4j compatibility
                properties = self._flatten_properties(properties)
                
                # Add metadata
                properties['source'] = rel.get('source', 'unknown')
                properties['created_at'] = rel.get('created_at', datetime.utcnow().isoformat())
                
                # Create relationship using direct Cypher (like simple loader)
                relationship_id = self._create_relationship_direct(
                    from_id, to_id, rel_type, properties
                )
                
                if relationship_id:
                    self.loaded_relationships.append(relationship_id)
                    batch_results['loaded'] += 1
                else:
                    batch_results['failed'] += 1
                    batch_results['errors'].append({
                        'relationship': f"{from_id} -> {to_id}",
                        'error': 'Failed to create relationship',
                        'timestamp': datetime.utcnow().isoformat()
                    })
                    
            except Exception as e:
                batch_results['failed'] += 1
                batch_results['errors'].append({
                    'relationship': f"{rel.get('from_id', 'unknown')} -> {rel.get('to_id', 'unknown')}",
                    'error': str(e),
                    'timestamp': datetime.utcnow().isoformat()
                })
                logger.warning(f"Failed to load relationship: {e}")
        
        return batch_results
    
    def _create_relationship_direct(self, from_id: str, to_id: str, rel_type: str, properties: Dict[str, Any]) -> Optional[str]:
        """Create a relationship using direct Cypher statements (like simple loader)."""
        try:
            if not self.client._driver:
                raise RuntimeError("Not connected to Neo4j")
            
            with self.client._driver.session(database=self.client.database) as session:
                # Use direct Cypher like the simple loader
                query = f"""
                MATCH (a {{graph_id: $from_id}}), (b {{graph_id: $to_id}})
                MERGE (a)-[r:{rel_type}]->(b)
                SET r += $properties
                RETURN id(r) as rel_id
                """
                
                result = session.run(query, from_id=from_id, to_id=to_id, properties=properties)
                record = result.single()
                
                if record:
                    rel_id = record["rel_id"]
                    logger.debug(f"Created relationship with ID: {rel_id}")
                    return str(rel_id)
                else:
                    logger.warning(f"Failed to create relationship {from_id} -> {to_id}")
                    return None
                    
        except Exception as e:
            logger.error(f"Failed to create relationship {from_id} -> {to_id}: {e}")
            return None
    
    def _create_relationship(self, rel: Dict[str, Any], from_id: str, to_id: str) -> Optional[str]:
        """Create a single relationship in Neo4j."""
        try:
            # Prepare relationship data
            rel_type = rel['relationship_type']
            properties = rel.get('properties', {})
            
            # Add metadata
            properties['created_at'] = datetime.utcnow().isoformat()
            
            # Create relationship using Neo4j client
            result = self.client.create_relationship(
                from_id=from_id,
                to_id=to_id,
                rel_type=rel_type,
                properties=properties
            )
            
            if result:
                # Neo4jClient.create_relationship returns an integer ID, not a dictionary
                return str(result)
            
            return None
            
        except Exception as e:
            logger.debug(f"Error creating relationship: {e}")
            return None
    
    def clear_database(self) -> bool:
        """Clear all data from the Neo4j database."""
        try:
            logger.info("🧹 Clearing Neo4j database...")
            result = self.client.execute_cypher("MATCH (n) DETACH DELETE n")
            logger.info("✅ Database cleared successfully")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to clear database: {e}")
            return False
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        try:
            # Get node count
            node_result = self.client.execute_cypher("MATCH (n) RETURN count(n) as node_count")
            node_count = node_result[0]['node_count'] if node_result else 0
            
            # Get relationship count
            rel_result = self.client.execute_cypher("MATCH ()-[r]->() RETURN count(r) as rel_count")
            rel_count = rel_result[0]['rel_count'] if rel_result else 0
            
            # Get node types
            type_result = self.client.execute_cypher("""
                MATCH (n) 
                RETURN labels(n) as labels, count(n) as count 
                ORDER BY count DESC
            """)
            
            node_types = {}
            for row in type_result:
                labels = row['labels']
                count = row['count']
                label_key = ':'.join(labels) if labels else 'Unlabeled'
                node_types[label_key] = count
            
            return {
                'node_count': node_count,
                'relationship_count': rel_count,
                'node_types': node_types,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {
                'node_count': 0,
                'relationship_count': 0,
                'node_types': {},
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    def get_loading_summary(self) -> Dict[str, Any]:
        """Get loading summary."""
        return {
            'loaded_nodes': len(self.loaded_nodes),
            'loaded_relationships': len(self.loaded_relationships),
            'errors': len(self.errors),
            'timestamp': datetime.utcnow().isoformat()
        }
