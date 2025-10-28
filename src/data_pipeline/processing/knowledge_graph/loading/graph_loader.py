"""
Graph Loader for Knowledge Graph

This module orchestrates the complete graph loading pipeline.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import json

from .neo4j_loader import Neo4jLoader
from .batch_processor import BatchProcessor
from .validation_engine import ValidationEngine

logger = logging.getLogger(__name__)


class GraphLoader:
    """
    Orchestrates the complete knowledge graph loading pipeline.
    
    Features:
    - End-to-end graph loading
    - Data validation
    - Batch processing
    - Error handling and recovery
    - Progress tracking
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the graph loader.
        
        Args:
            config: Optional Neo4j configuration
        """
        self.neo4j_loader = Neo4jLoader(config)
        self.batch_processor = BatchProcessor()
        self.validation_engine = ValidationEngine()
        
        self.loading_stats: Dict[str, Any] = {}
        
    def load_complete_graph(self, 
                          nodes: List[Dict[str, Any]], 
                          relationships: List[Dict[str, Any]],
                          clear_database: bool = False,
                          validate_data: bool = True,
                          batch_size: int = 100) -> Dict[str, Any]:
        """
        Load a complete knowledge graph.
        
        Args:
            nodes: List of nodes to load
            relationships: List of relationships to load
            clear_database: Whether to clear database before loading
            validate_data: Whether to validate data before loading
            batch_size: Batch size for processing
            
        Returns:
            Dict[str, Any]: Loading results
        """
        logger.info("🚀 Starting complete knowledge graph loading...")
        logger.info(f"   📊 Nodes: {len(nodes)}")
        logger.info(f"   🔗 Relationships: {len(relationships)}")
        
        start_time = datetime.utcnow()
        
        try:
            # Connect to Neo4j
            if not self.neo4j_loader.connect():
                return self._create_error_result("Failed to connect to Neo4j")
            
            # Clear database if requested
            if clear_database:
                logger.info("🧹 Clearing database...")
                if not self.neo4j_loader.clear_database():
                    return self._create_error_result("Failed to clear database")
            
            # Validate data if requested
            if validate_data:
                logger.info("🔍 Validating data...")
                validation_results = self._validate_all_data(nodes, relationships)
                if not validation_results['valid']:
                    logger.warning("⚠️ Data validation failed, but continuing...")
            
            # Load nodes
            logger.info("📊 Loading nodes...")
            node_results = self._load_nodes_batch(nodes, batch_size)
            
            # Load relationships
            logger.info("🔗 Loading relationships...")
            relationship_results = self._load_relationships_batch(relationships, batch_size)
            
            # Get final statistics
            final_stats = self.neo4j_loader.get_database_stats()
            
            # Create results
            results = {
                'success': True,
                'loading_time': (datetime.utcnow() - start_time).total_seconds(),
                'nodes': node_results,
                'relationships': relationship_results,
                'final_stats': final_stats,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            logger.info("✅ Knowledge graph loading completed successfully!")
            logger.info(f"   ⏱️ Loading time: {results['loading_time']:.2f}s")
            logger.info(f"   📊 Final nodes: {final_stats.get('node_count', 0)}")
            logger.info(f"   🔗 Final relationships: {final_stats.get('relationship_count', 0)}")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Graph loading failed: {e}")
            return self._create_error_result(str(e))
        
        finally:
            # Always disconnect
            self.neo4j_loader.disconnect()
    
    def _validate_all_data(self, nodes: List[Dict[str, Any]], 
                          relationships: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate all data before loading."""
        # Validate nodes
        node_validation = self.validation_engine.validate_nodes(nodes)
        
        # Get valid node IDs for relationship validation
        valid_node_ids = set()
        for node in nodes:
            if 'graph_id' in node:
                valid_node_ids.add(node['graph_id'])
        
        # Validate relationships
        relationship_validation = self.validation_engine.validate_relationships(
            relationships, valid_node_ids
        )
        
        return {
            'valid': (node_validation['invalid_nodes'] == 0 and 
                     relationship_validation['invalid_relationships'] == 0),
            'nodes': node_validation,
            'relationships': relationship_validation
        }
    
    def _load_nodes_batch(self, nodes: List[Dict[str, Any]], batch_size: int) -> Dict[str, Any]:
        """Load nodes using batch processing."""
        def process_batch(batch):
            return self.neo4j_loader.load_nodes(batch, batch_size=len(batch))
        
        return self.batch_processor.process_batches(
            nodes, process_batch, batch_size
        )
    
    def _load_relationships_batch(self, relationships: List[Dict[str, Any]], 
                                 batch_size: int) -> Dict[str, Any]:
        """Load relationships using batch processing."""
        def process_batch(batch):
            return self.neo4j_loader.load_relationships(batch, batch_size=len(batch))
        
        return self.batch_processor.process_batches(
            relationships, process_batch, batch_size
        )
    
    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """Create an error result."""
        return {
            'success': False,
            'error': error_message,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def load_from_transformation_pipeline(self, 
                                        transformation_results: Dict[str, Any],
                                        clear_database: bool = False,
                                        validate_data: bool = True,
                                        batch_size: int = 100) -> Dict[str, Any]:
        """
        Load graph from transformation pipeline results.
        
        Args:
            transformation_results: Results from transformation pipeline
            clear_database: Whether to clear database before loading
            validate_data: Whether to validate data before loading
            batch_size: Batch size for processing
            
        Returns:
            Dict[str, Any]: Loading results
        """
        logger.info("🔄 Loading from transformation pipeline...")
        
        # Extract nodes and relationships from transformation results
        nodes = list(transformation_results.get('nodes', {}).values())
        relationships = transformation_results.get('relationships', [])
        
        if not nodes:
            return self._create_error_result("No nodes found in transformation results")
        
        logger.info(f"📊 Extracted {len(nodes)} nodes and {len(relationships)} relationships")
        
        # Load the graph
        return self.load_complete_graph(
            nodes=nodes,
            relationships=relationships,
            clear_database=clear_database,
            validate_data=validate_data,
            batch_size=batch_size
        )
    
    def get_loading_statistics(self) -> Dict[str, Any]:
        """Get loading statistics."""
        return {
            'neo4j_stats': self.neo4j_loader.get_loading_summary(),
            'batch_stats': self.batch_processor.get_processing_stats(),
            'validation_stats': self.validation_engine.get_validation_summary(),
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def export_loading_report(self, results: Dict[str, Any]) -> str:
        """Export a detailed loading report."""
        report = {
            'loading_summary': {
                'success': results.get('success', False),
                'loading_time': results.get('loading_time', 0),
                'timestamp': results.get('timestamp', datetime.utcnow().isoformat())
            },
            'node_loading': results.get('nodes', {}),
            'relationship_loading': results.get('relationships', {}),
            'final_database_stats': results.get('final_stats', {}),
            'errors': results.get('errors', [])
        }
        
        return json.dumps(report, indent=2)
