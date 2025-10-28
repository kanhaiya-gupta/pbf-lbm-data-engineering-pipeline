"""
Transformation Manager for Knowledge Graph Pipeline

This module orchestrates the complete transformation pipeline, coordinating
node processing, relationship mapping, and graph building with optimized
batch processing and comprehensive error handling.
"""

import logging
import time
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timezone
from pathlib import Path
import json
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from .node_processor import NodeProcessor
from .relationship_mapper import RelationshipMapper
from .graph_builder import GraphBuilder

logger = logging.getLogger(__name__)


class TransformationManager:
    """
    Orchestrates the complete knowledge graph transformation pipeline.
    
    Handles:
    - End-to-end ETL pipeline coordination
    - Multi-source data processing with batching
    - Parallel processing for performance optimization
    - Comprehensive error handling and recovery
    - Progress tracking and monitoring
    - Schema validation and data quality assurance
    """
    
    def __init__(self, 
                 output_dir: str = "data_lake/kg_neo4j",
                 batch_size: int = 100,
                 max_workers: int = 4,
                 enable_parallel: bool = True):
        """
        Initialize the transformation manager.
        
        Args:
            output_dir: Output directory for kg_neo4j data lake
            batch_size: Number of nodes to process in each batch
            max_workers: Maximum number of parallel workers
            enable_parallel: Enable parallel processing
        """
        self.output_dir = Path(output_dir)
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.enable_parallel = enable_parallel
        
        # Initialize components
        self.node_processor = NodeProcessor(output_dir=str(self.output_dir))
        self.relationship_mapper = None  # Will be initialized with processed nodes
        self.graph_builder = GraphBuilder(output_dir=str(self.output_dir))
        
        # Pipeline state
        self.processed_nodes: Dict[str, Dict[str, Any]] = {}
        self.processed_relationships: List[Dict[str, Any]] = []
        self.pipeline_metadata: Dict[str, Any] = {}
        self.processing_stats: Dict[str, Any] = {}
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "pipeline_logs").mkdir(exist_ok=True)
        (self.output_dir / "batch_results").mkdir(exist_ok=True)
        
        logger.info(f"🚀 TransformationManager initialized")
        logger.info(f"   📁 Output: {self.output_dir}")
        logger.info(f"   📦 Batch size: {self.batch_size}")
        logger.info(f"   ⚡ Parallel: {self.enable_parallel} ({self.max_workers} workers)")
    
    def process_complete_pipeline(self, 
                                 data_sources: Dict[str, Dict[str, List[Dict[str, Any]]]],
                                 pipeline_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process the complete transformation pipeline for all data sources.
        
        Args:
            data_sources: Dictionary of data sources and their extracted data
            pipeline_config: Optional pipeline configuration
            
        Returns:
            Dict[str, Any]: Complete pipeline results with metadata
        """
        start_time = time.time()
        logger.info("🚀 Starting complete transformation pipeline...")
        logger.info("=" * 60)
        
        try:
            # Initialize pipeline metadata
            self.pipeline_metadata = {
                'pipeline_id': str(uuid.uuid4()),
                'start_time': datetime.now(timezone.utc).isoformat(),
                'data_sources': list(data_sources.keys()),
                'batch_size': self.batch_size,
                'parallel_processing': self.enable_parallel,
                'max_workers': self.max_workers,
                'config': pipeline_config or {}
            }
            
            # Phase 1: Process all nodes from all sources
            logger.info("📊 Phase 1: Processing nodes from all sources...")
            all_processed_nodes = self._process_all_sources_nodes(data_sources)
            
            # Phase 2: Extract relationships
            logger.info("🔗 Phase 2: Extracting relationships...")
            all_relationships = self._extract_all_relationships(all_processed_nodes, data_sources)
            
            # Phase 3: Build complete graph
            logger.info("🏗️ Phase 3: Building complete knowledge graph...")
            complete_graph = self._build_complete_graph(all_processed_nodes, all_relationships)
            
            # Calculate final statistics
            processing_time = time.time() - start_time
            self.pipeline_metadata.update({
                'end_time': datetime.now(timezone.utc).isoformat(),
                'processing_time_seconds': processing_time,
                'total_nodes': len(all_processed_nodes),
                'total_relationships': len(all_relationships),
                'success': True
            })
            
            # Export pipeline results
            self._export_pipeline_results(complete_graph)
            
            logger.info("✅ Complete transformation pipeline completed successfully!")
            logger.info(f"   ⏱️ Processing time: {processing_time:.2f} seconds")
            logger.info(f"   📊 Total nodes: {len(all_processed_nodes)}")
            logger.info(f"   🔗 Total relationships: {len(all_relationships)}")
            logger.info(f"   📁 Results exported to: {self.output_dir}")
            
            return {
                'success': True,
                'graph': complete_graph,
                'metadata': self.pipeline_metadata,
                'stats': self.processing_stats
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Pipeline failed after {processing_time:.2f} seconds: {str(e)}"
            logger.error(f"❌ {error_msg}")
            
            self.pipeline_metadata.update({
                'end_time': datetime.now(timezone.utc).isoformat(),
                'processing_time_seconds': processing_time,
                'success': False,
                'error': str(e)
            })
            
            # Export error results
            self._export_error_results(str(e))
            
            return {
                'success': False,
                'error': str(e),
                'metadata': self.pipeline_metadata
            }
    
    def _process_all_sources_nodes(self, data_sources: Dict[str, Dict[str, List[Dict[str, Any]]]]) -> Dict[str, Dict[str, Any]]:
        """Process nodes from all data sources with batching and parallel processing."""
        all_processed_nodes = {}
        source_stats = {}
        
        for source_name, source_data in data_sources.items():
            logger.info(f"📊 Processing {source_name} nodes...")
            source_start_time = time.time()
            
            try:
                # Flatten source data into nodes with proper node_type
                source_nodes = self._flatten_source_data(source_name, source_data)
                logger.info(f"   📦 {source_name}: {len(source_nodes)} raw nodes")
                
                # Process nodes in batches
                if self.enable_parallel and len(source_nodes) > self.batch_size:
                    processed_nodes = self._process_nodes_parallel(source_name, source_nodes)
                else:
                    processed_nodes = self._process_nodes_sequential(source_name, source_nodes)
                
                # Merge into global processed nodes
                with self._lock:
                    all_processed_nodes.update(processed_nodes)
                
                source_time = time.time() - source_start_time
                source_stats[source_name] = {
                    'raw_nodes': len(source_nodes),
                    'processed_nodes': len(processed_nodes),
                    'processing_time': source_time,
                    'success': True
                }
                
                logger.info(f"   ✅ {source_name}: {len(processed_nodes)} processed nodes in {source_time:.2f}s")
                
            except Exception as e:
                source_time = time.time() - source_start_time
                logger.error(f"   ❌ {source_name} failed: {e}")
                source_stats[source_name] = {
                    'raw_nodes': 0,
                    'processed_nodes': 0,
                    'processing_time': source_time,
                    'success': False,
                    'error': str(e)
                }
        
        self.processing_stats['sources'] = source_stats
        logger.info(f"📊 Total processed nodes: {len(all_processed_nodes)}")
        
        return all_processed_nodes
    
    def _flatten_source_data(self, source_name: str, source_data: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Flatten source data into a list of nodes with proper node_type."""
        flattened_nodes = []
        
        # Node type mapping for different sources
        node_type_mapping = {
            'postgresql': {
                'processes': 'Process',
                'machines': 'Machine', 
                'parts': 'Part',
                'builds': 'Build',
                'sensors': 'Sensor'
            },
            'mongodb': {
                'process_images': 'process_image',
                'ct_scan_images': 'ct_scan_image',
                'powder_bed_images': 'powder_bed_image',
                'machine_build_files': 'machine_build_file',
                'model_3d_files': 'model_3d_file',
                'raw_sensor_data': 'raw_sensor_data',
                'process_logs': 'process_log',
                'machine_configurations': 'machine_configuration'
            },
            'cassandra': {
                'sensor_readings': 'SensorReading',
                'process_monitoring': 'ProcessMonitoring',
                'machine_status': 'MachineStatus',
                'alert_events': 'AlertEvent'
            },
            'redis': {
                'process_cache': 'ProcessCache',
                'analytics_cache': 'AnalyticsCache',
                'job_queue_items': 'JobQueue',
                'user_sessions': 'UserSession'
            }
        }
        
        mapping = node_type_mapping.get(source_name, {})
        
        for data_type, nodes in source_data.items():
            if isinstance(nodes, list):
                node_type = mapping.get(data_type, data_type)
                for node in nodes:
                    node['node_type'] = node_type
                    node['source'] = source_name
                    flattened_nodes.append(node)
        
        return flattened_nodes
    
    def _process_nodes_parallel(self, source_name: str, nodes: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Process nodes in parallel batches."""
        logger.info(f"   ⚡ Processing {len(nodes)} {source_name} nodes in parallel...")
        
        # Split nodes into batches
        batches = [nodes[i:i + self.batch_size] for i in range(0, len(nodes), self.batch_size)]
        logger.info(f"   📦 Created {len(batches)} batches of max {self.batch_size} nodes each")
        
        all_processed_nodes = {}
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit batch processing tasks
            future_to_batch = {
                executor.submit(self._process_node_batch, source_name, batch, i): (batch, i)
                for i, batch in enumerate(batches)
            }
            
            # Collect results
            for future in as_completed(future_to_batch):
                batch, batch_num = future_to_batch[future]
                try:
                    batch_processed_nodes = future.result()
                    with self._lock:
                        all_processed_nodes.update(batch_processed_nodes)
                    logger.info(f"   ✅ Batch {batch_num + 1}/{len(batches)} completed: {len(batch_processed_nodes)} nodes")
                except Exception as e:
                    logger.error(f"   ❌ Batch {batch_num + 1}/{len(batches)} failed: {e}")
        
        return all_processed_nodes
    
    def _process_nodes_sequential(self, source_name: str, nodes: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Process nodes sequentially."""
        logger.info(f"   📊 Processing {len(nodes)} {source_name} nodes sequentially...")
        
        # Process in batches
        all_processed_nodes = {}
        batches = [nodes[i:i + self.batch_size] for i in range(0, len(nodes), self.batch_size)]
        
        for i, batch in enumerate(batches):
            try:
                batch_processed_nodes = self._process_node_batch(source_name, batch, i)
                all_processed_nodes.update(batch_processed_nodes)
                logger.info(f"   ✅ Batch {i + 1}/{len(batches)} completed: {len(batch_processed_nodes)} nodes")
            except Exception as e:
                logger.error(f"   ❌ Batch {i + 1}/{len(batches)} failed: {e}")
        
        return all_processed_nodes
    
    def _process_node_batch(self, source_name: str, batch: List[Dict[str, Any]], batch_num: int) -> Dict[str, Dict[str, Any]]:
        """Process a single batch of nodes."""
        try:
            # Create a new NodeProcessor for this batch
            batch_processor = NodeProcessor(output_dir=str(self.output_dir))
            
            # Process based on source type
            if source_name == 'postgresql':
                processed_nodes = batch_processor.process_postgresql_nodes(batch)
            elif source_name == 'mongodb':
                processed_nodes = batch_processor.process_mongodb_nodes(batch)
            elif source_name == 'cassandra':
                processed_nodes = batch_processor.process_cassandra_nodes(batch)
            elif source_name == 'redis':
                processed_nodes = batch_processor.process_redis_nodes(batch)
            else:
                raise ValueError(f"Unknown source type: {source_name}")
            
            # Convert to dictionary format
            processed_dict = {}
            for node in processed_nodes:
                graph_id = node.get('graph_id')
                if graph_id:
                    processed_dict[graph_id] = node
                    # Also add by other ID fields for relationship lookup
                    for id_field in ['process_id', 'machine_id', 'part_id', 'build_id', 'sensor_id', 'document_id']:
                        id_value = node.get(id_field)
                        if id_value:
                            processed_dict[id_value] = node
            
            return processed_dict
            
        except Exception as e:
            logger.error(f"Batch {batch_num} processing failed: {e}")
            return {}
    
    def _extract_all_relationships(self, processed_nodes: Dict[str, Dict[str, Any]], 
                                 data_sources: Dict[str, Dict[str, List[Dict[str, Any]]]]) -> List[Dict[str, Any]]:
        """Extract relationships from all data sources."""
        logger.info("🔗 Extracting relationships from all sources...")
        
        # Initialize relationship mapper with all processed nodes
        self.relationship_mapper = RelationshipMapper(processed_nodes=processed_nodes, output_dir=str(self.output_dir))
        
        all_relationships = []
        relationship_stats = {}
        
        for source_name, source_data in data_sources.items():
            logger.info(f"   🔗 Extracting {source_name} relationships...")
            source_start_time = time.time()
            
            try:
                # Use processed nodes for relationship extraction instead of raw nodes
                # Filter processed nodes by source based on node types
                if source_name == 'postgresql':
                    source_processed_nodes = [node for node in processed_nodes.values() 
                                            if node.get('node_type') in ['Process', 'Machine', 'Part', 'Build', 'Sensor', 'Quality', 'Material']]
                elif source_name == 'mongodb':
                    source_processed_nodes = [node for node in processed_nodes.values() 
                                            if node.get('node_type') in ['ProcessImage', 'CTScanImage', 'PowderBedImage', 'BuildFile', 'ModelFile', 'SensorReading', 'LogFile', 'MachineConfig']]
                elif source_name == 'cassandra':
                    source_processed_nodes = [node for node in processed_nodes.values() 
                                            if node.get('node_type') in ['SensorReading', 'ProcessMonitoring', 'MachineStatus', 'AlertEvent', 'SensorType']]
                elif source_name == 'redis':
                    source_processed_nodes = [node for node in processed_nodes.values() 
                                            if node.get('node_type') in ['ProcessCache', 'AnalyticsCache', 'JobQueue', 'UserSession']]
                else:
                    source_processed_nodes = []
                
                # Extract relationships from each processed node
                source_relationships = []
                for node in source_processed_nodes:
                    try:
                        relationships = self.relationship_mapper._extract_relationships_from_node(node, source_name)
                        source_relationships.extend(relationships)
                    except Exception as e:
                        logger.warning(f"Failed to extract relationships from {source_name} node: {e}")
                        continue
                
                source_time = time.time() - source_start_time
                relationship_stats[source_name] = {
                    'relationships_found': len(source_relationships),
                    'processing_time': source_time,
                    'success': True
                }
                
                all_relationships.extend(source_relationships)
                logger.info(f"   ✅ {source_name}: {len(source_relationships)} relationships in {source_time:.2f}s")
                
            except Exception as e:
                source_time = time.time() - source_start_time
                logger.error(f"   ❌ {source_name} relationship extraction failed: {e}")
                relationship_stats[source_name] = {
                    'relationships_found': 0,
                    'processing_time': source_time,
                    'success': False,
                    'error': str(e)
                }
        
        self.processing_stats['relationships'] = relationship_stats
        logger.info(f"🔗 Total relationships extracted: {len(all_relationships)}")
        
        return all_relationships
    
    def _build_complete_graph(self, processed_nodes: Dict[str, Dict[str, Any]], 
                            relationships: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build the complete knowledge graph."""
        logger.info("🏗️ Building complete knowledge graph...")
        
        # Build graph metadata
        graph_metadata = {
            'transformation_manager': True,
            'pipeline_id': self.pipeline_metadata.get('pipeline_id'),
            'batch_size': self.batch_size,
            'parallel_processing': self.enable_parallel,
            'max_workers': self.max_workers
        }
        
        # Build the complete graph
        complete_graph = self.graph_builder.build_graph(
            processed_nodes=processed_nodes,
            relationships=relationships,
            metadata=graph_metadata
        )
        
        logger.info("✅ Complete knowledge graph built successfully")
        
        return complete_graph
    
    def _export_pipeline_results(self, complete_graph: Dict[str, Any]):
        """Export pipeline results and metadata."""
        logger.info("💾 Exporting pipeline results...")
        
        # Export pipeline metadata
        metadata_file = self.output_dir / "pipeline_logs" / "pipeline_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(self.pipeline_metadata, f, indent=2, default=str)
        
        # Export processing statistics
        stats_file = self.output_dir / "pipeline_logs" / "processing_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(self.processing_stats, f, indent=2, default=str)
        
        # Export complete graph summary
        summary = {
            'pipeline_metadata': self.pipeline_metadata,
            'processing_stats': self.processing_stats,
            'graph_summary': {
                'total_nodes': len(complete_graph.get('nodes', {})),
                'total_relationships': len(complete_graph.get('relationships', [])),
                'statistics': complete_graph.get('statistics', {})
            }
        }
        
        summary_file = self.output_dir / "pipeline_logs" / "pipeline_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"📁 Pipeline results exported to: {self.output_dir}/pipeline_logs")
    
    def _export_error_results(self, error_message: str):
        """Export error results when pipeline fails."""
        error_data = {
            'pipeline_metadata': self.pipeline_metadata,
            'error': error_message,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        error_file = self.output_dir / "pipeline_logs" / "pipeline_error.json"
        with open(error_file, 'w') as f:
            json.dump(error_data, f, indent=2, default=str)
        
        logger.error(f"📁 Error results exported to: {error_file}")
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status."""
        return {
            'metadata': self.pipeline_metadata,
            'stats': self.processing_stats,
            'processed_nodes': len(self.processed_nodes),
            'processed_relationships': len(self.processed_relationships)
        }
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get detailed processing statistics."""
        return self.processing_stats
