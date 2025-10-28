"""
Batch Processor for Knowledge Graph Loading

This module handles batch processing operations for efficient graph loading.
"""

import logging
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
import time

logger = logging.getLogger(__name__)


class BatchProcessor:
    """
    Handles batch processing operations for graph loading.
    
    Features:
    - Configurable batch sizes
    - Progress tracking
    - Error handling and recovery
    - Performance monitoring
    """
    
    def __init__(self, batch_size: int = 100, max_retries: int = 3):
        """
        Initialize the batch processor.
        
        Args:
            batch_size: Default batch size for processing
            max_retries: Maximum number of retries for failed batches
        """
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.processing_stats: Dict[str, Any] = {}
        
    def process_batches(self, 
                       data: List[Any], 
                       processor_func: Callable[[List[Any]], Dict[str, Any]],
                       batch_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Process data in batches.
        
        Args:
            data: List of data items to process
            processor_func: Function to process each batch
            batch_size: Optional batch size override
            
        Returns:
            Dict[str, Any]: Processing results
        """
        if batch_size is None:
            batch_size = self.batch_size
        
        logger.info(f"🚀 Processing {len(data)} items in batches of {batch_size}")
        
        results = {
            'total_items': len(data),
            'total_batches': 0,
            'successful_batches': 0,
            'failed_batches': 0,
            'total_processed': 0,
            'total_failed': 0,
            'processing_time': 0,
            'errors': []
        }
        
        start_time = time.time()
        
        # Process data in batches
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(data) + batch_size - 1) // batch_size
            
            logger.info(f"📦 Processing batch {batch_num}/{total_batches} ({len(batch)} items)")
            
            # Process batch with retries
            batch_result = self._process_batch_with_retries(batch, processor_func, batch_num)
            
            # Update results
            results['total_batches'] += 1
            results['total_processed'] += batch_result.get('processed', 0)
            results['total_failed'] += batch_result.get('failed', 0)
            
            if batch_result['success']:
                results['successful_batches'] += 1
            else:
                results['failed_batches'] += 1
                results['errors'].extend(batch_result.get('errors', []))
        
        results['processing_time'] = time.time() - start_time
        
        logger.info(f"✅ Batch processing completed:")
        logger.info(f"   📊 Total batches: {results['total_batches']}")
        logger.info(f"   ✅ Successful: {results['successful_batches']}")
        logger.info(f"   ❌ Failed: {results['failed_batches']}")
        logger.info(f"   ⏱️ Processing time: {results['processing_time']:.2f}s")
        
        return results
    
    def _process_batch_with_retries(self, 
                                   batch: List[Any], 
                                   processor_func: Callable[[List[Any]], Dict[str, Any]],
                                   batch_num: int) -> Dict[str, Any]:
        """Process a batch with retry logic."""
        for attempt in range(self.max_retries + 1):
            try:
                result = processor_func(batch)
                result['success'] = True
                result['attempt'] = attempt + 1
                return result
                
            except Exception as e:
                if attempt < self.max_retries:
                    logger.warning(f"⚠️ Batch {batch_num} failed (attempt {attempt + 1}), retrying...")
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.error(f"❌ Batch {batch_num} failed after {self.max_retries + 1} attempts: {e}")
                    return {
                        'success': False,
                        'processed': 0,
                        'failed': len(batch),
                        'errors': [{'batch': batch_num, 'error': str(e), 'attempt': attempt + 1}],
                        'attempt': attempt + 1
                    }
        
        return {
            'success': False,
            'processed': 0,
            'failed': len(batch),
            'errors': [{'batch': batch_num, 'error': 'Max retries exceeded'}],
            'attempt': self.max_retries + 1
        }
    
    def process_nodes_batch(self, nodes: List[Dict[str, Any]], loader) -> Dict[str, Any]:
        """Process a batch of nodes."""
        try:
            result = loader.load_nodes(nodes, batch_size=len(nodes))
            return {
                'processed': result.get('loaded_nodes', 0),
                'failed': result.get('failed_nodes', 0),
                'errors': result.get('errors', [])
            }
        except Exception as e:
            return {
                'processed': 0,
                'failed': len(nodes),
                'errors': [{'error': str(e)}]
            }
    
    def process_relationships_batch(self, relationships: List[Dict[str, Any]], loader) -> Dict[str, Any]:
        """Process a batch of relationships."""
        try:
            result = loader.load_relationships(relationships, batch_size=len(relationships))
            return {
                'processed': result.get('loaded_relationships', 0),
                'failed': result.get('failed_relationships', 0),
                'errors': result.get('errors', [])
            }
        except Exception as e:
            return {
                'processed': 0,
                'failed': len(relationships),
                'errors': [{'error': str(e)}]
            }
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return self.processing_stats.copy()
    
    def update_processing_stats(self, stats: Dict[str, Any]) -> None:
        """Update processing statistics."""
        self.processing_stats.update(stats)
        self.processing_stats['last_updated'] = datetime.utcnow().isoformat()
