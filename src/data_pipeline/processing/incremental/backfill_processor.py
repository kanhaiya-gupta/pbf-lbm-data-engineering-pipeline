"""
Backfill Processor

This module provides backfill processing capabilities for handling late-arriving data in the PBF-LB/M data pipeline.
"""

import pandas as pd
from typing import Dict, List, Optional, Any, Union, Tuple
import logging
from datetime import datetime, timedelta
import json
import os

from src.data_pipeline.config.pipeline_config import get_pipeline_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BackfillProcessor:
    """
    Backfill processor for handling late-arriving data.
    """
    
    def __init__(self):
        self.config = get_pipeline_config()
        self.backfill_config = self._load_backfill_config()
        self.backfill_queue = []
        self.backfill_history = {}
        self.backfill_file = self._get_backfill_file_path()
        self._load_backfill_queue()
    
    def _load_backfill_config(self) -> Dict[str, Any]:
        """Load backfill configuration."""
        try:
            return self.config.get('backfill', {
                'max_late_arrival_hours': 24,
                'backfill_batch_size': 1000,
                'backfill_interval': 300,  # 5 minutes
                'max_retry_attempts': 3,
                'retry_delay_seconds': 60,
                'enabled': True,
                'streams': {
                    'pbf_process_stream': {
                        'enabled': True,
                        'max_late_arrival_hours': 24,
                        'priority': 'high'
                    },
                    'ispm_monitoring_stream': {
                        'enabled': True,
                        'max_late_arrival_hours': 12,
                        'priority': 'high'
                    },
                    'ct_scan_stream': {
                        'enabled': True,
                        'max_late_arrival_hours': 48,
                        'priority': 'medium'
                    },
                    'powder_bed_stream': {
                        'enabled': True,
                        'max_late_arrival_hours': 36,
                        'priority': 'medium'
                    }
                }
            })
        except Exception as e:
            logger.error(f"Error loading backfill configuration: {e}")
            return {}
    
    def _get_backfill_file_path(self) -> str:
        """Get backfill queue file path."""
        try:
            base_dir = self.config.get('data_directory', './data')
            backfill_dir = os.path.join(base_dir, 'backfill')
            os.makedirs(backfill_dir, exist_ok=True)
            return os.path.join(backfill_dir, 'backfill_queue.json')
        except Exception as e:
            logger.error(f"Error getting backfill file path: {e}")
            return './backfill_queue.json'
    
    def _load_backfill_queue(self) -> None:
        """Load backfill queue from persistent storage."""
        try:
            if os.path.exists(self.backfill_file):
                with open(self.backfill_file, 'r') as f:
                    data = json.load(f)
                    self.backfill_queue = data.get('queue', [])
                    self.backfill_history = data.get('history', {})
                logger.info(f"Loaded {len(self.backfill_queue)} items from backfill queue")
            else:
                logger.info("No existing backfill queue found, starting fresh")
        except Exception as e:
            logger.error(f"Error loading backfill queue: {e}")
            self.backfill_queue = []
            self.backfill_history = {}
    
    def _save_backfill_queue(self) -> None:
        """Save backfill queue to persistent storage."""
        try:
            data = {
                'queue': self.backfill_queue,
                'history': self.backfill_history,
                'last_updated': datetime.now().isoformat()
            }
            
            with open(self.backfill_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            logger.debug("Backfill queue saved to persistent storage")
        except Exception as e:
            logger.error(f"Error saving backfill queue: {e}")
    
    def add_late_data(self, stream_name: str, data: Dict[str, Any], 
                     data_timestamp: Union[datetime, str]) -> bool:
        """Add late-arriving data to backfill queue."""
        try:
            if not self._is_backfill_enabled(stream_name):
                logger.warning(f"Backfill disabled for stream: {stream_name}")
                return False
            
            if isinstance(data_timestamp, str):
                data_timestamp = datetime.fromisoformat(data_timestamp.replace('Z', '+00:00'))
            
            # Check if data is within backfill window
            if not self._is_within_backfill_window(stream_name, data_timestamp):
                logger.warning(f"Data outside backfill window for {stream_name}: {data_timestamp}")
                return False
            
            # Create backfill item
            backfill_item = {
                'id': self._generate_backfill_id(),
                'stream_name': stream_name,
                'data': data,
                'data_timestamp': data_timestamp.isoformat(),
                'added_at': datetime.now().isoformat(),
                'priority': self._get_stream_priority(stream_name),
                'retry_count': 0,
                'status': 'pending'
            }
            
            # Add to queue
            self.backfill_queue.append(backfill_item)
            
            # Sort by priority and timestamp
            self._sort_backfill_queue()
            
            # Save queue
            self._save_backfill_queue()
            
            logger.info(f"Late data added to backfill queue: {stream_name}, ID: {backfill_item['id']}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding late data to backfill queue: {e}")
            return False
    
    def process_backfill_queue(self) -> Dict[str, Any]:
        """Process backfill queue."""
        try:
            if not self.backfill_config.get('enabled', True):
                return {'status': 'disabled', 'processed_count': 0}
            
            processed_count = 0
            failed_count = 0
            skipped_count = 0
            
            # Process items in queue
            for item in self.backfill_queue[:]:
                try:
                    if item['status'] == 'pending':
                        result = self._process_backfill_item(item)
                        if result['success']:
                            processed_count += 1
                            item['status'] = 'completed'
                            item['completed_at'] = datetime.now().isoformat()
                        else:
                            failed_count += 1
                            item['retry_count'] += 1
                            
                            if item['retry_count'] >= self.backfill_config.get('max_retry_attempts', 3):
                                item['status'] = 'failed'
                                item['failed_at'] = datetime.now().isoformat()
                            else:
                                item['status'] = 'retry'
                                item['next_retry_at'] = (
                                    datetime.now() + 
                                    timedelta(seconds=self.backfill_config.get('retry_delay_seconds', 60))
                                ).isoformat()
                    
                    elif item['status'] == 'retry':
                        if self._should_retry_item(item):
                            result = self._process_backfill_item(item)
                            if result['success']:
                                processed_count += 1
                                item['status'] = 'completed'
                                item['completed_at'] = datetime.now().isoformat()
                            else:
                                item['retry_count'] += 1
                                if item['retry_count'] >= self.backfill_config.get('max_retry_attempts', 3):
                                    item['status'] = 'failed'
                                    item['failed_at'] = datetime.now().isoformat()
                    
                    elif item['status'] in ['completed', 'failed']:
                        skipped_count += 1
                        
                except Exception as e:
                    logger.error(f"Error processing backfill item {item.get('id', 'unknown')}: {e}")
                    failed_count += 1
                    item['status'] = 'failed'
                    item['failed_at'] = datetime.now().isoformat()
            
            # Remove completed and failed items
            self.backfill_queue = [item for item in self.backfill_queue 
                                 if item['status'] not in ['completed', 'failed']]
            
            # Update history
            self._update_backfill_history(processed_count, failed_count, skipped_count)
            
            # Save queue
            self._save_backfill_queue()
            
            result = {
                'status': 'success',
                'processed_count': processed_count,
                'failed_count': failed_count,
                'skipped_count': skipped_count,
                'queue_size': len(self.backfill_queue),
                'processed_at': datetime.now().isoformat()
            }
            
            logger.info(f"Backfill queue processed: {processed_count} successful, {failed_count} failed")
            return result
            
        except Exception as e:
            logger.error(f"Error processing backfill queue: {e}")
            return {'status': 'error', 'error': str(e), 'processed_count': 0}
    
    def _process_backfill_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single backfill item."""
        try:
            stream_name = item['stream_name']
            data = item['data']
            data_timestamp = item['data_timestamp']
            
            # Process based on stream type
            if stream_name == 'pbf_process_stream':
                result = self._process_pbf_process_backfill(data, data_timestamp)
            elif stream_name == 'ispm_monitoring_stream':
                result = self._process_ispm_monitoring_backfill(data, data_timestamp)
            elif stream_name == 'ct_scan_stream':
                result = self._process_ct_scan_backfill(data, data_timestamp)
            elif stream_name == 'powder_bed_stream':
                result = self._process_powder_bed_backfill(data, data_timestamp)
            else:
                result = {'success': False, 'error': f'Unknown stream: {stream_name}'}
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing backfill item: {e}")
            return {'success': False, 'error': str(e)}
    
    def _process_pbf_process_backfill(self, data: Dict[str, Any], timestamp: str) -> Dict[str, Any]:
        """Process PBF process backfill data."""
        try:
            # Add backfill metadata
            data['backfill_processed'] = True
            data['backfill_timestamp'] = timestamp
            data['backfill_processed_at'] = datetime.now().isoformat()
            
            # Process the data (implement actual processing logic)
            # This would typically involve writing to the appropriate storage system
            
            return {'success': True, 'message': 'PBF process backfill processed successfully'}
            
        except Exception as e:
            logger.error(f"Error processing PBF process backfill: {e}")
            return {'success': False, 'error': str(e)}
    
    def _process_ispm_monitoring_backfill(self, data: Dict[str, Any], timestamp: str) -> Dict[str, Any]:
        """Process ISPM monitoring backfill data."""
        try:
            # Add backfill metadata
            data['backfill_processed'] = True
            data['backfill_timestamp'] = timestamp
            data['backfill_processed_at'] = datetime.now().isoformat()
            
            # Process the data (implement actual processing logic)
            
            return {'success': True, 'message': 'ISPM monitoring backfill processed successfully'}
            
        except Exception as e:
            logger.error(f"Error processing ISPM monitoring backfill: {e}")
            return {'success': False, 'error': str(e)}
    
    def _process_ct_scan_backfill(self, data: Dict[str, Any], timestamp: str) -> Dict[str, Any]:
        """Process CT scan backfill data."""
        try:
            # Add backfill metadata
            data['backfill_processed'] = True
            data['backfill_timestamp'] = timestamp
            data['backfill_processed_at'] = datetime.now().isoformat()
            
            # Process the data (implement actual processing logic)
            
            return {'success': True, 'message': 'CT scan backfill processed successfully'}
            
        except Exception as e:
            logger.error(f"Error processing CT scan backfill: {e}")
            return {'success': False, 'error': str(e)}
    
    def _process_powder_bed_backfill(self, data: Dict[str, Any], timestamp: str) -> Dict[str, Any]:
        """Process powder bed backfill data."""
        try:
            # Add backfill metadata
            data['backfill_processed'] = True
            data['backfill_timestamp'] = timestamp
            data['backfill_processed_at'] = datetime.now().isoformat()
            
            # Process the data (implement actual processing logic)
            
            return {'success': True, 'message': 'Powder bed backfill processed successfully'}
            
        except Exception as e:
            logger.error(f"Error processing powder bed backfill: {e}")
            return {'success': False, 'error': str(e)}
    
    def _is_backfill_enabled(self, stream_name: str) -> bool:
        """Check if backfill is enabled for a stream."""
        try:
            if not self.backfill_config.get('enabled', True):
                return False
            
            stream_config = self.backfill_config.get('streams', {}).get(stream_name, {})
            return stream_config.get('enabled', False)
        except Exception as e:
            logger.error(f"Error checking backfill enablement: {e}")
            return False
    
    def _is_within_backfill_window(self, stream_name: str, data_timestamp: datetime) -> bool:
        """Check if data is within backfill window."""
        try:
            stream_config = self.backfill_config.get('streams', {}).get(stream_name, {})
            max_hours = stream_config.get('max_late_arrival_hours', 
                                        self.backfill_config.get('max_late_arrival_hours', 24))
            
            cutoff_time = datetime.now() - timedelta(hours=max_hours)
            return data_timestamp >= cutoff_time
            
        except Exception as e:
            logger.error(f"Error checking backfill window: {e}")
            return False
    
    def _get_stream_priority(self, stream_name: str) -> str:
        """Get priority for a stream."""
        try:
            stream_config = self.backfill_config.get('streams', {}).get(stream_name, {})
            return stream_config.get('priority', 'medium')
        except Exception as e:
            logger.error(f"Error getting stream priority: {e}")
            return 'medium'
    
    def _generate_backfill_id(self) -> str:
        """Generate unique backfill ID."""
        try:
            return f"backfill_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.backfill_queue)}"
        except Exception as e:
            logger.error(f"Error generating backfill ID: {e}")
            return f"backfill_{datetime.now().isoformat()}"
    
    def _sort_backfill_queue(self) -> None:
        """Sort backfill queue by priority and timestamp."""
        try:
            priority_order = {'high': 1, 'medium': 2, 'low': 3}
            
            self.backfill_queue.sort(key=lambda x: (
                priority_order.get(x.get('priority', 'medium'), 2),
                x.get('data_timestamp', '')
            ))
        except Exception as e:
            logger.error(f"Error sorting backfill queue: {e}")
    
    def _should_retry_item(self, item: Dict[str, Any]) -> bool:
        """Check if item should be retried."""
        try:
            next_retry_at = item.get('next_retry_at')
            if not next_retry_at:
                return True
            
            retry_time = datetime.fromisoformat(next_retry_at.replace('Z', '+00:00'))
            return datetime.now() >= retry_time
            
        except Exception as e:
            logger.error(f"Error checking retry condition: {e}")
            return False
    
    def _update_backfill_history(self, processed_count: int, failed_count: int, skipped_count: int) -> None:
        """Update backfill history."""
        try:
            timestamp = datetime.now().isoformat()
            self.backfill_history[timestamp] = {
                'processed_count': processed_count,
                'failed_count': failed_count,
                'skipped_count': skipped_count,
                'queue_size': len(self.backfill_queue)
            }
            
            # Keep only last 100 history entries
            if len(self.backfill_history) > 100:
                oldest_key = min(self.backfill_history.keys())
                del self.backfill_history[oldest_key]
                
        except Exception as e:
            logger.error(f"Error updating backfill history: {e}")
    
    def get_backfill_statistics(self) -> Dict[str, Any]:
        """Get backfill statistics."""
        try:
            stats = {
                'queue_size': len(self.backfill_queue),
                'enabled': self.backfill_config.get('enabled', True),
                'configuration': self.backfill_config.copy(),
                'queue_summary': self._get_queue_summary(),
                'history_summary': self._get_history_summary()
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting backfill statistics: {e}")
            return {}
    
    def _get_queue_summary(self) -> Dict[str, Any]:
        """Get queue summary."""
        try:
            summary = {
                'total_items': len(self.backfill_queue),
                'by_status': {},
                'by_stream': {},
                'by_priority': {}
            }
            
            for item in self.backfill_queue:
                # By status
                status = item.get('status', 'unknown')
                summary['by_status'][status] = summary['by_status'].get(status, 0) + 1
                
                # By stream
                stream = item.get('stream_name', 'unknown')
                summary['by_stream'][stream] = summary['by_stream'].get(stream, 0) + 1
                
                # By priority
                priority = item.get('priority', 'unknown')
                summary['by_priority'][priority] = summary['by_priority'].get(priority, 0) + 1
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting queue summary: {e}")
            return {}
    
    def _get_history_summary(self) -> Dict[str, Any]:
        """Get history summary."""
        try:
            if not self.backfill_history:
                return {'total_runs': 0, 'total_processed': 0, 'total_failed': 0}
            
            total_runs = len(self.backfill_history)
            total_processed = sum(entry.get('processed_count', 0) for entry in self.backfill_history.values())
            total_failed = sum(entry.get('failed_count', 0) for entry in self.backfill_history.values())
            
            return {
                'total_runs': total_runs,
                'total_processed': total_processed,
                'total_failed': total_failed,
                'success_rate': (total_processed / (total_processed + total_failed)) * 100 if (total_processed + total_failed) > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error getting history summary: {e}")
            return {}
    
    def clear_backfill_queue(self) -> Dict[str, Any]:
        """Clear backfill queue."""
        try:
            cleared_count = len(self.backfill_queue)
            self.backfill_queue = []
            self._save_backfill_queue()
            
            result = {
                'status': 'success',
                'cleared_count': cleared_count,
                'cleared_at': datetime.now().isoformat()
            }
            
            logger.info(f"Backfill queue cleared: {cleared_count} items removed")
            return result
            
        except Exception as e:
            logger.error(f"Error clearing backfill queue: {e}")
            return {'status': 'error', 'error': str(e)}
