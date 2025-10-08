"""
Watermark Manager

This module provides watermark management capabilities for incremental processing in the PBF-LB/M data pipeline.
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

class WatermarkManager:
    """
    Manages processing watermarks for incremental data processing.
    """
    
    def __init__(self):
        self.config = get_pipeline_config()
        self.watermark_config = self._load_watermark_config()
        self.watermarks = {}
        self.watermark_file = self._get_watermark_file_path()
        self._load_watermarks()
    
    def _load_watermark_config(self) -> Dict[str, Any]:
        """Load watermark configuration."""
        try:
            return self.config.get('watermarks', {
                'default_watermark_delay': 30,  # seconds
                'max_watermark_delay': 300,     # 5 minutes
                'watermark_update_interval': 60, # 1 minute
                'persistence_enabled': True,
                'streams': {
                    'pbf_process_stream': {
                        'watermark_delay': 30,
                        'enabled': True
                    },
                    'ispm_monitoring_stream': {
                        'watermark_delay': 15,
                        'enabled': True
                    },
                    'ct_scan_stream': {
                        'watermark_delay': 60,
                        'enabled': True
                    },
                    'powder_bed_stream': {
                        'watermark_delay': 45,
                        'enabled': True
                    }
                }
            })
        except Exception as e:
            logger.error(f"Error loading watermark configuration: {e}")
            return {}
    
    def _get_watermark_file_path(self) -> str:
        """Get watermark file path."""
        try:
            base_dir = self.config.get('data_directory', './data')
            watermark_dir = os.path.join(base_dir, 'watermarks')
            os.makedirs(watermark_dir, exist_ok=True)
            return os.path.join(watermark_dir, 'watermarks.json')
        except Exception as e:
            logger.error(f"Error getting watermark file path: {e}")
            return './watermarks.json'
    
    def _load_watermarks(self) -> None:
        """Load watermarks from persistent storage."""
        try:
            if os.path.exists(self.watermark_file):
                with open(self.watermark_file, 'r') as f:
                    self.watermarks = json.load(f)
                logger.info(f"Loaded {len(self.watermarks)} watermarks from {self.watermark_file}")
            else:
                logger.info("No existing watermarks found, starting fresh")
        except Exception as e:
            logger.error(f"Error loading watermarks: {e}")
            self.watermarks = {}
    
    def _save_watermarks(self) -> None:
        """Save watermarks to persistent storage."""
        try:
            if self.watermark_config.get('persistence_enabled', True):
                with open(self.watermark_file, 'w') as f:
                    json.dump(self.watermarks, f, indent=2, default=str)
                logger.debug("Watermarks saved to persistent storage")
        except Exception as e:
            logger.error(f"Error saving watermarks: {e}")
    
    def get_watermark(self, stream_name: str) -> Optional[datetime]:
        """Get current watermark for a stream."""
        try:
            watermark_data = self.watermarks.get(stream_name)
            if watermark_data:
                timestamp_str = watermark_data.get('timestamp')
                if timestamp_str:
                    return datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            return None
        except Exception as e:
            logger.error(f"Error getting watermark for {stream_name}: {e}")
            return None
    
    def update_watermark(self, stream_name: str, timestamp: Union[datetime, str], 
                        metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Update watermark for a stream."""
        try:
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            
            # Validate timestamp
            if not self._validate_watermark_timestamp(stream_name, timestamp):
                logger.warning(f"Invalid watermark timestamp for {stream_name}: {timestamp}")
                return False
            
            # Update watermark
            self.watermarks[stream_name] = {
                'timestamp': timestamp.isoformat(),
                'updated_at': datetime.now().isoformat(),
                'metadata': metadata or {}
            }
            
            # Save to persistent storage
            self._save_watermarks()
            
            logger.info(f"Watermark updated for {stream_name}: {timestamp}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating watermark for {stream_name}: {e}")
            return False
    
    def _validate_watermark_timestamp(self, stream_name: str, timestamp: datetime) -> bool:
        """Validate watermark timestamp."""
        try:
            # Check if stream is enabled
            stream_config = self.watermark_config.get('streams', {}).get(stream_name, {})
            if not stream_config.get('enabled', True):
                return False
            
            # Check if timestamp is not too far in the future
            max_delay = self.watermark_config.get('max_watermark_delay', 300)
            max_future_time = datetime.now() + timedelta(seconds=max_delay)
            
            if timestamp > max_future_time:
                logger.warning(f"Watermark timestamp too far in future: {timestamp}")
                return False
            
            # Check if timestamp is not too far in the past
            current_watermark = self.get_watermark(stream_name)
            if current_watermark and timestamp < current_watermark:
                logger.warning(f"Watermark timestamp is older than current watermark: {timestamp}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating watermark timestamp: {e}")
            return False
    
    def get_watermark_delay(self, stream_name: str) -> int:
        """Get watermark delay for a stream."""
        try:
            stream_config = self.watermark_config.get('streams', {}).get(stream_name, {})
            return stream_config.get('watermark_delay', 
                                   self.watermark_config.get('default_watermark_delay', 30))
        except Exception as e:
            logger.error(f"Error getting watermark delay: {e}")
            return 30
    
    def calculate_processing_watermark(self, stream_name: str) -> Optional[datetime]:
        """Calculate processing watermark for a stream."""
        try:
            current_watermark = self.get_watermark(stream_name)
            if not current_watermark:
                return None
            
            delay_seconds = self.get_watermark_delay(stream_name)
            processing_watermark = current_watermark - timedelta(seconds=delay_seconds)
            
            return processing_watermark
            
        except Exception as e:
            logger.error(f"Error calculating processing watermark: {e}")
            return None
    
    def get_late_data_threshold(self, stream_name: str) -> Optional[datetime]:
        """Get late data threshold for a stream."""
        try:
            current_watermark = self.get_watermark(stream_name)
            if not current_watermark:
                return None
            
            # Late data threshold is typically 2x the watermark delay
            delay_seconds = self.get_watermark_delay(stream_name)
            late_threshold = current_watermark - timedelta(seconds=delay_seconds * 2)
            
            return late_threshold
            
        except Exception as e:
            logger.error(f"Error getting late data threshold: {e}")
            return None
    
    def is_late_data(self, stream_name: str, data_timestamp: Union[datetime, str]) -> bool:
        """Check if data is late based on watermark."""
        try:
            if isinstance(data_timestamp, str):
                data_timestamp = datetime.fromisoformat(data_timestamp.replace('Z', '+00:00'))
            
            late_threshold = self.get_late_data_threshold(stream_name)
            if not late_threshold:
                return False
            
            return data_timestamp < late_threshold
            
        except Exception as e:
            logger.error(f"Error checking if data is late: {e}")
            return False
    
    def get_watermark_statistics(self) -> Dict[str, Any]:
        """Get watermark statistics."""
        try:
            stats = {
                'total_streams': len(self.watermarks),
                'enabled_streams': 0,
                'stream_details': {},
                'configuration': self.watermark_config.copy()
            }
            
            for stream_name, watermark_data in self.watermarks.items():
                stream_config = self.watermark_config.get('streams', {}).get(stream_name, {})
                if stream_config.get('enabled', True):
                    stats['enabled_streams'] += 1
                
                stats['stream_details'][stream_name] = {
                    'watermark_timestamp': watermark_data.get('timestamp'),
                    'updated_at': watermark_data.get('updated_at'),
                    'enabled': stream_config.get('enabled', True),
                    'watermark_delay': self.get_watermark_delay(stream_name),
                    'processing_watermark': self.calculate_processing_watermark(stream_name),
                    'late_data_threshold': self.get_late_data_threshold(stream_name)
                }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting watermark statistics: {e}")
            return {}
    
    def cleanup_old_watermarks(self, retention_days: int = 30) -> Dict[str, Any]:
        """Clean up old watermarks."""
        try:
            cutoff_time = datetime.now() - timedelta(days=retention_days)
            cleaned_count = 0
            
            for stream_name, watermark_data in list(self.watermarks.items()):
                updated_at_str = watermark_data.get('updated_at', '')
                if updated_at_str:
                    updated_at = datetime.fromisoformat(updated_at_str.replace('Z', '+00:00'))
                    if updated_at < cutoff_time:
                        del self.watermarks[stream_name]
                        cleaned_count += 1
            
            if cleaned_count > 0:
                self._save_watermarks()
            
            result = {
                'status': 'success',
                'cleaned_count': cleaned_count,
                'retention_days': retention_days,
                'cleaned_at': datetime.now().isoformat()
            }
            
            logger.info(f"Watermark cleanup completed: {cleaned_count} old watermarks removed")
            return result
            
        except Exception as e:
            logger.error(f"Error cleaning up old watermarks: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def reset_watermark(self, stream_name: str) -> bool:
        """Reset watermark for a stream."""
        try:
            if stream_name in self.watermarks:
                del self.watermarks[stream_name]
                self._save_watermarks()
                logger.info(f"Watermark reset for {stream_name}")
                return True
            else:
                logger.warning(f"No watermark found for {stream_name}")
                return False
                
        except Exception as e:
            logger.error(f"Error resetting watermark for {stream_name}: {e}")
            return False
    
    def get_all_watermarks(self) -> Dict[str, Any]:
        """Get all watermarks."""
        try:
            return self.watermarks.copy()
        except Exception as e:
            logger.error(f"Error getting all watermarks: {e}")
            return {}
    
    def set_watermark_batch(self, watermark_updates: Dict[str, Union[datetime, str]]) -> Dict[str, bool]:
        """Set multiple watermarks in batch."""
        try:
            results = {}
            
            for stream_name, timestamp in watermark_updates.items():
                try:
                    success = self.update_watermark(stream_name, timestamp)
                    results[stream_name] = success
                except Exception as e:
                    logger.error(f"Error updating watermark for {stream_name}: {e}")
                    results[stream_name] = False
            
            return results
            
        except Exception as e:
            logger.error(f"Error setting watermark batch: {e}")
            return {}
