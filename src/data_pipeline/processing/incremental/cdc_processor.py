"""
CDC Processor

This module provides Change Data Capture (CDC) processing capabilities for the PBF-LB/M data pipeline.
"""

import pandas as pd
from typing import Dict, List, Optional, Any, Union
import logging
from datetime import datetime, timedelta
import json

from src.data_pipeline.config.pipeline_config import get_pipeline_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CDCProcessor:
    """
    Change Data Capture processor for incremental data processing.
    """
    
    def __init__(self):
        self.config = get_pipeline_config()
        self.cdc_config = self._load_cdc_config()
        self.change_buffer = {}
        self.last_processed_timestamps = {}
    
    def _load_cdc_config(self) -> Dict[str, Any]:
        """Load CDC configuration."""
        try:
            return self.config.get('cdc', {
                'batch_size': 1000,
                'processing_interval': 30,  # seconds
                'retention_hours': 24,
                'tables': {
                    'pbf_process_data': {
                        'primary_key': 'process_id',
                        'timestamp_column': 'updated_at',
                        'enabled': True
                    },
                    'ispm_monitoring_data': {
                        'primary_key': 'monitoring_id',
                        'timestamp_column': 'timestamp',
                        'enabled': True
                    },
                    'ct_scan_data': {
                        'primary_key': 'scan_id',
                        'timestamp_column': 'created_at',
                        'enabled': True
                    },
                    'powder_bed_data': {
                        'primary_key': 'bed_id',
                        'timestamp_column': 'timestamp',
                        'enabled': True
                    }
                }
            })
        except Exception as e:
            logger.error(f"Error loading CDC configuration: {e}")
            return {}
    
    def process_cdc_changes(self, table_name: str, changes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process CDC changes for a specific table."""
        try:
            if not self._is_table_enabled(table_name):
                logger.warning(f"CDC processing disabled for table: {table_name}")
                return {'status': 'disabled', 'processed_count': 0}
            
            table_config = self.cdc_config['tables'].get(table_name, {})
            primary_key = table_config.get('primary_key', 'id')
            timestamp_column = table_config.get('timestamp_column', 'updated_at')
            
            processed_changes = []
            for change in changes:
                try:
                    processed_change = self._process_single_change(
                        change, table_name, primary_key, timestamp_column
                    )
                    if processed_change:
                        processed_changes.append(processed_change)
                        
                except Exception as e:
                    logger.error(f"Error processing single change: {e}")
                    continue
            
            # Update last processed timestamp
            if processed_changes:
                latest_timestamp = max(
                    change.get(timestamp_column, '') for change in processed_changes
                )
                self.last_processed_timestamps[table_name] = latest_timestamp
            
            result = {
                'status': 'success',
                'table_name': table_name,
                'processed_count': len(processed_changes),
                'total_received': len(changes),
                'processed_at': datetime.now().isoformat()
            }
            
            logger.info(f"CDC changes processed for {table_name}: {len(processed_changes)}/{len(changes)}")
            return result
            
        except Exception as e:
            logger.error(f"Error processing CDC changes for {table_name}: {e}")
            return {'status': 'error', 'error': str(e), 'processed_count': 0}
    
    def _process_single_change(self, change: Dict[str, Any], table_name: str, 
                             primary_key: str, timestamp_column: str) -> Optional[Dict[str, Any]]:
        """Process a single CDC change record."""
        try:
            # Extract change metadata
            change_type = change.get('change_type', 'unknown')
            change_data = change.get('data', {})
            change_timestamp = change.get(timestamp_column, datetime.now().isoformat())
            
            # Validate change data
            if not self._validate_change_data(change_data, table_name):
                logger.warning(f"Invalid change data for {table_name}: {change_data}")
                return None
            
            # Process based on change type
            processed_change = {
                'table_name': table_name,
                'change_type': change_type,
                'primary_key': change_data.get(primary_key),
                'timestamp': change_timestamp,
                'processed_at': datetime.now().isoformat()
            }
            
            if change_type == 'INSERT':
                processed_change.update(self._process_insert_change(change_data, table_name))
            elif change_type == 'UPDATE':
                processed_change.update(self._process_update_change(change_data, table_name))
            elif change_type == 'DELETE':
                processed_change.update(self._process_delete_change(change_data, table_name))
            else:
                logger.warning(f"Unknown change type: {change_type}")
                return None
            
            # Add change metadata
            processed_change['change_metadata'] = {
                'source': change.get('source', 'unknown'),
                'transaction_id': change.get('transaction_id'),
                'sequence_number': change.get('sequence_number'),
                'schema_version': change.get('schema_version')
            }
            
            return processed_change
            
        except Exception as e:
            logger.error(f"Error processing single change: {e}")
            return None
    
    def _process_insert_change(self, data: Dict[str, Any], table_name: str) -> Dict[str, Any]:
        """Process INSERT change."""
        try:
            return {
                'operation': 'insert',
                'data': data,
                'change_summary': f"New record inserted in {table_name}",
                'impact_level': 'low'
            }
        except Exception as e:
            logger.error(f"Error processing INSERT change: {e}")
            return {'operation': 'insert', 'error': str(e)}
    
    def _process_update_change(self, data: Dict[str, Any], table_name: str) -> Dict[str, Any]:
        """Process UPDATE change."""
        try:
            # Identify changed fields
            changed_fields = []
            if 'old_data' in data and 'new_data' in data:
                old_data = data['old_data']
                new_data = data['new_data']
                
                for key in new_data:
                    if key in old_data and old_data[key] != new_data[key]:
                        changed_fields.append({
                            'field': key,
                            'old_value': old_data[key],
                            'new_value': new_data[key]
                        })
            
            return {
                'operation': 'update',
                'data': data,
                'changed_fields': changed_fields,
                'change_summary': f"Record updated in {table_name}: {len(changed_fields)} fields changed",
                'impact_level': 'medium' if len(changed_fields) > 3 else 'low'
            }
        except Exception as e:
            logger.error(f"Error processing UPDATE change: {e}")
            return {'operation': 'update', 'error': str(e)}
    
    def _process_delete_change(self, data: Dict[str, Any], table_name: str) -> Dict[str, Any]:
        """Process DELETE change."""
        try:
            return {
                'operation': 'delete',
                'data': data,
                'change_summary': f"Record deleted from {table_name}",
                'impact_level': 'high'
            }
        except Exception as e:
            logger.error(f"Error processing DELETE change: {e}")
            return {'operation': 'delete', 'error': str(e)}
    
    def _validate_change_data(self, data: Dict[str, Any], table_name: str) -> bool:
        """Validate change data."""
        try:
            if not data:
                return False
            
            # Check required fields based on table
            required_fields = self._get_required_fields(table_name)
            for field in required_fields:
                if field not in data:
                    logger.warning(f"Missing required field {field} in {table_name}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating change data: {e}")
            return False
    
    def _get_required_fields(self, table_name: str) -> List[str]:
        """Get required fields for a table."""
        try:
            required_fields_map = {
                'pbf_process_data': ['process_id', 'timestamp'],
                'ispm_monitoring_data': ['monitoring_id', 'timestamp'],
                'ct_scan_data': ['scan_id', 'created_at'],
                'powder_bed_data': ['bed_id', 'timestamp']
            }
            
            return required_fields_map.get(table_name, ['id'])
            
        except Exception as e:
            logger.error(f"Error getting required fields: {e}")
            return ['id']
    
    def _is_table_enabled(self, table_name: str) -> bool:
        """Check if CDC is enabled for a table."""
        try:
            table_config = self.cdc_config['tables'].get(table_name, {})
            return table_config.get('enabled', False)
        except Exception as e:
            logger.error(f"Error checking table enablement: {e}")
            return False
    
    def get_last_processed_timestamp(self, table_name: str) -> Optional[str]:
        """Get last processed timestamp for a table."""
        try:
            return self.last_processed_timestamps.get(table_name)
        except Exception as e:
            logger.error(f"Error getting last processed timestamp: {e}")
            return None
    
    def set_last_processed_timestamp(self, table_name: str, timestamp: str) -> None:
        """Set last processed timestamp for a table."""
        try:
            self.last_processed_timestamps[table_name] = timestamp
            logger.info(f"Last processed timestamp updated for {table_name}: {timestamp}")
        except Exception as e:
            logger.error(f"Error setting last processed timestamp: {e}")
    
    def get_cdc_statistics(self) -> Dict[str, Any]:
        """Get CDC processing statistics."""
        try:
            stats = {
                'enabled_tables': [],
                'last_processed_timestamps': self.last_processed_timestamps.copy(),
                'total_tables': len(self.cdc_config.get('tables', {})),
                'processing_config': {
                    'batch_size': self.cdc_config.get('batch_size', 1000),
                    'processing_interval': self.cdc_config.get('processing_interval', 30),
                    'retention_hours': self.cdc_config.get('retention_hours', 24)
                }
            }
            
            # Get enabled tables
            for table_name, table_config in self.cdc_config.get('tables', {}).items():
                if table_config.get('enabled', False):
                    stats['enabled_tables'].append(table_name)
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting CDC statistics: {e}")
            return {}
    
    def cleanup_old_changes(self, retention_hours: Optional[int] = None) -> Dict[str, Any]:
        """Clean up old CDC changes."""
        try:
            if retention_hours is None:
                retention_hours = self.cdc_config.get('retention_hours', 24)
            
            cutoff_time = datetime.now() - timedelta(hours=retention_hours)
            cleaned_count = 0
            
            # Clean up change buffer
            for table_name, changes in list(self.change_buffer.items()):
                filtered_changes = []
                for change in changes:
                    change_time = self._parse_timestamp(change.get('timestamp', ''))
                    if change_time and change_time > cutoff_time:
                        filtered_changes.append(change)
                    else:
                        cleaned_count += 1
                
                if filtered_changes:
                    self.change_buffer[table_name] = filtered_changes
                else:
                    del self.change_buffer[table_name]
            
            result = {
                'status': 'success',
                'cleaned_count': cleaned_count,
                'retention_hours': retention_hours,
                'cleaned_at': datetime.now().isoformat()
            }
            
            logger.info(f"CDC cleanup completed: {cleaned_count} old changes removed")
            return result
            
        except Exception as e:
            logger.error(f"Error cleaning up old changes: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _parse_timestamp(self, timestamp_str: str) -> Optional[datetime]:
        """Parse timestamp string to datetime object."""
        try:
            if isinstance(timestamp_str, str):
                return datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            return None
        except Exception as e:
            logger.error(f"Error parsing timestamp: {e}")
            return None
