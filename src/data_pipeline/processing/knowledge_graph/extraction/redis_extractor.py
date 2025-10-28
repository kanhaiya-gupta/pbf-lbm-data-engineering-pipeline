"""
Redis Data Extractor for Knowledge Graph

This module extracts cache data from Redis to build knowledge graph nodes
and relationships for PBF-LB/M manufacturing processes.
"""

import sys
import os
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
from src.data_pipeline.processing.knowledge_graph.utils.json_parser import safe_json_loads_with_fallback

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from src.data_pipeline.config.redis_config import get_redis_config
from src.data_pipeline.storage.operational.redis_client import RedisClient

logger = logging.getLogger(__name__)


class RedisExtractor:
    """
    Extracts cache data from Redis for knowledge graph construction.
    
    Focuses on PBF-LB/M manufacturing cache entities:
    - Process data cache, machine status cache, sensor readings cache
    - Analytics cache, job queue items, user sessions
    """
    
    def __init__(self):
        """Initialize Redis extractor."""
        self.config = get_redis_config()
        # Convert config to dictionary for RedisClient
        config_dict = {
            'host': self.config.host,
            'port': self.config.port,
            'db': self.config.db,
            'password': self.config.password,
            'max_connections': self.config.max_connections,
            'default_ttl': self.config.default_ttl
        }
        self.client = RedisClient(config=config_dict)
        self.connected = False
        
    def connect(self) -> bool:
        """Connect to Redis database."""
        try:
            if self.client.connect():
                self.connected = True
                logger.info("✅ Connected to Redis for knowledge graph extraction")
                return True
            else:
                logger.error("❌ Failed to connect to Redis")
                return False
        except Exception as e:
            logger.error(f"❌ Redis connection error: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from Redis."""
        if self.connected:
            self.client.disconnect()
            self.connected = False
            logger.info("Disconnected from Redis")
    
    def extract_process_cache(self) -> List[Dict[str, Any]]:
        """
        Extract process cache data for knowledge graph nodes.
        
        Returns:
            List[Dict[str, Any]]: Process cache data for graph nodes
        """
        try:
            if not self.connected:
                raise RuntimeError("Not connected to Redis")
            
            # Get all process cache keys
            process_keys = self.client.get_keys("process:*")
            
            process_cache = []
            for key in process_keys:
                try:
                    # Get the cached data
                    cached_data = self.client.get_hash(key)
                    if cached_data:
                        # Parse the cached data
                        if isinstance(cached_data, str):
                            data = safe_json_loads_with_fallback(cached_data, 'cached_data', 5000, {})
                        else:
                            data = cached_data
                        
                        cache_data = {
                            'node_type': 'ProcessCache',
                            'cache_key': key,
                            'process_id': data.get('process_id'),
                            'status': data.get('status'),
                            'quality_score': data.get('quality_score'),
                            'last_updated': data.get('last_updated'),
                            'metadata': data.get('metadata', {}),
                            'extraction_timestamp': datetime.utcnow().isoformat()
                        }
                        process_cache.append(cache_data)
                        
                except Exception as e:
                    logger.warning(f"Failed to process cache key {key}: {e}")
                    continue
            
            logger.info(f"📊 Extracted {len(process_cache)} process cache entries from Redis")
            return process_cache
            
        except Exception as e:
            logger.error(f"❌ Failed to extract process cache: {e}")
            return []
    
    def extract_machine_status_cache(self) -> List[Dict[str, Any]]:
        """
        Extract machine status cache data for knowledge graph nodes.
        
        Returns:
            List[Dict[str, Any]]: Machine status cache data for graph nodes
        """
        try:
            if not self.connected:
                raise RuntimeError("Not connected to Redis")
            
            # Get all machine status cache keys
            machine_keys = self.client.get_keys("machine:*")
            
            machine_cache = []
            for key in machine_keys:
                try:
                    # Get the cached data
                    cached_data = self.client.get_hash(key)
                    if cached_data:
                        # Parse the cached data
                        if isinstance(cached_data, str):
                            data = safe_json_loads_with_fallback(cached_data, 'cached_data', 5000, {})
                        else:
                            data = cached_data
                        
                        cache_data = {
                            'node_type': 'MachineStatusCache',
                            'cache_key': key,
                            'machine_id': data.get('machine_id'),
                            'status': data.get('status'),
                            'operational_hours': data.get('operational_hours'),
                            'temperature': data.get('temperature'),
                            'last_maintenance': data.get('last_maintenance'),
                            'performance_metrics': data.get('performance_metrics', {}),
                            'extraction_timestamp': datetime.utcnow().isoformat()
                        }
                        machine_cache.append(cache_data)
                        
                except Exception as e:
                    logger.warning(f"Failed to process cache key {key}: {e}")
                    continue
            
            logger.info(f"📊 Extracted {len(machine_cache)} machine status cache entries from Redis")
            return machine_cache
            
        except Exception as e:
            logger.error(f"❌ Failed to extract machine status cache: {e}")
            return []
    
    def extract_sensor_readings_cache(self) -> List[Dict[str, Any]]:
        """
        Extract sensor readings cache data for knowledge graph nodes.
        
        Returns:
            List[Dict[str, Any]]: Sensor readings cache data for graph nodes
        """
        try:
            if not self.connected:
                raise RuntimeError("Not connected to Redis")
            
            # Get all sensor readings cache keys
            sensor_keys = self.client.get_keys("sensor:*")
            
            sensor_cache = []
            for key in sensor_keys:
                try:
                    # Get the cached data
                    cached_data = self.client.get_hash(key)
                    if cached_data:
                        # Parse the cached data
                        if isinstance(cached_data, str):
                            data = safe_json_loads_with_fallback(cached_data, 'cached_data', 5000, {})
                        else:
                            data = cached_data
                        
                        cache_data = {
                            'node_type': 'SensorReadingCache',
                            'cache_key': key,
                            'sensor_id': data.get('sensor_id'),
                            'sensor_type': data.get('sensor_type'),
                            'value': data.get('value'),
                            'unit': data.get('unit'),
                            'quality_score': data.get('quality_score'),
                            'timestamp': data.get('timestamp'),
                            'process_id': data.get('process_id'),
                            'extraction_timestamp': datetime.utcnow().isoformat()
                        }
                        sensor_cache.append(cache_data)
                        
                except Exception as e:
                    logger.warning(f"Failed to process cache key {key}: {e}")
                    continue
            
            logger.info(f"📊 Extracted {len(sensor_cache)} sensor readings cache entries from Redis")
            return sensor_cache
            
        except Exception as e:
            logger.error(f"❌ Failed to extract sensor readings cache: {e}")
            return []
    
    def extract_analytics_cache(self) -> List[Dict[str, Any]]:
        """
        Extract analytics cache data for knowledge graph nodes.
        
        Returns:
            List[Dict[str, Any]]: Analytics cache data for graph nodes
        """
        try:
            if not self.connected:
                raise RuntimeError("Not connected to Redis")
            
            # Get all analytics cache keys
            analytics_keys = self.client.get_keys("analytics:*")
            
            analytics_cache = []
            for key in analytics_keys:
                try:
                    # Get the cached data
                    cached_data = self.client.get_hash(key)
                    if cached_data:
                        # Parse the cached data
                        if isinstance(cached_data, str):
                            data = safe_json_loads_with_fallback(cached_data, 'cached_data', 5000, {})
                        else:
                            data = cached_data
                        
                        cache_data = {
                            'node_type': 'AnalyticsCache',
                            'cache_key': key,
                            'analytics_type': data.get('analytics_type'),
                            'process_id': data.get('process_id'),
                            'statistics': data.get('statistics', {}),
                            'percentiles': data.get('percentiles', {}),
                            'quality_metrics': data.get('quality_metrics', {}),
                            'generated_at': data.get('generated_at'),
                            'extraction_timestamp': datetime.utcnow().isoformat()
                        }
                        analytics_cache.append(cache_data)
                        
                except Exception as e:
                    logger.warning(f"Failed to process cache key {key}: {e}")
                    continue
            
            logger.info(f"📊 Extracted {len(analytics_cache)} analytics cache entries from Redis")
            return analytics_cache
            
        except Exception as e:
            logger.error(f"❌ Failed to extract analytics cache: {e}")
            return []
    
    def extract_job_queue_items(self) -> List[Dict[str, Any]]:
        """
        Extract job queue items for knowledge graph nodes.
        
        Returns:
            List[Dict[str, Any]]: Job queue item data for graph nodes
        """
        try:
            if not self.connected:
                raise RuntimeError("Not connected to Redis")
            
            # Get all job queue keys
            job_keys = self.client.get_keys("job:*")
            
            job_items = []
            for key in job_keys:
                try:
                    # Get the cached data
                    cached_data = self.client.get_hash(key)
                    if cached_data:
                        # Parse the cached data
                        if isinstance(cached_data, str):
                            data = safe_json_loads_with_fallback(cached_data, 'cached_data', 5000, {})
                        else:
                            data = cached_data
                        
                        job_data = {
                            'node_type': 'JobQueueItem',
                            'cache_key': key,
                            'job_id': data.get('job_id'),
                            'job_type': data.get('job_type'),
                            'status': data.get('status'),
                            'priority': data.get('priority'),
                            'process_id': data.get('process_id'),
                            'machine_id': data.get('machine_id'),
                            'created_at': data.get('created_at'),
                            'scheduled_at': data.get('scheduled_at'),
                            'extraction_timestamp': datetime.utcnow().isoformat()
                        }
                        job_items.append(job_data)
                        
                except Exception as e:
                    logger.warning(f"Failed to process cache key {key}: {e}")
                    continue
            
            logger.info(f"📊 Extracted {len(job_items)} job queue items from Redis")
            return job_items
            
        except Exception as e:
            logger.error(f"❌ Failed to extract job queue items: {e}")
            return []
    
    def extract_user_sessions(self) -> List[Dict[str, Any]]:
        """
        Extract user session data for knowledge graph nodes.
        
        Returns:
            List[Dict[str, Any]]: User session data for graph nodes
        """
        try:
            if not self.connected:
                raise RuntimeError("Not connected to Redis")
            
            # Get all user session keys
            session_keys = self.client.get_keys("session:*")
            
            user_sessions = []
            failed_keys = 0
            for key in session_keys:
                try:
                    # Get the cached data
                    cached_data = self.client.get_hash(key)
                    if cached_data:
                        # Parse the cached data
                        if isinstance(cached_data, str):
                            data = safe_json_loads_with_fallback(cached_data, 'cached_data', 5000, {})
                        else:
                            data = cached_data
                        
                        session_data = {
                            'node_type': 'UserSession',
                            'cache_key': key,
                            'session_id': data.get('session_id'),
                            'user_id': data.get('user_id'),
                            'username': data.get('username'),
                            'role': data.get('role'),
                            'status': data.get('status'),
                            'last_activity': data.get('last_activity'),
                            'ip_address': data.get('ip_address'),
                            'extraction_timestamp': datetime.utcnow().isoformat()
                        }
                        user_sessions.append(session_data)
                        
                except Exception as e:
                    # Count failed keys but don't log each one individually
                    failed_keys += 1
                    continue
            
            # Log summary of failed keys if any
            if failed_keys > 0:
                logger.warning(f"Failed to process {failed_keys} session keys (wrong data type)")
            
            logger.info(f"📊 Extracted {len(user_sessions)} user sessions from Redis")
            return user_sessions
            
        except Exception as e:
            logger.error(f"❌ Failed to extract user sessions: {e}")
            return []
    
    def extract_all_data(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract all data from Redis for knowledge graph construction.
        
        Returns:
            Dict[str, List[Dict[str, Any]]]: All extracted data organized by type
        """
        logger.info("🚀 Starting comprehensive Redis data extraction...")
        
        if not self.connected:
            if not self.connect():
                return {}
        
        try:
            extracted_data = {
                'process_cache': self.extract_process_cache(),
                'machine_status_cache': self.extract_machine_status_cache(),
                'sensor_readings_cache': self.extract_sensor_readings_cache(),
                'analytics_cache': self.extract_analytics_cache(),
                'job_queue_items': self.extract_job_queue_items(),
                'user_sessions': self.extract_user_sessions()
            }
            
            # Calculate totals
            total_entries = sum(len(data) for data in extracted_data.values())
            
            logger.info(f"✅ Redis extraction completed:")
            logger.info(f"   📊 Total Cache Entries: {total_entries}")
            logger.info(f"   🔄 Process Cache: {len(extracted_data['process_cache'])}")
            logger.info(f"   🏭 Machine Status Cache: {len(extracted_data['machine_status_cache'])}")
            logger.info(f"   📡 Sensor Readings Cache: {len(extracted_data['sensor_readings_cache'])}")
            logger.info(f"   📈 Analytics Cache: {len(extracted_data['analytics_cache'])}")
            logger.info(f"   📋 Job Queue Items: {len(extracted_data['job_queue_items'])}")
            logger.info(f"   👤 User Sessions: {len(extracted_data['user_sessions'])}")
            
            return extracted_data
            
        except Exception as e:
            logger.error(f"❌ Failed to extract all Redis data: {e}")
            return {}
    
    def get_extraction_summary(self) -> Dict[str, Any]:
        """
        Get summary of available data for extraction.
        
        Returns:
            Dict[str, Any]: Extraction summary
        """
        try:
            if not self.connected:
                if not self.connect():
                    return {}
            
            # Get key counts by pattern
            patterns = ['process:*', 'machine:*', 'sensor:*', 'analytics:*', 'job:*', 'session:*']
            counts = {}
            
            for pattern in patterns:
                try:
                    keys = self.client.get_keys(pattern)
                    counts[pattern] = len(keys)
                except Exception as e:
                    logger.warning(f"Could not get keys for pattern {pattern}: {e}")
                    counts[pattern] = 0
            
            summary = {
                'database': 'Redis',
                'connection_status': 'Connected' if self.connected else 'Disconnected',
                'key_patterns': counts,
                'total_keys': sum(counts.values()),
                'extraction_timestamp': datetime.utcnow().isoformat()
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"❌ Failed to get extraction summary: {e}")
            return {}
    
    def close_connection(self) -> None:
        """Close the Redis connection."""
        try:
            if self.client and self.client.connected:
                self.client.close_connection()
                self.connected = False
                logger.info("✅ Redis connection closed")
        except Exception as e:
            logger.error(f"❌ Failed to close Redis connection: {e}")
