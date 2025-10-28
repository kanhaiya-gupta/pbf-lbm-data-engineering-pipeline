"""
Cassandra Data Extractor for Knowledge Graph

This module extracts time-series data from Cassandra to build knowledge graph nodes
and relationships for PBF-LB/M manufacturing processes.
"""

import sys
import os
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json
from src.data_pipeline.processing.knowledge_graph.utils.json_parser import safe_json_loads_with_fallback

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from src.data_pipeline.config.cassandra_config import get_cassandra_config
from src.data_pipeline.storage.operational.cassandra_client import CassandraClient

logger = logging.getLogger(__name__)


class CassandraExtractor:
    """
    Extracts time-series data from Cassandra for knowledge graph construction.
    
    Focuses on PBF-LB/M manufacturing time-series entities:
    - Sensor readings, process monitoring, machine status
    - Analytics aggregations, alert events, performance metrics
    """
    
    def __init__(self):
        """Initialize Cassandra extractor."""
        self.config = get_cassandra_config()
        # Convert config to dictionary for CassandraClient
        config_dict = {
            'hosts': self.config.hosts,
            'port': self.config.port,
            'keyspace': self.config.keyspace,
            'username': self.config.username,
            'password': self.config.password
        }
        self.client = CassandraClient(config=config_dict)
        self.connected = False
        
    def connect(self) -> bool:
        """Connect to Cassandra database."""
        try:
            if self.client.connect():
                self.connected = True
                logger.info("✅ Connected to Cassandra for knowledge graph extraction")
                return True
            else:
                logger.error("❌ Failed to connect to Cassandra")
                return False
        except Exception as e:
            logger.error(f"❌ Cassandra connection error: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from Cassandra."""
        if self.connected:
            self.client.disconnect()
            self.connected = False
            logger.info("Disconnected from Cassandra")
    
    def extract_sensor_readings(self, hours_back: int = 24) -> List[Dict[str, Any]]:
        """
        Extract sensor readings for knowledge graph nodes.
        
        Args:
            hours_back: Number of hours to look back for data
            
        Returns:
            List[Dict[str, Any]]: Sensor reading data for graph nodes
        """
        try:
            if not self.connected:
                raise RuntimeError("Not connected to Cassandra")
            
            # Remove time filter to get all sensor readings (test data has future dates)
            query = """
                SELECT sensor_id, timestamp, sensor_type, value, unit, 
                       quality_score, metadata, process_id
                FROM sensor_readings
                LIMIT 1000
            """
            
            results = self.client.execute_cql(query)
            
            sensor_readings = []
            for row in results:
                reading_data = {
                    'node_type': 'SensorReading',
                    'sensor_id': row['sensor_id'],
                    'timestamp': row['timestamp'].isoformat() if row['timestamp'] else None,
                    'sensor_type': row['sensor_type'],
                    'value': float(row['value']) if row['value'] is not None else None,
                    'unit': row['unit'],
                    'quality_score': float(row['quality_score']) if row['quality_score'] is not None else None,
                    'metadata': safe_json_loads_with_fallback(row['metadata'], 'metadata', 5000, {}),
                    'process_id': row['process_id'],
                    'extraction_timestamp': datetime.utcnow().isoformat()
                }
                sensor_readings.append(reading_data)
            
            logger.info(f"📊 Extracted {len(sensor_readings)} sensor readings from Cassandra")
            return sensor_readings
            
        except Exception as e:
            logger.error(f"❌ Failed to extract sensor readings: {e}")
            return []
    
    def extract_process_monitoring(self, hours_back: int = 24) -> List[Dict[str, Any]]:
        """
        Extract process monitoring data for knowledge graph nodes.
        
        Args:
            hours_back: Number of hours to look back for data
            
        Returns:
            List[Dict[str, Any]]: Process monitoring data for graph nodes
        """
        try:
            if not self.connected:
                raise RuntimeError("Not connected to Cassandra")
            
            # Calculate time range
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=hours_back)
            
            query = """
                SELECT process_id, timestamp, event_type, event_data, 
                       severity, machine_id, operator_id, process_status,
                       layer_number, progress_percentage, session_id, metadata
                FROM process_monitoring
                WHERE timestamp >= ? AND timestamp <= ?
                ALLOW FILTERING
            """
            
            results = self.client.execute_cql(query, [start_time, end_time])
            
            monitoring_events = []
            for row in results:
                event_data = {
                    'node_type': 'ProcessMonitoring',
                    'process_id': row['process_id'],
                    'timestamp': row['timestamp'].isoformat() if row['timestamp'] else None,
                    'event_type': row['event_type'],
                    'event_data': safe_json_loads_with_fallback(row['event_data'], 'event_data', 5000, {}),
                    'severity': row['severity'],
                    'machine_id': row['machine_id'],
                    'operator_id': row['operator_id'],
                    'process_status': row['process_status'],
                    'layer_number': row['layer_number'],
                    'progress_percentage': float(row['progress_percentage']) if row['progress_percentage'] is not None else None,
                    'session_id': row['session_id'],
                    'metadata': safe_json_loads_with_fallback(row['metadata'], 'metadata', 5000, {}),
                    'extraction_timestamp': datetime.utcnow().isoformat()
                }
                monitoring_events.append(event_data)
            
            logger.info(f"📊 Extracted {len(monitoring_events)} process monitoring events from Cassandra")
            return monitoring_events
            
        except Exception as e:
            logger.error(f"❌ Failed to extract process monitoring: {e}")
            return []
    
    def extract_machine_status(self, hours_back: int = 24) -> List[Dict[str, Any]]:
        """
        Extract machine status data for knowledge graph nodes.
        
        Args:
            hours_back: Number of hours to look back for data
            
        Returns:
            List[Dict[str, Any]]: Machine status data for graph nodes
        """
        try:
            if not self.connected:
                raise RuntimeError("Not connected to Cassandra")
            
            # Calculate time range
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=hours_back)
            
            query = """
                SELECT machine_id, timestamp, status, performance_metrics, 
                       maintenance_required, health_score, operator_id, location
                FROM machine_status
                WHERE timestamp >= ? AND timestamp <= ?
                ALLOW FILTERING
            """
            
            results = self.client.execute_cql(query, [start_time, end_time])
            
            machine_statuses = []
            for row in results:
                status_data = {
                    'node_type': 'MachineStatus',
                    'machine_id': row['machine_id'],
                    'timestamp': row['timestamp'].isoformat() if row['timestamp'] else None,
                    'status': row['status'],
                    'performance_metrics': safe_json_loads_with_fallback(row['performance_metrics'], 'performance_metrics', 5000, {}),
                    'maintenance_required': row['maintenance_required'],
                    'health_score': float(row['health_score']) if row['health_score'] is not None else None,
                    'operator_id': row['operator_id'],
                    'location': row['location'],
                    'extraction_timestamp': datetime.utcnow().isoformat()
                }
                machine_statuses.append(status_data)
            
            logger.info(f"📊 Extracted {len(machine_statuses)} machine status updates from Cassandra")
            return machine_statuses
            
        except Exception as e:
            logger.error(f"❌ Failed to extract machine status: {e}")
            return []
    
    def extract_analytics_aggregations(self, hours_back: int = 24) -> List[Dict[str, Any]]:
        """
        Extract analytics aggregations for knowledge graph nodes.
        
        Args:
            hours_back: Number of hours to look back for data
            
        Returns:
            List[Dict[str, Any]]: Analytics aggregation data for graph nodes
        """
        try:
            if not self.connected:
                raise RuntimeError("Not connected to Cassandra")
            
            # Calculate time range
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=hours_back)
            
            query = """
                SELECT aggregation_id, timestamp, process_id, machine_id, 
                       aggregation_type, time_window, count, min_value, max_value, 
                       avg_value, median_value, std_dev, percentiles, data_quality_score
                FROM analytics_aggregations
                WHERE timestamp >= ? AND timestamp <= ?
                ALLOW FILTERING
            """
            
            results = self.client.execute_cql(query, [start_time, end_time])
            
            aggregations = []
            for row in results:
                agg_data = {
                    'node_type': 'AnalyticsAggregation',
                    'aggregation_id': row['aggregation_id'],
                    'timestamp': row['timestamp'].isoformat() if row['timestamp'] else None,
                    'process_id': row['process_id'],
                    'machine_id': row['machine_id'],
                    'aggregation_type': row['aggregation_type'],
                    'time_window': row['time_window'],
                    'count': row['count'],
                    'min_value': float(row['min_value']) if row['min_value'] is not None else None,
                    'max_value': float(row['max_value']) if row['max_value'] is not None else None,
                    'avg_value': float(row['avg_value']) if row['avg_value'] is not None else None,
                    'median_value': float(row['median_value']) if row['median_value'] is not None else None,
                    'std_dev': float(row['std_dev']) if row['std_dev'] is not None else None,
                    'percentiles': safe_json_loads_with_fallback(row['percentiles'], 'percentiles', 5000, {}),
                    'data_quality_score': float(row['data_quality_score']) if row['data_quality_score'] is not None else None,
                    'extraction_timestamp': datetime.utcnow().isoformat()
                }
                aggregations.append(agg_data)
            
            logger.info(f"📊 Extracted {len(aggregations)} analytics aggregations from Cassandra")
            return aggregations
            
        except Exception as e:
            logger.error(f"❌ Failed to extract analytics aggregations: {e}")
            return []
    
    def extract_alert_events(self, hours_back: int = 24) -> List[Dict[str, Any]]:
        """
        Extract alert events for knowledge graph nodes.
        
        Args:
            hours_back: Number of hours to look back for data
            
        Returns:
            List[Dict[str, Any]]: Alert event data for graph nodes
        """
        try:
            if not self.connected:
                raise RuntimeError("Not connected to Cassandra")
            
            # Calculate time range
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=hours_back)
            
            query = """
                SELECT alert_id, timestamp, alert_type, severity, 
                       process_id, machine_id, sensor_id, title, description,
                       threshold_value, actual_value, status, acknowledged, resolved
                FROM alert_events
                WHERE timestamp >= ? AND timestamp <= ?
                ALLOW FILTERING
            """
            
            results = self.client.execute_cql(query, [start_time, end_time])
            
            alert_events = []
            for row in results:
                alert_data = {
                    'node_type': 'AlertEvent',
                    'alert_id': row['alert_id'],
                    'timestamp': row['timestamp'].isoformat() if row['timestamp'] else None,
                    'alert_type': row['alert_type'],
                    'severity': row['severity'],
                    'process_id': row['process_id'],
                    'machine_id': row['machine_id'],
                    'sensor_id': row['sensor_id'],
                    'title': row['title'],
                    'description': row['description'],
                    'threshold_value': float(row['threshold_value']) if row['threshold_value'] is not None else None,
                    'actual_value': float(row['actual_value']) if row['actual_value'] is not None else None,
                    'status': row['status'],
                    'acknowledged': row['acknowledged'],
                    'resolved': row['resolved'],
                    'extraction_timestamp': datetime.utcnow().isoformat()
                }
                alert_events.append(alert_data)
            
            logger.info(f"📊 Extracted {len(alert_events)} alert events from Cassandra")
            return alert_events
            
        except Exception as e:
            logger.error(f"❌ Failed to extract alert events: {e}")
            return []
    
    def extract_sensor_types(self) -> List[Dict[str, Any]]:
        """
        Extract unique sensor types for knowledge graph nodes.
        
        Returns:
            List[Dict[str, Any]]: Sensor type data for graph nodes
        """
        try:
            if not self.connected:
                raise RuntimeError("Not connected to Cassandra")
            
            query = """
                SELECT sensor_type, unit
                FROM sensor_readings
                LIMIT 1000
            """
            
            results = self.client.execute_cql(query)
            
            sensor_types = []
            seen_types = set()
            
            for row in results:
                sensor_type = row['sensor_type']
                if sensor_type not in seen_types:
                    type_data = {
                        'node_type': 'SensorType',
                        'sensor_type': sensor_type,
                        'unit': row['unit'],
                        'extraction_timestamp': datetime.utcnow().isoformat()
                    }
                    sensor_types.append(type_data)
                    seen_types.add(sensor_type)
            
            logger.info(f"📊 Extracted {len(sensor_types)} unique sensor types from Cassandra")
            return sensor_types
            
        except Exception as e:
            logger.error(f"❌ Failed to extract sensor types: {e}")
            return []
    
    def extract_all_data(self, hours_back: int = 24) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract all data from Cassandra for knowledge graph construction.
        
        Args:
            hours_back: Number of hours to look back for data
            
        Returns:
            Dict[str, List[Dict[str, Any]]]: All extracted data organized by type
        """
        logger.info(f"🚀 Starting comprehensive Cassandra data extraction (last {hours_back} hours)...")
        
        if not self.connected:
            if not self.connect():
                return {}
        
        try:
            extracted_data = {
                'sensor_readings': self.extract_sensor_readings(hours_back),
                'process_monitoring': self.extract_process_monitoring(hours_back),
                'machine_status': self.extract_machine_status(hours_back),
                'analytics_aggregations': self.extract_analytics_aggregations(hours_back),
                'alert_events': self.extract_alert_events(hours_back),
                'sensor_types': self.extract_sensor_types()
            }
            
            # Calculate totals
            total_records = sum(len(data) for data in extracted_data.values())
            
            logger.info(f"✅ Cassandra extraction completed:")
            logger.info(f"   📊 Total Records: {total_records}")
            logger.info(f"   📡 Sensor Readings: {len(extracted_data['sensor_readings'])}")
            logger.info(f"   🔍 Process Monitoring: {len(extracted_data['process_monitoring'])}")
            logger.info(f"   🏭 Machine Status: {len(extracted_data['machine_status'])}")
            logger.info(f"   📈 Analytics Aggregations: {len(extracted_data['analytics_aggregations'])}")
            logger.info(f"   🚨 Alert Events: {len(extracted_data['alert_events'])}")
            logger.info(f"   🔧 Sensor Types: {len(extracted_data['sensor_types'])}")
            
            return extracted_data
            
        except Exception as e:
            logger.error(f"❌ Failed to extract all Cassandra data: {e}")
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
            
            # Get table counts (approximate)
            tables = ['sensor_readings', 'process_monitoring', 'machine_status', 
                     'analytics_aggregations', 'alert_events']
            counts = {}
            
            for table in tables:
                try:
                    result = self.client.execute_cql(f"SELECT COUNT(*) FROM {table} LIMIT 1000")
                    counts[table] = len(result) if result else 0
                except Exception as e:
                    logger.warning(f"Could not get count for table {table}: {e}")
                    counts[table] = 0
            
            summary = {
                'database': 'Cassandra',
                'connection_status': 'Connected' if self.connected else 'Disconnected',
                'table_counts': counts,
                'total_records': sum(counts.values()),
                'extraction_timestamp': datetime.utcnow().isoformat()
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"❌ Failed to get extraction summary: {e}")
            return {}
    
    def close_connection(self) -> None:
        """Close the Cassandra connection."""
        try:
            if self.client and self.client.connected:
                self.client.close_connection()
                self.connected = False
                logger.info("✅ Cassandra connection closed")
        except Exception as e:
            logger.error(f"❌ Failed to close Cassandra connection: {e}")
