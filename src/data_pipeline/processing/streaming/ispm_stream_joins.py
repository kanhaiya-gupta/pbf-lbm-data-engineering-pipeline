"""
ISPM Stream Joins

This module provides stream-to-stream join capabilities for ISPM monitoring data.
"""

import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
import logging
from datetime import datetime, timedelta

from src.data_pipeline.config.pipeline_config import get_pipeline_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ISPMStreamJoins:
    """
    Stream-to-stream joins for ISPM monitoring data.
    """
    
    def __init__(self):
        self.config = get_pipeline_config()
        self.join_windows = self._load_join_windows()
        self.stream_buffers = {}
    
    def _load_join_windows(self) -> Dict[str, int]:
        """Load join window configurations."""
        try:
            return self.config.get('join_windows', {
                'temperature_pressure': 30,  # 30 seconds
                'sensor_correlation': 60,    # 60 seconds
                'process_monitoring': 120    # 120 seconds
            })
        except Exception as e:
            logger.error(f"Error loading join windows: {e}")
            return {}
    
    def join_temperature_pressure_streams(self, temp_data: Dict[str, Any], 
                                        pressure_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Join temperature and pressure streams."""
        try:
            # Check if data is within join window
            if not self._is_within_join_window(temp_data, pressure_data, 'temperature_pressure'):
                return None
            
            # Create joined record
            joined_data = {
                'timestamp': temp_data.get('timestamp'),
                'process_id': temp_data.get('process_id'),
                'temperature': temp_data.get('value'),
                'pressure': pressure_data.get('value'),
                'temperature_sensor_id': temp_data.get('sensor_id'),
                'pressure_sensor_id': pressure_data.get('sensor_id'),
                'joined_at': datetime.now().isoformat()
            }
            
            # Calculate derived metrics
            joined_data['temp_pressure_ratio'] = self._calculate_temp_pressure_ratio(
                joined_data['temperature'], joined_data['pressure']
            )
            
            joined_data['process_stability'] = self._calculate_process_stability(
                joined_data['temperature'], joined_data['pressure']
            )
            
            logger.info("Temperature and pressure streams joined successfully")
            return joined_data
            
        except Exception as e:
            logger.error(f"Error joining temperature and pressure streams: {e}")
            return None
    
    def join_sensor_correlation_streams(self, sensor_data_list: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Join multiple sensor streams for correlation analysis."""
        try:
            if len(sensor_data_list) < 2:
                return None
            
            # Check if all data is within join window
            if not self._are_within_join_window(sensor_data_list, 'sensor_correlation'):
                return None
            
            # Group data by sensor type
            sensor_groups = {}
            for data in sensor_data_list:
                sensor_type = data.get('sensor_type', 'unknown')
                if sensor_type not in sensor_groups:
                    sensor_groups[sensor_type] = []
                sensor_groups[sensor_type].append(data)
            
            # Create joined record
            joined_data = {
                'timestamp': sensor_data_list[0].get('timestamp'),
                'process_id': sensor_data_list[0].get('process_id'),
                'joined_at': datetime.now().isoformat()
            }
            
            # Add sensor data
            for sensor_type, data_list in sensor_groups.items():
                if data_list:
                    latest_data = max(data_list, key=lambda x: x.get('timestamp', ''))
                    joined_data[f'{sensor_type}_value'] = latest_data.get('value')
                    joined_data[f'{sensor_type}_sensor_id'] = latest_data.get('sensor_id')
            
            # Calculate correlation metrics
            joined_data['sensor_correlation_score'] = self._calculate_sensor_correlation(
                sensor_groups
            )
            
            joined_data['sensor_consistency'] = self._calculate_sensor_consistency(
                sensor_groups
            )
            
            logger.info("Sensor correlation streams joined successfully")
            return joined_data
            
        except Exception as e:
            logger.error(f"Error joining sensor correlation streams: {e}")
            return None
    
    def join_process_monitoring_streams(self, process_data: Dict[str, Any], 
                                      monitoring_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Join process and monitoring streams."""
        try:
            # Check if data is within join window
            if not self._is_within_join_window(process_data, monitoring_data, 'process_monitoring'):
                return None
            
            # Create joined record
            joined_data = {
                'timestamp': process_data.get('timestamp'),
                'process_id': process_data.get('process_id'),
                'process_parameters': {
                    'laser_power': process_data.get('laser_power'),
                    'scan_speed': process_data.get('scan_speed'),
                    'layer_height': process_data.get('layer_height')
                },
                'monitoring_data': {
                    'sensor_id': monitoring_data.get('sensor_id'),
                    'sensor_type': monitoring_data.get('sensor_type'),
                    'value': monitoring_data.get('value'),
                    'unit': monitoring_data.get('unit')
                },
                'joined_at': datetime.now().isoformat()
            }
            
            # Calculate process-monitoring correlation
            joined_data['process_monitoring_correlation'] = self._calculate_process_monitoring_correlation(
                process_data, monitoring_data
            )
            
            # Calculate process efficiency with monitoring
            joined_data['monitored_process_efficiency'] = self._calculate_monitored_process_efficiency(
                process_data, monitoring_data
            )
            
            logger.info("Process and monitoring streams joined successfully")
            return joined_data
            
        except Exception as e:
            logger.error(f"Error joining process and monitoring streams: {e}")
            return None
    
    def _is_within_join_window(self, data1: Dict[str, Any], data2: Dict[str, Any], 
                              join_type: str) -> bool:
        """Check if two data records are within join window."""
        try:
            window_seconds = self.join_windows.get(join_type, 60)
            
            timestamp1 = self._parse_timestamp(data1.get('timestamp'))
            timestamp2 = self._parse_timestamp(data2.get('timestamp'))
            
            if not timestamp1 or not timestamp2:
                return False
            
            time_diff = abs((timestamp1 - timestamp2).total_seconds())
            return time_diff <= window_seconds
            
        except Exception as e:
            logger.error(f"Error checking join window: {e}")
            return False
    
    def _are_within_join_window(self, data_list: List[Dict[str, Any]], join_type: str) -> bool:
        """Check if multiple data records are within join window."""
        try:
            window_seconds = self.join_windows.get(join_type, 60)
            
            timestamps = []
            for data in data_list:
                timestamp = self._parse_timestamp(data.get('timestamp'))
                if timestamp:
                    timestamps.append(timestamp)
            
            if len(timestamps) < 2:
                return False
            
            # Check if all timestamps are within the window
            min_timestamp = min(timestamps)
            max_timestamp = max(timestamps)
            
            time_diff = (max_timestamp - min_timestamp).total_seconds()
            return time_diff <= window_seconds
            
        except Exception as e:
            logger.error(f"Error checking join window for multiple records: {e}")
            return False
    
    def _parse_timestamp(self, timestamp_str: str) -> Optional[datetime]:
        """Parse timestamp string to datetime object."""
        try:
            if isinstance(timestamp_str, str):
                return datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            return None
        except Exception as e:
            logger.error(f"Error parsing timestamp: {e}")
            return None
    
    def _calculate_temp_pressure_ratio(self, temperature: float, pressure: float) -> float:
        """Calculate temperature to pressure ratio."""
        try:
            if pressure == 0:
                return 0.0
            return temperature / pressure
        except Exception as e:
            logger.error(f"Error calculating temp-pressure ratio: {e}")
            return 0.0
    
    def _calculate_process_stability(self, temperature: float, pressure: float) -> float:
        """Calculate process stability score."""
        try:
            # Simple stability calculation based on temperature and pressure
            temp_stability = 1.0 - abs(temperature - 1000.0) / 1000.0  # Assuming 1000°C is optimal
            pressure_stability = 1.0 - abs(pressure - 500.0) / 500.0   # Assuming 500 mbar is optimal
            
            return (temp_stability + pressure_stability) / 2.0
            
        except Exception as e:
            logger.error(f"Error calculating process stability: {e}")
            return 0.0
    
    def _calculate_sensor_correlation(self, sensor_groups: Dict[str, List[Dict[str, Any]]]) -> float:
        """Calculate sensor correlation score."""
        try:
            if len(sensor_groups) < 2:
                return 0.0
            
            # Simple correlation calculation
            values = []
            for sensor_type, data_list in sensor_groups.items():
                if data_list:
                    latest_data = max(data_list, key=lambda x: x.get('timestamp', ''))
                    values.append(latest_data.get('value', 0))
            
            if len(values) < 2:
                return 0.0
            
            # Calculate simple correlation coefficient
            mean_val = sum(values) / len(values)
            numerator = sum((v - mean_val) ** 2 for v in values)
            denominator = len(values) * (mean_val ** 2) if mean_val != 0 else 1
            
            correlation = 1.0 - (numerator / denominator) if denominator != 0 else 0.0
            return max(0.0, min(1.0, correlation))
            
        except Exception as e:
            logger.error(f"Error calculating sensor correlation: {e}")
            return 0.0
    
    def _calculate_sensor_consistency(self, sensor_groups: Dict[str, List[Dict[str, Any]]]) -> float:
        """Calculate sensor consistency score."""
        try:
            consistency_scores = []
            
            for sensor_type, data_list in sensor_groups.items():
                if len(data_list) > 1:
                    values = [data.get('value', 0) for data in data_list]
                    if values:
                        mean_val = sum(values) / len(values)
                        variance = sum((v - mean_val) ** 2 for v in values) / len(values)
                        consistency = 1.0 / (1.0 + variance) if variance > 0 else 1.0
                        consistency_scores.append(consistency)
            
            return sum(consistency_scores) / len(consistency_scores) if consistency_scores else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating sensor consistency: {e}")
            return 0.0
    
    def _calculate_process_monitoring_correlation(self, process_data: Dict[str, Any], 
                                                monitoring_data: Dict[str, Any]) -> float:
        """Calculate correlation between process and monitoring data."""
        try:
            # Simple correlation based on process parameters and monitoring values
            process_score = 0.0
            if 'laser_power' in process_data:
                process_score += process_data['laser_power'] / 1000.0  # Normalize
            
            monitoring_score = 0.0
            if 'value' in monitoring_data:
                monitoring_score = monitoring_data['value'] / 1000.0  # Normalize
            
            correlation = 1.0 - abs(process_score - monitoring_score)
            return max(0.0, min(1.0, correlation))
            
        except Exception as e:
            logger.error(f"Error calculating process-monitoring correlation: {e}")
            return 0.0
    
    def _calculate_monitored_process_efficiency(self, process_data: Dict[str, Any], 
                                              monitoring_data: Dict[str, Any]) -> float:
        """Calculate process efficiency considering monitoring data."""
        try:
            # Base process efficiency
            base_efficiency = 0.0
            if all(key in process_data for key in ['laser_power', 'scan_speed', 'layer_height']):
                base_efficiency = (process_data['laser_power'] * process_data['scan_speed']) / (
                    process_data['layer_height'] * 1000
                )
            
            # Monitoring adjustment
            monitoring_adjustment = 1.0
            if 'value' in monitoring_data:
                # Adjust based on monitoring value (assuming optimal range)
                optimal_value = 500.0  # Example optimal value
                actual_value = monitoring_data['value']
                adjustment_factor = 1.0 - abs(actual_value - optimal_value) / optimal_value
                monitoring_adjustment = max(0.5, min(1.5, adjustment_factor))
            
            return base_efficiency * monitoring_adjustment
            
        except Exception as e:
            logger.error(f"Error calculating monitored process efficiency: {e}")
            return 0.0
