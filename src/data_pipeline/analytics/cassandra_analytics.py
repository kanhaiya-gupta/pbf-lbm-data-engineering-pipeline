"""
Cassandra Analytics Module for PBF-LB/M Time-Series Data

This module provides comprehensive analytics capabilities for time-series data
stored in Cassandra, including statistical analysis, real-time aggregations,
and advanced querying features for PBF-LB/M manufacturing processes.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
import statistics
import json
from collections import defaultdict

from src.data_pipeline.storage.operational.cassandra_client import CassandraClient
from src.data_pipeline.config.cassandra_config import get_cassandra_config
from src.data_pipeline.processing.schema.models.nosql_models.cassandra_time_series_models import (
    SensorReading, ProcessMonitoring, MachineStatusUpdate, AnalyticsAggregation, AlertEvent
)

logger = logging.getLogger(__name__)


@dataclass
class TimeSeriesStats:
    """Time-series statistics container."""
    count: int
    mean: float
    median: float
    std_dev: float
    min_value: float
    max_value: float
    q25: float
    q75: float
    variance: float
    range: float


@dataclass
class ProcessAnalytics:
    """Process analytics results."""
    process_id: str
    total_sensor_readings: int
    avg_temperature: float
    avg_pressure: float
    avg_vibration: float
    quality_score: float
    anomalies_detected: int
    alerts_triggered: int
    processing_time_minutes: float
    efficiency_score: float


class CassandraAnalytics:
    """
    Comprehensive analytics engine for Cassandra time-series data.
    
    Provides statistical analysis, real-time aggregations, and advanced
    querying capabilities for PBF-LB/M manufacturing data.
    """
    
    def __init__(self, cassandra_client: Optional[CassandraClient] = None):
        """
        Initialize Cassandra analytics engine.
        
        Args:
            cassandra_client: Optional Cassandra client instance
        """
        self.client = cassandra_client or CassandraClient(get_cassandra_config())
        self.connected = False
        
        # Analytics cache for performance
        self._cache = {}
        self._cache_ttl = 300  # 5 minutes
        
    def connect(self) -> bool:
        """Connect to Cassandra cluster."""
        try:
            if not self.client.connect():
                logger.error("Failed to connect to Cassandra")
                return False
            
            self.connected = True
            logger.info("✅ Connected to Cassandra for analytics")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Cassandra: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from Cassandra."""
        if self.client:
            self.client.disconnect()
        self.connected = False
    
    def get_sensor_statistics(self, 
                            sensor_id: str, 
                            start_time: datetime, 
                            end_time: datetime,
                            sensor_type: Optional[str] = None) -> TimeSeriesStats:
        """
        Calculate comprehensive statistics for sensor data.
        
        Args:
            sensor_id: Sensor identifier
            start_time: Start of time range
            end_time: End of time range
            sensor_type: Optional sensor type filter
            
        Returns:
            TimeSeriesStats object with statistical measures
        """
        try:
            # Build query for sensor readings - use proper partition key
            query = """
                SELECT value, timestamp 
                FROM sensor_readings 
                WHERE sensor_id = %s 
                AND timestamp >= %s 
                AND timestamp <= %s
            """
            params = [sensor_id, start_time, end_time]
            
            if sensor_type:
                query += " AND sensor_type = %s"
                params.append(sensor_type)
            
            query += " ORDER BY timestamp"
            
            results = self.client.execute_cql(query, params)
            
            if not results:
                return TimeSeriesStats(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
            
            values = [row['value'] for row in results]
            
            return TimeSeriesStats(
                count=len(values),
                mean=statistics.mean(values),
                median=statistics.median(values),
                std_dev=statistics.stdev(values) if len(values) > 1 else 0,
                min_value=min(values),
                max_value=max(values),
                q25=statistics.quantiles(values, n=4)[0] if len(values) >= 4 else min(values),
                q75=statistics.quantiles(values, n=4)[2] if len(values) >= 4 else max(values),
                variance=statistics.variance(values) if len(values) > 1 else 0,
                range=max(values) - min(values)
            )
            
        except Exception as e:
            logger.error(f"Failed to calculate sensor statistics: {e}")
            return TimeSeriesStats(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    
    def get_process_analytics(self, process_id: str) -> ProcessAnalytics:
        """
        Generate comprehensive analytics for a manufacturing process.
        
        Args:
            process_id: Process identifier
            
        Returns:
            ProcessAnalytics object with process metrics
        """
        try:
            # Get sensor readings for the process - need to query by sensor_id first
            # Since process_id is not a partition key, we need to use ALLOW FILTERING
            sensor_query = """
                SELECT sensor_type, value, quality_score, timestamp
                FROM sensor_readings 
                WHERE process_id = %s
                ALLOW FILTERING
            """
            sensor_results = self.client.execute_cql(sensor_query, [process_id])
            
            # Get process monitoring events
            monitoring_query = """
                SELECT event_type, timestamp, metadata
                FROM process_monitoring 
                WHERE process_id = %s
                ALLOW FILTERING
            """
            monitoring_results = self.client.execute_cql(monitoring_query, [process_id])
            
            # Get alerts for the process
            alerts_query = """
                SELECT alert_type, severity, timestamp
                FROM alert_events 
                WHERE process_id = %s
                ALLOW FILTERING
            """
            alerts_results = self.client.execute_cql(alerts_query, [process_id])
            
            # Calculate metrics
            total_readings = len(sensor_results)
            
            # Temperature analysis
            temp_readings = [r['value'] for r in sensor_results if r['sensor_type'] == 'THERMAL']
            avg_temperature = statistics.mean(temp_readings) if temp_readings else 0
            
            # Pressure analysis
            pressure_readings = [r['value'] for r in sensor_results if r['sensor_type'] == 'PRESSURE']
            avg_pressure = statistics.mean(pressure_readings) if pressure_readings else 0
            
            # Vibration analysis
            vibration_readings = [r['value'] for r in sensor_results if r['sensor_type'] == 'VIBRATION']
            avg_vibration = statistics.mean(vibration_readings) if vibration_readings else 0
            
            # Quality score calculation
            quality_scores = [r['quality_score'] for r in sensor_results if r['quality_score'] is not None]
            avg_quality_score = statistics.mean(quality_scores) if quality_scores else 0
            
            # Anomaly detection (simplified)
            anomalies = len([r for r in sensor_results if r['quality_score'] and r['quality_score'] < 50])
            
            # Alert analysis
            alerts_triggered = len(alerts_results)
            
            # Processing time calculation
            if sensor_results:
                start_time = min(r['timestamp'] for r in sensor_results)
                end_time = max(r['timestamp'] for r in sensor_results)
                processing_time = (end_time - start_time).total_seconds() / 60  # minutes
            else:
                processing_time = 0
            
            # Efficiency score (simplified calculation)
            efficiency_score = max(0, min(100, avg_quality_score - (anomalies * 5) - (alerts_triggered * 2)))
            
            return ProcessAnalytics(
                process_id=process_id,
                total_sensor_readings=total_readings,
                avg_temperature=avg_temperature,
                avg_pressure=avg_pressure,
                avg_vibration=avg_vibration,
                quality_score=avg_quality_score,
                anomalies_detected=anomalies,
                alerts_triggered=alerts_triggered,
                processing_time_minutes=processing_time,
                efficiency_score=efficiency_score
            )
            
        except Exception as e:
            logger.error(f"Failed to generate process analytics: {e}")
            return ProcessAnalytics(
                process_id=process_id,
                total_sensor_readings=0,
                avg_temperature=0,
                avg_pressure=0,
                avg_vibration=0,
                quality_score=0,
                anomalies_detected=0,
                alerts_triggered=0,
                processing_time_minutes=0,
                efficiency_score=0
            )
    
    def get_machine_performance(self, machine_id: str, days: int = 7) -> Dict[str, Any]:
        """
        Analyze machine performance over a specified period.
        
        Args:
            machine_id: Machine identifier
            days: Number of days to analyze
            
        Returns:
            Dictionary with machine performance metrics
        """
        try:
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(days=days)
            
            # Get machine status updates
            status_query = """
                SELECT status, performance_metrics, timestamp
                FROM machine_status 
                WHERE machine_id = %s 
                AND timestamp >= %s 
                AND timestamp <= %s
            """
            status_results = self.client.execute_cql(status_query, [machine_id, start_time, end_time])
            
            # Get sensor data for the machine
            sensor_query = """
                SELECT sensor_type, value, quality_score, timestamp
                FROM sensor_readings 
                WHERE machine_id = %s 
                AND timestamp >= %s 
                AND timestamp <= %s
                ALLOW FILTERING
            """
            sensor_results = self.client.execute_cql(sensor_query, [machine_id, start_time, end_time])
            
            # Calculate performance metrics
            total_operations = len(status_results)
            uptime_percentage = 0
            avg_efficiency = 0
            maintenance_events = 0
            
            if status_results:
                # Calculate uptime
                operational_statuses = [s for s in status_results if s['status'] in ['OPERATIONAL', 'RUNNING']]
                uptime_percentage = (len(operational_statuses) / total_operations) * 100
                
                # Calculate average efficiency from performance metrics
                efficiency_scores = []
                for status in status_results:
                    if status['performance_metrics']:
                        try:
                            metrics = json.loads(status['performance_metrics']) if isinstance(status['performance_metrics'], str) else status['performance_metrics']
                            if 'efficiency' in metrics:
                                efficiency_scores.append(metrics['efficiency'])
                        except:
                            pass
                
                avg_efficiency = statistics.mean(efficiency_scores) if efficiency_scores else 0
                
                # Count maintenance events
                maintenance_events = len([s for s in status_results if s['status'] == 'MAINTENANCE'])
            
            # Sensor data analysis
            sensor_stats = {}
            for sensor_type in ['THERMAL', 'PRESSURE', 'VIBRATION', 'OPTICAL']:
                type_readings = [s for s in sensor_results if s['sensor_type'] == sensor_type]
                if type_readings:
                    values = [r['value'] for r in type_readings]
                    sensor_stats[sensor_type] = {
                        'count': len(values),
                        'mean': statistics.mean(values),
                        'std_dev': statistics.stdev(values) if len(values) > 1 else 0,
                        'min': min(values),
                        'max': max(values)
                    }
            
            return {
                'machine_id': machine_id,
                'analysis_period_days': days,
                'total_operations': total_operations,
                'uptime_percentage': uptime_percentage,
                'avg_efficiency': avg_efficiency,
                'maintenance_events': maintenance_events,
                'sensor_statistics': sensor_stats,
                'data_points_analyzed': len(sensor_results),
                'analysis_timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze machine performance: {e}")
            return {
                'machine_id': machine_id,
                'error': str(e),
                'analysis_timestamp': datetime.utcnow().isoformat()
            }
    
    def get_quality_trends(self, process_id: str, hours: int = 24) -> Dict[str, Any]:
        """
        Analyze quality trends for a process over time.
        
        Args:
            process_id: Process identifier
            hours: Number of hours to analyze
            
        Returns:
            Dictionary with quality trend analysis
        """
        try:
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=hours)
            
            # Get quality scores over time
            quality_query = """
                SELECT quality_score, timestamp, sensor_type
                FROM sensor_readings 
                WHERE process_id = %s 
                AND timestamp >= %s 
                AND timestamp <= %s
                ALLOW FILTERING
            """
            quality_results = self.client.execute_cql(quality_query, [process_id, start_time, end_time])
            
            if not quality_results:
                return {
                    'process_id': process_id,
                    'trend': 'no_data',
                    'avg_quality': 0,
                    'quality_variance': 0,
                    'trend_direction': 'stable'
                }
            
            # Calculate trend metrics - filter out null quality scores
            quality_scores = [r['quality_score'] for r in quality_results if r['quality_score'] is not None]
            timestamps = [r['timestamp'] for r in quality_results if r['quality_score'] is not None]
            
            avg_quality = statistics.mean(quality_scores)
            quality_variance = statistics.variance(quality_scores) if len(quality_scores) > 1 else 0
            
            # Determine trend direction
            if len(quality_scores) >= 2:
                first_half = quality_scores[:len(quality_scores)//2]
                second_half = quality_scores[len(quality_scores)//2:]
                
                first_avg = statistics.mean(first_half)
                second_avg = statistics.mean(second_half)
                
                if second_avg > first_avg + 5:
                    trend_direction = 'improving'
                elif second_avg < first_avg - 5:
                    trend_direction = 'declining'
                else:
                    trend_direction = 'stable'
            else:
                trend_direction = 'stable'
            
            # Quality distribution by sensor type
            quality_by_sensor = defaultdict(list)
            for result in quality_results:
                quality_by_sensor[result['sensor_type']].append(result['quality_score'])
            
            sensor_quality = {}
            for sensor_type, scores in quality_by_sensor.items():
                sensor_quality[sensor_type] = {
                    'avg_quality': statistics.mean(scores),
                    'count': len(scores),
                    'std_dev': statistics.stdev(scores) if len(scores) > 1 else 0
                }
            
            return {
                'process_id': process_id,
                'analysis_hours': hours,
                'avg_quality': avg_quality,
                'quality_variance': quality_variance,
                'trend_direction': trend_direction,
                'total_measurements': len(quality_scores),
                'sensor_quality_breakdown': sensor_quality,
                'analysis_timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze quality trends: {e}")
            return {
                'process_id': process_id,
                'error': str(e),
                'analysis_timestamp': datetime.utcnow().isoformat()
            }
    
    def get_anomaly_analysis(self, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """
        Perform anomaly detection analysis across all processes.
        
        Args:
            start_time: Start of analysis period
            end_time: End of analysis period
            
        Returns:
            Dictionary with anomaly analysis results
        """
        try:
            # Get all sensor readings in the time range
            sensor_query = """
                SELECT process_id, sensor_id, sensor_type, value, quality_score, timestamp
                FROM sensor_readings 
                WHERE timestamp >= %s 
                AND timestamp <= %s
                ALLOW FILTERING
            """
            sensor_results = self.client.execute_cql(sensor_query, [start_time, end_time])
            
            # Get alerts in the time range
            alerts_query = """
                SELECT process_id, alert_type, severity, timestamp
                FROM alert_events 
                WHERE timestamp >= %s 
                AND timestamp <= %s
                ALLOW FILTERING
            """
            alerts_results = self.client.execute_cql(alerts_query, [start_time, end_time])
            
            # Analyze anomalies
            anomalies_by_process = defaultdict(list)
            anomalies_by_sensor_type = defaultdict(list)
            
            for reading in sensor_results:
                # Simple anomaly detection based on quality score and value ranges
                is_anomaly = False
                anomaly_reasons = []
                
                if reading['quality_score'] and reading['quality_score'] < 30:
                    is_anomaly = True
                    anomaly_reasons.append('low_quality_score')
                
                # Value-based anomaly detection (simplified)
                if reading['sensor_type'] == 'THERMAL' and (reading['value'] < -50 or reading['value'] > 1000):
                    is_anomaly = True
                    anomaly_reasons.append('temperature_out_of_range')
                elif reading['sensor_type'] == 'PRESSURE' and reading['value'] < 0:
                    is_anomaly = True
                    anomaly_reasons.append('negative_pressure')
                elif reading['sensor_type'] == 'VIBRATION' and reading['value'] > 10:
                    is_anomaly = True
                    anomaly_reasons.append('excessive_vibration')
                
                if is_anomaly:
                    anomaly_data = {
                        'sensor_id': reading['sensor_id'],
                        'sensor_type': reading['sensor_type'],
                        'value': reading['value'],
                        'quality_score': reading['quality_score'],
                        'timestamp': reading['timestamp'],
                        'reasons': anomaly_reasons
                    }
                    anomalies_by_process[reading['process_id']].append(anomaly_data)
                    anomalies_by_sensor_type[reading['sensor_type']].append(anomaly_data)
            
            # Calculate anomaly statistics
            total_anomalies = sum(len(anomalies) for anomalies in anomalies_by_process.values())
            processes_with_anomalies = len(anomalies_by_process)
            total_processes = len(set(r['process_id'] for r in sensor_results))
            
            # Alert correlation
            alert_correlation = {}
            for alert in alerts_results:
                process_id = alert['process_id']
                if process_id in anomalies_by_process:
                    if process_id not in alert_correlation:
                        alert_correlation[process_id] = {'alerts': 0, 'anomalies': 0}
                    alert_correlation[process_id]['alerts'] += 1
                    alert_correlation[process_id]['anomalies'] = len(anomalies_by_process[process_id])
            
            return {
                'analysis_period': {
                    'start': start_time.isoformat(),
                    'end': end_time.isoformat()
                },
                'total_measurements': len(sensor_results),
                'total_anomalies': total_anomalies,
                'anomaly_rate': (total_anomalies / len(sensor_results)) * 100 if sensor_results else 0,
                'processes_analyzed': total_processes,
                'processes_with_anomalies': processes_with_anomalies,
                'anomalies_by_sensor_type': {
                    sensor_type: len(anomalies) 
                    for sensor_type, anomalies in anomalies_by_sensor_type.items()
                },
                'alert_correlation': alert_correlation,
                'top_anomaly_processes': sorted(
                    [(pid, len(anomalies)) for pid, anomalies in anomalies_by_process.items()],
                    key=lambda x: x[1], reverse=True
                )[:10],
                'analysis_timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to perform anomaly analysis: {e}")
            return {
                'error': str(e),
                'analysis_timestamp': datetime.utcnow().isoformat()
            }
    
    def get_real_time_dashboard_data(self) -> Dict[str, Any]:
        """
        Generate real-time dashboard data for monitoring.
        
        Returns:
            Dictionary with real-time metrics and alerts
        """
        try:
            # Get recent sensor readings (last hour)
            recent_time = datetime.utcnow() - timedelta(hours=1)
            
            recent_sensors_query = """
                SELECT sensor_type, value, quality_score, process_id, timestamp
                FROM sensor_readings 
                WHERE timestamp >= %s
                ALLOW FILTERING
            """
            recent_sensors = self.client.execute_cql(recent_sensors_query, [recent_time])
            # Sort by timestamp descending and limit to 1000
            recent_sensors = sorted(recent_sensors, key=lambda x: x['timestamp'], reverse=True)[:1000]
            
            # Get active alerts
            active_alerts_query = """
                SELECT alert_type, severity, process_id, timestamp
                FROM alert_events 
                WHERE timestamp >= %s
                ALLOW FILTERING
            """
            active_alerts = self.client.execute_cql(active_alerts_query, [recent_time])
            # Sort by timestamp descending and limit to 50
            active_alerts = sorted(active_alerts, key=lambda x: x['timestamp'], reverse=True)[:50]
            
            # Get machine status
            machine_status_query = """
                SELECT machine_id, status, timestamp
                FROM machine_status 
                WHERE timestamp >= %s
                ALLOW FILTERING
            """
            machine_status = self.client.execute_cql(machine_status_query, [recent_time])
            # Sort by timestamp descending
            machine_status = sorted(machine_status, key=lambda x: x['timestamp'], reverse=True)
            
            # Calculate real-time metrics
            if recent_sensors:
                # Current sensor averages
                sensor_averages = {}
                for sensor_type in ['THERMAL', 'PRESSURE', 'VIBRATION', 'OPTICAL']:
                    type_readings = [s for s in recent_sensors if s['sensor_type'] == sensor_type]
                    if type_readings:
                        values = [r['value'] for r in type_readings]
                        sensor_averages[sensor_type] = {
                            'current_avg': statistics.mean(values),
                            'count': len(values),
                            'latest_timestamp': max(r['timestamp'] for r in type_readings)
                        }
                
                # Quality score
                quality_scores = [s['quality_score'] for s in recent_sensors if s['quality_score'] is not None]
                avg_quality = statistics.mean(quality_scores) if quality_scores else 0
                
                # Active processes
                active_processes = len(set(s['process_id'] for s in recent_sensors))
            else:
                sensor_averages = {}
                avg_quality = 0
                active_processes = 0
            
            # Machine status summary
            machine_status_summary = {}
            for status in machine_status:
                machine_id = status['machine_id']
                if machine_id not in machine_status_summary:
                    machine_status_summary[machine_id] = status['status']
            
            # Alert summary
            alert_summary = {
                'total_alerts': len(active_alerts),
                'by_severity': {},
                'by_type': {}
            }
            
            for alert in active_alerts:
                severity = alert['severity']
                alert_type = alert['alert_type']
                
                alert_summary['by_severity'][severity] = alert_summary['by_severity'].get(severity, 0) + 1
                alert_summary['by_type'][alert_type] = alert_summary['by_type'].get(alert_type, 0) + 1
            
            return {
                'timestamp': datetime.utcnow().isoformat(),
                'active_processes': active_processes,
                'sensor_averages': sensor_averages,
                'overall_quality_score': avg_quality,
                'machine_status': machine_status_summary,
                'alert_summary': alert_summary,
                'recent_alerts': [
                    {
                        'alert_type': alert['alert_type'],
                        'severity': alert['severity'],
                        'process_id': alert['process_id'],
                        'timestamp': alert['timestamp'].isoformat()
                    }
                    for alert in active_alerts[:10]
                ],
                'data_freshness': {
                    'latest_sensor_reading': max(s['timestamp'] for s in recent_sensors).isoformat() if recent_sensors else None,
                    'sensor_data_age_minutes': (
                        (datetime.utcnow() - max(s['timestamp'] for s in recent_sensors)).total_seconds() / 60
                    ) if recent_sensors else None
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to generate real-time dashboard data: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
