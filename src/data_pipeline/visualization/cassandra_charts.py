"""
Cassandra Data Visualization Module

This module provides data visualization capabilities for Cassandra time-series data,
including charts, graphs, and reports for PBF-LB/M manufacturing analytics.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import json
import base64
from io import BytesIO

try:
    import matplotlib
    # Set backend before importing pyplot to avoid display issues
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import seaborn as sns
    import pandas as pd
    import numpy as np
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("Matplotlib/Seaborn not available. Chart generation will be limited.")

from src.data_pipeline.analytics.cassandra_analytics import CassandraAnalytics

logger = logging.getLogger(__name__)


class CassandraCharts:
    """
    Data visualization engine for Cassandra time-series data.
    
    Generates charts, graphs, and reports for PBF-LB/M manufacturing analytics.
    """
    
    def __init__(self, analytics: Optional[CassandraAnalytics] = None):
        """
        Initialize Cassandra charts generator.
        
        Args:
            analytics: Optional CassandraAnalytics instance
        """
        self.analytics = analytics or CassandraAnalytics()
        self.connected = False
        
        if MATPLOTLIB_AVAILABLE:
            # Set up matplotlib style
            plt.style.use('seaborn-v0_8')
            sns.set_palette("husl")
    
    def connect(self) -> bool:
        """Connect to Cassandra analytics."""
        try:
            if not self.analytics.connect():
                return False
            self.connected = True
            return True
        except Exception as e:
            logger.error(f"Failed to connect to analytics: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from analytics."""
        if self.analytics:
            self.analytics.disconnect()
        self.connected = False
    
    def generate_sensor_trend_chart(self, 
                                   sensor_id: str, 
                                   start_time: datetime, 
                                   end_time: datetime,
                                   title: str = None) -> Optional[str]:
        """
        Generate a trend chart for sensor data.
        
        Args:
            sensor_id: Sensor identifier
            start_time: Start of time range
            end_time: End of time range
            title: Optional chart title
            
        Returns:
            Base64 encoded chart image or None if matplotlib not available
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib not available. Cannot generate charts.")
            return None
        
        try:
            # Get sensor data
            query = """
                SELECT value, timestamp, quality_score
                FROM sensor_readings 
                WHERE sensor_id = %s 
                AND timestamp >= %s 
                AND timestamp <= %s
                ORDER BY timestamp
            """
            results = self.analytics.client.execute_cql(query, [sensor_id, start_time, end_time])
            
            if not results:
                logger.warning(f"No data found for sensor {sensor_id}")
                return None
            
            # Prepare data
            timestamps = [row['timestamp'] for row in results]
            values = [row['value'] for row in results]
            quality_scores = [row['quality_score'] for row in results if row['quality_score'] is not None]
            
            # Create figure
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            # Plot sensor values
            ax1.plot(timestamps, values, 'b-', linewidth=2, alpha=0.7)
            ax1.set_ylabel('Sensor Value')
            ax1.set_title(title or f'Sensor {sensor_id} Trend Analysis')
            ax1.grid(True, alpha=0.3)
            
            # Format x-axis
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            ax1.xaxis.set_major_locator(mdates.HourLocator(interval=1))
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
            
            # Plot quality scores if available
            if quality_scores:
                quality_timestamps = [row['timestamp'] for row in results if row['quality_score'] is not None]
                ax2.plot(quality_timestamps, quality_scores, 'g-', linewidth=2, alpha=0.7)
                ax2.set_ylabel('Quality Score')
                ax2.set_xlabel('Time')
                ax2.grid(True, alpha=0.3)
                ax2.set_ylim(0, 100)
                
                # Format x-axis
                ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                ax2.xaxis.set_major_locator(mdates.HourLocator(interval=1))
                plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
            else:
                ax2.set_visible(False)
            
            plt.tight_layout()
            
            # Convert to base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return image_base64
            
        except Exception as e:
            logger.error(f"Failed to generate sensor trend chart: {e}")
            return None
    
    def generate_process_analytics_chart(self, process_id: str) -> Optional[str]:
        """
        Generate a comprehensive process analytics chart.
        
        Args:
            process_id: Process identifier
            
        Returns:
            Base64 encoded chart image or None if matplotlib not available
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib not available. Cannot generate charts.")
            return None
        
        try:
            # Get process analytics
            analytics = self.analytics.get_process_analytics(process_id)
            
            # Create figure with subplots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            # 1. Quality Score Gauge
            quality_score = analytics.quality_score
            ax1.pie([quality_score, 100-quality_score], 
                   labels=['Quality', 'Remaining'], 
                   colors=['green', 'lightgray'],
                   startangle=90, counterclock=False)
            ax1.set_title(f'Quality Score: {quality_score:.1f}%')
            
            # 2. Sensor Averages
            sensor_data = {
                'Temperature': analytics.avg_temperature,
                'Pressure': analytics.avg_pressure,
                'Vibration': analytics.avg_vibration
            }
            
            bars = ax2.bar(sensor_data.keys(), sensor_data.values(), 
                          color=['red', 'blue', 'orange'], alpha=0.7)
            ax2.set_title('Average Sensor Values')
            ax2.set_ylabel('Value')
            
            # Add value labels on bars
            for bar, value in zip(bars, sensor_data.values()):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        f'{value:.2f}', ha='center', va='bottom')
            
            # 3. Process Metrics
            metrics = {
                'Total Readings': analytics.total_sensor_readings,
                'Anomalies': analytics.anomalies_detected,
                'Alerts': analytics.alerts_triggered
            }
            
            bars = ax3.bar(metrics.keys(), metrics.values(), 
                          color=['blue', 'orange', 'red'], alpha=0.7)
            ax3.set_title('Process Metrics')
            ax3.set_ylabel('Count')
            
            # Add value labels
            for bar, value in zip(bars, metrics.values()):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        f'{value}', ha='center', va='bottom')
            
            # 4. Efficiency Score
            efficiency = analytics.efficiency_score
            ax4.barh(['Efficiency'], [efficiency], color='green' if efficiency > 70 else 'orange' if efficiency > 50 else 'red')
            ax4.set_xlim(0, 100)
            ax4.set_title(f'Efficiency Score: {efficiency:.1f}%')
            ax4.set_xlabel('Percentage')
            
            plt.suptitle(f'Process Analytics: {process_id}', fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            # Convert to base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return image_base64
            
        except Exception as e:
            logger.error(f"Failed to generate process analytics chart: {e}")
            return None
    
    def generate_machine_performance_chart(self, machine_id: str, days: int = 7) -> Optional[str]:
        """
        Generate a machine performance chart.
        
        Args:
            machine_id: Machine identifier
            days: Number of days to analyze
            
        Returns:
            Base64 encoded chart image or None if matplotlib not available
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib not available. Cannot generate charts.")
            return None
        
        try:
            # Get machine performance data
            performance = self.analytics.get_machine_performance(machine_id, days)
            
            # Create figure
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            # 1. Uptime and Efficiency
            uptime = performance.get('uptime_percentage', 0)
            efficiency = performance.get('avg_efficiency', 0)
            
            categories = ['Uptime', 'Efficiency']
            values = [uptime, efficiency]
            colors = ['blue', 'green']
            
            bars = ax1.bar(categories, values, color=colors, alpha=0.7)
            ax1.set_title('Machine Performance Overview')
            ax1.set_ylabel('Percentage')
            ax1.set_ylim(0, 100)
            
            # Add value labels
            for bar, value in zip(bars, values):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{value:.1f}%', ha='center', va='bottom')
            
            # 2. Operations and Maintenance
            operations = performance.get('total_operations', 0)
            maintenance = performance.get('maintenance_events', 0)
            
            ax2.pie([operations, maintenance], 
                   labels=['Operations', 'Maintenance'], 
                   colors=['lightblue', 'orange'],
                   autopct='%1.1f%%')
            ax2.set_title('Operations vs Maintenance')
            
            # 3. Sensor Statistics
            sensor_stats = performance.get('sensor_statistics', {})
            if sensor_stats:
                sensor_types = list(sensor_stats.keys())
                sensor_counts = [stats['count'] for stats in sensor_stats.values()]
                
                bars = ax3.bar(sensor_types, sensor_counts, alpha=0.7)
                ax3.set_title('Sensor Data Points')
                ax3.set_ylabel('Count')
                ax3.tick_params(axis='x', rotation=45)
                
                # Add value labels
                for bar, value in zip(bars, sensor_counts):
                    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                            f'{value}', ha='center', va='bottom')
            else:
                ax3.text(0.5, 0.5, 'No sensor data available', 
                        ha='center', va='center', transform=ax3.transAxes)
                ax3.set_title('Sensor Statistics')
            
            # 4. Performance Timeline (simplified)
            data_points = performance.get('data_points_analyzed', 0)
            days_analyzed = days
            
            ax4.bar(['Data Points', 'Days Analyzed'], [data_points, days_analyzed * 100], 
                   color=['purple', 'gray'], alpha=0.7)
            ax4.set_title('Analysis Summary')
            ax4.set_ylabel('Count')
            
            plt.suptitle(f'Machine Performance: {machine_id}', fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            # Convert to base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return image_base64
            
        except Exception as e:
            logger.error(f"Failed to generate machine performance chart: {e}")
            return None
    
    def generate_anomaly_analysis_chart(self, 
                                      start_time: datetime, 
                                      end_time: datetime) -> Optional[str]:
        """
        Generate an anomaly analysis chart.
        
        Args:
            start_time: Start of analysis period
            end_time: End of analysis period
            
        Returns:
            Base64 encoded chart image or None if matplotlib not available
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib not available. Cannot generate charts.")
            return None
        
        try:
            # Get anomaly analysis data
            analysis = self.analytics.get_anomaly_analysis(start_time, end_time)
            
            # Create figure
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            # 1. Anomaly Rate
            total_measurements = analysis.get('total_measurements', 0)
            total_anomalies = analysis.get('total_anomalies', 0)
            normal_measurements = total_measurements - total_anomalies
            
            if total_measurements > 0:
                ax1.pie([normal_measurements, total_anomalies], 
                       labels=['Normal', 'Anomalies'], 
                       colors=['green', 'red'],
                       autopct='%1.1f%%')
                ax1.set_title(f'Anomaly Rate: {analysis.get("anomaly_rate", 0):.2f}%')
            else:
                ax1.text(0.5, 0.5, 'No data available', 
                        ha='center', va='center', transform=ax1.transAxes)
                ax1.set_title('Anomaly Rate')
            
            # 2. Anomalies by Sensor Type
            anomalies_by_type = analysis.get('anomalies_by_sensor_type', {})
            if anomalies_by_type:
                sensor_types = list(anomalies_by_type.keys())
                anomaly_counts = list(anomalies_by_type.values())
                
                bars = ax2.bar(sensor_types, anomaly_counts, alpha=0.7)
                ax2.set_title('Anomalies by Sensor Type')
                ax2.set_ylabel('Count')
                ax2.tick_params(axis='x', rotation=45)
                
                # Add value labels
                for bar, value in zip(bars, anomaly_counts):
                    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                            f'{value}', ha='center', va='bottom')
            else:
                ax2.text(0.5, 0.5, 'No anomalies by sensor type', 
                        ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title('Anomalies by Sensor Type')
            
            # 3. Process Analysis
            processes_analyzed = analysis.get('processes_analyzed', 0)
            processes_with_anomalies = analysis.get('processes_with_anomalies', 0)
            processes_normal = processes_analyzed - processes_with_anomalies
            
            if processes_analyzed > 0:
                ax3.pie([processes_normal, processes_with_anomalies], 
                       labels=['Normal Processes', 'Processes with Anomalies'], 
                       colors=['lightgreen', 'orange'],
                       autopct='%1.1f%%')
                ax3.set_title('Process Anomaly Distribution')
            else:
                ax3.text(0.5, 0.5, 'No processes analyzed', 
                        ha='center', va='center', transform=ax3.transAxes)
                ax3.set_title('Process Anomaly Distribution')
            
            # 4. Top Anomaly Processes
            top_anomalies = analysis.get('top_anomaly_processes', [])
            if top_anomalies:
                # Take top 5
                top_5 = top_anomalies[:5]
                process_ids = [f"Process {i+1}" for i in range(len(top_5))]
                anomaly_counts = [count for _, count in top_5]
                
                bars = ax4.bar(process_ids, anomaly_counts, alpha=0.7, color='red')
                ax4.set_title('Top Anomaly Processes')
                ax4.set_ylabel('Anomaly Count')
                ax4.tick_params(axis='x', rotation=45)
                
                # Add value labels
                for bar, value in zip(bars, anomaly_counts):
                    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                            f'{value}', ha='center', va='bottom')
            else:
                ax4.text(0.5, 0.5, 'No top anomaly processes', 
                        ha='center', va='center', transform=ax4.transAxes)
                ax4.set_title('Top Anomaly Processes')
            
            plt.suptitle('Anomaly Analysis Report', fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            # Convert to base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return image_base64
            
        except Exception as e:
            logger.error(f"Failed to generate anomaly analysis chart: {e}")
            return None
    
    def generate_dashboard_summary_chart(self) -> Optional[str]:
        """
        Generate a real-time dashboard summary chart.
        
        Returns:
            Base64 encoded chart image or None if matplotlib not available
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib not available. Cannot generate charts.")
            return None
        
        try:
            # Get dashboard data
            dashboard_data = self.analytics.get_real_time_dashboard_data()
            
            # Create figure
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            # 1. System Overview
            active_processes = dashboard_data.get('active_processes', 0)
            overall_quality = dashboard_data.get('overall_quality_score', 0)
            
            ax1.bar(['Active Processes', 'Quality Score'], 
                   [active_processes, overall_quality], 
                   color=['blue', 'green'], alpha=0.7)
            ax1.set_title('System Overview')
            ax1.set_ylabel('Count/Score')
            
            # Add value labels
            ax1.text(0, active_processes + 0.1, f'{active_processes}', ha='center', va='bottom')
            ax1.text(1, overall_quality + 0.1, f'{overall_quality:.1f}', ha='center', va='bottom')
            
            # 2. Sensor Averages
            sensor_averages = dashboard_data.get('sensor_averages', {})
            if sensor_averages:
                sensor_types = list(sensor_averages.keys())
                sensor_values = [data['current_avg'] for data in sensor_averages.values()]
                
                bars = ax2.bar(sensor_types, sensor_values, alpha=0.7)
                ax2.set_title('Current Sensor Averages')
                ax2.set_ylabel('Value')
                ax2.tick_params(axis='x', rotation=45)
                
                # Add value labels
                for bar, value in zip(bars, sensor_values):
                    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                            f'{value:.2f}', ha='center', va='bottom')
            else:
                ax2.text(0.5, 0.5, 'No sensor data available', 
                        ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title('Current Sensor Averages')
            
            # 3. Alert Summary
            alert_summary = dashboard_data.get('alert_summary', {})
            total_alerts = alert_summary.get('total_alerts', 0)
            by_severity = alert_summary.get('by_severity', {})
            
            if by_severity:
                severities = list(by_severity.keys())
                alert_counts = list(by_severity.values())
                
                bars = ax3.bar(severities, alert_counts, alpha=0.7)
                ax3.set_title('Alerts by Severity')
                ax3.set_ylabel('Count')
                
                # Add value labels
                for bar, value in zip(bars, alert_counts):
                    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                            f'{value}', ha='center', va='bottom')
            else:
                ax3.text(0.5, 0.5, 'No alerts available', 
                        ha='center', va='center', transform=ax3.transAxes)
                ax3.set_title('Alerts by Severity')
            
            # 4. Machine Status
            machine_status = dashboard_data.get('machine_status', {})
            if machine_status:
                status_counts = {}
                for status in machine_status.values():
                    status_counts[status] = status_counts.get(status, 0) + 1
                
                if status_counts:
                    statuses = list(status_counts.keys())
                    counts = list(status_counts.values())
                    
                    ax4.pie(counts, labels=statuses, autopct='%1.1f%%')
                    ax4.set_title('Machine Status Distribution')
                else:
                    ax4.text(0.5, 0.5, 'No machine status data', 
                            ha='center', va='center', transform=ax4.transAxes)
                    ax4.set_title('Machine Status Distribution')
            else:
                ax4.text(0.5, 0.5, 'No machine status data', 
                        ha='center', va='center', transform=ax4.transAxes)
                ax4.set_title('Machine Status Distribution')
            
            plt.suptitle('Real-Time Dashboard Summary', fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            # Convert to base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return image_base64
            
        except Exception as e:
            logger.error(f"Failed to generate dashboard summary chart: {e}")
            return None
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
