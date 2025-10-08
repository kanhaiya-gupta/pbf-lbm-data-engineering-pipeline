"""
ISPM monitoring repository interface.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from datetime import datetime

from ...domain.entities.ispm_monitoring import ISPMMonitoring
from ...domain.enums import MonitoringType
from .base_repository import BaseRepository


class ISPMMonitoringRepository(BaseRepository[ISPMMonitoring]):
    """
    Repository interface for ISPM monitoring entities.
    
    This interface defines the contract for ISPM monitoring data access operations
    with domain-specific methods for monitoring session management.
    """
    
    @abstractmethod
    async def get_by_monitoring_name(self, monitoring_name: str) -> Optional[ISPMMonitoring]:
        """
        Get an ISPM monitoring session by its name.
        
        Args:
            monitoring_name: The name of the monitoring session to retrieve
            
        Returns:
            The monitoring session if found, None otherwise
        """
        pass
    
    @abstractmethod
    async def get_by_monitoring_session_id(self, session_id: str) -> Optional[ISPMMonitoring]:
        """
        Get an ISPM monitoring session by its session ID.
        
        Args:
            session_id: The session ID to retrieve
            
        Returns:
            The monitoring session if found, None otherwise
        """
        pass
    
    @abstractmethod
    async def get_by_process_id(self, process_id: str) -> List[ISPMMonitoring]:
        """
        Get all ISPM monitoring sessions for a specific process.
        
        Args:
            process_id: The process ID to filter by
            
        Returns:
            List of monitoring sessions for the process
        """
        pass
    
    @abstractmethod
    async def get_by_build_id(self, build_id: str) -> List[ISPMMonitoring]:
        """
        Get all ISPM monitoring sessions for a specific build.
        
        Args:
            build_id: The build ID to filter by
            
        Returns:
            List of monitoring sessions for the build
        """
        pass
    
    @abstractmethod
    async def get_by_monitoring_type(self, monitoring_type: MonitoringType) -> List[ISPMMonitoring]:
        """
        Get all ISPM monitoring sessions of a specific type.
        
        Args:
            monitoring_type: The monitoring type to filter by
            
        Returns:
            List of monitoring sessions of the specified type
        """
        pass
    
    @abstractmethod
    async def get_by_sensor_id(self, sensor_id: str) -> List[ISPMMonitoring]:
        """
        Get all ISPM monitoring sessions for a specific sensor.
        
        Args:
            sensor_id: The sensor ID to filter by
            
        Returns:
            List of monitoring sessions for the sensor
        """
        pass
    
    @abstractmethod
    async def get_active_monitoring_sessions(self) -> List[ISPMMonitoring]:
        """
        Get all currently active ISPM monitoring sessions.
        
        Returns:
            List of active monitoring sessions
        """
        pass
    
    @abstractmethod
    async def get_paused_monitoring_sessions(self) -> List[ISPMMonitoring]:
        """
        Get all currently paused ISPM monitoring sessions.
        
        Returns:
            List of paused monitoring sessions
        """
        pass
    
    @abstractmethod
    async def get_completed_monitoring_sessions(self) -> List[ISPMMonitoring]:
        """
        Get all completed ISPM monitoring sessions.
        
        Returns:
            List of completed monitoring sessions
        """
        pass
    
    @abstractmethod
    async def get_monitoring_sessions_by_date_range(self, start_date: datetime, end_date: datetime) -> List[ISPMMonitoring]:
        """
        Get ISPM monitoring sessions within a date range.
        
        Args:
            start_date: Start date for the range
            end_date: End date for the range
            
        Returns:
            List of monitoring sessions within the date range
        """
        pass
    
    @abstractmethod
    async def get_monitoring_sessions_by_duration_range(self, min_duration: float, max_duration: float) -> List[ISPMMonitoring]:
        """
        Get ISPM monitoring sessions within a duration range.
        
        Args:
            min_duration: Minimum duration in seconds
            max_duration: Maximum duration in seconds
            
        Returns:
            List of monitoring sessions within the duration range
        """
        pass
    
    @abstractmethod
    async def get_monitoring_sessions_by_data_points_range(self, min_points: int, max_points: int) -> List[ISPMMonitoring]:
        """
        Get ISPM monitoring sessions within a data points range.
        
        Args:
            min_points: Minimum number of data points
            max_points: Maximum number of data points
            
        Returns:
            List of monitoring sessions within the data points range
        """
        pass
    
    @abstractmethod
    async def get_monitoring_sessions_by_anomaly_count_range(self, min_anomalies: int, max_anomalies: int) -> List[ISPMMonitoring]:
        """
        Get ISPM monitoring sessions within an anomaly count range.
        
        Args:
            min_anomalies: Minimum number of anomalies
            max_anomalies: Maximum number of anomalies
            
        Returns:
            List of monitoring sessions within the anomaly count range
        """
        pass
    
    @abstractmethod
    async def get_monitoring_sessions_by_threshold_violations_range(self, min_violations: int, max_violations: int) -> List[ISPMMonitoring]:
        """
        Get ISPM monitoring sessions within a threshold violations range.
        
        Args:
            min_violations: Minimum number of threshold violations
            max_violations: Maximum number of threshold violations
            
        Returns:
            List of monitoring sessions within the violations range
        """
        pass
    
    @abstractmethod
    async def get_high_anomaly_monitoring_sessions(self, threshold: int = 10) -> List[ISPMMonitoring]:
        """
        Get ISPM monitoring sessions with high anomaly counts.
        
        Args:
            threshold: Minimum anomaly count threshold
            
        Returns:
            List of monitoring sessions with high anomaly counts
        """
        pass
    
    @abstractmethod
    async def get_high_violation_monitoring_sessions(self, threshold: int = 5) -> List[ISPMMonitoring]:
        """
        Get ISPM monitoring sessions with high threshold violation counts.
        
        Args:
            threshold: Minimum violation count threshold
            
        Returns:
            List of monitoring sessions with high violation counts
        """
        pass
    
    @abstractmethod
    async def get_monitoring_sessions_with_failed_sensors(self) -> List[ISPMMonitoring]:
        """
        Get ISPM monitoring sessions with failed sensors.
        
        Returns:
            List of monitoring sessions with failed sensors
        """
        pass
    
    @abstractmethod
    async def get_monitoring_sessions_with_stale_data(self, max_age_minutes: float = 5) -> List[ISPMMonitoring]:
        """
        Get ISPM monitoring sessions with stale data.
        
        Args:
            max_age_minutes: Maximum age in minutes for data to be considered fresh
            
        Returns:
            List of monitoring sessions with stale data
        """
        pass
    
    @abstractmethod
    async def get_monitoring_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about ISPM monitoring sessions.
        
        Returns:
            Dictionary containing monitoring statistics
        """
        pass
    
    @abstractmethod
    async def get_monitoring_analytics(self, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Get analytics and insights about ISPM monitoring sessions.
        
        Args:
            filters: Optional filters to apply to the analytics
            
        Returns:
            Dictionary containing monitoring analytics
        """
        pass
    
    @abstractmethod
    async def get_monitoring_trends(self, time_period: str = "month") -> Dict[str, Any]:
        """
        Get trends and patterns in ISPM monitoring over time.
        
        Args:
            time_period: Time period for trend analysis (day, week, month, year)
            
        Returns:
            Dictionary containing monitoring trends
        """
        pass
    
    @abstractmethod
    async def get_sensor_performance_analysis(self, sensor_id: str) -> Dict[str, Any]:
        """
        Get performance analysis for a specific sensor.
        
        Args:
            sensor_id: The ID of the sensor to analyze
            
        Returns:
            Dictionary containing sensor performance analysis
        """
        pass
    
    @abstractmethod
    async def get_anomaly_pattern_analysis(self, monitoring_session_id: str) -> Dict[str, Any]:
        """
        Get anomaly pattern analysis for a monitoring session.
        
        Args:
            monitoring_session_id: The ID of the monitoring session to analyze
            
        Returns:
            Dictionary containing anomaly pattern analysis
        """
        pass
    
    @abstractmethod
    async def get_threshold_effectiveness_analysis(self, monitoring_session_id: str) -> Dict[str, Any]:
        """
        Get threshold effectiveness analysis for a monitoring session.
        
        Args:
            monitoring_session_id: The ID of the monitoring session to analyze
            
        Returns:
            Dictionary containing threshold effectiveness analysis
        """
        pass
    
    @abstractmethod
    async def get_monitoring_recommendations(self, monitoring_session_id: str) -> List[str]:
        """
        Get recommendations for improving a monitoring session.
        
        Args:
            monitoring_session_id: The ID of the monitoring session to get recommendations for
            
        Returns:
            List of improvement recommendations
        """
        pass
    
    @abstractmethod
    async def get_monitoring_session_comparison(self, session_ids: List[str]) -> Dict[str, Any]:
        """
        Compare multiple ISPM monitoring sessions.
        
        Args:
            session_ids: List of monitoring session IDs to compare
            
        Returns:
            Dictionary containing monitoring session comparison data
        """
        pass
    
    @abstractmethod
    async def get_monitoring_impact_analysis(self, monitoring_session_id: str) -> Dict[str, Any]:
        """
        Get impact analysis for an ISPM monitoring session.
        
        Args:
            monitoring_session_id: The ID of the monitoring session to analyze
            
        Returns:
            Dictionary containing impact analysis results
        """
        pass
    
    @abstractmethod
    async def get_monitoring_quality_assessment(self, monitoring_session_id: str) -> Dict[str, Any]:
        """
        Get quality assessment for an ISPM monitoring session.
        
        Args:
            monitoring_session_id: The ID of the monitoring session to assess
            
        Returns:
            Dictionary containing quality assessment results
        """
        pass
