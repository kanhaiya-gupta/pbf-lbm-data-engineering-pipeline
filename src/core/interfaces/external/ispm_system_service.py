"""
ISPM system service interface for PBF-LB/M operations.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from datetime import datetime

from ...domain.enums import MonitoringType


class ISPMSystemService(ABC):
    """
    Interface for ISPM (In-Situ Process Monitoring) system services.
    
    This interface defines the contract for ISPM system integration,
    sensor management, and real-time monitoring operations.
    """
    
    @abstractmethod
    async def start_monitoring(self, monitoring_config: Dict[str, Any]) -> str:
        """
        Start ISPM monitoring session.
        
        Args:
            monitoring_config: Configuration for the monitoring session
            
        Returns:
            Monitoring session ID
            
        Raises:
            ISPMException: If monitoring start fails
        """
        pass
    
    @abstractmethod
    async def stop_monitoring(self, session_id: str) -> bool:
        """
        Stop ISPM monitoring session.
        
        Args:
            session_id: The monitoring session ID to stop
            
        Returns:
            True if monitoring stopped successfully
            
        Raises:
            ISPMException: If monitoring stop fails
        """
        pass
    
    @abstractmethod
    async def pause_monitoring(self, session_id: str) -> bool:
        """
        Pause ISPM monitoring session.
        
        Args:
            session_id: The monitoring session ID to pause
            
        Returns:
            True if monitoring paused successfully
            
        Raises:
            ISPMException: If monitoring pause fails
        """
        pass
    
    @abstractmethod
    async def resume_monitoring(self, session_id: str) -> bool:
        """
        Resume ISPM monitoring session.
        
        Args:
            session_id: The monitoring session ID to resume
            
        Returns:
            True if monitoring resumed successfully
            
        Raises:
            ISPMException: If monitoring resume fails
        """
        pass
    
    @abstractmethod
    async def get_monitoring_status(self, session_id: str) -> Dict[str, Any]:
        """
        Get status of ISPM monitoring session.
        
        Args:
            session_id: The monitoring session ID to check
            
        Returns:
            Dictionary containing monitoring status
            
        Raises:
            ISPMException: If status retrieval fails
        """
        pass
    
    @abstractmethod
    async def get_sensor_data(self, session_id: str, sensor_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get sensor data from ISPM monitoring session.
        
        Args:
            session_id: The monitoring session ID
            sensor_id: Optional specific sensor ID to get data for
            
        Returns:
            List of sensor data points
            
        Raises:
            ISPMException: If data retrieval fails
        """
        pass
    
    @abstractmethod
    async def get_real_time_data(self, session_id: str) -> Dict[str, Any]:
        """
        Get real-time data from ISPM monitoring session.
        
        Args:
            session_id: The monitoring session ID
            
        Returns:
            Dictionary containing real-time data
            
        Raises:
            ISPMException: If real-time data retrieval fails
        """
        pass
    
    @abstractmethod
    async def configure_sensors(self, session_id: str, sensor_config: Dict[str, Any]) -> bool:
        """
        Configure sensors for ISPM monitoring session.
        
        Args:
            session_id: The monitoring session ID
            sensor_config: Configuration for sensors
            
        Returns:
            True if sensors configured successfully
            
        Raises:
            ISPMException: If sensor configuration fails
        """
        pass
    
    @abstractmethod
    async def get_sensor_status(self, session_id: str) -> Dict[str, str]:
        """
        Get status of all sensors in monitoring session.
        
        Args:
            session_id: The monitoring session ID
            
        Returns:
            Dictionary mapping sensor IDs to their status
            
        Raises:
            ISPMException: If sensor status retrieval fails
        """
        pass
    
    @abstractmethod
    async def calibrate_sensor(self, session_id: str, sensor_id: str, calibration_data: Dict[str, Any]) -> bool:
        """
        Calibrate a sensor in the monitoring session.
        
        Args:
            session_id: The monitoring session ID
            sensor_id: The sensor ID to calibrate
            calibration_data: Calibration data and parameters
            
        Returns:
            True if sensor calibrated successfully
            
        Raises:
            ISPMException: If sensor calibration fails
        """
        pass
    
    @abstractmethod
    async def set_monitoring_thresholds(self, session_id: str, thresholds: Dict[str, Any]) -> bool:
        """
        Set monitoring thresholds for the session.
        
        Args:
            session_id: The monitoring session ID
            thresholds: Threshold values for monitoring
            
        Returns:
            True if thresholds set successfully
            
        Raises:
            ISPMException: If threshold setting fails
        """
        pass
    
    @abstractmethod
    async def get_monitoring_thresholds(self, session_id: str) -> Dict[str, Any]:
        """
        Get monitoring thresholds for the session.
        
        Args:
            session_id: The monitoring session ID
            
        Returns:
            Dictionary containing monitoring thresholds
            
        Raises:
            ISPMException: If threshold retrieval fails
        """
        pass
    
    @abstractmethod
    async def detect_anomalies(self, session_id: str, anomaly_config: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Detect anomalies in monitoring data.
        
        Args:
            session_id: The monitoring session ID
            anomaly_config: Optional anomaly detection configuration
            
        Returns:
            List of detected anomalies
            
        Raises:
            ISPMException: If anomaly detection fails
        """
        pass
    
    @abstractmethod
    async def check_threshold_violations(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Check for threshold violations in monitoring data.
        
        Args:
            session_id: The monitoring session ID
            
        Returns:
            List of threshold violations
            
        Raises:
            ISPMException: If threshold check fails
        """
        pass
    
    @abstractmethod
    async def get_monitoring_statistics(self, session_id: str) -> Dict[str, Any]:
        """
        Get statistics for monitoring session.
        
        Args:
            session_id: The monitoring session ID
            
        Returns:
            Dictionary containing monitoring statistics
            
        Raises:
            ISPMException: If statistics retrieval fails
        """
        pass
    
    @abstractmethod
    async def get_monitoring_analytics(self, session_id: str) -> Dict[str, Any]:
        """
        Get analytics for monitoring session.
        
        Args:
            session_id: The monitoring session ID
            
        Returns:
            Dictionary containing monitoring analytics
            
        Raises:
            ISPMException: If analytics retrieval fails
        """
        pass
    
    @abstractmethod
    async def export_monitoring_data(self, session_id: str, export_config: Dict[str, Any]) -> str:
        """
        Export monitoring data.
        
        Args:
            session_id: The monitoring session ID
            export_config: Configuration for data export
            
        Returns:
            Export file path or URL
            
        Raises:
            ISPMException: If data export fails
        """
        pass
    
    @abstractmethod
    async def get_monitoring_history(self, session_id: str, time_range: Optional[Dict[str, datetime]] = None) -> List[Dict[str, Any]]:
        """
        Get monitoring history for session.
        
        Args:
            session_id: The monitoring session ID
            time_range: Optional time range for history
            
        Returns:
            List of historical monitoring data
            
        Raises:
            ISPMException: If history retrieval fails
        """
        pass
    
    @abstractmethod
    async def get_monitoring_trends(self, session_id: str, time_period: str = "hour") -> Dict[str, Any]:
        """
        Get monitoring trends for session.
        
        Args:
            session_id: The monitoring session ID
            time_period: Time period for trend analysis
            
        Returns:
            Dictionary containing monitoring trends
            
        Raises:
            ISPMException: If trend analysis fails
        """
        pass
    
    @abstractmethod
    async def get_monitoring_alerts(self, session_id: str, severity: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get monitoring alerts for session.
        
        Args:
            session_id: The monitoring session ID
            severity: Optional severity level to filter alerts
            
        Returns:
            List of monitoring alerts
            
        Raises:
            ISPMException: If alert retrieval fails
        """
        pass
    
    @abstractmethod
    async def create_monitoring_alert(self, session_id: str, alert_data: Dict[str, Any]) -> str:
        """
        Create a monitoring alert.
        
        Args:
            session_id: The monitoring session ID
            alert_data: The alert data to create
            
        Returns:
            Alert ID
            
        Raises:
            ISPMException: If alert creation fails
        """
        pass
    
    @abstractmethod
    async def resolve_monitoring_alert(self, alert_id: str, resolution_data: Dict[str, Any]) -> bool:
        """
        Resolve a monitoring alert.
        
        Args:
            alert_id: The ID of the alert to resolve
            resolution_data: The resolution data
            
        Returns:
            True if alert resolved successfully
            
        Raises:
            ISPMException: If alert resolution fails
        """
        pass
    
    @abstractmethod
    async def get_monitoring_dashboard_data(self, session_id: str) -> Dict[str, Any]:
        """
        Get data for monitoring dashboard.
        
        Args:
            session_id: The monitoring session ID
            
        Returns:
            Dictionary containing dashboard data
            
        Raises:
            ISPMException: If dashboard data retrieval fails
        """
        pass
    
    @abstractmethod
    async def get_supported_monitoring_types(self) -> List[MonitoringType]:
        """
        Get supported monitoring types.
        
        Returns:
            List of supported monitoring types
        """
        pass
    
    @abstractmethod
    async def get_monitoring_capabilities(self) -> Dict[str, Any]:
        """
        Get monitoring system capabilities.
        
        Returns:
            Dictionary containing monitoring capabilities
        """
        pass
    
    @abstractmethod
    async def get_service_health(self) -> Dict[str, Any]:
        """
        Get health status of the ISPM system service.
        
        Returns:
            Dictionary containing health status
        """
        pass
    
    @abstractmethod
    async def get_service_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics of the ISPM system service.
        
        Returns:
            Dictionary containing service metrics
        """
        pass
