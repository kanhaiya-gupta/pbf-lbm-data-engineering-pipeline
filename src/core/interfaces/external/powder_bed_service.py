"""
Powder bed service interface for PBF-LB/M operations.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from datetime import datetime


class PowderBedService(ABC):
    """
    Interface for powder bed services in PBF-LB/M operations.
    
    This interface defines the contract for powder bed management,
    preparation, monitoring, and maintenance operations.
    """
    
    @abstractmethod
    async def prepare_bed(self, bed_config: Dict[str, Any]) -> str:
        """
        Prepare a powder bed.
        
        Args:
            bed_config: Configuration for bed preparation
            
        Returns:
            Bed ID
            
        Raises:
            PowderBedException: If bed preparation fails
        """
        pass
    
    @abstractmethod
    async def activate_bed(self, bed_id: str) -> bool:
        """
        Activate a powder bed.
        
        Args:
            bed_id: The bed ID to activate
            
        Returns:
            True if bed activated successfully
            
        Raises:
            PowderBedException: If bed activation fails
        """
        pass
    
    @abstractmethod
    async def deactivate_bed(self, bed_id: str) -> bool:
        """
        Deactivate a powder bed.
        
        Args:
            bed_id: The bed ID to deactivate
            
        Returns:
            True if bed deactivated successfully
            
        Raises:
            PowderBedException: If bed deactivation fails
        """
        pass
    
    @abstractmethod
    async def get_bed_status(self, bed_id: str) -> Dict[str, Any]:
        """
        Get status of a powder bed.
        
        Args:
            bed_id: The bed ID to check
            
        Returns:
            Dictionary containing bed status
            
        Raises:
            PowderBedException: If status retrieval fails
        """
        pass
    
    @abstractmethod
    async def get_bed_quality(self, bed_id: str) -> Dict[str, Any]:
        """
        Get quality assessment of a powder bed.
        
        Args:
            bed_id: The bed ID to assess
            
        Returns:
            Dictionary containing bed quality assessment
            
        Raises:
            PowderBedException: If quality assessment fails
        """
        pass
    
    @abstractmethod
    async def check_bed_quality(self, bed_id: str, quality_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Perform quality check on a powder bed.
        
        Args:
            bed_id: The bed ID to check
            quality_config: Optional quality check configuration
            
        Returns:
            Dictionary containing quality check results
            
        Raises:
            PowderBedException: If quality check fails
        """
        pass
    
    @abstractmethod
    async def clean_bed(self, bed_id: str, cleaning_config: Dict[str, Any]) -> bool:
        """
        Clean a powder bed.
        
        Args:
            bed_id: The bed ID to clean
            cleaning_config: Configuration for bed cleaning
            
        Returns:
            True if bed cleaned successfully
            
        Raises:
            PowderBedException: If bed cleaning fails
        """
        pass
    
    @abstractmethod
    async def refill_powder(self, bed_id: str, powder_config: Dict[str, Any]) -> bool:
        """
        Refill powder in a bed.
        
        Args:
            bed_id: The bed ID to refill
            powder_config: Configuration for powder refill
            
        Returns:
            True if powder refilled successfully
            
        Raises:
            PowderBedException: If powder refill fails
        """
        pass
    
    @abstractmethod
    async def get_powder_level(self, bed_id: str) -> Dict[str, Any]:
        """
        Get powder level in a bed.
        
        Args:
            bed_id: The bed ID to check
            
        Returns:
            Dictionary containing powder level information
            
        Raises:
            PowderBedException: If powder level check fails
        """
        pass
    
    @abstractmethod
    async def monitor_bed(self, bed_id: str, monitoring_config: Dict[str, Any]) -> str:
        """
        Start monitoring a powder bed.
        
        Args:
            bed_id: The bed ID to monitor
            monitoring_config: Configuration for bed monitoring
            
        Returns:
            Monitoring session ID
            
        Raises:
            PowderBedException: If monitoring start fails
        """
        pass
    
    @abstractmethod
    async def stop_monitoring(self, monitoring_session_id: str) -> bool:
        """
        Stop monitoring a powder bed.
        
        Args:
            monitoring_session_id: The monitoring session ID to stop
            
        Returns:
            True if monitoring stopped successfully
            
        Raises:
            PowderBedException: If monitoring stop fails
        """
        pass
    
    @abstractmethod
    async def get_monitoring_data(self, monitoring_session_id: str) -> List[Dict[str, Any]]:
        """
        Get monitoring data for a bed.
        
        Args:
            monitoring_session_id: The monitoring session ID
            
        Returns:
            List of monitoring data points
            
        Raises:
            PowderBedException: If monitoring data retrieval fails
        """
        pass
    
    @abstractmethod
    async def detect_disturbance(self, bed_id: str, disturbance_config: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Detect disturbances in a powder bed.
        
        Args:
            bed_id: The bed ID to check
            disturbance_config: Optional disturbance detection configuration
            
        Returns:
            List of detected disturbances
            
        Raises:
            PowderBedException: If disturbance detection fails
        """
        pass
    
    @abstractmethod
    async def get_bed_temperature(self, bed_id: str) -> float:
        """
        Get temperature of a powder bed.
        
        Args:
            bed_id: The bed ID to check
            
        Returns:
            Bed temperature in Celsius
            
        Raises:
            PowderBedException: If temperature retrieval fails
        """
        pass
    
    @abstractmethod
    async def set_bed_temperature(self, bed_id: str, temperature: float) -> bool:
        """
        Set temperature of a powder bed.
        
        Args:
            bed_id: The bed ID to set temperature for
            temperature: Target temperature in Celsius
            
        Returns:
            True if temperature set successfully
            
        Raises:
            PowderBedException: If temperature setting fails
        """
        pass
    
    @abstractmethod
    async def get_bed_dimensions(self, bed_id: str) -> Dict[str, float]:
        """
        Get dimensions of a powder bed.
        
        Args:
            bed_id: The bed ID to check
            
        Returns:
            Dictionary containing bed dimensions
            
        Raises:
            PowderBedException: If dimension retrieval fails
        """
        pass
    
    @abstractmethod
    async def get_bed_material_info(self, bed_id: str) -> Dict[str, Any]:
        """
        Get material information for a powder bed.
        
        Args:
            bed_id: The bed ID to check
            
        Returns:
            Dictionary containing material information
            
        Raises:
            PowderBedException: If material info retrieval fails
        """
        pass
    
    @abstractmethod
    async def get_bed_statistics(self, bed_id: str) -> Dict[str, Any]:
        """
        Get statistics for a powder bed.
        
        Args:
            bed_id: The bed ID to get statistics for
            
        Returns:
            Dictionary containing bed statistics
            
        Raises:
            PowderBedException: If statistics retrieval fails
        """
        pass
    
    @abstractmethod
    async def get_bed_history(self, bed_id: str, time_range: Optional[Dict[str, datetime]] = None) -> List[Dict[str, Any]]:
        """
        Get history for a powder bed.
        
        Args:
            bed_id: The bed ID to get history for
            time_range: Optional time range for history
            
        Returns:
            List of historical bed data
            
        Raises:
            PowderBedException: If history retrieval fails
        """
        pass
    
    @abstractmethod
    async def get_bed_trends(self, bed_id: str, time_period: str = "month") -> Dict[str, Any]:
        """
        Get trends for a powder bed.
        
        Args:
            bed_id: The bed ID to get trends for
            time_period: Time period for trend analysis
            
        Returns:
            Dictionary containing bed trends
            
        Raises:
            PowderBedException: If trend analysis fails
        """
        pass
    
    @abstractmethod
    async def get_bed_analytics(self, bed_id: str) -> Dict[str, Any]:
        """
        Get analytics for a powder bed.
        
        Args:
            bed_id: The bed ID to get analytics for
            
        Returns:
            Dictionary containing bed analytics
            
        Raises:
            PowderBedException: If analytics retrieval fails
        """
        pass
    
    @abstractmethod
    async def get_bed_recommendations(self, bed_id: str) -> List[str]:
        """
        Get recommendations for a powder bed.
        
        Args:
            bed_id: The bed ID to get recommendations for
            
        Returns:
            List of bed recommendations
            
        Raises:
            PowderBedException: If recommendations retrieval fails
        """
        pass
    
    @abstractmethod
    async def get_bed_maintenance_schedule(self, bed_id: str) -> Dict[str, Any]:
        """
        Get maintenance schedule for a powder bed.
        
        Args:
            bed_id: The bed ID to get schedule for
            
        Returns:
            Dictionary containing maintenance schedule
            
        Raises:
            PowderBedException: If schedule retrieval fails
        """
        pass
    
    @abstractmethod
    async def schedule_maintenance(self, bed_id: str, maintenance_config: Dict[str, Any]) -> str:
        """
        Schedule maintenance for a powder bed.
        
        Args:
            bed_id: The bed ID to schedule maintenance for
            maintenance_config: Configuration for maintenance
            
        Returns:
            Maintenance ID
            
        Raises:
            PowderBedException: If maintenance scheduling fails
        """
        pass
    
    @abstractmethod
    async def get_bed_alerts(self, bed_id: str, severity: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get alerts for a powder bed.
        
        Args:
            bed_id: The bed ID to get alerts for
            severity: Optional severity level to filter alerts
            
        Returns:
            List of bed alerts
            
        Raises:
            PowderBedException: If alert retrieval fails
        """
        pass
    
    @abstractmethod
    async def create_bed_alert(self, bed_id: str, alert_data: Dict[str, Any]) -> str:
        """
        Create an alert for a powder bed.
        
        Args:
            bed_id: The bed ID to create alert for
            alert_data: The alert data to create
            
        Returns:
            Alert ID
            
        Raises:
            PowderBedException: If alert creation fails
        """
        pass
    
    @abstractmethod
    async def resolve_bed_alert(self, alert_id: str, resolution_data: Dict[str, Any]) -> bool:
        """
        Resolve an alert for a powder bed.
        
        Args:
            alert_id: The ID of the alert to resolve
            resolution_data: The resolution data
            
        Returns:
            True if alert resolved successfully
            
        Raises:
            PowderBedException: If alert resolution fails
        """
        pass
    
    @abstractmethod
    async def get_bed_dashboard_data(self, bed_id: str) -> Dict[str, Any]:
        """
        Get data for powder bed dashboard.
        
        Args:
            bed_id: The bed ID to get dashboard data for
            
        Returns:
            Dictionary containing dashboard data
            
        Raises:
            PowderBedException: If dashboard data retrieval fails
        """
        pass
    
    @abstractmethod
    async def export_bed_data(self, bed_id: str, export_config: Dict[str, Any]) -> str:
        """
        Export data for a powder bed.
        
        Args:
            bed_id: The bed ID to export data for
            export_config: Configuration for data export
            
        Returns:
            Export file path or URL
            
        Raises:
            PowderBedException: If data export fails
        """
        pass
    
    @abstractmethod
    async def get_service_health(self) -> Dict[str, Any]:
        """
        Get health status of the powder bed service.
        
        Returns:
            Dictionary containing health status
        """
        pass
    
    @abstractmethod
    async def get_service_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics of the powder bed service.
        
        Returns:
            Dictionary containing service metrics
        """
        pass
