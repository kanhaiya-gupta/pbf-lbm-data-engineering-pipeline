"""
CT scanner service interface for PBF-LB/M operations.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from datetime import datetime


class CTScannerService(ABC):
    """
    Interface for CT (Computed Tomography) scanner services.
    
    This interface defines the contract for CT scanner integration,
    scan management, and image analysis operations.
    """
    
    @abstractmethod
    async def start_scan(self, scan_config: Dict[str, Any]) -> str:
        """
        Start a CT scan.
        
        Args:
            scan_config: Configuration for the scan
            
        Returns:
            Scan ID
            
        Raises:
            CTScannerException: If scan start fails
        """
        pass
    
    @abstractmethod
    async def stop_scan(self, scan_id: str) -> bool:
        """
        Stop a CT scan.
        
        Args:
            scan_id: The scan ID to stop
            
        Returns:
            True if scan stopped successfully
            
        Raises:
            CTScannerException: If scan stop fails
        """
        pass
    
    @abstractmethod
    async def pause_scan(self, scan_id: str) -> bool:
        """
        Pause a CT scan.
        
        Args:
            scan_id: The scan ID to pause
            
        Returns:
            True if scan paused successfully
            
        Raises:
            CTScannerException: If scan pause fails
        """
        pass
    
    @abstractmethod
    async def resume_scan(self, scan_id: str) -> bool:
        """
        Resume a CT scan.
        
        Args:
            scan_id: The scan ID to resume
            
        Returns:
            True if scan resumed successfully
            
        Raises:
            CTScannerException: If scan resume fails
        """
        pass
    
    @abstractmethod
    async def get_scan_status(self, scan_id: str) -> Dict[str, Any]:
        """
        Get status of a CT scan.
        
        Args:
            scan_id: The scan ID to check
            
        Returns:
            Dictionary containing scan status
            
        Raises:
            CTScannerException: If status retrieval fails
        """
        pass
    
    @abstractmethod
    async def get_scan_progress(self, scan_id: str) -> Dict[str, Any]:
        """
        Get progress of a CT scan.
        
        Args:
            scan_id: The scan ID to check
            
        Returns:
            Dictionary containing scan progress
            
        Raises:
            CTScannerException: If progress retrieval fails
        """
        pass
    
    @abstractmethod
    async def configure_scan_parameters(self, scan_id: str, parameters: Dict[str, Any]) -> bool:
        """
        Configure scan parameters.
        
        Args:
            scan_id: The scan ID to configure
            parameters: Scan parameters to set
            
        Returns:
            True if parameters configured successfully
            
        Raises:
            CTScannerException: If parameter configuration fails
        """
        pass
    
    @abstractmethod
    async def get_scan_parameters(self, scan_id: str) -> Dict[str, Any]:
        """
        Get scan parameters.
        
        Args:
            scan_id: The scan ID to get parameters for
            
        Returns:
            Dictionary containing scan parameters
            
        Raises:
            CTScannerException: If parameter retrieval fails
        """
        pass
    
    @abstractmethod
    async def calibrate_scanner(self, calibration_config: Dict[str, Any]) -> bool:
        """
        Calibrate the CT scanner.
        
        Args:
            calibration_config: Configuration for scanner calibration
            
        Returns:
            True if scanner calibrated successfully
            
        Raises:
            CTScannerException: If scanner calibration fails
        """
        pass
    
    @abstractmethod
    async def get_scanner_status(self) -> Dict[str, Any]:
        """
        Get status of the CT scanner.
        
        Returns:
            Dictionary containing scanner status
            
        Raises:
            CTScannerException: If status retrieval fails
        """
        pass
    
    @abstractmethod
    async def get_scanner_capabilities(self) -> Dict[str, Any]:
        """
        Get capabilities of the CT scanner.
        
        Returns:
            Dictionary containing scanner capabilities
            
        Raises:
            CTScannerException: If capability retrieval fails
        """
        pass
    
    @abstractmethod
    async def start_image_analysis(self, scan_id: str, analysis_config: Dict[str, Any]) -> str:
        """
        Start image analysis for a scan.
        
        Args:
            scan_id: The scan ID to analyze
            analysis_config: Configuration for image analysis
            
        Returns:
            Analysis ID
            
        Raises:
            CTScannerException: If analysis start fails
        """
        pass
    
    @abstractmethod
    async def get_analysis_status(self, analysis_id: str) -> Dict[str, Any]:
        """
        Get status of image analysis.
        
        Args:
            analysis_id: The analysis ID to check
            
        Returns:
            Dictionary containing analysis status
            
        Raises:
            CTScannerException: If status retrieval fails
        """
        pass
    
    @abstractmethod
    async def get_analysis_results(self, analysis_id: str) -> Dict[str, Any]:
        """
        Get results of image analysis.
        
        Args:
            analysis_id: The analysis ID to get results for
            
        Returns:
            Dictionary containing analysis results
            
        Raises:
            CTScannerException: If results retrieval fails
        """
        pass
    
    @abstractmethod
    async def detect_defects(self, scan_id: str, defect_config: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Detect defects in scan images.
        
        Args:
            scan_id: The scan ID to analyze
            defect_config: Optional defect detection configuration
            
        Returns:
            List of detected defects
            
        Raises:
            CTScannerException: If defect detection fails
        """
        pass
    
    @abstractmethod
    async def analyze_quality(self, scan_id: str, quality_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyze quality of scan images.
        
        Args:
            scan_id: The scan ID to analyze
            quality_config: Optional quality analysis configuration
            
        Returns:
            Dictionary containing quality analysis results
            
        Raises:
            CTScannerException: If quality analysis fails
        """
        pass
    
    @abstractmethod
    async def generate_report(self, scan_id: str, report_config: Dict[str, Any]) -> str:
        """
        Generate a report for a scan.
        
        Args:
            scan_id: The scan ID to generate report for
            report_config: Configuration for report generation
            
        Returns:
            Report file path or URL
            
        Raises:
            CTScannerException: If report generation fails
        """
        pass
    
    @abstractmethod
    async def export_scan_data(self, scan_id: str, export_config: Dict[str, Any]) -> str:
        """
        Export scan data.
        
        Args:
            scan_id: The scan ID to export
            export_config: Configuration for data export
            
        Returns:
            Export file path or URL
            
        Raises:
            CTScannerException: If data export fails
        """
        pass
    
    @abstractmethod
    async def get_scan_images(self, scan_id: str, image_config: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Get scan images.
        
        Args:
            scan_id: The scan ID to get images for
            image_config: Optional image configuration
            
        Returns:
            List of scan images
            
        Raises:
            CTScannerException: If image retrieval fails
        """
        pass
    
    @abstractmethod
    async def get_scan_metadata(self, scan_id: str) -> Dict[str, Any]:
        """
        Get metadata for a scan.
        
        Args:
            scan_id: The scan ID to get metadata for
            
        Returns:
            Dictionary containing scan metadata
            
        Raises:
            CTScannerException: If metadata retrieval fails
        """
        pass
    
    @abstractmethod
    async def get_scan_statistics(self, scan_id: str) -> Dict[str, Any]:
        """
        Get statistics for a scan.
        
        Args:
            scan_id: The scan ID to get statistics for
            
        Returns:
            Dictionary containing scan statistics
            
        Raises:
            CTScannerException: If statistics retrieval fails
        """
        pass
    
    @abstractmethod
    async def get_scan_history(self, time_range: Optional[Dict[str, datetime]] = None) -> List[Dict[str, Any]]:
        """
        Get scan history.
        
        Args:
            time_range: Optional time range for history
            
        Returns:
            List of historical scan data
            
        Raises:
            CTScannerException: If history retrieval fails
        """
        pass
    
    @abstractmethod
    async def get_scan_trends(self, time_period: str = "month") -> Dict[str, Any]:
        """
        Get scan trends over time.
        
        Args:
            time_period: Time period for trend analysis
            
        Returns:
            Dictionary containing scan trends
            
        Raises:
            CTScannerException: If trend analysis fails
        """
        pass
    
    @abstractmethod
    async def get_scanner_utilization(self, time_range: Optional[Dict[str, datetime]] = None) -> Dict[str, Any]:
        """
        Get scanner utilization statistics.
        
        Args:
            time_range: Optional time range for utilization analysis
            
        Returns:
            Dictionary containing utilization statistics
            
        Raises:
            CTScannerException: If utilization analysis fails
        """
        pass
    
    @abstractmethod
    async def get_scanner_performance_metrics(self) -> Dict[str, Any]:
        """
        Get scanner performance metrics.
        
        Returns:
            Dictionary containing performance metrics
            
        Raises:
            CTScannerException: If metrics retrieval fails
        """
        pass
    
    @abstractmethod
    async def get_scanner_maintenance_schedule(self) -> Dict[str, Any]:
        """
        Get scanner maintenance schedule.
        
        Returns:
            Dictionary containing maintenance schedule
            
        Raises:
            CTScannerException: If schedule retrieval fails
        """
        pass
    
    @abstractmethod
    async def schedule_maintenance(self, maintenance_config: Dict[str, Any]) -> str:
        """
        Schedule scanner maintenance.
        
        Args:
            maintenance_config: Configuration for maintenance
            
        Returns:
            Maintenance ID
            
        Raises:
            CTScannerException: If maintenance scheduling fails
        """
        pass
    
    @abstractmethod
    async def get_scanner_alerts(self, severity: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get scanner alerts.
        
        Args:
            severity: Optional severity level to filter alerts
            
        Returns:
            List of scanner alerts
            
        Raises:
            CTScannerException: If alert retrieval fails
        """
        pass
    
    @abstractmethod
    async def create_scanner_alert(self, alert_data: Dict[str, Any]) -> str:
        """
        Create a scanner alert.
        
        Args:
            alert_data: The alert data to create
            
        Returns:
            Alert ID
            
        Raises:
            CTScannerException: If alert creation fails
        """
        pass
    
    @abstractmethod
    async def resolve_scanner_alert(self, alert_id: str, resolution_data: Dict[str, Any]) -> bool:
        """
        Resolve a scanner alert.
        
        Args:
            alert_id: The ID of the alert to resolve
            resolution_data: The resolution data
            
        Returns:
            True if alert resolved successfully
            
        Raises:
            CTScannerException: If alert resolution fails
        """
        pass
    
    @abstractmethod
    async def get_service_health(self) -> Dict[str, Any]:
        """
        Get health status of the CT scanner service.
        
        Returns:
            Dictionary containing health status
        """
        pass
    
    @abstractmethod
    async def get_service_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics of the CT scanner service.
        
        Returns:
            Dictionary containing service metrics
        """
        pass
