"""
CT scan repository interface.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from datetime import datetime

from ...domain.entities.ct_scan import CTScan
from ...domain.enums import QualityTier, DefectType
from ...domain.value_objects import QualityMetrics, DefectClassification
from .base_repository import BaseRepository


class CTScanRepository(BaseRepository[CTScan]):
    """
    Repository interface for CT scan entities.
    
    This interface defines the contract for CT scan data access operations
    with domain-specific methods for scan management and analysis.
    """
    
    @abstractmethod
    async def get_by_scan_name(self, scan_name: str) -> Optional[CTScan]:
        """
        Get a CT scan by its name.
        
        Args:
            scan_name: The name of the scan to retrieve
            
        Returns:
            The scan if found, None otherwise
        """
        pass
    
    @abstractmethod
    async def get_by_scan_session_id(self, session_id: str) -> Optional[CTScan]:
        """
        Get a CT scan by its session ID.
        
        Args:
            session_id: The session ID to retrieve
            
        Returns:
            The scan if found, None otherwise
        """
        pass
    
    @abstractmethod
    async def get_by_process_id(self, process_id: str) -> List[CTScan]:
        """
        Get all CT scans for a specific process.
        
        Args:
            process_id: The process ID to filter by
            
        Returns:
            List of scans for the process
        """
        pass
    
    @abstractmethod
    async def get_by_build_id(self, build_id: str) -> List[CTScan]:
        """
        Get all CT scans for a specific build.
        
        Args:
            build_id: The build ID to filter by
            
        Returns:
            List of scans for the build
        """
        pass
    
    @abstractmethod
    async def get_by_part_id(self, part_id: str) -> List[CTScan]:
        """
        Get all CT scans for a specific part.
        
        Args:
            part_id: The part ID to filter by
            
        Returns:
            List of scans for the part
        """
        pass
    
    @abstractmethod
    async def get_by_scan_status(self, status: str) -> List[CTScan]:
        """
        Get all CT scans with a specific status.
        
        Args:
            status: The scan status to filter by
            
        Returns:
            List of scans with the specified status
        """
        pass
    
    @abstractmethod
    async def get_by_scan_quality(self, quality: str) -> List[CTScan]:
        """
        Get all CT scans with a specific quality.
        
        Args:
            quality: The scan quality to filter by
            
        Returns:
            List of scans with the specified quality
        """
        pass
    
    @abstractmethod
    async def get_by_scanner_id(self, scanner_id: str) -> List[CTScan]:
        """
        Get all CT scans for a specific scanner.
        
        Args:
            scanner_id: The scanner ID to filter by
            
        Returns:
            List of scans for the scanner
        """
        pass
    
    @abstractmethod
    async def get_by_operator_id(self, operator_id: str) -> List[CTScan]:
        """
        Get all CT scans for a specific operator.
        
        Args:
            operator_id: The operator ID to filter by
            
        Returns:
            List of scans for the operator
        """
        pass
    
    @abstractmethod
    async def get_completed_scans(self) -> List[CTScan]:
        """
        Get all completed CT scans.
        
        Returns:
            List of completed scans
        """
        pass
    
    @abstractmethod
    async def get_failed_scans(self) -> List[CTScan]:
        """
        Get all failed CT scans.
        
        Returns:
            List of failed scans
        """
        pass
    
    @abstractmethod
    async def get_scans_by_date_range(self, start_date: datetime, end_date: datetime) -> List[CTScan]:
        """
        Get CT scans within a date range.
        
        Args:
            start_date: Start date for the range
            end_date: End date for the range
            
        Returns:
            List of scans within the date range
        """
        pass
    
    @abstractmethod
    async def get_scans_by_duration_range(self, min_duration: float, max_duration: float) -> List[CTScan]:
        """
        Get CT scans within a duration range.
        
        Args:
            min_duration: Minimum duration in seconds
            max_duration: Maximum duration in seconds
            
        Returns:
            List of scans within the duration range
        """
        pass
    
    @abstractmethod
    async def get_scans_by_image_count_range(self, min_images: int, max_images: int) -> List[CTScan]:
        """
        Get CT scans within an image count range.
        
        Args:
            min_images: Minimum number of images
            max_images: Maximum number of images
            
        Returns:
            List of scans within the image count range
        """
        pass
    
    @abstractmethod
    async def get_scans_by_file_size_range(self, min_size: float, max_size: float) -> List[CTScan]:
        """
        Get CT scans within a file size range.
        
        Args:
            min_size: Minimum file size in MB
            max_size: Maximum file size in MB
            
        Returns:
            List of scans within the file size range
        """
        pass
    
    @abstractmethod
    async def get_scans_by_resolution_range(self, min_resolution: float, max_resolution: float) -> List[CTScan]:
        """
        Get CT scans within a resolution range.
        
        Args:
            min_resolution: Minimum resolution in mm/voxel
            max_resolution: Maximum resolution in mm/voxel
            
        Returns:
            List of scans within the resolution range
        """
        pass
    
    @abstractmethod
    async def get_scans_by_quality_score_range(self, min_score: float, max_score: float) -> List[CTScan]:
        """
        Get CT scans within a quality score range.
        
        Args:
            min_score: Minimum quality score (0-100)
            max_score: Maximum quality score (0-100)
            
        Returns:
            List of scans within the quality score range
        """
        pass
    
    @abstractmethod
    async def get_scans_with_critical_defects(self) -> List[CTScan]:
        """
        Get CT scans with critical defects.
        
        Returns:
            List of scans with critical defects
        """
        pass
    
    @abstractmethod
    async def get_scans_without_defects(self) -> List[CTScan]:
        """
        Get CT scans without any defects.
        
        Returns:
            List of scans without defects
        """
        pass
    
    @abstractmethod
    async def get_scans_by_defect_type(self, defect_type: DefectType) -> List[CTScan]:
        """
        Get CT scans with a specific defect type.
        
        Args:
            defect_type: The defect type to filter by
            
        Returns:
            List of scans with the specified defect type
        """
        pass
    
    @abstractmethod
    async def get_scans_by_quality_tier(self, quality_tier: QualityTier) -> List[CTScan]:
        """
        Get CT scans with a specific quality tier.
        
        Args:
            quality_tier: The quality tier to filter by
            
        Returns:
            List of scans with the specified quality tier
        """
        pass
    
    @abstractmethod
    async def get_top_quality_scans(self, limit: int = 10) -> List[CTScan]:
        """
        Get top quality CT scans by quality score.
        
        Args:
            limit: Maximum number of scans to return
            
        Returns:
            List of top quality scans
        """
        pass
    
    @abstractmethod
    async def get_lowest_quality_scans(self, limit: int = 10) -> List[CTScan]:
        """
        Get lowest quality CT scans by quality score.
        
        Args:
            limit: Maximum number of scans to return
            
        Returns:
            List of lowest quality scans
        """
        pass
    
    @abstractmethod
    async def get_scan_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about CT scans.
        
        Returns:
            Dictionary containing scan statistics
        """
        pass
    
    @abstractmethod
    async def get_scan_analytics(self, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Get analytics and insights about CT scans.
        
        Args:
            filters: Optional filters to apply to the analytics
            
        Returns:
            Dictionary containing scan analytics
        """
        pass
    
    @abstractmethod
    async def get_scan_trends(self, time_period: str = "month") -> Dict[str, Any]:
        """
        Get trends and patterns in CT scans over time.
        
        Args:
            time_period: Time period for trend analysis (day, week, month, year)
            
        Returns:
            Dictionary containing scan trends
        """
        pass
    
    @abstractmethod
    async def get_defect_analysis(self, scan_id: str) -> Dict[str, Any]:
        """
        Get defect analysis for a specific CT scan.
        
        Args:
            scan_id: The ID of the scan to analyze
            
        Returns:
            Dictionary containing defect analysis results
        """
        pass
    
    @abstractmethod
    async def get_quality_analysis(self, scan_id: str) -> Dict[str, Any]:
        """
        Get quality analysis for a specific CT scan.
        
        Args:
            scan_id: The ID of the scan to analyze
            
        Returns:
            Dictionary containing quality analysis results
        """
        pass
    
    @abstractmethod
    async def get_scan_recommendations(self, scan_id: str) -> List[str]:
        """
        Get recommendations for improving a CT scan.
        
        Args:
            scan_id: The ID of the scan to get recommendations for
            
        Returns:
            List of improvement recommendations
        """
        pass
    
    @abstractmethod
    async def get_scan_comparison(self, scan_ids: List[str]) -> Dict[str, Any]:
        """
        Compare multiple CT scans.
        
        Args:
            scan_ids: List of scan IDs to compare
            
        Returns:
            Dictionary containing scan comparison data
        """
        pass
    
    @abstractmethod
    async def get_scan_impact_analysis(self, scan_id: str) -> Dict[str, Any]:
        """
        Get impact analysis for a CT scan.
        
        Args:
            scan_id: The ID of the scan to analyze
            
        Returns:
            Dictionary containing impact analysis results
        """
        pass
    
    @abstractmethod
    async def get_defect_trends(self, time_period: str = "month") -> Dict[str, Any]:
        """
        Get trends in defect detection over time.
        
        Args:
            time_period: Time period for trend analysis (day, week, month, year)
            
        Returns:
            Dictionary containing defect trends
        """
        pass
    
    @abstractmethod
    async def get_quality_trends(self, time_period: str = "month") -> Dict[str, Any]:
        """
        Get trends in scan quality over time.
        
        Args:
            time_period: Time period for trend analysis (day, week, month, year)
            
        Returns:
            Dictionary containing quality trends
        """
        pass
    
    @abstractmethod
    async def get_scanner_performance_analysis(self, scanner_id: str) -> Dict[str, Any]:
        """
        Get performance analysis for a specific scanner.
        
        Args:
            scanner_id: The ID of the scanner to analyze
            
        Returns:
            Dictionary containing scanner performance analysis
        """
        pass
    
    @abstractmethod
    async def get_operator_performance_analysis(self, operator_id: str) -> Dict[str, Any]:
        """
        Get performance analysis for a specific operator.
        
        Args:
            operator_id: The ID of the operator to analyze
            
        Returns:
            Dictionary containing operator performance analysis
        """
        pass
