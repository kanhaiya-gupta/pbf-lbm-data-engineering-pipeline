"""
Powder bed repository interface.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from datetime import datetime

from ...domain.entities.powder_bed import PowderBed
from .base_repository import BaseRepository


class PowderBedRepository(BaseRepository[PowderBed]):
    """
    Repository interface for powder bed entities.
    
    This interface defines the contract for powder bed data access operations
    with domain-specific methods for bed management and monitoring.
    """
    
    @abstractmethod
    async def get_by_bed_name(self, bed_name: str) -> Optional[PowderBed]:
        """
        Get a powder bed by its name.
        
        Args:
            bed_name: The name of the bed to retrieve
            
        Returns:
            The bed if found, None otherwise
        """
        pass
    
    @abstractmethod
    async def get_by_bed_session_id(self, session_id: str) -> Optional[PowderBed]:
        """
        Get a powder bed by its session ID.
        
        Args:
            session_id: The session ID to retrieve
            
        Returns:
            The bed if found, None otherwise
        """
        pass
    
    @abstractmethod
    async def get_by_process_id(self, process_id: str) -> List[PowderBed]:
        """
        Get all powder beds for a specific process.
        
        Args:
            process_id: The process ID to filter by
            
        Returns:
            List of beds for the process
        """
        pass
    
    @abstractmethod
    async def get_by_build_id(self, build_id: str) -> List[PowderBed]:
        """
        Get all powder beds for a specific build.
        
        Args:
            build_id: The build ID to filter by
            
        Returns:
            List of beds for the build
        """
        pass
    
    @abstractmethod
    async def get_by_bed_status(self, status: str) -> List[PowderBed]:
        """
        Get all powder beds with a specific status.
        
        Args:
            status: The bed status to filter by
            
        Returns:
            List of beds with the specified status
        """
        pass
    
    @abstractmethod
    async def get_by_bed_quality(self, quality: str) -> List[PowderBed]:
        """
        Get all powder beds with a specific quality.
        
        Args:
            quality: The bed quality to filter by
            
        Returns:
            List of beds with the specified quality
        """
        pass
    
    @abstractmethod
    async def get_by_powder_type(self, powder_type: str) -> List[PowderBed]:
        """
        Get all powder beds for a specific powder type.
        
        Args:
            powder_type: The powder type to filter by
            
        Returns:
            List of beds for the powder type
        """
        pass
    
    @abstractmethod
    async def get_by_material(self, material: str) -> List[PowderBed]:
        """
        Get all powder beds for a specific material.
        
        Args:
            material: The material to filter by
            
        Returns:
            List of beds for the material
        """
        pass
    
    @abstractmethod
    async def get_prepared_beds(self) -> List[PowderBed]:
        """
        Get all prepared powder beds.
        
        Returns:
            List of prepared beds
        """
        pass
    
    @abstractmethod
    async def get_active_beds(self) -> List[PowderBed]:
        """
        Get all active powder beds.
        
        Returns:
            List of active beds
        """
        pass
    
    @abstractmethod
    async def get_disturbed_beds(self) -> List[PowderBed]:
        """
        Get all disturbed powder beds.
        
        Returns:
            List of disturbed beds
        """
        pass
    
    @abstractmethod
    async def get_cleaned_beds(self) -> List[PowderBed]:
        """
        Get all cleaned powder beds.
        
        Returns:
            List of cleaned beds
        """
        pass
    
    @abstractmethod
    async def get_beds_by_date_range(self, start_date: datetime, end_date: datetime) -> List[PowderBed]:
        """
        Get powder beds within a date range.
        
        Args:
            start_date: Start date for the range
            end_date: End date for the range
            
        Returns:
            List of beds within the date range
        """
        pass
    
    @abstractmethod
    async def get_beds_by_temperature_range(self, min_temp: float, max_temp: float) -> List[PowderBed]:
        """
        Get powder beds within a temperature range.
        
        Args:
            min_temp: Minimum temperature in Celsius
            max_temp: Maximum temperature in Celsius
            
        Returns:
            List of beds within the temperature range
        """
        pass
    
    @abstractmethod
    async def get_beds_by_powder_quantity_range(self, min_quantity: float, max_quantity: float) -> List[PowderBed]:
        """
        Get powder beds within a powder quantity range.
        
        Args:
            min_quantity: Minimum powder quantity in grams
            max_quantity: Maximum powder quantity in grams
            
        Returns:
            List of beds within the powder quantity range
        """
        pass
    
    @abstractmethod
    async def get_beds_by_powder_remaining_range(self, min_remaining: float, max_remaining: float) -> List[PowderBed]:
        """
        Get powder beds within a powder remaining range.
        
        Args:
            min_remaining: Minimum powder remaining in grams
            max_remaining: Maximum powder remaining in grams
            
        Returns:
            List of beds within the powder remaining range
        """
        pass
    
    @abstractmethod
    async def get_beds_by_quality_score_range(self, min_score: float, max_score: float) -> List[PowderBed]:
        """
        Get powder beds within a quality score range.
        
        Args:
            min_score: Minimum quality score (0-100)
            max_score: Maximum quality score (0-100)
            
        Returns:
            List of beds within the quality score range
        """
        pass
    
    @abstractmethod
    async def get_beds_by_disturbance_count_range(self, min_disturbances: int, max_disturbances: int) -> List[PowderBed]:
        """
        Get powder beds within a disturbance count range.
        
        Args:
            min_disturbances: Minimum number of disturbances
            max_disturbances: Maximum number of disturbances
            
        Returns:
            List of beds within the disturbance count range
        """
        pass
    
    @abstractmethod
    async def get_beds_by_cleaning_cycles_range(self, min_cycles: int, max_cycles: int) -> List[PowderBed]:
        """
        Get powder beds within a cleaning cycles range.
        
        Args:
            min_cycles: Minimum number of cleaning cycles
            max_cycles: Maximum number of cleaning cycles
            
        Returns:
            List of beds within the cleaning cycles range
        """
        pass
    
    @abstractmethod
    async def get_low_powder_beds(self, threshold_percentage: float = 20) -> List[PowderBed]:
        """
        Get powder beds with low powder levels.
        
        Args:
            threshold_percentage: Threshold percentage for low powder
            
        Returns:
            List of beds with low powder levels
        """
        pass
    
    @abstractmethod
    async def get_depleted_powder_beds(self, threshold_percentage: float = 5) -> List[PowderBed]:
        """
        Get powder beds with depleted powder levels.
        
        Args:
            threshold_percentage: Threshold percentage for depleted powder
            
        Returns:
            List of beds with depleted powder levels
        """
        pass
    
    @abstractmethod
    async def get_high_disturbance_beds(self, threshold: int = 5) -> List[PowderBed]:
        """
        Get powder beds with high disturbance counts.
        
        Args:
            threshold: Minimum disturbance count threshold
            
        Returns:
            List of beds with high disturbance counts
        """
        pass
    
    @abstractmethod
    async def get_beds_needing_cleaning(self, max_cycles: int = 10) -> List[PowderBed]:
        """
        Get powder beds that need cleaning.
        
        Args:
            max_cycles: Maximum cleaning cycles before needing cleaning
            
        Returns:
            List of beds that need cleaning
        """
        pass
    
    @abstractmethod
    async def get_ready_beds(self) -> List[PowderBed]:
        """
        Get powder beds that are ready for use.
        
        Returns:
            List of ready beds
        """
        pass
    
    @abstractmethod
    async def get_bed_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about powder beds.
        
        Returns:
            Dictionary containing bed statistics
        """
        pass
    
    @abstractmethod
    async def get_bed_analytics(self, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Get analytics and insights about powder beds.
        
        Args:
            filters: Optional filters to apply to the analytics
            
        Returns:
            Dictionary containing bed analytics
        """
        pass
    
    @abstractmethod
    async def get_bed_trends(self, time_period: str = "month") -> Dict[str, Any]:
        """
        Get trends and patterns in powder beds over time.
        
        Args:
            time_period: Time period for trend analysis (day, week, month, year)
            
        Returns:
            Dictionary containing bed trends
        """
        pass
    
    @abstractmethod
    async def get_powder_usage_analysis(self, bed_id: str) -> Dict[str, Any]:
        """
        Get powder usage analysis for a specific bed.
        
        Args:
            bed_id: The ID of the bed to analyze
            
        Returns:
            Dictionary containing powder usage analysis
        """
        pass
    
    @abstractmethod
    async def get_bed_quality_analysis(self, bed_id: str) -> Dict[str, Any]:
        """
        Get bed quality analysis for a specific bed.
        
        Args:
            bed_id: The ID of the bed to analyze
            
        Returns:
            Dictionary containing bed quality analysis
        """
        pass
    
    @abstractmethod
    async def get_disturbance_analysis(self, bed_id: str) -> Dict[str, Any]:
        """
        Get disturbance analysis for a specific bed.
        
        Args:
            bed_id: The ID of the bed to analyze
            
        Returns:
            Dictionary containing disturbance analysis
        """
        pass
    
    @abstractmethod
    async def get_bed_recommendations(self, bed_id: str) -> List[str]:
        """
        Get recommendations for improving a powder bed.
        
        Args:
            bed_id: The ID of the bed to get recommendations for
            
        Returns:
            List of improvement recommendations
        """
        pass
    
    @abstractmethod
    async def get_bed_comparison(self, bed_ids: List[str]) -> Dict[str, Any]:
        """
        Compare multiple powder beds.
        
        Args:
            bed_ids: List of bed IDs to compare
            
        Returns:
            Dictionary containing bed comparison data
        """
        pass
    
    @abstractmethod
    async def get_bed_impact_analysis(self, bed_id: str) -> Dict[str, Any]:
        """
        Get impact analysis for a powder bed.
        
        Args:
            bed_id: The ID of the bed to analyze
            
        Returns:
            Dictionary containing impact analysis results
        """
        pass
    
    @abstractmethod
    async def get_powder_consumption_trends(self, time_period: str = "month") -> Dict[str, Any]:
        """
        Get trends in powder consumption over time.
        
        Args:
            time_period: Time period for trend analysis (day, week, month, year)
            
        Returns:
            Dictionary containing powder consumption trends
        """
        pass
    
    @abstractmethod
    async def get_bed_quality_trends(self, time_period: str = "month") -> Dict[str, Any]:
        """
        Get trends in bed quality over time.
        
        Args:
            time_period: Time period for trend analysis (day, week, month, year)
            
        Returns:
            Dictionary containing bed quality trends
        """
        pass
    
    @abstractmethod
    async def get_disturbance_trends(self, time_period: str = "month") -> Dict[str, Any]:
        """
        Get trends in bed disturbances over time.
        
        Args:
            time_period: Time period for trend analysis (day, week, month, year)
            
        Returns:
            Dictionary containing disturbance trends
        """
        pass
    
    @abstractmethod
    async def get_maintenance_analysis(self, bed_id: str) -> Dict[str, Any]:
        """
        Get maintenance analysis for a specific bed.
        
        Args:
            bed_id: The ID of the bed to analyze
            
        Returns:
            Dictionary containing maintenance analysis
        """
        pass
    
    @abstractmethod
    async def get_bed_utilization_analysis(self, bed_id: str) -> Dict[str, Any]:
        """
        Get utilization analysis for a specific bed.
        
        Args:
            bed_id: The ID of the bed to analyze
            
        Returns:
            Dictionary containing utilization analysis
        """
        pass
