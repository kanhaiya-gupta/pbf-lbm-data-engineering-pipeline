"""
PBF process repository interface.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from datetime import datetime

from ...domain.entities.pbf_process import PBFProcess
from ...domain.enums import ProcessStatus, QualityTier
from ...domain.value_objects import ProcessParameters, QualityMetrics
from .base_repository import BaseRepository


class PBFProcessRepository(BaseRepository[PBFProcess]):
    """
    Repository interface for PBF process entities.
    
    This interface defines the contract for PBF process data access operations
    with domain-specific methods for process management.
    """
    
    @abstractmethod
    async def get_by_process_name(self, process_name: str) -> Optional[PBFProcess]:
        """
        Get a PBF process by its name.
        
        Args:
            process_name: The name of the process to retrieve
            
        Returns:
            The process if found, None otherwise
        """
        pass
    
    @abstractmethod
    async def get_by_build_id(self, build_id: str) -> List[PBFProcess]:
        """
        Get all PBF processes for a specific build.
        
        Args:
            build_id: The build ID to filter by
            
        Returns:
            List of processes for the build
        """
        pass
    
    @abstractmethod
    async def get_by_part_id(self, part_id: str) -> List[PBFProcess]:
        """
        Get all PBF processes for a specific part.
        
        Args:
            part_id: The part ID to filter by
            
        Returns:
            List of processes for the part
        """
        pass
    
    @abstractmethod
    async def get_by_status(self, status: ProcessStatus) -> List[PBFProcess]:
        """
        Get all PBF processes with a specific status.
        
        Args:
            status: The process status to filter by
            
        Returns:
            List of processes with the specified status
        """
        pass
    
    @abstractmethod
    async def get_by_quality_tier(self, quality_tier: QualityTier) -> List[PBFProcess]:
        """
        Get all PBF processes with a specific quality tier.
        
        Args:
            quality_tier: The quality tier to filter by
            
        Returns:
            List of processes with the specified quality tier
        """
        pass
    
    @abstractmethod
    async def get_by_material_type(self, material_type: str) -> List[PBFProcess]:
        """
        Get all PBF processes for a specific material type.
        
        Args:
            material_type: The material type to filter by
            
        Returns:
            List of processes for the material type
        """
        pass
    
    @abstractmethod
    async def get_by_equipment_id(self, equipment_id: str) -> List[PBFProcess]:
        """
        Get all PBF processes for a specific equipment.
        
        Args:
            equipment_id: The equipment ID to filter by
            
        Returns:
            List of processes for the equipment
        """
        pass
    
    @abstractmethod
    async def get_by_operator_id(self, operator_id: str) -> List[PBFProcess]:
        """
        Get all PBF processes for a specific operator.
        
        Args:
            operator_id: The operator ID to filter by
            
        Returns:
            List of processes for the operator
        """
        pass
    
    @abstractmethod
    async def get_running_processes(self) -> List[PBFProcess]:
        """
        Get all currently running PBF processes.
        
        Returns:
            List of running processes
        """
        pass
    
    @abstractmethod
    async def get_completed_processes(self) -> List[PBFProcess]:
        """
        Get all completed PBF processes.
        
        Returns:
            List of completed processes
        """
        pass
    
    @abstractmethod
    async def get_failed_processes(self) -> List[PBFProcess]:
        """
        Get all failed PBF processes.
        
        Returns:
            List of failed processes
        """
        pass
    
    @abstractmethod
    async def get_processes_by_date_range(self, start_date: datetime, end_date: datetime) -> List[PBFProcess]:
        """
        Get PBF processes within a date range.
        
        Args:
            start_date: Start date for the range
            end_date: End date for the range
            
        Returns:
            List of processes within the date range
        """
        pass
    
    @abstractmethod
    async def get_processes_by_duration_range(self, min_duration: float, max_duration: float) -> List[PBFProcess]:
        """
        Get PBF processes within a duration range.
        
        Args:
            min_duration: Minimum duration in seconds
            max_duration: Maximum duration in seconds
            
        Returns:
            List of processes within the duration range
        """
        pass
    
    @abstractmethod
    async def get_processes_by_energy_range(self, min_energy: float, max_energy: float) -> List[PBFProcess]:
        """
        Get PBF processes within an energy usage range.
        
        Args:
            min_energy: Minimum energy usage in Joules
            max_energy: Maximum energy usage in Joules
            
        Returns:
            List of processes within the energy range
        """
        pass
    
    @abstractmethod
    async def get_processes_by_powder_consumption_range(self, min_consumption: float, max_consumption: float) -> List[PBFProcess]:
        """
        Get PBF processes within a powder consumption range.
        
        Args:
            min_consumption: Minimum powder consumption in grams
            max_consumption: Maximum powder consumption in grams
            
        Returns:
            List of processes within the consumption range
        """
        pass
    
    @abstractmethod
    async def get_processes_by_quality_score_range(self, min_score: float, max_score: float) -> List[PBFProcess]:
        """
        Get PBF processes within a quality score range.
        
        Args:
            min_score: Minimum quality score (0-100)
            max_score: Maximum quality score (0-100)
            
        Returns:
            List of processes within the quality score range
        """
        pass
    
    @abstractmethod
    async def get_top_performing_processes(self, limit: int = 10) -> List[PBFProcess]:
        """
        Get top performing PBF processes by quality score.
        
        Args:
            limit: Maximum number of processes to return
            
        Returns:
            List of top performing processes
        """
        pass
    
    @abstractmethod
    async def get_worst_performing_processes(self, limit: int = 10) -> List[PBFProcess]:
        """
        Get worst performing PBF processes by quality score.
        
        Args:
            limit: Maximum number of processes to return
            
        Returns:
            List of worst performing processes
        """
        pass
    
    @abstractmethod
    async def get_most_energy_efficient_processes(self, limit: int = 10) -> List[PBFProcess]:
        """
        Get most energy efficient PBF processes.
        
        Args:
            limit: Maximum number of processes to return
            
        Returns:
            List of most energy efficient processes
        """
        pass
    
    @abstractmethod
    async def get_least_energy_efficient_processes(self, limit: int = 10) -> List[PBFProcess]:
        """
        Get least energy efficient PBF processes.
        
        Args:
            limit: Maximum number of processes to return
            
        Returns:
            List of least energy efficient processes
        """
        pass
    
    @abstractmethod
    async def get_process_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about PBF processes.
        
        Returns:
            Dictionary containing process statistics
        """
        pass
    
    @abstractmethod
    async def get_process_analytics(self, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Get analytics and insights about PBF processes.
        
        Args:
            filters: Optional filters to apply to the analytics
            
        Returns:
            Dictionary containing process analytics
        """
        pass
    
    @abstractmethod
    async def get_process_trends(self, time_period: str = "month") -> Dict[str, Any]:
        """
        Get trends and patterns in PBF processes over time.
        
        Args:
            time_period: Time period for trend analysis (day, week, month, year)
            
        Returns:
            Dictionary containing process trends
        """
        pass
    
    @abstractmethod
    async def get_process_comparison(self, process_ids: List[str]) -> Dict[str, Any]:
        """
        Compare multiple PBF processes.
        
        Args:
            process_ids: List of process IDs to compare
            
        Returns:
            Dictionary containing process comparison data
        """
        pass
    
    @abstractmethod
    async def get_process_recommendations(self, process_id: str) -> List[str]:
        """
        Get recommendations for improving a PBF process.
        
        Args:
            process_id: The ID of the process to get recommendations for
            
        Returns:
            List of improvement recommendations
        """
        pass
    
    @abstractmethod
    async def get_process_parameters_history(self, process_id: str) -> List[ProcessParameters]:
        """
        Get the history of process parameters for a PBF process.
        
        Args:
            process_id: The ID of the process to get parameter history for
            
        Returns:
            List of process parameters in chronological order
        """
        pass
    
    @abstractmethod
    async def get_quality_metrics_history(self, process_id: str) -> List[QualityMetrics]:
        """
        Get the history of quality metrics for a PBF process.
        
        Args:
            process_id: The ID of the process to get quality history for
            
        Returns:
            List of quality metrics in chronological order
        """
        pass
    
    @abstractmethod
    async def get_process_dependencies(self, process_id: str) -> List[PBFProcess]:
        """
        Get processes that depend on or are related to a specific process.
        
        Args:
            process_id: The ID of the process to get dependencies for
            
        Returns:
            List of related processes
        """
        pass
    
    @abstractmethod
    async def get_process_impact_analysis(self, process_id: str) -> Dict[str, Any]:
        """
        Get impact analysis for a PBF process.
        
        Args:
            process_id: The ID of the process to analyze
            
        Returns:
            Dictionary containing impact analysis results
        """
        pass
