"""
PBF process domain entity.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass, field

from .base_entity import BaseEntity
from ..enums import ProcessStatus, QualityTier
from ..value_objects import ProcessParameters, QualityMetrics


@dataclass
class PBFProcess(BaseEntity):
    """
    Domain entity representing a PBF (Powder Bed Fusion) process.
    
    This entity encapsulates all information about a PBF manufacturing process,
    including process parameters, status, quality metrics, and related data.
    """
    
    # Process identification (required fields first)
    process_name: str = field(default="")
    
    # Process identification (optional fields)
    build_id: Optional[str] = None
    part_id: Optional[str] = None
    
    # Process status and state
    status: ProcessStatus = ProcessStatus.INITIALIZED
    quality_tier: Optional[QualityTier] = None
    
    # Process parameters
    process_parameters: Optional[ProcessParameters] = None
    
    # Quality information
    quality_metrics: Optional[QualityMetrics] = None
    
    # Process timing
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    estimated_duration: Optional[float] = None  # seconds
    
    # Process results
    success: Optional[bool] = None
    failure_reason: Optional[str] = None
    error_code: Optional[str] = None
    
    # Material information
    material_type: Optional[str] = None
    material_batch_id: Optional[str] = None
    
    # Equipment information
    equipment_id: Optional[str] = None
    operator_id: Optional[str] = None
    
    # Process data
    layer_count: Optional[int] = None
    total_energy_used: Optional[float] = None  # Joules
    powder_consumed: Optional[float] = None  # grams
    
    def validate(self) -> None:
        """Validate PBF process entity."""
        if not self.process_name:
            raise ValueError("Process name cannot be empty")
        
        if self.start_time and self.end_time and self.start_time > self.end_time:
            raise ValueError("Start time cannot be after end time")
        
        if self.estimated_duration is not None and self.estimated_duration < 0:
            raise ValueError("Estimated duration cannot be negative")
        
        if self.total_energy_used is not None and self.total_energy_used < 0:
            raise ValueError("Total energy used cannot be negative")
        
        if self.powder_consumed is not None and self.powder_consumed < 0:
            raise ValueError("Powder consumed cannot be negative")
        
        if self.layer_count is not None and self.layer_count < 0:
            raise ValueError("Layer count cannot be negative")
    
    def start_process(self, start_time: Optional[datetime] = None) -> 'PBFProcess':
        """Start the PBF process."""
        if start_time is None:
            start_time = datetime.utcnow()
        
        return self.update(
            status=ProcessStatus.RUNNING,
            start_time=start_time,
            updated_by="PBFProcessService"
        )
    
    def pause_process(self, reason: str) -> 'PBFProcess':
        """Pause the PBF process."""
        return self.update(
            status=ProcessStatus.PAUSED,
            data={**self.data, "pause_reason": reason},
            updated_by="PBFProcessService"
        )
    
    def resume_process(self, reason: str) -> 'PBFProcess':
        """Resume the PBF process."""
        return self.update(
            status=ProcessStatus.RUNNING,
            data={**self.data, "resume_reason": reason},
            updated_by="PBFProcessService"
        )
    
    def complete_process(self, end_time: Optional[datetime] = None, success: bool = True) -> 'PBFProcess':
        """Complete the PBF process."""
        if end_time is None:
            end_time = datetime.utcnow()
        
        new_status = ProcessStatus.COMPLETED if success else ProcessStatus.FAILED
        
        return self.update(
            status=new_status,
            end_time=end_time,
            success=success,
            updated_by="PBFProcessService"
        )
    
    def fail_process(self, failure_reason: str, error_code: str, end_time: Optional[datetime] = None) -> 'PBFProcess':
        """Mark the PBF process as failed."""
        if end_time is None:
            end_time = datetime.utcnow()
        
        return self.update(
            status=ProcessStatus.FAILED,
            end_time=end_time,
            success=False,
            failure_reason=failure_reason,
            error_code=error_code,
            updated_by="PBFProcessService"
        )
    
    def cancel_process(self, cancellation_reason: str) -> 'PBFProcess':
        """Cancel the PBF process."""
        return self.update(
            status=ProcessStatus.CANCELLED,
            data={**self.data, "cancellation_reason": cancellation_reason},
            updated_by="PBFProcessService"
        )
    
    def update_quality_metrics(self, quality_metrics: QualityMetrics) -> 'PBFProcess':
        """Update quality metrics for the process."""
        return self.update(
            quality_metrics=quality_metrics,
            quality_tier=quality_metrics.get_quality_tier(),
            updated_by="QualityService"
        )
    
    def update_process_parameters(self, process_parameters: ProcessParameters) -> 'PBFProcess':
        """Update process parameters."""
        return self.update(
            process_parameters=process_parameters,
            updated_by="PBFProcessService"
        )
    
    def get_duration(self) -> Optional[float]:
        """Get process duration in seconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None
    
    def get_remaining_time(self) -> Optional[float]:
        """Get estimated remaining time in seconds."""
        if self.estimated_duration and self.start_time and not self.end_time:
            elapsed = (datetime.utcnow() - self.start_time).total_seconds()
            return max(0, self.estimated_duration - elapsed)
        return None
    
    def get_progress_percentage(self) -> Optional[float]:
        """Get process progress percentage."""
        if self.estimated_duration and self.start_time:
            if self.end_time:
                return 100.0
            else:
                elapsed = (datetime.utcnow() - self.start_time).total_seconds()
                return min(100.0, (elapsed / self.estimated_duration) * 100)
        return None
    
    def is_running(self) -> bool:
        """Check if process is currently running."""
        return self.status == ProcessStatus.RUNNING
    
    def is_completed(self) -> bool:
        """Check if process is completed."""
        return self.status in [ProcessStatus.COMPLETED, ProcessStatus.SUCCESS]
    
    def is_failed(self) -> bool:
        """Check if process has failed."""
        return self.status in [ProcessStatus.FAILED, ProcessStatus.CANCELLED, ProcessStatus.ABORTED]
    
    def is_paused(self) -> bool:
        """Check if process is paused."""
        return self.status == ProcessStatus.PAUSED
    
    def is_quality_checked(self) -> bool:
        """Check if process has been quality checked."""
        return self.quality_metrics is not None
    
    def get_energy_efficiency(self) -> Optional[float]:
        """Get energy efficiency (Joules per gram of powder)."""
        if self.total_energy_used and self.powder_consumed and self.powder_consumed > 0:
            return self.total_energy_used / self.powder_consumed
        return None
    
    def get_powder_efficiency(self) -> Optional[float]:
        """Get powder efficiency (percentage of powder used vs estimated)."""
        if self.process_parameters and self.powder_consumed:
            estimated_consumption = self.process_parameters.get_powder_consumption()
            if estimated_consumption > 0:
                return (self.powder_consumed / estimated_consumption) * 100
        return None
    
    def get_process_summary(self) -> Dict[str, Any]:
        """Get comprehensive process summary."""
        return {
            "process_id": self.id,
            "process_name": self.process_name,
            "status": self.status.value,
            "quality_tier": self.quality_tier.value if self.quality_tier else None,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration": self.get_duration(),
            "estimated_duration": self.estimated_duration,
            "remaining_time": self.get_remaining_time(),
            "progress_percentage": self.get_progress_percentage(),
            "success": self.success,
            "failure_reason": self.failure_reason,
            "error_code": self.error_code,
            "material_type": self.material_type,
            "equipment_id": self.equipment_id,
            "operator_id": self.operator_id,
            "layer_count": self.layer_count,
            "total_energy_used": self.total_energy_used,
            "powder_consumed": self.powder_consumed,
            "energy_efficiency": self.get_energy_efficiency(),
            "powder_efficiency": self.get_powder_efficiency(),
            "is_running": self.is_running(),
            "is_completed": self.is_completed(),
            "is_failed": self.is_failed(),
            "is_paused": self.is_paused(),
            "is_quality_checked": self.is_quality_checked(),
            "quality_score": self.quality_metrics.get_overall_quality_score() if self.quality_metrics else None
        }
    
    def get_process_analytics(self) -> Dict[str, Any]:
        """Get process analytics and insights."""
        analytics = {
            "performance_metrics": {
                "duration": self.get_duration(),
                "energy_efficiency": self.get_energy_efficiency(),
                "powder_efficiency": self.get_powder_efficiency(),
                "progress": self.get_progress_percentage()
            },
            "quality_metrics": {
                "overall_score": self.quality_metrics.get_overall_quality_score() if self.quality_metrics else None,
                "quality_tier": self.quality_tier.value if self.quality_tier else None,
                "is_acceptable": self.quality_metrics.is_acceptable_quality() if self.quality_metrics else None,
                "is_production_ready": self.quality_metrics.is_production_ready() if self.quality_metrics else None
            },
            "process_parameters": {
                "energy_density": self.process_parameters.get_energy_density() if self.process_parameters else None,
                "scan_time_per_layer": self.process_parameters.get_scan_time_per_layer() if self.process_parameters else None,
                "total_build_time": self.process_parameters.get_total_build_time() if self.process_parameters else None
            },
            "recommendations": []
        }
        
        # Add recommendations based on process state
        if self.is_failed():
            analytics["recommendations"].append("Investigate failure reason and improve process parameters")
        
        if self.quality_metrics and not self.quality_metrics.is_acceptable_quality():
            analytics["recommendations"].extend(self.quality_metrics.get_improvement_recommendations())
        
        if self.get_energy_efficiency() and self.get_energy_efficiency() > 1000:  # High energy usage
            analytics["recommendations"].append("Optimize energy usage through parameter tuning")
        
        if self.get_powder_efficiency() and self.get_powder_efficiency() > 110:  # High powder usage
            analytics["recommendations"].append("Optimize powder usage and reduce waste")
        
        return analytics