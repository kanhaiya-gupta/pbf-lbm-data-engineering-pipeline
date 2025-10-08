"""
Powder bed domain entity.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass, field

from .base_entity import BaseEntity


@dataclass
class PowderBed(BaseEntity):
    """
    Domain entity representing a powder bed in PBF operations.
    
    This entity encapsulates all information about powder bed preparation,
    monitoring, quality assessment, and maintenance operations.
    """
    
    # Bed identification
    bed_name: str = field(default="")
    bed_session_id: Optional[str] = None
    
    # Process information
    process_id: Optional[str] = None
    build_id: Optional[str] = None
    
    # Bed configuration
    bed_dimensions: Optional[Dict[str, float]] = None  # x, y, z dimensions in mm
    bed_material: Optional[str] = None
    bed_temperature: Optional[float] = None  # Celsius
    
    # Bed state
    bed_status: str = "initialized"  # initialized, prepared, active, disturbed, cleaned
    bed_quality: Optional[str] = None  # excellent, good, fair, poor
    
    # Timing information
    preparation_time: Optional[datetime] = None
    last_quality_check: Optional[datetime] = None
    last_cleaning_time: Optional[datetime] = None
    
    # Powder information
    powder_type: Optional[str] = None
    powder_batch_id: Optional[str] = None
    powder_quantity: Optional[float] = None  # grams
    powder_remaining: Optional[float] = None  # grams
    powder_consumption_rate: Optional[float] = None  # grams/hour
    
    # Bed quality metrics
    bed_height: Optional[float] = None  # mm
    bed_density: Optional[float] = None  # g/cm³
    bed_roughness: Optional[float] = None  # μm
    bed_flatness: Optional[float] = None  # mm
    
    # Disturbance tracking
    disturbance_count: int = 0
    disturbance_types: Optional[List[str]] = None
    last_disturbance: Optional[datetime] = None
    
    # Quality assessments
    quality_checks: Optional[List[Dict[str, Any]]] = None
    quality_score: Optional[float] = None  # 0-100
    
    # Maintenance information
    cleaning_cycles: int = 0
    maintenance_schedule: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Initialize default values."""
        if self.disturbance_types is None:
            object.__setattr__(self, 'disturbance_types', [])
        if self.quality_checks is None:
            object.__setattr__(self, 'quality_checks', [])
        super().__post_init__()
    
    def validate(self) -> None:
        """Validate powder bed entity."""
        if not self.bed_name:
            raise ValueError("Bed name cannot be empty")
        
        if self.bed_temperature is not None and self.bed_temperature < 0:
            raise ValueError("Bed temperature cannot be negative")
        
        if self.powder_quantity is not None and self.powder_quantity < 0:
            raise ValueError("Powder quantity cannot be negative")
        
        if self.powder_remaining is not None and self.powder_remaining < 0:
            raise ValueError("Powder remaining cannot be negative")
        
        if self.powder_consumption_rate is not None and self.powder_consumption_rate < 0:
            raise ValueError("Powder consumption rate cannot be negative")
        
        if self.bed_height is not None and self.bed_height < 0:
            raise ValueError("Bed height cannot be negative")
        
        if self.bed_density is not None and self.bed_density < 0:
            raise ValueError("Bed density cannot be negative")
        
        if self.bed_roughness is not None and self.bed_roughness < 0:
            raise ValueError("Bed roughness cannot be negative")
        
        if self.bed_flatness is not None and self.bed_flatness < 0:
            raise ValueError("Bed flatness cannot be negative")
        
        if self.quality_score is not None and not 0 <= self.quality_score <= 100:
            raise ValueError("Quality score must be between 0 and 100")
        
        if self.disturbance_count < 0:
            raise ValueError("Disturbance count cannot be negative")
        
        if self.cleaning_cycles < 0:
            raise ValueError("Cleaning cycles cannot be negative")
    
    def prepare_bed(self, preparation_time: Optional[datetime] = None) -> 'PowderBed':
        """Prepare the powder bed."""
        if preparation_time is None:
            preparation_time = datetime.utcnow()
        
        return self.update(
            bed_status="prepared",
            preparation_time=preparation_time,
            updated_by="PowderBedService"
        )
    
    def activate_bed(self) -> 'PowderBed':
        """Activate the powder bed for use."""
        return self.update(
            bed_status="active",
            updated_by="PowderBedService"
        )
    
    def disturb_bed(self, disturbance_type: str, severity: str = "minor") -> 'PowderBed':
        """Record a bed disturbance."""
        new_disturbance_types = self.disturbance_types + [disturbance_type]
        
        return self.update(
            bed_status="disturbed",
            disturbance_count=self.disturbance_count + 1,
            disturbance_types=new_disturbance_types,
            last_disturbance=datetime.utcnow(),
            data={**self.data, "last_disturbance_severity": severity},
            updated_by="PowderBedService"
        )
    
    def clean_bed(self, cleaning_method: str, cleaning_time: Optional[datetime] = None) -> 'PowderBed':
        """Clean the powder bed."""
        if cleaning_time is None:
            cleaning_time = datetime.utcnow()
        
        return self.update(
            bed_status="cleaned",
            cleaning_cycles=self.cleaning_cycles + 1,
            last_cleaning_time=cleaning_time,
            data={**self.data, "last_cleaning_method": cleaning_method},
            updated_by="PowderBedService"
        )
    
    def update_powder_consumption(self, consumed_amount: float) -> 'PowderBed':
        """Update powder consumption."""
        new_remaining = self.powder_remaining - consumed_amount if self.powder_remaining else 0
        
        return self.update(
            powder_remaining=max(0, new_remaining),
            updated_by="PowderBedService"
        )
    
    def add_quality_check(self, quality_check: Dict[str, Any]) -> 'PowderBed':
        """Add a quality check to the bed."""
        new_quality_checks = self.quality_checks + [quality_check]
        
        return self.update(
            quality_checks=new_quality_checks,
            last_quality_check=datetime.utcnow(),
            updated_by="PowderBedService"
        )
    
    def update_quality_score(self, quality_score: float) -> 'PowderBed':
        """Update bed quality score."""
        return self.update(
            quality_score=quality_score,
            bed_quality=self._get_quality_tier(quality_score),
            updated_by="PowderBedService"
        )
    
    def _get_quality_tier(self, score: float) -> str:
        """Get quality tier based on score."""
        if score >= 90:
            return "excellent"
        elif score >= 80:
            return "good"
        elif score >= 70:
            return "fair"
        else:
            return "poor"
    
    def get_powder_usage_percentage(self) -> Optional[float]:
        """Get powder usage percentage."""
        if self.powder_quantity and self.powder_remaining is not None:
            used = self.powder_quantity - self.powder_remaining
            return (used / self.powder_quantity) * 100
        return None
    
    def get_remaining_capacity_percentage(self) -> Optional[float]:
        """Get remaining powder capacity percentage."""
        if self.powder_quantity and self.powder_remaining is not None:
            return (self.powder_remaining / self.powder_quantity) * 100
        return None
    
    def is_powder_low(self, threshold_percentage: float = 20) -> bool:
        """Check if powder is running low."""
        remaining_percentage = self.get_remaining_capacity_percentage()
        return remaining_percentage is not None and remaining_percentage < threshold_percentage
    
    def is_powder_depleted(self, threshold_percentage: float = 5) -> bool:
        """Check if powder is depleted."""
        remaining_percentage = self.get_remaining_capacity_percentage()
        return remaining_percentage is not None and remaining_percentage < threshold_percentage
    
    def is_bed_ready(self) -> bool:
        """Check if bed is ready for use."""
        return self.bed_status == "prepared" and self.quality_score and self.quality_score >= 70
    
    def is_bed_disturbed(self) -> bool:
        """Check if bed is disturbed."""
        return self.bed_status == "disturbed"
    
    def needs_cleaning(self, max_cycles: int = 10) -> bool:
        """Check if bed needs cleaning."""
        return self.cleaning_cycles >= max_cycles or self.disturbance_count > 5
    
    def get_bed_summary(self) -> Dict[str, Any]:
        """Get comprehensive bed summary."""
        return {
            "bed_id": self.id,
            "bed_name": self.bed_name,
            "bed_session_id": self.bed_session_id,
            "process_id": self.process_id,
            "build_id": self.build_id,
            "bed_status": self.bed_status,
            "bed_quality": self.bed_quality,
            "bed_dimensions": self.bed_dimensions,
            "bed_material": self.bed_material,
            "bed_temperature": self.bed_temperature,
            "preparation_time": self.preparation_time.isoformat() if self.preparation_time else None,
            "last_quality_check": self.last_quality_check.isoformat() if self.last_quality_check else None,
            "last_cleaning_time": self.last_cleaning_time.isoformat() if self.last_cleaning_time else None,
            "powder_type": self.powder_type,
            "powder_batch_id": self.powder_batch_id,
            "powder_quantity": self.powder_quantity,
            "powder_remaining": self.powder_remaining,
            "powder_usage_percentage": self.get_powder_usage_percentage(),
            "remaining_capacity_percentage": self.get_remaining_capacity_percentage(),
            "powder_consumption_rate": self.powder_consumption_rate,
            "bed_height": self.bed_height,
            "bed_density": self.bed_density,
            "bed_roughness": self.bed_roughness,
            "bed_flatness": self.bed_flatness,
            "disturbance_count": self.disturbance_count,
            "disturbance_types": self.disturbance_types,
            "last_disturbance": self.last_disturbance.isoformat() if self.last_disturbance else None,
            "quality_score": self.quality_score,
            "quality_checks_count": len(self.quality_checks),
            "cleaning_cycles": self.cleaning_cycles,
            "is_powder_low": self.is_powder_low(),
            "is_powder_depleted": self.is_powder_depleted(),
            "is_bed_ready": self.is_bed_ready(),
            "is_bed_disturbed": self.is_bed_disturbed(),
            "needs_cleaning": self.needs_cleaning()
        }
    
    def get_bed_analytics(self) -> Dict[str, Any]:
        """Get bed analytics and insights."""
        analytics = {
            "powder_management": {
                "usage_percentage": self.get_powder_usage_percentage(),
                "remaining_capacity": self.get_remaining_capacity_percentage(),
                "consumption_rate": self.powder_consumption_rate,
                "is_low": self.is_powder_low(),
                "is_depleted": self.is_powder_depleted()
            },
            "bed_quality": {
                "quality_score": self.quality_score,
                "quality_tier": self.bed_quality,
                "bed_height": self.bed_height,
                "bed_density": self.bed_density,
                "bed_roughness": self.bed_roughness,
                "bed_flatness": self.bed_flatness,
                "quality_checks_count": len(self.quality_checks)
            },
            "disturbance_analysis": {
                "total_disturbances": self.disturbance_count,
                "disturbance_types": self.disturbance_types,
                "last_disturbance": self.last_disturbance.isoformat() if self.last_disturbance else None,
                "disturbance_rate": "high" if self.disturbance_count > 5 else "normal"
            },
            "maintenance": {
                "cleaning_cycles": self.cleaning_cycles,
                "last_cleaning": self.last_cleaning_time.isoformat() if self.last_cleaning_time else None,
                "needs_cleaning": self.needs_cleaning(),
                "maintenance_schedule": self.maintenance_schedule
            },
            "bed_status": {
                "current_status": self.bed_status,
                "is_ready": self.is_bed_ready(),
                "is_disturbed": self.is_bed_disturbed(),
                "preparation_time": self.preparation_time.isoformat() if self.preparation_time else None
            },
            "recommendations": []
        }
        
        # Add recommendations based on bed state
        if self.is_powder_depleted():
            analytics["recommendations"].append("Refill powder immediately")
        elif self.is_powder_low():
            analytics["recommendations"].append("Plan powder refill soon")
        
        if self.needs_cleaning():
            analytics["recommendations"].append("Clean powder bed")
        
        if self.is_bed_disturbed():
            analytics["recommendations"].append("Address bed disturbance and re-prepare")
        
        if self.quality_score and self.quality_score < 70:
            analytics["recommendations"].append("Improve bed quality through better preparation")
        
        if self.disturbance_count > 5:
            analytics["recommendations"].append("Investigate frequent disturbances")
        
        if not self.is_bed_ready() and self.bed_status == "prepared":
            analytics["recommendations"].append("Complete bed quality check")
        
        return analytics