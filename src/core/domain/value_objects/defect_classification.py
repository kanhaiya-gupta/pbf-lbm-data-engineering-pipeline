"""
Defect classification value object for PBF-LB/M operations.
"""

from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

from .base_value_object import BaseValueObject
from ..enums import DefectType


class DefectSeverity(Enum):
    """Defect severity levels."""
    CRITICAL = "critical"
    MAJOR = "major"
    MINOR = "minor"
    COSMETIC = "cosmetic"


class DefectLocation(Enum):
    """Defect location categories."""
    SURFACE = "surface"
    INTERNAL = "internal"
    BOUNDARY = "boundary"
    INTERFACE = "interface"
    CORNER = "corner"
    EDGE = "edge"


@dataclass(frozen=True)
class DefectClassification(BaseValueObject):
    """
    Value object representing defect classification for PBF-LB/M operations.
    
    This immutable object contains comprehensive information about
    defects found in PBF processes and manufactured parts.
    """
    
    # Basic defect information
    defect_id: str
    defect_type: DefectType
    severity: DefectSeverity
    location: DefectLocation
    
    # Defect characteristics
    size: float  # mm or mm³
    shape: str  # e.g., "spherical", "elongated", "irregular"
    orientation: Optional[str] = None  # e.g., "horizontal", "vertical", "diagonal"
    
    # Location coordinates
    x_coordinate: float = 0.0  # mm
    y_coordinate: float = 0.0  # mm
    z_coordinate: float = 0.0  # mm
    
    # Defect properties
    density: Optional[float] = None  # g/cm³
    porosity: Optional[float] = None  # percentage
    connectivity: Optional[float] = None  # 0-1 scale
    
    # Impact assessment
    structural_impact: float = 0.0  # 0-1 scale
    functional_impact: float = 0.0  # 0-1 scale
    aesthetic_impact: float = 0.0  # 0-1 scale
    
    # Detection information
    detection_method: str = "unknown"  # e.g., "CT_scan", "visual", "ultrasonic"
    detection_confidence: float = 1.0  # 0-1 scale
    detection_timestamp: Optional[datetime] = None
    
    # Analysis results
    root_cause: Optional[str] = None
    contributing_factors: List[str] = None
    prevention_measures: List[str] = None
    
    # Quality impact
    quality_score_impact: float = 0.0  # 0-100
    rework_required: bool = False
    scrap_required: bool = False
    
    def __post_init__(self):
        """Initialize default values and validate."""
        if self.contributing_factors is None:
            object.__setattr__(self, 'contributing_factors', [])
        if self.prevention_measures is None:
            object.__setattr__(self, 'prevention_measures', [])
        if self.detection_timestamp is None:
            object.__setattr__(self, 'detection_timestamp', datetime.utcnow())
        super().__post_init__()
    
    def validate(self) -> None:
        """Validate defect classification."""
        # Basic validation
        if not self.defect_id:
            raise ValueError("Defect ID cannot be empty")
        if self.size <= 0:
            raise ValueError("Defect size must be positive")
        
        # Coordinate validation
        if any(coord < 0 for coord in [self.x_coordinate, self.y_coordinate, self.z_coordinate]):
            raise ValueError("Coordinates cannot be negative")
        
        # Property validation
        if self.density is not None and self.density <= 0:
            raise ValueError("Density must be positive")
        if self.porosity is not None and not 0 <= self.porosity <= 100:
            raise ValueError("Porosity must be between 0 and 100")
        if self.connectivity is not None and not 0 <= self.connectivity <= 1:
            raise ValueError("Connectivity must be between 0 and 1")
        
        # Impact validation
        for impact in [self.structural_impact, self.functional_impact, self.aesthetic_impact]:
            if not 0 <= impact <= 1:
                raise ValueError("Impact values must be between 0 and 1")
        
        # Detection validation
        if not 0 <= self.detection_confidence <= 1:
            raise ValueError("Detection confidence must be between 0 and 1")
        
        # Quality impact validation
        if not 0 <= self.quality_score_impact <= 100:
            raise ValueError("Quality score impact must be between 0 and 100")
    
    def get_severity_score(self) -> int:
        """Get numeric severity score (1-4)."""
        severity_scores = {
            DefectSeverity.CRITICAL: 4,
            DefectSeverity.MAJOR: 3,
            DefectSeverity.MINOR: 2,
            DefectSeverity.COSMETIC: 1
        }
        return severity_scores[self.severity]
    
    def get_total_impact_score(self) -> float:
        """Calculate total impact score (0-1)."""
        return (self.structural_impact + self.functional_impact + self.aesthetic_impact) / 3
    
    def get_risk_level(self) -> str:
        """Determine risk level based on severity and impact."""
        severity_score = self.get_severity_score()
        impact_score = self.get_total_impact_score()
        
        if severity_score >= 4 and impact_score >= 0.8:
            return "EXTREME"
        elif severity_score >= 3 and impact_score >= 0.6:
            return "HIGH"
        elif severity_score >= 2 and impact_score >= 0.4:
            return "MEDIUM"
        else:
            return "LOW"
    
    def is_critical(self) -> bool:
        """Check if defect is critical."""
        return self.severity == DefectSeverity.CRITICAL or self.get_risk_level() == "EXTREME"
    
    def is_acceptable(self) -> bool:
        """Check if defect is acceptable for production."""
        return (self.severity in [DefectSeverity.MINOR, DefectSeverity.COSMETIC] and 
                self.get_total_impact_score() < 0.3)
    
    def requires_immediate_action(self) -> bool:
        """Check if defect requires immediate action."""
        return (self.is_critical() or 
                self.rework_required or 
                self.scrap_required or
                self.get_risk_level() in ["EXTREME", "HIGH"])
    
    def get_location_description(self) -> str:
        """Get human-readable location description."""
        return f"({self.x_coordinate:.2f}, {self.y_coordinate:.2f}, {self.z_coordinate:.2f})"
    
    def get_size_category(self) -> str:
        """Get size category based on defect size."""
        if self.size < 0.1:
            return "micro"
        elif self.size < 1.0:
            return "small"
        elif self.size < 10.0:
            return "medium"
        else:
            return "large"
    
    def get_defect_summary(self) -> Dict[str, Any]:
        """Get comprehensive defect summary."""
        return {
            "defect_id": self.defect_id,
            "type": self.defect_type.value,
            "severity": self.severity.value,
            "location": self.location.value,
            "coordinates": self.get_location_description(),
            "size": self.size,
            "size_category": self.get_size_category(),
            "shape": self.shape,
            "severity_score": self.get_severity_score(),
            "total_impact": self.get_total_impact_score(),
            "risk_level": self.get_risk_level(),
            "is_critical": self.is_critical(),
            "is_acceptable": self.is_acceptable(),
            "requires_action": self.requires_immediate_action(),
            "detection_method": self.detection_method,
            "detection_confidence": self.detection_confidence,
            "quality_impact": self.quality_score_impact,
            "rework_required": self.rework_required,
            "scrap_required": self.scrap_required
        }
    
    def get_recommended_actions(self) -> List[str]:
        """Get recommended actions for this defect."""
        actions = []
        
        if self.is_critical():
            actions.append("IMMEDIATE: Stop production and investigate root cause")
            actions.append("IMMEDIATE: Implement containment measures")
        
        if self.rework_required:
            actions.append("Rework the affected area or component")
        
        if self.scrap_required:
            actions.append("Scrap the affected component")
        
        if self.get_risk_level() == "HIGH":
            actions.append("Review process parameters and quality controls")
        
        if self.severity == DefectSeverity.MAJOR:
            actions.append("Investigate contributing factors")
            actions.append("Update process monitoring procedures")
        
        if self.detection_confidence < 0.8:
            actions.append("Re-inspect using alternative detection methods")
        
        if self.quality_score_impact > 20:
            actions.append("Review quality acceptance criteria")
        
        return actions
    
    def get_prevention_strategies(self) -> List[str]:
        """Get prevention strategies for this defect type."""
        strategies = {
            DefectType.POROSITY: [
                "Optimize laser power and speed parameters",
                "Improve atmosphere control",
                "Ensure proper powder quality and handling"
            ],
            DefectType.KEYHOLE_POROSITY: [
                "Reduce laser power density",
                "Optimize scan speed",
                "Improve focus control"
            ],
            DefectType.LACK_OF_FUSION: [
                "Increase energy density",
                "Reduce hatch spacing",
                "Improve layer adhesion"
            ],
            DefectType.HOT_CRACKING: [
                "Optimize cooling rate",
                "Improve preheating",
                "Use crack-resistant materials"
            ],
            DefectType.COLD_CRACKING: [
                "Improve material ductility",
                "Optimize stress relief",
                "Control hydrogen content"
            ],
            DefectType.WARPAGE: [
                "Optimize support structures",
                "Improve thermal management",
                "Use stress-relieving strategies"
            ],
            DefectType.SURFACE_ROUGHNESS: [
                "Optimize laser parameters",
                "Improve scan patterns",
                "Use post-processing techniques"
            ]
        }
        
        return strategies.get(self.defect_type, [
            "Review process parameters",
            "Improve quality monitoring",
            "Update standard operating procedures"
        ])
    
    def calculate_repair_cost(self) -> float:
        """Calculate estimated repair cost based on defect characteristics."""
        base_cost = 100.0  # Base cost in currency units
        
        # Size factor
        size_factor = 1.0 + (self.size / 10.0)
        
        # Severity factor
        severity_factor = self.get_severity_score()
        
        # Impact factor
        impact_factor = 1.0 + self.get_total_impact_score()
        
        # Location factor
        location_factors = {
            DefectLocation.SURFACE: 1.0,
            DefectLocation.INTERNAL: 2.0,
            DefectLocation.BOUNDARY: 1.5,
            DefectLocation.INTERFACE: 1.8,
            DefectLocation.CORNER: 1.3,
            DefectLocation.EDGE: 1.2
        }
        location_factor = location_factors.get(self.location, 1.0)
        
        total_cost = base_cost * size_factor * severity_factor * impact_factor * location_factor
        
        return round(total_cost, 2)
    
    def get_quality_impact_assessment(self) -> Dict[str, Any]:
        """Get detailed quality impact assessment."""
        return {
            "overall_impact": self.get_total_impact_score(),
            "structural_impact": self.structural_impact,
            "functional_impact": self.functional_impact,
            "aesthetic_impact": self.aesthetic_impact,
            "quality_score_impact": self.quality_score_impact,
            "risk_level": self.get_risk_level(),
            "repair_cost": self.calculate_repair_cost(),
            "recommended_actions": self.get_recommended_actions(),
            "prevention_strategies": self.get_prevention_strategies(),
            "acceptability": "acceptable" if self.is_acceptable() else "not_acceptable"
        }