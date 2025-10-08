"""
Quality metrics value object for PBF-LB/M operations.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass

from .base_value_object import BaseValueObject
from ..enums import QualityTier, DefectType


@dataclass(frozen=True)
class QualityMetrics(BaseValueObject):
    """
    Value object representing quality metrics for PBF-LB/M operations.
    
    This immutable object contains comprehensive quality measurements
    and assessments for PBF processes and manufactured parts.
    """
    
    # Overall quality assessment
    overall_quality_tier: QualityTier
    quality_score: float  # 0-100
    pass_fail_status: bool
    
    # Dimensional quality
    dimensional_accuracy: float  # mm
    dimensional_tolerance: float  # mm
    flatness_deviation: float  # mm
    perpendicularity_deviation: float  # mm
    surface_roughness_ra: float  # μm
    surface_roughness_rz: float  # μm
    
    # Density and porosity
    relative_density: float  # percentage
    porosity_percentage: float  # percentage
    pore_size_distribution: Dict[str, float]  # size ranges and percentages
    pore_connectivity: float  # 0-1 scale
    
    # Mechanical properties
    ultimate_tensile_strength: Optional[float] = None  # MPa
    yield_strength: Optional[float] = None  # MPa
    elongation_at_break: Optional[float] = None  # percentage
    hardness_hv: Optional[float] = None  # Vickers hardness
    fatigue_strength: Optional[float] = None  # MPa
    
    # Defect analysis
    defect_count: int = 0
    defect_types: List[DefectType] = None
    critical_defects: int = 0
    major_defects: int = 0
    minor_defects: int = 0
    cosmetic_defects: int = 0
    
    # Microstructural quality
    grain_size: Optional[float] = None  # μm
    grain_size_distribution: Optional[Dict[str, float]] = None
    phase_composition: Optional[Dict[str, float]] = None
    residual_stress: Optional[float] = None  # MPa
    
    # Process quality indicators
    build_success_rate: float = 100.0  # percentage
    layer_adhesion_quality: float = 100.0  # percentage
    powder_utilization_efficiency: float = 100.0  # percentage
    energy_efficiency: float = 100.0  # percentage
    
    def __post_init__(self):
        """Initialize default values and validate."""
        if self.defect_types is None:
            object.__setattr__(self, 'defect_types', [])
        super().__post_init__()
    
    def validate(self) -> None:
        """Validate quality metrics."""
        # Quality score validation
        if not 0 <= self.quality_score <= 100:
            raise ValueError("Quality score must be between 0 and 100")
        
        # Dimensional validation
        if self.dimensional_accuracy < 0:
            raise ValueError("Dimensional accuracy cannot be negative")
        if self.dimensional_tolerance < 0:
            raise ValueError("Dimensional tolerance cannot be negative")
        if self.flatness_deviation < 0:
            raise ValueError("Flatness deviation cannot be negative")
        if self.perpendicularity_deviation < 0:
            raise ValueError("Perpendicularity deviation cannot be negative")
        
        # Surface roughness validation
        if self.surface_roughness_ra < 0:
            raise ValueError("Surface roughness Ra cannot be negative")
        if self.surface_roughness_rz < 0:
            raise ValueError("Surface roughness Rz cannot be negative")
        
        # Density validation
        if not 0 <= self.relative_density <= 100:
            raise ValueError("Relative density must be between 0 and 100")
        if not 0 <= self.porosity_percentage <= 100:
            raise ValueError("Porosity percentage must be between 0 and 100")
        if not 0 <= self.pore_connectivity <= 1:
            raise ValueError("Pore connectivity must be between 0 and 1")
        
        # Mechanical properties validation
        if self.ultimate_tensile_strength is not None and self.ultimate_tensile_strength < 0:
            raise ValueError("Ultimate tensile strength cannot be negative")
        if self.yield_strength is not None and self.yield_strength < 0:
            raise ValueError("Yield strength cannot be negative")
        if self.elongation_at_break is not None and self.elongation_at_break < 0:
            raise ValueError("Elongation at break cannot be negative")
        if self.hardness_hv is not None and self.hardness_hv < 0:
            raise ValueError("Hardness cannot be negative")
        if self.fatigue_strength is not None and self.fatigue_strength < 0:
            raise ValueError("Fatigue strength cannot be negative")
        
        # Defect validation
        if self.defect_count < 0:
            raise ValueError("Defect count cannot be negative")
        if self.critical_defects < 0:
            raise ValueError("Critical defects cannot be negative")
        if self.major_defects < 0:
            raise ValueError("Major defects cannot be negative")
        if self.minor_defects < 0:
            raise ValueError("Minor defects cannot be negative")
        if self.cosmetic_defects < 0:
            raise ValueError("Cosmetic defects cannot be negative")
        
        # Process quality validation
        if not 0 <= self.build_success_rate <= 100:
            raise ValueError("Build success rate must be between 0 and 100")
        if not 0 <= self.layer_adhesion_quality <= 100:
            raise ValueError("Layer adhesion quality must be between 0 and 100")
        if not 0 <= self.powder_utilization_efficiency <= 100:
            raise ValueError("Powder utilization efficiency must be between 0 and 100")
        if not 0 <= self.energy_efficiency <= 100:
            raise ValueError("Energy efficiency must be between 0 and 100")
    
    def get_dimensional_quality_score(self) -> float:
        """Calculate dimensional quality score (0-100)."""
        if self.dimensional_tolerance == 0:
            return 100.0
        
        accuracy_ratio = self.dimensional_accuracy / self.dimensional_tolerance
        if accuracy_ratio <= 0.5:
            return 100.0
        elif accuracy_ratio <= 1.0:
            return 100.0 - (accuracy_ratio - 0.5) * 100
        else:
            return max(0.0, 50.0 - (accuracy_ratio - 1.0) * 50)
    
    def get_surface_quality_score(self) -> float:
        """Calculate surface quality score (0-100)."""
        # Based on Ra and Rz values (lower is better)
        ra_score = max(0, 100 - self.surface_roughness_ra * 10)
        rz_score = max(0, 100 - self.surface_roughness_rz * 2)
        return (ra_score + rz_score) / 2
    
    def get_density_quality_score(self) -> float:
        """Calculate density quality score (0-100)."""
        # Higher density is better
        density_score = self.relative_density
        porosity_penalty = self.porosity_percentage * 2
        return max(0, density_score - porosity_penalty)
    
    def get_defect_quality_score(self) -> float:
        """Calculate defect quality score (0-100)."""
        if self.defect_count == 0:
            return 100.0
        
        # Weighted penalty based on defect severity
        total_penalty = (
            self.critical_defects * 25 +
            self.major_defects * 15 +
            self.minor_defects * 5 +
            self.cosmetic_defects * 1
        )
        
        return max(0, 100 - total_penalty)
    
    def get_mechanical_quality_score(self) -> float:
        """Calculate mechanical quality score (0-100)."""
        if not all([self.ultimate_tensile_strength, self.yield_strength, self.elongation_at_break]):
            return 0.0
        
        # Normalize mechanical properties (assuming target values)
        uts_score = min(100, (self.ultimate_tensile_strength / 800) * 100)  # Target: 800 MPa
        ys_score = min(100, (self.yield_strength / 600) * 100)  # Target: 600 MPa
        elongation_score = min(100, (self.elongation_at_break / 15) * 100)  # Target: 15%
        
        return (uts_score + ys_score + elongation_score) / 3
    
    def get_overall_quality_score(self) -> float:
        """Calculate overall quality score (0-100)."""
        dimensional_score = self.get_dimensional_quality_score()
        surface_score = self.get_surface_quality_score()
        density_score = self.get_density_quality_score()
        defect_score = self.get_defect_quality_score()
        mechanical_score = self.get_mechanical_quality_score()
        
        # Weighted average
        weights = {
            'dimensional': 0.25,
            'surface': 0.20,
            'density': 0.20,
            'defect': 0.25,
            'mechanical': 0.10
        }
        
        overall_score = (
            dimensional_score * weights['dimensional'] +
            surface_score * weights['surface'] +
            density_score * weights['density'] +
            defect_score * weights['defect'] +
            mechanical_score * weights['mechanical']
        )
        
        return round(overall_score, 2)
    
    def get_quality_tier(self) -> QualityTier:
        """Determine quality tier based on overall score."""
        score = self.get_overall_quality_score()
        
        if score >= 95:
            return QualityTier.OUTSTANDING
        elif score >= 90:
            return QualityTier.EXCELLENT
        elif score >= 85:
            return QualityTier.PREMIUM
        elif score >= 80:
            return QualityTier.HIGH
        elif score >= 75:
            return QualityTier.GOOD
        elif score >= 70:
            return QualityTier.STANDARD
        elif score >= 60:
            return QualityTier.ACCEPTABLE
        elif score >= 45:
            return QualityTier.FAIR
        elif score >= 30:
            return QualityTier.POOR
        elif score >= 15:
            return QualityTier.SUBSTANDARD
        else:
            return QualityTier.UNACCEPTABLE
    
    def is_acceptable_quality(self) -> bool:
        """Check if quality meets minimum acceptable standards."""
        return self.get_overall_quality_score() >= 70
    
    def is_production_ready(self) -> bool:
        """Check if quality is suitable for production."""
        return self.get_overall_quality_score() >= 75
    
    def is_customer_ready(self) -> bool:
        """Check if quality is suitable for customer delivery."""
        return self.get_overall_quality_score() >= 80
    
    def has_critical_defects(self) -> bool:
        """Check if there are any critical defects."""
        return self.critical_defects > 0
    
    def get_quality_summary(self) -> Dict[str, Any]:
        """Get comprehensive quality summary."""
        return {
            "overall_score": self.get_overall_quality_score(),
            "quality_tier": self.get_quality_tier().value,
            "dimensional_score": self.get_dimensional_quality_score(),
            "surface_score": self.get_surface_quality_score(),
            "density_score": self.get_density_quality_score(),
            "defect_score": self.get_defect_quality_score(),
            "mechanical_score": self.get_mechanical_quality_score(),
            "is_acceptable": self.is_acceptable_quality(),
            "is_production_ready": self.is_production_ready(),
            "is_customer_ready": self.is_customer_ready(),
            "has_critical_defects": self.has_critical_defects(),
            "total_defects": self.defect_count,
            "defect_breakdown": {
                "critical": self.critical_defects,
                "major": self.major_defects,
                "minor": self.minor_defects,
                "cosmetic": self.cosmetic_defects
            }
        }
    
    def get_improvement_recommendations(self) -> List[str]:
        """Get recommendations for quality improvement."""
        recommendations = []
        
        if self.get_dimensional_quality_score() < 80:
            recommendations.append("Improve dimensional accuracy through process parameter optimization")
        
        if self.get_surface_quality_score() < 80:
            recommendations.append("Optimize surface finish through laser parameter adjustment")
        
        if self.get_density_quality_score() < 80:
            recommendations.append("Increase density through energy density optimization")
        
        if self.get_defect_quality_score() < 80:
            recommendations.append("Reduce defects through process monitoring and control")
        
        if self.has_critical_defects():
            recommendations.append("Address critical defects immediately - process may need redesign")
        
        if self.relative_density < 95:
            recommendations.append("Increase relative density to improve mechanical properties")
        
        if self.porosity_percentage > 5:
            recommendations.append("Reduce porosity through process optimization")
        
        return recommendations