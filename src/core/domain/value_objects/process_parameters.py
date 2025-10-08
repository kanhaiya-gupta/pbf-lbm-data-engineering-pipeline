"""
Process parameters value object for PBF-LB/M operations.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass

from .base_value_object import BaseValueObject


@dataclass(frozen=True)
class ProcessParameters(BaseValueObject):
    """
    Value object representing process parameters for PBF-LB/M operations.
    
    This immutable object contains all the parameters needed to configure
    and control a PBF process, including laser settings, scan patterns,
    and environmental conditions.
    """
    
    # Laser parameters
    laser_power: float  # Watts
    laser_speed: float  # mm/s
    laser_focus: float  # mm
    laser_spot_size: float  # mm
    
    # Scan parameters
    scan_pattern: str  # e.g., "zigzag", "spiral", "contour"
    hatch_spacing: float  # mm
    contour_offset: float  # mm
    scan_vector_length: float  # mm
    
    # Layer parameters
    layer_height: float  # mm
    layer_thickness: float  # mm
    overlap_ratio: float  # 0.0 to 1.0
    
    # Temperature parameters
    build_plate_temperature: float  # Celsius
    chamber_temperature: float  # Celsius
    preheat_temperature: float  # Celsius
    
    # Environmental parameters
    atmosphere_pressure: float  # Pa
    oxygen_level: float  # ppm
    inert_gas_flow_rate: float  # L/min
    
    # Powder parameters
    powder_type: str
    powder_size_distribution: Dict[str, float]  # size ranges and percentages
    powder_density: float  # g/cm³
    powder_flow_rate: float  # g/min
    
    # Build parameters
    build_volume_x: float  # mm
    build_volume_y: float  # mm
    build_volume_z: float  # mm
    support_structure: bool
    support_angle: float  # degrees
    
    # Quality parameters
    surface_finish_target: str  # e.g., "rough", "smooth", "polished"
    dimensional_tolerance: float  # mm
    density_target: float  # percentage
    
    def validate(self) -> None:
        """Validate process parameters."""
        # Laser parameter validation
        if self.laser_power <= 0:
            raise ValueError("Laser power must be positive")
        if self.laser_speed <= 0:
            raise ValueError("Laser speed must be positive")
        if self.laser_spot_size <= 0:
            raise ValueError("Laser spot size must be positive")
        
        # Scan parameter validation
        if self.hatch_spacing <= 0:
            raise ValueError("Hatch spacing must be positive")
        if self.scan_vector_length <= 0:
            raise ValueError("Scan vector length must be positive")
        
        # Layer parameter validation
        if self.layer_height <= 0:
            raise ValueError("Layer height must be positive")
        if not 0 <= self.overlap_ratio <= 1:
            raise ValueError("Overlap ratio must be between 0 and 1")
        
        # Temperature validation
        if self.build_plate_temperature < 0:
            raise ValueError("Build plate temperature cannot be negative")
        if self.chamber_temperature < 0:
            raise ValueError("Chamber temperature cannot be negative")
        
        # Environmental validation
        if self.atmosphere_pressure <= 0:
            raise ValueError("Atmosphere pressure must be positive")
        if self.oxygen_level < 0:
            raise ValueError("Oxygen level cannot be negative")
        
        # Powder validation
        if self.powder_density <= 0:
            raise ValueError("Powder density must be positive")
        if self.powder_flow_rate < 0:
            raise ValueError("Powder flow rate cannot be negative")
        
        # Build volume validation
        if any(dim <= 0 for dim in [self.build_volume_x, self.build_volume_y, self.build_volume_z]):
            raise ValueError("Build volume dimensions must be positive")
        
        # Quality validation
        if self.dimensional_tolerance < 0:
            raise ValueError("Dimensional tolerance cannot be negative")
        if not 0 <= self.density_target <= 100:
            raise ValueError("Density target must be between 0 and 100")
    
    def get_energy_density(self) -> float:
        """Calculate energy density (J/mm³)."""
        return (self.laser_power * 1000) / (self.laser_speed * self.hatch_spacing * self.layer_height)
    
    def get_scan_time_per_layer(self) -> float:
        """Calculate estimated scan time per layer (seconds)."""
        area = self.build_volume_x * self.build_volume_y
        scan_length = area / self.hatch_spacing
        return scan_length / self.laser_speed
    
    def get_total_build_time(self) -> float:
        """Calculate estimated total build time (seconds)."""
        layers = int(self.build_volume_z / self.layer_height)
        time_per_layer = self.get_scan_time_per_layer()
        return layers * time_per_layer
    
    def get_powder_consumption(self) -> float:
        """Calculate estimated powder consumption (grams)."""
        volume = self.build_volume_x * self.build_volume_y * self.build_volume_z
        return volume * self.powder_density / 1000  # Convert mm³ to cm³
    
    def is_high_energy_density(self) -> bool:
        """Check if energy density is high (> 100 J/mm³)."""
        return self.get_energy_density() > 100
    
    def is_low_energy_density(self) -> bool:
        """Check if energy density is low (< 50 J/mm³)."""
        return self.get_energy_density() < 50
    
    def requires_support(self) -> bool:
        """Check if support structures are required."""
        return self.support_structure
    
    def get_optimal_scan_pattern(self) -> str:
        """Get optimal scan pattern based on parameters."""
        if self.build_volume_x > 100 or self.build_volume_y > 100:
            return "zigzag"
        elif self.surface_finish_target == "smooth":
            return "spiral"
        else:
            return "contour"
    
    def get_quality_indicators(self) -> Dict[str, Any]:
        """Get quality indicators based on parameters."""
        return {
            "energy_density": self.get_energy_density(),
            "scan_time_per_layer": self.get_scan_time_per_layer(),
            "total_build_time": self.get_total_build_time(),
            "powder_consumption": self.get_powder_consumption(),
            "is_high_energy": self.is_high_energy_density(),
            "is_low_energy": self.is_low_energy_density(),
            "requires_support": self.requires_support(),
            "optimal_pattern": self.get_optimal_scan_pattern()
        }
    
    def get_parameter_ranges(self) -> Dict[str, Dict[str, float]]:
        """Get acceptable parameter ranges for optimization."""
        return {
            "laser_power": {"min": 50, "max": 1000, "current": self.laser_power},
            "laser_speed": {"min": 100, "max": 5000, "current": self.laser_speed},
            "hatch_spacing": {"min": 0.05, "max": 0.5, "current": self.hatch_spacing},
            "layer_height": {"min": 0.02, "max": 0.2, "current": self.layer_height},
            "build_plate_temp": {"min": 20, "max": 200, "current": self.build_plate_temperature},
            "chamber_temp": {"min": 20, "max": 100, "current": self.chamber_temperature}
        }
    
    def optimize_for_quality(self) -> 'ProcessParameters':
        """Create optimized parameters for quality."""
        # This would typically involve ML models or optimization algorithms
        # For now, return adjusted parameters based on heuristics
        optimized_params = {
            "laser_power": min(self.laser_power * 1.1, 1000),
            "laser_speed": max(self.laser_speed * 0.9, 100),
            "hatch_spacing": max(self.hatch_spacing * 0.95, 0.05),
            "layer_height": max(self.layer_height * 0.95, 0.02),
            "overlap_ratio": min(self.overlap_ratio * 1.05, 1.0),
            "dimensional_tolerance": max(self.dimensional_tolerance * 0.9, 0.01),
            "density_target": min(self.density_target * 1.02, 100)
        }
        
        # Create new instance with optimized parameters
        current_dict = self.to_dict()
        current_dict.update(optimized_params)
        current_dict["updated_at"] = datetime.utcnow()
        
        return ProcessParameters(**current_dict)
    
    def optimize_for_speed(self) -> 'ProcessParameters':
        """Create optimized parameters for speed."""
        optimized_params = {
            "laser_power": min(self.laser_power * 1.2, 1000),
            "laser_speed": min(self.laser_speed * 1.3, 5000),
            "hatch_spacing": min(self.hatch_spacing * 1.1, 0.5),
            "layer_height": min(self.layer_height * 1.1, 0.2),
            "overlap_ratio": max(self.overlap_ratio * 0.95, 0.0)
        }
        
        current_dict = self.to_dict()
        current_dict.update(optimized_params)
        current_dict["updated_at"] = datetime.utcnow()
        
        return ProcessParameters(**current_dict)