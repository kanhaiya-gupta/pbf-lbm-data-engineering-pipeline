"""
PBF Process Model

This module defines the Pydantic model for PBF-LB/M process data.
"""

from typing import Optional, Dict, Any, List
from datetime import datetime
from pydantic import Field, validator, root_validator
from enum import Enum

from .base_model import BaseDataModel

class AtmosphereType(str, Enum):
    """Enumeration of atmosphere types."""
    ARGON = "argon"
    NITROGEN = "nitrogen"
    HELIUM = "helium"
    VACUUM = "vacuum"
    AIR = "air"

class QualityGrade(str, Enum):
    """Enumeration of quality grades."""
    EXCELLENT = "EXCELLENT"
    GOOD = "GOOD"
    ACCEPTABLE = "ACCEPTABLE"
    POOR = "POOR"

class QualityMetrics(BaseDataModel):
    """Quality metrics for PBF process."""
    
    density: Optional[float] = Field(None, ge=0, le=100, description="Part density percentage")
    surface_roughness: Optional[float] = Field(None, ge=0, le=100, description="Surface roughness in micrometers")
    dimensional_accuracy: Optional[float] = Field(None, ge=0, le=10, description="Dimensional accuracy in mm")
    defect_count: Optional[int] = Field(None, ge=0, le=10000, description="Number of detected defects")
    
    def get_primary_key(self) -> str:
        return "id"
    
    def get_primary_key_value(self) -> Any:
        return getattr(self, "id", None)

class PBFProcessModel(BaseDataModel):
    """
    Pydantic model for PBF-LB/M process data.
    
    This model represents the core process data for Powder Bed Fusion - Laser Beam/Metal
    additive manufacturing, including process parameters, quality metrics, and metadata.
    """
    
    # Primary key and identifiers
    process_id: str = Field(..., min_length=1, max_length=100, description="Unique identifier for the PBF process")
    machine_id: str = Field(..., min_length=1, max_length=50, description="Identifier of the PBF machine")
    build_id: Optional[str] = Field(None, min_length=1, max_length=100, description="Build identifier for the manufacturing job")
    
    # Process timestamp
    timestamp: datetime = Field(..., description="Process timestamp in ISO format")
    
    # Process parameters
    layer_number: Optional[int] = Field(None, ge=0, le=10000, description="Current layer number being processed")
    temperature: float = Field(..., ge=0, le=2000, description="Process temperature in Celsius")
    pressure: float = Field(..., ge=0, le=1000, description="Chamber pressure in mbar")
    laser_power: float = Field(..., ge=0, le=1000, description="Laser power in watts")
    scan_speed: float = Field(..., ge=0, le=10000, description="Laser scan speed in mm/s")
    layer_height: float = Field(..., ge=0.01, le=1.0, description="Layer height in mm")
    hatch_spacing: Optional[float] = Field(None, ge=0.01, le=1.0, description="Hatch spacing in mm")
    exposure_time: Optional[float] = Field(None, ge=0, le=3600, description="Exposure time per layer in seconds")
    
    # Material and environment
    atmosphere: Optional[AtmosphereType] = Field(None, description="Atmosphere type")
    powder_material: Optional[str] = Field(None, min_length=1, max_length=100, description="Powder material type")
    powder_batch_id: Optional[str] = Field(None, min_length=1, max_length=100, description="Powder batch identifier")
    
    # Process parameters as key-value pairs
    process_parameters: Dict[str, str] = Field(default_factory=dict, description="Additional process parameters as key-value pairs")
    
    # Quality metrics
    quality_metrics: Optional[QualityMetrics] = Field(None, description="Quality metrics for the process")
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True
        validate_assignment = True
        schema_extra = {
            "example": {
                "process_id": "PBF_2024_001",
                "machine_id": "MACHINE_001",
                "build_id": "BUILD_2024_001",
                "timestamp": "2024-01-15T10:30:00Z",
                "layer_number": 150,
                "temperature": 1200.5,
                "pressure": 1.2,
                "laser_power": 200.0,
                "scan_speed": 1000.0,
                "layer_height": 0.05,
                "hatch_spacing": 0.1,
                "exposure_time": 120.0,
                "atmosphere": "argon",
                "powder_material": "Ti6Al4V",
                "powder_batch_id": "BATCH_001",
                "process_parameters": {
                    "laser_spot_size": "0.1",
                    "scan_pattern": "zigzag"
                },
                "quality_metrics": {
                    "density": 99.5,
                    "surface_roughness": 8.2,
                    "dimensional_accuracy": 0.05,
                    "defect_count": 0
                }
            }
        }
    
    @validator('process_id')
    def validate_process_id(cls, v):
        """Validate process ID format."""
        if not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError("Process ID must contain only alphanumeric characters, underscores, and hyphens")
        return v
    
    @validator('machine_id')
    def validate_machine_id(cls, v):
        """Validate machine ID format."""
        if not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError("Machine ID must contain only alphanumeric characters, underscores, and hyphens")
        return v
    
    @validator('build_id')
    def validate_build_id(cls, v):
        """Validate build ID format if provided."""
        if v is not None and not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError("Build ID must contain only alphanumeric characters, underscores, and hyphens")
        return v
    
    @validator('powder_material')
    def validate_powder_material(cls, v):
        """Validate powder material format."""
        if v is not None and len(v.strip()) == 0:
            raise ValueError("Powder material cannot be empty")
        return v
    
    @validator('powder_batch_id')
    def validate_powder_batch_id(cls, v):
        """Validate powder batch ID format."""
        if v is not None and not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError("Powder batch ID must contain only alphanumeric characters, underscores, and hyphens")
        return v
    
    @root_validator
    def validate_process_consistency(cls, values):
        """Validate process parameter consistency."""
        temperature = values.get('temperature')
        pressure = values.get('pressure')
        laser_power = values.get('laser_power')
        scan_speed = values.get('scan_speed')
        
        # Check for reasonable parameter combinations
        if temperature and pressure:
            if temperature > 1500 and pressure > 10:
                # High temperature and high pressure might indicate an issue
                pass  # Could add warning logic here
        
        if laser_power and scan_speed:
            if laser_power > 500 and scan_speed > 5000:
                # Very high power and speed combination
                pass  # Could add warning logic here
        
        return values
    
    def get_primary_key(self) -> str:
        """Get the primary key field name."""
        return "process_id"
    
    def get_primary_key_value(self) -> Any:
        """Get the primary key value."""
        return self.process_id
    
    def calculate_quality_score(self) -> float:
        """
        Calculate overall quality score based on quality metrics.
        
        Returns:
            Quality score between 0 and 100
        """
        if not self.quality_metrics:
            return 0.0
        
        metrics = self.quality_metrics
        score = 0.0
        factors = 0
        
        if metrics.density is not None:
            score += metrics.density
            factors += 1
        
        if metrics.surface_roughness is not None:
            # Lower surface roughness is better (invert the scale)
            roughness_score = max(0, 100 - metrics.surface_roughness)
            score += roughness_score
            factors += 1
        
        if metrics.dimensional_accuracy is not None:
            # Lower dimensional accuracy is better (invert the scale)
            accuracy_score = max(0, 100 - (metrics.dimensional_accuracy * 100))
            score += accuracy_score
            factors += 1
        
        if metrics.defect_count is not None:
            # Lower defect count is better
            defect_score = max(0, 100 - (metrics.defect_count * 10))
            score += defect_score
            factors += 1
        
        return score / factors if factors > 0 else 0.0
    
    def get_quality_grade(self) -> QualityGrade:
        """
        Get quality grade based on quality metrics.
        
        Returns:
            Quality grade
        """
        score = self.calculate_quality_score()
        
        if score >= 90:
            return QualityGrade.EXCELLENT
        elif score >= 80:
            return QualityGrade.GOOD
        elif score >= 70:
            return QualityGrade.ACCEPTABLE
        else:
            return QualityGrade.POOR
    
    def is_high_quality(self) -> bool:
        """
        Check if the process meets high quality standards.
        
        Returns:
            True if quality is excellent or good
        """
        return self.get_quality_grade() in [QualityGrade.EXCELLENT, QualityGrade.GOOD]
    
    def get_process_efficiency(self) -> float:
        """
        Calculate process efficiency based on parameters.
        
        Returns:
            Efficiency score between 0 and 1
        """
        # Simple efficiency calculation based on scan speed and layer height
        if self.scan_speed > 0 and self.layer_height > 0:
            # Higher scan speed and appropriate layer height indicate efficiency
            efficiency = min(1.0, (self.scan_speed / 1000) * (self.layer_height / 0.1))
            return efficiency
        return 0.0
    
    def get_energy_consumption(self) -> float:
        """
        Estimate energy consumption for the process.
        
        Returns:
            Estimated energy consumption in kWh
        """
        if self.laser_power and self.exposure_time:
            # Simple energy calculation: Power * Time
            energy_wh = self.laser_power * (self.exposure_time / 3600)  # Convert to hours
            return energy_wh / 1000  # Convert to kWh
        return 0.0
    
    def validate_process_parameters(self) -> Dict[str, Any]:
        """
        Validate process parameters against recommended ranges.
        
        Returns:
            Dictionary containing validation results
        """
        validation_results = {
            'valid': True,
            'warnings': [],
            'errors': []
        }
        
        # Temperature validation
        if self.temperature < 800 or self.temperature > 1600:
            validation_results['warnings'].append(f"Temperature {self.temperature}°C is outside typical range (800-1600°C)")
        
        # Pressure validation
        if self.pressure < 0.1 or self.pressure > 100:
            validation_results['warnings'].append(f"Pressure {self.pressure} mbar is outside typical range (0.1-100 mbar)")
        
        # Laser power validation
        if self.laser_power < 50 or self.laser_power > 500:
            validation_results['warnings'].append(f"Laser power {self.laser_power}W is outside typical range (50-500W)")
        
        # Scan speed validation
        if self.scan_speed < 100 or self.scan_speed > 5000:
            validation_results['warnings'].append(f"Scan speed {self.scan_speed} mm/s is outside typical range (100-5000 mm/s)")
        
        # Layer height validation
        if self.layer_height < 0.02 or self.layer_height > 0.2:
            validation_results['warnings'].append(f"Layer height {self.layer_height} mm is outside typical range (0.02-0.2 mm)")
        
        if validation_results['warnings']:
            validation_results['valid'] = False
        
        return validation_results
    
    def get_recommended_parameters(self) -> Dict[str, Any]:
        """
        Get recommended parameter ranges for the material.
        
        Returns:
            Dictionary containing recommended parameter ranges
        """
        recommendations = {
            'temperature': {'min': 800, 'max': 1600, 'optimal': 1200},
            'pressure': {'min': 0.1, 'max': 100, 'optimal': 1.0},
            'laser_power': {'min': 50, 'max': 500, 'optimal': 200},
            'scan_speed': {'min': 100, 'max': 5000, 'optimal': 1000},
            'layer_height': {'min': 0.02, 'max': 0.2, 'optimal': 0.05}
        }
        
        # Adjust recommendations based on material type
        if self.powder_material:
            material = self.powder_material.lower()
            if 'ti6al4v' in material or 'titanium' in material:
                recommendations['temperature']['optimal'] = 1200
                recommendations['laser_power']['optimal'] = 200
            elif 'inconel' in material or 'nickel' in material:
                recommendations['temperature']['optimal'] = 1400
                recommendations['laser_power']['optimal'] = 250
            elif 'stainless' in material or 'steel' in material:
                recommendations['temperature']['optimal'] = 1000
                recommendations['laser_power']['optimal'] = 180
        
        return recommendations
    
    def _calculate_consistency(self) -> float:
        """Calculate data consistency score (0-1)."""
        # Check if process parameters are within expected ranges
        validation = self.validate_process_parameters()
        if validation['valid']:
            return 1.0
        else:
            # Reduce score based on number of warnings
            warning_penalty = len(validation['warnings']) * 0.1
            return max(0.0, 1.0 - warning_penalty)
    
    def _calculate_accuracy(self) -> float:
        """Calculate data accuracy score (0-1)."""
        # Check if quality metrics are reasonable
        if not self.quality_metrics:
            return 0.5  # Partial score if no quality metrics
        
        metrics = self.quality_metrics
        accuracy_score = 1.0
        
        # Check density
        if metrics.density is not None:
            if metrics.density < 80 or metrics.density > 100:
                accuracy_score -= 0.2
        
        # Check surface roughness
        if metrics.surface_roughness is not None:
            if metrics.surface_roughness < 0 or metrics.surface_roughness > 50:
                accuracy_score -= 0.2
        
        # Check dimensional accuracy
        if metrics.dimensional_accuracy is not None:
            if metrics.dimensional_accuracy < 0 or metrics.dimensional_accuracy > 1:
                accuracy_score -= 0.2
        
        # Check defect count
        if metrics.defect_count is not None:
            if metrics.defect_count < 0 or metrics.defect_count > 1000:
                accuracy_score -= 0.2
        
        return max(0.0, accuracy_score)
    
    def _calculate_validity(self) -> float:
        """Calculate data validity score (0-1)."""
        # Check if all required fields are present and valid
        required_fields = ['process_id', 'machine_id', 'timestamp', 'temperature', 'pressure', 'laser_power', 'scan_speed', 'layer_height']
        valid_fields = 0
        
        for field in required_fields:
            value = getattr(self, field)
            if value is not None:
                valid_fields += 1
        
        return valid_fields / len(required_fields)
