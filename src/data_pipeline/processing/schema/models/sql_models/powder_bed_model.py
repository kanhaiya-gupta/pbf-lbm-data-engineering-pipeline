"""
Powder Bed Model

This module defines the Pydantic model for powder bed monitoring data.
"""

from typing import Optional, Dict, Any, List
from datetime import datetime
from pydantic import Field, field_validator, model_validator
from enum import Enum

from .base_model import BaseDataModel

class ImageFormat(str, Enum):
    """Enumeration of image formats."""
    JPEG = "JPEG"
    PNG = "PNG"
    TIFF = "TIFF"
    RAW = "RAW"
    BMP = "BMP"

class ProcessingStatus(str, Enum):
    """Enumeration of processing statuses."""
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"

class PowderBedDefectType(str, Enum):
    """Enumeration of powder bed defect types."""
    INSUFFICIENT_POWDER = "INSUFFICIENT_POWDER"
    EXCESS_POWDER = "EXCESS_POWDER"
    CONTAMINATION = "CONTAMINATION"
    AGGLOMERATION = "AGGLOMERATION"
    SEGREGATION = "SEGREGATION"
    SURFACE_IRREGULARITY = "SURFACE_IRREGULARITY"

class DefectSeverity(str, Enum):
    """Enumeration of defect severity levels."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class QualityAssessment(str, Enum):
    """Enumeration of quality assessments."""
    EXCELLENT = "EXCELLENT"
    GOOD = "GOOD"
    ACCEPTABLE = "ACCEPTABLE"
    POOR = "POOR"
    UNACCEPTABLE = "UNACCEPTABLE"

class CaptureSettings(BaseDataModel):
    """Camera capture settings."""
    
    exposure_time: float = Field(..., ge=0.001, le=60, description="Exposure time in seconds")
    aperture: float = Field(..., ge=0.5, le=32, description="Camera aperture value")
    iso: int = Field(..., ge=50, le=25600, description="ISO sensitivity")
    white_balance: str = Field(..., min_length=1, max_length=50, description="White balance setting")
    lighting_conditions: str = Field(..., min_length=1, max_length=100, description="Lighting conditions during capture")
    
    def get_primary_key(self) -> str:
        return "id"
    
    def get_primary_key_value(self) -> Any:
        return getattr(self, "id", None)

class ImageMetadata(BaseDataModel):
    """Powder bed image metadata."""
    
    image_id: str = Field(..., min_length=1, max_length=100, description="Unique identifier for the powder bed image")
    camera_id: str = Field(..., min_length=1, max_length=50, description="Camera identifier used for capture")
    image_format: ImageFormat = Field(..., description="Image file format")
    resolution: str = Field(..., description="Image resolution")
    file_size: int = Field(..., ge=0, description="Image file size in bytes")
    file_path: str = Field(..., min_length=1, max_length=500, description="Path to the image file")
    capture_settings: CaptureSettings = Field(..., description="Camera capture settings")
    
    def get_primary_key(self) -> str:
        return "image_id"
    
    def get_primary_key_value(self) -> Any:
        return self.image_id

class ParticleSizeDistribution(BaseDataModel):
    """Particle size distribution."""
    
    d10: float = Field(..., ge=0.1, le=1000, description="10th percentile particle size in micrometers")
    d50: float = Field(..., ge=0.1, le=1000, description="50th percentile particle size in micrometers")
    d90: float = Field(..., ge=0.1, le=1000, description="90th percentile particle size in micrometers")
    span: float = Field(..., ge=0, description="Particle size distribution span")
    
    @field_validator('d50')
    @classmethod
    def validate_d50(cls, v):
        """Validate d50 is reasonable."""
        if v <= 0:
            raise ValueError("d50 must be positive")
        return v
    
    @field_validator('span')
    @classmethod
    def validate_span(cls, v):
        """Validate span is reasonable."""
        if v < 0:
            raise ValueError("Span must be non-negative")
        return v
    
    def get_primary_key(self) -> str:
        return "id"
    
    def get_primary_key_value(self) -> Any:
        return getattr(self, "id", None)

class PowderCharacteristics(BaseDataModel):
    """Powder characteristics."""
    
    material_type: str = Field(..., min_length=1, max_length=100, description="Powder material type")
    particle_size_distribution: Optional[ParticleSizeDistribution] = Field(None, description="Particle size distribution")
    powder_density: Optional[float] = Field(None, ge=0.1, le=20, description="Powder density in g/cm³")
    flowability: Optional[float] = Field(None, ge=0, le=100, description="Powder flowability index")
    moisture_content: Optional[float] = Field(None, ge=0, le=100, description="Moisture content percentage")
    
    def get_primary_key(self) -> str:
        return "id"
    
    def get_primary_key_value(self) -> Any:
        return getattr(self, "id", None)

class BedQualityMetrics(BaseDataModel):
    """Powder bed quality metrics."""
    
    uniformity_score: float = Field(..., ge=0, le=100, description="Powder bed uniformity score (0-100)")
    coverage_percentage: float = Field(..., ge=0, le=100, description="Powder coverage percentage")
    thickness_consistency: float = Field(..., ge=0, le=100, description="Layer thickness consistency score (0-100)")
    surface_roughness: Optional[float] = Field(None, ge=0, le=100, description="Surface roughness in micrometers")
    density_variation: Optional[float] = Field(None, ge=0, le=1, description="Density variation coefficient")
    defect_density: Optional[float] = Field(None, ge=0, description="Defect density per cm²")
    
    def get_primary_key(self) -> str:
        return "id"
    
    def get_primary_key_value(self) -> Any:
        return getattr(self, "id", None)

class ColorBalance(BaseDataModel):
    """Color balance analysis."""
    
    red_channel: float = Field(..., ge=0, le=255, description="Red channel average value")
    green_channel: float = Field(..., ge=0, le=255, description="Green channel average value")
    blue_channel: float = Field(..., ge=0, le=255, description="Blue channel average value")
    
    def get_primary_key(self) -> str:
        return "id"
    
    def get_primary_key_value(self) -> Any:
        return getattr(self, "id", None)

class TextureAnalysis(BaseDataModel):
    """Texture analysis results."""
    
    homogeneity: float = Field(..., ge=0, le=1, description="Texture homogeneity")
    contrast: float = Field(..., ge=0, description="Texture contrast")
    energy: float = Field(..., ge=0, le=1, description="Texture energy")
    entropy: float = Field(..., ge=0, description="Texture entropy")
    
    def get_primary_key(self) -> str:
        return "id"
    
    def get_primary_key_value(self) -> Any:
        return getattr(self, "id", None)

class ImageAnalysis(BaseDataModel):
    """Image analysis results."""
    
    brightness: float = Field(..., ge=0, le=255, description="Average image brightness (0-255)")
    contrast: float = Field(..., ge=0, description="Image contrast value")
    sharpness: float = Field(..., ge=0, description="Image sharpness score")
    noise_level: float = Field(..., ge=0, description="Image noise level")
    color_balance: ColorBalance = Field(..., description="Color balance analysis")
    texture_analysis: Optional[TextureAnalysis] = Field(None, description="Texture analysis results")
    
    def get_primary_key(self) -> str:
        return "id"
    
    def get_primary_key_value(self) -> Any:
        return getattr(self, "id", None)

class DefectLocation(BaseDataModel):
    """Defect location and size."""
    
    x_coordinate: float = Field(..., description="X coordinate in mm")
    y_coordinate: float = Field(..., description="Y coordinate in mm")
    area: float = Field(..., ge=0, description="Defect area in mm²")
    
    def get_primary_key(self) -> str:
        return "id"
    
    def get_primary_key_value(self) -> Any:
        return getattr(self, "id", None)

class PowderBedDefect(BaseDataModel):
    """Powder bed defect information."""
    
    defect_id: str = Field(..., min_length=1, max_length=100, description="Unique defect identifier")
    type: PowderBedDefectType = Field(..., description="Type of powder bed defect")
    location: DefectLocation = Field(..., description="Defect location and size")
    severity: DefectSeverity = Field(..., description="Defect severity level")
    confidence_score: float = Field(..., ge=0, le=100, description="Detection confidence score (0-100)")
    
    def get_primary_key(self) -> str:
        return "defect_id"
    
    def get_primary_key_value(self) -> Any:
        return self.defect_id

class DefectDetection(BaseDataModel):
    """Defect detection results."""
    
    defects_detected: bool = Field(..., description="Whether defects were detected")
    defect_count: int = Field(..., ge=0, description="Number of detected defects")
    defect_types: List[PowderBedDefect] = Field(..., description="Detected defects")
    overall_quality_assessment: QualityAssessment = Field(..., description="Overall powder bed quality assessment")
    
    def get_primary_key(self) -> str:
        return "id"
    
    def get_primary_key_value(self) -> Any:
        return getattr(self, "id", None)

class EnvironmentalConditions(BaseDataModel):
    """Environmental conditions during monitoring."""
    
    temperature: float = Field(..., ge=-50, le=100, description="Ambient temperature in Celsius")
    humidity: float = Field(..., ge=0, le=100, description="Relative humidity percentage")
    pressure: float = Field(..., ge=0, le=2000, description="Atmospheric pressure in mbar")
    vibration_level: Optional[float] = Field(None, ge=0, description="Vibration level in g")
    
    def get_primary_key(self) -> str:
        return "id"
    
    def get_primary_key_value(self) -> Any:
        return getattr(self, "id", None)

class PowderBedModel(BaseDataModel):
    """
    Pydantic model for powder bed monitoring data matching the SQL schema structure.
    
    This model represents powder bed monitoring data for PBF-LB/M additive manufacturing
    with flat fields that match the PostgreSQL schema exactly.
    """
    
    # Primary key and identifiers
    bed_id: str = Field(..., min_length=1, max_length=100, description="Unique identifier for the powder bed record")
    process_id: str = Field(..., min_length=1, max_length=100, description="Associated PBF process identifier")
    layer_number: int = Field(..., ge=0, le=10000, description="Current layer number")
    
    # Monitoring timestamp
    timestamp: datetime = Field(..., description="Monitoring timestamp in ISO format")
    
    # Image metadata (flat fields matching SQL schema)
    image_id: str = Field(..., min_length=1, max_length=100, description="Unique identifier for the image")
    camera_id: str = Field(..., min_length=1, max_length=50, description="Camera identifier")
    image_format: str = Field(..., description="Image format")
    resolution: str = Field(..., min_length=1, max_length=20, description="Image resolution")
    file_size: int = Field(..., ge=0, description="File size in bytes")
    file_path: str = Field(..., min_length=1, max_length=500, description="Path to the image file")
    
    # Camera capture settings (flat fields matching SQL schema)
    exposure_time: float = Field(..., ge=0.001, le=60, description="Exposure time in seconds")
    aperture: float = Field(..., ge=0.5, le=32, description="Camera aperture")
    iso: int = Field(..., ge=50, le=25600, description="ISO setting")
    white_balance: str = Field(..., min_length=1, max_length=50, description="White balance setting")
    lighting_conditions: str = Field(..., min_length=1, max_length=100, description="Lighting conditions")
    
    # Powder characteristics (flat fields matching SQL schema)
    material_type: str = Field(..., min_length=1, max_length=100, description="Powder material type")
    particle_size_d10: Optional[float] = Field(None, description="Particle size D10 in micrometers")
    particle_size_d50: Optional[float] = Field(None, description="Particle size D50 in micrometers")
    particle_size_d90: Optional[float] = Field(None, description="Particle size D90 in micrometers")
    particle_size_span: Optional[float] = Field(None, description="Particle size span")
    powder_density: Optional[float] = Field(None, ge=0.1, le=20, description="Powder density in g/cm³")
    flowability: Optional[float] = Field(None, ge=0, le=100, description="Powder flowability percentage")
    moisture_content: Optional[float] = Field(None, ge=0, le=100, description="Moisture content percentage")
    
    # Bed quality metrics (flat fields matching SQL schema)
    uniformity_score: float = Field(..., ge=0, le=100, description="Powder bed uniformity score")
    coverage_percentage: float = Field(..., ge=0, le=100, description="Powder coverage percentage")
    thickness_consistency: float = Field(..., ge=0, le=100, description="Thickness consistency score")
    surface_roughness: Optional[float] = Field(None, ge=0, le=100, description="Surface roughness in micrometers")
    density_variation: Optional[float] = Field(None, ge=0, le=1, description="Density variation coefficient")
    defect_density: Optional[float] = Field(None, ge=0, description="Defect density per unit area")
    
    # Image analysis (flat fields matching SQL schema)
    brightness: Optional[float] = Field(None, ge=0, le=255, description="Image brightness")
    contrast: Optional[float] = Field(None, description="Image contrast")
    sharpness: Optional[float] = Field(None, description="Image sharpness")
    noise_level: Optional[float] = Field(None, description="Image noise level")
    red_channel: Optional[float] = Field(None, ge=0, le=255, description="Red channel value")
    green_channel: Optional[float] = Field(None, ge=0, le=255, description="Green channel value")
    blue_channel: Optional[float] = Field(None, ge=0, le=255, description="Blue channel value")
    
    # Texture analysis (flat fields matching SQL schema)
    texture_homogeneity: Optional[float] = Field(None, ge=0, le=1, description="Texture homogeneity")
    texture_contrast: Optional[float] = Field(None, description="Texture contrast")
    texture_energy: Optional[float] = Field(None, ge=0, le=1, description="Texture energy")
    texture_entropy: Optional[float] = Field(None, description="Texture entropy")
    
    # Defect detection (flat fields matching SQL schema)
    defects_detected: Optional[bool] = Field(None, description="Whether defects were detected")
    defect_count: Optional[int] = Field(None, ge=0, description="Number of detected defects")
    overall_quality_assessment: Optional[str] = Field(None, description="Overall quality assessment")
    
    # Environmental conditions (flat fields matching SQL schema)
    ambient_temperature: Optional[float] = Field(None, ge=-50, le=100, description="Ambient temperature in Celsius")
    relative_humidity: Optional[float] = Field(None, ge=0, le=100, description="Relative humidity percentage")
    atmospheric_pressure: Optional[float] = Field(None, ge=0, le=2000, description="Atmospheric pressure in mbar")
    vibration_level: Optional[float] = Field(None, description="Vibration level")
    
    # Processing status
    processing_status: str = Field(..., description="Processing status")
    
    # Additional data (JSONB field)
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True
        validate_assignment = True
        schema_extra = {
            "example": {
                "bed_id": "BED_2024_001",
                "process_id": "PBF_2024_001",
                "layer_number": 150,
                "timestamp": "2024-01-15T10:30:00Z",
                "image_metadata": {
                    "image_id": "IMG_2024_001",
                    "camera_id": "CAM_001",
                    "image_format": "PNG",
                    "resolution": "1920x1080",
                    "file_size": 2048576,
                    "file_path": "/data/powder_bed_images/IMG_2024_001.png",
                    "capture_settings": {
                        "exposure_time": 0.1,
                        "aperture": 5.6,
                        "iso": 400,
                        "white_balance": "auto",
                        "lighting_conditions": "LED_ring_light"
                    }
                },
                "powder_characteristics": {
                    "material_type": "Ti6Al4V",
                    "particle_size_distribution": {
                        "d10": 15.0,
                        "d50": 25.0,
                        "d90": 40.0,
                        "span": 1.0
                    },
                    "powder_density": 4.43,
                    "flowability": 85.0,
                    "moisture_content": 0.1
                },
                "uniformity_score": 92.5,
                "coverage_percentage": 98.0,
                "thickness_consistency": 89.0,
                "surface_roughness": 12.5,
                "density_variation": 0.05,
                "defect_density": 0.2,
                "brightness": 128.5,
                "contrast": 45.2,
                "sharpness": 78.3,
                "noise_level": 2.1,
                "red_channel": 130.2,
                "green_channel": 128.8,
                "blue_channel": 126.1,
                "texture_homogeneity": 0.85,
                "texture_contrast": 12.3,
                "texture_energy": 0.92,
                "texture_entropy": 3.45,
                "defects_detected": True,
                "defect_count": 3,
                "overall_quality_assessment": "GOOD",
                "ambient_temperature": 22.5,
                "relative_humidity": 45.0,
                "atmospheric_pressure": 1013.25,
                "vibration_level": 0.01,
                "processing_status": "COMPLETED"
            }
        }
    
    @field_validator('bed_id')
    @classmethod
    def validate_bed_id(cls, v):
        """Validate bed ID format."""
        if not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError("Bed ID must contain only alphanumeric characters, underscores, and hyphens")
        return v
    
    @field_validator('layer_number')
    @classmethod
    def validate_layer_number(cls, v):
        """Validate layer number."""
        if v < 0:
            raise ValueError("Layer number cannot be negative")
        return v
    
    
    @model_validator(mode='after')
    def validate_defect_consistency(self):
        """Validate defect detection consistency."""
        if self.defects_detected is not None and self.defect_count is not None:
            if self.defects_detected != (self.defect_count > 0):
                raise ValueError("defects_detected flag must match defect_count > 0")
        
        return self
    
    def get_primary_key(self) -> str:
        """Get the primary key field name."""
        return "bed_id"
    
    def get_primary_key_value(self) -> Any:
        """Get the primary key value."""
        return self.bed_id
    
    def get_overall_quality_score(self) -> float:
        """
        Calculate overall quality score based on bed quality metrics.
        
        Returns:
            Quality score between 0 and 100
        """
        metrics = self.bed_quality_metrics
        
        # Weighted average of quality metrics
        weights = {
            'uniformity_score': 0.3,
            'coverage_percentage': 0.3,
            'thickness_consistency': 0.2,
            'surface_roughness': 0.1,
            'density_variation': 0.1
        }
        
        score = 0.0
        total_weight = 0.0
        
        # Uniformity score (higher is better)
        score += metrics.uniformity_score * weights['uniformity_score']
        total_weight += weights['uniformity_score']
        
        # Coverage percentage (higher is better)
        score += metrics.coverage_percentage * weights['coverage_percentage']
        total_weight += weights['coverage_percentage']
        
        # Thickness consistency (higher is better)
        score += metrics.thickness_consistency * weights['thickness_consistency']
        total_weight += weights['thickness_consistency']
        
        # Surface roughness (lower is better, invert the scale)
        if metrics.surface_roughness is not None:
            roughness_score = max(0, 100 - metrics.surface_roughness)
            score += roughness_score * weights['surface_roughness']
            total_weight += weights['surface_roughness']
        
        # Density variation (lower is better, invert the scale)
        if metrics.density_variation is not None:
            density_score = max(0, 100 - (metrics.density_variation * 100))
            score += density_score * weights['density_variation']
            total_weight += weights['density_variation']
        
        return score / total_weight if total_weight > 0 else 0.0
    
    def get_quality_grade(self) -> QualityAssessment:
        """
        Get quality grade based on bed quality metrics.
        
        Returns:
            Quality grade
        """
        score = self.get_overall_quality_score()
        
        if score >= 90:
            return QualityAssessment.EXCELLENT
        elif score >= 80:
            return QualityAssessment.GOOD
        elif score >= 70:
            return QualityAssessment.ACCEPTABLE
        elif score >= 50:
            return QualityAssessment.POOR
        else:
            return QualityAssessment.UNACCEPTABLE
    
    def is_high_quality(self) -> bool:
        """
        Check if the powder bed meets high quality standards.
        
        Returns:
            True if quality is excellent or good
        """
        return self.get_quality_grade() in [QualityAssessment.EXCELLENT, QualityAssessment.GOOD]
    
    def get_defect_severity_score(self) -> float:
        """
        Calculate defect severity score.
        
        Returns:
            Severity score between 0 and 100
        """
        if not self.defect_detection or not self.defect_detection.defects_detected:
            return 100.0  # No defects = perfect score
        
        severity_score = 100.0
        
        for defect in self.defect_detection.defect_types:
            severity_penalty = {
                DefectSeverity.LOW: 10,
                DefectSeverity.MEDIUM: 25,
                DefectSeverity.HIGH: 50,
                DefectSeverity.CRITICAL: 80
            }.get(defect.severity, 0)
            
            # Penalty increases with defect area
            area_penalty = min(20, defect.location.area * 2)
            severity_score -= (severity_penalty + area_penalty)
        
        return max(0, severity_score)
    
    def get_critical_defects(self) -> List[PowderBedDefect]:
        """
        Get list of critical defects.
        
        Returns:
            List of critical defects
        """
        if not self.defect_detection:
            return []
        
        return [defect for defect in self.defect_detection.defect_types 
                if defect.severity == DefectSeverity.CRITICAL]
    
    def get_defect_summary(self) -> Dict[str, Any]:
        """
        Get summary of all defects.
        
        Returns:
            Dictionary containing defect summary
        """
        if not self.defect_detection:
            return {
                'defects_detected': False,
                'defect_count': 0,
                'defect_types': [],
                'severity_distribution': {},
                'total_defect_area': 0.0,
                'largest_defect_area': 0.0
            }
        
        detection = self.defect_detection
        severity_distribution = {}
        total_area = 0.0
        largest_area = 0.0
        
        for defect in detection.defect_types:
            severity = defect.severity.value
            severity_distribution[severity] = severity_distribution.get(severity, 0) + 1
            total_area += defect.location.area
            largest_area = max(largest_area, defect.location.area)
        
        return {
            'defects_detected': detection.defects_detected,
            'defect_count': detection.defect_count,
            'defect_types': [defect.type.value for defect in detection.defect_types],
            'severity_distribution': severity_distribution,
            'total_defect_area': total_area,
            'largest_defect_area': largest_area
        }
    
    def get_image_quality_score(self) -> float:
        """
        Calculate image quality score based on image analysis.
        
        Returns:
            Image quality score between 0 and 100
        """
        if not self.image_analysis:
            return 0.0
        
        analysis = self.image_analysis
        
        # Brightness score (optimal range: 100-200)
        brightness_score = 100 - abs(analysis.brightness - 150) / 1.5
        
        # Contrast score (higher is better, normalized)
        contrast_score = min(100, analysis.contrast * 2)
        
        # Sharpness score (higher is better, normalized)
        sharpness_score = min(100, analysis.sharpness * 1.3)
        
        # Noise penalty (lower is better)
        noise_penalty = min(30, analysis.noise_level * 10)
        
        # Color balance score (closer to equal RGB is better)
        color_balance_score = 100 - abs(analysis.color_balance.red_channel - 
                                       analysis.color_balance.green_channel) / 2.55
        
        # Calculate weighted average
        score = (brightness_score * 0.2 + contrast_score * 0.25 + 
                sharpness_score * 0.25 + color_balance_score * 0.2) - noise_penalty
        
        return max(0, min(100, score))
    
    def get_powder_flowability_score(self) -> float:
        """
        Get powder flowability score.
        
        Returns:
            Flowability score between 0 and 100
        """
        if not self.powder_characteristics.flowability:
            return 0.0
        
        return self.powder_characteristics.flowability
    
    def get_moisture_risk_score(self) -> float:
        """
        Calculate moisture risk score.
        
        Returns:
            Risk score between 0 and 100 (higher = more risk)
        """
        if not self.powder_characteristics.moisture_content:
            return 0.0
        
        moisture = self.powder_characteristics.moisture_content
        
        if moisture < 0.1:
            return 0.0  # Very low risk
        elif moisture < 0.5:
            return moisture * 20  # Low risk
        elif moisture < 1.0:
            return 10 + (moisture - 0.5) * 40  # Medium risk
        else:
            return 30 + min(70, (moisture - 1.0) * 70)  # High risk
    
    def get_environmental_risk_score(self) -> float:
        """
        Calculate environmental risk score.
        
        Returns:
            Risk score between 0 and 100 (higher = more risk)
        """
        if not self.environmental_conditions:
            return 0.0
        
        env = self.environmental_conditions
        risk_score = 0.0
        
        # Temperature risk (optimal: 20-25°C)
        temp_deviation = abs(env.temperature - 22.5)
        if temp_deviation > 5:
            risk_score += min(30, temp_deviation * 3)
        
        # Humidity risk (optimal: 30-50%)
        if env.humidity < 30 or env.humidity > 50:
            humidity_risk = min(40, abs(env.humidity - 40) * 2)
            risk_score += humidity_risk
        
        # Vibration risk
        if env.vibration_level and env.vibration_level > 0.1:
            risk_score += min(30, env.vibration_level * 100)
        
        return min(100, risk_score)
    
    def get_recommended_action(self) -> str:
        """
        Get recommended action based on the powder bed data.
        
        Returns:
            Recommended action string
        """
        quality_grade = self.get_quality_grade()
        defect_summary = self.get_defect_summary()
        moisture_risk = self.get_moisture_risk_score()
        environmental_risk = self.get_environmental_risk_score()
        
        if quality_grade == QualityAssessment.UNACCEPTABLE:
            return "REJECT_LAYER"
        elif quality_grade == QualityAssessment.POOR:
            return "INVESTIGATE_ISSUES"
        elif defect_summary['defect_count'] > 10:
            return "ADDRESS_DEFECTS"
        elif moisture_risk > 50:
            return "ADDRESS_MOISTURE"
        elif environmental_risk > 60:
            return "ADJUST_ENVIRONMENT"
        elif quality_grade in [QualityAssessment.EXCELLENT, QualityAssessment.GOOD]:
            return "CONTINUE_PROCESS"
        else:
            return "MONITOR_CLOSELY"
    
    def validate_powder_bed_data(self) -> Dict[str, Any]:
        """
        Validate powder bed data quality and consistency.
        
        Returns:
            Dictionary containing validation results
        """
        validation_results = {
            'valid': True,
            'warnings': [],
            'errors': []
        }
        
        # Check quality metrics
        if self.bed_quality_metrics.uniformity_score < 50:
            validation_results['warnings'].append("Low uniformity score")
        
        if self.bed_quality_metrics.coverage_percentage < 80:
            validation_results['warnings'].append("Low coverage percentage")
        
        if self.bed_quality_metrics.thickness_consistency < 70:
            validation_results['warnings'].append("Low thickness consistency")
        
        # Check powder characteristics
        if self.powder_characteristics.moisture_content and self.powder_characteristics.moisture_content > 1.0:
            validation_results['warnings'].append("High moisture content detected")
        
        if self.powder_characteristics.flowability and self.powder_characteristics.flowability < 50:
            validation_results['warnings'].append("Poor powder flowability")
        
        # Check environmental conditions
        if self.environmental_conditions:
            if self.environmental_conditions.humidity > 60:
                validation_results['warnings'].append("High humidity detected")
            
            if self.environmental_conditions.temperature < 15 or self.environmental_conditions.temperature > 30:
                validation_results['warnings'].append("Temperature outside optimal range")
        
        # Check defect consistency
        if self.defect_detection:
            if self.defect_detection.defects_detected and self.defect_detection.defect_count == 0:
                validation_results['errors'].append("Defects detected but count is zero")
        
        if validation_results['warnings'] or validation_results['errors']:
            validation_results['valid'] = False
        
        return validation_results
    
    def _calculate_consistency(self) -> float:
        """Calculate data consistency score (0-1)."""
        validation = self.validate_powder_bed_data()
        if validation['valid']:
            return 1.0
        else:
            # Reduce score based on number of issues
            issue_penalty = (len(validation['warnings']) * 0.1 + 
                           len(validation['errors']) * 0.2)
            return max(0.0, 1.0 - issue_penalty)
    
    def _calculate_accuracy(self) -> float:
        """Calculate data accuracy score (0-1)."""
        # Base accuracy on quality metrics and image analysis
        quality_score = self.get_overall_quality_score()
        image_score = self.get_image_quality_score()
        
        return (quality_score + image_score) / 200  # Normalize to 0-1
    
    def _calculate_validity(self) -> float:
        """Calculate data validity score (0-1)."""
        # Check if all required fields are present and valid
        required_fields = ['bed_id', 'process_id', 'layer_number', 'timestamp', 
                          'image_metadata', 'powder_characteristics', 'bed_quality_metrics', 'processing_status']
        valid_fields = 0
        
        for field in required_fields:
            value = getattr(self, field)
            if value is not None:
                valid_fields += 1
        
        return valid_fields / len(required_fields)
