"""
CT Scan Model

This module defines the Pydantic model for CT scan data.
"""

from typing import Optional, Dict, Any, List
from datetime import datetime
from pydantic import Field, field_validator, model_validator
from enum import Enum

from .base_model import BaseDataModel

class ScanType(str, Enum):
    """Enumeration of CT scan types."""
    QUALITY_CONTROL = "QUALITY_CONTROL"
    DEFECT_ANALYSIS = "DEFECT_ANALYSIS"
    DIMENSIONAL_MEASUREMENT = "DIMENSIONAL_MEASUREMENT"
    MATERIAL_ANALYSIS = "MATERIAL_ANALYSIS"
    RESEARCH = "RESEARCH"

class FileFormat(str, Enum):
    """Enumeration of file formats."""
    DICOM = "DICOM"
    TIFF = "TIFF"
    RAW = "RAW"
    NIFTI = "NIFTI"
    MHD = "MHD"

class ProcessingStatus(str, Enum):
    """Enumeration of processing statuses."""
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"

class ArtifactSeverity(str, Enum):
    """Enumeration of artifact severity levels."""
    NONE = "NONE"
    MINIMAL = "MINIMAL"
    MODERATE = "MODERATE"
    SEVERE = "SEVERE"

class DefectType(str, Enum):
    """Enumeration of defect types."""
    POROSITY = "POROSITY"
    CRACK = "CRACK"
    INCLUSION = "INCLUSION"
    DELAMINATION = "DELAMINATION"
    WARPAGE = "WARPAGE"
    SHRINKAGE = "SHRINKAGE"
    UNMELTED_POWDER = "UNMELTED_POWDER"

class DefectSeverity(str, Enum):
    """Enumeration of defect severity levels."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class AcceptanceStatus(str, Enum):
    """Enumeration of acceptance statuses."""
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"
    CONDITIONAL = "CONDITIONAL"
    REQUIRES_REVIEW = "REQUIRES_REVIEW"

class ScanParameters(BaseDataModel):
    """CT scan acquisition parameters."""
    
    voltage: float = Field(..., ge=10, le=500, description="X-ray tube voltage in kV")
    current: float = Field(..., ge=0.1, le=1000, description="X-ray tube current in mA")
    exposure_time: float = Field(..., ge=0.001, le=60, description="Exposure time per projection in seconds")
    number_of_projections: int = Field(..., ge=100, le=10000, description="Number of X-ray projections")
    detector_resolution: str = Field(..., description="Detector resolution")
    voxel_size: float = Field(..., ge=0.1, le=1000, description="Voxel size in micrometers")
    scan_duration: float = Field(..., ge=0.1, le=1440, description="Total scan duration in minutes")
    
    def get_primary_key(self) -> str:
        return "id"
    
    def get_primary_key_value(self) -> Any:
        return getattr(self, "id", None)

class FileMetadata(BaseDataModel):
    """CT scan file metadata."""
    
    file_path: str = Field(..., min_length=1, max_length=500, description="Path to the CT scan file")
    file_format: FileFormat = Field(..., description="File format of the CT scan")
    file_size: int = Field(..., ge=0, description="File size in bytes")
    compression: Optional[str] = Field(None, description="Compression type if applicable")
    checksum: Optional[str] = Field(None, description="File checksum for integrity verification")
    
    def get_primary_key(self) -> str:
        return "id"
    
    def get_primary_key_value(self) -> Any:
        return getattr(self, "id", None)

class ImageDimensions(BaseDataModel):
    """CT scan image dimensions."""
    
    width: int = Field(..., ge=1, le=10000, description="Image width in pixels")
    height: int = Field(..., ge=1, le=10000, description="Image height in pixels")
    depth: int = Field(..., ge=1, le=10000, description="Number of slices")
    physical_width: float = Field(..., ge=0.1, le=1000, description="Physical width in mm")
    physical_height: float = Field(..., ge=0.1, le=1000, description="Physical height in mm")
    physical_depth: float = Field(..., ge=0.1, le=1000, description="Physical depth in mm")
    
    def get_primary_key(self) -> str:
        return "id"
    
    def get_primary_key_value(self) -> Any:
        return getattr(self, "id", None)

class CTQualityMetrics(BaseDataModel):
    """CT scan quality metrics."""
    
    contrast_to_noise_ratio: Optional[float] = Field(None, ge=0, description="Contrast-to-noise ratio")
    signal_to_noise_ratio: Optional[float] = Field(None, ge=0, description="Signal-to-noise ratio")
    spatial_resolution: Optional[float] = Field(None, ge=0, description="Spatial resolution in line pairs per mm")
    uniformity: Optional[float] = Field(None, ge=0, le=100, description="Image uniformity percentage")
    artifacts_detected: Optional[bool] = Field(None, description="Whether artifacts were detected")
    artifact_severity: Optional[ArtifactSeverity] = Field(None, description="Severity of detected artifacts")
    
    def get_primary_key(self) -> str:
        return "id"
    
    def get_primary_key_value(self) -> Any:
        return getattr(self, "id", None)

class DefectTypeInfo(BaseDataModel):
    """Information about a specific defect type."""
    
    type: DefectType = Field(..., description="Type of defect")
    count: int = Field(..., ge=0, description="Number of defects of this type")
    average_size: float = Field(..., ge=0, description="Average defect size in mm")
    max_size: float = Field(..., ge=0, description="Maximum defect size in mm")
    severity: DefectSeverity = Field(..., description="Defect severity level")
    
    def get_primary_key(self) -> str:
        return "id"
    
    def get_primary_key_value(self) -> Any:
        return getattr(self, "id", None)

class DefectAnalysis(BaseDataModel):
    """Defect analysis results."""
    
    total_defects: int = Field(..., ge=0, description="Total number of detected defects")
    defect_types: List[DefectTypeInfo] = Field(..., description="Analysis of different defect types")
    overall_quality_score: float = Field(..., ge=0, le=100, description="Overall quality score (0-100)")
    acceptance_status: AcceptanceStatus = Field(..., description="Part acceptance status based on CT analysis")
    
    def get_primary_key(self) -> str:
        return "id"
    
    def get_primary_key_value(self) -> Any:
        return getattr(self, "id", None)

class DimensionalAnalysis(BaseDataModel):
    """Dimensional analysis results."""
    
    measured_dimensions: Dict[str, float] = Field(..., description="Measured dimensions in mm")
    tolerance_deviations: Dict[str, float] = Field(..., description="Tolerance deviations in mm")
    dimensional_accuracy: float = Field(..., ge=0, le=100, description="Overall dimensional accuracy percentage")
    
    def get_primary_key(self) -> str:
        return "id"
    
    def get_primary_key_value(self) -> Any:
        return getattr(self, "id", None)

class CTScanModel(BaseDataModel):
    """
    Pydantic model for CT scan data matching the SQL schema structure.
    
    This model represents CT scan data for PBF-LB/M additive manufacturing quality assessment,
    with flat fields that match the PostgreSQL schema exactly.
    """
    
    # Primary key and identifiers
    scan_id: str = Field(..., min_length=1, max_length=100, description="Unique identifier for the CT scan")
    process_id: str = Field(..., min_length=1, max_length=100, description="Associated PBF process identifier")
    part_id: Optional[str] = Field(None, min_length=1, max_length=100, description="Manufactured part identifier")
    
    # Scan information
    scan_type: ScanType = Field(..., description="Type of CT scan")
    processing_status: ProcessingStatus = Field(..., description="CT scan processing status")
    
    # Scan parameters (flat fields matching SQL schema)
    voltage: float = Field(..., ge=10, le=500, description="X-ray tube voltage in kV")
    current: float = Field(..., ge=0.1, le=1000, description="X-ray tube current in mA")
    exposure_time: float = Field(..., ge=0.001, le=60, description="Exposure time per projection in seconds")
    number_of_projections: int = Field(..., ge=100, le=10000, description="Number of X-ray projections")
    detector_resolution: str = Field(..., description="Detector resolution")
    voxel_size: float = Field(..., ge=0.1, le=1000, description="Voxel size in micrometers")
    scan_duration: float = Field(..., ge=0.1, le=1440, description="Total scan duration in minutes")
    
    # File metadata (flat fields matching SQL schema)
    file_path: str = Field(..., min_length=1, max_length=500, description="Path to the CT scan file")
    file_format: FileFormat = Field(..., description="File format of the CT scan")
    file_size: int = Field(..., ge=0, description="File size in bytes")
    compression: Optional[str] = Field(None, description="Compression type if applicable")
    checksum: Optional[str] = Field(None, description="File checksum for integrity verification")
    
    # Image dimensions (flat fields matching SQL schema)
    image_width: int = Field(..., ge=1, le=10000, description="Image width in pixels")
    image_height: int = Field(..., ge=1, le=10000, description="Image height in pixels")
    image_depth: int = Field(..., ge=1, le=10000, description="Image depth in pixels")
    physical_width: float = Field(..., ge=0.1, le=1000, description="Physical width in mm")
    physical_height: float = Field(..., ge=0.1, le=1000, description="Physical height in mm")
    physical_depth: float = Field(..., ge=0.1, le=1000, description="Physical depth in mm")
    
    # Quality metrics (flat fields matching SQL schema)
    contrast_to_noise_ratio: Optional[float] = Field(None, description="Contrast to noise ratio")
    signal_to_noise_ratio: Optional[float] = Field(None, description="Signal to noise ratio")
    spatial_resolution: Optional[float] = Field(None, description="Spatial resolution in mm")
    uniformity: Optional[float] = Field(None, ge=0, le=100, description="Image uniformity percentage")
    artifacts_detected: Optional[bool] = Field(None, description="Whether artifacts were detected")
    artifact_severity: Optional[ArtifactSeverity] = Field(None, description="Severity of detected artifacts")
    
    # Defect analysis (flat fields matching SQL schema)
    total_defects: Optional[int] = Field(None, ge=0, description="Total number of detected defects")
    overall_quality_score: Optional[float] = Field(None, ge=0, le=100, description="Overall quality score (0-100)")
    acceptance_status: Optional[AcceptanceStatus] = Field(None, description="Part acceptance status")
    
    # Dimensional analysis (flat fields matching SQL schema)
    dimensional_accuracy: Optional[float] = Field(None, ge=0, le=100, description="Dimensional accuracy percentage")
    
    # Processing metadata (JSONB fields)
    processing_metadata: Optional[Dict[str, Any]] = Field(None, description="Processing metadata and parameters")
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True
        validate_assignment = True
        schema_extra = {
            "example": {
                "scan_id": "CT_2024_001",
                "process_id": "PBF_2024_001",
                "part_id": "PART_001",
                "scan_type": "QUALITY_CONTROL",
                "processing_status": "COMPLETED",
                "scan_parameters": {
                    "voltage": 120.0,
                    "current": 100.0,
                    "exposure_time": 0.5,
                    "number_of_projections": 1000,
                    "detector_resolution": "2048x2048",
                    "voxel_size": 50.0,
                    "scan_duration": 30.0
                },
                "file_metadata": {
                    "file_path": "/data/ct_scans/CT_2024_001.dicom",
                    "file_format": "DICOM",
                    "file_size": 1073741824,
                    "compression": "GZIP",
                    "checksum": "abc123def456"
                },
                "image_dimensions": {
                    "width": 2048,
                    "height": 2048,
                    "depth": 1000,
                    "physical_width": 100.0,
                    "physical_height": 100.0,
                    "physical_depth": 50.0
                },
                "quality_metrics": {
                    "contrast_to_noise_ratio": 15.5,
                    "signal_to_noise_ratio": 25.0,
                    "spatial_resolution": 10.0,
                    "uniformity": 95.0,
                    "artifacts_detected": False,
                    "artifact_severity": "NONE"
                },
                "defect_analysis": {
                    "total_defects": 5,
                    "defect_types": [
                        {
                            "type": "POROSITY",
                            "count": 3,
                            "average_size": 0.1,
                            "max_size": 0.2,
                            "severity": "LOW"
                        },
                        {
                            "type": "CRACK",
                            "count": 2,
                            "average_size": 0.05,
                            "max_size": 0.08,
                            "severity": "MEDIUM"
                        }
                    ],
                    "overall_quality_score": 85.0,
                    "acceptance_status": "ACCEPTED"
                },
                "dimensional_analysis": {
                    "measured_dimensions": {
                        "length": 99.8,
                        "width": 100.1,
                        "height": 49.9
                    },
                    "tolerance_deviations": {
                        "length": -0.2,
                        "width": 0.1,
                        "height": -0.1
                    },
                    "dimensional_accuracy": 98.5
                }
            }
        }
    
    @field_validator('scan_id')
    @classmethod
    def validate_scan_id(cls, v):
        """Validate scan ID format."""
        if not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError("Scan ID must contain only alphanumeric characters, underscores, and hyphens")
        return v
    
    @field_validator('part_id')
    @classmethod
    def validate_part_id(cls, v):
        """Validate part ID format."""
        if v is not None and not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError("Part ID must contain only alphanumeric characters, underscores, and hyphens")
        return v
    
    
    
    @model_validator(mode='after')
    def validate_processing_status(self):
        """Validate processing status consistency."""
        if self.processing_status == ProcessingStatus.COMPLETED:
            if not self.total_defects and not self.overall_quality_score:
                # Completed scans should have some analysis results
                pass  # Could add warning logic here
        
        return self
    
    def get_primary_key(self) -> str:
        """Get the primary key field name."""
        return "scan_id"
    
    def get_primary_key_value(self) -> Any:
        """Get the primary key value."""
        return self.scan_id
    
    def get_scan_quality_score(self) -> float:
        """
        Calculate overall scan quality score.
        
        Returns:
            Quality score between 0 and 100
        """
        if not self.quality_metrics:
            return 0.0
        
        metrics = self.quality_metrics
        score = 0.0
        factors = 0
        
        # Contrast-to-noise ratio (target: >10)
        if metrics.contrast_to_noise_ratio is not None:
            cnr_score = min(100, (metrics.contrast_to_noise_ratio / 10) * 100)
            score += cnr_score
            factors += 1
        
        # Signal-to-noise ratio (target: >20)
        if metrics.signal_to_noise_ratio is not None:
            snr_score = min(100, (metrics.signal_to_noise_ratio / 20) * 100)
            score += snr_score
            factors += 1
        
        # Spatial resolution (target: >5 lp/mm)
        if metrics.spatial_resolution is not None:
            resolution_score = min(100, (metrics.spatial_resolution / 5) * 100)
            score += resolution_score
            factors += 1
        
        # Uniformity (target: >90%)
        if metrics.uniformity is not None:
            score += metrics.uniformity
            factors += 1
        
        # Artifact penalty
        if metrics.artifacts_detected:
            artifact_penalty = {
                ArtifactSeverity.NONE: 0,
                ArtifactSeverity.MINIMAL: 5,
                ArtifactSeverity.MODERATE: 15,
                ArtifactSeverity.SEVERE: 30
            }.get(metrics.artifact_severity, 0)
            score -= artifact_penalty
        
        return max(0, score / factors) if factors > 0 else 0.0
    
    def get_defect_severity_score(self) -> float:
        """
        Calculate defect severity score.
        
        Returns:
            Severity score between 0 and 100
        """
        if not self.defect_analysis:
            return 0.0
        
        analysis = self.defect_analysis
        severity_score = 100.0
        
        # Reduce score based on defect types and counts
        for defect_type in analysis.defect_types:
            severity_penalty = {
                DefectSeverity.LOW: 5,
                DefectSeverity.MEDIUM: 15,
                DefectSeverity.HIGH: 30,
                DefectSeverity.CRITICAL: 50
            }.get(defect_type.severity, 0)
            
            # Penalty increases with defect count
            count_penalty = min(20, defect_type.count * 2)
            severity_score -= (severity_penalty + count_penalty)
        
        return max(0, severity_score)
    
    def get_dimensional_accuracy_score(self) -> float:
        """
        Calculate dimensional accuracy score.
        
        Returns:
            Accuracy score between 0 and 100
        """
        if not self.dimensional_analysis:
            return 0.0
        
        return self.dimensional_analysis.dimensional_accuracy
    
    def get_overall_quality_assessment(self) -> str:
        """
        Get overall quality assessment.
        
        Returns:
            Quality assessment string
        """
        scan_quality = self.get_scan_quality_score()
        defect_severity = self.get_defect_severity_score()
        dimensional_accuracy = self.get_dimensional_accuracy_score()
        
        # Calculate weighted average
        overall_score = (scan_quality * 0.4 + defect_severity * 0.4 + dimensional_accuracy * 0.2)
        
        if overall_score >= 90:
            return "EXCELLENT"
        elif overall_score >= 80:
            return "GOOD"
        elif overall_score >= 70:
            return "ACCEPTABLE"
        else:
            return "POOR"
    
    def is_acceptable_quality(self) -> bool:
        """
        Check if the scan meets acceptable quality standards.
        
        Returns:
            True if quality is acceptable or better
        """
        return self.get_overall_quality_assessment() in ["EXCELLENT", "GOOD", "ACCEPTABLE"]
    
    def get_critical_defects(self) -> List[DefectTypeInfo]:
        """
        Get list of critical defects.
        
        Returns:
            List of critical defect type information
        """
        if not self.defect_analysis:
            return []
        
        return [defect for defect in self.defect_analysis.defect_types 
                if defect.severity == DefectSeverity.CRITICAL]
    
    def get_defect_summary(self) -> Dict[str, Any]:
        """
        Get summary of all defects.
        
        Returns:
            Dictionary containing defect summary
        """
        if not self.defect_analysis:
            return {
                'total_defects': 0,
                'defect_types': [],
                'severity_distribution': {},
                'largest_defect_size': 0.0
            }
        
        analysis = self.defect_analysis
        severity_distribution = {}
        largest_size = 0.0
        
        for defect_type in analysis.defect_types:
            severity = defect_type.severity.value
            severity_distribution[severity] = severity_distribution.get(severity, 0) + defect_type.count
            largest_size = max(largest_size, defect_type.max_size)
        
        return {
            'total_defects': analysis.total_defects,
            'defect_types': [defect.type.value for defect in analysis.defect_types],
            'severity_distribution': severity_distribution,
            'largest_defect_size': largest_size
        }
    
    def get_scan_efficiency(self) -> float:
        """
        Calculate scan efficiency based on parameters.
        
        Returns:
            Efficiency score between 0 and 1
        """
        params = self.scan_parameters
        
        # Efficiency based on scan duration and resolution
        expected_duration = (params.number_of_projections * params.exposure_time) / 60  # minutes
        actual_duration = params.scan_duration
        
        if actual_duration > 0:
            efficiency = min(1.0, expected_duration / actual_duration)
        else:
            efficiency = 0.0
        
        return efficiency
    
    def get_file_size_mb(self) -> float:
        """
        Get file size in megabytes.
        
        Returns:
            File size in MB
        """
        return self.file_size / (1024 * 1024)
    
    def get_voxel_volume_mm3(self) -> float:
        """
        Get voxel volume in cubic millimeters.
        
        Returns:
            Voxel volume in mm³
        """
        voxel_size_mm = self.scan_parameters.voxel_size / 1000  # Convert micrometers to mm
        return voxel_size_mm ** 3
    
    def get_total_volume_mm3(self) -> float:
        """
        Get total scanned volume in cubic millimeters.
        
        Returns:
            Total volume in mm³
        """
        dims = self.image_dimensions
        return dims.physical_width * dims.physical_height * dims.physical_depth
    
    def validate_scan_data(self) -> Dict[str, Any]:
        """
        Validate CT scan data quality and consistency.
        
        Returns:
            Dictionary containing validation results
        """
        validation_results = {
            'valid': True,
            'warnings': [],
            'errors': []
        }
        
        # Check scan parameters
        if self.voltage < 50 or self.voltage > 300:
            validation_results['warnings'].append(
                f"Voltage {self.voltage} kV is outside typical range (50-300 kV)"
            )
        
        if self.current < 10 or self.current > 500:
            validation_results['warnings'].append(
                f"Current {self.current} mA is outside typical range (10-500 mA)"
            )
        
        # Check file size
        if self.file_size < 1024:  # Less than 1KB
            validation_results['errors'].append("File size is too small")
        
        # Check image dimensions
        if (self.image_width * self.image_height * self.image_depth) == 0:
            validation_results['errors'].append("Image dimensions cannot be zero")
        
        # Check quality metrics
        if self.contrast_to_noise_ratio and self.contrast_to_noise_ratio < 5:
            validation_results['warnings'].append("Low contrast-to-noise ratio")
            
        if self.signal_to_noise_ratio and self.signal_to_noise_ratio < 10:
            validation_results['warnings'].append("Low signal-to-noise ratio")
        
        if validation_results['warnings'] or validation_results['errors']:
            validation_results['valid'] = False
        
        return validation_results
    
    def _calculate_consistency(self) -> float:
        """Calculate data consistency score (0-1)."""
        validation = self.validate_scan_data()
        if validation['valid']:
            return 1.0
        else:
            # Reduce score based on number of issues
            issue_penalty = (len(validation['warnings']) * 0.1 + 
                           len(validation['errors']) * 0.2)
            return max(0.0, 1.0 - issue_penalty)
    
    def _calculate_accuracy(self) -> float:
        """Calculate data accuracy score (0-1)."""
        # Base accuracy on quality metrics and defect analysis
        if not self.quality_metrics and not self.defect_analysis:
            return 0.5  # Partial score if no analysis available
        
        quality_score = self.get_scan_quality_score()
        return quality_score / 100
    
    def _calculate_validity(self) -> float:
        """Calculate data validity score (0-1)."""
        # Check if all required fields are present and valid
        required_fields = ['scan_id', 'process_id', 'scan_type', 'processing_status', 
                          'voltage', 'current', 'file_path', 'file_size', 'image_width', 'image_height', 'image_depth']
        valid_fields = 0
        
        for field in required_fields:
            value = getattr(self, field)
            if value is not None:
                valid_fields += 1
        
        return valid_fields / len(required_fields)
