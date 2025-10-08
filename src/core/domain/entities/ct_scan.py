"""
CT scan domain entity.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass, field

from .base_entity import BaseEntity
from ..enums import QualityTier, DefectType
from ..value_objects import QualityMetrics, DefectClassification


@dataclass
class CTScan(BaseEntity):
    """
    Domain entity representing a CT (Computed Tomography) scan.
    
    This entity encapsulates all information about CT scan operations,
    including scan parameters, image data, analysis results, and quality assessments.
    """
    
    # Scan identification
    scan_name: str = field(default="")
    scan_session_id: Optional[str] = None
    
    # Process information
    process_id: Optional[str] = None
    build_id: Optional[str] = None
    part_id: Optional[str] = None
    
    # Scan configuration
    scan_parameters: Optional[Dict[str, Any]] = None
    scan_resolution: Optional[float] = None  # mm/voxel
    scan_volume: Optional[Dict[str, float]] = None  # x, y, z dimensions
    
    # Scan state
    scan_status: str = "initialized"  # initialized, running, completed, failed
    scan_quality: Optional[str] = None  # excellent, good, fair, poor
    
    # Timing information
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    analysis_start_time: Optional[datetime] = None
    analysis_end_time: Optional[datetime] = None
    
    # Scan results
    image_count: int = 0
    image_resolution: Optional[Dict[str, int]] = None  # width, height
    file_size: Optional[float] = None  # MB
    scan_duration: Optional[float] = None  # seconds
    
    # Analysis results
    quality_metrics: Optional[QualityMetrics] = None
    defect_detections: Optional[List[DefectClassification]] = None
    analysis_results: Optional[Dict[str, Any]] = None
    
    # Equipment information
    scanner_id: Optional[str] = None
    operator_id: Optional[str] = None
    
    # Data storage
    image_path: Optional[str] = None
    analysis_path: Optional[str] = None
    report_path: Optional[str] = None
    
    def __post_init__(self):
        """Initialize default values."""
        if self.defect_detections is None:
            object.__setattr__(self, 'defect_detections', [])
        if self.analysis_results is None:
            object.__setattr__(self, 'analysis_results', {})
        super().__post_init__()
    
    def validate(self) -> None:
        """Validate CT scan entity."""
        if not self.scan_name:
            raise ValueError("Scan name cannot be empty")
        
        if self.start_time and self.end_time and self.start_time > self.end_time:
            raise ValueError("Start time cannot be after end time")
        
        if self.analysis_start_time and self.analysis_end_time and self.analysis_start_time > self.analysis_end_time:
            raise ValueError("Analysis start time cannot be after analysis end time")
        
        if self.scan_resolution is not None and self.scan_resolution <= 0:
            raise ValueError("Scan resolution must be positive")
        
        if self.image_count < 0:
            raise ValueError("Image count cannot be negative")
        
        if self.file_size is not None and self.file_size < 0:
            raise ValueError("File size cannot be negative")
        
        if self.scan_duration is not None and self.scan_duration < 0:
            raise ValueError("Scan duration cannot be negative")
    
    def start_scan(self, start_time: Optional[datetime] = None) -> 'CTScan':
        """Start the CT scan."""
        if start_time is None:
            start_time = datetime.utcnow()
        
        return self.update(
            scan_status="running",
            start_time=start_time,
            updated_by="CTScannerService"
        )
    
    def complete_scan(self, end_time: Optional[datetime] = None, scan_results: Optional[Dict[str, Any]] = None) -> 'CTScan':
        """Complete the CT scan."""
        if end_time is None:
            end_time = datetime.utcnow()
        
        update_data = {
            "scan_status": "completed",
            "end_time": end_time,
            "updated_by": "CTScannerService"
        }
        
        if scan_results:
            update_data.update({
                "image_count": scan_results.get("image_count", self.image_count),
                "image_resolution": scan_results.get("image_resolution", self.image_resolution),
                "file_size": scan_results.get("file_size", self.file_size),
                "scan_duration": scan_results.get("scan_duration", self.scan_duration),
                "scan_quality": scan_results.get("scan_quality", self.scan_quality)
            })
        
        return self.update(**update_data)
    
    def fail_scan(self, failure_reason: str, end_time: Optional[datetime] = None) -> 'CTScan':
        """Mark the CT scan as failed."""
        if end_time is None:
            end_time = datetime.utcnow()
        
        return self.update(
            scan_status="failed",
            end_time=end_time,
            data={**self.data, "failure_reason": failure_reason},
            updated_by="CTScannerService"
        )
    
    def start_analysis(self, analysis_start_time: Optional[datetime] = None) -> 'CTScan':
        """Start CT scan analysis."""
        if analysis_start_time is None:
            analysis_start_time = datetime.utcnow()
        
        return self.update(
            analysis_start_time=analysis_start_time,
            updated_by="CTAnalysisService"
        )
    
    def complete_analysis(self, analysis_end_time: Optional[datetime] = None, analysis_results: Optional[Dict[str, Any]] = None) -> 'CTScan':
        """Complete CT scan analysis."""
        if analysis_end_time is None:
            analysis_end_time = datetime.utcnow()
        
        update_data = {
            "analysis_end_time": analysis_end_time,
            "updated_by": "CTAnalysisService"
        }
        
        if analysis_results:
            update_data["analysis_results"] = analysis_results
        
        return self.update(**update_data)
    
    def add_defect_detection(self, defect: DefectClassification) -> 'CTScan':
        """Add a defect detection to the scan."""
        new_defect_detections = self.defect_detections + [defect]
        
        return self.update(
            defect_detections=new_defect_detections,
            updated_by="CTAnalysisService"
        )
    
    def update_quality_metrics(self, quality_metrics: QualityMetrics) -> 'CTScan':
        """Update quality metrics for the scan."""
        return self.update(
            quality_metrics=quality_metrics,
            updated_by="CTAnalysisService"
        )
    
    def get_scan_duration(self) -> Optional[float]:
        """Get scan duration in seconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None
    
    def get_analysis_duration(self) -> Optional[float]:
        """Get analysis duration in seconds."""
        if self.analysis_start_time and self.analysis_end_time:
            return (self.analysis_end_time - self.analysis_start_time).total_seconds()
        return None
    
    def get_total_duration(self) -> Optional[float]:
        """Get total duration (scan + analysis) in seconds."""
        scan_duration = self.get_scan_duration()
        analysis_duration = self.get_analysis_duration()
        
        if scan_duration and analysis_duration:
            return scan_duration + analysis_duration
        return None
    
    def get_defect_count_by_type(self) -> Dict[str, int]:
        """Get count of defects by type."""
        defect_counts = {}
        for defect in self.defect_detections:
            defect_type = defect.defect_type.value
            defect_counts[defect_type] = defect_counts.get(defect_type, 0) + 1
        return defect_counts
    
    def get_defect_count_by_severity(self) -> Dict[str, int]:
        """Get count of defects by severity."""
        severity_counts = {}
        for defect in self.defect_detections:
            severity = defect.severity.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        return severity_counts
    
    def get_critical_defect_count(self) -> int:
        """Get count of critical defects."""
        return sum(1 for defect in self.defect_detections if defect.is_critical())
    
    def get_acceptable_defect_count(self) -> int:
        """Get count of acceptable defects."""
        return sum(1 for defect in self.defect_detections if defect.is_acceptable())
    
    def is_scan_completed(self) -> bool:
        """Check if scan is completed."""
        return self.scan_status == "completed"
    
    def is_scan_failed(self) -> bool:
        """Check if scan has failed."""
        return self.scan_status == "failed"
    
    def is_analysis_completed(self) -> bool:
        """Check if analysis is completed."""
        return self.analysis_end_time is not None
    
    def has_critical_defects(self) -> bool:
        """Check if scan has critical defects."""
        return self.get_critical_defect_count() > 0
    
    def get_scan_summary(self) -> Dict[str, Any]:
        """Get comprehensive scan summary."""
        return {
            "scan_id": self.id,
            "scan_name": self.scan_name,
            "scan_session_id": self.scan_session_id,
            "process_id": self.process_id,
            "build_id": self.build_id,
            "part_id": self.part_id,
            "scan_status": self.scan_status,
            "scan_quality": self.scan_quality,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "analysis_start_time": self.analysis_start_time.isoformat() if self.analysis_start_time else None,
            "analysis_end_time": self.analysis_end_time.isoformat() if self.analysis_end_time else None,
            "scan_duration": self.get_scan_duration(),
            "analysis_duration": self.get_analysis_duration(),
            "total_duration": self.get_total_duration(),
            "image_count": self.image_count,
            "image_resolution": self.image_resolution,
            "file_size": self.file_size,
            "scan_resolution": self.scan_resolution,
            "scan_volume": self.scan_volume,
            "scanner_id": self.scanner_id,
            "operator_id": self.operator_id,
            "image_path": self.image_path,
            "analysis_path": self.analysis_path,
            "report_path": self.report_path,
            "is_scan_completed": self.is_scan_completed(),
            "is_scan_failed": self.is_scan_failed(),
            "is_analysis_completed": self.is_analysis_completed(),
            "quality_score": self.quality_metrics.get_overall_quality_score() if self.quality_metrics else None,
            "quality_tier": self.quality_metrics.get_quality_tier().value if self.quality_metrics else None,
            "total_defects": len(self.defect_detections),
            "critical_defects": self.get_critical_defect_count(),
            "acceptable_defects": self.get_acceptable_defect_count(),
            "has_critical_defects": self.has_critical_defects()
        }
    
    def get_scan_analytics(self) -> Dict[str, Any]:
        """Get scan analytics and insights."""
        analytics = {
            "performance_metrics": {
                "scan_duration": self.get_scan_duration(),
                "analysis_duration": self.get_analysis_duration(),
                "total_duration": self.get_total_duration(),
                "image_count": self.image_count,
                "file_size": self.file_size,
                "scan_resolution": self.scan_resolution
            },
            "quality_metrics": {
                "overall_score": self.quality_metrics.get_overall_quality_score() if self.quality_metrics else None,
                "quality_tier": self.quality_metrics.get_quality_tier().value if self.quality_metrics else None,
                "is_acceptable": self.quality_metrics.is_acceptable_quality() if self.quality_metrics else None,
                "is_production_ready": self.quality_metrics.is_production_ready() if self.quality_metrics else None
            },
            "defect_analysis": {
                "total_defects": len(self.defect_detections),
                "critical_defects": self.get_critical_defect_count(),
                "acceptable_defects": self.get_acceptable_defect_count(),
                "defects_by_type": self.get_defect_count_by_type(),
                "defects_by_severity": self.get_defect_count_by_severity(),
                "has_critical_defects": self.has_critical_defects()
            },
            "scan_quality": {
                "scan_quality": self.scan_quality,
                "image_resolution": self.image_resolution,
                "scan_resolution": self.scan_resolution,
                "scan_volume": self.scan_volume
            },
            "recommendations": []
        }
        
        # Add recommendations based on scan results
        if self.is_scan_failed():
            analytics["recommendations"].append("Investigate scan failure and retry scan")
        
        if self.has_critical_defects():
            analytics["recommendations"].append("Address critical defects immediately")
        
        if self.quality_metrics and not self.quality_metrics.is_acceptable_quality():
            analytics["recommendations"].extend(self.quality_metrics.get_improvement_recommendations())
        
        if self.scan_quality == "poor":
            analytics["recommendations"].append("Improve scan quality through parameter optimization")
        
        if not self.is_analysis_completed() and self.is_scan_completed():
            analytics["recommendations"].append("Complete scan analysis")
        
        return analytics