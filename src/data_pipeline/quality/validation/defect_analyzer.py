"""
Defect Analyzer for PBF-LB/M Data Pipeline

This module provides defect analysis capabilities for identifying
and classifying defects in PBF-LB/M manufacturing data, particularly
for CT scan and powder bed data.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

from .data_quality_service import QualityResult, QualityRule

logger = logging.getLogger(__name__)


class DefectType(Enum):
    """Types of defects that can be detected."""
    POROSITY = "porosity"
    CRACK = "crack"
    DELAMINATION = "delamination"
    INCLUSION = "inclusion"
    SURFACE_ROUGHNESS = "surface_roughness"
    DIMENSIONAL_DEVIATION = "dimensional_deviation"
    UNKNOWN = "unknown"


class DefectSeverity(Enum):
    """Severity levels for defects."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class DefectConfig:
    """Configuration for defect analysis."""
    porosity_threshold: float = 0.05  # 5% porosity threshold
    crack_length_threshold: float = 0.1  # 0.1mm crack length threshold
    surface_roughness_threshold: float = 10.0  # 10μm surface roughness threshold
    dimensional_tolerance: float = 0.1  # 0.1mm dimensional tolerance
    enable_porosity_analysis: bool = True
    enable_crack_detection: bool = True
    enable_surface_analysis: bool = True
    enable_dimensional_analysis: bool = True
    min_defect_size: float = 0.01  # Minimum defect size in mm
    max_defect_density: float = 0.1  # Maximum defect density per unit volume


@dataclass
class Defect:
    """Individual defect information."""
    defect_id: str
    defect_type: DefectType
    severity: DefectSeverity
    location: Tuple[float, float, float]  # x, y, z coordinates
    size: float  # Defect size in mm
    confidence: float  # Detection confidence (0-1)
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class DefectAnalysisResult:
    """Result of defect analysis."""
    total_records: int
    records_with_defects: int
    total_defects: int
    defects_by_type: Dict[str, int]
    defects_by_severity: Dict[str, int]
    average_defect_density: float
    quality_results: List[QualityResult]
    processing_time_seconds: float
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class DefectAnalyzer:
    """
    Defect analyzer for PBF-LB/M manufacturing data.
    
    This analyzer identifies and classifies various types of defects
    in PBF-LB/M parts, particularly from CT scan and powder bed data.
    """
    
    def __init__(self, config: Optional[DefectConfig] = None):
        """
        Initialize the defect analyzer.
        
        Args:
            config: Defect analysis configuration
        """
        self.config = config or DefectConfig()
        
        logger.info("Defect Analyzer initialized")
    
    def analyze_defects(self, data: List[Dict[str, Any]], 
                       data_type: str = 'generic') -> DefectAnalysisResult:
        """
        Analyze defects in the provided data.
        
        Args:
            data: List of data records
            data_type: Type of data (ct_scan, powder_bed, etc.)
            
        Returns:
            DefectAnalysisResult: Defect analysis result
        """
        start_time = datetime.now()
        
        try:
            logger.info(f"Starting defect analysis for {len(data)} records of type {data_type}")
            
            # Convert data to DataFrame for easier processing
            df = pd.DataFrame(data)
            
            if df.empty:
                return DefectAnalysisResult(
                    total_records=0,
                    records_with_defects=0,
                    total_defects=0,
                    defects_by_type={},
                    defects_by_severity={},
                    average_defect_density=0.0,
                    quality_results=[],
                    processing_time_seconds=0.0
                )
            
            # Analyze defects based on data type
            if data_type == 'ct_scan':
                defects = self._analyze_ct_scan_defects(df)
            elif data_type == 'powder_bed':
                defects = self._analyze_powder_bed_defects(df)
            else:
                defects = self._analyze_generic_defects(df)
            
            # Calculate statistics
            records_with_defects = len(set(defect.metadata.get('record_id', 0) for defect in defects))
            total_defects = len(defects)
            
            defects_by_type = {}
            defects_by_severity = {}
            
            for defect in defects:
                # Count by type
                defect_type = defect.defect_type.value
                defects_by_type[defect_type] = defects_by_type.get(defect_type, 0) + 1
                
                # Count by severity
                severity = defect.severity.value
                defects_by_severity[severity] = defects_by_severity.get(severity, 0) + 1
            
            # Calculate average defect density
            total_volume = self._calculate_total_volume(df, data_type)
            average_defect_density = total_defects / total_volume if total_volume > 0 else 0.0
            
            # Create quality results
            quality_results = self._create_quality_results(data, defects, data_type)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Create result
            result = DefectAnalysisResult(
                total_records=len(data),
                records_with_defects=records_with_defects,
                total_defects=total_defects,
                defects_by_type=defects_by_type,
                defects_by_severity=defects_by_severity,
                average_defect_density=average_defect_density,
                quality_results=quality_results,
                processing_time_seconds=processing_time,
                metadata={
                    'data_type': data_type,
                    'analysis_config': self.config.__dict__,
                    'analysis_timestamp': datetime.now().isoformat()
                }
            )
            
            logger.info(f"Defect analysis completed: {total_defects} defects found in "
                       f"{records_with_defects}/{len(data)} records")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in defect analysis: {e}")
            return DefectAnalysisResult(
                total_records=len(data),
                records_with_defects=0,
                total_defects=0,
                defects_by_type={},
                defects_by_severity={},
                average_defect_density=0.0,
                quality_results=[],
                processing_time_seconds=(datetime.now() - start_time).total_seconds(),
                metadata={'error': str(e)}
            )
    
    def _analyze_ct_scan_defects(self, df: pd.DataFrame) -> List[Defect]:
        """Analyze defects in CT scan data."""
        defects = []
        
        try:
            for idx, row in df.iterrows():
                record_defects = []
                
                # Porosity analysis
                if self.config.enable_porosity_analysis:
                    porosity_defects = self._detect_porosity_defects(row, idx)
                    record_defects.extend(porosity_defects)
                
                # Crack detection
                if self.config.enable_crack_detection:
                    crack_defects = self._detect_crack_defects(row, idx)
                    record_defects.extend(crack_defects)
                
                # Inclusion detection
                inclusion_defects = self._detect_inclusion_defects(row, idx)
                record_defects.extend(inclusion_defects)
                
                defects.extend(record_defects)
            
        except Exception as e:
            logger.error(f"Error analyzing CT scan defects: {e}")
        
        return defects
    
    def _analyze_powder_bed_defects(self, df: pd.DataFrame) -> List[Defect]:
        """Analyze defects in powder bed data."""
        defects = []
        
        try:
            for idx, row in df.iterrows():
                record_defects = []
                
                # Surface roughness analysis
                if self.config.enable_surface_analysis:
                    surface_defects = self._detect_surface_roughness_defects(row, idx)
                    record_defects.extend(surface_defects)
                
                # Dimensional analysis
                if self.config.enable_dimensional_analysis:
                    dimensional_defects = self._detect_dimensional_defects(row, idx)
                    record_defects.extend(dimensional_defects)
                
                defects.extend(record_defects)
            
        except Exception as e:
            logger.error(f"Error analyzing powder bed defects: {e}")
        
        return defects
    
    def _analyze_generic_defects(self, df: pd.DataFrame) -> List[Defect]:
        """Analyze defects in generic data."""
        defects = []
        
        try:
            for idx, row in df.iterrows():
                # Look for common defect indicators
                if 'porosity' in row and row['porosity'] > self.config.porosity_threshold:
                    defect = Defect(
                        defect_id=f"porosity_{idx}",
                        defect_type=DefectType.POROSITY,
                        severity=self._determine_porosity_severity(row['porosity']),
                        location=(0.0, 0.0, 0.0),  # Default location
                        size=row.get('porosity', 0.0),
                        confidence=0.8,
                        metadata={'record_id': idx, 'source': 'generic_analysis'}
                    )
                    defects.append(defect)
                
                if 'surface_roughness' in row and row['surface_roughness'] > self.config.surface_roughness_threshold:
                    defect = Defect(
                        defect_id=f"surface_roughness_{idx}",
                        defect_type=DefectType.SURFACE_ROUGHNESS,
                        severity=self._determine_surface_roughness_severity(row['surface_roughness']),
                        location=(0.0, 0.0, 0.0),  # Default location
                        size=row['surface_roughness'],
                        confidence=0.7,
                        metadata={'record_id': idx, 'source': 'generic_analysis'}
                    )
                    defects.append(defect)
            
        except Exception as e:
            logger.error(f"Error analyzing generic defects: {e}")
        
        return defects
    
    def _detect_porosity_defects(self, row: pd.Series, record_id: int) -> List[Defect]:
        """Detect porosity defects."""
        defects = []
        
        try:
            porosity_value = row.get('porosity', 0.0)
            if porosity_value > self.config.porosity_threshold:
                defect = Defect(
                    defect_id=f"porosity_{record_id}",
                    defect_type=DefectType.POROSITY,
                    severity=self._determine_porosity_severity(porosity_value),
                    location=(
                        row.get('x_coordinate', 0.0),
                        row.get('y_coordinate', 0.0),
                        row.get('z_coordinate', 0.0)
                    ),
                    size=porosity_value,
                    confidence=0.9,
                    metadata={
                        'record_id': record_id,
                        'porosity_value': porosity_value,
                        'threshold': self.config.porosity_threshold
                    }
                )
                defects.append(defect)
        
        except Exception as e:
            logger.error(f"Error detecting porosity defects: {e}")
        
        return defects
    
    def _detect_crack_defects(self, row: pd.Series, record_id: int) -> List[Defect]:
        """Detect crack defects."""
        defects = []
        
        try:
            crack_length = row.get('crack_length', 0.0)
            if crack_length > self.config.crack_length_threshold:
                defect = Defect(
                    defect_id=f"crack_{record_id}",
                    defect_type=DefectType.CRACK,
                    severity=self._determine_crack_severity(crack_length),
                    location=(
                        row.get('x_coordinate', 0.0),
                        row.get('y_coordinate', 0.0),
                        row.get('z_coordinate', 0.0)
                    ),
                    size=crack_length,
                    confidence=0.85,
                    metadata={
                        'record_id': record_id,
                        'crack_length': crack_length,
                        'threshold': self.config.crack_length_threshold
                    }
                )
                defects.append(defect)
        
        except Exception as e:
            logger.error(f"Error detecting crack defects: {e}")
        
        return defects
    
    def _detect_inclusion_defects(self, row: pd.Series, record_id: int) -> List[Defect]:
        """Detect inclusion defects."""
        defects = []
        
        try:
            inclusion_size = row.get('inclusion_size', 0.0)
            if inclusion_size > self.config.min_defect_size:
                defect = Defect(
                    defect_id=f"inclusion_{record_id}",
                    defect_type=DefectType.INCLUSION,
                    severity=self._determine_inclusion_severity(inclusion_size),
                    location=(
                        row.get('x_coordinate', 0.0),
                        row.get('y_coordinate', 0.0),
                        row.get('z_coordinate', 0.0)
                    ),
                    size=inclusion_size,
                    confidence=0.8,
                    metadata={
                        'record_id': record_id,
                        'inclusion_size': inclusion_size,
                        'min_size_threshold': self.config.min_defect_size
                    }
                )
                defects.append(defect)
        
        except Exception as e:
            logger.error(f"Error detecting inclusion defects: {e}")
        
        return defects
    
    def _detect_surface_roughness_defects(self, row: pd.Series, record_id: int) -> List[Defect]:
        """Detect surface roughness defects."""
        defects = []
        
        try:
            surface_roughness = row.get('surface_roughness', 0.0)
            if surface_roughness > self.config.surface_roughness_threshold:
                defect = Defect(
                    defect_id=f"surface_roughness_{record_id}",
                    defect_type=DefectType.SURFACE_ROUGHNESS,
                    severity=self._determine_surface_roughness_severity(surface_roughness),
                    location=(
                        row.get('x_coordinate', 0.0),
                        row.get('y_coordinate', 0.0),
                        row.get('z_coordinate', 0.0)
                    ),
                    size=surface_roughness,
                    confidence=0.75,
                    metadata={
                        'record_id': record_id,
                        'surface_roughness': surface_roughness,
                        'threshold': self.config.surface_roughness_threshold
                    }
                )
                defects.append(defect)
        
        except Exception as e:
            logger.error(f"Error detecting surface roughness defects: {e}")
        
        return defects
    
    def _detect_dimensional_defects(self, row: pd.Series, record_id: int) -> List[Defect]:
        """Detect dimensional deviation defects."""
        defects = []
        
        try:
            # Check various dimensional parameters
            dimensional_params = ['length', 'width', 'height', 'diameter', 'thickness']
            
            for param in dimensional_params:
                if param in row:
                    deviation = abs(row.get(f'{param}_deviation', 0.0))
                    if deviation > self.config.dimensional_tolerance:
                        defect = Defect(
                            defect_id=f"dimensional_{param}_{record_id}",
                            defect_type=DefectType.DIMENSIONAL_DEVIATION,
                            severity=self._determine_dimensional_severity(deviation),
                            location=(
                                row.get('x_coordinate', 0.0),
                                row.get('y_coordinate', 0.0),
                                row.get('z_coordinate', 0.0)
                            ),
                            size=deviation,
                            confidence=0.8,
                            metadata={
                                'record_id': record_id,
                                'parameter': param,
                                'deviation': deviation,
                                'tolerance': self.config.dimensional_tolerance
                            }
                        )
                        defects.append(defect)
        
        except Exception as e:
            logger.error(f"Error detecting dimensional defects: {e}")
        
        return defects
    
    def _determine_porosity_severity(self, porosity: float) -> DefectSeverity:
        """Determine severity based on porosity value."""
        if porosity > 0.2:
            return DefectSeverity.CRITICAL
        elif porosity > 0.1:
            return DefectSeverity.HIGH
        elif porosity > 0.05:
            return DefectSeverity.MEDIUM
        else:
            return DefectSeverity.LOW
    
    def _determine_crack_severity(self, crack_length: float) -> DefectSeverity:
        """Determine severity based on crack length."""
        if crack_length > 1.0:
            return DefectSeverity.CRITICAL
        elif crack_length > 0.5:
            return DefectSeverity.HIGH
        elif crack_length > 0.1:
            return DefectSeverity.MEDIUM
        else:
            return DefectSeverity.LOW
    
    def _determine_inclusion_severity(self, inclusion_size: float) -> DefectSeverity:
        """Determine severity based on inclusion size."""
        if inclusion_size > 0.5:
            return DefectSeverity.CRITICAL
        elif inclusion_size > 0.2:
            return DefectSeverity.HIGH
        elif inclusion_size > 0.05:
            return DefectSeverity.MEDIUM
        else:
            return DefectSeverity.LOW
    
    def _determine_surface_roughness_severity(self, roughness: float) -> DefectSeverity:
        """Determine severity based on surface roughness."""
        if roughness > 50.0:
            return DefectSeverity.CRITICAL
        elif roughness > 25.0:
            return DefectSeverity.HIGH
        elif roughness > 10.0:
            return DefectSeverity.MEDIUM
        else:
            return DefectSeverity.LOW
    
    def _determine_dimensional_severity(self, deviation: float) -> DefectSeverity:
        """Determine severity based on dimensional deviation."""
        if deviation > 1.0:
            return DefectSeverity.CRITICAL
        elif deviation > 0.5:
            return DefectSeverity.HIGH
        elif deviation > 0.1:
            return DefectSeverity.MEDIUM
        else:
            return DefectSeverity.LOW
    
    def _calculate_total_volume(self, df: pd.DataFrame, data_type: str) -> float:
        """Calculate total volume for defect density calculation."""
        try:
            if data_type == 'ct_scan':
                # Use voxel dimensions if available
                if all(col in df.columns for col in ['voxel_size_x', 'voxel_size_y', 'voxel_size_z']):
                    voxel_volume = (df['voxel_size_x'] * df['voxel_size_y'] * df['voxel_size_z']).sum()
                    return voxel_volume
            elif data_type == 'powder_bed':
                # Use layer dimensions if available
                if all(col in df.columns for col in ['layer_length', 'layer_width', 'layer_thickness']):
                    layer_volume = (df['layer_length'] * df['layer_width'] * df['layer_thickness']).sum()
                    return layer_volume
            
            # Default volume calculation
            return len(df) * 1.0  # Assume unit volume per record
            
        except Exception as e:
            logger.error(f"Error calculating total volume: {e}")
            return 1.0
    
    def _create_quality_results(self, data: List[Dict[str, Any]], 
                               defects: List[Defect], 
                               data_type: str) -> List[QualityResult]:
        """Create quality results from defect analysis."""
        try:
            quality_results = []
            
            for i, defect in enumerate(defects):
                quality_result = QualityResult(
                    rule_id=f"defect_analysis_{i}",
                    rule_name=f"Defect Analysis - {defect.defect_type.value}",
                    rule_type="defect",
                    passed=defect.severity in [DefectSeverity.LOW, DefectSeverity.MEDIUM],
                    quality_score=1.0 - (defect.confidence * 0.5),  # Convert to quality score
                    message=f"Defect detected: {defect.defect_type.value} "
                           f"(severity: {defect.severity.value}, size: {defect.size:.3f})",
                    record_id=str(defect.metadata.get('record_id', 0)),
                    field_name="defect_analysis",
                    expected_value="no_defects",
                    actual_value=f"{defect.defect_type.value}_{defect.severity.value}",
                    severity=defect.severity.value,
                    metadata={
                        'defect_id': defect.defect_id,
                        'defect_type': defect.defect_type.value,
                        'severity': defect.severity.value,
                        'location': defect.location,
                        'size': defect.size,
                        'confidence': defect.confidence,
                        'data_type': data_type,
                        'detection_timestamp': datetime.now().isoformat()
                    }
                )
                quality_results.append(quality_result)
            
            return quality_results
            
        except Exception as e:
            logger.error(f"Error creating quality results: {e}")
            return []
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of the defect analyzer."""
        try:
            return {
                'status': 'healthy',
                'config': self.config.__dict__,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }


# Convenience functions
def create_defect_analyzer(**kwargs) -> DefectAnalyzer:
    """Create a defect analyzer with custom configuration."""
    config = DefectConfig(**kwargs)
    return DefectAnalyzer(config)


def analyze_defects(data: List[Dict[str, Any]], data_type: str = 'generic', **kwargs) -> DefectAnalysisResult:
    """Convenience function for defect analysis."""
    analyzer = create_defect_analyzer(**kwargs)
    return analyzer.analyze_defects(data, data_type)
