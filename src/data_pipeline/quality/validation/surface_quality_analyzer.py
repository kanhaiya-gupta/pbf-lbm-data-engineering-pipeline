"""
Surface Quality Analyzer for PBF-LB/M Data Pipeline

This module provides surface quality analysis capabilities for evaluating
surface characteristics in PBF-LB/M manufacturing data, particularly
for powder bed and CT scan data.
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


class SurfaceQualityMetric(Enum):
    """Surface quality metrics."""
    ROUGHNESS = "roughness"
    WAVINESS = "waviness"
    FLATNESS = "flatness"
    STRAIGHTNESS = "straightness"
    CIRCULARITY = "circularity"
    CYLINDRICITY = "cylindricity"
    SURFACE_AREA = "surface_area"
    VOLUME = "volume"


class SurfaceQualityGrade(Enum):
    """Surface quality grades."""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    UNACCEPTABLE = "unacceptable"


@dataclass
class SurfaceQualityConfig:
    """Configuration for surface quality analysis."""
    roughness_threshold_excellent: float = 1.6  # μm
    roughness_threshold_good: float = 3.2  # μm
    roughness_threshold_fair: float = 6.3  # μm
    roughness_threshold_poor: float = 12.5  # μm
    
    waviness_threshold_excellent: float = 0.8  # μm
    waviness_threshold_good: float = 1.6  # μm
    waviness_threshold_fair: float = 3.2  # μm
    waviness_threshold_poor: float = 6.3  # μm
    
    flatness_tolerance_excellent: float = 0.01  # mm
    flatness_tolerance_good: float = 0.02  # mm
    flatness_tolerance_fair: float = 0.05  # mm
    flatness_tolerance_poor: float = 0.1  # mm
    
    enable_roughness_analysis: bool = True
    enable_waviness_analysis: bool = True
    enable_flatness_analysis: bool = True
    enable_geometric_analysis: bool = True
    enable_surface_area_analysis: bool = True


@dataclass
class SurfaceQualityMeasurement:
    """Individual surface quality measurement."""
    metric: SurfaceQualityMetric
    value: float
    unit: str
    grade: SurfaceQualityGrade
    location: Tuple[float, float, float]  # x, y, z coordinates
    confidence: float  # Measurement confidence (0-1)
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class SurfaceQualityResult:
    """Result of surface quality analysis."""
    total_records: int
    records_analyzed: int
    overall_quality_grade: SurfaceQualityGrade
    quality_score: float  # Overall quality score (0-1)
    measurements: List[SurfaceQualityMeasurement]
    quality_results: List[QualityResult]
    processing_time_seconds: float
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class SurfaceQualityAnalyzer:
    """
    Surface quality analyzer for PBF-LB/M manufacturing data.
    
    This analyzer evaluates surface characteristics including roughness,
    waviness, flatness, and geometric properties of PBF-LB/M parts.
    """
    
    def __init__(self, config: Optional[SurfaceQualityConfig] = None):
        """
        Initialize the surface quality analyzer.
        
        Args:
            config: Surface quality analysis configuration
        """
        self.config = config or SurfaceQualityConfig()
        
        logger.info("Surface Quality Analyzer initialized")
    
    def analyze_surface_quality(self, data: List[Dict[str, Any]], 
                               data_type: str = 'generic') -> SurfaceQualityResult:
        """
        Analyze surface quality in the provided data.
        
        Args:
            data: List of data records
            data_type: Type of data (powder_bed, ct_scan, etc.)
            
        Returns:
            SurfaceQualityResult: Surface quality analysis result
        """
        start_time = datetime.now()
        
        try:
            logger.info(f"Starting surface quality analysis for {len(data)} records of type {data_type}")
            
            # Convert data to DataFrame for easier processing
            df = pd.DataFrame(data)
            
            if df.empty:
                return SurfaceQualityResult(
                    total_records=0,
                    records_analyzed=0,
                    overall_quality_grade=SurfaceQualityGrade.UNACCEPTABLE,
                    quality_score=0.0,
                    measurements=[],
                    quality_results=[],
                    processing_time_seconds=0.0
                )
            
            # Analyze surface quality based on data type
            if data_type == 'powder_bed':
                measurements = self._analyze_powder_bed_surface_quality(df)
            elif data_type == 'ct_scan':
                measurements = self._analyze_ct_scan_surface_quality(df)
            else:
                measurements = self._analyze_generic_surface_quality(df)
            
            # Calculate overall quality grade and score
            overall_grade, quality_score = self._calculate_overall_quality(measurements)
            
            # Create quality results
            quality_results = self._create_quality_results(data, measurements, data_type)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Create result
            result = SurfaceQualityResult(
                total_records=len(data),
                records_analyzed=len(data),
                overall_quality_grade=overall_grade,
                quality_score=quality_score,
                measurements=measurements,
                quality_results=quality_results,
                processing_time_seconds=processing_time,
                metadata={
                    'data_type': data_type,
                    'analysis_config': self.config.__dict__,
                    'analysis_timestamp': datetime.now().isoformat()
                }
            )
            
            logger.info(f"Surface quality analysis completed: {overall_grade.value} grade "
                       f"(score: {quality_score:.3f})")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in surface quality analysis: {e}")
            return SurfaceQualityResult(
                total_records=len(data),
                records_analyzed=0,
                overall_quality_grade=SurfaceQualityGrade.UNACCEPTABLE,
                quality_score=0.0,
                measurements=[],
                quality_results=[],
                processing_time_seconds=(datetime.now() - start_time).total_seconds(),
                metadata={'error': str(e)}
            )
    
    def _analyze_powder_bed_surface_quality(self, df: pd.DataFrame) -> List[SurfaceQualityMeasurement]:
        """Analyze surface quality in powder bed data."""
        measurements = []
        
        try:
            for idx, row in df.iterrows():
                record_measurements = []
                
                # Roughness analysis
                if self.config.enable_roughness_analysis and 'surface_roughness' in row:
                    roughness_measurement = self._analyze_roughness(row, idx)
                    if roughness_measurement:
                        record_measurements.append(roughness_measurement)
                
                # Waviness analysis
                if self.config.enable_waviness_analysis and 'surface_waviness' in row:
                    waviness_measurement = self._analyze_waviness(row, idx)
                    if waviness_measurement:
                        record_measurements.append(waviness_measurement)
                
                # Flatness analysis
                if self.config.enable_flatness_analysis and 'flatness_deviation' in row:
                    flatness_measurement = self._analyze_flatness(row, idx)
                    if flatness_measurement:
                        record_measurements.append(flatness_measurement)
                
                # Surface area analysis
                if self.config.enable_surface_area_analysis and 'surface_area' in row:
                    surface_area_measurement = self._analyze_surface_area(row, idx)
                    if surface_area_measurement:
                        record_measurements.append(surface_area_measurement)
                
                measurements.extend(record_measurements)
            
        except Exception as e:
            logger.error(f"Error analyzing powder bed surface quality: {e}")
        
        return measurements
    
    def _analyze_ct_scan_surface_quality(self, df: pd.DataFrame) -> List[SurfaceQualityMeasurement]:
        """Analyze surface quality in CT scan data."""
        measurements = []
        
        try:
            for idx, row in df.iterrows():
                record_measurements = []
                
                # Roughness analysis from CT data
                if self.config.enable_roughness_analysis and 'surface_roughness' in row:
                    roughness_measurement = self._analyze_roughness(row, idx)
                    if roughness_measurement:
                        record_measurements.append(roughness_measurement)
                
                # Geometric analysis
                if self.config.enable_geometric_analysis:
                    geometric_measurements = self._analyze_geometric_properties(row, idx)
                    record_measurements.extend(geometric_measurements)
                
                # Volume analysis
                if 'volume' in row:
                    volume_measurement = self._analyze_volume(row, idx)
                    if volume_measurement:
                        record_measurements.append(volume_measurement)
                
                measurements.extend(record_measurements)
            
        except Exception as e:
            logger.error(f"Error analyzing CT scan surface quality: {e}")
        
        return measurements
    
    def _analyze_generic_surface_quality(self, df: pd.DataFrame) -> List[SurfaceQualityMeasurement]:
        """Analyze surface quality in generic data."""
        measurements = []
        
        try:
            for idx, row in df.iterrows():
                record_measurements = []
                
                # Look for common surface quality parameters
                if 'surface_roughness' in row:
                    roughness_measurement = self._analyze_roughness(row, idx)
                    if roughness_measurement:
                        record_measurements.append(roughness_measurement)
                
                if 'surface_waviness' in row:
                    waviness_measurement = self._analyze_waviness(row, idx)
                    if waviness_measurement:
                        record_measurements.append(waviness_measurement)
                
                if 'flatness_deviation' in row:
                    flatness_measurement = self._analyze_flatness(row, idx)
                    if flatness_measurement:
                        record_measurements.append(flatness_measurement)
                
                measurements.extend(record_measurements)
            
        except Exception as e:
            logger.error(f"Error analyzing generic surface quality: {e}")
        
        return measurements
    
    def _analyze_roughness(self, row: pd.Series, record_id: int) -> Optional[SurfaceQualityMeasurement]:
        """Analyze surface roughness."""
        try:
            roughness_value = row.get('surface_roughness', 0.0)
            if roughness_value > 0:
                grade = self._determine_roughness_grade(roughness_value)
                return SurfaceQualityMeasurement(
                    metric=SurfaceQualityMetric.ROUGHNESS,
                    value=roughness_value,
                    unit="μm",
                    grade=grade,
                    location=(
                        row.get('x_coordinate', 0.0),
                        row.get('y_coordinate', 0.0),
                        row.get('z_coordinate', 0.0)
                    ),
                    confidence=0.9,
                    metadata={
                        'record_id': record_id,
                        'measurement_type': 'surface_roughness',
                        'measurement_timestamp': datetime.now().isoformat()
                    }
                )
        except Exception as e:
            logger.error(f"Error analyzing roughness: {e}")
        
        return None
    
    def _analyze_waviness(self, row: pd.Series, record_id: int) -> Optional[SurfaceQualityMeasurement]:
        """Analyze surface waviness."""
        try:
            waviness_value = row.get('surface_waviness', 0.0)
            if waviness_value > 0:
                grade = self._determine_waviness_grade(waviness_value)
                return SurfaceQualityMeasurement(
                    metric=SurfaceQualityMetric.WAVINESS,
                    value=waviness_value,
                    unit="μm",
                    grade=grade,
                    location=(
                        row.get('x_coordinate', 0.0),
                        row.get('y_coordinate', 0.0),
                        row.get('z_coordinate', 0.0)
                    ),
                    confidence=0.85,
                    metadata={
                        'record_id': record_id,
                        'measurement_type': 'surface_waviness',
                        'measurement_timestamp': datetime.now().isoformat()
                    }
                )
        except Exception as e:
            logger.error(f"Error analyzing waviness: {e}")
        
        return None
    
    def _analyze_flatness(self, row: pd.Series, record_id: int) -> Optional[SurfaceQualityMeasurement]:
        """Analyze surface flatness."""
        try:
            flatness_deviation = row.get('flatness_deviation', 0.0)
            if flatness_deviation >= 0:
                grade = self._determine_flatness_grade(flatness_deviation)
                return SurfaceQualityMeasurement(
                    metric=SurfaceQualityMetric.FLATNESS,
                    value=flatness_deviation,
                    unit="mm",
                    grade=grade,
                    location=(
                        row.get('x_coordinate', 0.0),
                        row.get('y_coordinate', 0.0),
                        row.get('z_coordinate', 0.0)
                    ),
                    confidence=0.8,
                    metadata={
                        'record_id': record_id,
                        'measurement_type': 'flatness_deviation',
                        'measurement_timestamp': datetime.now().isoformat()
                    }
                )
        except Exception as e:
            logger.error(f"Error analyzing flatness: {e}")
        
        return None
    
    def _analyze_surface_area(self, row: pd.Series, record_id: int) -> Optional[SurfaceQualityMeasurement]:
        """Analyze surface area."""
        try:
            surface_area = row.get('surface_area', 0.0)
            if surface_area > 0:
                # Surface area doesn't have a grade, but we can assess it
                grade = SurfaceQualityGrade.GOOD  # Default grade for surface area
                return SurfaceQualityMeasurement(
                    metric=SurfaceQualityMetric.SURFACE_AREA,
                    value=surface_area,
                    unit="mm²",
                    grade=grade,
                    location=(
                        row.get('x_coordinate', 0.0),
                        row.get('y_coordinate', 0.0),
                        row.get('z_coordinate', 0.0)
                    ),
                    confidence=0.9,
                    metadata={
                        'record_id': record_id,
                        'measurement_type': 'surface_area',
                        'measurement_timestamp': datetime.now().isoformat()
                    }
                )
        except Exception as e:
            logger.error(f"Error analyzing surface area: {e}")
        
        return None
    
    def _analyze_volume(self, row: pd.Series, record_id: int) -> Optional[SurfaceQualityMeasurement]:
        """Analyze volume."""
        try:
            volume = row.get('volume', 0.0)
            if volume > 0:
                grade = SurfaceQualityGrade.GOOD  # Default grade for volume
                return SurfaceQualityMeasurement(
                    metric=SurfaceQualityMetric.VOLUME,
                    value=volume,
                    unit="mm³",
                    grade=grade,
                    location=(
                        row.get('x_coordinate', 0.0),
                        row.get('y_coordinate', 0.0),
                        row.get('z_coordinate', 0.0)
                    ),
                    confidence=0.9,
                    metadata={
                        'record_id': record_id,
                        'measurement_type': 'volume',
                        'measurement_timestamp': datetime.now().isoformat()
                    }
                )
        except Exception as e:
            logger.error(f"Error analyzing volume: {e}")
        
        return None
    
    def _analyze_geometric_properties(self, row: pd.Series, record_id: int) -> List[SurfaceQualityMeasurement]:
        """Analyze geometric properties."""
        measurements = []
        
        try:
            # Analyze various geometric properties
            geometric_params = {
                'circularity': SurfaceQualityMetric.CIRCULARITY,
                'cylindricity': SurfaceQualityMetric.CYLINDRICITY,
                'straightness': SurfaceQualityMetric.STRAIGHTNESS
            }
            
            for param, metric in geometric_params.items():
                if param in row:
                    value = row[param]
                    if value > 0:
                        grade = SurfaceQualityGrade.GOOD  # Default grade
                        measurement = SurfaceQualityMeasurement(
                            metric=metric,
                            value=value,
                            unit="mm",
                            grade=grade,
                            location=(
                                row.get('x_coordinate', 0.0),
                                row.get('y_coordinate', 0.0),
                                row.get('z_coordinate', 0.0)
                            ),
                            confidence=0.8,
                            metadata={
                                'record_id': record_id,
                                'measurement_type': param,
                                'measurement_timestamp': datetime.now().isoformat()
                            }
                        )
                        measurements.append(measurement)
        
        except Exception as e:
            logger.error(f"Error analyzing geometric properties: {e}")
        
        return measurements
    
    def _determine_roughness_grade(self, roughness: float) -> SurfaceQualityGrade:
        """Determine grade based on roughness value."""
        if roughness <= self.config.roughness_threshold_excellent:
            return SurfaceQualityGrade.EXCELLENT
        elif roughness <= self.config.roughness_threshold_good:
            return SurfaceQualityGrade.GOOD
        elif roughness <= self.config.roughness_threshold_fair:
            return SurfaceQualityGrade.FAIR
        elif roughness <= self.config.roughness_threshold_poor:
            return SurfaceQualityGrade.POOR
        else:
            return SurfaceQualityGrade.UNACCEPTABLE
    
    def _determine_waviness_grade(self, waviness: float) -> SurfaceQualityGrade:
        """Determine grade based on waviness value."""
        if waviness <= self.config.waviness_threshold_excellent:
            return SurfaceQualityGrade.EXCELLENT
        elif waviness <= self.config.waviness_threshold_good:
            return SurfaceQualityGrade.GOOD
        elif waviness <= self.config.waviness_threshold_fair:
            return SurfaceQualityGrade.FAIR
        elif waviness <= self.config.waviness_threshold_poor:
            return SurfaceQualityGrade.POOR
        else:
            return SurfaceQualityGrade.UNACCEPTABLE
    
    def _determine_flatness_grade(self, flatness_deviation: float) -> SurfaceQualityGrade:
        """Determine grade based on flatness deviation."""
        if flatness_deviation <= self.config.flatness_tolerance_excellent:
            return SurfaceQualityGrade.EXCELLENT
        elif flatness_deviation <= self.config.flatness_tolerance_good:
            return SurfaceQualityGrade.GOOD
        elif flatness_deviation <= self.config.flatness_tolerance_fair:
            return SurfaceQualityGrade.FAIR
        elif flatness_deviation <= self.config.flatness_tolerance_poor:
            return SurfaceQualityGrade.POOR
        else:
            return SurfaceQualityGrade.UNACCEPTABLE
    
    def _calculate_overall_quality(self, measurements: List[SurfaceQualityMeasurement]) -> Tuple[SurfaceQualityGrade, float]:
        """Calculate overall quality grade and score."""
        try:
            if not measurements:
                return SurfaceQualityGrade.UNACCEPTABLE, 0.0
            
            # Weight different metrics
            metric_weights = {
                SurfaceQualityMetric.ROUGHNESS: 0.4,
                SurfaceQualityMetric.WAVINESS: 0.2,
                SurfaceQualityMetric.FLATNESS: 0.2,
                SurfaceQualityMetric.STRAIGHTNESS: 0.1,
                SurfaceQualityMetric.CIRCULARITY: 0.05,
                SurfaceQualityMetric.CYLINDRICITY: 0.05
            }
            
            # Calculate weighted score
            total_score = 0.0
            total_weight = 0.0
            
            for measurement in measurements:
                weight = metric_weights.get(measurement.metric, 0.1)
                grade_score = self._grade_to_score(measurement.grade)
                total_score += grade_score * weight
                total_weight += weight
            
            overall_score = total_score / total_weight if total_weight > 0 else 0.0
            overall_grade = self._score_to_grade(overall_score)
            
            return overall_grade, overall_score
            
        except Exception as e:
            logger.error(f"Error calculating overall quality: {e}")
            return SurfaceQualityGrade.UNACCEPTABLE, 0.0
    
    def _grade_to_score(self, grade: SurfaceQualityGrade) -> float:
        """Convert grade to numerical score."""
        grade_scores = {
            SurfaceQualityGrade.EXCELLENT: 1.0,
            SurfaceQualityGrade.GOOD: 0.8,
            SurfaceQualityGrade.FAIR: 0.6,
            SurfaceQualityGrade.POOR: 0.4,
            SurfaceQualityGrade.UNACCEPTABLE: 0.0
        }
        return grade_scores.get(grade, 0.0)
    
    def _score_to_grade(self, score: float) -> SurfaceQualityGrade:
        """Convert numerical score to grade."""
        if score >= 0.9:
            return SurfaceQualityGrade.EXCELLENT
        elif score >= 0.7:
            return SurfaceQualityGrade.GOOD
        elif score >= 0.5:
            return SurfaceQualityGrade.FAIR
        elif score >= 0.3:
            return SurfaceQualityGrade.POOR
        else:
            return SurfaceQualityGrade.UNACCEPTABLE
    
    def _create_quality_results(self, data: List[Dict[str, Any]], 
                               measurements: List[SurfaceQualityMeasurement], 
                               data_type: str) -> List[QualityResult]:
        """Create quality results from surface quality analysis."""
        try:
            quality_results = []
            
            for i, measurement in enumerate(measurements):
                quality_result = QualityResult(
                    rule_id=f"surface_quality_{i}",
                    rule_name=f"Surface Quality - {measurement.metric.value}",
                    rule_type="surface_quality",
                    passed=measurement.grade in [SurfaceQualityGrade.EXCELLENT, 
                                                SurfaceQualityGrade.GOOD, 
                                                SurfaceQualityGrade.FAIR],
                    quality_score=self._grade_to_score(measurement.grade),
                    message=f"Surface quality measurement: {measurement.metric.value} = "
                           f"{measurement.value:.3f} {measurement.unit} (grade: {measurement.grade.value})",
                    record_id=str(measurement.metadata.get('record_id', 0)),
                    field_name=measurement.metric.value,
                    expected_value=f"grade_{SurfaceQualityGrade.GOOD.value}",
                    actual_value=f"grade_{measurement.grade.value}",
                    severity="medium" if measurement.grade in [SurfaceQualityGrade.POOR, 
                                                             SurfaceQualityGrade.UNACCEPTABLE] else "low",
                    metadata={
                        'metric': measurement.metric.value,
                        'value': measurement.value,
                        'unit': measurement.unit,
                        'grade': measurement.grade.value,
                        'location': measurement.location,
                        'confidence': measurement.confidence,
                        'data_type': data_type,
                        'measurement_timestamp': measurement.metadata.get('measurement_timestamp')
                    }
                )
                quality_results.append(quality_result)
            
            return quality_results
            
        except Exception as e:
            logger.error(f"Error creating quality results: {e}")
            return []
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of the surface quality analyzer."""
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
def create_surface_quality_analyzer(**kwargs) -> SurfaceQualityAnalyzer:
    """Create a surface quality analyzer with custom configuration."""
    config = SurfaceQualityConfig(**kwargs)
    return SurfaceQualityAnalyzer(config)


def analyze_surface_quality(data: List[Dict[str, Any]], data_type: str = 'generic', **kwargs) -> SurfaceQualityResult:
    """Convenience function for surface quality analysis."""
    analyzer = create_surface_quality_analyzer(**kwargs)
    return analyzer.analyze_surface_quality(data, data_type)
