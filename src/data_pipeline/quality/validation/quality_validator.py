"""
Quality Validator for PBF-LB/M Data Pipeline

This module provides a comprehensive quality validator that combines
multiple validation strategies for PBF-LB/M manufacturing data.
"""

import logging
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime
from dataclasses import dataclass
from abc import ABC, abstractmethod

from .data_quality_service import DataQualityService, QualityResult, QualityRule
from .schema_validator import SchemaValidator, SchemaValidationResult
from .business_rule_validator import BusinessRuleValidator, BusinessRuleResult
from .data_type_validator import DataTypeValidator, DataTypeValidationResult
from .anomaly_detector import AnomalyDetector, AnomalyResult
from .defect_analyzer import DefectAnalyzer, DefectAnalysisResult
from .surface_quality_analyzer import SurfaceQualityAnalyzer, SurfaceQualityResult

logger = logging.getLogger(__name__)


@dataclass
class QualityValidationConfig:
    """Configuration for quality validation."""
    enable_schema_validation: bool = True
    enable_business_rule_validation: bool = True
    enable_data_type_validation: bool = True
    enable_anomaly_detection: bool = True
    enable_defect_analysis: bool = True
    enable_surface_quality_analysis: bool = True
    validation_timeout_seconds: int = 300
    parallel_validation: bool = True
    max_validation_errors: int = 1000
    quality_threshold: float = 0.8


@dataclass
class ComprehensiveQualityResult:
    """Comprehensive quality validation result."""
    overall_quality_score: float
    validation_passed: bool
    total_records: int
    valid_records: int
    invalid_records: int
    validation_time_seconds: float
    
    # Individual validation results
    schema_validation: Optional[SchemaValidationResult] = None
    business_rule_validation: Optional[BusinessRuleResult] = None
    data_type_validation: Optional[DataTypeValidationResult] = None
    anomaly_detection: Optional[AnomalyResult] = None
    defect_analysis: Optional[DefectAnalysisResult] = None
    surface_quality_analysis: Optional[SurfaceQualityResult] = None
    
    # Aggregated results
    all_quality_results: List[QualityResult] = None
    errors: List[str] = None
    warnings: List[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.all_quality_results is None:
            self.all_quality_results = []
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []
        if self.metadata is None:
            self.metadata = {}


class QualityValidator:
    """
    Comprehensive quality validator for PBF-LB/M data pipeline.
    
    This validator combines multiple validation strategies to provide
    comprehensive quality assessment of PBF-LB/M manufacturing data.
    """
    
    def __init__(self, config: Optional[QualityValidationConfig] = None):
        """
        Initialize the quality validator.
        
        Args:
            config: Quality validation configuration
        """
        self.config = config or QualityValidationConfig()
        
        # Initialize validation components
        self.data_quality_service = DataQualityService()
        self.schema_validator = SchemaValidator() if self.config.enable_schema_validation else None
        self.business_rule_validator = BusinessRuleValidator() if self.config.enable_business_rule_validation else None
        self.data_type_validator = DataTypeValidator() if self.config.enable_data_type_validation else None
        self.anomaly_detector = AnomalyDetector() if self.config.enable_anomaly_detection else None
        self.defect_analyzer = DefectAnalyzer() if self.config.enable_defect_analysis else None
        self.surface_quality_analyzer = SurfaceQualityAnalyzer() if self.config.enable_surface_quality_analysis else None
        
        logger.info("Quality Validator initialized with comprehensive validation capabilities")
    
    def validate_data_quality(self, data: List[Dict[str, Any]], 
                            data_type: str = 'generic') -> ComprehensiveQualityResult:
        """
        Perform comprehensive quality validation on data.
        
        Args:
            data: List of data records to validate
            data_type: Type of data (ispm, powder_bed, pbf_process, ct_scan)
            
        Returns:
            ComprehensiveQualityResult: Comprehensive validation result
        """
        start_time = datetime.now()
        errors = []
        warnings = []
        all_quality_results = []
        
        try:
            logger.info(f"Starting comprehensive quality validation for {len(data)} records of type {data_type}")
            
            # Perform individual validations
            schema_result = None
            business_rule_result = None
            data_type_result = None
            anomaly_result = None
            defect_result = None
            surface_quality_result = None
            
            # Schema validation
            if self.schema_validator and self.config.enable_schema_validation:
                try:
                    schema_result = self.schema_validator.validate_schema(data, data_type)
                    all_quality_results.extend(schema_result.quality_results)
                except Exception as e:
                    error_msg = f"Schema validation failed: {e}"
                    logger.error(error_msg)
                    errors.append(error_msg)
            
            # Business rule validation
            if self.business_rule_validator and self.config.enable_business_rule_validation:
                try:
                    business_rule_result = self.business_rule_validator.validate_business_rules(data, data_type)
                    all_quality_results.extend(business_rule_result.quality_results)
                except Exception as e:
                    error_msg = f"Business rule validation failed: {e}"
                    logger.error(error_msg)
                    errors.append(error_msg)
            
            # Data type validation
            if self.data_type_validator and self.config.enable_data_type_validation:
                try:
                    data_type_result = self.data_type_validator.validate_data_types(data, data_type)
                    all_quality_results.extend(data_type_result.quality_results)
                except Exception as e:
                    error_msg = f"Data type validation failed: {e}"
                    logger.error(error_msg)
                    errors.append(error_msg)
            
            # Anomaly detection
            if self.anomaly_detector and self.config.enable_anomaly_detection:
                try:
                    anomaly_result = self.anomaly_detector.detect_anomalies(data, data_type)
                    all_quality_results.extend(anomaly_result.quality_results)
                except Exception as e:
                    error_msg = f"Anomaly detection failed: {e}"
                    logger.error(error_msg)
                    errors.append(error_msg)
            
            # Defect analysis (for specific data types)
            if self.defect_analyzer and self.config.enable_defect_analysis and data_type in ['ct_scan', 'powder_bed']:
                try:
                    defect_result = self.defect_analyzer.analyze_defects(data, data_type)
                    all_quality_results.extend(defect_result.quality_results)
                except Exception as e:
                    error_msg = f"Defect analysis failed: {e}"
                    logger.error(error_msg)
                    errors.append(error_msg)
            
            # Surface quality analysis (for specific data types)
            if self.surface_quality_analyzer and self.config.enable_surface_quality_analysis and data_type in ['powder_bed', 'ct_scan']:
                try:
                    surface_quality_result = self.surface_quality_analyzer.analyze_surface_quality(data, data_type)
                    all_quality_results.extend(surface_quality_result.quality_results)
                except Exception as e:
                    error_msg = f"Surface quality analysis failed: {e}"
                    logger.error(error_msg)
                    errors.append(error_msg)
            
            # Calculate overall quality score
            overall_quality_score = self._calculate_overall_quality_score(all_quality_results)
            
            # Count valid/invalid records
            valid_records = sum(1 for result in all_quality_results if result.passed)
            invalid_records = len(data) - valid_records
            
            # Determine if validation passed
            validation_passed = (overall_quality_score >= self.config.quality_threshold and 
                               len(errors) == 0 and 
                               invalid_records <= self.config.max_validation_errors)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Create comprehensive result
            result = ComprehensiveQualityResult(
                overall_quality_score=overall_quality_score,
                validation_passed=validation_passed,
                total_records=len(data),
                valid_records=valid_records,
                invalid_records=invalid_records,
                validation_time_seconds=processing_time,
                schema_validation=schema_result,
                business_rule_validation=business_rule_result,
                data_type_validation=data_type_result,
                anomaly_detection=anomaly_result,
                defect_analysis=defect_result,
                surface_quality_analysis=surface_quality_result,
                all_quality_results=all_quality_results,
                errors=errors,
                warnings=warnings,
                metadata={
                    'data_type': data_type,
                    'validation_config': self.config.__dict__,
                    'validation_timestamp': datetime.now().isoformat()
                }
            )
            
            logger.info(f"Quality validation completed: {overall_quality_score:.3f} score, "
                       f"{valid_records}/{len(data)} records valid")
            
            return result
            
        except Exception as e:
            error_msg = f"Critical error in quality validation: {e}"
            logger.error(error_msg)
            return ComprehensiveQualityResult(
                overall_quality_score=0.0,
                validation_passed=False,
                total_records=len(data),
                valid_records=0,
                invalid_records=len(data),
                validation_time_seconds=(datetime.now() - start_time).total_seconds(),
                errors=[error_msg],
                warnings=warnings
            )
    
    def _calculate_overall_quality_score(self, quality_results: List[QualityResult]) -> float:
        """Calculate overall quality score from individual results."""
        if not quality_results:
            return 0.0
        
        # Weight different types of validations
        weights = {
            'schema': 0.2,
            'business_rule': 0.3,
            'data_type': 0.2,
            'anomaly': 0.15,
            'defect': 0.1,
            'surface_quality': 0.05
        }
        
        weighted_score = 0.0
        total_weight = 0.0
        
        for result in quality_results:
            weight = weights.get(result.rule_type, 0.1)
            weighted_score += result.quality_score * weight
            total_weight += weight
        
        return weighted_score / total_weight if total_weight > 0 else 0.0
    
    def validate_ispm_data(self, data: List[Dict[str, Any]]) -> ComprehensiveQualityResult:
        """Validate ISPM monitoring data."""
        return self.validate_data_quality(data, 'ispm')
    
    def validate_powder_bed_data(self, data: List[Dict[str, Any]]) -> ComprehensiveQualityResult:
        """Validate powder bed monitoring data."""
        return self.validate_data_quality(data, 'powder_bed')
    
    def validate_pbf_process_data(self, data: List[Dict[str, Any]]) -> ComprehensiveQualityResult:
        """Validate PBF process data."""
        return self.validate_data_quality(data, 'pbf_process')
    
    def validate_ct_scan_data(self, data: List[Dict[str, Any]]) -> ComprehensiveQualityResult:
        """Validate CT scan data."""
        return self.validate_data_quality(data, 'ct_scan')
    
    def get_validation_metrics(self) -> Dict[str, Any]:
        """Get validation metrics and statistics."""
        try:
            metrics = {
                'validator_status': 'active',
                'config': self.config.__dict__,
                'enabled_validators': {
                    'schema_validation': self.config.enable_schema_validation,
                    'business_rule_validation': self.config.enable_business_rule_validation,
                    'data_type_validation': self.config.enable_data_type_validation,
                    'anomaly_detection': self.config.enable_anomaly_detection,
                    'defect_analysis': self.config.enable_defect_analysis,
                    'surface_quality_analysis': self.config.enable_surface_quality_analysis
                },
                'timestamp': datetime.now().isoformat()
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting validation metrics: {e}")
            return {'error': str(e)}
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of the quality validator."""
        try:
            status = {
                'status': 'healthy',
                'components': {},
                'timestamp': datetime.now().isoformat()
            }
            
            # Check component health
            components = [
                ('data_quality_service', self.data_quality_service),
                ('schema_validator', self.schema_validator),
                ('business_rule_validator', self.business_rule_validator),
                ('data_type_validator', self.data_type_validator),
                ('anomaly_detector', self.anomaly_detector),
                ('defect_analyzer', self.defect_analyzer),
                ('surface_quality_analyzer', self.surface_quality_analyzer)
            ]
            
            for name, component in components:
                if component:
                    try:
                        if hasattr(component, 'get_health_status'):
                            component_health = component.get_health_status()
                            status['components'][name] = component_health
                        else:
                            status['components'][name] = {'status': 'active'}
                    except Exception as e:
                        status['components'][name] = {'status': 'error', 'error': str(e)}
                else:
                    status['components'][name] = {'status': 'disabled'}
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting health status: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }


# Convenience functions for common operations
def create_quality_validator(**kwargs) -> QualityValidator:
    """
    Create a quality validator with custom configuration.
    
    Args:
        **kwargs: Configuration parameters
        
    Returns:
        Configured QualityValidator instance
    """
    config = QualityValidationConfig(**kwargs)
    return QualityValidator(config)


def validate_data_quality(data: List[Dict[str, Any]], data_type: str = 'generic', **kwargs) -> ComprehensiveQualityResult:
    """
    Convenience function for data quality validation.
    
    Args:
        data: Data records to validate
        data_type: Type of data
        **kwargs: Additional configuration parameters
        
    Returns:
        ComprehensiveQualityResult: Validation result
    """
    validator = create_quality_validator(**kwargs)
    return validator.validate_data_quality(data, data_type)
