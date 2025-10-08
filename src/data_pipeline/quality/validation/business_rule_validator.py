"""
Business Rule Validator

This module provides business rule validation capabilities for the PBF-LB/M data pipeline.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging

from src.data_pipeline.config.pipeline_config import get_pipeline_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BusinessRuleType(Enum):
    """Business rule type enumeration."""
    TEMPERATURE_CORRELATION = "temperature_correlation"
    PRESSURE_STABILITY = "pressure_stability"
    LASER_PARAMETER_CONSISTENCY = "laser_parameter_consistency"
    BUILD_QUALITY_THRESHOLD = "build_quality_threshold"
    PROCESS_EFFICIENCY = "process_efficiency"
    MATERIAL_PROPERTY_COMPLIANCE = "material_property_compliance"
    SAFETY_THRESHOLD = "safety_threshold"

class RuleSeverity(Enum):
    """Rule severity enumeration."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class BusinessRule:
    """Business rule data class."""
    id: str
    name: str
    description: str
    rule_type: BusinessRuleType
    severity: RuleSeverity
    enabled: bool = True
    threshold_value: Optional[float] = None
    parameters: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BusinessRuleResult:
    """Business rule validation result data class."""
    rule_id: str
    rule_name: str
    passed: bool
    severity: RuleSeverity
    message: str
    affected_records: List[int] = field(default_factory=list)
    total_records: int = 0
    passed_records: int = 0
    failed_records: int = 0
    confidence_score: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class BusinessRuleValidationSummary:
    """Business rule validation summary data class."""
    source_name: str
    total_rules: int
    passed_rules: int
    failed_rules: int
    critical_violations: int
    warning_violations: int
    overall_compliance_score: float
    results: List[BusinessRuleResult] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

class BusinessRuleValidator:
    """
    Business rule validation service for PBF-LB/M data pipeline.
    """
    
    def __init__(self):
        self.config = get_pipeline_config()
        self.business_rules: Dict[str, BusinessRule] = {}
        self.validation_results: Dict[str, BusinessRuleValidationSummary] = {}
        
        # Initialize business rules
        self._initialize_business_rules()
        
    def validate_pbf_process_business_rules(self, data: List[Dict[str, Any]]) -> BusinessRuleValidationSummary:
        """
        Validate PBF process data against business rules.
        
        Args:
            data: List of PBF process data records
            
        Returns:
            BusinessRuleValidationSummary: Business rule validation results
        """
        try:
            logger.info(f"Validating PBF process business rules for {len(data)} records")
            
            df = pd.DataFrame(data)
            results = []
            
            # Apply PBF-specific business rules
            pbf_rules = [rule for rule in self.business_rules.values() 
                        if rule.enabled and "pbf_process" in rule.parameters.get("applicable_sources", [])]
            
            for rule in pbf_rules:
                result = self._apply_business_rule(df, rule)
                results.append(result)
            
            # Calculate summary
            summary = self._calculate_validation_summary("pbf_process", results)
            self.validation_results["pbf_process"] = summary
            
            logger.info(f"PBF process business rule validation completed. Compliance score: {summary.overall_compliance_score:.2f}")
            return summary
            
        except Exception as e:
            logger.error(f"Error validating PBF process business rules: {e}")
            raise
    
    def validate_ispm_monitoring_business_rules(self, data: List[Dict[str, Any]]) -> BusinessRuleValidationSummary:
        """
        Validate ISPM monitoring data against business rules.
        
        Args:
            data: List of ISPM monitoring data records
            
        Returns:
            BusinessRuleValidationSummary: Business rule validation results
        """
        try:
            logger.info(f"Validating ISPM monitoring business rules for {len(data)} records")
            
            df = pd.DataFrame(data)
            results = []
            
            # Apply ISPM-specific business rules
            ispm_rules = [rule for rule in self.business_rules.values() 
                         if rule.enabled and "ispm_monitoring" in rule.parameters.get("applicable_sources", [])]
            
            for rule in ispm_rules:
                result = self._apply_business_rule(df, rule)
                results.append(result)
            
            # Calculate summary
            summary = self._calculate_validation_summary("ispm_monitoring", results)
            self.validation_results["ispm_monitoring"] = summary
            
            logger.info(f"ISPM monitoring business rule validation completed. Compliance score: {summary.overall_compliance_score:.2f}")
            return summary
            
        except Exception as e:
            logger.error(f"Error validating ISPM monitoring business rules: {e}")
            raise
    
    def validate_ct_scan_business_rules(self, data: List[Dict[str, Any]]) -> BusinessRuleValidationSummary:
        """
        Validate CT scan data against business rules.
        
        Args:
            data: List of CT scan data records
            
        Returns:
            BusinessRuleValidationSummary: Business rule validation results
        """
        try:
            logger.info(f"Validating CT scan business rules for {len(data)} records")
            
            df = pd.DataFrame(data)
            results = []
            
            # Apply CT scan-specific business rules
            ct_rules = [rule for rule in self.business_rules.values() 
                       if rule.enabled and "ct_scan" in rule.parameters.get("applicable_sources", [])]
            
            for rule in ct_rules:
                result = self._apply_business_rule(df, rule)
                results.append(result)
            
            # Calculate summary
            summary = self._calculate_validation_summary("ct_scan", results)
            self.validation_results["ct_scan"] = summary
            
            logger.info(f"CT scan business rule validation completed. Compliance score: {summary.overall_compliance_score:.2f}")
            return summary
            
        except Exception as e:
            logger.error(f"Error validating CT scan business rules: {e}")
            raise
    
    def validate_powder_bed_business_rules(self, data: List[Dict[str, Any]]) -> BusinessRuleValidationSummary:
        """
        Validate powder bed data against business rules.
        
        Args:
            data: List of powder bed data records
            
        Returns:
            BusinessRuleValidationSummary: Business rule validation results
        """
        try:
            logger.info(f"Validating powder bed business rules for {len(data)} records")
            
            df = pd.DataFrame(data)
            results = []
            
            # Apply powder bed-specific business rules
            pb_rules = [rule for rule in self.business_rules.values() 
                       if rule.enabled and "powder_bed" in rule.parameters.get("applicable_sources", [])]
            
            for rule in pb_rules:
                result = self._apply_business_rule(df, rule)
                results.append(result)
            
            # Calculate summary
            summary = self._calculate_validation_summary("powder_bed", results)
            self.validation_results["powder_bed"] = summary
            
            logger.info(f"Powder bed business rule validation completed. Compliance score: {summary.overall_compliance_score:.2f}")
            return summary
            
        except Exception as e:
            logger.error(f"Error validating powder bed business rules: {e}")
            raise
    
    def _apply_business_rule(self, df: pd.DataFrame, rule: BusinessRule) -> BusinessRuleResult:
        """Apply a specific business rule to the data."""
        try:
            if rule.rule_type == BusinessRuleType.TEMPERATURE_CORRELATION:
                return self._validate_temperature_correlation(df, rule)
            elif rule.rule_type == BusinessRuleType.PRESSURE_STABILITY:
                return self._validate_pressure_stability(df, rule)
            elif rule.rule_type == BusinessRuleType.LASER_PARAMETER_CONSISTENCY:
                return self._validate_laser_parameter_consistency(df, rule)
            elif rule.rule_type == BusinessRuleType.BUILD_QUALITY_THRESHOLD:
                return self._validate_build_quality_threshold(df, rule)
            elif rule.rule_type == BusinessRuleType.PROCESS_EFFICIENCY:
                return self._validate_process_efficiency(df, rule)
            elif rule.rule_type == BusinessRuleType.MATERIAL_PROPERTY_COMPLIANCE:
                return self._validate_material_property_compliance(df, rule)
            elif rule.rule_type == BusinessRuleType.SAFETY_THRESHOLD:
                return self._validate_safety_threshold(df, rule)
            else:
                return BusinessRuleResult(
                    rule_id=rule.id,
                    rule_name=rule.name,
                    passed=True,
                    severity=RuleSeverity.INFO,
                    message=f"Unknown rule type: {rule.rule_type}",
                    total_records=len(df)
                )
                
        except Exception as e:
            logger.error(f"Error applying business rule {rule.id}: {e}")
            return BusinessRuleResult(
                rule_id=rule.id,
                rule_name=rule.name,
                passed=False,
                severity=RuleSeverity.ERROR,
                message=f"Rule application error: {e}",
                total_records=len(df)
            )
    
    def _validate_temperature_correlation(self, df: pd.DataFrame, rule: BusinessRule) -> BusinessRuleResult:
        """Validate temperature correlation business rule."""
        try:
            if "chamber_temperature" not in df.columns or "build_plate_temperature" not in df.columns:
                return BusinessRuleResult(
                    rule_id=rule.id,
                    rule_name=rule.name,
                    passed=True,
                    severity=RuleSeverity.INFO,
                    message="Required temperature columns not found",
                    total_records=len(df)
                )
            
            # Convert to numeric
            chamber_temp = pd.to_numeric(df["chamber_temperature"], errors='coerce')
            build_plate_temp = pd.to_numeric(df["build_plate_temperature"], errors='coerce')
            
            # Remove NaN values
            valid_mask = chamber_temp.notna() & build_plate_temp.notna()
            valid_chamber = chamber_temp[valid_mask]
            valid_build_plate = build_plate_temp[valid_mask]
            
            if len(valid_chamber) < 2:
                return BusinessRuleResult(
                    rule_id=rule.id,
                    rule_name=rule.name,
                    passed=True,
                    severity=RuleSeverity.INFO,
                    message="Insufficient valid temperature data",
                    total_records=len(df)
                )
            
            # Calculate correlation
            correlation = valid_chamber.corr(valid_build_plate)
            min_correlation = rule.parameters.get("min_correlation", 0.7)
            
            passed = correlation >= min_correlation
            failed_records = df[~valid_mask].index.tolist()
            
            return BusinessRuleResult(
                rule_id=rule.id,
                rule_name=rule.name,
                passed=passed,
                severity=rule.severity,
                message=f"Temperature correlation: {correlation:.3f} (threshold: {min_correlation})",
                affected_records=failed_records,
                total_records=len(df),
                passed_records=len(valid_chamber),
                failed_records=len(failed_records),
                confidence_score=abs(correlation)
            )
            
        except Exception as e:
            logger.error(f"Error validating temperature correlation: {e}")
            return BusinessRuleResult(
                rule_id=rule.id,
                rule_name=rule.name,
                passed=False,
                severity=RuleSeverity.ERROR,
                message=f"Temperature correlation validation error: {e}",
                total_records=len(df)
            )
    
    def _validate_pressure_stability(self, df: pd.DataFrame, rule: BusinessRule) -> BusinessRuleResult:
        """Validate pressure stability business rule."""
        try:
            if "chamber_pressure" not in df.columns:
                return BusinessRuleResult(
                    rule_id=rule.id,
                    rule_name=rule.name,
                    passed=True,
                    severity=RuleSeverity.INFO,
                    message="Chamber pressure column not found",
                    total_records=len(df)
                )
            
            pressure = pd.to_numeric(df["chamber_pressure"], errors='coerce')
            valid_pressure = pressure.dropna()
            
            if len(valid_pressure) < 2:
                return BusinessRuleResult(
                    rule_id=rule.id,
                    rule_name=rule.name,
                    passed=True,
                    severity=RuleSeverity.INFO,
                    message="Insufficient valid pressure data",
                    total_records=len(df)
                )
            
            # Calculate pressure stability (coefficient of variation)
            mean_pressure = valid_pressure.mean()
            std_pressure = valid_pressure.std()
            cv = std_pressure / mean_pressure if mean_pressure > 0 else 0
            
            max_cv = rule.parameters.get("max_coefficient_of_variation", 0.1)
            passed = cv <= max_cv
            
            # Find records with pressure outside acceptable range
            target_pressure = rule.parameters.get("target_pressure", mean_pressure)
            tolerance = rule.parameters.get("pressure_tolerance", 0.5)
            lower_bound = target_pressure - tolerance
            upper_bound = target_pressure + tolerance
            
            failed_mask = (pressure < lower_bound) | (pressure > upper_bound)
            failed_records = df[failed_mask].index.tolist()
            
            return BusinessRuleResult(
                rule_id=rule.id,
                rule_name=rule.name,
                passed=passed,
                severity=rule.severity,
                message=f"Pressure stability CV: {cv:.3f} (threshold: {max_cv})",
                affected_records=failed_records,
                total_records=len(df),
                passed_records=len(valid_pressure) - len(failed_records),
                failed_records=len(failed_records),
                confidence_score=1.0 - cv
            )
            
        except Exception as e:
            logger.error(f"Error validating pressure stability: {e}")
            return BusinessRuleResult(
                rule_id=rule.id,
                rule_name=rule.name,
                passed=False,
                severity=RuleSeverity.ERROR,
                message=f"Pressure stability validation error: {e}",
                total_records=len(df)
            )
    
    def _validate_laser_parameter_consistency(self, df: pd.DataFrame, rule: BusinessRule) -> BusinessRuleResult:
        """Validate laser parameter consistency business rule."""
        try:
            laser_columns = ["laser_power", "laser_speed"]
            available_columns = [col for col in laser_columns if col in df.columns]
            
            if not available_columns:
                return BusinessRuleResult(
                    rule_id=rule.id,
                    rule_name=rule.name,
                    passed=True,
                    severity=RuleSeverity.INFO,
                    message="No laser parameter columns found",
                    total_records=len(df)
                )
            
            # Check for parameter consistency within acceptable ranges
            failed_records = []
            total_valid_records = 0
            
            for col in available_columns:
                values = pd.to_numeric(df[col], errors='coerce')
                valid_values = values.dropna()
                total_valid_records += len(valid_values)
                
                # Check for values outside expected range
                min_val = rule.parameters.get(f"{col}_min", valid_values.quantile(0.05))
                max_val = rule.parameters.get(f"{col}_max", valid_values.quantile(0.95))
                
                out_of_range = (values < min_val) | (values > max_val)
                failed_records.extend(df[out_of_range].index.tolist())
            
            # Remove duplicates
            failed_records = list(set(failed_records))
            passed = len(failed_records) == 0
            
            return BusinessRuleResult(
                rule_id=rule.id,
                rule_name=rule.name,
                passed=passed,
                severity=rule.severity,
                message=f"Laser parameter consistency check: {len(failed_records)} violations",
                affected_records=failed_records,
                total_records=len(df),
                passed_records=total_valid_records - len(failed_records),
                failed_records=len(failed_records),
                confidence_score=1.0 - (len(failed_records) / total_valid_records) if total_valid_records > 0 else 1.0
            )
            
        except Exception as e:
            logger.error(f"Error validating laser parameter consistency: {e}")
            return BusinessRuleResult(
                rule_id=rule.id,
                rule_name=rule.name,
                passed=False,
                severity=RuleSeverity.ERROR,
                message=f"Laser parameter consistency validation error: {e}",
                total_records=len(df)
            )
    
    def _validate_build_quality_threshold(self, df: pd.DataFrame, rule: BusinessRule) -> BusinessRuleResult:
        """Validate build quality threshold business rule."""
        try:
            # This rule applies to CT scan and powder bed data
            quality_columns = ["porosity_percentage", "porosity_metric", "num_defects"]
            available_columns = [col for col in quality_columns if col in df.columns]
            
            if not available_columns:
                return BusinessRuleResult(
                    rule_id=rule.id,
                    rule_name=rule.name,
                    passed=True,
                    severity=RuleSeverity.INFO,
                    message="No quality metric columns found",
                    total_records=len(df)
                )
            
            failed_records = []
            total_valid_records = 0
            
            for col in available_columns:
                values = pd.to_numeric(df[col], errors='coerce')
                valid_values = values.dropna()
                total_valid_records += len(valid_values)
                
                # Define quality thresholds
                if "porosity" in col:
                    max_porosity = rule.parameters.get("max_porosity", 5.0)
                    if col == "porosity_percentage":
                        threshold = max_porosity
                    else:  # porosity_metric (0-1 scale)
                        threshold = max_porosity / 100.0
                    
                    quality_violations = values > threshold
                    failed_records.extend(df[quality_violations].index.tolist())
                
                elif col == "num_defects":
                    max_defects = rule.parameters.get("max_defects", 10)
                    defect_violations = values > max_defects
                    failed_records.extend(df[defect_violations].index.tolist())
            
            # Remove duplicates
            failed_records = list(set(failed_records))
            passed = len(failed_records) == 0
            
            return BusinessRuleResult(
                rule_id=rule.id,
                rule_name=rule.name,
                passed=passed,
                severity=rule.severity,
                message=f"Build quality threshold check: {len(failed_records)} violations",
                affected_records=failed_records,
                total_records=len(df),
                passed_records=total_valid_records - len(failed_records),
                failed_records=len(failed_records),
                confidence_score=1.0 - (len(failed_records) / total_valid_records) if total_valid_records > 0 else 1.0
            )
            
        except Exception as e:
            logger.error(f"Error validating build quality threshold: {e}")
            return BusinessRuleResult(
                rule_id=rule.id,
                rule_name=rule.name,
                passed=False,
                severity=RuleSeverity.ERROR,
                message=f"Build quality threshold validation error: {e}",
                total_records=len(df)
            )
    
    def _validate_process_efficiency(self, df: pd.DataFrame, rule: BusinessRule) -> BusinessRuleResult:
        """Validate process efficiency business rule."""
        try:
            # Check for process efficiency indicators
            efficiency_indicators = []
            
            # Temperature efficiency (chamber vs build plate temperature ratio)
            if "chamber_temperature" in df.columns and "build_plate_temperature" in df.columns:
                chamber_temp = pd.to_numeric(df["chamber_temperature"], errors='coerce')
                build_plate_temp = pd.to_numeric(df["build_plate_temperature"], errors='coerce')
                
                valid_mask = chamber_temp.notna() & build_plate_temp.notna()
                temp_ratio = chamber_temp[valid_mask] / build_plate_temp[valid_mask]
                
                # Optimal ratio should be between 1.5 and 3.0
                optimal_ratio_min = rule.parameters.get("optimal_temp_ratio_min", 1.5)
                optimal_ratio_max = rule.parameters.get("optimal_temp_ratio_max", 3.0)
                
                efficiency_indicators.append({
                    "name": "temperature_ratio",
                    "values": temp_ratio,
                    "valid_mask": valid_mask,
                    "optimal_range": (optimal_ratio_min, optimal_ratio_max)
                })
            
            if not efficiency_indicators:
                return BusinessRuleResult(
                    rule_id=rule.id,
                    rule_name=rule.name,
                    passed=True,
                    severity=RuleSeverity.INFO,
                    message="No efficiency indicators found",
                    total_records=len(df)
                )
            
            # Calculate overall efficiency score
            total_efficiency_score = 0.0
            total_indicators = 0
            failed_records = []
            
            for indicator in efficiency_indicators:
                values = indicator["values"]
                valid_mask = indicator["valid_mask"]
                min_val, max_val = indicator["optimal_range"]
                
                # Calculate efficiency score (1.0 = optimal, 0.0 = poor)
                efficiency_scores = np.where(
                    (values >= min_val) & (values <= max_val),
                    1.0,
                    np.maximum(0.0, 1.0 - np.abs(values - (min_val + max_val) / 2) / (max_val - min_val))
                )
                
                total_efficiency_score += efficiency_scores.mean()
                total_indicators += 1
                
                # Find records with poor efficiency
                poor_efficiency = efficiency_scores < 0.5
                failed_records.extend(df[valid_mask][poor_efficiency].index.tolist())
            
            # Remove duplicates
            failed_records = list(set(failed_records))
            
            overall_efficiency = total_efficiency_score / total_indicators if total_indicators > 0 else 0.0
            min_efficiency = rule.parameters.get("min_efficiency", 0.7)
            passed = overall_efficiency >= min_efficiency
            
            return BusinessRuleResult(
                rule_id=rule.id,
                rule_name=rule.name,
                passed=passed,
                severity=rule.severity,
                message=f"Process efficiency: {overall_efficiency:.3f} (threshold: {min_efficiency})",
                affected_records=failed_records,
                total_records=len(df),
                passed_records=len(df) - len(failed_records),
                failed_records=len(failed_records),
                confidence_score=overall_efficiency
            )
            
        except Exception as e:
            logger.error(f"Error validating process efficiency: {e}")
            return BusinessRuleResult(
                rule_id=rule.id,
                rule_name=rule.name,
                passed=False,
                severity=RuleSeverity.ERROR,
                message=f"Process efficiency validation error: {e}",
                total_records=len(df)
            )
    
    def _validate_material_property_compliance(self, df: pd.DataFrame, rule: BusinessRule) -> BusinessRuleResult:
        """Validate material property compliance business rule."""
        try:
            # This is a placeholder for material property compliance
            # In a real system, this would check against material specifications
            
            material_type = rule.parameters.get("material_type", "unknown")
            compliance_threshold = rule.parameters.get("compliance_threshold", 0.95)
            
            # Simulate compliance check
            total_records = len(df)
            compliant_records = int(total_records * compliance_threshold)
            non_compliant_records = total_records - compliant_records
            
            failed_records = list(range(compliant_records, total_records)) if non_compliant_records > 0 else []
            passed = len(failed_records) == 0
            
            return BusinessRuleResult(
                rule_id=rule.id,
                rule_name=rule.name,
                passed=passed,
                severity=rule.severity,
                message=f"Material property compliance for {material_type}: {compliant_records}/{total_records} compliant",
                affected_records=failed_records,
                total_records=total_records,
                passed_records=compliant_records,
                failed_records=non_compliant_records,
                confidence_score=compliance_threshold
            )
            
        except Exception as e:
            logger.error(f"Error validating material property compliance: {e}")
            return BusinessRuleResult(
                rule_id=rule.id,
                rule_name=rule.name,
                passed=False,
                severity=RuleSeverity.ERROR,
                message=f"Material property compliance validation error: {e}",
                total_records=len(df)
            )
    
    def _validate_safety_threshold(self, df: pd.DataFrame, rule: BusinessRule) -> BusinessRuleResult:
        """Validate safety threshold business rule."""
        try:
            safety_columns = ["chamber_temperature", "chamber_pressure", "laser_power"]
            available_columns = [col for col in safety_columns if col in df.columns]
            
            if not available_columns:
                return BusinessRuleResult(
                    rule_id=rule.id,
                    rule_name=rule.name,
                    passed=True,
                    severity=RuleSeverity.INFO,
                    message="No safety parameter columns found",
                    total_records=len(df)
                )
            
            failed_records = []
            total_valid_records = 0
            
            for col in available_columns:
                values = pd.to_numeric(df[col], errors='coerce')
                valid_values = values.dropna()
                total_valid_records += len(valid_values)
                
                # Define safety thresholds
                if col == "chamber_temperature":
                    max_temp = rule.parameters.get("max_chamber_temperature", 800.0)
                    safety_violations = values > max_temp
                elif col == "chamber_pressure":
                    max_pressure = rule.parameters.get("max_chamber_pressure", 8.0)
                    safety_violations = values > max_pressure
                elif col == "laser_power":
                    max_power = rule.parameters.get("max_laser_power", 800.0)
                    safety_violations = values > max_power
                else:
                    continue
                
                failed_records.extend(df[safety_violations].index.tolist())
            
            # Remove duplicates
            failed_records = list(set(failed_records))
            passed = len(failed_records) == 0
            
            return BusinessRuleResult(
                rule_id=rule.id,
                rule_name=rule.name,
                passed=passed,
                severity=RuleSeverity.CRITICAL if not passed else RuleSeverity.INFO,
                message=f"Safety threshold check: {len(failed_records)} violations",
                affected_records=failed_records,
                total_records=len(df),
                passed_records=total_valid_records - len(failed_records),
                failed_records=len(failed_records),
                confidence_score=1.0 - (len(failed_records) / total_valid_records) if total_valid_records > 0 else 1.0
            )
            
        except Exception as e:
            logger.error(f"Error validating safety threshold: {e}")
            return BusinessRuleResult(
                rule_id=rule.id,
                rule_name=rule.name,
                passed=False,
                severity=RuleSeverity.ERROR,
                message=f"Safety threshold validation error: {e}",
                total_records=len(df)
            )
    
    def _calculate_validation_summary(self, source_name: str, results: List[BusinessRuleResult]) -> BusinessRuleValidationSummary:
        """Calculate validation summary from individual results."""
        try:
            total_rules = len(results)
            passed_rules = sum(1 for result in results if result.passed)
            failed_rules = total_rules - passed_rules
            
            critical_violations = sum(1 for result in results 
                                    if not result.passed and result.severity == RuleSeverity.CRITICAL)
            warning_violations = sum(1 for result in results 
                                   if not result.passed and result.severity == RuleSeverity.WARNING)
            
            # Calculate overall compliance score
            if total_rules > 0:
                overall_compliance_score = passed_rules / total_rules
            else:
                overall_compliance_score = 1.0
            
            return BusinessRuleValidationSummary(
                source_name=source_name,
                total_rules=total_rules,
                passed_rules=passed_rules,
                failed_rules=failed_rules,
                critical_violations=critical_violations,
                warning_violations=warning_violations,
                overall_compliance_score=overall_compliance_score,
                results=results
            )
            
        except Exception as e:
            logger.error(f"Error calculating validation summary: {e}")
            return BusinessRuleValidationSummary(
                source_name=source_name,
                total_rules=0,
                passed_rules=0,
                failed_rules=0,
                critical_violations=0,
                warning_violations=0,
                overall_compliance_score=0.0,
                results=[]
            )
    
    def _initialize_business_rules(self):
        """Initialize business rules for different data sources."""
        try:
            # PBF Process Business Rules
            self.business_rules["pbf_temp_correlation"] = BusinessRule(
                id="pbf_temp_correlation",
                name="PBF Temperature Correlation",
                description="Chamber and build plate temperatures should be correlated",
                rule_type=BusinessRuleType.TEMPERATURE_CORRELATION,
                severity=RuleSeverity.WARNING,
                parameters={
                    "applicable_sources": ["pbf_process"],
                    "min_correlation": 0.7
                }
            )
            
            self.business_rules["pbf_pressure_stability"] = BusinessRule(
                id="pbf_pressure_stability",
                name="PBF Pressure Stability",
                description="Chamber pressure should remain stable during process",
                rule_type=BusinessRuleType.PRESSURE_STABILITY,
                severity=RuleSeverity.WARNING,
                parameters={
                    "applicable_sources": ["pbf_process"],
                    "max_coefficient_of_variation": 0.1,
                    "target_pressure": 1.0,
                    "pressure_tolerance": 0.5
                }
            )
            
            self.business_rules["pbf_laser_consistency"] = BusinessRule(
                id="pbf_laser_consistency",
                name="PBF Laser Parameter Consistency",
                description="Laser parameters should be within expected ranges",
                rule_type=BusinessRuleType.LASER_PARAMETER_CONSISTENCY,
                severity=RuleSeverity.WARNING,
                parameters={
                    "applicable_sources": ["pbf_process"],
                    "laser_power_min": 100.0,
                    "laser_power_max": 800.0,
                    "laser_speed_min": 100.0,
                    "laser_speed_max": 5000.0
                }
            )
            
            self.business_rules["pbf_safety_threshold"] = BusinessRule(
                id="pbf_safety_threshold",
                name="PBF Safety Threshold",
                description="Process parameters must not exceed safety limits",
                rule_type=BusinessRuleType.SAFETY_THRESHOLD,
                severity=RuleSeverity.CRITICAL,
                parameters={
                    "applicable_sources": ["pbf_process"],
                    "max_chamber_temperature": 800.0,
                    "max_chamber_pressure": 8.0,
                    "max_laser_power": 800.0
                }
            )
            
            # CT Scan Business Rules
            self.business_rules["ct_build_quality"] = BusinessRule(
                id="ct_build_quality",
                name="CT Scan Build Quality",
                description="Build quality must meet minimum standards",
                rule_type=BusinessRuleType.BUILD_QUALITY_THRESHOLD,
                severity=RuleSeverity.ERROR,
                parameters={
                    "applicable_sources": ["ct_scan"],
                    "max_porosity": 5.0,
                    "max_defects": 10
                }
            )
            
            # Powder Bed Business Rules
            self.business_rules["pb_build_quality"] = BusinessRule(
                id="pb_build_quality",
                name="Powder Bed Build Quality",
                description="Powder bed quality must meet minimum standards",
                rule_type=BusinessRuleType.BUILD_QUALITY_THRESHOLD,
                severity=RuleSeverity.WARNING,
                parameters={
                    "applicable_sources": ["powder_bed"],
                    "max_porosity": 0.3
                }
            )
            
            logger.info(f"Initialized {len(self.business_rules)} business rules")
            
        except Exception as e:
            logger.error(f"Error initializing business rules: {e}")
    
    def add_business_rule(self, rule: BusinessRule) -> bool:
        """Add a new business rule."""
        try:
            self.business_rules[rule.id] = rule
            logger.info(f"Added business rule: {rule.id}")
            return True
        except Exception as e:
            logger.error(f"Error adding business rule {rule.id}: {e}")
            return False
    
    def get_business_rule(self, rule_id: str) -> Optional[BusinessRule]:
        """Get a business rule by ID."""
        return self.business_rules.get(rule_id)
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get a summary of all validation results."""
        try:
            total_sources = len(self.validation_results)
            total_rules = sum(summary.total_rules for summary in self.validation_results.values())
            total_passed = sum(summary.passed_rules for summary in self.validation_results.values())
            total_failed = sum(summary.failed_rules for summary in self.validation_results.values())
            total_critical = sum(summary.critical_violations for summary in self.validation_results.values())
            
            overall_compliance = total_passed / total_rules if total_rules > 0 else 1.0
            
            return {
                "total_sources": total_sources,
                "total_rules": total_rules,
                "total_passed": total_passed,
                "total_failed": total_failed,
                "total_critical_violations": total_critical,
                "overall_compliance_score": overall_compliance,
                "source_summaries": {
                    source: {
                        "compliance_score": summary.overall_compliance_score,
                        "total_rules": summary.total_rules,
                        "passed_rules": summary.passed_rules,
                        "failed_rules": summary.failed_rules,
                        "critical_violations": summary.critical_violations
                    }
                    for source, summary in self.validation_results.items()
                },
                "last_updated": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting validation summary: {e}")
            return {}
