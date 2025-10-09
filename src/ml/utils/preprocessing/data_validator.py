"""
Data Validator

This module implements utilities for data validation and quality assessment
in PBF-LB/M manufacturing processes.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Union, Tuple, Callable
from datetime import datetime, timedelta
import re
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Validation severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationResult:
    """Data structure for validation results."""
    column: str
    rule: str
    severity: ValidationSeverity
    message: str
    invalid_count: int
    invalid_percentage: float
    invalid_indices: List[int]
    invalid_values: List[Any]


class DataValidator:
    """
    Utility class for data validation and quality assessment.
    
    This class handles:
    - Data type validation
    - Range validation
    - Pattern validation
    - Consistency validation
    - Completeness validation
    - Custom validation rules
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the data validator.
        
        Args:
            config: Configuration dictionary with validation settings
        """
        self.config = config or {}
        
        # Built-in validation rules
        self.built_in_rules = {
            'not_null': self._validate_not_null,
            'not_empty': self._validate_not_empty,
            'numeric': self._validate_numeric,
            'integer': self._validate_integer,
            'positive': self._validate_positive,
            'negative': self._validate_negative,
            'range': self._validate_range,
            'min_value': self._validate_min_value,
            'max_value': self._validate_max_value,
            'pattern': self._validate_pattern,
            'email': self._validate_email,
            'phone': self._validate_phone,
            'date': self._validate_date,
            'datetime': self._validate_datetime,
            'unique': self._validate_unique,
            'in_list': self._validate_in_list,
            'length': self._validate_length,
            'min_length': self._validate_min_length,
            'max_length': self._validate_max_length
        }
        
        # Quality thresholds
        self.quality_thresholds = {
            'completeness_threshold': 0.95,  # 95% completeness
            'accuracy_threshold': 0.98,      # 98% accuracy
            'consistency_threshold': 0.95,   # 95% consistency
            'validity_threshold': 0.95       # 95% validity
        }
        
        logger.info("Initialized DataValidator")
    
    def validate_dataframe(self, df: pd.DataFrame, 
                          validation_rules: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Validate DataFrame against specified rules.
        
        Args:
            df: Input DataFrame
            validation_rules: Dictionary of validation rules
            
        Returns:
            Dictionary with validation results
        """
        if df.empty:
            logger.warning("Input DataFrame is empty")
            return {'status': 'empty', 'results': []}
        
        validation_rules = validation_rules or self.config
        validation_results = []
        
        for column, rules in validation_rules.items():
            if column not in df.columns:
                logger.warning(f"Column {column} not found in DataFrame")
                continue
            
            column_results = self._validate_column(df[column], column, rules)
            validation_results.extend(column_results)
        
        # Calculate overall validation status
        overall_status = self._calculate_overall_status(validation_results)
        
        return {
            'status': overall_status,
            'results': validation_results,
            'summary': self._generate_validation_summary(validation_results),
            'recommendations': self._generate_recommendations(validation_results)
        }
    
    def _validate_column(self, series: pd.Series, column: str, rules: Dict[str, Any]) -> List[ValidationResult]:
        """Validate a single column against rules."""
        results = []
        
        for rule_name, rule_config in rules.items():
            if rule_name in self.built_in_rules:
                try:
                    result = self.built_in_rules[rule_name](series, column, rule_config)
                    if result:
                        results.append(result)
                except Exception as e:
                    logger.error(f"Failed to validate {column} with rule {rule_name}: {e}")
                    continue
            else:
                logger.warning(f"Unknown validation rule: {rule_name}")
        
        return results
    
    def _validate_not_null(self, series: pd.Series, column: str, config: Dict[str, Any]) -> Optional[ValidationResult]:
        """Validate that values are not null."""
        null_mask = series.isnull()
        invalid_count = null_mask.sum()
        invalid_percentage = invalid_count / len(series) * 100
        
        if invalid_count > 0:
            severity = ValidationSeverity.ERROR if invalid_percentage > 10 else ValidationSeverity.WARNING
            return ValidationResult(
                column=column,
                rule='not_null',
                severity=severity,
                message=f"Found {invalid_count} null values ({invalid_percentage:.2f}%)",
                invalid_count=invalid_count,
                invalid_percentage=invalid_percentage,
                invalid_indices=series.index[null_mask].tolist(),
                invalid_values=[None] * invalid_count
            )
        return None
    
    def _validate_not_empty(self, series: pd.Series, column: str, config: Dict[str, Any]) -> Optional[ValidationResult]:
        """Validate that string values are not empty."""
        if not pd.api.types.is_string_dtype(series):
            return None
        
        empty_mask = (series == '') | (series.isnull())
        invalid_count = empty_mask.sum()
        invalid_percentage = invalid_count / len(series) * 100
        
        if invalid_count > 0:
            severity = ValidationSeverity.ERROR if invalid_percentage > 10 else ValidationSeverity.WARNING
            return ValidationResult(
                column=column,
                rule='not_empty',
                severity=severity,
                message=f"Found {invalid_count} empty values ({invalid_percentage:.2f}%)",
                invalid_count=invalid_count,
                invalid_percentage=invalid_percentage,
                invalid_indices=series.index[empty_mask].tolist(),
                invalid_values=series[empty_mask].tolist()
            )
        return None
    
    def _validate_numeric(self, series: pd.Series, column: str, config: Dict[str, Any]) -> Optional[ValidationResult]:
        """Validate that values are numeric."""
        if pd.api.types.is_numeric_dtype(series):
            return None
        
        # Try to convert to numeric
        numeric_series = pd.to_numeric(series, errors='coerce')
        invalid_mask = numeric_series.isnull() & series.notnull()
        invalid_count = invalid_mask.sum()
        invalid_percentage = invalid_count / len(series) * 100
        
        if invalid_count > 0:
            severity = ValidationSeverity.ERROR if invalid_percentage > 10 else ValidationSeverity.WARNING
            return ValidationResult(
                column=column,
                rule='numeric',
                severity=severity,
                message=f"Found {invalid_count} non-numeric values ({invalid_percentage:.2f}%)",
                invalid_count=invalid_count,
                invalid_percentage=invalid_percentage,
                invalid_indices=series.index[invalid_mask].tolist(),
                invalid_values=series[invalid_mask].tolist()
            )
        return None
    
    def _validate_integer(self, series: pd.Series, column: str, config: Dict[str, Any]) -> Optional[ValidationResult]:
        """Validate that values are integers."""
        if not pd.api.types.is_numeric_dtype(series):
            return self._validate_numeric(series, column, config)
        
        # Check if values are integers
        integer_mask = series != series.round()
        invalid_count = integer_mask.sum()
        invalid_percentage = invalid_count / len(series) * 100
        
        if invalid_count > 0:
            severity = ValidationSeverity.ERROR if invalid_percentage > 10 else ValidationSeverity.WARNING
            return ValidationResult(
                column=column,
                rule='integer',
                severity=severity,
                message=f"Found {invalid_count} non-integer values ({invalid_percentage:.2f}%)",
                invalid_count=invalid_count,
                invalid_percentage=invalid_percentage,
                invalid_indices=series.index[integer_mask].tolist(),
                invalid_values=series[integer_mask].tolist()
            )
        return None
    
    def _validate_positive(self, series: pd.Series, column: str, config: Dict[str, Any]) -> Optional[ValidationResult]:
        """Validate that values are positive."""
        if not pd.api.types.is_numeric_dtype(series):
            return None
        
        negative_mask = series < 0
        invalid_count = negative_mask.sum()
        invalid_percentage = invalid_count / len(series) * 100
        
        if invalid_count > 0:
            severity = ValidationSeverity.ERROR if invalid_percentage > 10 else ValidationSeverity.WARNING
            return ValidationResult(
                column=column,
                rule='positive',
                severity=severity,
                message=f"Found {invalid_count} negative values ({invalid_percentage:.2f}%)",
                invalid_count=invalid_count,
                invalid_percentage=invalid_percentage,
                invalid_indices=series.index[negative_mask].tolist(),
                invalid_values=series[negative_mask].tolist()
            )
        return None
    
    def _validate_negative(self, series: pd.Series, column: str, config: Dict[str, Any]) -> Optional[ValidationResult]:
        """Validate that values are negative."""
        if not pd.api.types.is_numeric_dtype(series):
            return None
        
        positive_mask = series > 0
        invalid_count = positive_mask.sum()
        invalid_percentage = invalid_count / len(series) * 100
        
        if invalid_count > 0:
            severity = ValidationSeverity.ERROR if invalid_percentage > 10 else ValidationSeverity.WARNING
            return ValidationResult(
                column=column,
                rule='negative',
                severity=severity,
                message=f"Found {invalid_count} positive values ({invalid_percentage:.2f}%)",
                invalid_count=invalid_count,
                invalid_percentage=invalid_percentage,
                invalid_indices=series.index[positive_mask].tolist(),
                invalid_values=series[positive_mask].tolist()
            )
        return None
    
    def _validate_range(self, series: pd.Series, column: str, config: Dict[str, Any]) -> Optional[ValidationResult]:
        """Validate that values are within a specified range."""
        if not pd.api.types.is_numeric_dtype(series):
            return None
        
        min_val = config.get('min_value')
        max_val = config.get('max_value')
        
        if min_val is None and max_val is None:
            return None
        
        invalid_mask = pd.Series(False, index=series.index)
        
        if min_val is not None:
            invalid_mask |= series < min_val
        if max_val is not None:
            invalid_mask |= series > max_val
        
        invalid_count = invalid_mask.sum()
        invalid_percentage = invalid_count / len(series) * 100
        
        if invalid_count > 0:
            severity = ValidationSeverity.ERROR if invalid_percentage > 10 else ValidationSeverity.WARNING
            range_str = f"[{min_val}, {max_val}]" if min_val is not None and max_val is not None else f">= {min_val}" if min_val is not None else f"<= {max_val}"
            return ValidationResult(
                column=column,
                rule='range',
                severity=severity,
                message=f"Found {invalid_count} values outside range {range_str} ({invalid_percentage:.2f}%)",
                invalid_count=invalid_count,
                invalid_percentage=invalid_percentage,
                invalid_indices=series.index[invalid_mask].tolist(),
                invalid_values=series[invalid_mask].tolist()
            )
        return None
    
    def _validate_min_value(self, series: pd.Series, column: str, config: Dict[str, Any]) -> Optional[ValidationResult]:
        """Validate that values are above minimum value."""
        min_val = config.get('value')
        if min_val is None:
            return None
        
        return self._validate_range(series, column, {'min_value': min_val})
    
    def _validate_max_value(self, series: pd.Series, column: str, config: Dict[str, Any]) -> Optional[ValidationResult]:
        """Validate that values are below maximum value."""
        max_val = config.get('value')
        if max_val is None:
            return None
        
        return self._validate_range(series, column, {'max_value': max_val})
    
    def _validate_pattern(self, series: pd.Series, column: str, config: Dict[str, Any]) -> Optional[ValidationResult]:
        """Validate that values match a specified pattern."""
        pattern = config.get('pattern')
        if pattern is None:
            return None
        
        if not pd.api.types.is_string_dtype(series):
            return None
        
        try:
            regex = re.compile(pattern)
            invalid_mask = ~series.astype(str).str.match(regex, na=False)
            invalid_count = invalid_mask.sum()
            invalid_percentage = invalid_count / len(series) * 100
            
            if invalid_count > 0:
                severity = ValidationSeverity.ERROR if invalid_percentage > 10 else ValidationSeverity.WARNING
                return ValidationResult(
                    column=column,
                    rule='pattern',
                    severity=severity,
                    message=f"Found {invalid_count} values not matching pattern '{pattern}' ({invalid_percentage:.2f}%)",
                    invalid_count=invalid_count,
                    invalid_percentage=invalid_percentage,
                    invalid_indices=series.index[invalid_mask].tolist(),
                    invalid_values=series[invalid_mask].tolist()
                )
        except re.error as e:
            logger.error(f"Invalid regex pattern '{pattern}': {e}")
        
        return None
    
    def _validate_email(self, series: pd.Series, column: str, config: Dict[str, Any]) -> Optional[ValidationResult]:
        """Validate that values are valid email addresses."""
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return self._validate_pattern(series, column, {'pattern': email_pattern})
    
    def _validate_phone(self, series: pd.Series, column: str, config: Dict[str, Any]) -> Optional[ValidationResult]:
        """Validate that values are valid phone numbers."""
        phone_pattern = r'^\+?1?-?\.?\s?\(?(\d{3})\)?[-.\s]?(\d{3})[-.\s]?(\d{4})$'
        return self._validate_pattern(series, column, {'pattern': phone_pattern})
    
    def _validate_date(self, series: pd.Series, column: str, config: Dict[str, Any]) -> Optional[ValidationResult]:
        """Validate that values are valid dates."""
        if pd.api.types.is_datetime64_any_dtype(series):
            return None
        
        try:
            date_format = config.get('format')
            if date_format:
                pd.to_datetime(series, format=date_format, errors='raise')
            else:
                pd.to_datetime(series, errors='raise')
        except (ValueError, TypeError) as e:
            # Count invalid dates
            invalid_count = 0
            invalid_indices = []
            invalid_values = []
            
            for idx, value in series.items():
                try:
                    if date_format:
                        pd.to_datetime(value, format=date_format, errors='raise')
                    else:
                        pd.to_datetime(value, errors='raise')
                except (ValueError, TypeError):
                    invalid_count += 1
                    invalid_indices.append(idx)
                    invalid_values.append(value)
            
            invalid_percentage = invalid_count / len(series) * 100
            severity = ValidationSeverity.ERROR if invalid_percentage > 10 else ValidationSeverity.WARNING
            
            return ValidationResult(
                column=column,
                rule='date',
                severity=severity,
                message=f"Found {invalid_count} invalid date values ({invalid_percentage:.2f}%)",
                invalid_count=invalid_count,
                invalid_percentage=invalid_percentage,
                invalid_indices=invalid_indices,
                invalid_values=invalid_values
            )
        
        return None
    
    def _validate_datetime(self, series: pd.Series, column: str, config: Dict[str, Any]) -> Optional[ValidationResult]:
        """Validate that values are valid datetime values."""
        return self._validate_date(series, column, config)
    
    def _validate_unique(self, series: pd.Series, column: str, config: Dict[str, Any]) -> Optional[ValidationResult]:
        """Validate that values are unique."""
        duplicate_mask = series.duplicated()
        invalid_count = duplicate_mask.sum()
        invalid_percentage = invalid_count / len(series) * 100
        
        if invalid_count > 0:
            severity = ValidationSeverity.ERROR if invalid_percentage > 10 else ValidationSeverity.WARNING
            return ValidationResult(
                column=column,
                rule='unique',
                severity=severity,
                message=f"Found {invalid_count} duplicate values ({invalid_percentage:.2f}%)",
                invalid_count=invalid_count,
                invalid_percentage=invalid_percentage,
                invalid_indices=series.index[duplicate_mask].tolist(),
                invalid_values=series[duplicate_mask].tolist()
            )
        return None
    
    def _validate_in_list(self, series: pd.Series, column: str, config: Dict[str, Any]) -> Optional[ValidationResult]:
        """Validate that values are in a specified list."""
        allowed_values = config.get('values', [])
        if not allowed_values:
            return None
        
        invalid_mask = ~series.isin(allowed_values)
        invalid_count = invalid_mask.sum()
        invalid_percentage = invalid_count / len(series) * 100
        
        if invalid_count > 0:
            severity = ValidationSeverity.ERROR if invalid_percentage > 10 else ValidationSeverity.WARNING
            return ValidationResult(
                column=column,
                rule='in_list',
                severity=severity,
                message=f"Found {invalid_count} values not in allowed list ({invalid_percentage:.2f}%)",
                invalid_count=invalid_count,
                invalid_percentage=invalid_percentage,
                invalid_indices=series.index[invalid_mask].tolist(),
                invalid_values=series[invalid_mask].tolist()
            )
        return None
    
    def _validate_length(self, series: pd.Series, column: str, config: Dict[str, Any]) -> Optional[ValidationResult]:
        """Validate that string values have a specific length."""
        if not pd.api.types.is_string_dtype(series):
            return None
        
        expected_length = config.get('length')
        if expected_length is None:
            return None
        
        length_mask = series.astype(str).str.len() != expected_length
        invalid_count = length_mask.sum()
        invalid_percentage = invalid_count / len(series) * 100
        
        if invalid_count > 0:
            severity = ValidationSeverity.ERROR if invalid_percentage > 10 else ValidationSeverity.WARNING
            return ValidationResult(
                column=column,
                rule='length',
                severity=severity,
                message=f"Found {invalid_count} values with length != {expected_length} ({invalid_percentage:.2f}%)",
                invalid_count=invalid_count,
                invalid_percentage=invalid_percentage,
                invalid_indices=series.index[length_mask].tolist(),
                invalid_values=series[length_mask].tolist()
            )
        return None
    
    def _validate_min_length(self, series: pd.Series, column: str, config: Dict[str, Any]) -> Optional[ValidationResult]:
        """Validate that string values have minimum length."""
        if not pd.api.types.is_string_dtype(series):
            return None
        
        min_length = config.get('length')
        if min_length is None:
            return None
        
        length_mask = series.astype(str).str.len() < min_length
        invalid_count = length_mask.sum()
        invalid_percentage = invalid_count / len(series) * 100
        
        if invalid_count > 0:
            severity = ValidationSeverity.ERROR if invalid_percentage > 10 else ValidationSeverity.WARNING
            return ValidationResult(
                column=column,
                rule='min_length',
                severity=severity,
                message=f"Found {invalid_count} values with length < {min_length} ({invalid_percentage:.2f}%)",
                invalid_count=invalid_count,
                invalid_percentage=invalid_percentage,
                invalid_indices=series.index[length_mask].tolist(),
                invalid_values=series[length_mask].tolist()
            )
        return None
    
    def _validate_max_length(self, series: pd.Series, column: str, config: Dict[str, Any]) -> Optional[ValidationResult]:
        """Validate that string values have maximum length."""
        if not pd.api.types.is_string_dtype(series):
            return None
        
        max_length = config.get('length')
        if max_length is None:
            return None
        
        length_mask = series.astype(str).str.len() > max_length
        invalid_count = length_mask.sum()
        invalid_percentage = invalid_count / len(series) * 100
        
        if invalid_count > 0:
            severity = ValidationSeverity.ERROR if invalid_percentage > 10 else ValidationSeverity.WARNING
            return ValidationResult(
                column=column,
                rule='max_length',
                severity=severity,
                message=f"Found {invalid_count} values with length > {max_length} ({invalid_percentage:.2f}%)",
                invalid_count=invalid_count,
                invalid_percentage=invalid_percentage,
                invalid_indices=series.index[length_mask].tolist(),
                invalid_values=series[length_mask].tolist()
            )
        return None
    
    def _calculate_overall_status(self, validation_results: List[ValidationResult]) -> str:
        """Calculate overall validation status."""
        if not validation_results:
            return 'valid'
        
        # Check for critical errors
        critical_errors = [r for r in validation_results if r.severity == ValidationSeverity.CRITICAL]
        if critical_errors:
            return 'critical'
        
        # Check for errors
        errors = [r for r in validation_results if r.severity == ValidationSeverity.ERROR]
        if errors:
            return 'error'
        
        # Check for warnings
        warnings = [r for r in validation_results if r.severity == ValidationSeverity.WARNING]
        if warnings:
            return 'warning'
        
        return 'valid'
    
    def _generate_validation_summary(self, validation_results: List[ValidationResult]) -> Dict[str, Any]:
        """Generate validation summary."""
        summary = {
            'total_rules_checked': len(validation_results),
            'rules_passed': len([r for r in validation_results if r.invalid_count == 0]),
            'rules_failed': len([r for r in validation_results if r.invalid_count > 0]),
            'severity_counts': {
                'critical': len([r for r in validation_results if r.severity == ValidationSeverity.CRITICAL]),
                'error': len([r for r in validation_results if r.severity == ValidationSeverity.ERROR]),
                'warning': len([r for r in validation_results if r.severity == ValidationSeverity.WARNING]),
                'info': len([r for r in validation_results if r.severity == ValidationSeverity.INFO])
            },
            'total_invalid_values': sum(r.invalid_count for r in validation_results),
            'columns_with_issues': list(set(r.column for r in validation_results if r.invalid_count > 0))
        }
        
        return summary
    
    def _generate_recommendations(self, validation_results: List[ValidationResult]) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        # Group results by severity
        critical_issues = [r for r in validation_results if r.severity == ValidationSeverity.CRITICAL]
        error_issues = [r for r in validation_results if r.severity == ValidationSeverity.ERROR]
        warning_issues = [r for r in validation_results if r.severity == ValidationSeverity.WARNING]
        
        if critical_issues:
            recommendations.append("CRITICAL: Immediate action required for critical data quality issues.")
        
        if error_issues:
            recommendations.append("ERROR: Data quality issues detected that need attention.")
        
        if warning_issues:
            recommendations.append("WARNING: Minor data quality issues detected.")
        
        # Specific recommendations
        null_issues = [r for r in validation_results if r.rule == 'not_null' and r.invalid_count > 0]
        if null_issues:
            recommendations.append("Consider implementing data imputation strategies for missing values.")
        
        range_issues = [r for r in validation_results if r.rule == 'range' and r.invalid_count > 0]
        if range_issues:
            recommendations.append("Review data collection processes for out-of-range values.")
        
        duplicate_issues = [r for r in validation_results if r.rule == 'unique' and r.invalid_count > 0]
        if duplicate_issues:
            recommendations.append("Implement deduplication processes.")
        
        return recommendations
    
    def add_custom_rule(self, rule_name: str, rule_function: Callable):
        """
        Add a custom validation rule.
        
        Args:
            rule_name: Name of the rule
            rule_function: Function that implements the rule
        """
        self.built_in_rules[rule_name] = rule_function
        logger.info(f"Added custom validation rule: {rule_name}")
    
    def validate_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform comprehensive data quality assessment.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with data quality metrics
        """
        quality_report = {
            'completeness': self._assess_completeness(df),
            'accuracy': self._assess_accuracy(df),
            'consistency': self._assess_consistency(df),
            'validity': self._assess_validity(df),
            'overall_score': 0.0
        }
        
        # Calculate overall quality score
        scores = [quality_report['completeness'], quality_report['accuracy'], 
                 quality_report['consistency'], quality_report['validity']]
        quality_report['overall_score'] = np.mean(scores)
        
        return quality_report
    
    def _assess_completeness(self, df: pd.DataFrame) -> float:
        """Assess data completeness."""
        total_cells = df.size
        missing_cells = df.isnull().sum().sum()
        completeness = (total_cells - missing_cells) / total_cells
        return completeness
    
    def _assess_accuracy(self, df: pd.DataFrame) -> float:
        """Assess data accuracy (placeholder implementation)."""
        # This would typically involve comparing with ground truth data
        # For now, return a default score
        return 0.95
    
    def _assess_consistency(self, df: pd.DataFrame) -> float:
        """Assess data consistency."""
        # Check for duplicate rows
        duplicate_rows = df.duplicated().sum()
        total_rows = len(df)
        consistency = 1 - (duplicate_rows / total_rows)
        return consistency
    
    def _assess_validity(self, df: pd.DataFrame) -> float:
        """Assess data validity."""
        # This would typically involve checking against business rules
        # For now, return a default score
        return 0.95
    
    def get_validation_report(self, validation_results: Dict[str, Any]) -> str:
        """
        Generate a human-readable validation report.
        
        Args:
            validation_results: Results from validation
            
        Returns:
            Formatted validation report
        """
        report = []
        report.append("=" * 50)
        report.append("DATA VALIDATION REPORT")
        report.append("=" * 50)
        report.append("")
        
        # Overall status
        status = validation_results.get('status', 'unknown')
        report.append(f"Overall Status: {status.upper()}")
        report.append("")
        
        # Summary
        summary = validation_results.get('summary', {})
        report.append("SUMMARY:")
        report.append(f"  Total Rules Checked: {summary.get('total_rules_checked', 0)}")
        report.append(f"  Rules Passed: {summary.get('rules_passed', 0)}")
        report.append(f"  Rules Failed: {summary.get('rules_failed', 0)}")
        report.append(f"  Total Invalid Values: {summary.get('total_invalid_values', 0)}")
        report.append("")
        
        # Severity breakdown
        severity_counts = summary.get('severity_counts', {})
        report.append("SEVERITY BREAKDOWN:")
        for severity, count in severity_counts.items():
            report.append(f"  {severity.upper()}: {count}")
        report.append("")
        
        # Detailed results
        results = validation_results.get('results', [])
        if results:
            report.append("DETAILED RESULTS:")
            for result in results:
                report.append(f"  Column: {result.column}")
                report.append(f"  Rule: {result.rule}")
                report.append(f"  Severity: {result.severity.value}")
                report.append(f"  Message: {result.message}")
                report.append("")
        
        # Recommendations
        recommendations = validation_results.get('recommendations', [])
        if recommendations:
            report.append("RECOMMENDATIONS:")
            for i, rec in enumerate(recommendations, 1):
                report.append(f"  {i}. {rec}")
            report.append("")
        
        report.append("=" * 50)
        
        return "\n".join(report)
