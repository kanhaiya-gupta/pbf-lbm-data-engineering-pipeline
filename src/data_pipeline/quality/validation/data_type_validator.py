"""
Data Type Validator

This module provides data type validation capabilities for the PBF-LB/M data pipeline.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import re

from src.data_pipeline.config.pipeline_config import get_pipeline_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataType(Enum):
    """Data type enumeration."""
    INTEGER = "integer"
    FLOAT = "float"
    STRING = "string"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    DATE = "date"
    TIME = "time"
    EMAIL = "email"
    URL = "url"
    UUID = "uuid"
    JSON = "json"
    ARRAY = "array"
    OBJECT = "object"

class ValidationSeverity(Enum):
    """Validation severity enumeration."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class DataTypeRule:
    """Data type validation rule data class."""
    column_name: str
    expected_type: DataType
    nullable: bool = True
    format_pattern: Optional[str] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    allowed_values: Optional[List[Any]] = None
    custom_validator: Optional[callable] = None

@dataclass
class DataTypeValidationResult:
    """Data type validation result data class."""
    column_name: str
    expected_type: DataType
    actual_type: str
    is_valid: bool
    severity: ValidationSeverity
    message: str
    total_records: int = 0
    valid_records: int = 0
    invalid_records: int = 0
    null_records: int = 0
    type_mismatches: int = 0
    format_errors: int = 0
    range_violations: int = 0
    invalid_values: List[Any] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class DataTypeValidationSummary:
    """Data type validation summary data class."""
    source_name: str
    total_columns: int
    valid_columns: int
    invalid_columns: int
    overall_score: float
    results: List[DataTypeValidationResult] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

class DataTypeValidator:
    """
    Data type validation service for PBF-LB/M data pipeline.
    """
    
    def __init__(self):
        self.config = get_pipeline_config()
        self.validation_rules: Dict[str, List[DataTypeRule]] = {}
        self.validation_results: Dict[str, DataTypeValidationSummary] = {}
        
        # Initialize validation rules
        self._initialize_validation_rules()
        
    def validate_pbf_process_data_types(self, data: List[Dict[str, Any]]) -> DataTypeValidationSummary:
        """
        Validate PBF process data types.
        
        Args:
            data: List of PBF process data records
            
        Returns:
            DataTypeValidationSummary: Data type validation results
        """
        try:
            logger.info(f"Validating PBF process data types for {len(data)} records")
            
            df = pd.DataFrame(data)
            rules = self.validation_rules.get("pbf_process", [])
            
            results = []
            for rule in rules:
                if rule.column_name in df.columns:
                    result = self._validate_column_data_type(df, rule)
                    results.append(result)
                else:
                    # Column not found
                    result = DataTypeValidationResult(
                        column_name=rule.column_name,
                        expected_type=rule.expected_type,
                        actual_type="missing",
                        is_valid=False,
                        severity=ValidationSeverity.ERROR,
                        message=f"Column '{rule.column_name}' not found in data",
                        total_records=len(df)
                    )
                    results.append(result)
            
            summary = self._calculate_validation_summary("pbf_process", results)
            self.validation_results["pbf_process"] = summary
            
            logger.info(f"PBF process data type validation completed. Overall score: {summary.overall_score:.2f}")
            return summary
            
        except Exception as e:
            logger.error(f"Error validating PBF process data types: {e}")
            raise
    
    def validate_ispm_monitoring_data_types(self, data: List[Dict[str, Any]]) -> DataTypeValidationSummary:
        """
        Validate ISPM monitoring data types.
        
        Args:
            data: List of ISPM monitoring data records
            
        Returns:
            DataTypeValidationSummary: Data type validation results
        """
        try:
            logger.info(f"Validating ISPM monitoring data types for {len(data)} records")
            
            df = pd.DataFrame(data)
            rules = self.validation_rules.get("ispm_monitoring", [])
            
            results = []
            for rule in rules:
                if rule.column_name in df.columns:
                    result = self._validate_column_data_type(df, rule)
                    results.append(result)
                else:
                    result = DataTypeValidationResult(
                        column_name=rule.column_name,
                        expected_type=rule.expected_type,
                        actual_type="missing",
                        is_valid=False,
                        severity=ValidationSeverity.ERROR,
                        message=f"Column '{rule.column_name}' not found in data",
                        total_records=len(df)
                    )
                    results.append(result)
            
            summary = self._calculate_validation_summary("ispm_monitoring", results)
            self.validation_results["ispm_monitoring"] = summary
            
            logger.info(f"ISPM monitoring data type validation completed. Overall score: {summary.overall_score:.2f}")
            return summary
            
        except Exception as e:
            logger.error(f"Error validating ISPM monitoring data types: {e}")
            raise
    
    def validate_ct_scan_data_types(self, data: List[Dict[str, Any]]) -> DataTypeValidationSummary:
        """
        Validate CT scan data types.
        
        Args:
            data: List of CT scan data records
            
        Returns:
            DataTypeValidationSummary: Data type validation results
        """
        try:
            logger.info(f"Validating CT scan data types for {len(data)} records")
            
            df = pd.DataFrame(data)
            rules = self.validation_rules.get("ct_scan", [])
            
            results = []
            for rule in rules:
                if rule.column_name in df.columns:
                    result = self._validate_column_data_type(df, rule)
                    results.append(result)
                else:
                    result = DataTypeValidationResult(
                        column_name=rule.column_name,
                        expected_type=rule.expected_type,
                        actual_type="missing",
                        is_valid=False,
                        severity=ValidationSeverity.ERROR,
                        message=f"Column '{rule.column_name}' not found in data",
                        total_records=len(df)
                    )
                    results.append(result)
            
            summary = self._calculate_validation_summary("ct_scan", results)
            self.validation_results["ct_scan"] = summary
            
            logger.info(f"CT scan data type validation completed. Overall score: {summary.overall_score:.2f}")
            return summary
            
        except Exception as e:
            logger.error(f"Error validating CT scan data types: {e}")
            raise
    
    def validate_powder_bed_data_types(self, data: List[Dict[str, Any]]) -> DataTypeValidationSummary:
        """
        Validate powder bed data types.
        
        Args:
            data: List of powder bed data records
            
        Returns:
            DataTypeValidationSummary: Data type validation results
        """
        try:
            logger.info(f"Validating powder bed data types for {len(data)} records")
            
            df = pd.DataFrame(data)
            rules = self.validation_rules.get("powder_bed", [])
            
            results = []
            for rule in rules:
                if rule.column_name in df.columns:
                    result = self._validate_column_data_type(df, rule)
                    results.append(result)
                else:
                    result = DataTypeValidationResult(
                        column_name=rule.column_name,
                        expected_type=rule.expected_type,
                        actual_type="missing",
                        is_valid=False,
                        severity=ValidationSeverity.ERROR,
                        message=f"Column '{rule.column_name}' not found in data",
                        total_records=len(df)
                    )
                    results.append(result)
            
            summary = self._calculate_validation_summary("powder_bed", results)
            self.validation_results["powder_bed"] = summary
            
            logger.info(f"Powder bed data type validation completed. Overall score: {summary.overall_score:.2f}")
            return summary
            
        except Exception as e:
            logger.error(f"Error validating powder bed data types: {e}")
            raise
    
    def _validate_column_data_type(self, df: pd.DataFrame, rule: DataTypeRule) -> DataTypeValidationResult:
        """Validate a single column against its data type rule."""
        try:
            column = df[rule.column_name]
            total_records = len(column)
            
            # Count null values
            null_records = column.isnull().sum()
            non_null_records = total_records - null_records
            
            # Check if nulls are allowed
            if null_records > 0 and not rule.nullable:
                return DataTypeValidationResult(
                    column_name=rule.column_name,
                    expected_type=rule.expected_type,
                    actual_type="nullable",
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    message=f"Column '{rule.column_name}' contains {null_records} null values but nulls are not allowed",
                    total_records=total_records,
                    null_records=null_records
                )
            
            # If all values are null and nulls are allowed, consider it valid
            if non_null_records == 0 and rule.nullable:
                return DataTypeValidationResult(
                    column_name=rule.column_name,
                    expected_type=rule.expected_type,
                    actual_type="all_null",
                    is_valid=True,
                    severity=ValidationSeverity.INFO,
                    message=f"Column '{rule.column_name}' contains only null values (allowed)",
                    total_records=total_records,
                    null_records=null_records
                )
            
            # Validate non-null values
            non_null_column = column.dropna()
            validation_result = self._validate_data_type(non_null_column, rule)
            
            # Update counts
            validation_result.total_records = total_records
            validation_result.null_records = null_records
            validation_result.valid_records = validation_result.valid_records + null_records if rule.nullable else validation_result.valid_records
            validation_result.invalid_records = total_records - validation_result.valid_records
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Error validating column {rule.column_name}: {e}")
            return DataTypeValidationResult(
                column_name=rule.column_name,
                expected_type=rule.expected_type,
                actual_type="error",
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"Validation error: {e}",
                total_records=len(df)
            )
    
    def _validate_data_type(self, column: pd.Series, rule: DataTypeRule) -> DataTypeValidationResult:
        """Validate data type for a non-null column."""
        try:
            total_records = len(column)
            valid_records = 0
            type_mismatches = 0
            format_errors = 0
            range_violations = 0
            invalid_values = []
            
            # Determine actual data type
            actual_type = str(column.dtype)
            
            # Validate based on expected type
            if rule.expected_type == DataType.INTEGER:
                valid_records, type_mismatches, range_violations, invalid_values = self._validate_integer_type(column, rule)
            elif rule.expected_type == DataType.FLOAT:
                valid_records, type_mismatches, range_violations, invalid_values = self._validate_float_type(column, rule)
            elif rule.expected_type == DataType.STRING:
                valid_records, type_mismatches, format_errors, invalid_values = self._validate_string_type(column, rule)
            elif rule.expected_type == DataType.BOOLEAN:
                valid_records, type_mismatches, invalid_values = self._validate_boolean_type(column, rule)
            elif rule.expected_type == DataType.DATETIME:
                valid_records, type_mismatches, format_errors, invalid_values = self._validate_datetime_type(column, rule)
            elif rule.expected_type == DataType.DATE:
                valid_records, type_mismatches, format_errors, invalid_values = self._validate_date_type(column, rule)
            elif rule.expected_type == DataType.TIME:
                valid_records, type_mismatches, format_errors, invalid_values = self._validate_time_type(column, rule)
            elif rule.expected_type == DataType.EMAIL:
                valid_records, type_mismatches, format_errors, invalid_values = self._validate_email_type(column, rule)
            elif rule.expected_type == DataType.URL:
                valid_records, type_mismatches, format_errors, invalid_values = self._validate_url_type(column, rule)
            elif rule.expected_type == DataType.UUID:
                valid_records, type_mismatches, format_errors, invalid_values = self._validate_uuid_type(column, rule)
            elif rule.expected_type == DataType.JSON:
                valid_records, type_mismatches, format_errors, invalid_values = self._validate_json_type(column, rule)
            else:
                # Unknown type
                valid_records = 0
                type_mismatches = total_records
                invalid_values = column.tolist()
            
            # Determine severity
            if type_mismatches > 0 or format_errors > 0:
                severity = ValidationSeverity.ERROR
            elif range_violations > 0:
                severity = ValidationSeverity.WARNING
            else:
                severity = ValidationSeverity.INFO
            
            # Determine if valid
            is_valid = (valid_records == total_records and 
                       type_mismatches == 0 and 
                       format_errors == 0 and 
                       range_violations == 0)
            
            # Create message
            message_parts = []
            if type_mismatches > 0:
                message_parts.append(f"{type_mismatches} type mismatches")
            if format_errors > 0:
                message_parts.append(f"{format_errors} format errors")
            if range_violations > 0:
                message_parts.append(f"{range_violations} range violations")
            
            if message_parts:
                message = f"Column '{rule.column_name}': {', '.join(message_parts)}"
            else:
                message = f"Column '{rule.column_name}': All {valid_records} records valid"
            
            return DataTypeValidationResult(
                column_name=rule.column_name,
                expected_type=rule.expected_type,
                actual_type=actual_type,
                is_valid=is_valid,
                severity=severity,
                message=message,
                valid_records=valid_records,
                type_mismatches=type_mismatches,
                format_errors=format_errors,
                range_violations=range_violations,
                invalid_values=invalid_values[:10]  # Limit to first 10 invalid values
            )
            
        except Exception as e:
            logger.error(f"Error validating data type for column {rule.column_name}: {e}")
            return DataTypeValidationResult(
                column_name=rule.column_name,
                expected_type=rule.expected_type,
                actual_type="error",
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"Data type validation error: {e}",
                total_records=len(column)
            )
    
    def _validate_integer_type(self, column: pd.Series, rule: DataTypeRule) -> tuple[int, int, int, List[Any]]:
        """Validate integer data type."""
        try:
            total_records = len(column)
            valid_records = 0
            type_mismatches = 0
            range_violations = 0
            invalid_values = []
            
            for idx, value in column.items():
                try:
                    # Try to convert to integer
                    int_value = int(float(value))  # Handle "123.0" -> 123
                    
                    # Check range
                    if rule.min_value is not None and int_value < rule.min_value:
                        range_violations += 1
                        invalid_values.append(value)
                    elif rule.max_value is not None and int_value > rule.max_value:
                        range_violations += 1
                        invalid_values.append(value)
                    else:
                        valid_records += 1
                        
                except (ValueError, TypeError):
                    type_mismatches += 1
                    invalid_values.append(value)
            
            return valid_records, type_mismatches, range_violations, invalid_values
            
        except Exception as e:
            logger.error(f"Error validating integer type: {e}")
            return 0, len(column), 0, column.tolist()
    
    def _validate_float_type(self, column: pd.Series, rule: DataTypeRule) -> tuple[int, int, int, List[Any]]:
        """Validate float data type."""
        try:
            total_records = len(column)
            valid_records = 0
            type_mismatches = 0
            range_violations = 0
            invalid_values = []
            
            for idx, value in column.items():
                try:
                    # Try to convert to float
                    float_value = float(value)
                    
                    # Check range
                    if rule.min_value is not None and float_value < rule.min_value:
                        range_violations += 1
                        invalid_values.append(value)
                    elif rule.max_value is not None and float_value > rule.max_value:
                        range_violations += 1
                        invalid_values.append(value)
                    else:
                        valid_records += 1
                        
                except (ValueError, TypeError):
                    type_mismatches += 1
                    invalid_values.append(value)
            
            return valid_records, type_mismatches, range_violations, invalid_values
            
        except Exception as e:
            logger.error(f"Error validating float type: {e}")
            return 0, len(column), 0, column.tolist()
    
    def _validate_string_type(self, column: pd.Series, rule: DataTypeRule) -> tuple[int, int, int, List[Any]]:
        """Validate string data type."""
        try:
            total_records = len(column)
            valid_records = 0
            type_mismatches = 0
            format_errors = 0
            invalid_values = []
            
            for idx, value in column.items():
                try:
                    # Convert to string
                    str_value = str(value)
                    
                    # Check length constraints
                    if rule.min_length is not None and len(str_value) < rule.min_length:
                        format_errors += 1
                        invalid_values.append(value)
                    elif rule.max_length is not None and len(str_value) > rule.max_length:
                        format_errors += 1
                        invalid_values.append(value)
                    # Check format pattern
                    elif rule.format_pattern is not None and not re.match(rule.format_pattern, str_value):
                        format_errors += 1
                        invalid_values.append(value)
                    # Check allowed values
                    elif rule.allowed_values is not None and str_value not in rule.allowed_values:
                        format_errors += 1
                        invalid_values.append(value)
                    else:
                        valid_records += 1
                        
                except Exception:
                    type_mismatches += 1
                    invalid_values.append(value)
            
            return valid_records, type_mismatches, format_errors, invalid_values
            
        except Exception as e:
            logger.error(f"Error validating string type: {e}")
            return 0, len(column), 0, column.tolist()
    
    def _validate_boolean_type(self, column: pd.Series, rule: DataTypeRule) -> tuple[int, int, List[Any]]:
        """Validate boolean data type."""
        try:
            total_records = len(column)
            valid_records = 0
            type_mismatches = 0
            invalid_values = []
            
            for idx, value in column.items():
                try:
                    # Convert to boolean
                    if isinstance(value, bool):
                        valid_records += 1
                    elif isinstance(value, str):
                        if value.lower() in ['true', 'false', '1', '0', 'yes', 'no']:
                            valid_records += 1
                        else:
                            type_mismatches += 1
                            invalid_values.append(value)
                    elif isinstance(value, (int, float)):
                        if value in [0, 1]:
                            valid_records += 1
                        else:
                            type_mismatches += 1
                            invalid_values.append(value)
                    else:
                        type_mismatches += 1
                        invalid_values.append(value)
                        
                except Exception:
                    type_mismatches += 1
                    invalid_values.append(value)
            
            return valid_records, type_mismatches, invalid_values
            
        except Exception as e:
            logger.error(f"Error validating boolean type: {e}")
            return 0, len(column), column.tolist()
    
    def _validate_datetime_type(self, column: pd.Series, rule: DataTypeRule) -> tuple[int, int, int, List[Any]]:
        """Validate datetime data type."""
        try:
            total_records = len(column)
            valid_records = 0
            type_mismatches = 0
            format_errors = 0
            invalid_values = []
            
            for idx, value in column.items():
                try:
                    # Try to parse as datetime
                    if isinstance(value, datetime):
                        valid_records += 1
                    else:
                        pd.to_datetime(value)
                        valid_records += 1
                        
                except (ValueError, TypeError):
                    type_mismatches += 1
                    invalid_values.append(value)
            
            return valid_records, type_mismatches, format_errors, invalid_values
            
        except Exception as e:
            logger.error(f"Error validating datetime type: {e}")
            return 0, len(column), 0, column.tolist()
    
    def _validate_date_type(self, column: pd.Series, rule: DataTypeRule) -> tuple[int, int, int, List[Any]]:
        """Validate date data type."""
        try:
            total_records = len(column)
            valid_records = 0
            type_mismatches = 0
            format_errors = 0
            invalid_values = []
            
            for idx, value in column.items():
                try:
                    # Try to parse as date
                    if isinstance(value, datetime):
                        valid_records += 1
                    else:
                        pd.to_datetime(value).date()
                        valid_records += 1
                        
                except (ValueError, TypeError):
                    type_mismatches += 1
                    invalid_values.append(value)
            
            return valid_records, type_mismatches, format_errors, invalid_values
            
        except Exception as e:
            logger.error(f"Error validating date type: {e}")
            return 0, len(column), 0, column.tolist()
    
    def _validate_time_type(self, column: pd.Series, rule: DataTypeRule) -> tuple[int, int, int, List[Any]]:
        """Validate time data type."""
        try:
            total_records = len(column)
            valid_records = 0
            type_mismatches = 0
            format_errors = 0
            invalid_values = []
            
            for idx, value in column.items():
                try:
                    # Try to parse as time
                    if isinstance(value, datetime):
                        valid_records += 1
                    else:
                        pd.to_datetime(value).time()
                        valid_records += 1
                        
                except (ValueError, TypeError):
                    type_mismatches += 1
                    invalid_values.append(value)
            
            return valid_records, type_mismatches, format_errors, invalid_values
            
        except Exception as e:
            logger.error(f"Error validating time type: {e}")
            return 0, len(column), 0, column.tolist()
    
    def _validate_email_type(self, column: pd.Series, rule: DataTypeRule) -> tuple[int, int, int, List[Any]]:
        """Validate email data type."""
        try:
            total_records = len(column)
            valid_records = 0
            type_mismatches = 0
            format_errors = 0
            invalid_values = []
            
            email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            
            for idx, value in column.items():
                try:
                    str_value = str(value)
                    if re.match(email_pattern, str_value):
                        valid_records += 1
                    else:
                        format_errors += 1
                        invalid_values.append(value)
                        
                except Exception:
                    type_mismatches += 1
                    invalid_values.append(value)
            
            return valid_records, type_mismatches, format_errors, invalid_values
            
        except Exception as e:
            logger.error(f"Error validating email type: {e}")
            return 0, len(column), 0, column.tolist()
    
    def _validate_url_type(self, column: pd.Series, rule: DataTypeRule) -> tuple[int, int, int, List[Any]]:
        """Validate URL data type."""
        try:
            total_records = len(column)
            valid_records = 0
            type_mismatches = 0
            format_errors = 0
            invalid_values = []
            
            url_pattern = r'^https?://[^\s/$.?#].[^\s]*$'
            
            for idx, value in column.items():
                try:
                    str_value = str(value)
                    if re.match(url_pattern, str_value):
                        valid_records += 1
                    else:
                        format_errors += 1
                        invalid_values.append(value)
                        
                except Exception:
                    type_mismatches += 1
                    invalid_values.append(value)
            
            return valid_records, type_mismatches, format_errors, invalid_values
            
        except Exception as e:
            logger.error(f"Error validating URL type: {e}")
            return 0, len(column), 0, column.tolist()
    
    def _validate_uuid_type(self, column: pd.Series, rule: DataTypeRule) -> tuple[int, int, int, List[Any]]:
        """Validate UUID data type."""
        try:
            total_records = len(column)
            valid_records = 0
            type_mismatches = 0
            format_errors = 0
            invalid_values = []
            
            uuid_pattern = r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
            
            for idx, value in column.items():
                try:
                    str_value = str(value).lower()
                    if re.match(uuid_pattern, str_value):
                        valid_records += 1
                    else:
                        format_errors += 1
                        invalid_values.append(value)
                        
                except Exception:
                    type_mismatches += 1
                    invalid_values.append(value)
            
            return valid_records, type_mismatches, format_errors, invalid_values
            
        except Exception as e:
            logger.error(f"Error validating UUID type: {e}")
            return 0, len(column), 0, column.tolist()
    
    def _validate_json_type(self, column: pd.Series, rule: DataTypeRule) -> tuple[int, int, int, List[Any]]:
        """Validate JSON data type."""
        try:
            total_records = len(column)
            valid_records = 0
            type_mismatches = 0
            format_errors = 0
            invalid_values = []
            
            for idx, value in column.items():
                try:
                    if isinstance(value, (dict, list)):
                        valid_records += 1
                    else:
                        # Try to parse as JSON string
                        import json
                        json.loads(str(value))
                        valid_records += 1
                        
                except (ValueError, TypeError, json.JSONDecodeError):
                    format_errors += 1
                    invalid_values.append(value)
            
            return valid_records, type_mismatches, format_errors, invalid_values
            
        except Exception as e:
            logger.error(f"Error validating JSON type: {e}")
            return 0, len(column), 0, column.tolist()
    
    def _calculate_validation_summary(self, source_name: str, results: List[DataTypeValidationResult]) -> DataTypeValidationSummary:
        """Calculate validation summary from individual results."""
        try:
            total_columns = len(results)
            valid_columns = sum(1 for result in results if result.is_valid)
            invalid_columns = total_columns - valid_columns
            
            # Calculate overall score
            if total_columns > 0:
                overall_score = valid_columns / total_columns
            else:
                overall_score = 1.0
            
            return DataTypeValidationSummary(
                source_name=source_name,
                total_columns=total_columns,
                valid_columns=valid_columns,
                invalid_columns=invalid_columns,
                overall_score=overall_score,
                results=results
            )
            
        except Exception as e:
            logger.error(f"Error calculating validation summary: {e}")
            return DataTypeValidationSummary(
                source_name=source_name,
                total_columns=0,
                valid_columns=0,
                invalid_columns=0,
                overall_score=0.0,
                results=[]
            )
    
    def _initialize_validation_rules(self):
        """Initialize data type validation rules for different sources."""
        try:
            # PBF Process validation rules
            self.validation_rules["pbf_process"] = [
                DataTypeRule("machine_id", DataType.STRING, nullable=False, format_pattern=r"^PBF-[A-Z]{2}-\d{3}$"),
                DataTypeRule("event_timestamp", DataType.DATETIME, nullable=False),
                DataTypeRule("chamber_temperature", DataType.FLOAT, nullable=False, min_value=20.0, max_value=1000.0),
                DataTypeRule("build_plate_temperature", DataType.FLOAT, nullable=False, min_value=20.0, max_value=500.0),
                DataTypeRule("chamber_pressure", DataType.FLOAT, nullable=False, min_value=0.0, max_value=10.0),
                DataTypeRule("laser_power", DataType.FLOAT, nullable=True, min_value=0.0, max_value=1000.0),
                DataTypeRule("laser_speed", DataType.FLOAT, nullable=True, min_value=0.0, max_value=10000.0)
            ]
            
            # ISPM Monitoring validation rules
            self.validation_rules["ispm_monitoring"] = [
                DataTypeRule("sensor_id", DataType.STRING, nullable=False, format_pattern=r"^ISPM-\d{3}$"),
                DataTypeRule("event_timestamp", DataType.DATETIME, nullable=False),
                DataTypeRule("melt_pool_temperature", DataType.FLOAT, nullable=False, min_value=1000.0, max_value=3000.0),
                DataTypeRule("plume_intensity", DataType.FLOAT, nullable=True, min_value=0.0, max_value=100.0),
                DataTypeRule("acoustic_emissions", DataType.FLOAT, nullable=True, min_value=0.0, max_value=1000.0)
            ]
            
            # CT Scan validation rules
            self.validation_rules["ct_scan"] = [
                DataTypeRule("scan_id", DataType.STRING, nullable=False, format_pattern=r"^CT-\d{6}$"),
                DataTypeRule("part_id", DataType.STRING, nullable=False),
                DataTypeRule("scan_date", DataType.DATETIME, nullable=False),
                DataTypeRule("porosity_percentage", DataType.FLOAT, nullable=False, min_value=0.0, max_value=100.0),
                DataTypeRule("num_defects", DataType.INTEGER, nullable=True, min_value=0),
                DataTypeRule("scan_volume_mm3", DataType.FLOAT, nullable=True, min_value=0.0)
            ]
            
            # Powder Bed validation rules
            self.validation_rules["powder_bed"] = [
                DataTypeRule("image_id", DataType.STRING, nullable=False, format_pattern=r"^PB-\d{8}$"),
                DataTypeRule("layer_number", DataType.INTEGER, nullable=False, min_value=1, max_value=5000),
                DataTypeRule("capture_timestamp", DataType.DATETIME, nullable=False),
                DataTypeRule("image_path", DataType.STRING, nullable=False),
                DataTypeRule("porosity_metric", DataType.FLOAT, nullable=True, min_value=0.0, max_value=1.0),
                DataTypeRule("roughness_metric", DataType.FLOAT, nullable=True, min_value=0.0, max_value=100.0)
            ]
            
            logger.info("Initialized data type validation rules for all sources")
            
        except Exception as e:
            logger.error(f"Error initializing validation rules: {e}")
    
    def add_validation_rule(self, source_name: str, rule: DataTypeRule) -> bool:
        """Add a validation rule for a specific source."""
        try:
            if source_name not in self.validation_rules:
                self.validation_rules[source_name] = []
            
            self.validation_rules[source_name].append(rule)
            logger.info(f"Added validation rule for {source_name}: {rule.column_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding validation rule: {e}")
            return False
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get a summary of all validation results."""
        try:
            total_sources = len(self.validation_results)
            total_columns = sum(summary.total_columns for summary in self.validation_results.values())
            valid_columns = sum(summary.valid_columns for summary in self.validation_results.values())
            invalid_columns = sum(summary.invalid_columns for summary in self.validation_results.values())
            
            overall_score = valid_columns / total_columns if total_columns > 0 else 1.0
            
            return {
                "total_sources": total_sources,
                "total_columns": total_columns,
                "valid_columns": valid_columns,
                "invalid_columns": invalid_columns,
                "overall_score": overall_score,
                "source_summaries": {
                    source: {
                        "overall_score": summary.overall_score,
                        "total_columns": summary.total_columns,
                        "valid_columns": summary.valid_columns,
                        "invalid_columns": summary.invalid_columns
                    }
                    for source, summary in self.validation_results.items()
                },
                "last_updated": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting validation summary: {e}")
            return {}
