"""
Data Cleanser

This module provides data cleansing capabilities for the PBF-LB/M data pipeline.
"""

import pandas as pd
import numpy as np
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import unicodedata

from src.data_pipeline.config.pipeline_config import get_pipeline_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CleansingOperation(Enum):
    """Data cleansing operation enumeration."""
    NORMALIZE = "normalize"
    STANDARDIZE = "standardize"
    DEDUPLICATE = "deduplicate"
    VALIDATE = "validate"
    TRANSFORM = "transform"
    ENRICH = "enrich"
    AGGREGATE = "aggregate"
    FILTER = "filter"

class CleansingStatus(Enum):
    """Cleansing status enumeration."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"

@dataclass
class CleansingRule:
    """Data cleansing rule data class."""
    id: str
    name: str
    description: str
    operation: CleansingOperation
    column: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    priority: int = 1

@dataclass
class CleansingResult:
    """Data cleansing result data class."""
    rule_id: str
    operation: CleansingOperation
    status: CleansingStatus
    records_processed: int
    records_cleaned: int
    records_removed: int
    records_transformed: int
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class CleansingJob:
    """Data cleansing job data class."""
    job_id: str
    source_name: str
    data: List[Dict[str, Any]]
    rules: List[CleansingRule]
    status: CleansingStatus
    results: List[CleansingResult] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    total_records: int = 0
    processed_records: int = 0
    cleaned_records: int = 0

class DataCleanser:
    """
    Data cleansing service for PBF-LB/M data pipeline.
    """
    
    def __init__(self):
        self.config = get_pipeline_config()
        self.cleansing_rules: Dict[str, CleansingRule] = {}
        self.cleansing_jobs: Dict[str, CleansingJob] = {}
        self.cleansing_results: Dict[str, List[CleansingResult]] = {}
        
        # Initialize cleansing rules
        self._initialize_cleansing_rules()
        
    def cleanse_pbf_process_data(self, data: List[Dict[str, Any]]) -> CleansingJob:
        """
        Cleanse PBF process data.
        
        Args:
            data: List of PBF process data records
            
        Returns:
            CleansingJob: The cleansing job results
        """
        try:
            logger.info(f"Starting PBF process data cleansing for {len(data)} records")
            
            # Create cleansing job
            job = CleansingJob(
                job_id=f"pbf_process_cleansing_{int(datetime.now().timestamp())}",
                source_name="pbf_process",
                data=data,
                rules=self._get_applicable_rules("pbf_process"),
                status=CleansingStatus.PENDING,
                total_records=len(data)
            )
            
            # Execute cleansing
            job = self._execute_cleansing_job(job)
            
            # Store job
            self.cleansing_jobs[job.job_id] = job
            
            logger.info(f"PBF process data cleansing completed. Job ID: {job.job_id}")
            return job
            
        except Exception as e:
            logger.error(f"Error cleansing PBF process data: {e}")
            raise
    
    def cleanse_ispm_monitoring_data(self, data: List[Dict[str, Any]]) -> CleansingJob:
        """
        Cleanse ISPM monitoring data.
        
        Args:
            data: List of ISPM monitoring data records
            
        Returns:
            CleansingJob: The cleansing job results
        """
        try:
            logger.info(f"Starting ISPM monitoring data cleansing for {len(data)} records")
            
            # Create cleansing job
            job = CleansingJob(
                job_id=f"ispm_monitoring_cleansing_{int(datetime.now().timestamp())}",
                source_name="ispm_monitoring",
                data=data,
                rules=self._get_applicable_rules("ispm_monitoring"),
                status=CleansingStatus.PENDING,
                total_records=len(data)
            )
            
            # Execute cleansing
            job = self._execute_cleansing_job(job)
            
            # Store job
            self.cleansing_jobs[job.job_id] = job
            
            logger.info(f"ISPM monitoring data cleansing completed. Job ID: {job.job_id}")
            return job
            
        except Exception as e:
            logger.error(f"Error cleansing ISPM monitoring data: {e}")
            raise
    
    def cleanse_ct_scan_data(self, data: List[Dict[str, Any]]) -> CleansingJob:
        """
        Cleanse CT scan data.
        
        Args:
            data: List of CT scan data records
            
        Returns:
            CleansingJob: The cleansing job results
        """
        try:
            logger.info(f"Starting CT scan data cleansing for {len(data)} records")
            
            # Create cleansing job
            job = CleansingJob(
                job_id=f"ct_scan_cleansing_{int(datetime.now().timestamp())}",
                source_name="ct_scan",
                data=data,
                rules=self._get_applicable_rules("ct_scan"),
                status=CleansingStatus.PENDING,
                total_records=len(data)
            )
            
            # Execute cleansing
            job = self._execute_cleansing_job(job)
            
            # Store job
            self.cleansing_jobs[job.job_id] = job
            
            logger.info(f"CT scan data cleansing completed. Job ID: {job.job_id}")
            return job
            
        except Exception as e:
            logger.error(f"Error cleansing CT scan data: {e}")
            raise
    
    def cleanse_powder_bed_data(self, data: List[Dict[str, Any]]) -> CleansingJob:
        """
        Cleanse powder bed data.
        
        Args:
            data: List of powder bed data records
            
        Returns:
            CleansingJob: The cleansing job results
        """
        try:
            logger.info(f"Starting powder bed data cleansing for {len(data)} records")
            
            # Create cleansing job
            job = CleansingJob(
                job_id=f"powder_bed_cleansing_{int(datetime.now().timestamp())}",
                source_name="powder_bed",
                data=data,
                rules=self._get_applicable_rules("powder_bed"),
                status=CleansingStatus.PENDING,
                total_records=len(data)
            )
            
            # Execute cleansing
            job = self._execute_cleansing_job(job)
            
            # Store job
            self.cleansing_jobs[job.job_id] = job
            
            logger.info(f"Powder bed data cleansing completed. Job ID: {job.job_id}")
            return job
            
        except Exception as e:
            logger.error(f"Error cleansing powder bed data: {e}")
            raise
    
    def get_cleansing_job(self, job_id: str) -> Optional[CleansingJob]:
        """
        Get a cleansing job by ID.
        
        Args:
            job_id: The job ID
            
        Returns:
            CleansingJob: The cleansing job, or None if not found
        """
        return self.cleansing_jobs.get(job_id)
    
    def get_cleansing_results(self, source_name: str) -> List[CleansingResult]:
        """
        Get cleansing results for a specific source.
        
        Args:
            source_name: The data source name
            
        Returns:
            List[CleansingResult]: List of cleansing results
        """
        return self.cleansing_results.get(source_name, [])
    
    def add_cleansing_rule(self, rule: CleansingRule) -> bool:
        """
        Add a new cleansing rule.
        
        Args:
            rule: The cleansing rule to add
            
        Returns:
            bool: True if rule was added successfully, False otherwise
        """
        try:
            self.cleansing_rules[rule.id] = rule
            logger.info(f"Added cleansing rule: {rule.id}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding cleansing rule {rule.id}: {e}")
            return False
    
    def get_cleansing_rule(self, rule_id: str) -> Optional[CleansingRule]:
        """
        Get a cleansing rule by ID.
        
        Args:
            rule_id: The rule ID
            
        Returns:
            CleansingRule: The cleansing rule, or None if not found
        """
        return self.cleansing_rules.get(rule_id)
    
    def get_cleansing_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all cleansing activities.
        
        Returns:
            Dict[str, Any]: Cleansing summary
        """
        try:
            total_jobs = len(self.cleansing_jobs)
            completed_jobs = len([job for job in self.cleansing_jobs.values() 
                                if job.status == CleansingStatus.COMPLETED])
            failed_jobs = len([job for job in self.cleansing_jobs.values() 
                             if job.status == CleansingStatus.FAILED])
            
            # Calculate total records processed
            total_records_processed = sum(job.processed_records for job in self.cleansing_jobs.values())
            total_records_cleaned = sum(job.cleaned_records for job in self.cleansing_jobs.values())
            
            # Calculate success rate
            success_rate = completed_jobs / total_jobs if total_jobs > 0 else 0.0
            
            return {
                "total_jobs": total_jobs,
                "completed_jobs": completed_jobs,
                "failed_jobs": failed_jobs,
                "success_rate": success_rate,
                "total_records_processed": total_records_processed,
                "total_records_cleaned": total_records_cleaned,
                "total_rules": len(self.cleansing_rules),
                "enabled_rules": len([rule for rule in self.cleansing_rules.values() if rule.enabled]),
                "last_updated": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting cleansing summary: {e}")
            return {}
    
    def _execute_cleansing_job(self, job: CleansingJob) -> CleansingJob:
        """Execute a cleansing job."""
        try:
            job.status = CleansingStatus.IN_PROGRESS
            job.started_at = datetime.now()
            
            # Process each rule
            for rule in job.rules:
                if not rule.enabled:
                    continue
                
                result = self._execute_cleansing_rule(job, rule)
                job.results.append(result)
                
                # Update job progress
                job.processed_records += result.records_processed
                job.cleaned_records += result.records_cleaned
            
            # Mark job as completed
            job.status = CleansingStatus.COMPLETED
            job.completed_at = datetime.now()
            
            # Store results
            if job.source_name not in self.cleansing_results:
                self.cleansing_results[job.source_name] = []
            self.cleansing_results[job.source_name].extend(job.results)
            
            return job
            
        except Exception as e:
            logger.error(f"Error executing cleansing job {job.job_id}: {e}")
            job.status = CleansingStatus.FAILED
            job.completed_at = datetime.now()
            return job
    
    def _execute_cleansing_rule(self, job: CleansingJob, rule: CleansingRule) -> CleansingResult:
        """Execute a specific cleansing rule."""
        try:
            start_time = datetime.now()
            
            # Initialize result
            result = CleansingResult(
                rule_id=rule.id,
                operation=rule.operation,
                status=CleansingStatus.PENDING
            )
            
            # Apply cleansing based on operation type
            if rule.operation == CleansingOperation.NORMALIZE:
                result = self._normalize_data(job.data, rule, result)
            elif rule.operation == CleansingOperation.STANDARDIZE:
                result = self._standardize_data(job.data, rule, result)
            elif rule.operation == CleansingOperation.DEDUPLICATE:
                result = self._deduplicate_data(job.data, rule, result)
            elif rule.operation == CleansingOperation.VALIDATE:
                result = self._validate_data(job.data, rule, result)
            elif rule.operation == CleansingOperation.TRANSFORM:
                result = self._transform_data(job.data, rule, result)
            elif rule.operation == CleansingOperation.ENRICH:
                result = self._enrich_data(job.data, rule, result)
            elif rule.operation == CleansingOperation.AGGREGATE:
                result = self._aggregate_data(job.data, rule, result)
            elif rule.operation == CleansingOperation.FILTER:
                result = self._filter_data(job.data, rule, result)
            else:
                result.status = CleansingStatus.FAILED
                result.errors.append(f"Unknown cleansing operation: {rule.operation}")
            
            # Calculate execution time
            result.execution_time = (datetime.now() - start_time).total_seconds()
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing cleansing rule {rule.id}: {e}")
            result.status = CleansingStatus.FAILED
            result.errors.append(str(e))
            return result
    
    def _normalize_data(self, data: List[Dict[str, Any]], rule: CleansingRule, 
                       result: CleansingResult) -> CleansingResult:
        """Normalize data based on rule parameters."""
        try:
            result.records_processed = len(data)
            
            column = rule.column
            normalization_type = rule.parameters.get("type", "text")
            
            if normalization_type == "text":
                result = self._normalize_text_data(data, column, rule, result)
            elif normalization_type == "numeric":
                result = self._normalize_numeric_data(data, column, rule, result)
            elif normalization_type == "datetime":
                result = self._normalize_datetime_data(data, column, rule, result)
            else:
                result.warnings.append(f"Unknown normalization type: {normalization_type}")
            
            result.status = CleansingStatus.COMPLETED
            return result
            
        except Exception as e:
            logger.error(f"Error normalizing data: {e}")
            result.status = CleansingStatus.FAILED
            result.errors.append(str(e))
            return result
    
    def _normalize_text_data(self, data: List[Dict[str, Any]], column: str, 
                           rule: CleansingRule, result: CleansingResult) -> CleansingResult:
        """Normalize text data."""
        try:
            cleaned_count = 0
            
            for record in data:
                if column in record and record[column] is not None:
                    original_value = str(record[column])
                    cleaned_value = original_value
                    
                    # Remove extra whitespace
                    cleaned_value = re.sub(r'\s+', ' ', cleaned_value.strip())
                    
                    # Remove special characters if specified
                    if rule.parameters.get("remove_special_chars", False):
                        cleaned_value = re.sub(r'[^\w\s]', '', cleaned_value)
                    
                    # Convert to lowercase if specified
                    if rule.parameters.get("to_lowercase", False):
                        cleaned_value = cleaned_value.lower()
                    
                    # Convert to uppercase if specified
                    if rule.parameters.get("to_uppercase", False):
                        cleaned_value = cleaned_value.upper()
                    
                    # Remove accents if specified
                    if rule.parameters.get("remove_accents", False):
                        cleaned_value = unicodedata.normalize('NFD', cleaned_value)
                        cleaned_value = ''.join(c for c in cleaned_value if unicodedata.category(c) != 'Mn')
                    
                    if cleaned_value != original_value:
                        record[column] = cleaned_value
                        cleaned_count += 1
            
            result.records_cleaned = cleaned_count
            return result
            
        except Exception as e:
            logger.error(f"Error normalizing text data: {e}")
            result.errors.append(str(e))
            return result
    
    def _normalize_numeric_data(self, data: List[Dict[str, Any]], column: str, 
                              rule: CleansingRule, result: CleansingResult) -> CleansingResult:
        """Normalize numeric data."""
        try:
            cleaned_count = 0
            
            for record in data:
                if column in record and record[column] is not None:
                    try:
                        original_value = record[column]
                        
                        # Convert to numeric
                        if isinstance(original_value, str):
                            # Remove non-numeric characters except decimal point and minus
                            cleaned_str = re.sub(r'[^\d.-]', '', original_value)
                            if cleaned_str:
                                numeric_value = float(cleaned_str)
                            else:
                                numeric_value = 0.0
                        else:
                            numeric_value = float(original_value)
                        
                        # Apply rounding if specified
                        if "decimal_places" in rule.parameters:
                            decimal_places = rule.parameters["decimal_places"]
                            numeric_value = round(numeric_value, decimal_places)
                        
                        # Apply scaling if specified
                        if "scale_factor" in rule.parameters:
                            scale_factor = rule.parameters["scale_factor"]
                            numeric_value = numeric_value * scale_factor
                        
                        if numeric_value != original_value:
                            record[column] = numeric_value
                            cleaned_count += 1
                            
                    except (ValueError, TypeError):
                        # Invalid numeric value, replace with default
                        default_value = rule.parameters.get("default_value", 0.0)
                        record[column] = default_value
                        cleaned_count += 1
            
            result.records_cleaned = cleaned_count
            return result
            
        except Exception as e:
            logger.error(f"Error normalizing numeric data: {e}")
            result.errors.append(str(e))
            return result
    
    def _normalize_datetime_data(self, data: List[Dict[str, Any]], column: str, 
                               rule: CleansingRule, result: CleansingResult) -> CleansingResult:
        """Normalize datetime data."""
        try:
            cleaned_count = 0
            date_format = rule.parameters.get("date_format", "%Y-%m-%d %H:%M:%S")
            
            for record in data:
                if column in record and record[column] is not None:
                    try:
                        original_value = record[column]
                        
                        # Parse datetime
                        if isinstance(original_value, str):
                            parsed_datetime = pd.to_datetime(original_value)
                        else:
                            parsed_datetime = pd.to_datetime(original_value)
                        
                        # Format datetime
                        formatted_datetime = parsed_datetime.strftime(date_format)
                        
                        if str(formatted_datetime) != str(original_value):
                            record[column] = formatted_datetime
                            cleaned_count += 1
                            
                    except (ValueError, TypeError):
                        # Invalid datetime, replace with default
                        default_value = rule.parameters.get("default_value", datetime.now().strftime(date_format))
                        record[column] = default_value
                        cleaned_count += 1
            
            result.records_cleaned = cleaned_count
            return result
            
        except Exception as e:
            logger.error(f"Error normalizing datetime data: {e}")
            result.errors.append(str(e))
            return result
    
    def _standardize_data(self, data: List[Dict[str, Any]], rule: CleansingRule, 
                         result: CleansingResult) -> CleansingResult:
        """Standardize data based on rule parameters."""
        try:
            result.records_processed = len(data)
            
            column = rule.column
            standardization_type = rule.parameters.get("type", "format")
            
            if standardization_type == "format":
                result = self._standardize_format(data, column, rule, result)
            elif standardization_type == "units":
                result = self._standardize_units(data, column, rule, result)
            elif standardization_type == "encoding":
                result = self._standardize_encoding(data, column, rule, result)
            else:
                result.warnings.append(f"Unknown standardization type: {standardization_type}")
            
            result.status = CleansingStatus.COMPLETED
            return result
            
        except Exception as e:
            logger.error(f"Error standardizing data: {e}")
            result.status = CleansingStatus.FAILED
            result.errors.append(str(e))
            return result
    
    def _standardize_format(self, data: List[Dict[str, Any]], column: str, 
                          rule: CleansingRule, result: CleansingResult) -> CleansingResult:
        """Standardize data format."""
        try:
            cleaned_count = 0
            target_format = rule.parameters.get("target_format")
            
            if not target_format:
                result.warnings.append("No target format specified")
                return result
            
            for record in data:
                if column in record and record[column] is not None:
                    original_value = str(record[column])
                    
                    # Apply format standardization
                    if target_format == "uuid":
                        # Standardize UUID format
                        cleaned_value = re.sub(r'[^a-fA-F0-9-]', '', original_value)
                        if len(cleaned_value) == 32:
                            cleaned_value = f"{cleaned_value[:8]}-{cleaned_value[8:12]}-{cleaned_value[12:16]}-{cleaned_value[16:20]}-{cleaned_value[20:]}"
                    elif target_format == "email":
                        # Standardize email format
                        cleaned_value = original_value.lower().strip()
                    elif target_format == "phone":
                        # Standardize phone format
                        cleaned_value = re.sub(r'[^\d]', '', original_value)
                        if len(cleaned_value) == 10:
                            cleaned_value = f"({cleaned_value[:3]}) {cleaned_value[3:6]}-{cleaned_value[6:]}"
                    else:
                        cleaned_value = original_value
                    
                    if cleaned_value != original_value:
                        record[column] = cleaned_value
                        cleaned_count += 1
            
            result.records_cleaned = cleaned_count
            return result
            
        except Exception as e:
            logger.error(f"Error standardizing format: {e}")
            result.errors.append(str(e))
            return result
    
    def _standardize_units(self, data: List[Dict[str, Any]], column: str, 
                         rule: CleansingRule, result: CleansingResult) -> CleansingResult:
        """Standardize units of measurement."""
        try:
            cleaned_count = 0
            target_unit = rule.parameters.get("target_unit")
            conversion_factors = rule.parameters.get("conversion_factors", {})
            
            if not target_unit:
                result.warnings.append("No target unit specified")
                return result
            
            for record in data:
                if column in record and record[column] is not None:
                    try:
                        original_value = str(record[column])
                        
                        # Extract numeric value and unit
                        match = re.match(r'([\d.]+)\s*([a-zA-Z]+)', original_value)
                        if match:
                            numeric_value = float(match.group(1))
                            unit = match.group(2).lower()
                            
                            # Convert to target unit
                            if unit in conversion_factors:
                                converted_value = numeric_value * conversion_factors[unit]
                                standardized_value = f"{converted_value} {target_unit}"
                                
                                if standardized_value != original_value:
                                    record[column] = standardized_value
                                    cleaned_count += 1
                        
                    except (ValueError, TypeError):
                        # Invalid format, keep original
                        pass
            
            result.records_cleaned = cleaned_count
            return result
            
        except Exception as e:
            logger.error(f"Error standardizing units: {e}")
            result.errors.append(str(e))
            return result
    
    def _standardize_encoding(self, data: List[Dict[str, Any]], column: str, 
                            rule: CleansingRule, result: CleansingResult) -> CleansingResult:
        """Standardize text encoding."""
        try:
            cleaned_count = 0
            target_encoding = rule.parameters.get("target_encoding", "utf-8")
            
            for record in data:
                if column in record and record[column] is not None:
                    try:
                        original_value = str(record[column])
                        
                        # Encode and decode to standardize encoding
                        encoded_value = original_value.encode(target_encoding, errors='ignore')
                        standardized_value = encoded_value.decode(target_encoding)
                        
                        if standardized_value != original_value:
                            record[column] = standardized_value
                            cleaned_count += 1
                            
                    except (UnicodeEncodeError, UnicodeDecodeError):
                        # Encoding error, keep original
                        pass
            
            result.records_cleaned = cleaned_count
            return result
            
        except Exception as e:
            logger.error(f"Error standardizing encoding: {e}")
            result.errors.append(str(e))
            return result
    
    def _deduplicate_data(self, data: List[Dict[str, Any]], rule: CleansingRule, 
                         result: CleansingResult) -> CleansingResult:
        """Remove duplicate records."""
        try:
            result.records_processed = len(data)
            
            # Convert to DataFrame for easier processing
            df = pd.DataFrame(data)
            
            # Remove duplicates
            initial_count = len(df)
            df = df.drop_duplicates(subset=[rule.column] if rule.column else None)
            removed_count = initial_count - len(df)
            
            result.records_removed = removed_count
            
            # Update data
            data.clear()
            data.extend(df.to_dict('records'))
            
            result.status = CleansingStatus.COMPLETED
            return result
            
        except Exception as e:
            logger.error(f"Error deduplicating data: {e}")
            result.status = CleansingStatus.FAILED
            result.errors.append(str(e))
            return result
    
    def _validate_data(self, data: List[Dict[str, Any]], rule: CleansingRule, 
                      result: CleansingResult) -> CleansingResult:
        """Validate data against rules."""
        try:
            result.records_processed = len(data)
            
            column = rule.column
            validation_rules = rule.parameters.get("validation_rules", {})
            
            validated_count = 0
            
            for record in data:
                if column in record:
                    value = record[column]
                    is_valid = True
                    
                    # Check required
                    if validation_rules.get("required", False) and (value is None or value == ""):
                        is_valid = False
                    
                    # Check data type
                    expected_type = validation_rules.get("type")
                    if expected_type and not isinstance(value, expected_type):
                        is_valid = False
                    
                    # Check range
                    if "min" in validation_rules and value < validation_rules["min"]:
                        is_valid = False
                    if "max" in validation_rules and value > validation_rules["max"]:
                        is_valid = False
                    
                    # Check pattern
                    pattern = validation_rules.get("pattern")
                    if pattern and not re.match(pattern, str(value)):
                        is_valid = False
                    
                    if is_valid:
                        validated_count += 1
            
            result.records_cleaned = validated_count
            result.status = CleansingStatus.COMPLETED
            return result
            
        except Exception as e:
            logger.error(f"Error validating data: {e}")
            result.status = CleansingStatus.FAILED
            result.errors.append(str(e))
            return result
    
    def _transform_data(self, data: List[Dict[str, Any]], rule: CleansingRule, 
                       result: CleansingResult) -> CleansingResult:
        """Transform data based on rule parameters."""
        try:
            result.records_processed = len(data)
            
            column = rule.column
            transformation_type = rule.parameters.get("type", "default")
            
            if transformation_type == "log":
                result = self._log_transform(data, column, rule, result)
            elif transformation_type == "sqrt":
                result = self._sqrt_transform(data, column, rule, result)
            elif transformation_type == "box_cox":
                result = self._box_cox_transform(data, column, rule, result)
            else:
                result = self._default_transform(data, column, rule, result)
            
            result.status = CleansingStatus.COMPLETED
            return result
            
        except Exception as e:
            logger.error(f"Error transforming data: {e}")
            result.status = CleansingStatus.FAILED
            result.errors.append(str(e))
            return result
    
    def _log_transform(self, data: List[Dict[str, Any]], column: str, 
                      rule: CleansingRule, result: CleansingResult) -> CleansingResult:
        """Apply log transformation."""
        try:
            transformed_count = 0
            
            for record in data:
                if column in record and record[column] is not None:
                    try:
                        value = float(record[column])
                        if value > 0:
                            log_value = np.log(value)
                            record[column] = log_value
                            transformed_count += 1
                    except (ValueError, TypeError):
                        pass
            
            result.records_transformed = transformed_count
            return result
            
        except Exception as e:
            logger.error(f"Error applying log transform: {e}")
            result.errors.append(str(e))
            return result
    
    def _sqrt_transform(self, data: List[Dict[str, Any]], column: str, 
                       rule: CleansingRule, result: CleansingResult) -> CleansingResult:
        """Apply square root transformation."""
        try:
            transformed_count = 0
            
            for record in data:
                if column in record and record[column] is not None:
                    try:
                        value = float(record[column])
                        if value >= 0:
                            sqrt_value = np.sqrt(value)
                            record[column] = sqrt_value
                            transformed_count += 1
                    except (ValueError, TypeError):
                        pass
            
            result.records_transformed = transformed_count
            return result
            
        except Exception as e:
            logger.error(f"Error applying sqrt transform: {e}")
            result.errors.append(str(e))
            return result
    
    def _box_cox_transform(self, data: List[Dict[str, Any]], column: str, 
                          rule: CleansingRule, result: CleansingResult) -> CleansingResult:
        """Apply Box-Cox transformation."""
        try:
            transformed_count = 0
            lambda_param = rule.parameters.get("lambda", 0.5)
            
            for record in data:
                if column in record and record[column] is not None:
                    try:
                        value = float(record[column])
                        if value > 0:
                            if lambda_param == 0:
                                box_cox_value = np.log(value)
                            else:
                                box_cox_value = (value ** lambda_param - 1) / lambda_param
                            record[column] = box_cox_value
                            transformed_count += 1
                    except (ValueError, TypeError):
                        pass
            
            result.records_transformed = transformed_count
            return result
            
        except Exception as e:
            logger.error(f"Error applying Box-Cox transform: {e}")
            result.errors.append(str(e))
            return result
    
    def _default_transform(self, data: List[Dict[str, Any]], column: str, 
                          rule: CleansingRule, result: CleansingResult) -> CleansingResult:
        """Apply default transformation."""
        try:
            transformed_count = 0
            
            for record in data:
                if column in record and record[column] is not None:
                    # Simple data type conversion
                    try:
                        value = record[column]
                        if isinstance(value, str):
                            if '.' in value:
                                record[column] = float(value)
                            else:
                                record[column] = int(value)
                            transformed_count += 1
                    except (ValueError, TypeError):
                        pass
            
            result.records_transformed = transformed_count
            return result
            
        except Exception as e:
            logger.error(f"Error applying default transform: {e}")
            result.errors.append(str(e))
            return result
    
    def _enrich_data(self, data: List[Dict[str, Any]], rule: CleansingRule, 
                    result: CleansingResult) -> CleansingResult:
        """Enrich data with additional information."""
        try:
            result.records_processed = len(data)
            
            enrichment_type = rule.parameters.get("type", "default")
            
            if enrichment_type == "geocoding":
                result = self._enrich_geocoding(data, rule, result)
            elif enrichment_type == "lookup":
                result = self._enrich_lookup(data, rule, result)
            else:
                result = self._enrich_default(data, rule, result)
            
            result.status = CleansingStatus.COMPLETED
            return result
            
        except Exception as e:
            logger.error(f"Error enriching data: {e}")
            result.status = CleansingStatus.FAILED
            result.errors.append(str(e))
            return result
    
    def _enrich_geocoding(self, data: List[Dict[str, Any]], rule: CleansingRule, 
                         result: CleansingResult) -> CleansingResult:
        """Enrich data with geocoding information."""
        try:
            enriched_count = 0
            
            for record in data:
                # Add geocoding information (placeholder)
                record["_geocoded"] = True
                record["_geocoding_timestamp"] = datetime.now().isoformat()
                enriched_count += 1
            
            result.records_cleaned = enriched_count
            return result
            
        except Exception as e:
            logger.error(f"Error enriching geocoding: {e}")
            result.errors.append(str(e))
            return result
    
    def _enrich_lookup(self, data: List[Dict[str, Any]], rule: CleansingRule, 
                      result: CleansingResult) -> CleansingResult:
        """Enrich data with lookup information."""
        try:
            enriched_count = 0
            lookup_table = rule.parameters.get("lookup_table", {})
            source_column = rule.parameters.get("source_column")
            target_column = rule.parameters.get("target_column")
            
            if not source_column or not target_column:
                result.warnings.append("Source and target columns must be specified for lookup")
                return result
            
            for record in data:
                if source_column in record:
                    source_value = record[source_column]
                    if source_value in lookup_table:
                        record[target_column] = lookup_table[source_value]
                        enriched_count += 1
            
            result.records_cleaned = enriched_count
            return result
            
        except Exception as e:
            logger.error(f"Error enriching lookup: {e}")
            result.errors.append(str(e))
            return result
    
    def _enrich_default(self, data: List[Dict[str, Any]], rule: CleansingRule, 
                       result: CleansingResult) -> CleansingResult:
        """Apply default enrichment."""
        try:
            enriched_count = 0
            
            for record in data:
                # Add metadata
                record["_enriched"] = True
                record["_enrichment_timestamp"] = datetime.now().isoformat()
                enriched_count += 1
            
            result.records_cleaned = enriched_count
            return result
            
        except Exception as e:
            logger.error(f"Error applying default enrichment: {e}")
            result.errors.append(str(e))
            return result
    
    def _aggregate_data(self, data: List[Dict[str, Any]], rule: CleansingRule, 
                       result: CleansingResult) -> CleansingResult:
        """Aggregate data based on rule parameters."""
        try:
            result.records_processed = len(data)
            
            # Convert to DataFrame for easier processing
            df = pd.DataFrame(data)
            
            # Apply aggregation
            group_by = rule.parameters.get("group_by", [])
            aggregation_rules = rule.parameters.get("aggregation_rules", {})
            
            if group_by and aggregation_rules:
                aggregated_df = df.groupby(group_by).agg(aggregation_rules).reset_index()
                
                # Update data
                data.clear()
                data.extend(aggregated_df.to_dict('records'))
                
                result.records_cleaned = len(aggregated_df)
            else:
                result.warnings.append("Group by and aggregation rules must be specified")
            
            result.status = CleansingStatus.COMPLETED
            return result
            
        except Exception as e:
            logger.error(f"Error aggregating data: {e}")
            result.status = CleansingStatus.FAILED
            result.errors.append(str(e))
            return result
    
    def _filter_data(self, data: List[Dict[str, Any]], rule: CleansingRule, 
                    result: CleansingResult) -> CleansingResult:
        """Filter data based on rule parameters."""
        try:
            result.records_processed = len(data)
            
            # Convert to DataFrame for easier processing
            df = pd.DataFrame(data)
            
            # Apply filters
            filters = rule.parameters.get("filters", {})
            
            if filters:
                for column, filter_rule in filters.items():
                    if column in df.columns:
                        if "min" in filter_rule:
                            df = df[df[column] >= filter_rule["min"]]
                        if "max" in filter_rule:
                            df = df[df[column] <= filter_rule["max"]]
                        if "values" in filter_rule:
                            df = df[df[column].isin(filter_rule["values"])]
                        if "pattern" in filter_rule:
                            df = df[df[column].str.match(filter_rule["pattern"], na=False)]
                
                # Update data
                data.clear()
                data.extend(df.to_dict('records'))
                
                result.records_cleaned = len(df)
            else:
                result.warnings.append("No filters specified")
            
            result.status = CleansingStatus.COMPLETED
            return result
            
        except Exception as e:
            logger.error(f"Error filtering data: {e}")
            result.status = CleansingStatus.FAILED
            result.errors.append(str(e))
            return result
    
    def _get_applicable_rules(self, source_name: str) -> List[CleansingRule]:
        """Get applicable cleansing rules for a source."""
        try:
            applicable_rules = []
            
            for rule in self.cleansing_rules.values():
                if rule.enabled:
                    # Check if rule applies to this source
                    source_types = rule.parameters.get("source_types", [])
                    if not source_types or source_name in source_types:
                        applicable_rules.append(rule)
            
            # Sort by priority
            applicable_rules.sort(key=lambda r: r.priority)
            
            return applicable_rules
            
        except Exception as e:
            logger.error(f"Error getting applicable rules: {e}")
            return []
    
    def _initialize_cleansing_rules(self):
        """Initialize default cleansing rules."""
        try:
            # PBF Process rules
            self.cleansing_rules["pbf_temp_normalize"] = CleansingRule(
                id="pbf_temp_normalize",
                name="PBF Temperature Normalization",
                description="Normalize temperature values",
                operation=CleansingOperation.NORMALIZE,
                column="chamber_temperature",
                parameters={
                    "type": "numeric",
                    "decimal_places": 2,
                    "source_types": ["pbf_process"]
                },
                priority=1
            )
            
            self.cleansing_rules["pbf_id_standardize"] = CleansingRule(
                id="pbf_id_standardize",
                name="PBF ID Standardization",
                description="Standardize machine ID format",
                operation=CleansingOperation.STANDARDIZE,
                column="machine_id",
                parameters={
                    "type": "format",
                    "target_format": "uppercase",
                    "source_types": ["pbf_process"]
                },
                priority=2
            )
            
            # ISPM Monitoring rules
            self.cleansing_rules["ispm_deduplicate"] = CleansingRule(
                id="ispm_deduplicate",
                name="ISPM Deduplication",
                description="Remove duplicate ISPM records",
                operation=CleansingOperation.DEDUPLICATE,
                column="sensor_id",
                parameters={
                    "source_types": ["ispm_monitoring"]
                },
                priority=1
            )
            
            # CT Scan rules
            self.cleansing_rules["ct_validate"] = CleansingRule(
                id="ct_validate",
                name="CT Scan Validation",
                description="Validate CT scan data",
                operation=CleansingOperation.VALIDATE,
                column="porosity_percentage",
                parameters={
                    "validation_rules": {
                        "required": True,
                        "type": float,
                        "min": 0.0,
                        "max": 100.0
                    },
                    "source_types": ["ct_scan"]
                },
                priority=1
            )
            
            # Powder Bed rules
            self.cleansing_rules["pb_transform"] = CleansingRule(
                id="pb_transform",
                name="Powder Bed Transformation",
                description="Transform powder bed data",
                operation=CleansingOperation.TRANSFORM,
                column="porosity_metric",
                parameters={
                    "type": "log",
                    "source_types": ["powder_bed"]
                },
                priority=1
            )
            
            logger.info(f"Initialized {len(self.cleansing_rules)} cleansing rules")
            
        except Exception as e:
            logger.error(f"Error initializing cleansing rules: {e}")
