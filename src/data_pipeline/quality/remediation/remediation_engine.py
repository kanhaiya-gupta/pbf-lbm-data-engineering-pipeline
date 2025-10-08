"""
Remediation Engine

This module provides automated data quality remediation capabilities for the PBF-LB/M data pipeline.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import json

from src.data_pipeline.config.pipeline_config import get_pipeline_config
from src.data_pipeline.quality.validation.data_quality_service import QualityResult, QualityRule
from src.data_pipeline.quality.monitoring.quality_monitor import QualityAlert

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RemediationAction(Enum):
    """Remediation action enumeration."""
    FIX = "fix"
    REMOVE = "remove"
    FLAG = "flag"
    TRANSFORM = "transform"
    INTERPOLATE = "interpolate"
    REPLACE = "replace"
    IGNORE = "ignore"
    ESCALATE = "escalate"

class RemediationStatus(Enum):
    """Remediation status enumeration."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

class RemediationPriority(Enum):
    """Remediation priority enumeration."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class RemediationRule:
    """Remediation rule data class."""
    id: str
    name: str
    description: str
    condition: str
    action: RemediationAction
    priority: RemediationPriority
    enabled: bool = True
    parameters: Dict[str, Any] = field(default_factory=dict)
    source_types: List[str] = field(default_factory=list)

@dataclass
class RemediationResult:
    """Remediation result data class."""
    rule_id: str
    source_name: str
    action: RemediationAction
    status: RemediationStatus
    records_processed: int
    records_fixed: int
    records_removed: int
    records_flagged: int
    records_transformed: int
    records_interpolated: int
    records_replaced: int
    records_ignored: int
    records_escalated: int
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class RemediationJob:
    """Remediation job data class."""
    job_id: str
    source_name: str
    data: List[Dict[str, Any]]
    rules: List[RemediationRule]
    priority: RemediationPriority
    status: RemediationStatus
    results: List[RemediationResult] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    total_records: int = 0
    processed_records: int = 0
    failed_records: int = 0

class RemediationEngine:
    """
    Automated data quality remediation engine for PBF-LB/M data pipeline.
    """
    
    def __init__(self):
        self.config = get_pipeline_config()
        self.remediation_rules: Dict[str, RemediationRule] = {}
        self.remediation_jobs: Dict[str, RemediationJob] = {}
        self.remediation_results: Dict[str, List[RemediationResult]] = {}
        
        # Initialize remediation rules
        self._initialize_remediation_rules()
        
    def remediate_pbf_process_data(self, data: List[Dict[str, Any]], 
                                 quality_results: List[QualityResult]) -> RemediationJob:
        """
        Remediate PBF process data based on quality results.
        
        Args:
            data: List of PBF process data records
            quality_results: List of quality validation results
            
        Returns:
            RemediationJob: The remediation job results
        """
        try:
            logger.info(f"Starting PBF process data remediation for {len(data)} records")
            
            # Create remediation job
            job = RemediationJob(
                job_id=f"pbf_process_remediation_{int(datetime.now().timestamp())}",
                source_name="pbf_process",
                data=data,
                rules=self._get_applicable_rules("pbf_process"),
                priority=RemediationPriority.HIGH,
                status=RemediationStatus.PENDING,
                total_records=len(data)
            )
            
            # Execute remediation
            job = self._execute_remediation_job(job, quality_results)
            
            # Store job
            self.remediation_jobs[job.job_id] = job
            
            logger.info(f"PBF process data remediation completed. Job ID: {job.job_id}")
            return job
            
        except Exception as e:
            logger.error(f"Error remediating PBF process data: {e}")
            raise
    
    def remediate_ispm_monitoring_data(self, data: List[Dict[str, Any]], 
                                     quality_results: List[QualityResult]) -> RemediationJob:
        """
        Remediate ISPM monitoring data based on quality results.
        
        Args:
            data: List of ISPM monitoring data records
            quality_results: List of quality validation results
            
        Returns:
            RemediationJob: The remediation job results
        """
        try:
            logger.info(f"Starting ISPM monitoring data remediation for {len(data)} records")
            
            # Create remediation job
            job = RemediationJob(
                job_id=f"ispm_monitoring_remediation_{int(datetime.now().timestamp())}",
                source_name="ispm_monitoring",
                data=data,
                rules=self._get_applicable_rules("ispm_monitoring"),
                priority=RemediationPriority.HIGH,
                status=RemediationStatus.PENDING,
                total_records=len(data)
            )
            
            # Execute remediation
            job = self._execute_remediation_job(job, quality_results)
            
            # Store job
            self.remediation_jobs[job.job_id] = job
            
            logger.info(f"ISPM monitoring data remediation completed. Job ID: {job.job_id}")
            return job
            
        except Exception as e:
            logger.error(f"Error remediating ISPM monitoring data: {e}")
            raise
    
    def remediate_ct_scan_data(self, data: List[Dict[str, Any]], 
                             quality_results: List[QualityResult]) -> RemediationJob:
        """
        Remediate CT scan data based on quality results.
        
        Args:
            data: List of CT scan data records
            quality_results: List of quality validation results
            
        Returns:
            RemediationJob: The remediation job results
        """
        try:
            logger.info(f"Starting CT scan data remediation for {len(data)} records")
            
            # Create remediation job
            job = RemediationJob(
                job_id=f"ct_scan_remediation_{int(datetime.now().timestamp())}",
                source_name="ct_scan",
                data=data,
                rules=self._get_applicable_rules("ct_scan"),
                priority=RemediationPriority.MEDIUM,
                status=RemediationStatus.PENDING,
                total_records=len(data)
            )
            
            # Execute remediation
            job = self._execute_remediation_job(job, quality_results)
            
            # Store job
            self.remediation_jobs[job.job_id] = job
            
            logger.info(f"CT scan data remediation completed. Job ID: {job.job_id}")
            return job
            
        except Exception as e:
            logger.error(f"Error remediating CT scan data: {e}")
            raise
    
    def remediate_powder_bed_data(self, data: List[Dict[str, Any]], 
                                quality_results: List[QualityResult]) -> RemediationJob:
        """
        Remediate powder bed data based on quality results.
        
        Args:
            data: List of powder bed data records
            quality_results: List of quality validation results
            
        Returns:
            RemediationJob: The remediation job results
        """
        try:
            logger.info(f"Starting powder bed data remediation for {len(data)} records")
            
            # Create remediation job
            job = RemediationJob(
                job_id=f"powder_bed_remediation_{int(datetime.now().timestamp())}",
                source_name="powder_bed",
                data=data,
                rules=self._get_applicable_rules("powder_bed"),
                priority=RemediationPriority.MEDIUM,
                status=RemediationStatus.PENDING,
                total_records=len(data)
            )
            
            # Execute remediation
            job = self._execute_remediation_job(job, quality_results)
            
            # Store job
            self.remediation_jobs[job.job_id] = job
            
            logger.info(f"Powder bed data remediation completed. Job ID: {job.job_id}")
            return job
            
        except Exception as e:
            logger.error(f"Error remediating powder bed data: {e}")
            raise
    
    def get_remediation_job(self, job_id: str) -> Optional[RemediationJob]:
        """
        Get a remediation job by ID.
        
        Args:
            job_id: The job ID
            
        Returns:
            RemediationJob: The remediation job, or None if not found
        """
        return self.remediation_jobs.get(job_id)
    
    def get_remediation_results(self, source_name: str) -> List[RemediationResult]:
        """
        Get remediation results for a specific source.
        
        Args:
            source_name: The data source name
            
        Returns:
            List[RemediationResult]: List of remediation results
        """
        return self.remediation_results.get(source_name, [])
    
    def add_remediation_rule(self, rule: RemediationRule) -> bool:
        """
        Add a new remediation rule.
        
        Args:
            rule: The remediation rule to add
            
        Returns:
            bool: True if rule was added successfully, False otherwise
        """
        try:
            self.remediation_rules[rule.id] = rule
            logger.info(f"Added remediation rule: {rule.id}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding remediation rule {rule.id}: {e}")
            return False
    
    def get_remediation_rule(self, rule_id: str) -> Optional[RemediationRule]:
        """
        Get a remediation rule by ID.
        
        Args:
            rule_id: The rule ID
            
        Returns:
            RemediationRule: The remediation rule, or None if not found
        """
        return self.remediation_rules.get(rule_id)
    
    def get_remediation_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all remediation activities.
        
        Returns:
            Dict[str, Any]: Remediation summary
        """
        try:
            total_jobs = len(self.remediation_jobs)
            completed_jobs = len([job for job in self.remediation_jobs.values() 
                                if job.status == RemediationStatus.COMPLETED])
            failed_jobs = len([job for job in self.remediation_jobs.values() 
                             if job.status == RemediationStatus.FAILED])
            
            # Calculate total records processed
            total_records_processed = sum(job.processed_records for job in self.remediation_jobs.values())
            total_records_fixed = sum(
                sum(result.records_fixed for result in job.results) 
                for job in self.remediation_jobs.values()
            )
            
            # Calculate success rate
            success_rate = completed_jobs / total_jobs if total_jobs > 0 else 0.0
            
            return {
                "total_jobs": total_jobs,
                "completed_jobs": completed_jobs,
                "failed_jobs": failed_jobs,
                "success_rate": success_rate,
                "total_records_processed": total_records_processed,
                "total_records_fixed": total_records_fixed,
                "total_rules": len(self.remediation_rules),
                "enabled_rules": len([rule for rule in self.remediation_rules.values() if rule.enabled]),
                "last_updated": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting remediation summary: {e}")
            return {}
    
    def _execute_remediation_job(self, job: RemediationJob, quality_results: List[QualityResult]) -> RemediationJob:
        """Execute a remediation job."""
        try:
            job.status = RemediationStatus.IN_PROGRESS
            job.started_at = datetime.now()
            
            # Process each rule
            for rule in job.rules:
                if not rule.enabled:
                    continue
                
                result = self._execute_remediation_rule(job, rule, quality_results)
                job.results.append(result)
                
                # Update job progress
                job.processed_records += result.records_processed
                if result.status == RemediationStatus.FAILED:
                    job.failed_records += result.records_processed
            
            # Mark job as completed
            job.status = RemediationStatus.COMPLETED
            job.completed_at = datetime.now()
            
            # Store results
            if job.source_name not in self.remediation_results:
                self.remediation_results[job.source_name] = []
            self.remediation_results[job.source_name].extend(job.results)
            
            return job
            
        except Exception as e:
            logger.error(f"Error executing remediation job {job.job_id}: {e}")
            job.status = RemediationStatus.FAILED
            job.completed_at = datetime.now()
            return job
    
    def _execute_remediation_rule(self, job: RemediationJob, rule: RemediationRule, 
                                quality_results: List[QualityResult]) -> RemediationResult:
        """Execute a specific remediation rule."""
        try:
            start_time = datetime.now()
            
            # Initialize result
            result = RemediationResult(
                rule_id=rule.id,
                source_name=job.source_name,
                action=rule.action,
                status=RemediationStatus.PENDING
            )
            
            # Apply remediation based on action type
            if rule.action == RemediationAction.FIX:
                result = self._fix_data(job.data, rule, quality_results, result)
            elif rule.action == RemediationAction.REMOVE:
                result = self._remove_data(job.data, rule, quality_results, result)
            elif rule.action == RemediationAction.FLAG:
                result = self._flag_data(job.data, rule, quality_results, result)
            elif rule.action == RemediationAction.TRANSFORM:
                result = self._transform_data(job.data, rule, quality_results, result)
            elif rule.action == RemediationAction.INTERPOLATE:
                result = self._interpolate_data(job.data, rule, quality_results, result)
            elif rule.action == RemediationAction.REPLACE:
                result = self._replace_data(job.data, rule, quality_results, result)
            elif rule.action == RemediationAction.IGNORE:
                result = self._ignore_data(job.data, rule, quality_results, result)
            elif rule.action == RemediationAction.ESCALATE:
                result = self._escalate_data(job.data, rule, quality_results, result)
            else:
                result.status = RemediationStatus.SKIPPED
                result.warnings.append(f"Unknown remediation action: {rule.action}")
            
            # Calculate execution time
            result.execution_time = (datetime.now() - start_time).total_seconds()
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing remediation rule {rule.id}: {e}")
            result.status = RemediationStatus.FAILED
            result.errors.append(str(e))
            return result
    
    def _fix_data(self, data: List[Dict[str, Any]], rule: RemediationRule, 
                 quality_results: List[QualityResult], result: RemediationResult) -> RemediationResult:
        """Fix data based on rule conditions."""
        try:
            result.records_processed = len(data)
            
            # Apply fixes based on rule parameters
            fix_type = rule.parameters.get("fix_type", "default")
            
            if fix_type == "outlier_correction":
                result = self._fix_outliers(data, rule, result)
            elif fix_type == "missing_value_imputation":
                result = self._fix_missing_values(data, rule, result)
            elif fix_type == "format_correction":
                result = self._fix_format_issues(data, rule, result)
            elif fix_type == "range_correction":
                result = self._fix_range_issues(data, rule, result)
            else:
                result = self._fix_default(data, rule, result)
            
            result.status = RemediationStatus.COMPLETED
            return result
            
        except Exception as e:
            logger.error(f"Error fixing data: {e}")
            result.status = RemediationStatus.FAILED
            result.errors.append(str(e))
            return result
    
    def _fix_outliers(self, data: List[Dict[str, Any]], rule: RemediationRule, 
                     result: RemediationResult) -> RemediationResult:
        """Fix outliers in the data."""
        try:
            column = rule.parameters.get("column")
            method = rule.parameters.get("method", "iqr")
            
            if not column:
                result.warnings.append("No column specified for outlier correction")
                return result
            
            # Convert to DataFrame for easier processing
            df = pd.DataFrame(data)
            
            if column not in df.columns:
                result.warnings.append(f"Column {column} not found in data")
                return result
            
            # Apply outlier correction
            if method == "iqr":
                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Cap outliers
                df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
                result.records_fixed = len(df[(df[column] < lower_bound) | (df[column] > upper_bound)])
                
            elif method == "zscore":
                z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
                threshold = rule.parameters.get("threshold", 3.0)
                
                # Replace outliers with mean
                outlier_mask = z_scores > threshold
                df.loc[outlier_mask, column] = df[column].mean()
                result.records_fixed = outlier_mask.sum()
            
            # Update data
            data.clear()
            data.extend(df.to_dict('records'))
            
            return result
            
        except Exception as e:
            logger.error(f"Error fixing outliers: {e}")
            result.errors.append(str(e))
            return result
    
    def _fix_missing_values(self, data: List[Dict[str, Any]], rule: RemediationRule, 
                          result: RemediationResult) -> RemediationResult:
        """Fix missing values in the data."""
        try:
            column = rule.parameters.get("column")
            method = rule.parameters.get("method", "mean")
            
            if not column:
                result.warnings.append("No column specified for missing value imputation")
                return result
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            if column not in df.columns:
                result.warnings.append(f"Column {column} not found in data")
                return result
            
            # Count missing values
            missing_count = df[column].isnull().sum()
            result.records_fixed = missing_count
            
            # Apply imputation
            if method == "mean":
                df[column].fillna(df[column].mean(), inplace=True)
            elif method == "median":
                df[column].fillna(df[column].median(), inplace=True)
            elif method == "mode":
                df[column].fillna(df[column].mode()[0], inplace=True)
            elif method == "forward_fill":
                df[column].fillna(method='ffill', inplace=True)
            elif method == "backward_fill":
                df[column].fillna(method='bfill', inplace=True)
            elif method == "interpolate":
                df[column].interpolate(inplace=True)
            else:
                # Default value
                default_value = rule.parameters.get("default_value", 0)
                df[column].fillna(default_value, inplace=True)
            
            # Update data
            data.clear()
            data.extend(df.to_dict('records'))
            
            return result
            
        except Exception as e:
            logger.error(f"Error fixing missing values: {e}")
            result.errors.append(str(e))
            return result
    
    def _fix_format_issues(self, data: List[Dict[str, Any]], rule: RemediationRule, 
                         result: RemediationResult) -> RemediationResult:
        """Fix format issues in the data."""
        try:
            column = rule.parameters.get("column")
            expected_format = rule.parameters.get("expected_format")
            
            if not column or not expected_format:
                result.warnings.append("Column and expected format must be specified")
                return result
            
            fixed_count = 0
            
            for record in data:
                if column in record:
                    value = record[column]
                    if isinstance(value, str):
                        # Apply format corrections
                        if expected_format == "uppercase":
                            record[column] = value.upper()
                            fixed_count += 1
                        elif expected_format == "lowercase":
                            record[column] = value.lower()
                            fixed_count += 1
                        elif expected_format == "title_case":
                            record[column] = value.title()
                            fixed_count += 1
                        elif expected_format == "strip_whitespace":
                            record[column] = value.strip()
                            fixed_count += 1
            
            result.records_fixed = fixed_count
            return result
            
        except Exception as e:
            logger.error(f"Error fixing format issues: {e}")
            result.errors.append(str(e))
            return result
    
    def _fix_range_issues(self, data: List[Dict[str, Any]], rule: RemediationRule, 
                        result: RemediationResult) -> RemediationResult:
        """Fix range issues in the data."""
        try:
            column = rule.parameters.get("column")
            min_value = rule.parameters.get("min_value")
            max_value = rule.parameters.get("max_value")
            
            if not column or min_value is None or max_value is None:
                result.warnings.append("Column, min_value, and max_value must be specified")
                return result
            
            fixed_count = 0
            
            for record in data:
                if column in record:
                    try:
                        value = float(record[column])
                        if value < min_value:
                            record[column] = min_value
                            fixed_count += 1
                        elif value > max_value:
                            record[column] = max_value
                            fixed_count += 1
                    except (ValueError, TypeError):
                        # Invalid value, replace with default
                        default_value = rule.parameters.get("default_value", min_value)
                        record[column] = default_value
                        fixed_count += 1
            
            result.records_fixed = fixed_count
            return result
            
        except Exception as e:
            logger.error(f"Error fixing range issues: {e}")
            result.errors.append(str(e))
            return result
    
    def _fix_default(self, data: List[Dict[str, Any]], rule: RemediationRule, 
                    result: RemediationResult) -> RemediationResult:
        """Apply default fixes."""
        try:
            # Simple validation and correction
            fixed_count = 0
            
            for record in data:
                for key, value in record.items():
                    if value is None or value == "":
                        # Replace with default value
                        default_value = rule.parameters.get("default_value", "N/A")
                        record[key] = default_value
                        fixed_count += 1
            
            result.records_fixed = fixed_count
            return result
            
        except Exception as e:
            logger.error(f"Error applying default fixes: {e}")
            result.errors.append(str(e))
            return result
    
    def _remove_data(self, data: List[Dict[str, Any]], rule: RemediationRule, 
                    quality_results: List[QualityResult], result: RemediationResult) -> RemediationResult:
        """Remove data based on rule conditions."""
        try:
            result.records_processed = len(data)
            
            # Apply removal based on rule parameters
            removal_type = rule.parameters.get("removal_type", "default")
            
            if removal_type == "duplicates":
                result = self._remove_duplicates(data, rule, result)
            elif removal_type == "outliers":
                result = self._remove_outliers(data, rule, result)
            elif removal_type == "invalid_records":
                result = self._remove_invalid_records(data, rule, result)
            else:
                result = self._remove_default(data, rule, result)
            
            result.status = RemediationStatus.COMPLETED
            return result
            
        except Exception as e:
            logger.error(f"Error removing data: {e}")
            result.status = RemediationStatus.FAILED
            result.errors.append(str(e))
            return result
    
    def _remove_duplicates(self, data: List[Dict[str, Any]], rule: RemediationRule, 
                         result: RemediationResult) -> RemediationResult:
        """Remove duplicate records."""
        try:
            # Convert to DataFrame for easier processing
            df = pd.DataFrame(data)
            
            # Remove duplicates
            initial_count = len(df)
            df = df.drop_duplicates()
            removed_count = initial_count - len(df)
            
            result.records_removed = removed_count
            
            # Update data
            data.clear()
            data.extend(df.to_dict('records'))
            
            return result
            
        except Exception as e:
            logger.error(f"Error removing duplicates: {e}")
            result.errors.append(str(e))
            return result
    
    def _remove_outliers(self, data: List[Dict[str, Any]], rule: RemediationRule, 
                        result: RemediationResult) -> RemediationResult:
        """Remove outlier records."""
        try:
            column = rule.parameters.get("column")
            method = rule.parameters.get("method", "iqr")
            
            if not column:
                result.warnings.append("No column specified for outlier removal")
                return result
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            if column not in df.columns:
                result.warnings.append(f"Column {column} not found in data")
                return result
            
            # Identify outliers
            if method == "iqr":
                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outlier_mask = (df[column] < lower_bound) | (df[column] > upper_bound)
                
            elif method == "zscore":
                z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
                threshold = rule.parameters.get("threshold", 3.0)
                outlier_mask = z_scores > threshold
            
            else:
                result.warnings.append(f"Unknown outlier removal method: {method}")
                return result
            
            # Remove outliers
            removed_count = outlier_mask.sum()
            df = df[~outlier_mask]
            
            result.records_removed = removed_count
            
            # Update data
            data.clear()
            data.extend(df.to_dict('records'))
            
            return result
            
        except Exception as e:
            logger.error(f"Error removing outliers: {e}")
            result.errors.append(str(e))
            return result
    
    def _remove_invalid_records(self, data: List[Dict[str, Any]], rule: RemediationRule, 
                              result: RemediationResult) -> RemediationResult:
        """Remove invalid records."""
        try:
            validation_rules = rule.parameters.get("validation_rules", {})
            removed_count = 0
            
            # Filter out invalid records
            valid_data = []
            for record in data:
                is_valid = True
                
                for column, rules in validation_rules.items():
                    if column in record:
                        value = record[column]
                        
                        # Check required
                        if rules.get("required", False) and (value is None or value == ""):
                            is_valid = False
                            break
                        
                        # Check data type
                        expected_type = rules.get("type")
                        if expected_type and not isinstance(value, expected_type):
                            is_valid = False
                            break
                        
                        # Check range
                        if "min" in rules and value < rules["min"]:
                            is_valid = False
                            break
                        if "max" in rules and value > rules["max"]:
                            is_valid = False
                            break
                
                if is_valid:
                    valid_data.append(record)
                else:
                    removed_count += 1
            
            result.records_removed = removed_count
            
            # Update data
            data.clear()
            data.extend(valid_data)
            
            return result
            
        except Exception as e:
            logger.error(f"Error removing invalid records: {e}")
            result.errors.append(str(e))
            return result
    
    def _remove_default(self, data: List[Dict[str, Any]], rule: RemediationRule, 
                       result: RemediationResult) -> RemediationResult:
        """Apply default removal logic."""
        try:
            # Remove records with all null values
            removed_count = 0
            valid_data = []
            
            for record in data:
                if any(value is not None and value != "" for value in record.values()):
                    valid_data.append(record)
                else:
                    removed_count += 1
            
            result.records_removed = removed_count
            
            # Update data
            data.clear()
            data.extend(valid_data)
            
            return result
            
        except Exception as e:
            logger.error(f"Error applying default removal: {e}")
            result.errors.append(str(e))
            return result
    
    def _flag_data(self, data: List[Dict[str, Any]], rule: RemediationRule, 
                  quality_results: List[QualityResult], result: RemediationResult) -> RemediationResult:
        """Flag data based on rule conditions."""
        try:
            result.records_processed = len(data)
            
            # Add quality flags to records
            flagged_count = 0
            
            for record in data:
                # Add quality flag
                record["_quality_flag"] = True
                record["_quality_flag_reason"] = rule.description
                record["_quality_flag_timestamp"] = datetime.now().isoformat()
                flagged_count += 1
            
            result.records_flagged = flagged_count
            result.status = RemediationStatus.COMPLETED
            
            return result
            
        except Exception as e:
            logger.error(f"Error flagging data: {e}")
            result.status = RemediationStatus.FAILED
            result.errors.append(str(e))
            return result
    
    def _transform_data(self, data: List[Dict[str, Any]], rule: RemediationRule, 
                       quality_results: List[QualityResult], result: RemediationResult) -> RemediationResult:
        """Transform data based on rule conditions."""
        try:
            result.records_processed = len(data)
            
            # Apply transformations
            transformation_type = rule.parameters.get("transformation_type", "default")
            
            if transformation_type == "normalize":
                result = self._normalize_data(data, rule, result)
            elif transformation_type == "standardize":
                result = self._standardize_data(data, rule, result)
            elif transformation_type == "log_transform":
                result = self._log_transform_data(data, rule, result)
            else:
                result = self._transform_default(data, rule, result)
            
            result.status = RemediationStatus.COMPLETED
            return result
            
        except Exception as e:
            logger.error(f"Error transforming data: {e}")
            result.status = RemediationStatus.FAILED
            result.errors.append(str(e))
            return result
    
    def _normalize_data(self, data: List[Dict[str, Any]], rule: RemediationRule, 
                       result: RemediationResult) -> RemediationResult:
        """Normalize data to 0-1 range."""
        try:
            column = rule.parameters.get("column")
            
            if not column:
                result.warnings.append("No column specified for normalization")
                return result
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            if column not in df.columns:
                result.warnings.append(f"Column {column} not found in data")
                return result
            
            # Normalize to 0-1 range
            min_val = df[column].min()
            max_val = df[column].max()
            
            if max_val != min_val:
                df[column] = (df[column] - min_val) / (max_val - min_val)
                result.records_transformed = len(df)
            else:
                result.warnings.append(f"Column {column} has no variation")
            
            # Update data
            data.clear()
            data.extend(df.to_dict('records'))
            
            return result
            
        except Exception as e:
            logger.error(f"Error normalizing data: {e}")
            result.errors.append(str(e))
            return result
    
    def _standardize_data(self, data: List[Dict[str, Any]], rule: RemediationRule, 
                         result: RemediationResult) -> RemediationResult:
        """Standardize data to z-scores."""
        try:
            column = rule.parameters.get("column")
            
            if not column:
                result.warnings.append("No column specified for standardization")
                return result
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            if column not in df.columns:
                result.warnings.append(f"Column {column} not found in data")
                return result
            
            # Standardize to z-scores
            mean_val = df[column].mean()
            std_val = df[column].std()
            
            if std_val != 0:
                df[column] = (df[column] - mean_val) / std_val
                result.records_transformed = len(df)
            else:
                result.warnings.append(f"Column {column} has no variation")
            
            # Update data
            data.clear()
            data.extend(df.to_dict('records'))
            
            return result
            
        except Exception as e:
            logger.error(f"Error standardizing data: {e}")
            result.errors.append(str(e))
            return result
    
    def _log_transform_data(self, data: List[Dict[str, Any]], rule: RemediationRule, 
                           result: RemediationResult) -> RemediationResult:
        """Apply log transformation to data."""
        try:
            column = rule.parameters.get("column")
            
            if not column:
                result.warnings.append("No column specified for log transformation")
                return result
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            if column not in df.columns:
                result.warnings.append(f"Column {column} not found in data")
                return result
            
            # Apply log transformation
            df[column] = np.log1p(df[column])  # log1p handles zero values
            result.records_transformed = len(df)
            
            # Update data
            data.clear()
            data.extend(df.to_dict('records'))
            
            return result
            
        except Exception as e:
            logger.error(f"Error applying log transformation: {e}")
            result.errors.append(str(e))
            return result
    
    def _transform_default(self, data: List[Dict[str, Any]], rule: RemediationRule, 
                          result: RemediationResult) -> RemediationResult:
        """Apply default transformations."""
        try:
            # Simple data type conversions
            transformed_count = 0
            
            for record in data:
                for key, value in record.items():
                    if isinstance(value, str):
                        # Try to convert to numeric
                        try:
                            if '.' in value:
                                record[key] = float(value)
                            else:
                                record[key] = int(value)
                            transformed_count += 1
                        except ValueError:
                            pass
            
            result.records_transformed = transformed_count
            return result
            
        except Exception as e:
            logger.error(f"Error applying default transformations: {e}")
            result.errors.append(str(e))
            return result
    
    def _interpolate_data(self, data: List[Dict[str, Any]], rule: RemediationRule, 
                         quality_results: List[QualityResult], result: RemediationResult) -> RemediationResult:
        """Interpolate missing data."""
        try:
            result.records_processed = len(data)
            
            # Apply interpolation
            column = rule.parameters.get("column")
            method = rule.parameters.get("method", "linear")
            
            if not column:
                result.warnings.append("No column specified for interpolation")
                return result
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            if column not in df.columns:
                result.warnings.append(f"Column {column} not found in data")
                return result
            
            # Count missing values
            missing_count = df[column].isnull().sum()
            
            # Apply interpolation
            if method == "linear":
                df[column].interpolate(method='linear', inplace=True)
            elif method == "polynomial":
                df[column].interpolate(method='polynomial', order=2, inplace=True)
            elif method == "spline":
                df[column].interpolate(method='spline', order=3, inplace=True)
            else:
                df[column].interpolate(method='linear', inplace=True)
            
            result.records_interpolated = missing_count
            
            # Update data
            data.clear()
            data.extend(df.to_dict('records'))
            
            result.status = RemediationStatus.COMPLETED
            return result
            
        except Exception as e:
            logger.error(f"Error interpolating data: {e}")
            result.status = RemediationStatus.FAILED
            result.errors.append(str(e))
            return result
    
    def _replace_data(self, data: List[Dict[str, Any]], rule: RemediationRule, 
                     quality_results: List[QualityResult], result: RemediationResult) -> RemediationResult:
        """Replace data based on rule conditions."""
        try:
            result.records_processed = len(data)
            
            # Apply replacements
            replacements = rule.parameters.get("replacements", {})
            replaced_count = 0
            
            for record in data:
                for column, replacement_rules in replacements.items():
                    if column in record:
                        value = record[column]
                        
                        for old_value, new_value in replacement_rules.items():
                            if value == old_value:
                                record[column] = new_value
                                replaced_count += 1
            
            result.records_replaced = replaced_count
            result.status = RemediationStatus.COMPLETED
            
            return result
            
        except Exception as e:
            logger.error(f"Error replacing data: {e}")
            result.status = RemediationStatus.FAILED
            result.errors.append(str(e))
            return result
    
    def _ignore_data(self, data: List[Dict[str, Any]], rule: RemediationRule, 
                    quality_results: List[QualityResult], result: RemediationResult) -> RemediationResult:
        """Ignore data based on rule conditions."""
        try:
            result.records_processed = len(data)
            result.records_ignored = len(data)
            result.status = RemediationStatus.COMPLETED
            
            return result
            
        except Exception as e:
            logger.error(f"Error ignoring data: {e}")
            result.status = RemediationStatus.FAILED
            result.errors.append(str(e))
            return result
    
    def _escalate_data(self, data: List[Dict[str, Any]], rule: RemediationRule, 
                      quality_results: List[QualityResult], result: RemediationResult) -> RemediationResult:
        """Escalate data for manual review."""
        try:
            result.records_processed = len(data)
            result.records_escalated = len(data)
            result.status = RemediationStatus.COMPLETED
            
            # Add escalation metadata
            result.details["escalation_reason"] = rule.description
            result.details["escalation_timestamp"] = datetime.now().isoformat()
            result.details["escalation_priority"] = rule.priority.value
            
            return result
            
        except Exception as e:
            logger.error(f"Error escalating data: {e}")
            result.status = RemediationStatus.FAILED
            result.errors.append(str(e))
            return result
    
    def _get_applicable_rules(self, source_name: str) -> List[RemediationRule]:
        """Get applicable remediation rules for a source."""
        try:
            applicable_rules = []
            
            for rule in self.remediation_rules.values():
                if rule.enabled and (not rule.source_types or source_name in rule.source_types):
                    applicable_rules.append(rule)
            
            # Sort by priority
            priority_order = {
                RemediationPriority.CRITICAL: 0,
                RemediationPriority.HIGH: 1,
                RemediationPriority.MEDIUM: 2,
                RemediationPriority.LOW: 3
            }
            
            applicable_rules.sort(key=lambda r: priority_order.get(r.priority, 4))
            
            return applicable_rules
            
        except Exception as e:
            logger.error(f"Error getting applicable rules: {e}")
            return []
    
    def _initialize_remediation_rules(self):
        """Initialize default remediation rules."""
        try:
            # PBF Process rules
            self.remediation_rules["pbf_temp_outlier_fix"] = RemediationRule(
                id="pbf_temp_outlier_fix",
                name="PBF Temperature Outlier Fix",
                description="Fix temperature outliers using IQR method",
                condition="temperature_outlier",
                action=RemediationAction.FIX,
                priority=RemediationPriority.HIGH,
                source_types=["pbf_process"],
                parameters={
                    "column": "chamber_temperature",
                    "fix_type": "outlier_correction",
                    "method": "iqr"
                }
            )
            
            self.remediation_rules["pbf_missing_values_fix"] = RemediationRule(
                id="pbf_missing_values_fix",
                name="PBF Missing Values Fix",
                description="Fix missing values using interpolation",
                condition="missing_values",
                action=RemediationAction.FIX,
                priority=RemediationPriority.MEDIUM,
                source_types=["pbf_process"],
                parameters={
                    "column": "laser_power",
                    "fix_type": "missing_value_imputation",
                    "method": "interpolate"
                }
            )
            
            # ISPM Monitoring rules
            self.remediation_rules["ispm_outlier_remove"] = RemediationRule(
                id="ispm_outlier_remove",
                name="ISPM Outlier Removal",
                description="Remove extreme outliers from ISPM data",
                condition="extreme_outlier",
                action=RemediationAction.REMOVE,
                priority=RemediationPriority.HIGH,
                source_types=["ispm_monitoring"],
                parameters={
                    "column": "melt_pool_temperature",
                    "removal_type": "outliers",
                    "method": "zscore",
                    "threshold": 4.0
                }
            )
            
            # CT Scan rules
            self.remediation_rules["ct_duplicate_remove"] = RemediationRule(
                id="ct_duplicate_remove",
                name="CT Scan Duplicate Removal",
                description="Remove duplicate CT scan records",
                condition="duplicates",
                action=RemediationAction.REMOVE,
                priority=RemediationPriority.MEDIUM,
                source_types=["ct_scan"],
                parameters={
                    "removal_type": "duplicates"
                }
            )
            
            # Powder Bed rules
            self.remediation_rules["pb_format_fix"] = RemediationRule(
                id="pb_format_fix",
                name="Powder Bed Format Fix",
                description="Fix format issues in powder bed data",
                condition="format_issue",
                action=RemediationAction.FIX,
                priority=RemediationPriority.LOW,
                source_types=["powder_bed"],
                parameters={
                    "column": "image_id",
                    "fix_type": "format_correction",
                    "expected_format": "uppercase"
                }
            )
            
            logger.info(f"Initialized {len(self.remediation_rules)} remediation rules")
            
        except Exception as e:
            logger.error(f"Error initializing remediation rules: {e}")
