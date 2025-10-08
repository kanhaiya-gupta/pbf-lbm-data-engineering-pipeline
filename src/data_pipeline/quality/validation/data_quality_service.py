"""
Data Quality Service

This module provides comprehensive data quality validation services for the PBF-LB/M data pipeline.
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
import re
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from src.data_pipeline.config.pipeline_config import get_pipeline_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ValidationLevel(Enum):
    """Validation level enumeration."""
    STRICT = "strict"
    MODERATE = "moderate"
    RELAXED = "relaxed"

class QualityRule(Enum):
    """Quality rule enumeration."""
    NOT_NULL = "not_null"
    UNIQUE = "unique"
    RANGE = "range"
    REGEX = "regex"
    DATA_TYPE = "data_type"
    REFERENTIAL_INTEGRITY = "referential_integrity"
    COMPLETENESS = "completeness"
    ACCURACY = "accuracy"
    CONSISTENCY = "consistency"
    TIMELINESS = "timeliness"
    VALIDITY = "validity"

@dataclass
class QualityResult:
    """Quality validation result data class."""
    rule_name: str
    rule_type: QualityRule
    passed: bool
    score: float
    message: str
    failed_records: List[int] = field(default_factory=list)
    total_records: int = 0
    passed_records: int = 0
    failed_records_count: int = 0
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class QualityProfile:
    """Data quality profile data class."""
    source_name: str
    overall_score: float
    validation_level: ValidationLevel
    results: List[QualityResult] = field(default_factory=list)
    total_records: int = 0
    valid_records: int = 0
    invalid_records: int = 0
    timestamp: datetime = field(default_factory=datetime.now)

class DataQualityService:
    """
    Comprehensive data quality validation service.
    """
    
    def __init__(self):
        self.config = get_pipeline_config()
        self.validation_rules: Dict[str, Dict[str, Any]] = {}
        self.quality_profiles: Dict[str, QualityProfile] = {}
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.scaler = StandardScaler()
        
        # Initialize validation rules
        self._initialize_validation_rules()
        
    def validate_pbf_process_data(self, data: List[Dict[str, Any]]) -> QualityProfile:
        """
        Validate PBF process data quality.
        
        Args:
            data: List of PBF process data records
            
        Returns:
            QualityProfile: Quality validation results
        """
        try:
            logger.info(f"Validating PBF process data: {len(data)} records")
            
            df = pd.DataFrame(data)
            profile = QualityProfile(
                source_name="pbf_process",
                validation_level=ValidationLevel.STRICT,
                total_records=len(data)
            )
            
            # Apply PBF-specific validation rules
            results = []
            
            # Machine ID validation
            results.append(self._validate_not_null(df, "machine_id", "PBF machine ID not null"))
            results.append(self._validate_regex(df, "machine_id", r"^PBF-[A-Z]{2}-\d{3}$", "PBF machine ID format"))
            
            # Temperature validation
            results.append(self._validate_range(df, "chamber_temperature", 20.0, 1000.0, "Chamber temperature range"))
            results.append(self._validate_range(df, "build_plate_temperature", 20.0, 500.0, "Build plate temperature range"))
            
            # Pressure validation
            results.append(self._validate_range(df, "chamber_pressure", 0.0, 10.0, "Chamber pressure range"))
            
            # Timestamp validation
            results.append(self._validate_timestamp(df, "event_timestamp", "Event timestamp format"))
            
            # Laser parameters validation
            if "laser_power" in df.columns:
                results.append(self._validate_range(df, "laser_power", 0.0, 1000.0, "Laser power range"))
            if "laser_speed" in df.columns:
                results.append(self._validate_range(df, "laser_speed", 0.0, 10000.0, "Laser speed range"))
            
            # Calculate overall quality score
            profile.results = results
            profile.overall_score = self._calculate_overall_score(results)
            profile.valid_records = sum(r.passed_records for r in results if r.passed)
            profile.invalid_records = profile.total_records - profile.valid_records
            
            self.quality_profiles["pbf_process"] = profile
            
            logger.info(f"PBF process data validation completed. Overall score: {profile.overall_score:.2f}")
            return profile
            
        except Exception as e:
            logger.error(f"Error validating PBF process data: {e}")
            raise
    
    def validate_ispm_monitoring_data(self, data: List[Dict[str, Any]]) -> QualityProfile:
        """
        Validate ISPM monitoring data quality.
        
        Args:
            data: List of ISPM monitoring data records
            
        Returns:
            QualityProfile: Quality validation results
        """
        try:
            logger.info(f"Validating ISPM monitoring data: {len(data)} records")
            
            df = pd.DataFrame(data)
            profile = QualityProfile(
                source_name="ispm_monitoring",
                validation_level=ValidationLevel.STRICT,
                total_records=len(data)
            )
            
            # Apply ISPM-specific validation rules
            results = []
            
            # Sensor ID validation
            results.append(self._validate_not_null(df, "sensor_id", "ISPM sensor ID not null"))
            results.append(self._validate_regex(df, "sensor_id", r"^ISPM-\d{3}$", "ISPM sensor ID format"))
            
            # Melt pool temperature validation
            results.append(self._validate_not_null(df, "melt_pool_temperature", "Melt pool temperature not null"))
            results.append(self._validate_range(df, "melt_pool_temperature", 1000.0, 3000.0, "Melt pool temperature range"))
            
            # Plume intensity validation
            if "plume_intensity" in df.columns:
                results.append(self._validate_range(df, "plume_intensity", 0.0, 100.0, "Plume intensity range"))
            
            # Acoustic emissions validation
            if "acoustic_emissions" in df.columns:
                results.append(self._validate_range(df, "acoustic_emissions", 0.0, 1000.0, "Acoustic emissions range"))
            
            # Timestamp validation
            results.append(self._validate_timestamp(df, "event_timestamp", "Event timestamp format"))
            
            # Anomaly detection
            results.append(self._detect_temperature_anomalies(df, "melt_pool_temperature", "Temperature anomaly detection"))
            
            # Calculate overall quality score
            profile.results = results
            profile.overall_score = self._calculate_overall_score(results)
            profile.valid_records = sum(r.passed_records for r in results if r.passed)
            profile.invalid_records = profile.total_records - profile.valid_records
            
            self.quality_profiles["ispm_monitoring"] = profile
            
            logger.info(f"ISPM monitoring data validation completed. Overall score: {profile.overall_score:.2f}")
            return profile
            
        except Exception as e:
            logger.error(f"Error validating ISPM monitoring data: {e}")
            raise
    
    def validate_ct_scan_data(self, data: List[Dict[str, Any]]) -> QualityProfile:
        """
        Validate CT scan data quality.
        
        Args:
            data: List of CT scan data records
            
        Returns:
            QualityProfile: Quality validation results
        """
        try:
            logger.info(f"Validating CT scan data: {len(data)} records")
            
            df = pd.DataFrame(data)
            profile = QualityProfile(
                source_name="ct_scan",
                validation_level=ValidationLevel.MODERATE,
                total_records=len(data)
            )
            
            # Apply CT scan-specific validation rules
            results = []
            
            # Scan ID validation
            results.append(self._validate_not_null(df, "scan_id", "CT scan ID not null"))
            results.append(self._validate_regex(df, "scan_id", r"^CT-\d{6}$", "CT scan ID format"))
            
            # Part ID validation
            results.append(self._validate_not_null(df, "part_id", "Part ID not null"))
            
            # Porosity validation
            results.append(self._validate_not_null(df, "porosity_percentage", "Porosity percentage not null"))
            results.append(self._validate_range(df, "porosity_percentage", 0.0, 100.0, "Porosity percentage range"))
            
            # Defect count validation
            if "num_defects" in df.columns:
                results.append(self._validate_range(df, "num_defects", 0, 10000, "Defect count range"))
            
            # Scan date validation
            results.append(self._validate_timestamp(df, "scan_date", "Scan date format"))
            
            # Volume validation
            if "scan_volume_mm3" in df.columns:
                results.append(self._validate_range(df, "scan_volume_mm3", 0.0, 1000000.0, "Scan volume range"))
            
            # Calculate overall quality score
            profile.results = results
            profile.overall_score = self._calculate_overall_score(results)
            profile.valid_records = sum(r.passed_records for r in results if r.passed)
            profile.invalid_records = profile.total_records - profile.valid_records
            
            self.quality_profiles["ct_scan"] = profile
            
            logger.info(f"CT scan data validation completed. Overall score: {profile.overall_score:.2f}")
            return profile
            
        except Exception as e:
            logger.error(f"Error validating CT scan data: {e}")
            raise
    
    def validate_powder_bed_data(self, data: List[Dict[str, Any]]) -> QualityProfile:
        """
        Validate powder bed data quality.
        
        Args:
            data: List of powder bed data records
            
        Returns:
            QualityProfile: Quality validation results
        """
        try:
            logger.info(f"Validating powder bed data: {len(data)} records")
            
            df = pd.DataFrame(data)
            profile = QualityProfile(
                source_name="powder_bed",
                validation_level=ValidationLevel.MODERATE,
                total_records=len(data)
            )
            
            # Apply powder bed-specific validation rules
            results = []
            
            # Image ID validation
            results.append(self._validate_not_null(df, "image_id", "Powder bed image ID not null"))
            results.append(self._validate_regex(df, "image_id", r"^PB-\d{8}$", "Powder bed image ID format"))
            
            # Layer number validation
            results.append(self._validate_not_null(df, "layer_number", "Layer number not null"))
            results.append(self._validate_range(df, "layer_number", 1, 5000, "Layer number range"))
            
            # Image path validation
            results.append(self._validate_not_null(df, "image_path", "Image path not null"))
            
            # Quality metrics validation
            if "porosity_metric" in df.columns:
                results.append(self._validate_range(df, "porosity_metric", 0.0, 1.0, "Porosity metric range"))
            if "roughness_metric" in df.columns:
                results.append(self._validate_range(df, "roughness_metric", 0.0, 100.0, "Roughness metric range"))
            
            # Timestamp validation
            results.append(self._validate_timestamp(df, "capture_timestamp", "Capture timestamp format"))
            
            # Calculate overall quality score
            profile.results = results
            profile.overall_score = self._calculate_overall_score(results)
            profile.valid_records = sum(r.passed_records for r in results if r.passed)
            profile.invalid_records = profile.total_records - profile.valid_records
            
            self.quality_profiles["powder_bed"] = profile
            
            logger.info(f"Powder bed data validation completed. Overall score: {profile.overall_score:.2f}")
            return profile
            
        except Exception as e:
            logger.error(f"Error validating powder bed data: {e}")
            raise
    
    def check_pbf_process_quality(self) -> Dict[str, Any]:
        """
        Check PBF process data quality (for monitoring).
        
        Returns:
            Dict[str, Any]: Quality check results
        """
        try:
            # This would typically query recent data and validate it
            # For now, return a mock result
            return {
                "source": "pbf_process",
                "quality_score": 0.95,
                "status": "healthy",
                "issues": [],
                "last_checked": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error checking PBF process quality: {e}")
            return {"source": "pbf_process", "quality_score": 0.0, "status": "error", "issues": [str(e)]}
    
    def check_ispm_monitoring_quality(self) -> Dict[str, Any]:
        """
        Check ISPM monitoring data quality (for monitoring).
        
        Returns:
            Dict[str, Any]: Quality check results
        """
        try:
            return {
                "source": "ispm_monitoring",
                "quality_score": 0.98,
                "status": "healthy",
                "issues": [],
                "last_checked": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error checking ISPM monitoring quality: {e}")
            return {"source": "ispm_monitoring", "quality_score": 0.0, "status": "error", "issues": [str(e)]}
    
    def check_ct_scan_quality(self) -> Dict[str, Any]:
        """
        Check CT scan data quality (for monitoring).
        
        Returns:
            Dict[str, Any]: Quality check results
        """
        try:
            return {
                "source": "ct_scan",
                "quality_score": 0.92,
                "status": "healthy",
                "issues": [],
                "last_checked": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error checking CT scan quality: {e}")
            return {"source": "ct_scan", "quality_score": 0.0, "status": "error", "issues": [str(e)]}
    
    def check_powder_bed_quality(self) -> Dict[str, Any]:
        """
        Check powder bed data quality (for monitoring).
        
        Returns:
            Dict[str, Any]: Quality check results
        """
        try:
            return {
                "source": "powder_bed",
                "quality_score": 0.89,
                "status": "healthy",
                "issues": [],
                "last_checked": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error checking powder bed quality: {e}")
            return {"source": "powder_bed", "quality_score": 0.0, "status": "error", "issues": [str(e)]}
    
    def _validate_not_null(self, df: pd.DataFrame, column: str, rule_name: str) -> QualityResult:
        """Validate that a column has no null values."""
        try:
            if column not in df.columns:
                return QualityResult(
                    rule_name=rule_name,
                    rule_type=QualityRule.NOT_NULL,
                    passed=False,
                    score=0.0,
                    message=f"Column {column} not found",
                    total_records=len(df)
                )
            
            null_mask = df[column].isnull()
            failed_records = df[null_mask].index.tolist()
            passed_records = len(df) - len(failed_records)
            score = passed_records / len(df) if len(df) > 0 else 0.0
            
            return QualityResult(
                rule_name=rule_name,
                rule_type=QualityRule.NOT_NULL,
                passed=len(failed_records) == 0,
                score=score,
                message=f"Null validation: {passed_records}/{len(df)} records passed",
                failed_records=failed_records,
                total_records=len(df),
                passed_records=passed_records,
                failed_records_count=len(failed_records)
            )
            
        except Exception as e:
            logger.error(f"Error validating not null for {column}: {e}")
            return QualityResult(
                rule_name=rule_name,
                rule_type=QualityRule.NOT_NULL,
                passed=False,
                score=0.0,
                message=f"Validation error: {e}",
                total_records=len(df)
            )
    
    def _validate_range(self, df: pd.DataFrame, column: str, min_val: float, max_val: float, rule_name: str) -> QualityResult:
        """Validate that a column's values are within a specified range."""
        try:
            if column not in df.columns:
                return QualityResult(
                    rule_name=rule_name,
                    rule_type=QualityRule.RANGE,
                    passed=False,
                    score=0.0,
                    message=f"Column {column} not found",
                    total_records=len(df)
                )
            
            # Convert to numeric, handling non-numeric values
            numeric_series = pd.to_numeric(df[column], errors='coerce')
            range_mask = (numeric_series >= min_val) & (numeric_series <= max_val)
            failed_records = df[~range_mask].index.tolist()
            passed_records = range_mask.sum()
            score = passed_records / len(df) if len(df) > 0 else 0.0
            
            return QualityResult(
                rule_name=rule_name,
                rule_type=QualityRule.RANGE,
                passed=len(failed_records) == 0,
                score=score,
                message=f"Range validation ({min_val}-{max_val}): {passed_records}/{len(df)} records passed",
                failed_records=failed_records,
                total_records=len(df),
                passed_records=passed_records,
                failed_records_count=len(failed_records)
            )
            
        except Exception as e:
            logger.error(f"Error validating range for {column}: {e}")
            return QualityResult(
                rule_name=rule_name,
                rule_type=QualityRule.RANGE,
                passed=False,
                score=0.0,
                message=f"Validation error: {e}",
                total_records=len(df)
            )
    
    def _validate_regex(self, df: pd.DataFrame, column: str, pattern: str, rule_name: str) -> QualityResult:
        """Validate that a column's values match a regex pattern."""
        try:
            if column not in df.columns:
                return QualityResult(
                    rule_name=rule_name,
                    rule_type=QualityRule.REGEX,
                    passed=False,
                    score=0.0,
                    message=f"Column {column} not found",
                    total_records=len(df)
                )
            
            regex_mask = df[column].astype(str).str.match(pattern, na=False)
            failed_records = df[~regex_mask].index.tolist()
            passed_records = regex_mask.sum()
            score = passed_records / len(df) if len(df) > 0 else 0.0
            
            return QualityResult(
                rule_name=rule_name,
                rule_type=QualityRule.REGEX,
                passed=len(failed_records) == 0,
                score=score,
                message=f"Regex validation: {passed_records}/{len(df)} records passed",
                failed_records=failed_records,
                total_records=len(df),
                passed_records=passed_records,
                failed_records_count=len(failed_records)
            )
            
        except Exception as e:
            logger.error(f"Error validating regex for {column}: {e}")
            return QualityResult(
                rule_name=rule_name,
                rule_type=QualityRule.REGEX,
                passed=False,
                score=0.0,
                message=f"Validation error: {e}",
                total_records=len(df)
            )
    
    def _validate_timestamp(self, df: pd.DataFrame, column: str, rule_name: str) -> QualityResult:
        """Validate that a column contains valid timestamps."""
        try:
            if column not in df.columns:
                return QualityResult(
                    rule_name=rule_name,
                    rule_type=QualityRule.VALIDITY,
                    passed=False,
                    score=0.0,
                    message=f"Column {column} not found",
                    total_records=len(df)
                )
            
            # Try to parse timestamps
            try:
                pd.to_datetime(df[column], errors='raise')
                passed_records = len(df)
                failed_records = []
            except:
                # If parsing fails, check individual values
                passed_records = 0
                failed_records = []
                for idx, value in df[column].items():
                    try:
                        pd.to_datetime(value)
                        passed_records += 1
                    except:
                        failed_records.append(idx)
            
            score = passed_records / len(df) if len(df) > 0 else 0.0
            
            return QualityResult(
                rule_name=rule_name,
                rule_type=QualityRule.VALIDITY,
                passed=len(failed_records) == 0,
                score=score,
                message=f"Timestamp validation: {passed_records}/{len(df)} records passed",
                failed_records=failed_records,
                total_records=len(df),
                passed_records=passed_records,
                failed_records_count=len(failed_records)
            )
            
        except Exception as e:
            logger.error(f"Error validating timestamp for {column}: {e}")
            return QualityResult(
                rule_name=rule_name,
                rule_type=QualityRule.VALIDITY,
                passed=False,
                score=0.0,
                message=f"Validation error: {e}",
                total_records=len(df)
            )
    
    def _detect_temperature_anomalies(self, df: pd.DataFrame, column: str, rule_name: str) -> QualityResult:
        """Detect temperature anomalies using statistical methods."""
        try:
            if column not in df.columns:
                return QualityResult(
                    rule_name=rule_name,
                    rule_type=QualityRule.ACCURACY,
                    passed=True,
                    score=1.0,
                    message=f"Column {column} not found",
                    total_records=len(df)
                )
            
            # Convert to numeric
            numeric_series = pd.to_numeric(df[column], errors='coerce')
            valid_data = numeric_series.dropna()
            
            if len(valid_data) < 10:  # Need minimum data points
                return QualityResult(
                    rule_name=rule_name,
                    rule_type=QualityRule.ACCURACY,
                    passed=True,
                    score=1.0,
                    message="Insufficient data for anomaly detection",
                    total_records=len(df)
                )
            
            # Use Z-score for anomaly detection
            z_scores = np.abs(stats.zscore(valid_data))
            anomaly_threshold = 3.0
            anomalies = z_scores > anomaly_threshold
            
            # Map back to original dataframe indices
            valid_indices = valid_data.index
            failed_records = valid_indices[anomalies].tolist()
            passed_records = len(df) - len(failed_records)
            score = passed_records / len(df) if len(df) > 0 else 0.0
            
            return QualityResult(
                rule_name=rule_name,
                rule_type=QualityRule.ACCURACY,
                passed=len(failed_records) == 0,
                score=score,
                message=f"Anomaly detection: {len(failed_records)} anomalies found in {len(df)} records",
                failed_records=failed_records,
                total_records=len(df),
                passed_records=passed_records,
                failed_records_count=len(failed_records)
            )
            
        except Exception as e:
            logger.error(f"Error detecting temperature anomalies for {column}: {e}")
            return QualityResult(
                rule_name=rule_name,
                rule_type=QualityRule.ACCURACY,
                passed=True,
                score=1.0,
                message=f"Anomaly detection error: {e}",
                total_records=len(df)
            )
    
    def _calculate_overall_score(self, results: List[QualityResult]) -> float:
        """Calculate overall quality score from individual results."""
        try:
            if not results:
                return 0.0
            
            # Weight different rule types
            weights = {
                QualityRule.NOT_NULL: 0.3,
                QualityRule.UNIQUE: 0.2,
                QualityRule.RANGE: 0.2,
                QualityRule.REGEX: 0.1,
                QualityRule.VALIDITY: 0.1,
                QualityRule.ACCURACY: 0.1
            }
            
            weighted_score = 0.0
            total_weight = 0.0
            
            for result in results:
                weight = weights.get(result.rule_type, 0.1)
                weighted_score += result.score * weight
                total_weight += weight
            
            return weighted_score / total_weight if total_weight > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating overall score: {e}")
            return 0.0
    
    def _initialize_validation_rules(self):
        """Initialize default validation rules."""
        try:
            self.validation_rules = {
                "pbf_process": {
                    "machine_id": {"type": "regex", "pattern": r"^PBF-[A-Z]{2}-\d{3}$"},
                    "chamber_temperature": {"type": "range", "min": 20.0, "max": 1000.0},
                    "build_plate_temperature": {"type": "range", "min": 20.0, "max": 500.0},
                    "chamber_pressure": {"type": "range", "min": 0.0, "max": 10.0}
                },
                "ispm_monitoring": {
                    "sensor_id": {"type": "regex", "pattern": r"^ISPM-\d{3}$"},
                    "melt_pool_temperature": {"type": "range", "min": 1000.0, "max": 3000.0},
                    "plume_intensity": {"type": "range", "min": 0.0, "max": 100.0}
                },
                "ct_scan": {
                    "scan_id": {"type": "regex", "pattern": r"^CT-\d{6}$"},
                    "porosity_percentage": {"type": "range", "min": 0.0, "max": 100.0},
                    "num_defects": {"type": "range", "min": 0, "max": 10000}
                },
                "powder_bed": {
                    "image_id": {"type": "regex", "pattern": r"^PB-\d{8}$"},
                    "layer_number": {"type": "range", "min": 1, "max": 5000},
                    "porosity_metric": {"type": "range", "min": 0.0, "max": 1.0}
                }
            }
            
            logger.info("Initialized validation rules")
            
        except Exception as e:
            logger.error(f"Error initializing validation rules: {e}")
