"""
Quality Configuration

This module provides data quality configuration for PBF-LB/M data processing.
It handles quality rules, SLA settings, and remediation configurations.
"""

import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum


class QualityRuleType(Enum):
    """Quality rule types for PBF-LB/M data"""
    COMPLETENESS = "completeness"
    ACCURACY = "accuracy"
    CONSISTENCY = "consistency"
    VALIDITY = "validity"
    UNIQUENESS = "uniqueness"
    TIMELINESS = "timeliness"
    CUSTOM = "custom"


class QualitySeverity(Enum):
    """Quality issue severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class QualityRule:
    """Data quality rule configuration"""
    
    # Rule identification
    rule_id: str
    rule_name: str
    rule_type: QualityRuleType
    description: str
    
    # Rule settings
    enabled: bool = True
    severity: QualitySeverity = QualitySeverity.MEDIUM
    
    # Data source settings
    data_source: str = ""
    table_name: str = ""
    column_name: str = ""
    
    # Rule parameters
    parameters: Dict[str, Any] = None
    
    # Thresholds
    warning_threshold: float = 0.95
    error_threshold: float = 0.90
    
    # Actions
    actions: List[str] = None  # ["alert", "quarantine", "remediate"]
    
    def __post_init__(self):
        """Initialize default values if not provided"""
        if self.parameters is None:
            self.parameters = {}
        if self.actions is None:
            self.actions = ["alert"]


@dataclass
class SLASettings:
    """SLA settings for PBF-LB/M data quality"""
    
    # SLA identification
    sla_id: str
    sla_name: str
    data_source: str
    
    # SLA metrics
    availability_target: float = 0.99  # 99%
    latency_target: int = 300  # 5 minutes
    throughput_target: int = 1000  # records per second
    
    # Quality targets
    quality_score_target: float = 0.95
    error_rate_target: float = 0.01  # 1%
    
    # Monitoring settings
    monitoring_interval: int = 300  # 5 minutes
    alert_threshold: float = 0.90
    
    # Escalation settings
    escalation_levels: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Initialize default escalation levels if not provided"""
        if self.escalation_levels is None:
            self.escalation_levels = [
                {
                    "level": 1,
                    "threshold": 0.90,
                    "action": "alert",
                    "recipients": ["data-team@company.com"]
                },
                {
                    "level": 2,
                    "threshold": 0.80,
                    "action": "escalate",
                    "recipients": ["data-manager@company.com"]
                },
                {
                    "level": 3,
                    "threshold": 0.70,
                    "action": "critical",
                    "recipients": ["data-director@company.com"]
                }
            ]


@dataclass
class RemediationConfig:
    """Remediation configuration for PBF-LB/M data quality issues"""
    
    # Remediation identification
    remediation_id: str
    remediation_name: str
    quality_rule_id: str
    
    # Remediation settings
    enabled: bool = True
    auto_remediate: bool = False
    
    # Remediation actions
    actions: List[str] = None  # ["clean", "enrich", "validate", "reject"]
    
    # Remediation parameters
    parameters: Dict[str, Any] = None
    
    # Retry settings
    max_retries: int = 3
    retry_delay: int = 60  # seconds
    
    # Notification settings
    notify_on_success: bool = True
    notify_on_failure: bool = True
    notification_recipients: List[str] = None
    
    def __post_init__(self):
        """Initialize default values if not provided"""
        if self.actions is None:
            self.actions = ["clean", "validate"]
        if self.parameters is None:
            self.parameters = {}
        if self.notification_recipients is None:
            self.notification_recipients = ["data-team@company.com"]


class QualityConfig:
    """Quality configuration manager for PBF-LB/M data processing"""
    
    def __init__(self):
        self.quality_rules: Dict[str, QualityRule] = {}
        self.sla_settings: Dict[str, SLASettings] = {}
        self.remediation_configs: Dict[str, RemediationConfig] = {}
        self._load_default_configurations()
    
    def _load_default_configurations(self):
        """Load default quality configurations"""
        # Default quality rules for PBF-LB/M data
        self.quality_rules = {
            "pbf_process_completeness": QualityRule(
                rule_id="pbf_process_completeness",
                rule_name="PBF Process Data Completeness",
                rule_type=QualityRuleType.COMPLETENESS,
                description="Check completeness of PBF process data",
                data_source="pbf_process",
                table_name="pbf_process_data",
                parameters={"required_columns": ["process_id", "timestamp", "laser_power", "scan_speed"]},
                warning_threshold=0.95,
                error_threshold=0.90,
                actions=["alert", "quarantine"]
            ),
            "ispm_monitoring_accuracy": QualityRule(
                rule_id="ispm_monitoring_accuracy",
                rule_name="ISPM Monitoring Data Accuracy",
                rule_type=QualityRuleType.ACCURACY,
                description="Check accuracy of ISPM monitoring data",
                data_source="ispm_monitoring",
                table_name="ispm_monitoring_data",
                parameters={"valid_ranges": {"temperature": [20, 200], "pressure": [0, 100]}},
                warning_threshold=0.95,
                error_threshold=0.90,
                actions=["alert", "remediate"]
            ),
            "ct_scan_consistency": QualityRule(
                rule_id="ct_scan_consistency",
                rule_name="CT Scan Data Consistency",
                rule_type=QualityRuleType.CONSISTENCY,
                description="Check consistency of CT scan data",
                data_source="ct_scan",
                table_name="ct_scan_data",
                parameters={"consistency_checks": ["voxel_dimensions", "resolution", "scan_parameters"]},
                warning_threshold=0.95,
                error_threshold=0.90,
                actions=["alert", "validate"]
            ),
            "powder_bed_validity": QualityRule(
                rule_id="powder_bed_validity",
                rule_name="Powder Bed Data Validity",
                rule_type=QualityRuleType.VALIDITY,
                description="Check validity of powder bed data",
                data_source="powder_bed",
                table_name="powder_bed_data",
                parameters={"valid_formats": ["image/jpeg", "image/png"], "max_file_size": 10485760},
                warning_threshold=0.95,
                error_threshold=0.90,
                actions=["alert", "reject"]
            )
        }
        
        # Default SLA settings
        self.sla_settings = {
            "pbf_process_sla": SLASettings(
                sla_id="pbf_process_sla",
                sla_name="PBF Process Data SLA",
                data_source="pbf_process",
                availability_target=0.99,
                latency_target=300,
                throughput_target=1000,
                quality_score_target=0.95
            ),
            "ispm_monitoring_sla": SLASettings(
                sla_id="ispm_monitoring_sla",
                sla_name="ISPM Monitoring Data SLA",
                data_source="ispm_monitoring",
                availability_target=0.99,
                latency_target=60,
                throughput_target=5000,
                quality_score_target=0.95
            ),
            "ct_scan_sla": SLASettings(
                sla_id="ct_scan_sla",
                sla_name="CT Scan Data SLA",
                data_source="ct_scan",
                availability_target=0.95,
                latency_target=600,
                throughput_target=100,
                quality_score_target=0.90
            ),
            "powder_bed_sla": SLASettings(
                sla_id="powder_bed_sla",
                sla_name="Powder Bed Data SLA",
                data_source="powder_bed",
                availability_target=0.99,
                latency_target=120,
                throughput_target=2000,
                quality_score_target=0.95
            )
        }
        
        # Default remediation configurations
        self.remediation_configs = {
            "pbf_process_remediation": RemediationConfig(
                remediation_id="pbf_process_remediation",
                remediation_name="PBF Process Data Remediation",
                quality_rule_id="pbf_process_completeness",
                actions=["clean", "enrich", "validate"],
                parameters={"clean_method": "interpolation", "enrich_source": "historical_data"}
            ),
            "ispm_monitoring_remediation": RemediationConfig(
                remediation_id="ispm_monitoring_remediation",
                remediation_name="ISPM Monitoring Data Remediation",
                quality_rule_id="ispm_monitoring_accuracy",
                actions=["clean", "validate"],
                parameters={"clean_method": "outlier_removal", "validation_rules": "range_check"}
            ),
            "ct_scan_remediation": RemediationConfig(
                remediation_id="ct_scan_remediation",
                remediation_name="CT Scan Data Remediation",
                quality_rule_id="ct_scan_consistency",
                actions=["validate", "reject"],
                parameters={"validation_rules": "schema_check", "rejection_threshold": 0.80}
            ),
            "powder_bed_remediation": RemediationConfig(
                remediation_id="powder_bed_remediation",
                remediation_name="Powder Bed Data Remediation",
                quality_rule_id="powder_bed_validity",
                actions=["clean", "reject"],
                parameters={"clean_method": "format_conversion", "rejection_threshold": 0.70}
            )
        }
    
    @classmethod
    def from_environment(cls) -> 'QualityConfig':
        """Create quality configuration from environment variables"""
        config = cls()
        
        # Update configuration from environment variables if needed
        # This can be extended based on specific requirements
        
        return config
    
    def get_quality_rules(self) -> Dict[str, QualityRule]:
        """Get all quality rules"""
        return self.quality_rules.copy()
    
    def get_quality_rule(self, rule_id: str) -> Optional[QualityRule]:
        """Get quality rule by ID"""
        return self.quality_rules.get(rule_id)
    
    def get_sla_settings(self) -> Dict[str, SLASettings]:
        """Get all SLA settings"""
        return self.sla_settings.copy()
    
    def get_sla_setting(self, sla_id: str) -> Optional[SLASettings]:
        """Get SLA setting by ID"""
        return self.sla_settings.get(sla_id)
    
    def get_remediation_configs(self) -> Dict[str, RemediationConfig]:
        """Get all remediation configurations"""
        return self.remediation_configs.copy()
    
    def get_remediation_config(self, remediation_id: str) -> Optional[RemediationConfig]:
        """Get remediation configuration by ID"""
        return self.remediation_configs.get(remediation_id)
    
    def add_quality_rule(self, rule: QualityRule) -> None:
        """Add a new quality rule"""
        self.quality_rules[rule.rule_id] = rule
    
    def add_sla_setting(self, sla: SLASettings) -> None:
        """Add a new SLA setting"""
        self.sla_settings[sla.sla_id] = sla
    
    def add_remediation_config(self, remediation: RemediationConfig) -> None:
        """Add a new remediation configuration"""
        self.remediation_configs[remediation.remediation_id] = remediation
    
    def get_quality_rules_by_data_source(self, data_source: str) -> List[QualityRule]:
        """Get quality rules for a specific data source"""
        return [rule for rule in self.quality_rules.values() if rule.data_source == data_source]
    
    def get_enabled_quality_rules(self) -> List[QualityRule]:
        """Get all enabled quality rules"""
        return [rule for rule in self.quality_rules.values() if rule.enabled]


# Global configuration instance
_quality_config: Optional[QualityConfig] = None


def get_quality_config() -> QualityConfig:
    """
    Get the global quality configuration instance.
    
    Returns:
        QualityConfig: The global quality configuration
    """
    global _quality_config
    if _quality_config is None:
        _quality_config = QualityConfig.from_environment()
    return _quality_config


def set_quality_config(config: QualityConfig) -> None:
    """
    Set the global quality configuration instance.
    
    Args:
        config: The quality configuration to set
    """
    global _quality_config
    _quality_config = config


def reset_quality_config() -> None:
    """Reset the global quality configuration to None."""
    global _quality_config
    _quality_config = None


def load_quality_config(config_path: Optional[str] = None) -> QualityConfig:
    """
    Load quality configuration from file or environment.
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        QualityConfig: Loaded quality configuration
    """
    # For now, just return from environment
    # TODO: Add file-based configuration loading
    return QualityConfig.from_environment()