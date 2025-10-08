"""
Orchestration Configuration

This module provides orchestration configuration for PBF-LB/M data processing.
It handles Airflow, scheduling, and monitoring configurations.
"""

import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum


class ScheduleType(Enum):
    """Schedule types for PBF-LB/M data processing"""
    CRON = "cron"
    INTERVAL = "interval"
    ONCE = "once"
    TRIGGER = "trigger"


class JobPriority(Enum):
    """Job priority levels for PBF-LB/M data processing"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class AirflowConfig:
    """Airflow configuration for PBF-LB/M data processing"""
    
    # Connection settings
    webserver_url: str = "http://localhost:8080"
    api_url: str = "http://localhost:8080/api/v1"
    username: str = "admin"
    password: str = "admin"
    
    # DAG settings
    dag_folder: str = "dags"
    max_active_runs: int = 1
    catchup: bool = False
    schedule_interval: str = "@daily"
    
    # Task settings
    default_task_retries: int = 3
    default_task_retry_delay: int = 300  # 5 minutes
    default_task_timeout: int = 3600  # 1 hour
    
    # Resource settings
    default_pool: str = "default_pool"
    pool_slots: int = 128
    
    # Monitoring settings
    enable_monitoring: bool = True
    monitoring_interval: int = 60  # seconds
    
    # Email settings
    email_on_failure: bool = True
    email_on_retry: bool = False
    email_recipients: List[str] = None
    
    def __post_init__(self):
        """Initialize default email recipients if not provided"""
        if self.email_recipients is None:
            self.email_recipients = ["data-team@company.com"]


@dataclass
class SchedulingConfig:
    """Scheduling configuration for PBF-LB/M data processing"""
    
    # Schedule identification
    schedule_id: str
    schedule_name: str
    job_name: str
    
    # Schedule settings
    schedule_type: ScheduleType = ScheduleType.CRON
    schedule_expression: str = "0 0 * * *"  # Daily at midnight
    enabled: bool = True
    
    # Job settings
    priority: JobPriority = JobPriority.MEDIUM
    max_retries: int = 3
    retry_delay: int = 300  # 5 minutes
    timeout: int = 3600  # 1 hour
    
    # Dependencies
    dependencies: List[str] = None
    dependency_type: str = "all_success"  # "all_success", "one_success", "all_failed", "one_failed"
    
    # Resource requirements
    cpu_requirement: str = "1"
    memory_requirement: str = "1Gi"
    storage_requirement: str = "1Gi"
    
    # Notification settings
    notify_on_success: bool = False
    notify_on_failure: bool = True
    notification_recipients: List[str] = None
    
    def __post_init__(self):
        """Initialize default values if not provided"""
        if self.dependencies is None:
            self.dependencies = []
        if self.notification_recipients is None:
            self.notification_recipients = ["data-team@company.com"]


@dataclass
class MonitoringConfig:
    """Monitoring configuration for PBF-LB/M data processing"""
    
    # Monitoring settings
    enable_monitoring: bool = True
    monitoring_interval: int = 60  # seconds
    retention_days: int = 30
    
    # Metrics settings
    enable_metrics: bool = True
    metrics_endpoint: str = "http://localhost:9090"
    metrics_namespace: str = "pbf_lbm"
    
    # Logging settings
    enable_logging: bool = True
    log_level: str = "INFO"
    log_format: str = "json"
    log_retention_days: int = 7
    
    # Alerting settings
    enable_alerting: bool = True
    alert_endpoint: str = "http://localhost:9093"
    alert_rules: List[Dict[str, Any]] = None
    
    # Dashboard settings
    enable_dashboard: bool = True
    dashboard_url: str = "http://localhost:3000"
    dashboard_refresh_interval: int = 30  # seconds
    
    def __post_init__(self):
        """Initialize default alert rules if not provided"""
        if self.alert_rules is None:
            self.alert_rules = [
                {
                    "name": "pbf_process_failure",
                    "condition": "job_failure_rate > 0.1",
                    "severity": "high",
                    "description": "PBF process job failure rate is high"
                },
                {
                    "name": "ispm_monitoring_latency",
                    "condition": "processing_latency > 300",
                    "severity": "medium",
                    "description": "ISPM monitoring processing latency is high"
                },
                {
                    "name": "ct_scan_quality_degradation",
                    "condition": "quality_score < 0.90",
                    "severity": "high",
                    "description": "CT scan data quality has degraded"
                },
                {
                    "name": "powder_bed_throughput_drop",
                    "condition": "throughput < 1000",
                    "severity": "medium",
                    "description": "Powder bed data throughput has dropped"
                }
            ]


class OrchestrationConfig:
    """Orchestration configuration manager for PBF-LB/M data processing"""
    
    def __init__(self):
        self.airflow_config = AirflowConfig()
        self.scheduling_configs: Dict[str, SchedulingConfig] = {}
        self.monitoring_config = MonitoringConfig()
        self._load_default_configurations()
    
    def _load_default_configurations(self):
        """Load default orchestration configurations"""
        # Default Airflow configuration
        self.airflow_config = AirflowConfig()
        
        # Default scheduling configurations
        self.scheduling_configs = {
            "pbf_process_schedule": SchedulingConfig(
                schedule_id="pbf_process_schedule",
                schedule_name="PBF Process Data Processing Schedule",
                job_name="pbf_process_etl",
                schedule_type=ScheduleType.CRON,
                schedule_expression="0 2 * * *",  # Daily at 2 AM
                priority=JobPriority.HIGH,
                dependencies=[],
                cpu_requirement="2",
                memory_requirement="4Gi"
            ),
            "ispm_monitoring_schedule": SchedulingConfig(
                schedule_id="ispm_monitoring_schedule",
                schedule_name="ISPM Monitoring Data Processing Schedule",
                job_name="ispm_monitoring_etl",
                schedule_type=ScheduleType.INTERVAL,
                schedule_expression="300",  # Every 5 minutes
                priority=JobPriority.CRITICAL,
                dependencies=[],
                cpu_requirement="1",
                memory_requirement="2Gi"
            ),
            "ct_scan_schedule": SchedulingConfig(
                schedule_id="ct_scan_schedule",
                schedule_name="CT Scan Data Processing Schedule",
                job_name="ct_scan_etl",
                schedule_type=ScheduleType.CRON,
                schedule_expression="0 4 * * *",  # Daily at 4 AM
                priority=JobPriority.MEDIUM,
                dependencies=["pbf_process_schedule"],
                cpu_requirement="4",
                memory_requirement="8Gi"
            ),
            "powder_bed_schedule": SchedulingConfig(
                schedule_id="powder_bed_schedule",
                schedule_name="Powder Bed Data Processing Schedule",
                job_name="powder_bed_etl",
                schedule_type=ScheduleType.INTERVAL,
                schedule_expression="600",  # Every 10 minutes
                priority=JobPriority.HIGH,
                dependencies=[],
                cpu_requirement="1",
                memory_requirement="2Gi"
            ),
            "quality_monitoring_schedule": SchedulingConfig(
                schedule_id="quality_monitoring_schedule",
                schedule_name="Data Quality Monitoring Schedule",
                job_name="quality_monitoring",
                schedule_type=ScheduleType.INTERVAL,
                schedule_expression="1800",  # Every 30 minutes
                priority=JobPriority.MEDIUM,
                dependencies=[],
                cpu_requirement="1",
                memory_requirement="1Gi"
            )
        }
        
        # Default monitoring configuration
        self.monitoring_config = MonitoringConfig()
    
    @classmethod
    def from_environment(cls) -> 'OrchestrationConfig':
        """Create orchestration configuration from environment variables"""
        config = cls()
        
        # Update Airflow configuration from environment
        config.airflow_config.webserver_url = os.getenv("AIRFLOW_WEBSERVER_URL", "http://localhost:8080")
        config.airflow_config.api_url = os.getenv("AIRFLOW_API_URL", "http://localhost:8080/api/v1")
        config.airflow_config.username = os.getenv("AIRFLOW_USERNAME", "admin")
        config.airflow_config.password = os.getenv("AIRFLOW_PASSWORD", "admin")
        
        # Update monitoring configuration from environment
        config.monitoring_config.metrics_endpoint = os.getenv("MONITORING_METRICS_ENDPOINT", "http://localhost:9090")
        config.monitoring_config.dashboard_url = os.getenv("MONITORING_DASHBOARD_URL", "http://localhost:3000")
        
        return config
    
    def get_airflow_config(self) -> AirflowConfig:
        """Get Airflow configuration"""
        return self.airflow_config
    
    def get_scheduling_config(self, schedule_id: str) -> Optional[SchedulingConfig]:
        """Get scheduling configuration by ID"""
        return self.scheduling_configs.get(schedule_id)
    
    def get_all_scheduling_configs(self) -> Dict[str, SchedulingConfig]:
        """Get all scheduling configurations"""
        return self.scheduling_configs.copy()
    
    def get_monitoring_config(self) -> MonitoringConfig:
        """Get monitoring configuration"""
        return self.monitoring_config
    
    def get_scheduling_settings(self) -> Dict[str, Any]:
        """Get scheduling settings"""
        return {
            "schedules": {schedule_id: {
                "schedule_name": schedule.schedule_name,
                "job_name": schedule.job_name,
                "schedule_type": schedule.schedule_type.value,
                "schedule_expression": schedule.schedule_expression,
                "enabled": schedule.enabled,
                "priority": schedule.priority.value,
                "max_retries": schedule.max_retries,
                "retry_delay": schedule.retry_delay,
                "timeout": schedule.timeout,
                "dependencies": schedule.dependencies,
                "cpu_requirement": schedule.cpu_requirement,
                "memory_requirement": schedule.memory_requirement
            } for schedule_id, schedule in self.scheduling_configs.items()}
        }
    
    def get_monitoring_settings(self) -> Dict[str, Any]:
        """Get monitoring settings"""
        return {
            "enable_monitoring": self.monitoring_config.enable_monitoring,
            "monitoring_interval": self.monitoring_config.monitoring_interval,
            "retention_days": self.monitoring_config.retention_days,
            "enable_metrics": self.monitoring_config.enable_metrics,
            "metrics_endpoint": self.monitoring_config.metrics_endpoint,
            "metrics_namespace": self.monitoring_config.metrics_namespace,
            "enable_logging": self.monitoring_config.enable_logging,
            "log_level": self.monitoring_config.log_level,
            "log_retention_days": self.monitoring_config.log_retention_days,
            "enable_alerting": self.monitoring_config.enable_alerting,
            "alert_endpoint": self.monitoring_config.alert_endpoint,
            "alert_rules": self.monitoring_config.alert_rules,
            "enable_dashboard": self.monitoring_config.enable_dashboard,
            "dashboard_url": self.monitoring_config.dashboard_url
        }
    
    def add_scheduling_config(self, schedule_config: SchedulingConfig) -> None:
        """Add a new scheduling configuration"""
        self.scheduling_configs[schedule_config.schedule_id] = schedule_config
    
    def update_airflow_config(self, **kwargs) -> None:
        """Update Airflow configuration"""
        for key, value in kwargs.items():
            if hasattr(self.airflow_config, key):
                setattr(self.airflow_config, key, value)
    
    def update_monitoring_config(self, **kwargs) -> None:
        """Update monitoring configuration"""
        for key, value in kwargs.items():
            if hasattr(self.monitoring_config, key):
                setattr(self.monitoring_config, key, value)


# Global configuration instance
_orchestration_config: Optional[OrchestrationConfig] = None


def get_orchestration_config() -> OrchestrationConfig:
    """
    Get the global orchestration configuration instance.
    
    Returns:
        OrchestrationConfig: The global orchestration configuration
    """
    global _orchestration_config
    if _orchestration_config is None:
        _orchestration_config = OrchestrationConfig.from_environment()
    return _orchestration_config


def set_orchestration_config(config: OrchestrationConfig) -> None:
    """
    Set the global orchestration configuration instance.
    
    Args:
        config: The orchestration configuration to set
    """
    global _orchestration_config
    _orchestration_config = config


def reset_orchestration_config() -> None:
    """Reset the global orchestration configuration to None."""
    global _orchestration_config
    _orchestration_config = None


def load_orchestration_config(config_path: Optional[str] = None) -> OrchestrationConfig:
    """
    Load orchestration configuration from file or environment.
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        OrchestrationConfig: Loaded orchestration configuration
    """
    # For now, just return from environment
    # TODO: Add file-based configuration loading
    return OrchestrationConfig.from_environment()
    
    def get_enabled_schedules(self) -> List[SchedulingConfig]:
        """Get all enabled scheduling configurations"""
        return [schedule for schedule in self.scheduling_configs.values() if schedule.enabled]
    
    def get_schedules_by_priority(self, priority: JobPriority) -> List[SchedulingConfig]:
        """Get scheduling configurations by priority"""
        return [schedule for schedule in self.scheduling_configs.values() if schedule.priority == priority]
