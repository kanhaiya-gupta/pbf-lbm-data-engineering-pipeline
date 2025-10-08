"""
Main Pipeline Configuration

This module provides the main pipeline configuration for PBF-LB/M data processing.
It handles general pipeline settings, environment configuration, and feature flags.
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum


class Environment(Enum):
    """Environment types for PBF-LB/M pipeline"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass
class PipelineConfig:
    """Main pipeline configuration for PBF-LB/M data processing"""
    
    # Environment settings
    environment: Environment = Environment.DEVELOPMENT
    debug: bool = False
    log_level: str = "INFO"
    
    # Pipeline settings
    batch_size: int = 1000
    max_retries: int = 3
    retry_delay: int = 5
    timeout: int = 300
    
    # Performance settings
    max_workers: int = 4
    memory_limit: str = "2GB"
    cpu_limit: str = "2"
    
    # Data processing settings
    enable_streaming: bool = True
    enable_batch_processing: bool = True
    enable_cdc: bool = True
    enable_quality_checks: bool = True
    
    # Feature flags
    feature_flags: Dict[str, bool] = None
    
    def __post_init__(self):
        """Initialize default feature flags if not provided"""
        if self.feature_flags is None:
            self.feature_flags = {
                "enable_spark_optimization": True,
                "enable_delta_lake": True,
                "enable_real_time_monitoring": True,
                "enable_auto_scaling": False,
                "enable_ml_integration": True,
                "enable_graph_analytics": True
            }
    
    @classmethod
    def from_environment(cls) -> 'PipelineConfig':
        """Create configuration from environment variables"""
        return cls(
            environment=Environment(os.getenv("PIPELINE_ENV", "development")),
            debug=os.getenv("PIPELINE_DEBUG", "false").lower() == "true",
            log_level=os.getenv("PIPELINE_LOG_LEVEL", "INFO"),
            batch_size=int(os.getenv("PIPELINE_BATCH_SIZE", "1000")),
            max_retries=int(os.getenv("PIPELINE_MAX_RETRIES", "3")),
            retry_delay=int(os.getenv("PIPELINE_RETRY_DELAY", "5")),
            timeout=int(os.getenv("PIPELINE_TIMEOUT", "300")),
            max_workers=int(os.getenv("PIPELINE_MAX_WORKERS", "4")),
            memory_limit=os.getenv("PIPELINE_MEMORY_LIMIT", "2GB"),
            cpu_limit=os.getenv("PIPELINE_CPU_LIMIT", "2"),
            enable_streaming=os.getenv("PIPELINE_ENABLE_STREAMING", "true").lower() == "true",
            enable_batch_processing=os.getenv("PIPELINE_ENABLE_BATCH", "true").lower() == "true",
            enable_cdc=os.getenv("PIPELINE_ENABLE_CDC", "true").lower() == "true",
            enable_quality_checks=os.getenv("PIPELINE_ENABLE_QUALITY", "true").lower() == "true"
        )
    
    def get_pipeline_settings(self) -> Dict[str, Any]:
        """Get general pipeline settings"""
        return {
            "environment": self.environment.value,
            "debug": self.debug,
            "log_level": self.log_level,
            "batch_size": self.batch_size,
            "max_retries": self.max_retries,
            "retry_delay": self.retry_delay,
            "timeout": self.timeout,
            "max_workers": self.max_workers,
            "memory_limit": self.memory_limit,
            "cpu_limit": self.cpu_limit
        }
    
    def get_environment_config(self) -> Dict[str, Any]:
        """Get environment-specific configuration"""
        base_config = {
            "environment": self.environment.value,
            "debug": self.debug,
            "log_level": self.log_level
        }
        
        if self.environment == Environment.DEVELOPMENT:
            base_config.update({
                "enable_streaming": True,
                "enable_batch_processing": True,
                "enable_cdc": True,
                "enable_quality_checks": True,
                "max_workers": 2,
                "memory_limit": "1GB"
            })
        elif self.environment == Environment.STAGING:
            base_config.update({
                "enable_streaming": True,
                "enable_batch_processing": True,
                "enable_cdc": True,
                "enable_quality_checks": True,
                "max_workers": 4,
                "memory_limit": "2GB"
            })
        elif self.environment == Environment.PRODUCTION:
            base_config.update({
                "enable_streaming": True,
                "enable_batch_processing": True,
                "enable_cdc": True,
                "enable_quality_checks": True,
                "max_workers": 8,
                "memory_limit": "4GB"
            })
        
        return base_config
    
    def get_feature_flags(self) -> Dict[str, bool]:
        """Get feature flags configuration"""
        return self.feature_flags.copy()
    
    def is_feature_enabled(self, feature_name: str) -> bool:
        """Check if a feature is enabled"""
        return self.feature_flags.get(feature_name, False)
    
    def enable_feature(self, feature_name: str) -> None:
        """Enable a feature flag"""
        self.feature_flags[feature_name] = True
    
    def disable_feature(self, feature_name: str) -> None:
        """Disable a feature flag"""
        self.feature_flags[feature_name] = False


# Global configuration instance
_pipeline_config: Optional[PipelineConfig] = None


def get_pipeline_config() -> PipelineConfig:
    """
    Get the global pipeline configuration instance.
    
    Returns:
        PipelineConfig: The global pipeline configuration
    """
    global _pipeline_config
    if _pipeline_config is None:
        _pipeline_config = PipelineConfig.from_environment()
    return _pipeline_config


def set_pipeline_config(config: PipelineConfig) -> None:
    """
    Set the global pipeline configuration instance.
    
    Args:
        config: The pipeline configuration to set
    """
    global _pipeline_config
    _pipeline_config = config


def reset_pipeline_config() -> None:
    """Reset the global pipeline configuration to None."""
    global _pipeline_config
    _pipeline_config = None


def load_pipeline_config(config_path: Optional[str] = None) -> PipelineConfig:
    """
    Load pipeline configuration from file or environment.
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        PipelineConfig: Loaded pipeline configuration
    """
    # For now, just return from environment
    # TODO: Add file-based configuration loading
    return PipelineConfig.from_environment()


# Alias for backward compatibility
PipelineSettings = PipelineConfig