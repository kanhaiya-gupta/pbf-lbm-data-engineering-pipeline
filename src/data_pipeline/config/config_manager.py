"""
Configuration Manager

This module provides a centralized configuration manager for the PBF-LB/M data pipeline.
It aggregates all configuration modules and provides convenient access methods.
"""

from typing import Dict, Any, Optional
from .pipeline_config import get_pipeline_config, PipelineConfig
from .etl_config import get_etl_config, ETLConfig
from .streaming_config import get_streaming_config, StreamingConfig
from .storage_config import get_storage_config, StorageConfig
from .quality_config import get_quality_config, QualityConfig
from .orchestration_config import get_orchestration_config, OrchestrationConfig


class ConfigManager:
    """
    Centralized configuration manager for PBF-LB/M data pipeline.
    
    This class provides a single point of access to all configuration modules
    and manages the lifecycle of configuration instances.
    """
    
    def __init__(self):
        """Initialize the configuration manager."""
        self._pipeline_config: Optional[PipelineConfig] = None
        self._etl_config: Optional[ETLConfig] = None
        self._streaming_config: Optional[StreamingConfig] = None
        self._storage_config: Optional[StorageConfig] = None
        self._quality_config: Optional[QualityConfig] = None
        self._orchestration_config: Optional[OrchestrationConfig] = None
    
    @property
    def pipeline_config(self) -> PipelineConfig:
        """Get pipeline configuration."""
        if self._pipeline_config is None:
            self._pipeline_config = get_pipeline_config()
        return self._pipeline_config
    
    @property
    def etl_config(self) -> ETLConfig:
        """Get ETL configuration."""
        if self._etl_config is None:
            self._etl_config = get_etl_config()
        return self._etl_config
    
    @property
    def streaming_config(self) -> StreamingConfig:
        """Get streaming configuration."""
        if self._streaming_config is None:
            self._streaming_config = get_streaming_config()
        return self._streaming_config
    
    @property
    def storage_config(self) -> StorageConfig:
        """Get storage configuration."""
        if self._storage_config is None:
            self._storage_config = get_storage_config()
        return self._storage_config
    
    @property
    def quality_config(self) -> QualityConfig:
        """Get quality configuration."""
        if self._quality_config is None:
            self._quality_config = get_quality_config()
        return self._quality_config
    
    @property
    def orchestration_config(self) -> OrchestrationConfig:
        """Get orchestration configuration."""
        if self._orchestration_config is None:
            self._orchestration_config = get_orchestration_config()
        return self._orchestration_config
    
    def get_all_configs(self) -> Dict[str, Any]:
        """Get all configuration settings."""
        return {
            "pipeline": self.pipeline_config.get_pipeline_settings(),
            "etl": self.etl_config.get_etl_settings(),
            "streaming": self.streaming_config.get_streaming_settings(),
            "storage": self.storage_config.get_storage_settings(),
            "quality": self.quality_config.get_quality_settings(),
            "orchestration": self.orchestration_config.get_scheduling_settings()
        }
    
    def reload_configs(self) -> None:
        """Reload all configurations from environment."""
        self._pipeline_config = None
        self._etl_config = None
        self._streaming_config = None
        self._storage_config = None
        self._quality_config = None
        self._orchestration_config = None


# Global configuration manager instance
_config_manager: Optional[ConfigManager] = None


def get_config_manager() -> ConfigManager:
    """
    Get the global configuration manager instance.
    
    Returns:
        ConfigManager: The global configuration manager
    """
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


# Convenience functions for backward compatibility
def get_pipeline_settings() -> Dict[str, Any]:
    """Get pipeline settings."""
    return get_config_manager().pipeline_config.get_pipeline_settings()


def get_spark_config() -> Dict[str, Any]:
    """Get Spark configuration."""
    return get_config_manager().etl_config.get_spark_config().to_dict()


def get_kafka_config() -> Dict[str, Any]:
    """Get Kafka configuration."""
    return get_config_manager().streaming_config.get_kafka_config().__dict__


def get_s3_config() -> Dict[str, Any]:
    """Get S3 configuration."""
    return get_config_manager().storage_config.get_s3_config().__dict__


def get_quality_rules() -> Dict[str, Any]:
    """Get quality rules."""
    return {rule_id: rule.__dict__ for rule_id, rule in get_config_manager().quality_config.get_quality_rules().items()}


# Alias for backward compatibility
config_manager = get_config_manager
