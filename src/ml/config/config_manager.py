"""
Configuration Manager for ML Components

This module provides a high-level configuration manager that combines multiple
configuration files and provides environment-specific configuration management.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import logging
from .config_loader import ConfigLoader

logger = logging.getLogger(__name__)


class ConfigManager:
    """
    High-level configuration manager for ML components.
    
    This class provides methods to load and combine multiple configuration files,
    handle environment-specific overrides, and provide a unified configuration interface.
    """
    
    def __init__(self, base_path: Union[str, Path] = "config/ml", environment: Optional[str] = None):
        """
        Initialize the configuration manager.
        
        Args:
            base_path: Base path to the ML configuration directory
            environment: Environment name (development, staging, production, testing)
                        If None, will try to get from ML_ENV environment variable
        """
        self.loader = ConfigLoader(base_path)
        self.environment = environment or os.getenv('ML_ENV', 'development')
        self._combined_configs: Dict[str, Dict[str, Any]] = {}
        
        logger.info(f"Initialized ConfigManager for environment: {self.environment}")
    
    def get_model_config(self, model_name: str, include_environment: bool = True) -> Dict[str, Any]:
        """
        Get model configuration with optional environment overrides.
        
        Args:
            model_name: Name of the model configuration
            include_environment: Whether to include environment-specific overrides
            
        Returns:
            Combined model configuration
        """
        try:
            # Load base model configuration
            config = self.loader.load_model_config(model_name)
            
            # Apply environment-specific overrides if requested
            if include_environment:
                env_config = self._get_environment_overrides('models', model_name)
                config = self._merge_configs(config, env_config)
            
            return config
            
        except FileNotFoundError:
            logger.error(f"Model configuration not found: {model_name}")
            raise
        except Exception as e:
            logger.error(f"Error loading model configuration {model_name}: {e}")
            raise
    
    def get_pipeline_config(self, pipeline_name: str, include_environment: bool = True) -> Dict[str, Any]:
        """
        Get pipeline configuration with optional environment overrides.
        
        Args:
            pipeline_name: Name of the pipeline configuration
            include_environment: Whether to include environment-specific overrides
            
        Returns:
            Combined pipeline configuration
        """
        try:
            # Load base pipeline configuration
            config = self.loader.load_pipeline_config(pipeline_name)
            
            # Apply environment-specific overrides if requested
            if include_environment:
                env_config = self._get_environment_overrides('pipelines', pipeline_name)
                config = self._merge_configs(config, env_config)
            
            return config
            
        except FileNotFoundError:
            logger.error(f"Pipeline configuration not found: {pipeline_name}")
            raise
        except Exception as e:
            logger.error(f"Error loading pipeline configuration {pipeline_name}: {e}")
            raise
    
    def get_feature_config(self, feature_name: str, include_environment: bool = True) -> Dict[str, Any]:
        """
        Get feature configuration with optional environment overrides.
        
        Args:
            feature_name: Name of the feature configuration
            include_environment: Whether to include environment-specific overrides
            
        Returns:
            Combined feature configuration
        """
        try:
            # Load base feature configuration
            config = self.loader.load_feature_config(feature_name)
            
            # Apply environment-specific overrides if requested
            if include_environment:
                env_config = self._get_environment_overrides('features', feature_name)
                config = self._merge_configs(config, env_config)
            
            return config
            
        except FileNotFoundError:
            logger.error(f"Feature configuration not found: {feature_name}")
            raise
        except Exception as e:
            logger.error(f"Error loading feature configuration {feature_name}: {e}")
            raise
    
    def get_serving_config(self, serving_name: str, include_environment: bool = True) -> Dict[str, Any]:
        """
        Get serving configuration with optional environment overrides.
        
        Args:
            serving_name: Name of the serving configuration
            include_environment: Whether to include environment-specific overrides
            
        Returns:
            Combined serving configuration
        """
        try:
            # Load base serving configuration
            config = self.loader.load_serving_config(serving_name)
            
            # Apply environment-specific overrides if requested
            if include_environment:
                env_config = self._get_environment_overrides('serving', serving_name)
                config = self._merge_configs(config, env_config)
            
            return config
            
        except FileNotFoundError:
            logger.error(f"Serving configuration not found: {serving_name}")
            raise
        except Exception as e:
            logger.error(f"Error loading serving configuration {serving_name}: {e}")
            raise
    
    def get_monitoring_config(self, monitoring_name: str, include_environment: bool = True) -> Dict[str, Any]:
        """
        Get monitoring configuration with optional environment overrides.
        
        Args:
            monitoring_name: Name of the monitoring configuration
            include_environment: Whether to include environment-specific overrides
            
        Returns:
            Combined monitoring configuration
        """
        try:
            # Load base monitoring configuration
            config = self.loader.load_monitoring_config(monitoring_name)
            
            # Apply environment-specific overrides if requested
            if include_environment:
                env_config = self._get_environment_overrides('monitoring', monitoring_name)
                config = self._merge_configs(config, env_config)
            
            return config
            
        except FileNotFoundError:
            logger.error(f"Monitoring configuration not found: {monitoring_name}")
            raise
        except Exception as e:
            logger.error(f"Error loading monitoring configuration {monitoring_name}: {e}")
            raise
    
    def get_global_config(self, config_name: str, include_environment: bool = True) -> Dict[str, Any]:
        """
        Get global configuration with optional environment overrides.
        
        Args:
            config_name: Name of the global configuration
            include_environment: Whether to include environment-specific overrides
            
        Returns:
            Combined global configuration
        """
        try:
            # Load base global configuration
            config = self.loader.load_global_config(config_name)
            
            # Apply environment-specific overrides if requested
            if include_environment:
                env_config = self._get_environment_overrides('global', config_name)
                config = self._merge_configs(config, env_config)
            
            return config
            
        except FileNotFoundError:
            logger.error(f"Global configuration not found: {config_name}")
            raise
        except Exception as e:
            logger.error(f"Error loading global configuration {config_name}: {e}")
            raise
    
    def get_environment_config(self) -> Dict[str, Any]:
        """
        Get the current environment configuration.
        
        Returns:
            Environment configuration dictionary
        """
        try:
            return self.loader.load_environment_config(self.environment)
        except FileNotFoundError:
            logger.warning(f"Environment configuration not found: {self.environment}")
            return {}
        except Exception as e:
            logger.error(f"Error loading environment configuration {self.environment}: {e}")
            return {}
    
    def list_available_configs(self, config_type: str) -> List[str]:
        """
        List available configuration files of a specific type.
        
        Args:
            config_type: Type of configuration (models, pipelines, features, serving, monitoring, global)
            
        Returns:
            List of available configuration names
        """
        config_dir = self.loader.base_path / config_type
        if not config_dir.exists():
            return []
        
        configs = []
        for file_path in config_dir.rglob("*.yaml"):
            relative_path = file_path.relative_to(config_dir)
            config_name = str(relative_path.with_suffix(''))
            configs.append(config_name)
        
        return sorted(configs)
    
    def _get_environment_overrides(self, config_type: str, config_name: str) -> Dict[str, Any]:
        """
        Get environment-specific configuration overrides.
        
        Args:
            config_type: Type of configuration
            config_name: Name of the configuration
            
        Returns:
            Environment-specific overrides
        """
        try:
            env_config = self.get_environment_config()
            return env_config.get(config_type, {}).get(config_name, {})
        except Exception as e:
            logger.debug(f"No environment overrides found for {config_type}/{config_name}: {e}")
            return {}
    
    def _merge_configs(self, base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively merge configuration dictionaries.
        
        Args:
            base_config: Base configuration dictionary
            override_config: Override configuration dictionary
            
        Returns:
            Merged configuration dictionary
        """
        if not override_config:
            return base_config.copy()
        
        merged = base_config.copy()
        
        for key, value in override_config.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self._merge_configs(merged[key], value)
            else:
                merged[key] = value
        
        return merged
    
    def clear_cache(self) -> None:
        """Clear all configuration caches."""
        self.loader.clear_cache()
        self._combined_configs.clear()
        logger.debug("All configuration caches cleared")
    
    def set_environment(self, environment: str) -> None:
        """
        Set the current environment.
        
        Args:
            environment: Environment name
        """
        self.environment = environment
        logger.info(f"Environment changed to: {environment}")
    
    def get_environment(self) -> str:
        """
        Get the current environment.
        
        Returns:
            Current environment name
        """
        return self.environment
