"""
Simple YAML Configuration Loader

This module provides a lightweight configuration loader for reading YAML configuration files
from the config/ml/ directory structure.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
import logging

logger = logging.getLogger(__name__)


class ConfigLoader:
    """
    Simple YAML configuration loader for ML configurations.
    
    This class provides methods to load YAML configuration files from the config/ml/
    directory structure with caching and error handling.
    """
    
    def __init__(self, base_path: Union[str, Path] = "config/ml"):
        """
        Initialize the configuration loader.
        
        Args:
            base_path: Base path to the ML configuration directory
        """
        self.base_path = Path(base_path)
        self._cache: Dict[str, Dict[str, Any]] = {}
        
        if not self.base_path.exists():
            logger.warning(f"Configuration base path does not exist: {self.base_path}")
    
    def load_config(self, config_path: Union[str, Path], use_cache: bool = True) -> Dict[str, Any]:
        """
        Load a YAML configuration file.
        
        Args:
            config_path: Path to the configuration file (relative to base_path)
            use_cache: Whether to use cached configuration if available
            
        Returns:
            Dictionary containing the configuration data
            
        Raises:
            FileNotFoundError: If the configuration file doesn't exist
            yaml.YAMLError: If the YAML file is malformed
        """
        config_path = Path(config_path)
        full_path = self.base_path / config_path
        
        # Check cache first
        cache_key = str(config_path)
        if use_cache and cache_key in self._cache:
            logger.debug(f"Loading cached configuration: {config_path}")
            return self._cache[cache_key]
        
        # Load from file
        if not full_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {full_path}")
        
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            if config is None:
                config = {}
            
            # Cache the configuration
            if use_cache:
                self._cache[cache_key] = config
            
            logger.debug(f"Loaded configuration: {config_path}")
            return config
            
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML file {full_path}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading configuration file {full_path}: {e}")
            raise
    
    def load_model_config(self, model_name: str, use_cache: bool = True) -> Dict[str, Any]:
        """
        Load a model configuration file.
        
        Args:
            model_name: Name of the model configuration file (without .yaml extension)
            use_cache: Whether to use cached configuration if available
            
        Returns:
            Dictionary containing the model configuration
        """
        return self.load_config(f"models/{model_name}.yaml", use_cache)
    
    def load_pipeline_config(self, pipeline_name: str, use_cache: bool = True) -> Dict[str, Any]:
        """
        Load a pipeline configuration file.
        
        Args:
            pipeline_name: Name of the pipeline configuration file (without .yaml extension)
            use_cache: Whether to use cached configuration if available
            
        Returns:
            Dictionary containing the pipeline configuration
        """
        return self.load_config(f"pipelines/{pipeline_name}.yaml", use_cache)
    
    def load_feature_config(self, feature_name: str, use_cache: bool = True) -> Dict[str, Any]:
        """
        Load a feature configuration file.
        
        Args:
            feature_name: Name of the feature configuration file (without .yaml extension)
            use_cache: Whether to use cached configuration if available
            
        Returns:
            Dictionary containing the feature configuration
        """
        return self.load_config(f"features/{feature_name}.yaml", use_cache)
    
    def load_serving_config(self, serving_name: str, use_cache: bool = True) -> Dict[str, Any]:
        """
        Load a serving configuration file.
        
        Args:
            serving_name: Name of the serving configuration file (without .yaml extension)
            use_cache: Whether to use cached configuration if available
            
        Returns:
            Dictionary containing the serving configuration
        """
        return self.load_config(f"serving/{serving_name}.yaml", use_cache)
    
    def load_monitoring_config(self, monitoring_name: str, use_cache: bool = True) -> Dict[str, Any]:
        """
        Load a monitoring configuration file.
        
        Args:
            monitoring_name: Name of the monitoring configuration file (without .yaml extension)
            use_cache: Whether to use cached configuration if available
            
        Returns:
            Dictionary containing the monitoring configuration
        """
        return self.load_config(f"monitoring/{monitoring_name}.yaml", use_cache)
    
    def load_environment_config(self, environment: str, use_cache: bool = True) -> Dict[str, Any]:
        """
        Load an environment configuration file.
        
        Args:
            environment: Environment name (development, staging, production, testing)
            use_cache: Whether to use cached configuration if available
            
        Returns:
            Dictionary containing the environment configuration
        """
        return self.load_config(f"environments/{environment}.yaml", use_cache)
    
    def load_global_config(self, config_name: str, use_cache: bool = True) -> Dict[str, Any]:
        """
        Load a global configuration file.
        
        Args:
            config_name: Name of the global configuration file (without .yaml extension)
            use_cache: Whether to use cached configuration if available
            
        Returns:
            Dictionary containing the global configuration
        """
        return self.load_config(f"global/{config_name}.yaml", use_cache)
    
    def clear_cache(self) -> None:
        """Clear the configuration cache."""
        self._cache.clear()
        logger.debug("Configuration cache cleared")
    
    def get_cached_configs(self) -> list:
        """
        Get list of cached configuration file paths.
        
        Returns:
            List of cached configuration file paths
        """
        return list(self._cache.keys())
    
    def reload_config(self, config_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Force reload a configuration file, bypassing cache.
        
        Args:
            config_path: Path to the configuration file (relative to base_path)
            
        Returns:
            Dictionary containing the configuration data
        """
        return self.load_config(config_path, use_cache=False)
