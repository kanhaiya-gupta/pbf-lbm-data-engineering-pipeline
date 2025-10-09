"""
Base Model Class for ML Models

This module provides the base class that all ML models in the PBF-LB/M system inherit from.
It provides common functionality for model loading, training, prediction, and evaluation.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, List, Tuple
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import json
from datetime import datetime

from ..config import ConfigManager

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """
    Base class for all ML models in the PBF-LB/M system.
    
    This class provides common functionality for:
    - Configuration management
    - Model training and evaluation
    - Model saving and loading
    - Prediction and inference
    - Performance monitoring
    """
    
    def __init__(self, model_name: str, config_manager: Optional[ConfigManager] = None):
        """
        Initialize the base model.
        
        Args:
            model_name: Name of the model (should match YAML config file name)
            config_manager: Configuration manager instance
        """
        self.model_name = model_name
        self.config_manager = config_manager or ConfigManager()
        self.config = self._load_config()
        self.model = None
        self.is_trained = False
        self.training_history = None
        self.evaluation_metrics = None
        
        logger.info(f"Initialized {self.__class__.__name__} with model name: {model_name}")
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Load model configuration from YAML file.
        
        Returns:
            Model configuration dictionary
        """
        try:
            config = self.config_manager.get_model_config(self.model_name)
            logger.debug(f"Loaded configuration for model: {self.model_name}")
            return config
        except Exception as e:
            logger.error(f"Failed to load configuration for model {self.model_name}: {e}")
            raise
    
    @property
    def model_info(self) -> Dict[str, Any]:
        """
        Get model information from configuration.
        
        Returns:
            Model information dictionary
        """
        return self.config.get('model', {})
    
    @property
    def architecture(self) -> Dict[str, Any]:
        """
        Get model architecture configuration.
        
        Returns:
            Architecture configuration dictionary
        """
        return self.config.get('architecture', {})
    
    @property
    def training_config(self) -> Dict[str, Any]:
        """
        Get training configuration.
        
        Returns:
            Training configuration dictionary
        """
        return self.config.get('training', {})
    
    @property
    def data_config(self) -> Dict[str, Any]:
        """
        Get data configuration.
        
        Returns:
            Data configuration dictionary
        """
        return self.config.get('data', {})
    
    @property
    def performance_config(self) -> Dict[str, Any]:
        """
        Get performance requirements configuration.
        
        Returns:
            Performance configuration dictionary
        """
        return self.config.get('performance', {})
    
    @abstractmethod
    def build_model(self) -> Any:
        """
        Build the model architecture.
        
        Returns:
            Built model instance
        """
        pass
    
    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            
        Returns:
            Training history dictionary
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Input features
            
        Returns:
            Predictions
        """
        pass
    
    @abstractmethod
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance on test data.
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Evaluation metrics dictionary
        """
        pass
    
    def save_model(self, model_path: Union[str, Path]) -> None:
        """
        Save the trained model to disk.
        
        Args:
            model_path: Path to save the model
        """
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            if hasattr(self.model, 'save'):
                # For TensorFlow/Keras models
                self.model.save(str(model_path))
            else:
                # For scikit-learn models
                joblib.dump(self.model, model_path)
            
            # Save model metadata
            metadata = {
                'model_name': self.model_name,
                'model_type': self.model_info.get('type', 'unknown'),
                'algorithm': self.model_info.get('algorithm', 'unknown'),
                'version': self.model_info.get('version', '1.0.0'),
                'training_date': datetime.now().isoformat(),
                'is_trained': self.is_trained,
                'evaluation_metrics': self.evaluation_metrics,
                'config': self.config
            }
            
            metadata_path = model_path.parent / f"{model_path.stem}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            logger.info(f"Model saved to: {model_path}")
            
        except Exception as e:
            logger.error(f"Failed to save model to {model_path}: {e}")
            raise
    
    def load_model(self, model_path: Union[str, Path]) -> None:
        """
        Load a trained model from disk.
        
        Args:
            model_path: Path to the saved model
        """
        model_path = Path(model_path)
        
        try:
            if model_path.suffix == '.h5' or (model_path.is_dir() and 'saved_model.pb' in str(model_path)):
                # For TensorFlow/Keras models
                import tensorflow as tf
                self.model = tf.keras.models.load_model(str(model_path))
            else:
                # For scikit-learn models
                self.model = joblib.load(model_path)
            
            # Load model metadata
            metadata_path = model_path.parent / f"{model_path.stem}_metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    self.evaluation_metrics = metadata.get('evaluation_metrics')
                    self.is_trained = metadata.get('is_trained', False)
            
            logger.info(f"Model loaded from: {model_path}")
            
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}")
            raise
    
    def get_feature_names(self) -> List[str]:
        """
        Get feature names from configuration.
        
        Returns:
            List of feature names
        """
        features = self.data_config.get('features', [])
        if isinstance(features, list) and features and isinstance(features[0], dict):
            return [feature.get('name', f'feature_{i}') for i, feature in enumerate(features)]
        elif isinstance(features, list):
            return features
        else:
            return []
    
    def get_target_info(self) -> Dict[str, Any]:
        """
        Get target variable information from configuration.
        
        Returns:
            Target information dictionary
        """
        return self.data_config.get('target', {})
    
    def validate_input(self, X: np.ndarray) -> bool:
        """
        Validate input data against configuration requirements.
        
        Args:
            X: Input data to validate
            
        Returns:
            True if validation passes, False otherwise
        """
        try:
            # Check input shape
            expected_shape = self.architecture.get('input_shape')
            if expected_shape and X.shape[1:] != tuple(expected_shape[1:]):
                logger.warning(f"Input shape mismatch. Expected: {expected_shape}, Got: {X.shape}")
                return False
            
            # Check for NaN values
            if np.isnan(X).any():
                logger.warning("Input contains NaN values")
                return False
            
            # Check for infinite values
            if np.isinf(X).any():
                logger.warning("Input contains infinite values")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Input validation failed: {e}")
            return False
    
    def get_performance_thresholds(self) -> Dict[str, float]:
        """
        Get performance thresholds from configuration.
        
        Returns:
            Performance thresholds dictionary
        """
        return {
            'accuracy_threshold': self.performance_config.get('accuracy_threshold', 0.0),
            'precision_threshold': self.performance_config.get('precision_threshold', 0.0),
            'recall_threshold': self.performance_config.get('recall_threshold', 0.0),
            'f1_threshold': self.performance_config.get('f1_threshold', 0.0),
            'latency_threshold': self.performance_config.get('latency_threshold', float('inf'))
        }
    
    def meets_performance_requirements(self, metrics: Dict[str, float]) -> bool:
        """
        Check if model meets performance requirements.
        
        Args:
            metrics: Model performance metrics
            
        Returns:
            True if requirements are met, False otherwise
        """
        thresholds = self.get_performance_thresholds()
        
        for metric, threshold in thresholds.items():
            if metric in metrics and metrics[metric] < threshold:
                logger.warning(f"Performance requirement not met for {metric}: {metrics[metric]} < {threshold}")
                return False
        
        return True
    
    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get model summary information.
        
        Returns:
            Model summary dictionary
        """
        return {
            'model_name': self.model_name,
            'model_type': self.model_info.get('type', 'unknown'),
            'algorithm': self.model_info.get('algorithm', 'unknown'),
            'version': self.model_info.get('version', '1.0.0'),
            'is_trained': self.is_trained,
            'feature_count': len(self.get_feature_names()),
            'input_shape': self.architecture.get('input_shape'),
            'output_shape': self.architecture.get('output_shape'),
            'evaluation_metrics': self.evaluation_metrics,
            'performance_thresholds': self.get_performance_thresholds()
        }
    
    def __repr__(self) -> str:
        """String representation of the model."""
        return f"{self.__class__.__name__}(model_name='{self.model_name}', is_trained={self.is_trained})"
