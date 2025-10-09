"""
Laser Parameter Predictor Model

This module implements a model for predicting optimal laser parameters
(laser power, scan speed, hatch spacing) for PBF-LB/M processes.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time

from ..base_model import BaseModel

logger = logging.getLogger(__name__)


class LaserParameterPredictor(BaseModel):
    """
    Model for predicting optimal laser parameters for PBF-LB/M processes.
    
    This model predicts laser power, scan speed, and hatch spacing based on:
    - Material properties
    - Part geometry
    - Desired quality requirements
    - Environmental conditions
    """
    
    def __init__(self, config_manager=None):
        """
        Initialize the laser parameter predictor.
        
        Args:
            config_manager: Configuration manager instance
        """
        super().__init__('laser_parameter_predictor', config_manager)
        self.model_type = self.model_info.get('algorithm', 'neural_network')
        
        logger.info(f"Initialized LaserParameterPredictor with algorithm: {self.model_type}")
    
    def build_model(self) -> Any:
        """
        Build the model architecture based on configuration.
        
        Returns:
            Built model instance
        """
        try:
            arch_config = self.architecture
            algorithm = arch_config.get('algorithm', 'neural_network')
            
            if algorithm == 'neural_network':
                return self._build_neural_network()
            elif algorithm == 'random_forest':
                return self._build_random_forest()
            else:
                raise ValueError(f"Unsupported algorithm: {algorithm}")
                
        except Exception as e:
            logger.error(f"Failed to build model: {e}")
            raise
    
    def _build_neural_network(self) -> tf.keras.Model:
        """Build neural network model."""
        model = Sequential()
        
        # Input layer
        input_dim = len(self.get_feature_names())
        model.add(Dense(128, activation='relu', input_dim=input_dim))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        
        # Hidden layers
        model.add(Dense(64, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.2))
        
        # Output layer (3 parameters: laser_power, scan_speed, hatch_spacing)
        model.add(Dense(3, activation='linear'))
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=self.training_config.get('learning_rate', 0.001)),
            loss='mse',
            metrics=['mae', 'mse']
        )
        
        return model
    
    def _build_random_forest(self) -> RandomForestRegressor:
        """Build random forest model."""
        return RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training targets (laser_power, scan_speed, hatch_spacing)
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            
        Returns:
            Training history dictionary
        """
        try:
            # Build model if not already built
            if self.model is None:
                self.model = self.build_model()
            
            start_time = time.time()
            
            if self.model_type == 'neural_network':
                history = self._train_neural_network(X_train, y_train, X_val, y_val)
            else:
                history = self._train_random_forest(X_train, y_train, X_val, y_val)
            
            training_time = time.time() - start_time
            
            self.training_history = {
                'training_time': training_time,
                'history': history
            }
            
            self.is_trained = True
            
            logger.info(f"Model training completed in {training_time:.2f} seconds")
            
            return self.training_history
            
        except Exception as e:
            logger.error(f"Failed to train model: {e}")
            raise
    
    def _train_neural_network(self, X_train: np.ndarray, y_train: np.ndarray, 
                             X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Train neural network model."""
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=self.training_config.get('early_stopping', {}).get('patience', 10),
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        ]
        
        history = self.model.fit(
            X_train, y_train,
            batch_size=self.training_config.get('batch_size', 32),
            epochs=self.training_config.get('epochs', 100),
            validation_data=(X_val, y_val) if X_val is not None else None,
            callbacks=callbacks,
            verbose=1
        )
        
        return history.history
    
    def _train_random_forest(self, X_train: np.ndarray, y_train: np.ndarray, 
                            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Train random forest model."""
        self.model.fit(X_train, y_train)
        
        # Calculate training metrics
        train_pred = self.model.predict(X_train)
        train_mse = mean_squared_error(y_train, train_pred)
        train_mae = mean_absolute_error(y_train, train_pred)
        train_r2 = r2_score(y_train, train_pred)
        
        history = {
            'loss': [train_mse],
            'mae': [train_mae],
            'r2': [train_r2]
        }
        
        if X_val is not None and y_val is not None:
            val_pred = self.model.predict(X_val)
            val_mse = mean_squared_error(y_val, val_pred)
            val_mae = mean_absolute_error(y_val, val_pred)
            val_r2 = r2_score(y_val, val_pred)
            
            history.update({
                'val_loss': [val_mse],
                'val_mae': [val_mae],
                'val_r2': [val_r2]
            })
        
        return history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Input features
            
        Returns:
            Predictions array with shape (n_samples, 3) for [laser_power, scan_speed, hatch_spacing]
        """
        try:
            if not self.is_trained:
                raise ValueError("Model must be trained before making predictions")
            
            predictions = self.model.predict(X, verbose=0)
            
            # Ensure predictions are within valid ranges
            predictions = self._constrain_predictions(predictions)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Failed to make predictions: {e}")
            raise
    
    def _constrain_predictions(self, predictions: np.ndarray) -> np.ndarray:
        """
        Constrain predictions to valid parameter ranges.
        
        Args:
            predictions: Raw predictions
            
        Returns:
            Constrained predictions
        """
        # Get parameter ranges from configuration
        features = self.data_config.get('features', [])
        ranges = {}
        
        for feature in features:
            if isinstance(feature, dict):
                name = feature.get('name', '')
                if name in ['laser_power', 'scan_speed', 'hatch_spacing']:
                    ranges[name] = feature.get('range', [0, 1000])
        
        # Constrain predictions
        constrained = predictions.copy()
        
        if 'laser_power' in ranges:
            constrained[:, 0] = np.clip(constrained[:, 0], ranges['laser_power'][0], ranges['laser_power'][1])
        if 'scan_speed' in ranges:
            constrained[:, 1] = np.clip(constrained[:, 1], ranges['scan_speed'][0], ranges['scan_speed'][1])
        if 'hatch_spacing' in ranges:
            constrained[:, 2] = np.clip(constrained[:, 2], ranges['hatch_spacing'][0], ranges['hatch_spacing'][1])
        
        return constrained
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance on test data.
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Evaluation metrics dictionary
        """
        try:
            if not self.is_trained:
                raise ValueError("Model must be trained before evaluation")
            
            # Make predictions
            predictions = self.predict(X_test)
            
            # Calculate metrics for each parameter
            metrics = {}
            parameter_names = ['laser_power', 'scan_speed', 'hatch_spacing']
            
            for i, param_name in enumerate(parameter_names):
                mse = mean_squared_error(y_test[:, i], predictions[:, i])
                mae = mean_absolute_error(y_test[:, i], predictions[:, i])
                r2 = r2_score(y_test[:, i], predictions[:, i])
                
                metrics.update({
                    f'{param_name}_mse': mse,
                    f'{param_name}_mae': mae,
                    f'{param_name}_r2': r2
                })
            
            # Overall metrics
            overall_mse = mean_squared_error(y_test, predictions)
            overall_mae = mean_absolute_error(y_test, predictions)
            overall_r2 = r2_score(y_test, predictions)
            
            metrics.update({
                'overall_mse': overall_mse,
                'overall_mae': overall_mae,
                'overall_r2': overall_r2
            })
            
            # Calculate latency
            start_time = time.time()
            _ = self.predict(X_test[:10])
            latency = (time.time() - start_time) / 10 * 1000
            
            metrics['latency_ms'] = latency
            
            self.evaluation_metrics = metrics
            
            logger.info(f"Model evaluation completed:")
            logger.info(f"  Overall MSE: {overall_mse:.4f}")
            logger.info(f"  Overall MAE: {overall_mae:.4f}")
            logger.info(f"  Overall R²: {overall_r2:.4f}")
            logger.info(f"  Latency: {latency:.2f} ms")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to evaluate model: {e}")
            raise
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance.
        
        Returns:
            Feature importance dictionary
        """
        if self.model_type == 'random_forest' and hasattr(self.model, 'feature_importances_'):
            feature_names = self.get_feature_names()
            importances = self.model.feature_importances_
            return dict(zip(feature_names, importances))
        else:
            # For neural networks, return zero importance (could implement SHAP later)
            feature_names = self.get_feature_names()
            return {name: 0.0 for name in feature_names}
    
    def predict_optimal_parameters(self, material_type: str, part_geometry: Dict[str, float], 
                                 quality_requirements: Dict[str, float]) -> Dict[str, float]:
        """
        Predict optimal laser parameters for specific conditions.
        
        Args:
            material_type: Type of material
            part_geometry: Part geometry parameters
            quality_requirements: Quality requirements
            
        Returns:
            Dictionary with optimal parameters
        """
        try:
            # Create feature vector from inputs
            features = self._create_feature_vector(material_type, part_geometry, quality_requirements)
            
            # Make prediction
            prediction = self.predict(features.reshape(1, -1))[0]
            
            return {
                'laser_power': float(prediction[0]),
                'scan_speed': float(prediction[1]),
                'hatch_spacing': float(prediction[2])
            }
            
        except Exception as e:
            logger.error(f"Failed to predict optimal parameters: {e}")
            raise
    
    def _create_feature_vector(self, material_type: str, part_geometry: Dict[str, float], 
                              quality_requirements: Dict[str, float]) -> np.ndarray:
        """
        Create feature vector from input parameters.
        
        Args:
            material_type: Type of material
            part_geometry: Part geometry parameters
            quality_requirements: Quality requirements
            
        Returns:
            Feature vector
        """
        # This would be implemented based on the specific feature engineering
        # For now, return a placeholder
        feature_names = self.get_feature_names()
        features = np.zeros(len(feature_names))
        
        # Map inputs to features (this would be more sophisticated in practice)
        if 'material_type' in feature_names:
            material_idx = feature_names.index('material_type')
            features[material_idx] = hash(material_type) % 1000  # Simple encoding
        
        # Add geometry and quality features
        for key, value in part_geometry.items():
            if key in feature_names:
                features[feature_names.index(key)] = value
        
        for key, value in quality_requirements.items():
            if key in feature_names:
                features[feature_names.index(key)] = value
        
        return features
