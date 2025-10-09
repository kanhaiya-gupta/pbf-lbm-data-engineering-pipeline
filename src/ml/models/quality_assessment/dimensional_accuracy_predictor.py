"""
Dimensional Accuracy Predictor Model

This module implements a model for predicting dimensional accuracy
and providing compensation recommendations for PBF-LB/M manufactured parts.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time

from ..base_model import BaseModel

logger = logging.getLogger(__name__)


class DimensionalAccuracyPredictor(BaseModel):
    """
    Model for predicting dimensional accuracy and providing compensation for PBF-LB/M parts.
    
    This model predicts:
    - Dimensional deviations (X, Y, Z axes)
    - Geometric accuracy metrics
    - Compensation recommendations
    - Tolerance compliance
    """
    
    def __init__(self, config_manager=None):
        """
        Initialize the dimensional accuracy predictor.
        
        Args:
            config_manager: Configuration manager instance
        """
        super().__init__('dimensional_accuracy_predictor', config_manager)
        self.model_type = self.model_info.get('algorithm', 'neural_network')
        
        logger.info(f"Initialized DimensionalAccuracyPredictor with algorithm: {self.model_type}")
    
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
        model.add(Dense(256, activation='relu', input_dim=input_dim))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        
        # Hidden layers
        model.add(Dense(128, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.2))
        
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.1))
        
        # Output layer (3 dimensions: X, Y, Z deviations)
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
            n_estimators=200,
            max_depth=15,
            random_state=42,
            n_jobs=-1
        )
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training targets (dimensional deviations in X, Y, Z)
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
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=self.training_config.get('early_stopping', {}).get('patience', 15),
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=7,
                min_lr=1e-6
            )
        ]
        
        history = self.model.fit(
            X_train, y_train,
            batch_size=self.training_config.get('batch_size', 32),
            epochs=self.training_config.get('epochs', 150),
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
            Predictions array with dimensional deviations (X, Y, Z)
        """
        try:
            if not self.is_trained:
                raise ValueError("Model must be trained before making predictions")
            
            predictions = self.model.predict(X, verbose=0)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Failed to make predictions: {e}")
            raise
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance on test data.
        
        Args:
            X_test: Test features
            y_test: Test targets (dimensional deviations)
            
        Returns:
            Evaluation metrics dictionary
        """
        try:
            if not self.is_trained:
                raise ValueError("Model must be trained before evaluation")
            
            # Make predictions
            predictions = self.predict(X_test)
            
            # Calculate metrics for each dimension
            metrics = {}
            dimension_names = ['X_deviation', 'Y_deviation', 'Z_deviation']
            
            for i, dim_name in enumerate(dimension_names):
                mse = mean_squared_error(y_test[:, i], predictions[:, i])
                mae = mean_absolute_error(y_test[:, i], predictions[:, i])
                r2 = r2_score(y_test[:, i], predictions[:, i])
                
                metrics.update({
                    f'{dim_name}_mse': mse,
                    f'{dim_name}_mae': mae,
                    f'{dim_name}_r2': r2
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
            # For neural networks, return zero importance
            feature_names = self.get_feature_names()
            return {name: 0.0 for name in feature_names}
    
    def predict_dimensional_accuracy(self, part_geometry: Dict[str, float], 
                                   process_parameters: Dict[str, float],
                                   material_properties: Dict[str, float],
                                   build_orientation: Dict[str, float]) -> Dict[str, Any]:
        """
        Predict dimensional accuracy for specific part and process conditions.
        
        Args:
            part_geometry: Part geometry parameters
            process_parameters: Process parameters
            material_properties: Material properties
            build_orientation: Build orientation parameters
            
        Returns:
            Dictionary with dimensional accuracy predictions and compensation
        """
        try:
            # Create feature vector from inputs
            features = self._create_feature_vector(part_geometry, process_parameters, 
                                                 material_properties, build_orientation)
            
            # Make prediction
            deviations = self.predict(features.reshape(1, -1))[0]
            
            # Calculate compensation recommendations
            compensation = self._calculate_compensation(deviations, part_geometry)
            
            # Check tolerance compliance
            tolerance_compliance = self._check_tolerance_compliance(deviations)
            
            return {
                'dimensional_deviations': {
                    'X_deviation': float(deviations[0]),
                    'Y_deviation': float(deviations[1]),
                    'Z_deviation': float(deviations[2])
                },
                'compensation_recommendations': compensation,
                'tolerance_compliance': tolerance_compliance,
                'accuracy_grade': self._calculate_accuracy_grade(deviations)
            }
            
        except Exception as e:
            logger.error(f"Failed to predict dimensional accuracy: {e}")
            raise
    
    def _create_feature_vector(self, part_geometry: Dict[str, float], 
                              process_parameters: Dict[str, float],
                              material_properties: Dict[str, float],
                              build_orientation: Dict[str, float]) -> np.ndarray:
        """
        Create feature vector from input parameters.
        
        Args:
            part_geometry: Part geometry parameters
            process_parameters: Process parameters
            material_properties: Material properties
            build_orientation: Build orientation parameters
            
        Returns:
            Feature vector
        """
        feature_names = self.get_feature_names()
        features = np.zeros(len(feature_names))
        
        # Combine all input dictionaries
        all_inputs = {**part_geometry, **process_parameters, **material_properties, **build_orientation}
        
        # Map inputs to features
        for key, value in all_inputs.items():
            if key in feature_names:
                features[feature_names.index(key)] = value
        
        return features
    
    def _calculate_compensation(self, deviations: np.ndarray, part_geometry: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate compensation recommendations based on predicted deviations.
        
        Args:
            deviations: Predicted dimensional deviations
            part_geometry: Part geometry parameters
            
        Returns:
            Dictionary with compensation recommendations
        """
        # Simple compensation calculation (in practice, this would be more sophisticated)
        compensation = {
            'X_compensation': -deviations[0] * 0.8,  # 80% compensation
            'Y_compensation': -deviations[1] * 0.8,
            'Z_compensation': -deviations[2] * 0.8
        }
        
        # Add process parameter adjustments
        if abs(deviations[0]) > 0.1:  # X deviation > 0.1mm
            compensation['laser_power_adjustment'] = -deviations[0] * 10  # Adjust laser power
        if abs(deviations[1]) > 0.1:  # Y deviation > 0.1mm
            compensation['scan_speed_adjustment'] = deviations[1] * 50  # Adjust scan speed
        if abs(deviations[2]) > 0.1:  # Z deviation > 0.1mm
            compensation['layer_thickness_adjustment'] = -deviations[2] * 0.1  # Adjust layer thickness
        
        return compensation
    
    def _check_tolerance_compliance(self, deviations: np.ndarray) -> Dict[str, Any]:
        """
        Check if predicted deviations are within tolerance limits.
        
        Args:
            deviations: Predicted dimensional deviations
            
        Returns:
            Dictionary with tolerance compliance information
        """
        # Define tolerance limits (in practice, these would come from configuration)
        tolerance_limits = {
            'X_tolerance': 0.1,  # ±0.1mm
            'Y_tolerance': 0.1,  # ±0.1mm
            'Z_tolerance': 0.05  # ±0.05mm
        }
        
        compliance = {
            'X_compliant': abs(deviations[0]) <= tolerance_limits['X_tolerance'],
            'Y_compliant': abs(deviations[1]) <= tolerance_limits['Y_tolerance'],
            'Z_compliant': abs(deviations[2]) <= tolerance_limits['Z_tolerance'],
            'overall_compliant': all([
                abs(deviations[0]) <= tolerance_limits['X_tolerance'],
                abs(deviations[1]) <= tolerance_limits['Y_tolerance'],
                abs(deviations[2]) <= tolerance_limits['Z_tolerance']
            ])
        }
        
        return compliance
    
    def _calculate_accuracy_grade(self, deviations: np.ndarray) -> str:
        """
        Calculate accuracy grade based on deviations.
        
        Args:
            deviations: Predicted dimensional deviations
            
        Returns:
            Accuracy grade (A, B, C, D, F)
        """
        max_deviation = np.max(np.abs(deviations))
        
        if max_deviation <= 0.05:
            return 'A'  # Excellent
        elif max_deviation <= 0.1:
            return 'B'  # Good
        elif max_deviation <= 0.2:
            return 'C'  # Acceptable
        elif max_deviation <= 0.5:
            return 'D'  # Poor
        else:
            return 'F'  # Unacceptable
