"""
Surface Finish Predictor Model

This module implements a model for predicting surface finish parameters
(Ra, Rz, Rq) for PBF-LB/M manufactured parts based on process parameters.
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


class SurfaceFinishPredictor(BaseModel):
    """
    Model for predicting surface finish parameters for PBF-LB/M parts.
    
    This model predicts:
    - Ra (Arithmetic Average Roughness)
    - Rz (Maximum Height of the Profile)
    - Rq (Root Mean Square Roughness)
    - Surface texture characteristics
    """
    
    def __init__(self, config_manager=None):
        """
        Initialize the surface finish predictor.
        
        Args:
            config_manager: Configuration manager instance
        """
        super().__init__('surface_finish_predictor', config_manager)
        self.model_type = self.model_info.get('algorithm', 'neural_network')
        
        logger.info(f"Initialized SurfaceFinishPredictor with algorithm: {self.model_type}")
    
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
        
        # Output layer (3 surface finish parameters: Ra, Rz, Rq)
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
            y_train: Training targets (Ra, Rz, Rq values)
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
            Predictions array with surface finish parameters (Ra, Rz, Rq)
        """
        try:
            if not self.is_trained:
                raise ValueError("Model must be trained before making predictions")
            
            predictions = self.model.predict(X, verbose=0)
            
            # Ensure predictions are positive (surface roughness is always positive)
            predictions = np.maximum(predictions, 0.001)  # Minimum value of 0.001 μm
            
            return predictions
            
        except Exception as e:
            logger.error(f"Failed to make predictions: {e}")
            raise
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance on test data.
        
        Args:
            X_test: Test features
            y_test: Test targets (surface finish parameters)
            
        Returns:
            Evaluation metrics dictionary
        """
        try:
            if not self.is_trained:
                raise ValueError("Model must be trained before evaluation")
            
            # Make predictions
            predictions = self.predict(X_test)
            
            # Calculate metrics for each surface finish parameter
            metrics = {}
            parameter_names = ['Ra', 'Rz', 'Rq']
            
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
            # For neural networks, return zero importance
            feature_names = self.get_feature_names()
            return {name: 0.0 for name in feature_names}
    
    def predict_surface_finish(self, process_parameters: Dict[str, float], 
                              material_properties: Dict[str, float],
                              part_geometry: Dict[str, float],
                              build_orientation: Dict[str, float]) -> Dict[str, Any]:
        """
        Predict surface finish parameters for specific process conditions.
        
        Args:
            process_parameters: Process parameters (laser power, scan speed, etc.)
            material_properties: Material properties
            part_geometry: Part geometry parameters
            build_orientation: Build orientation parameters
            
        Returns:
            Dictionary with surface finish predictions and recommendations
        """
        try:
            # Create feature vector from inputs
            features = self._create_feature_vector(process_parameters, material_properties, 
                                                 part_geometry, build_orientation)
            
            # Make prediction
            surface_params = self.predict(features.reshape(1, -1))[0]
            
            # Calculate surface finish grade
            surface_grade = self._calculate_surface_grade(surface_params)
            
            # Generate improvement recommendations
            recommendations = self._generate_improvement_recommendations(surface_params, process_parameters)
            
            return {
                'surface_finish_parameters': {
                    'Ra': float(surface_params[0]),  # Arithmetic Average Roughness (μm)
                    'Rz': float(surface_params[1]),  # Maximum Height of the Profile (μm)
                    'Rq': float(surface_params[2])   # Root Mean Square Roughness (μm)
                },
                'surface_grade': surface_grade,
                'improvement_recommendations': recommendations,
                'surface_quality_assessment': self._assess_surface_quality(surface_params)
            }
            
        except Exception as e:
            logger.error(f"Failed to predict surface finish: {e}")
            raise
    
    def _create_feature_vector(self, process_parameters: Dict[str, float], 
                              material_properties: Dict[str, float],
                              part_geometry: Dict[str, float],
                              build_orientation: Dict[str, float]) -> np.ndarray:
        """
        Create feature vector from input parameters.
        
        Args:
            process_parameters: Process parameters
            material_properties: Material properties
            part_geometry: Part geometry parameters
            build_orientation: Build orientation parameters
            
        Returns:
            Feature vector
        """
        feature_names = self.get_feature_names()
        features = np.zeros(len(feature_names))
        
        # Combine all input dictionaries
        all_inputs = {**process_parameters, **material_properties, **part_geometry, **build_orientation}
        
        # Map inputs to features
        for key, value in all_inputs.items():
            if key in feature_names:
                features[feature_names.index(key)] = value
        
        return features
    
    def _calculate_surface_grade(self, surface_params: np.ndarray) -> str:
        """
        Calculate surface finish grade based on Ra value.
        
        Args:
            surface_params: Surface finish parameters [Ra, Rz, Rq]
            
        Returns:
            Surface finish grade (A, B, C, D, F)
        """
        ra = surface_params[0]
        
        if ra <= 0.8:
            return 'A'  # Excellent (Ra ≤ 0.8 μm)
        elif ra <= 1.6:
            return 'B'  # Good (Ra ≤ 1.6 μm)
        elif ra <= 3.2:
            return 'C'  # Acceptable (Ra ≤ 3.2 μm)
        elif ra <= 6.3:
            return 'D'  # Poor (Ra ≤ 6.3 μm)
        else:
            return 'F'  # Unacceptable (Ra > 6.3 μm)
    
    def _generate_improvement_recommendations(self, surface_params: np.ndarray, 
                                            process_parameters: Dict[str, float]) -> List[str]:
        """
        Generate recommendations for improving surface finish.
        
        Args:
            surface_params: Predicted surface finish parameters
            process_parameters: Current process parameters
            
        Returns:
            List of improvement recommendations
        """
        recommendations = []
        ra = surface_params[0]
        
        if ra > 3.2:  # Poor surface finish
            recommendations.append("Surface finish is poor. Consider the following improvements:")
            
            # Check laser power
            if 'laser_power' in process_parameters:
                if process_parameters['laser_power'] > 600:
                    recommendations.append("Reduce laser power to improve surface finish")
                elif process_parameters['laser_power'] < 300:
                    recommendations.append("Increase laser power for better fusion")
            
            # Check scan speed
            if 'scan_speed' in process_parameters:
                if process_parameters['scan_speed'] > 1200:
                    recommendations.append("Reduce scan speed for better surface quality")
                elif process_parameters['scan_speed'] < 600:
                    recommendations.append("Increase scan speed to reduce overheating")
            
            # Check hatch spacing
            if 'hatch_spacing' in process_parameters:
                if process_parameters['hatch_spacing'] > 0.15:
                    recommendations.append("Reduce hatch spacing for better surface coverage")
            
            # Check layer thickness
            if 'layer_thickness' in process_parameters:
                if process_parameters['layer_thickness'] > 0.05:
                    recommendations.append("Reduce layer thickness for finer surface finish")
        
        elif ra > 1.6:  # Acceptable but could be better
            recommendations.append("Surface finish is acceptable but could be improved:")
            recommendations.append("Consider fine-tuning process parameters for better surface quality")
        
        else:  # Good surface finish
            recommendations.append("Surface finish is excellent. Current parameters are well-optimized.")
        
        return recommendations
    
    def _assess_surface_quality(self, surface_params: np.ndarray) -> Dict[str, Any]:
        """
        Assess overall surface quality based on predicted parameters.
        
        Args:
            surface_params: Surface finish parameters [Ra, Rz, Rq]
            
        Returns:
            Dictionary with surface quality assessment
        """
        ra, rz, rq = surface_params
        
        # Calculate surface texture characteristics
        texture_ratio = rz / ra if ra > 0 else 0
        rms_ratio = rq / ra if ra > 0 else 0
        
        # Assess surface characteristics
        if texture_ratio > 8:
            texture_type = "Rough with high peaks"
        elif texture_ratio > 5:
            texture_type = "Moderately rough"
        else:
            texture_type = "Smooth"
        
        # Assess surface uniformity
        if rms_ratio > 1.2:
            uniformity = "Non-uniform"
        elif rms_ratio > 1.1:
            uniformity = "Moderately uniform"
        else:
            uniformity = "Uniform"
        
        return {
            'texture_type': texture_type,
            'uniformity': uniformity,
            'texture_ratio': float(texture_ratio),
            'rms_ratio': float(rms_ratio),
            'overall_quality': self._calculate_surface_grade(surface_params)
        }
