"""
Quality Score Predictor Model

This module implements a model for predicting overall quality scores
for PBF-LB/M manufactured parts based on process parameters and sensor data.
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


class QualityScorePredictor(BaseModel):
    """
    Model for predicting overall quality scores for PBF-LB/M parts.
    
    This model predicts quality scores (0-100) based on:
    - Process parameters (laser power, scan speed, etc.)
    - Sensor data (temperature, melt pool size, etc.)
    - Material properties
    - Part geometry
    - Environmental conditions
    """
    
    def __init__(self, config_manager=None):
        """
        Initialize the quality score predictor.
        
        Args:
            config_manager: Configuration manager instance
        """
        super().__init__('quality_score_predictor', config_manager)
        self.model_type = self.model_info.get('algorithm', 'neural_network')
        
        logger.info(f"Initialized QualityScorePredictor with algorithm: {self.model_type}")
    
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
        
        # Output layer (single quality score)
        model.add(Dense(1, activation='sigmoid'))  # Sigmoid for 0-1 range, will scale to 0-100
        
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
            y_train: Training targets (quality scores 0-100)
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            
        Returns:
            Training history dictionary
        """
        try:
            # Build model if not already built
            if self.model is None:
                self.model = self.build_model()
            
            # Normalize quality scores to 0-1 range for neural network
            if self.model_type == 'neural_network':
                y_train_norm = y_train / 100.0
                y_val_norm = y_val / 100.0 if y_val is not None else None
            else:
                y_train_norm = y_train
                y_val_norm = y_val
            
            start_time = time.time()
            
            if self.model_type == 'neural_network':
                history = self._train_neural_network(X_train, y_train_norm, X_val, y_val_norm)
            else:
                history = self._train_random_forest(X_train, y_train_norm, X_val, y_val_norm)
            
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
            Predictions array with quality scores (0-100)
        """
        try:
            if not self.is_trained:
                raise ValueError("Model must be trained before making predictions")
            
            predictions = self.model.predict(X, verbose=0)
            
            # Scale predictions back to 0-100 range for neural network
            if self.model_type == 'neural_network':
                predictions = predictions * 100.0
            
            # Ensure predictions are within valid range
            predictions = np.clip(predictions, 0, 100)
            
            return predictions.flatten()
            
        except Exception as e:
            logger.error(f"Failed to make predictions: {e}")
            raise
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance on test data.
        
        Args:
            X_test: Test features
            y_test: Test targets (quality scores 0-100)
            
        Returns:
            Evaluation metrics dictionary
        """
        try:
            if not self.is_trained:
                raise ValueError("Model must be trained before evaluation")
            
            # Make predictions
            predictions = self.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, predictions)
            mae = mean_absolute_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)
            
            # Calculate additional quality-specific metrics
            mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100  # Mean Absolute Percentage Error
            rmse = np.sqrt(mse)
            
            # Calculate latency
            start_time = time.time()
            _ = self.predict(X_test[:10])
            latency = (time.time() - start_time) / 10 * 1000
            
            self.evaluation_metrics = {
                'mse': mse,
                'mae': mae,
                'r2': r2,
                'mape': mape,
                'rmse': rmse,
                'latency_ms': latency,
                'test_samples': len(X_test)
            }
            
            logger.info(f"Model evaluation completed:")
            logger.info(f"  MSE: {mse:.4f}")
            logger.info(f"  MAE: {mae:.4f}")
            logger.info(f"  R²: {r2:.4f}")
            logger.info(f"  MAPE: {mape:.2f}%")
            logger.info(f"  RMSE: {rmse:.4f}")
            logger.info(f"  Latency: {latency:.2f} ms")
            
            return self.evaluation_metrics
            
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
    
    def predict_quality_score(self, process_parameters: Dict[str, float], 
                             sensor_data: Dict[str, float],
                             material_properties: Dict[str, float],
                             part_geometry: Dict[str, float]) -> Dict[str, float]:
        """
        Predict quality score for specific process conditions.
        
        Args:
            process_parameters: Process parameters (laser power, scan speed, etc.)
            sensor_data: Sensor data (temperature, melt pool size, etc.)
            material_properties: Material properties
            part_geometry: Part geometry parameters
            
        Returns:
            Dictionary with quality score and confidence
        """
        try:
            # Create feature vector from inputs
            features = self._create_feature_vector(process_parameters, sensor_data, 
                                                 material_properties, part_geometry)
            
            # Make prediction
            quality_score = self.predict(features.reshape(1, -1))[0]
            
            # Calculate confidence based on feature completeness and model uncertainty
            confidence = self._calculate_confidence(features, quality_score)
            
            return {
                'quality_score': float(quality_score),
                'confidence': float(confidence),
                'quality_grade': self._score_to_grade(quality_score)
            }
            
        except Exception as e:
            logger.error(f"Failed to predict quality score: {e}")
            raise
    
    def _create_feature_vector(self, process_parameters: Dict[str, float], 
                              sensor_data: Dict[str, float],
                              material_properties: Dict[str, float],
                              part_geometry: Dict[str, float]) -> np.ndarray:
        """
        Create feature vector from input parameters.
        
        Args:
            process_parameters: Process parameters
            sensor_data: Sensor data
            material_properties: Material properties
            part_geometry: Part geometry parameters
            
        Returns:
            Feature vector
        """
        feature_names = self.get_feature_names()
        features = np.zeros(len(feature_names))
        
        # Combine all input dictionaries
        all_inputs = {**process_parameters, **sensor_data, **material_properties, **part_geometry}
        
        # Map inputs to features
        for key, value in all_inputs.items():
            if key in feature_names:
                features[feature_names.index(key)] = value
        
        return features
    
    def _calculate_confidence(self, features: np.ndarray, quality_score: float) -> float:
        """
        Calculate confidence score for the prediction.
        
        Args:
            features: Feature vector
            quality_score: Predicted quality score
            
        Returns:
            Confidence score (0-1)
        """
        # Simple confidence calculation based on feature completeness
        # In practice, this could be more sophisticated using model uncertainty
        non_zero_features = np.count_nonzero(features)
        total_features = len(features)
        completeness = non_zero_features / total_features
        
        # Adjust confidence based on quality score (extreme scores are less confident)
        score_confidence = 1.0 - abs(quality_score - 50) / 50.0
        
        # Combine completeness and score confidence
        confidence = (completeness + score_confidence) / 2.0
        
        return np.clip(confidence, 0.0, 1.0)
    
    def _score_to_grade(self, score: float) -> str:
        """
        Convert quality score to letter grade.
        
        Args:
            score: Quality score (0-100)
            
        Returns:
            Quality grade (A, B, C, D, F)
        """
        if score >= 90:
            return 'A'
        elif score >= 80:
            return 'B'
        elif score >= 70:
            return 'C'
        elif score >= 60:
            return 'D'
        else:
            return 'F'
    
    def get_quality_insights(self, process_parameters: Dict[str, float], 
                           sensor_data: Dict[str, float],
                           material_properties: Dict[str, float],
                           part_geometry: Dict[str, float]) -> Dict[str, Any]:
        """
        Get detailed quality insights and recommendations.
        
        Args:
            process_parameters: Process parameters
            sensor_data: Sensor data
            material_properties: Material properties
            part_geometry: Part geometry parameters
            
        Returns:
            Dictionary with quality insights and recommendations
        """
        try:
            # Get quality score prediction
            quality_result = self.predict_quality_score(process_parameters, sensor_data, 
                                                      material_properties, part_geometry)
            
            # Get feature importance
            feature_importance = self.get_feature_importance()
            
            # Generate recommendations based on quality score
            recommendations = self._generate_recommendations(quality_result['quality_score'], 
                                                          process_parameters, feature_importance)
            
            return {
                'quality_score': quality_result['quality_score'],
                'quality_grade': quality_result['quality_grade'],
                'confidence': quality_result['confidence'],
                'feature_importance': feature_importance,
                'recommendations': recommendations,
                'risk_factors': self._identify_risk_factors(process_parameters, sensor_data)
            }
            
        except Exception as e:
            logger.error(f"Failed to get quality insights: {e}")
            raise
    
    def _generate_recommendations(self, quality_score: float, 
                                process_parameters: Dict[str, float],
                                feature_importance: Dict[str, float]) -> List[str]:
        """
        Generate recommendations based on quality score and feature importance.
        
        Args:
            quality_score: Predicted quality score
            process_parameters: Process parameters
            feature_importance: Feature importance scores
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        if quality_score < 70:
            recommendations.append("Quality score is below acceptable threshold. Consider adjusting process parameters.")
            
            # Find most important features with low values
            for feature, importance in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True):
                if feature in process_parameters and importance > 0.1:
                    recommendations.append(f"Consider optimizing {feature} (importance: {importance:.3f})")
        
        elif quality_score < 85:
            recommendations.append("Quality score is acceptable but could be improved.")
        
        else:
            recommendations.append("Quality score is excellent. Current parameters are well-optimized.")
        
        return recommendations
    
    def _identify_risk_factors(self, process_parameters: Dict[str, float], 
                             sensor_data: Dict[str, float]) -> List[str]:
        """
        Identify potential risk factors for quality issues.
        
        Args:
            process_parameters: Process parameters
            sensor_data: Sensor data
            
        Returns:
            List of identified risk factors
        """
        risk_factors = []
        
        # Check for extreme parameter values
        if 'laser_power' in process_parameters:
            if process_parameters['laser_power'] > 800:
                risk_factors.append("High laser power may cause overheating")
            elif process_parameters['laser_power'] < 200:
                risk_factors.append("Low laser power may cause incomplete melting")
        
        if 'scan_speed' in process_parameters:
            if process_parameters['scan_speed'] > 1500:
                risk_factors.append("High scan speed may cause poor fusion")
            elif process_parameters['scan_speed'] < 500:
                risk_factors.append("Low scan speed may cause overheating")
        
        # Check sensor data for anomalies
        if 'temperature' in sensor_data:
            if sensor_data['temperature'] > 2500:
                risk_factors.append("High temperature detected")
            elif sensor_data['temperature'] < 1000:
                risk_factors.append("Low temperature detected")
        
        return risk_factors
