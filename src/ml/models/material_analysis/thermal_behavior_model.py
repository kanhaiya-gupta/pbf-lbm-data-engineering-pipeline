"""
Thermal Behavior Model

This module implements a model for predicting thermal behavior and heat distribution
in PBF-LB/M manufacturing processes.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time

from ..base_model import BaseModel

logger = logging.getLogger(__name__)


class ThermalBehaviorModel(BaseModel):
    """
    Model for predicting thermal behavior and heat distribution in PBF-LB/M processes.
    
    This model predicts:
    - Temperature distribution in the build
    - Heat accumulation and dissipation
    - Thermal gradients and stress development
    - Cooling rates and solidification behavior
    - Thermal cycling effects
    - Melt pool temperature and size
    """
    
    def __init__(self, config_manager=None):
        """
        Initialize the thermal behavior model.
        
        Args:
            config_manager: Configuration manager instance
        """
        super().__init__('thermal_behavior_model', config_manager)
        self.model_type = self.model_info.get('algorithm', 'lstm')
        self.thermal_features = [
            'melt_pool_temperature', 'melt_pool_size', 'cooling_rate',
            'thermal_gradient', 'heat_accumulation', 'solidification_time',
            'thermal_stress', 'temperature_peak'
        ]
        
        logger.info(f"Initialized ThermalBehaviorModel with algorithm: {self.model_type}")
    
    def build_model(self) -> Any:
        """
        Build the model architecture based on configuration.
        
        Returns:
            Built model instance
        """
        try:
            arch_config = self.architecture
            algorithm = arch_config.get('algorithm', 'lstm')
            
            if algorithm == 'lstm':
                return self._build_lstm_model()
            elif algorithm == 'random_forest':
                return self._build_random_forest()
            else:
                raise ValueError(f"Unsupported algorithm: {algorithm}")
                
        except Exception as e:
            logger.error(f"Failed to build model: {e}")
            raise
    
    def _build_lstm_model(self) -> tf.keras.Model:
        """Build LSTM model for thermal behavior prediction."""
        model = Sequential()
        
        # Input layer
        input_shape = self.architecture.get('input_shape', [24, 12])
        model.add(LSTM(128, return_sequences=True, input_shape=tuple(input_shape[1:])))
        model.add(Dropout(0.3))
        
        # Hidden LSTM layers
        model.add(LSTM(64, return_sequences=True))
        model.add(Dropout(0.3))
        
        model.add(LSTM(32, return_sequences=False))
        model.add(Dropout(0.2))
        
        # Dense layers
        model.add(Dense(64, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.1))
        
        # Output layer (8 thermal features)
        model.add(Dense(8, activation='linear'))
        
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
            n_estimators=300,
            max_depth=20,
            random_state=42,
            n_jobs=-1
        )
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            X_train: Training features (time-series thermal data)
            y_train: Training targets (thermal behavior parameters)
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
            
            if self.model_type == 'lstm':
                history = self._train_lstm_model(X_train, y_train, X_val, y_val)
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
    
    def _train_lstm_model(self, X_train: np.ndarray, y_train: np.ndarray, 
                         X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Train LSTM model."""
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=self.training_config.get('early_stopping', {}).get('patience', 20),
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-6
            )
        ]
        
        history = self.model.fit(
            X_train, y_train,
            batch_size=self.training_config.get('batch_size', 32),
            epochs=self.training_config.get('epochs', 200),
            validation_data=(X_val, y_val) if X_val is not None else None,
            callbacks=callbacks,
            verbose=1
        )
        
        return history.history
    
    def _train_random_forest(self, X_train: np.ndarray, y_train: np.ndarray, 
                            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Train random forest model."""
        # Flatten time-series data for tree-based models
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        
        self.model.fit(X_train_flat, y_train)
        
        # Calculate training metrics
        train_pred = self.model.predict(X_train_flat)
        train_mse = mean_squared_error(y_train, train_pred)
        train_mae = mean_absolute_error(y_train, train_pred)
        train_r2 = r2_score(y_train, train_pred)
        
        history = {
            'loss': [train_mse],
            'mae': [train_mae],
            'r2': [train_r2]
        }
        
        if X_val is not None and y_val is not None:
            X_val_flat = X_val.reshape(X_val.shape[0], -1)
            val_pred = self.model.predict(X_val_flat)
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
            X: Input features (time-series thermal data)
            
        Returns:
            Predictions array with thermal behavior parameters
        """
        try:
            if not self.is_trained:
                raise ValueError("Model must be trained before making predictions")
            
            if self.model_type == 'lstm':
                predictions = self.model.predict(X, verbose=0)
            else:
                X_flat = X.reshape(X.shape[0], -1)
                predictions = self.model.predict(X_flat)
            
            # Ensure predictions are within valid ranges
            predictions = self._constrain_predictions(predictions)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Failed to make predictions: {e}")
            raise
    
    def _constrain_predictions(self, predictions: np.ndarray) -> np.ndarray:
        """
        Constrain predictions to valid ranges for thermal parameters.
        
        Args:
            predictions: Raw predictions
            
        Returns:
            Constrained predictions
        """
        constrained = predictions.copy()
        
        # Constrain melt pool temperature to reasonable range (1000-3000°C)
        constrained[:, 0] = np.clip(constrained[:, 0], 1000, 3000)
        
        # Constrain melt pool size to reasonable range (0.1-2.0 mm)
        constrained[:, 1] = np.clip(constrained[:, 1], 0.1, 2.0)
        
        # Constrain cooling rate to reasonable range (10-10000 K/s)
        constrained[:, 2] = np.clip(constrained[:, 2], 10, 10000)
        
        # Constrain thermal gradient to reasonable range (100-10000 K/mm)
        constrained[:, 3] = np.clip(constrained[:, 3], 100, 10000)
        
        # Constrain heat accumulation to reasonable range (0-1000 J)
        constrained[:, 4] = np.clip(constrained[:, 4], 0, 1000)
        
        # Constrain solidification time to reasonable range (0.001-1.0 s)
        constrained[:, 5] = np.clip(constrained[:, 5], 0.001, 1.0)
        
        # Constrain thermal stress to reasonable range (0-1000 MPa)
        constrained[:, 6] = np.clip(constrained[:, 6], 0, 1000)
        
        # Constrain temperature peak to reasonable range (1000-3000°C)
        constrained[:, 7] = np.clip(constrained[:, 7], 1000, 3000)
        
        return constrained
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance on test data.
        
        Args:
            X_test: Test features
            y_test: Test targets (thermal behavior parameters)
            
        Returns:
            Evaluation metrics dictionary
        """
        try:
            if not self.is_trained:
                raise ValueError("Model must be trained before evaluation")
            
            # Make predictions
            predictions = self.predict(X_test)
            
            # Calculate metrics for each thermal feature
            metrics = {}
            
            for i, feature_name in enumerate(self.thermal_features):
                mse = mean_squared_error(y_test[:, i], predictions[:, i])
                mae = mean_absolute_error(y_test[:, i], predictions[:, i])
                r2 = r2_score(y_test[:, i], predictions[:, i])
                
                metrics.update({
                    f'{feature_name}_mse': mse,
                    f'{feature_name}_mae': mae,
                    f'{feature_name}_r2': r2
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
            # For LSTM models, return zero importance
            feature_names = self.get_feature_names()
            return {name: 0.0 for name in feature_names}
    
    def predict_thermal_behavior(self, process_parameters: Dict[str, float], 
                               thermal_history: List[float],
                               build_geometry: Dict[str, float]) -> Dict[str, Any]:
        """
        Predict thermal behavior for specific process conditions.
        
        Args:
            process_parameters: Process parameters (laser power, scan speed, etc.)
            thermal_history: Historical thermal data
            build_geometry: Build geometry parameters
            
        Returns:
            Dictionary with thermal behavior predictions and analysis
        """
        try:
            # Create feature vector from inputs
            features = self._create_feature_vector(process_parameters, thermal_history, build_geometry)
            
            # Make prediction
            thermal_params = self.predict(features.reshape(1, -1))[0]
            
            # Analyze thermal behavior
            thermal_analysis = self._analyze_thermal_behavior(thermal_params)
            
            # Assess thermal stability
            stability_assessment = self._assess_thermal_stability(thermal_params, thermal_history)
            
            # Generate optimization recommendations
            recommendations = self._generate_thermal_optimization_recommendations(
                thermal_params, process_parameters
            )
            
            # Calculate thermal relationships
            thermal_relationships = self._calculate_thermal_relationships(thermal_params)
            
            return {
                'thermal_parameters': {
                    'melt_pool_temperature_c': float(thermal_params[0]),
                    'melt_pool_size_mm': float(thermal_params[1]),
                    'cooling_rate_k_per_s': float(thermal_params[2]),
                    'thermal_gradient_k_per_mm': float(thermal_params[3]),
                    'heat_accumulation_j': float(thermal_params[4]),
                    'solidification_time_s': float(thermal_params[5]),
                    'thermal_stress_mpa': float(thermal_params[6]),
                    'temperature_peak_c': float(thermal_params[7])
                },
                'thermal_analysis': thermal_analysis,
                'stability_assessment': stability_assessment,
                'optimization_recommendations': recommendations,
                'thermal_relationships': thermal_relationships,
                'thermal_grade': self._calculate_thermal_grade(thermal_params)
            }
            
        except Exception as e:
            logger.error(f"Failed to predict thermal behavior: {e}")
            raise
    
    def _create_feature_vector(self, process_parameters: Dict[str, float], 
                              thermal_history: List[float],
                              build_geometry: Dict[str, float]) -> np.ndarray:
        """
        Create feature vector from input parameters.
        
        Args:
            process_parameters: Process parameters
            thermal_history: Thermal history data
            build_geometry: Build geometry parameters
            
        Returns:
            Feature vector
        """
        feature_names = self.get_feature_names()
        features = np.zeros(len(feature_names))
        
        # Combine all input dictionaries
        all_inputs = {**process_parameters, **build_geometry}
        
        # Map inputs to features
        for key, value in all_inputs.items():
            if key in feature_names:
                features[feature_names.index(key)] = value
        
        # Add thermal history features
        if len(thermal_history) > 0:
            features[feature_names.index('thermal_history_mean')] = np.mean(thermal_history)
            features[feature_names.index('thermal_history_std')] = np.std(thermal_history)
            features[feature_names.index('thermal_history_trend')] = np.polyfit(range(len(thermal_history)), thermal_history, 1)[0]
        
        return features
    
    def _analyze_thermal_behavior(self, thermal_params: np.ndarray) -> Dict[str, Any]:
        """
        Analyze thermal behavior from predicted parameters.
        
        Args:
            thermal_params: Predicted thermal parameters
            
        Returns:
            Dictionary with thermal analysis
        """
        melt_pool_temp, melt_pool_size, cooling_rate, thermal_gradient, heat_accumulation, solidification_time, thermal_stress, temp_peak = thermal_params
        
        # Analyze melt pool characteristics
        if melt_pool_temp > 2500:
            melt_pool_category = 'high_temperature'
        elif melt_pool_temp > 2000:
            melt_pool_category = 'medium_temperature'
        else:
            melt_pool_category = 'low_temperature'
        
        if melt_pool_size > 1.0:
            melt_pool_size_category = 'large'
        elif melt_pool_size > 0.5:
            melt_pool_size_category = 'medium'
        else:
            melt_pool_size_category = 'small'
        
        # Analyze cooling behavior
        if cooling_rate > 5000:
            cooling_category = 'rapid'
        elif cooling_rate > 1000:
            cooling_category = 'moderate'
        else:
            cooling_category = 'slow'
        
        # Analyze thermal gradient
        if thermal_gradient > 5000:
            gradient_category = 'high'
        elif thermal_gradient > 2000:
            gradient_category = 'medium'
        else:
            gradient_category = 'low'
        
        # Analyze heat accumulation
        if heat_accumulation > 500:
            heat_accumulation_category = 'high'
        elif heat_accumulation > 200:
            heat_accumulation_category = 'medium'
        else:
            heat_accumulation_category = 'low'
        
        # Analyze thermal stress
        if thermal_stress > 500:
            stress_category = 'high'
        elif thermal_stress > 200:
            stress_category = 'medium'
        else:
            stress_category = 'low'
        
        return {
            'melt_pool_category': melt_pool_category,
            'melt_pool_size_category': melt_pool_size_category,
            'cooling_category': cooling_category,
            'gradient_category': gradient_category,
            'heat_accumulation_category': heat_accumulation_category,
            'stress_category': stress_category,
            'solidification_behavior': self._analyze_solidification_behavior(solidification_time, cooling_rate),
            'thermal_efficiency': self._calculate_thermal_efficiency(thermal_params)
        }
    
    def _analyze_solidification_behavior(self, solidification_time: float, cooling_rate: float) -> Dict[str, Any]:
        """
        Analyze solidification behavior.
        
        Args:
            solidification_time: Solidification time
            cooling_rate: Cooling rate
            
        Returns:
            Dictionary with solidification analysis
        """
        if solidification_time < 0.01:
            solidification_type = 'rapid'
        elif solidification_time < 0.1:
            solidification_type = 'moderate'
        else:
            solidification_type = 'slow'
        
        # Calculate solidification front velocity
        solidification_velocity = 1.0 / solidification_time if solidification_time > 0 else 0
        
        return {
            'solidification_type': solidification_type,
            'solidification_velocity': float(solidification_velocity),
            'cooling_efficiency': float(cooling_rate / solidification_time) if solidification_time > 0 else 0
        }
    
    def _calculate_thermal_efficiency(self, thermal_params: np.ndarray) -> float:
        """
        Calculate thermal efficiency based on thermal parameters.
        
        Args:
            thermal_params: Thermal parameters
            
        Returns:
            Thermal efficiency score (0-1)
        """
        melt_pool_temp, melt_pool_size, cooling_rate, thermal_gradient, heat_accumulation, solidification_time, thermal_stress, temp_peak = thermal_params
        
        # Efficiency factors
        temp_efficiency = 1.0 - abs(melt_pool_temp - 2000) / 2000  # Optimal around 2000°C
        size_efficiency = 1.0 - abs(melt_pool_size - 0.8) / 0.8  # Optimal around 0.8mm
        cooling_efficiency = min(1.0, cooling_rate / 3000)  # Optimal around 3000 K/s
        stress_efficiency = max(0.0, 1.0 - thermal_stress / 500)  # Lower stress is better
        
        # Weighted average
        efficiency = (temp_efficiency + size_efficiency + cooling_efficiency + stress_efficiency) / 4
        
        return np.clip(efficiency, 0.0, 1.0)
    
    def _assess_thermal_stability(self, thermal_params: np.ndarray, 
                                thermal_history: List[float]) -> Dict[str, Any]:
        """
        Assess thermal stability based on parameters and history.
        
        Args:
            thermal_params: Current thermal parameters
            thermal_history: Thermal history data
            
        Returns:
            Dictionary with stability assessment
        """
        melt_pool_temp, melt_pool_size, cooling_rate, thermal_gradient, heat_accumulation, solidification_time, thermal_stress, temp_peak = thermal_params
        
        # Assess parameter stability
        temp_stability = 1.0 - abs(melt_pool_temp - 2000) / 2000
        size_stability = 1.0 - abs(melt_pool_size - 0.8) / 0.8
        cooling_stability = 1.0 - abs(cooling_rate - 3000) / 3000
        
        # Assess historical stability
        if len(thermal_history) > 1:
            temp_variance = np.var(thermal_history)
            temp_trend = np.polyfit(range(len(thermal_history)), thermal_history, 1)[0]
            historical_stability = 1.0 / (1.0 + temp_variance / 10000)  # Normalize variance
        else:
            historical_stability = 0.5  # Neutral if no history
        
        # Calculate overall stability
        overall_stability = (temp_stability + size_stability + cooling_stability + historical_stability) / 4
        
        # Determine stability level
        if overall_stability >= 0.8:
            stability_level = 'excellent'
        elif overall_stability >= 0.6:
            stability_level = 'good'
        elif overall_stability >= 0.4:
            stability_level = 'fair'
        else:
            stability_level = 'poor'
        
        return {
            'overall_stability': float(overall_stability),
            'stability_level': stability_level,
            'parameter_stability': {
                'temperature_stability': float(temp_stability),
                'size_stability': float(size_stability),
                'cooling_stability': float(cooling_stability)
            },
            'historical_stability': float(historical_stability),
            'stability_summary': self._get_stability_summary(stability_level)
        }
    
    def _get_stability_summary(self, level: str) -> str:
        """Get stability summary based on level."""
        summaries = {
            'excellent': 'Excellent thermal stability - optimal for consistent quality',
            'good': 'Good thermal stability - suitable for most applications',
            'fair': 'Fair thermal stability - monitor for variations',
            'poor': 'Poor thermal stability - optimization required'
        }
        return summaries.get(level, 'Unknown stability level')
    
    def _generate_thermal_optimization_recommendations(self, thermal_params: np.ndarray, 
                                                     process_parameters: Dict[str, float]) -> List[str]:
        """
        Generate thermal optimization recommendations.
        
        Args:
            thermal_params: Predicted thermal parameters
            process_parameters: Current process parameters
            
        Returns:
            List of optimization recommendations
        """
        recommendations = []
        melt_pool_temp, melt_pool_size, cooling_rate, thermal_gradient, heat_accumulation, solidification_time, thermal_stress, temp_peak = thermal_params
        
        # Temperature optimization
        if melt_pool_temp > 2500:
            recommendations.append("Melt pool temperature is too high. Consider:")
            recommendations.append("- Reducing laser power")
            recommendations.append("- Increasing scan speed")
            recommendations.append("- Optimizing focus position")
        elif melt_pool_temp < 1500:
            recommendations.append("Melt pool temperature is too low. Consider:")
            recommendations.append("- Increasing laser power")
            recommendations.append("- Reducing scan speed")
            recommendations.append("- Optimizing preheating temperature")
        
        # Melt pool size optimization
        if melt_pool_size > 1.2:
            recommendations.append("Melt pool size is too large. Consider:")
            recommendations.append("- Reducing laser power")
            recommendations.append("- Increasing scan speed")
            recommendations.append("- Optimizing beam diameter")
        elif melt_pool_size < 0.3:
            recommendations.append("Melt pool size is too small. Consider:")
            recommendations.append("- Increasing laser power")
            recommendations.append("- Reducing scan speed")
            recommendations.append("- Optimizing beam focus")
        
        # Cooling rate optimization
        if cooling_rate > 8000:
            recommendations.append("Cooling rate is too high. Consider:")
            recommendations.append("- Reducing scan speed for slower cooling")
            recommendations.append("- Optimizing preheating temperature")
            recommendations.append("- Adjusting build chamber temperature")
        elif cooling_rate < 500:
            recommendations.append("Cooling rate is too low. Consider:")
            recommendations.append("- Increasing scan speed")
            recommendations.append("- Optimizing cooling system")
            recommendations.append("- Adjusting build atmosphere")
        
        # Thermal stress optimization
        if thermal_stress > 400:
            recommendations.append("High thermal stress detected. Consider:")
            recommendations.append("- Optimizing scan pattern for stress distribution")
            recommendations.append("- Adjusting preheating temperature")
            recommendations.append("- Implementing stress relief strategies")
        
        # Heat accumulation optimization
        if heat_accumulation > 600:
            recommendations.append("High heat accumulation detected. Consider:")
            recommendations.append("- Optimizing scan pattern for heat distribution")
            recommendations.append("- Increasing cooling time between layers")
            recommendations.append("- Adjusting build parameters for heat management")
        
        return recommendations
    
    def _calculate_thermal_relationships(self, thermal_params: np.ndarray) -> Dict[str, float]:
        """
        Calculate thermal relationships for process understanding.
        
        Args:
            thermal_params: Thermal parameters
            
        Returns:
            Dictionary with thermal relationships
        """
        melt_pool_temp, melt_pool_size, cooling_rate, thermal_gradient, heat_accumulation, solidification_time, thermal_stress, temp_peak = thermal_params
        
        relationships = {
            'temperature_to_size_ratio': melt_pool_temp / melt_pool_size if melt_pool_size > 0 else 0,
            'cooling_to_gradient_ratio': cooling_rate / thermal_gradient if thermal_gradient > 0 else 0,
            'heat_to_stress_ratio': heat_accumulation / thermal_stress if thermal_stress > 0 else 0,
            'solidification_to_cooling_ratio': solidification_time * cooling_rate,
            'peak_to_melt_ratio': temp_peak / melt_pool_temp if melt_pool_temp > 0 else 0,
            'thermal_efficiency_index': (melt_pool_temp * melt_pool_size) / (thermal_stress * solidification_time) if thermal_stress > 0 and solidification_time > 0 else 0
        }
        
        return relationships
    
    def _calculate_thermal_grade(self, thermal_params: np.ndarray) -> str:
        """
        Calculate overall thermal grade based on parameters.
        
        Args:
            thermal_params: Thermal parameters
            
        Returns:
            Thermal grade (A, B, C, D)
        """
        melt_pool_temp, melt_pool_size, cooling_rate, thermal_gradient, heat_accumulation, solidification_time, thermal_stress, temp_peak = thermal_params
        
        # Calculate scores for each aspect
        temp_score = 1.0 - abs(melt_pool_temp - 2000) / 2000
        size_score = 1.0 - abs(melt_pool_size - 0.8) / 0.8
        cooling_score = 1.0 - abs(cooling_rate - 3000) / 3000
        stress_score = max(0.0, 1.0 - thermal_stress / 500)
        
        # Calculate composite score
        composite_score = (temp_score + size_score + cooling_score + stress_score) / 4
        
        if composite_score >= 0.9:
            return 'A'  # Excellent
        elif composite_score >= 0.8:
            return 'B'  # Good
        elif composite_score >= 0.7:
            return 'C'  # Acceptable
        else:
            return 'D'  # Poor
