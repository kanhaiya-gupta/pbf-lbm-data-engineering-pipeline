"""
Material Property Predictor Model

This module implements a model for predicting material properties
from process parameters in PBF-LB/M manufacturing.
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


class MaterialPropertyPredictor(BaseModel):
    """
    Model for predicting material properties from PBF-LB/M process parameters.
    
    This model predicts:
    - Density (relative density percentage)
    - Hardness (HV - Vickers Hardness)
    - Tensile strength (MPa)
    - Yield strength (MPa)
    - Elongation (%)
    - Fatigue strength (MPa)
    - Thermal conductivity (W/m·K)
    - Electrical conductivity (S/m)
    """
    
    def __init__(self, config_manager=None):
        """
        Initialize the material property predictor.
        
        Args:
            config_manager: Configuration manager instance
        """
        super().__init__('material_property_predictor', config_manager)
        self.model_type = self.model_info.get('algorithm', 'neural_network')
        self.material_properties = [
            'density', 'hardness', 'tensile_strength', 'yield_strength', 
            'elongation', 'fatigue_strength', 'thermal_conductivity', 'electrical_conductivity'
        ]
        
        logger.info(f"Initialized MaterialPropertyPredictor with algorithm: {self.model_type}")
    
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
        model.add(Dense(512, activation='relu', input_dim=input_dim))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        
        # Hidden layers
        model.add(Dense(256, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        
        model.add(Dense(128, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.1))
        
        # Output layer (8 material properties)
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
            X_train: Training features (process parameters)
            y_train: Training targets (material properties)
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
            X: Input features (process parameters)
            
        Returns:
            Predictions array with material properties
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
        Constrain predictions to valid ranges for material properties.
        
        Args:
            predictions: Raw predictions
            
        Returns:
            Constrained predictions
        """
        constrained = predictions.copy()
        
        # Constrain density to 0-100%
        constrained[:, 0] = np.clip(constrained[:, 0], 0, 100)
        
        # Constrain hardness to reasonable range (100-1000 HV)
        constrained[:, 1] = np.clip(constrained[:, 1], 100, 1000)
        
        # Constrain tensile strength to reasonable range (200-3000 MPa)
        constrained[:, 2] = np.clip(constrained[:, 2], 200, 3000)
        
        # Constrain yield strength to reasonable range (100-2500 MPa)
        constrained[:, 3] = np.clip(constrained[:, 3], 100, 2500)
        
        # Constrain elongation to reasonable range (0-50%)
        constrained[:, 4] = np.clip(constrained[:, 4], 0, 50)
        
        # Constrain fatigue strength to reasonable range (100-2000 MPa)
        constrained[:, 5] = np.clip(constrained[:, 5], 100, 2000)
        
        # Constrain thermal conductivity to reasonable range (1-500 W/m·K)
        constrained[:, 6] = np.clip(constrained[:, 6], 1, 500)
        
        # Constrain electrical conductivity to reasonable range (1e3-1e8 S/m)
        constrained[:, 7] = np.clip(constrained[:, 7], 1e3, 1e8)
        
        return constrained
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance on test data.
        
        Args:
            X_test: Test features
            y_test: Test targets (material properties)
            
        Returns:
            Evaluation metrics dictionary
        """
        try:
            if not self.is_trained:
                raise ValueError("Model must be trained before evaluation")
            
            # Make predictions
            predictions = self.predict(X_test)
            
            # Calculate metrics for each material property
            metrics = {}
            
            for i, prop_name in enumerate(self.material_properties):
                mse = mean_squared_error(y_test[:, i], predictions[:, i])
                mae = mean_absolute_error(y_test[:, i], predictions[:, i])
                r2 = r2_score(y_test[:, i], predictions[:, i])
                
                metrics.update({
                    f'{prop_name}_mse': mse,
                    f'{prop_name}_mae': mae,
                    f'{prop_name}_r2': r2
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
    
    def predict_material_properties(self, process_parameters: Dict[str, float], 
                                  material_composition: Dict[str, float],
                                  build_conditions: Dict[str, float]) -> Dict[str, Any]:
        """
        Predict material properties for specific process conditions.
        
        Args:
            process_parameters: Process parameters (laser power, scan speed, etc.)
            material_composition: Material composition percentages
            build_conditions: Build conditions (temperature, atmosphere, etc.)
            
        Returns:
            Dictionary with material property predictions and analysis
        """
        try:
            # Create feature vector from inputs
            features = self._create_feature_vector(process_parameters, material_composition, build_conditions)
            
            # Make prediction
            properties = self.predict(features.reshape(1, -1))[0]
            
            # Analyze material quality
            quality_analysis = self._analyze_material_quality(properties)
            
            # Generate optimization recommendations
            recommendations = self._generate_optimization_recommendations(properties, process_parameters)
            
            # Calculate property relationships
            property_relationships = self._calculate_property_relationships(properties)
            
            return {
                'material_properties': {
                    'density': float(properties[0]),  # Relative density (%)
                    'hardness': float(properties[1]),  # Vickers Hardness (HV)
                    'tensile_strength': float(properties[2]),  # Tensile strength (MPa)
                    'yield_strength': float(properties[3]),  # Yield strength (MPa)
                    'elongation': float(properties[4]),  # Elongation (%)
                    'fatigue_strength': float(properties[5]),  # Fatigue strength (MPa)
                    'thermal_conductivity': float(properties[6]),  # Thermal conductivity (W/m·K)
                    'electrical_conductivity': float(properties[7])  # Electrical conductivity (S/m)
                },
                'quality_analysis': quality_analysis,
                'optimization_recommendations': recommendations,
                'property_relationships': property_relationships,
                'material_grade': self._calculate_material_grade(properties)
            }
            
        except Exception as e:
            logger.error(f"Failed to predict material properties: {e}")
            raise
    
    def _create_feature_vector(self, process_parameters: Dict[str, float], 
                              material_composition: Dict[str, float],
                              build_conditions: Dict[str, float]) -> np.ndarray:
        """
        Create feature vector from input parameters.
        
        Args:
            process_parameters: Process parameters
            material_composition: Material composition
            build_conditions: Build conditions
            
        Returns:
            Feature vector
        """
        feature_names = self.get_feature_names()
        features = np.zeros(len(feature_names))
        
        # Combine all input dictionaries
        all_inputs = {**process_parameters, **material_composition, **build_conditions}
        
        # Map inputs to features
        for key, value in all_inputs.items():
            if key in feature_names:
                features[feature_names.index(key)] = value
        
        return features
    
    def _analyze_material_quality(self, properties: np.ndarray) -> Dict[str, Any]:
        """
        Analyze material quality based on predicted properties.
        
        Args:
            properties: Predicted material properties
            
        Returns:
            Dictionary with quality analysis
        """
        density, hardness, tensile_strength, yield_strength, elongation, fatigue_strength, thermal_conductivity, electrical_conductivity = properties
        
        # Assess density quality
        if density >= 99:
            density_grade = 'A'  # Excellent
        elif density >= 97:
            density_grade = 'B'  # Good
        elif density >= 95:
            density_grade = 'C'  # Acceptable
        else:
            density_grade = 'D'  # Poor
        
        # Assess strength quality
        strength_ratio = yield_strength / tensile_strength if tensile_strength > 0 else 0
        
        if strength_ratio >= 0.9:
            strength_grade = 'A'  # Excellent
        elif strength_ratio >= 0.8:
            strength_grade = 'B'  # Good
        elif strength_ratio >= 0.7:
            strength_grade = 'C'  # Acceptable
        else:
            strength_grade = 'D'  # Poor
        
        # Assess ductility
        if elongation >= 15:
            ductility_grade = 'A'  # Excellent
        elif elongation >= 10:
            ductility_grade = 'B'  # Good
        elif elongation >= 5:
            ductility_grade = 'C'  # Acceptable
        else:
            ductility_grade = 'D'  # Poor
        
        # Assess fatigue resistance
        fatigue_ratio = fatigue_strength / tensile_strength if tensile_strength > 0 else 0
        
        if fatigue_ratio >= 0.5:
            fatigue_grade = 'A'  # Excellent
        elif fatigue_ratio >= 0.4:
            fatigue_grade = 'B'  # Good
        elif fatigue_ratio >= 0.3:
            fatigue_grade = 'C'  # Acceptable
        else:
            fatigue_grade = 'D'  # Poor
        
        # Overall quality assessment
        grades = [density_grade, strength_grade, ductility_grade, fatigue_grade]
        if all(grade == 'A' for grade in grades):
            overall_grade = 'A'
        elif all(grade in ['A', 'B'] for grade in grades):
            overall_grade = 'B'
        elif all(grade in ['A', 'B', 'C'] for grade in grades):
            overall_grade = 'C'
        else:
            overall_grade = 'D'
        
        return {
            'density_grade': density_grade,
            'strength_grade': strength_grade,
            'ductility_grade': ductility_grade,
            'fatigue_grade': fatigue_grade,
            'overall_grade': overall_grade,
            'strength_ratio': float(strength_ratio),
            'fatigue_ratio': float(fatigue_ratio),
            'quality_summary': self._get_quality_summary(overall_grade)
        }
    
    def _get_quality_summary(self, grade: str) -> str:
        """
        Get quality summary based on grade.
        
        Args:
            grade: Quality grade (A, B, C, D)
            
        Returns:
            Quality summary string
        """
        summaries = {
            'A': 'Excellent material properties - meets or exceeds specifications',
            'B': 'Good material properties - meets specifications with minor optimization potential',
            'C': 'Acceptable material properties - meets minimum specifications',
            'D': 'Poor material properties - below specifications, optimization required'
        }
        return summaries.get(grade, 'Unknown quality grade')
    
    def _generate_optimization_recommendations(self, properties: np.ndarray, 
                                             process_parameters: Dict[str, float]) -> List[str]:
        """
        Generate optimization recommendations based on predicted properties.
        
        Args:
            properties: Predicted material properties
            process_parameters: Current process parameters
            
        Returns:
            List of optimization recommendations
        """
        recommendations = []
        density, hardness, tensile_strength, yield_strength, elongation, fatigue_strength, thermal_conductivity, electrical_conductivity = properties
        
        # Density optimization
        if density < 97:
            recommendations.append("Density is below optimal. Consider:")
            recommendations.append("- Increasing laser power for better fusion")
            recommendations.append("- Reducing scan speed for more complete melting")
            recommendations.append("- Optimizing hatch spacing for better coverage")
        
        # Strength optimization
        if tensile_strength < 800:  # Assuming typical range for metals
            recommendations.append("Tensile strength is below optimal. Consider:")
            recommendations.append("- Optimizing laser parameters for better microstructure")
            recommendations.append("- Ensuring proper preheating temperature")
            recommendations.append("- Checking material quality and composition")
        
        # Ductility optimization
        if elongation < 10:
            recommendations.append("Ductility is below optimal. Consider:")
            recommendations.append("- Reducing laser power to avoid overheating")
            recommendations.append("- Optimizing cooling rate")
            recommendations.append("- Post-processing heat treatment if applicable")
        
        # Fatigue resistance optimization
        if fatigue_strength < 400:  # Assuming typical range
            recommendations.append("Fatigue strength is below optimal. Consider:")
            recommendations.append("- Optimizing surface finish")
            recommendations.append("- Reducing residual stresses")
            recommendations.append("- Post-processing treatments")
        
        # Process parameter specific recommendations
        if 'laser_power' in process_parameters:
            if process_parameters['laser_power'] > 700:
                recommendations.append("High laser power detected - monitor for overheating effects")
            elif process_parameters['laser_power'] < 300:
                recommendations.append("Low laser power detected - may result in incomplete fusion")
        
        return recommendations
    
    def _calculate_property_relationships(self, properties: np.ndarray) -> Dict[str, float]:
        """
        Calculate important property relationships for material characterization.
        
        Args:
            properties: Predicted material properties
            
        Returns:
            Dictionary with property relationships
        """
        density, hardness, tensile_strength, yield_strength, elongation, fatigue_strength, thermal_conductivity, electrical_conductivity = properties
        
        relationships = {
            'yield_to_tensile_ratio': yield_strength / tensile_strength if tensile_strength > 0 else 0,
            'hardness_to_strength_ratio': hardness / tensile_strength if tensile_strength > 0 else 0,
            'fatigue_to_tensile_ratio': fatigue_strength / tensile_strength if tensile_strength > 0 else 0,
            'strength_to_density_ratio': tensile_strength / density if density > 0 else 0,
            'thermal_to_electrical_ratio': thermal_conductivity / electrical_conductivity if electrical_conductivity > 0 else 0
        }
        
        return relationships
    
    def _calculate_material_grade(self, properties: np.ndarray) -> str:
        """
        Calculate overall material grade based on properties.
        
        Args:
            properties: Predicted material properties
            
        Returns:
            Material grade (A, B, C, D)
        """
        density, hardness, tensile_strength, yield_strength, elongation, fatigue_strength, thermal_conductivity, electrical_conductivity = properties
        
        # Calculate composite score
        density_score = min(1.0, density / 100)
        strength_score = min(1.0, tensile_strength / 2000)
        ductility_score = min(1.0, elongation / 20)
        fatigue_score = min(1.0, fatigue_strength / 1000)
        
        composite_score = (density_score + strength_score + ductility_score + fatigue_score) / 4
        
        if composite_score >= 0.9:
            return 'A'  # Excellent
        elif composite_score >= 0.8:
            return 'B'  # Good
        elif composite_score >= 0.7:
            return 'C'  # Acceptable
        else:
            return 'D'  # Poor
