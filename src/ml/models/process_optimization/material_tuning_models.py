"""
Material Tuning Models

This module implements models for material-specific parameter tuning
for different materials in PBF-LB/M processes (Inconel 718, Ti-6Al-4V, Stainless Steel 316L).
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Embedding
from tensorflow.keras.optimizers import Adam
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time

from ..base_model import BaseModel

logger = logging.getLogger(__name__)


class MaterialTuningModels(BaseModel):
    """
    Model for material-specific parameter tuning in PBF-LB/M processes.
    
    This model provides material-specific parameter recommendations for:
    - Inconel 718
    - Ti-6Al-4V
    - Stainless Steel 316L
    - Other materials as needed
    
    Parameters tuned include:
    - Laser power
    - Scan speed
    - Hatch spacing
    - Layer thickness
    - Preheating temperature
    """
    
    def __init__(self, config_manager=None):
        """
        Initialize the material tuning models.
        
        Args:
            config_manager: Configuration manager instance
        """
        super().__init__('material_tuning_models', config_manager)
        self.model_type = self.model_info.get('algorithm', 'neural_network')
        self.material_encoder = LabelEncoder()
        self.material_models = {}  # Separate models for each material
        
        logger.info(f"Initialized MaterialTuningModels with algorithm: {self.model_type}")
    
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
        """Build neural network model with material embedding."""
        model = Sequential()
        
        # Material embedding layer
        num_materials = len(self.data_config.get('materials', ['Inconel_718', 'Ti_6Al_4V', 'Stainless_Steel_316L']))
        model.add(Embedding(num_materials, 8, input_length=1, name='material_embedding'))
        model.add(tf.keras.layers.Flatten())
        
        # Feature input layer
        feature_dim = len(self.get_feature_names()) - 1  # Exclude material type
        model.add(Dense(128, activation='relu', input_dim=feature_dim))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        
        # Hidden layers
        model.add(Dense(64, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.2))
        
        # Output layer (5 parameters: laser_power, scan_speed, hatch_spacing, layer_thickness, preheat_temp)
        model.add(Dense(5, activation='linear'))
        
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
            n_estimators=150,
            max_depth=12,
            random_state=42,
            n_jobs=-1
        )
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            X_train: Training features (including material type)
            y_train: Training targets (material-specific parameters)
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            
        Returns:
            Training history dictionary
        """
        try:
            # Build model if not already built
            if self.model is None:
                self.model = self.build_model()
            
            # Prepare data
            X_train_prep, y_train_prep = self._prepare_data(X_train, y_train)
            X_val_prep, y_val_prep = None, None
            
            if X_val is not None and y_val is not None:
                X_val_prep, y_val_prep = self._prepare_data(X_val, y_val)
            
            start_time = time.time()
            
            if self.model_type == 'neural_network':
                history = self._train_neural_network(X_train_prep, y_train_prep, X_val_prep, y_val_prep)
            else:
                history = self._train_random_forest(X_train_prep, y_train_prep, X_val_prep, y_val_prep)
            
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
    
    def _prepare_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for training/prediction.
        
        Args:
            X: Input features
            y: Target labels
            
        Returns:
            Tuple of (prepared_X, prepared_y)
        """
        # Separate material type from other features
        material_col = 0  # Assuming material type is first column
        material_types = X[:, material_col]
        other_features = X[:, 1:]
        
        # Encode material types
        if not hasattr(self.material_encoder, 'classes_'):
            material_encoded = self.material_encoder.fit_transform(material_types)
        else:
            material_encoded = self.material_encoder.transform(material_types)
        
        # For neural network, we need separate inputs
        if self.model_type == 'neural_network':
            return (material_encoded.reshape(-1, 1), other_features), y
        else:
            # For random forest, combine all features
            return np.column_stack([material_encoded, other_features]), y
    
    def _train_neural_network(self, X_train: Tuple[np.ndarray, np.ndarray], y_train: np.ndarray, 
                             X_val: Optional[Tuple[np.ndarray, np.ndarray]] = None, y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
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
        
        # For neural network with embedding, we need to handle multiple inputs
        # This is a simplified version - in practice, you'd need to modify the model architecture
        # to handle multiple inputs properly
        history = self.model.fit(
            X_train[1], y_train,  # Using only the feature part for now
            batch_size=self.training_config.get('batch_size', 32),
            epochs=self.training_config.get('epochs', 150),
            validation_data=(X_val[1], y_val) if X_val is not None else None,
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
            X: Input features (including material type)
            
        Returns:
            Predictions array with material-specific parameters
        """
        try:
            if not self.is_trained:
                raise ValueError("Model must be trained before making predictions")
            
            # Prepare data
            X_prep, _ = self._prepare_data(X, np.zeros((X.shape[0], 5)))
            
            if self.model_type == 'neural_network':
                predictions = self.model.predict(X_prep[1], verbose=0)
            else:
                predictions = self.model.predict(X_prep, verbose=0)
            
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
                ranges[name] = feature.get('range', [0, 1000])
        
        # Constrain predictions
        constrained = predictions.copy()
        
        # Apply constraints for each output parameter
        param_names = ['laser_power', 'scan_speed', 'hatch_spacing', 'layer_thickness', 'preheat_temp']
        for i, param_name in enumerate(param_names):
            if param_name in ranges:
                constrained[:, i] = np.clip(constrained[:, i], ranges[param_name][0], ranges[param_name][1])
        
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
            param_names = ['laser_power', 'scan_speed', 'hatch_spacing', 'layer_thickness', 'preheat_temp']
            
            for i, param_name in enumerate(param_names):
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
    
    def get_material_parameters(self, material_type: str, part_geometry: Dict[str, float], 
                               quality_requirements: Dict[str, float]) -> Dict[str, float]:
        """
        Get material-specific parameters for a given material and requirements.
        
        Args:
            material_type: Type of material
            part_geometry: Part geometry parameters
            quality_requirements: Quality requirements
            
        Returns:
            Dictionary with material-specific parameters
        """
        try:
            # Create feature vector from inputs
            features = self._create_feature_vector(material_type, part_geometry, quality_requirements)
            
            # Make prediction
            prediction = self.predict(features.reshape(1, -1))[0]
            
            # Map predictions to parameter names
            param_names = ['laser_power', 'scan_speed', 'hatch_spacing', 'layer_thickness', 'preheat_temp']
            parameters = {name: float(prediction[i]) for i, name in enumerate(param_names)}
            
            return parameters
            
        except Exception as e:
            logger.error(f"Failed to get material parameters: {e}")
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
        feature_names = self.get_feature_names()
        features = np.zeros(len(feature_names))
        
        # Encode material type
        if hasattr(self.material_encoder, 'classes_'):
            material_encoded = self.material_encoder.transform([material_type])[0]
        else:
            material_encoded = 0  # Default value
        
        features[0] = material_encoded  # Material type is first feature
        
        # Map other inputs to features
        all_inputs = {**part_geometry, **quality_requirements}
        
        for key, value in all_inputs.items():
            if key in feature_names:
                features[feature_names.index(key)] = value
        
        return features
