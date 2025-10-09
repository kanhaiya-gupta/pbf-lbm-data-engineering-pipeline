"""
Microstructure Analyzer Model

This module implements a model for analyzing and predicting microstructure
characteristics in PBF-LB/M manufactured materials.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time

from ..base_model import BaseModel

logger = logging.getLogger(__name__)


class MicrostructureAnalyzer(BaseModel):
    """
    Model for analyzing and predicting microstructure characteristics.
    
    This model analyzes:
    - Grain size distribution
    - Phase composition (alpha, beta, gamma phases)
    - Precipitate distribution
    - Defect density (pores, cracks, inclusions)
    - Texture orientation
    - Microstructural homogeneity
    """
    
    def __init__(self, config_manager=None):
        """
        Initialize the microstructure analyzer.
        
        Args:
            config_manager: Configuration manager instance
        """
        super().__init__('microstructure_analyzer', config_manager)
        self.model_type = self.model_info.get('algorithm', 'cnn')
        self.microstructure_features = [
            'grain_size', 'phase_alpha', 'phase_beta', 'phase_gamma',
            'precipitate_density', 'defect_density', 'texture_strength', 'homogeneity'
        ]
        
        logger.info(f"Initialized MicrostructureAnalyzer with algorithm: {self.model_type}")
    
    def build_model(self) -> Any:
        """
        Build the model architecture based on configuration.
        
        Returns:
            Built model instance
        """
        try:
            arch_config = self.architecture
            algorithm = arch_config.get('algorithm', 'cnn')
            
            if algorithm == 'cnn':
                return self._build_cnn_model()
            elif algorithm == 'random_forest':
                return self._build_random_forest()
            else:
                raise ValueError(f"Unsupported algorithm: {algorithm}")
                
        except Exception as e:
            logger.error(f"Failed to build model: {e}")
            raise
    
    def _build_cnn_model(self) -> tf.keras.Model:
        """Build CNN model for microstructure image analysis."""
        model = Sequential()
        
        # Input layer (assuming 256x256 grayscale images)
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 1)))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2)))
        
        # Convolutional layers
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2)))
        
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2)))
        
        model.add(Conv2D(256, (3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2)))
        
        # Flatten and dense layers
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.3))
        
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.2))
        
        # Output layer (8 microstructure features)
        model.add(Dense(8, activation='linear'))
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=self.training_config.get('learning_rate', 0.001)),
            loss='mse',
            metrics=['mae', 'mse']
        )
        
        return model
    
    def _build_random_forest(self) -> RandomForestClassifier:
        """Build random forest model."""
        return RandomForestClassifier(
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
            X_train: Training features (microstructure images or features)
            y_train: Training targets (microstructure characteristics)
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
            
            if self.model_type == 'cnn':
                history = self._train_cnn_model(X_train, y_train, X_val, y_val)
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
    
    def _train_cnn_model(self, X_train: np.ndarray, y_train: np.ndarray, 
                        X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Train CNN model."""
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
            batch_size=self.training_config.get('batch_size', 16),
            epochs=self.training_config.get('epochs', 100),
            validation_data=(X_val, y_val) if X_val is not None else None,
            callbacks=callbacks,
            verbose=1
        )
        
        return history.history
    
    def _train_random_forest(self, X_train: np.ndarray, y_train: np.ndarray, 
                            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Train random forest model."""
        # For random forest, we need to flatten image data
        if len(X_train.shape) > 2:
            X_train_flat = X_train.reshape(X_train.shape[0], -1)
            X_val_flat = X_val.reshape(X_val.shape[0], -1) if X_val is not None else None
        else:
            X_train_flat = X_train
            X_val_flat = X_val
        
        self.model.fit(X_train_flat, y_train)
        
        # Calculate training metrics
        train_pred = self.model.predict(X_train_flat)
        train_accuracy = accuracy_score(y_train, train_pred)
        train_precision = precision_score(y_train, train_pred, average='weighted', zero_division=0)
        train_recall = recall_score(y_train, train_pred, average='weighted', zero_division=0)
        train_f1 = f1_score(y_train, train_pred, average='weighted', zero_division=0)
        
        history = {
            'accuracy': [train_accuracy],
            'precision': [train_precision],
            'recall': [train_recall],
            'f1_score': [train_f1]
        }
        
        if X_val_flat is not None and y_val is not None:
            val_pred = self.model.predict(X_val_flat)
            val_accuracy = accuracy_score(y_val, val_pred)
            val_precision = precision_score(y_val, val_pred, average='weighted', zero_division=0)
            val_recall = recall_score(y_val, val_pred, average='weighted', zero_division=0)
            val_f1 = f1_score(y_val, val_pred, average='weighted', zero_division=0)
            
            history.update({
                'val_accuracy': [val_accuracy],
                'val_precision': [val_precision],
                'val_recall': [val_recall],
                'val_f1_score': [val_f1]
            })
        
        return history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Input features (microstructure images or features)
            
        Returns:
            Predictions array with microstructure characteristics
        """
        try:
            if not self.is_trained:
                raise ValueError("Model must be trained before making predictions")
            
            if self.model_type == 'cnn':
                predictions = self.model.predict(X, verbose=0)
            else:
                # For random forest, flatten image data
                if len(X.shape) > 2:
                    X_flat = X.reshape(X.shape[0], -1)
                else:
                    X_flat = X
                predictions = self.model.predict(X_flat)
            
            # Ensure predictions are within valid ranges
            predictions = self._constrain_predictions(predictions)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Failed to make predictions: {e}")
            raise
    
    def _constrain_predictions(self, predictions: np.ndarray) -> np.ndarray:
        """
        Constrain predictions to valid ranges for microstructure characteristics.
        
        Args:
            predictions: Raw predictions
            
        Returns:
            Constrained predictions
        """
        constrained = predictions.copy()
        
        # Constrain grain size to reasonable range (0.1-100 μm)
        constrained[:, 0] = np.clip(constrained[:, 0], 0.1, 100)
        
        # Constrain phase fractions to 0-100%
        constrained[:, 1] = np.clip(constrained[:, 1], 0, 100)  # Alpha phase
        constrained[:, 2] = np.clip(constrained[:, 2], 0, 100)  # Beta phase
        constrained[:, 3] = np.clip(constrained[:, 3], 0, 100)  # Gamma phase
        
        # Constrain precipitate density to reasonable range (0-1000 /mm²)
        constrained[:, 4] = np.clip(constrained[:, 4], 0, 1000)
        
        # Constrain defect density to reasonable range (0-100 /mm²)
        constrained[:, 5] = np.clip(constrained[:, 5], 0, 100)
        
        # Constrain texture strength to 0-1
        constrained[:, 6] = np.clip(constrained[:, 6], 0, 1)
        
        # Constrain homogeneity to 0-1
        constrained[:, 7] = np.clip(constrained[:, 7], 0, 1)
        
        return constrained
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance on test data.
        
        Args:
            X_test: Test features
            y_test: Test targets (microstructure characteristics)
            
        Returns:
            Evaluation metrics dictionary
        """
        try:
            if not self.is_trained:
                raise ValueError("Model must be trained before evaluation")
            
            # Make predictions
            predictions = self.predict(X_test)
            
            # Calculate metrics for each microstructure feature
            metrics = {}
            
            for i, feature_name in enumerate(self.microstructure_features):
                mse = np.mean((y_test[:, i] - predictions[:, i]) ** 2)
                mae = np.mean(np.abs(y_test[:, i] - predictions[:, i]))
                r2 = 1 - (np.sum((y_test[:, i] - predictions[:, i]) ** 2) / np.sum((y_test[:, i] - np.mean(y_test[:, i])) ** 2))
                
                metrics.update({
                    f'{feature_name}_mse': mse,
                    f'{feature_name}_mae': mae,
                    f'{feature_name}_r2': r2
                })
            
            # Overall metrics
            overall_mse = np.mean((y_test - predictions) ** 2)
            overall_mae = np.mean(np.abs(y_test - predictions))
            overall_r2 = 1 - (np.sum((y_test - predictions) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2))
            
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
            # For CNN models, return zero importance
            feature_names = self.get_feature_names()
            return {name: 0.0 for name in feature_names}
    
    def analyze_microstructure(self, microstructure_image: np.ndarray, 
                             process_parameters: Dict[str, float],
                             material_type: str) -> Dict[str, Any]:
        """
        Analyze microstructure from image and process parameters.
        
        Args:
            microstructure_image: Microstructure image (grayscale)
            process_parameters: Process parameters used
            material_type: Type of material
            
        Returns:
            Dictionary with microstructure analysis results
        """
        try:
            # Preprocess image
            processed_image = self._preprocess_image(microstructure_image)
            
            # Make prediction
            features = self.predict(processed_image.reshape(1, *processed_image.shape, 1))
            
            # Analyze microstructure characteristics
            microstructure_analysis = self._analyze_microstructure_characteristics(features[0])
            
            # Assess microstructure quality
            quality_assessment = self._assess_microstructure_quality(features[0], material_type)
            
            # Generate optimization recommendations
            recommendations = self._generate_microstructure_recommendations(
                features[0], process_parameters, material_type
            )
            
            # Calculate phase relationships
            phase_relationships = self._calculate_phase_relationships(features[0])
            
            return {
                'microstructure_features': {
                    'grain_size_um': float(features[0][0]),
                    'alpha_phase_percent': float(features[0][1]),
                    'beta_phase_percent': float(features[0][2]),
                    'gamma_phase_percent': float(features[0][3]),
                    'precipitate_density_per_mm2': float(features[0][4]),
                    'defect_density_per_mm2': float(features[0][5]),
                    'texture_strength': float(features[0][6]),
                    'homogeneity_index': float(features[0][7])
                },
                'microstructure_analysis': microstructure_analysis,
                'quality_assessment': quality_assessment,
                'optimization_recommendations': recommendations,
                'phase_relationships': phase_relationships,
                'material_type': material_type
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze microstructure: {e}")
            raise
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess microstructure image for analysis.
        
        Args:
            image: Raw microstructure image
            
        Returns:
            Preprocessed image
        """
        # Normalize image to 0-1 range
        if image.max() > 1:
            image = image / 255.0
        
        # Resize to standard size if needed
        if image.shape != (256, 256):
            # In practice, use proper image resizing
            # For now, just ensure it's the right shape
            if len(image.shape) == 2:
                # Pad or crop to 256x256
                target_size = (256, 256)
                if image.shape[0] < target_size[0]:
                    pad_width = target_size[0] - image.shape[0]
                    image = np.pad(image, ((0, pad_width), (0, 0)), mode='constant')
                elif image.shape[0] > target_size[0]:
                    image = image[:target_size[0], :]
                
                if image.shape[1] < target_size[1]:
                    pad_width = target_size[1] - image.shape[1]
                    image = np.pad(image, ((0, 0), (0, pad_width)), mode='constant')
                elif image.shape[1] > target_size[1]:
                    image = image[:, :target_size[1]]
        
        return image
    
    def _analyze_microstructure_characteristics(self, features: np.ndarray) -> Dict[str, Any]:
        """
        Analyze microstructure characteristics from predicted features.
        
        Args:
            features: Predicted microstructure features
            
        Returns:
            Dictionary with microstructure analysis
        """
        grain_size, alpha_phase, beta_phase, gamma_phase, precipitate_density, defect_density, texture_strength, homogeneity = features
        
        # Analyze grain size distribution
        if grain_size < 5:
            grain_size_category = 'fine'
        elif grain_size < 20:
            grain_size_category = 'medium'
        else:
            grain_size_category = 'coarse'
        
        # Analyze phase composition
        total_phases = alpha_phase + beta_phase + gamma_phase
        if total_phases > 0:
            alpha_fraction = alpha_phase / total_phases
            beta_fraction = beta_phase / total_phases
            gamma_fraction = gamma_phase / total_phases
        else:
            alpha_fraction = beta_fraction = gamma_fraction = 0
        
        # Determine dominant phase
        if alpha_fraction > 0.5:
            dominant_phase = 'alpha'
        elif beta_fraction > 0.5:
            dominant_phase = 'beta'
        elif gamma_fraction > 0.5:
            dominant_phase = 'gamma'
        else:
            dominant_phase = 'mixed'
        
        # Analyze defect level
        if defect_density < 10:
            defect_level = 'low'
        elif defect_density < 50:
            defect_level = 'medium'
        else:
            defect_level = 'high'
        
        # Analyze texture strength
        if texture_strength < 0.3:
            texture_category = 'weak'
        elif texture_strength < 0.7:
            texture_category = 'moderate'
        else:
            texture_category = 'strong'
        
        # Analyze homogeneity
        if homogeneity > 0.8:
            homogeneity_category = 'excellent'
        elif homogeneity > 0.6:
            homogeneity_category = 'good'
        elif homogeneity > 0.4:
            homogeneity_category = 'fair'
        else:
            homogeneity_category = 'poor'
        
        return {
            'grain_size_category': grain_size_category,
            'dominant_phase': dominant_phase,
            'phase_fractions': {
                'alpha': float(alpha_fraction),
                'beta': float(beta_fraction),
                'gamma': float(gamma_fraction)
            },
            'defect_level': defect_level,
            'texture_category': texture_category,
            'homogeneity_category': homogeneity_category,
            'precipitate_density_category': 'high' if precipitate_density > 500 else 'medium' if precipitate_density > 100 else 'low'
        }
    
    def _assess_microstructure_quality(self, features: np.ndarray, material_type: str) -> Dict[str, Any]:
        """
        Assess microstructure quality based on features and material type.
        
        Args:
            features: Predicted microstructure features
            material_type: Type of material
            
        Returns:
            Dictionary with quality assessment
        """
        grain_size, alpha_phase, beta_phase, gamma_phase, precipitate_density, defect_density, texture_strength, homogeneity = features
        
        # Material-specific quality criteria
        quality_criteria = self._get_material_quality_criteria(material_type)
        
        # Assess each aspect
        grain_size_score = self._score_grain_size(grain_size, quality_criteria['grain_size_range'])
        phase_score = self._score_phase_composition(alpha_phase, beta_phase, gamma_phase, quality_criteria['phase_composition'])
        defect_score = self._score_defect_density(defect_density, quality_criteria['max_defect_density'])
        homogeneity_score = self._score_homogeneity(homogeneity, quality_criteria['min_homogeneity'])
        
        # Calculate overall quality score
        overall_score = (grain_size_score + phase_score + defect_score + homogeneity_score) / 4
        
        # Determine quality grade
        if overall_score >= 0.9:
            quality_grade = 'A'
        elif overall_score >= 0.8:
            quality_grade = 'B'
        elif overall_score >= 0.7:
            quality_grade = 'C'
        else:
            quality_grade = 'D'
        
        return {
            'overall_score': float(overall_score),
            'quality_grade': quality_grade,
            'aspect_scores': {
                'grain_size_score': float(grain_size_score),
                'phase_score': float(phase_score),
                'defect_score': float(defect_score),
                'homogeneity_score': float(homogeneity_score)
            },
            'quality_summary': self._get_quality_summary(quality_grade)
        }
    
    def _get_material_quality_criteria(self, material_type: str) -> Dict[str, Any]:
        """
        Get quality criteria for specific material type.
        
        Args:
            material_type: Type of material
            
        Returns:
            Dictionary with quality criteria
        """
        criteria = {
            'titanium': {
                'grain_size_range': (5, 20),
                'phase_composition': {'alpha': 0.6, 'beta': 0.4},
                'max_defect_density': 20,
                'min_homogeneity': 0.7
            },
            'steel': {
                'grain_size_range': (10, 50),
                'phase_composition': {'alpha': 0.8, 'gamma': 0.2},
                'max_defect_density': 30,
                'min_homogeneity': 0.6
            },
            'aluminum': {
                'grain_size_range': (20, 100),
                'phase_composition': {'alpha': 0.9, 'beta': 0.1},
                'max_defect_density': 15,
                'min_homogeneity': 0.8
            },
            'nickel': {
                'grain_size_range': (5, 25),
                'phase_composition': {'gamma': 0.9, 'alpha': 0.1},
                'max_defect_density': 25,
                'min_homogeneity': 0.7
            }
        }
        
        return criteria.get(material_type.lower(), criteria['steel'])  # Default to steel criteria
    
    def _score_grain_size(self, grain_size: float, target_range: Tuple[float, float]) -> float:
        """Score grain size based on target range."""
        min_size, max_size = target_range
        if min_size <= grain_size <= max_size:
            return 1.0
        else:
            # Penalty for being outside range
            if grain_size < min_size:
                return max(0.0, 1.0 - (min_size - grain_size) / min_size)
            else:
                return max(0.0, 1.0 - (grain_size - max_size) / max_size)
    
    def _score_phase_composition(self, alpha: float, beta: float, gamma: float, 
                               target_composition: Dict[str, float]) -> float:
        """Score phase composition based on target composition."""
        total = alpha + beta + gamma
        if total == 0:
            return 0.0
        
        alpha_fraction = alpha / total
        beta_fraction = beta / total
        gamma_fraction = gamma / total
        
        # Calculate deviation from target
        alpha_dev = abs(alpha_fraction - target_composition.get('alpha', 0))
        beta_dev = abs(beta_fraction - target_composition.get('beta', 0))
        gamma_dev = abs(gamma_fraction - target_composition.get('gamma', 0))
        
        total_deviation = alpha_dev + beta_dev + gamma_dev
        
        return max(0.0, 1.0 - total_deviation)
    
    def _score_defect_density(self, defect_density: float, max_allowed: float) -> float:
        """Score defect density based on maximum allowed."""
        if defect_density <= max_allowed:
            return 1.0
        else:
            return max(0.0, 1.0 - (defect_density - max_allowed) / max_allowed)
    
    def _score_homogeneity(self, homogeneity: float, min_required: float) -> float:
        """Score homogeneity based on minimum required."""
        if homogeneity >= min_required:
            return 1.0
        else:
            return max(0.0, homogeneity / min_required)
    
    def _get_quality_summary(self, grade: str) -> str:
        """Get quality summary based on grade."""
        summaries = {
            'A': 'Excellent microstructure - optimal for high-performance applications',
            'B': 'Good microstructure - suitable for most applications',
            'C': 'Acceptable microstructure - meets minimum requirements',
            'D': 'Poor microstructure - optimization required for better performance'
        }
        return summaries.get(grade, 'Unknown quality grade')
    
    def _generate_microstructure_recommendations(self, features: np.ndarray, 
                                               process_parameters: Dict[str, float],
                                               material_type: str) -> List[str]:
        """
        Generate microstructure optimization recommendations.
        
        Args:
            features: Predicted microstructure features
            process_parameters: Current process parameters
            material_type: Type of material
            
        Returns:
            List of optimization recommendations
        """
        recommendations = []
        grain_size, alpha_phase, beta_phase, gamma_phase, precipitate_density, defect_density, texture_strength, homogeneity = features
        
        # Grain size optimization
        if grain_size < 5:
            recommendations.append("Grain size is too fine. Consider:")
            recommendations.append("- Reducing laser power to allow grain growth")
            recommendations.append("- Increasing scan speed to reduce heat input")
        elif grain_size > 50:
            recommendations.append("Grain size is too coarse. Consider:")
            recommendations.append("- Increasing laser power for better grain refinement")
            recommendations.append("- Reducing scan speed for more controlled cooling")
        
        # Phase composition optimization
        total_phases = alpha_phase + beta_phase + gamma_phase
        if total_phases > 0:
            alpha_fraction = alpha_phase / total_phases
            if alpha_fraction < 0.5 and material_type.lower() in ['titanium', 'steel']:
                recommendations.append("Alpha phase fraction is low. Consider:")
                recommendations.append("- Optimizing cooling rate")
                recommendations.append("- Adjusting preheating temperature")
        
        # Defect density optimization
        if defect_density > 30:
            recommendations.append("High defect density detected. Consider:")
            recommendations.append("- Optimizing laser parameters for better fusion")
            recommendations.append("- Improving powder quality and distribution")
            recommendations.append("- Optimizing build atmosphere")
        
        # Homogeneity optimization
        if homogeneity < 0.6:
            recommendations.append("Poor microstructure homogeneity. Consider:")
            recommendations.append("- Optimizing scan pattern for uniform heat distribution")
            recommendations.append("- Adjusting layer thickness")
            recommendations.append("- Improving powder bed uniformity")
        
        # Process parameter specific recommendations
        if 'laser_power' in process_parameters:
            if process_parameters['laser_power'] > 600:
                recommendations.append("High laser power may cause excessive grain growth")
            elif process_parameters['laser_power'] < 200:
                recommendations.append("Low laser power may result in incomplete fusion and defects")
        
        return recommendations
    
    def _calculate_phase_relationships(self, features: np.ndarray) -> Dict[str, float]:
        """
        Calculate phase relationships for microstructure characterization.
        
        Args:
            features: Predicted microstructure features
            
        Returns:
            Dictionary with phase relationships
        """
        grain_size, alpha_phase, beta_phase, gamma_phase, precipitate_density, defect_density, texture_strength, homogeneity = features
        
        total_phases = alpha_phase + beta_phase + gamma_phase
        
        relationships = {
            'alpha_beta_ratio': alpha_phase / beta_phase if beta_phase > 0 else float('inf'),
            'alpha_gamma_ratio': alpha_phase / gamma_phase if gamma_phase > 0 else float('inf'),
            'beta_gamma_ratio': beta_phase / gamma_phase if gamma_phase > 0 else float('inf'),
            'total_phase_fraction': total_phases / 100.0,  # Assuming phases are in percentage
            'precipitate_to_defect_ratio': precipitate_density / defect_density if defect_density > 0 else float('inf'),
            'grain_size_to_defect_ratio': grain_size / defect_density if defect_density > 0 else float('inf')
        }
        
        return relationships
