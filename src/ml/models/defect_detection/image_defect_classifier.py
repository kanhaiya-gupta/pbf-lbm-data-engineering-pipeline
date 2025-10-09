"""
Image Defect Classifier Model

This module implements a model for classifying defects in images
from PBF-LB/M manufacturing processes.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import time

from ..base_model import BaseModel

logger = logging.getLogger(__name__)


class ImageDefectClassifier(BaseModel):
    """
    Model for classifying defects in images from PBF-LB/M processes.
    
    This model classifies:
    - Pores (spherical, irregular, keyhole)
    - Cracks (hot cracks, cold cracks, stress cracks)
    - Lack of fusion defects
    - Surface defects (roughness, waviness)
    - Dimensional defects (overhang, warping)
    - Contamination defects
    """
    
    def __init__(self, config_manager=None):
        """
        Initialize the image defect classifier.
        
        Args:
            config_manager: Configuration manager instance
        """
        super().__init__('image_defect_classifier', config_manager)
        self.model_type = self.model_info.get('algorithm', 'cnn')
        self.defect_classes = [
            'no_defect', 'pore_spherical', 'pore_irregular', 'pore_keyhole',
            'crack_hot', 'crack_cold', 'crack_stress', 'lack_of_fusion',
            'surface_roughness', 'surface_waviness', 'overhang', 'warping',
            'contamination', 'other_defect'
        ]
        
        logger.info(f"Initialized ImageDefectClassifier with algorithm: {self.model_type}")
    
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
            else:
                raise ValueError(f"Unsupported algorithm: {algorithm}")
                
        except Exception as e:
            logger.error(f"Failed to build model: {e}")
            raise
    
    def _build_cnn_model(self) -> tf.keras.Model:
        """Build CNN model for image defect classification."""
        model = Sequential()
        
        # Input layer (assuming 224x224 RGB images)
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
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
        
        model.add(Conv2D(512, (3, 3), activation='relu'))
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
        
        # Output layer (14 defect classes)
        model.add(Dense(len(self.defect_classes), activation='softmax'))
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=self.training_config.get('learning_rate', 0.001)),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            X_train: Training features (images)
            y_train: Training targets (defect class labels)
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
            
            history = self._train_cnn_model(X_train, y_train, X_val, y_val)
            
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
            batch_size=self.training_config.get('batch_size', 16),
            epochs=self.training_config.get('epochs', 100),
            validation_data=(X_val, y_val) if X_val is not None else None,
            callbacks=callbacks,
            verbose=1
        )
        
        return history.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Input features (images)
            
        Returns:
            Predictions array (defect class probabilities)
        """
        try:
            if not self.is_trained:
                raise ValueError("Model must be trained before making predictions")
            
            predictions = self.model.predict(X, verbose=0)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Failed to make predictions: {e}")
            raise
    
    def predict_defect_class(self, X: np.ndarray) -> np.ndarray:
        """
        Predict defect class (hard classification).
        
        Args:
            X: Input features (images)
            
        Returns:
            Defect class predictions
        """
        try:
            if not self.is_trained:
                raise ValueError("Model must be trained before making predictions")
            
            predictions = self.predict(X)
            class_predictions = np.argmax(predictions, axis=1)
            
            return class_predictions
            
        except Exception as e:
            logger.error(f"Failed to predict defect class: {e}")
            raise
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance on test data.
        
        Args:
            X_test: Test features (images)
            y_test: Test targets (defect class labels)
            
        Returns:
            Evaluation metrics dictionary
        """
        try:
            if not self.is_trained:
                raise ValueError("Model must be trained before evaluation")
            
            # Make predictions
            predictions = self.predict(X_test)
            class_predictions = self.predict_defect_class(X_test)
            
            # Convert y_test to class labels if needed
            if len(y_test.shape) > 1:
                y_test_classes = np.argmax(y_test, axis=1)
            else:
                y_test_classes = y_test
            
            # Calculate metrics
            accuracy = accuracy_score(y_test_classes, class_predictions)
            precision = precision_score(y_test_classes, class_predictions, average='weighted', zero_division=0)
            recall = recall_score(y_test_classes, class_predictions, average='weighted', zero_division=0)
            f1 = f1_score(y_test_classes, class_predictions, average='weighted', zero_division=0)
            
            # Calculate per-class metrics
            per_class_metrics = self._calculate_per_class_metrics(y_test_classes, class_predictions)
            
            # Calculate latency
            start_time = time.time()
            _ = self.predict(X_test[:10])
            latency = (time.time() - start_time) / 10 * 1000
            
            self.evaluation_metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'per_class_metrics': per_class_metrics,
                'latency_ms': latency,
                'test_samples': len(X_test)
            }
            
            logger.info(f"Model evaluation completed:")
            logger.info(f"  Accuracy: {accuracy:.4f}")
            logger.info(f"  Precision: {precision:.4f}")
            logger.info(f"  Recall: {recall:.4f}")
            logger.info(f"  F1-Score: {f1:.4f}")
            logger.info(f"  Latency: {latency:.2f} ms")
            
            return self.evaluation_metrics
            
        except Exception as e:
            logger.error(f"Failed to evaluate model: {e}")
            raise
    
    def _calculate_per_class_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Calculate per-class metrics.
        
        Args:
            y_true: True class labels
            y_pred: Predicted class labels
            
        Returns:
            Dictionary with per-class metrics
        """
        per_class_metrics = {}
        
        for i, class_name in enumerate(self.defect_classes):
            # Create binary labels for this class
            y_true_binary = (y_true == i).astype(int)
            y_pred_binary = (y_pred == i).astype(int)
            
            # Calculate metrics
            precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
            recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
            f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
            
            per_class_metrics[class_name] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
        
        return per_class_metrics
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance.
        
        Returns:
            Feature importance dictionary
        """
        # For CNN models, feature importance is not directly available
        # Return zero importance for all features
        feature_names = self.get_feature_names()
        return {name: 0.0 for name in feature_names}
    
    def classify_defect(self, image: np.ndarray, 
                       process_parameters: Dict[str, float],
                       image_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classify defect in an image.
        
        Args:
            image: Input image
            process_parameters: Process parameters
            image_metadata: Image metadata
            
        Returns:
            Dictionary with defect classification results
        """
        try:
            # Preprocess image
            processed_image = self._preprocess_image(image)
            
            # Make prediction
            predictions = self.predict(processed_image.reshape(1, *processed_image.shape))
            class_probabilities = predictions[0]
            predicted_class = np.argmax(class_probabilities)
            confidence = class_probabilities[predicted_class]
            
            # Analyze defect characteristics
            defect_analysis = self._analyze_defect_characteristics(class_probabilities, process_parameters)
            
            # Assess defect severity
            severity_assessment = self._assess_defect_severity(predicted_class, confidence, process_parameters)
            
            # Generate recommendations
            recommendations = self._generate_defect_recommendations(predicted_class, confidence, process_parameters)
            
            # Calculate defect metrics
            defect_metrics = self._calculate_defect_metrics(class_probabilities, image_metadata)
            
            return {
                'defect_classification': {
                    'predicted_class': self.defect_classes[predicted_class],
                    'confidence': float(confidence),
                    'class_probabilities': {self.defect_classes[i]: float(prob) for i, prob in enumerate(class_probabilities)}
                },
                'defect_analysis': defect_analysis,
                'severity_assessment': severity_assessment,
                'recommendations': recommendations,
                'defect_metrics': defect_metrics,
                'image_metadata': image_metadata
            }
            
        except Exception as e:
            logger.error(f"Failed to classify defect: {e}")
            raise
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for classification.
        
        Args:
            image: Raw image
            
        Returns:
            Preprocessed image
        """
        # Normalize image to 0-1 range
        if image.max() > 1:
            image = image / 255.0
        
        # Resize to standard size if needed
        if image.shape[:2] != (224, 224):
            # In practice, use proper image resizing
            # For now, just ensure it's the right shape
            if len(image.shape) == 2:
                # Grayscale to RGB
                image = np.stack([image, image, image], axis=-1)
            elif image.shape[2] == 1:
                # Single channel to RGB
                image = np.repeat(image, 3, axis=2)
        
        return image
    
    def _analyze_defect_characteristics(self, class_probabilities: np.ndarray, 
                                      process_parameters: Dict[str, float]) -> Dict[str, Any]:
        """
        Analyze defect characteristics from classification results.
        
        Args:
            class_probabilities: Class probabilities
            process_parameters: Process parameters
            
        Returns:
            Dictionary with defect analysis
        """
        # Find top 3 predicted classes
        top_indices = np.argsort(class_probabilities)[-3:][::-1]
        top_classes = [self.defect_classes[i] for i in top_indices]
        top_probabilities = [class_probabilities[i] for i in top_indices]
        
        # Analyze defect type
        predicted_class = self.defect_classes[np.argmax(class_probabilities)]
        
        if predicted_class == 'no_defect':
            defect_type = 'none'
            defect_category = 'none'
        elif 'pore' in predicted_class:
            defect_type = 'pore'
            defect_category = 'internal'
        elif 'crack' in predicted_class:
            defect_type = 'crack'
            defect_category = 'structural'
        elif 'surface' in predicted_class:
            defect_type = 'surface'
            defect_category = 'surface'
        elif predicted_class in ['overhang', 'warping']:
            defect_type = 'dimensional'
            defect_category = 'geometric'
        else:
            defect_type = 'other'
            defect_category = 'unknown'
        
        # Analyze confidence distribution
        confidence_distribution = {
            'high_confidence': np.sum(class_probabilities > 0.8),
            'medium_confidence': np.sum((class_probabilities > 0.5) & (class_probabilities <= 0.8)),
            'low_confidence': np.sum(class_probabilities <= 0.5)
        }
        
        return {
            'defect_type': defect_type,
            'defect_category': defect_category,
            'top_predictions': list(zip(top_classes, top_probabilities)),
            'confidence_distribution': confidence_distribution,
            'prediction_uncertainty': float(np.std(class_probabilities)),
            'defect_family': self._get_defect_family(predicted_class)
        }
    
    def _get_defect_family(self, defect_class: str) -> str:
        """
        Get defect family for a defect class.
        
        Args:
            defect_class: Defect class name
            
        Returns:
            Defect family name
        """
        if defect_class == 'no_defect':
            return 'none'
        elif 'pore' in defect_class:
            return 'porosity'
        elif 'crack' in defect_class:
            return 'cracking'
        elif 'surface' in defect_class:
            return 'surface_quality'
        elif defect_class in ['overhang', 'warping']:
            return 'dimensional'
        elif defect_class == 'contamination':
            return 'contamination'
        else:
            return 'other'
    
    def _assess_defect_severity(self, predicted_class: int, confidence: float, 
                               process_parameters: Dict[str, float]) -> Dict[str, Any]:
        """
        Assess defect severity based on classification and process parameters.
        
        Args:
            predicted_class: Predicted defect class index
            confidence: Classification confidence
            process_parameters: Process parameters
            
        Returns:
            Dictionary with severity assessment
        """
        defect_class = self.defect_classes[predicted_class]
        
        # Base severity assessment
        if defect_class == 'no_defect':
            severity_level = 'none'
            severity_score = 0.0
        elif defect_class in ['pore_spherical', 'surface_roughness']:
            severity_level = 'low'
            severity_score = 0.3
        elif defect_class in ['pore_irregular', 'surface_waviness', 'overhang']:
            severity_level = 'medium'
            severity_score = 0.6
        elif defect_class in ['pore_keyhole', 'crack_hot', 'crack_cold', 'lack_of_fusion']:
            severity_level = 'high'
            severity_score = 0.8
        elif defect_class in ['crack_stress', 'warping', 'contamination']:
            severity_level = 'critical'
            severity_score = 1.0
        else:
            severity_level = 'unknown'
            severity_score = 0.5
        
        # Adjust severity based on confidence
        if confidence < 0.5:
            severity_score *= 0.8  # Reduce severity for low confidence
        elif confidence > 0.9:
            severity_score *= 1.1  # Increase severity for high confidence
        
        # Adjust severity based on process parameters
        if 'laser_power' in process_parameters:
            if process_parameters['laser_power'] > 600:
                severity_score *= 1.1  # Higher power may increase severity
            elif process_parameters['laser_power'] < 200:
                severity_score *= 1.2  # Low power may indicate process issues
        
        severity_score = min(1.0, severity_score)
        
        return {
            'severity_level': severity_level,
            'severity_score': float(severity_score),
            'confidence_adjusted': confidence > 0.7,
            'process_adjusted': 'laser_power' in process_parameters,
            'severity_summary': self._get_severity_summary(severity_level, severity_score)
        }
    
    def _get_severity_summary(self, level: str, score: float) -> str:
        """
        Get severity summary based on level and score.
        
        Args:
            level: Severity level
            score: Severity score
            
        Returns:
            Severity summary string
        """
        summaries = {
            'none': 'No defects detected - part meets quality standards',
            'low': 'Minor defects detected - acceptable for most applications',
            'medium': 'Moderate defects detected - may require post-processing',
            'high': 'Significant defects detected - quality concerns, investigation required',
            'critical': 'Critical defects detected - part may be rejected, immediate action required'
        }
        return summaries.get(level, 'Unknown severity level')
    
    def _generate_defect_recommendations(self, predicted_class: int, confidence: float, 
                                       process_parameters: Dict[str, float]) -> List[str]:
        """
        Generate recommendations based on defect classification.
        
        Args:
            predicted_class: Predicted defect class index
            confidence: Classification confidence
            process_parameters: Process parameters
            
        Returns:
            List of recommendations
        """
        recommendations = []
        defect_class = self.defect_classes[predicted_class]
        
        if defect_class == 'no_defect':
            recommendations.append("No defects detected - continue current process parameters")
            recommendations.append("Maintain quality monitoring")
        
        elif 'pore' in defect_class:
            recommendations.append("Porosity detected. Consider:")
            recommendations.append("- Optimizing laser power for better fusion")
            recommendations.append("- Adjusting scan speed to reduce gas entrapment")
            recommendations.append("- Improving powder quality and distribution")
            recommendations.append("- Optimizing build atmosphere")
        
        elif 'crack' in defect_class:
            recommendations.append("Cracking detected. Consider:")
            recommendations.append("- Reducing thermal stress through scan pattern optimization")
            recommendations.append("- Adjusting preheating temperature")
            recommendations.append("- Optimizing cooling rate")
            recommendations.append("- Post-processing heat treatment if applicable")
        
        elif 'surface' in defect_class:
            recommendations.append("Surface quality issues detected. Consider:")
            recommendations.append("- Optimizing laser parameters for better surface finish")
            recommendations.append("- Adjusting layer thickness")
            recommendations.append("- Post-processing surface treatment")
        
        elif defect_class in ['overhang', 'warping']:
            recommendations.append("Dimensional issues detected. Consider:")
            recommendations.append("- Optimizing support structure design")
            recommendations.append("- Adjusting build orientation")
            recommendations.append("- Optimizing scan pattern for stress distribution")
        
        elif defect_class == 'contamination':
            recommendations.append("Contamination detected. Consider:")
            recommendations.append("- Improving powder handling procedures")
            recommendations.append("- Optimizing build chamber cleanliness")
            recommendations.append("- Checking powder quality and storage")
        
        # Add confidence-based recommendations
        if confidence < 0.7:
            recommendations.append("Low confidence in classification - consider manual inspection")
            recommendations.append("Collect more training data for this defect type")
        
        # Add process-specific recommendations
        if 'laser_power' in process_parameters:
            if process_parameters['laser_power'] > 700:
                recommendations.append("High laser power detected - monitor for overheating effects")
            elif process_parameters['laser_power'] < 200:
                recommendations.append("Low laser power detected - may result in incomplete fusion")
        
        return recommendations
    
    def _calculate_defect_metrics(self, class_probabilities: np.ndarray, 
                                image_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate defect metrics from classification results.
        
        Args:
            class_probabilities: Class probabilities
            image_metadata: Image metadata
            
        Returns:
            Dictionary with defect metrics
        """
        # Calculate entropy (uncertainty measure)
        entropy = -np.sum(class_probabilities * np.log(class_probabilities + 1e-10))
        
        # Calculate maximum probability
        max_probability = np.max(class_probabilities)
        
        # Calculate probability distribution
        probability_distribution = {
            'defect_probability': np.sum(class_probabilities[1:]),  # Sum of all defect classes
            'no_defect_probability': class_probabilities[0],
            'top_defect_probability': np.max(class_probabilities[1:])
        }
        
        # Calculate confidence metrics
        confidence_metrics = {
            'prediction_confidence': float(max_probability),
            'prediction_entropy': float(entropy),
            'confidence_level': 'high' if max_probability > 0.8 else 'medium' if max_probability > 0.5 else 'low'
        }
        
        return {
            'entropy': float(entropy),
            'max_probability': float(max_probability),
            'probability_distribution': probability_distribution,
            'confidence_metrics': confidence_metrics,
            'image_quality': image_metadata.get('quality', 'unknown'),
            'image_resolution': image_metadata.get('resolution', 'unknown')
        }
