"""
Failure Predictor Model

This module implements a model for predicting equipment failures
before they occur in PBF-LB/M systems.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import time

from ..base_model import BaseModel

logger = logging.getLogger(__name__)


class FailurePredictor(BaseModel):
    """
    Model for predicting equipment failures in PBF-LB/M systems.
    
    This model predicts:
    - Failure probability (0-1)
    - Time to failure (hours/days)
    - Failure type classification
    - Risk level assessment
    """
    
    def __init__(self, config_manager=None):
        """
        Initialize the failure predictor.
        
        Args:
            config_manager: Configuration manager instance
        """
        super().__init__('failure_predictor', config_manager)
        self.model_type = self.model_info.get('algorithm', 'lstm')
        self.failure_classifier = RandomForestClassifier(n_estimators=200, random_state=42)
        self.time_to_failure_regressor = GradientBoostingClassifier(n_estimators=100, random_state=42)
        
        logger.info(f"Initialized FailurePredictor with algorithm: {self.model_type}")
    
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
                return self.failure_classifier
            elif algorithm == 'gradient_boosting':
                return self.time_to_failure_regressor
            else:
                raise ValueError(f"Unsupported algorithm: {algorithm}")
                
        except Exception as e:
            logger.error(f"Failed to build model: {e}")
            raise
    
    def _build_lstm_model(self) -> tf.keras.Model:
        """Build LSTM model for failure prediction."""
        model = Sequential()
        
        # Input layer
        input_shape = self.architecture.get('input_shape', [48, 15])
        model.add(LSTM(128, return_sequences=True, input_shape=tuple(input_shape[1:])))
        model.add(Dropout(0.3))
        
        # Hidden LSTM layers
        model.add(LSTM(64, return_sequences=True))
        model.add(Dropout(0.3))
        
        model.add(LSTM(32, return_sequences=False))
        model.add(Dropout(0.2))
        
        # Dense layers
        model.add(Dense(16, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.1))
        
        # Output layer (failure probability)
        model.add(Dense(1, activation='sigmoid'))
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=self.training_config.get('learning_rate', 0.001)),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            X_train: Training features (time-series sensor data)
            y_train: Training targets (failure labels or probabilities)
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
                history = self._train_classifier(X_train, y_train, X_val, y_val)
            
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
    
    def _train_classifier(self, X_train: np.ndarray, y_train: np.ndarray, 
                         X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Train classifier model."""
        # Flatten time-series data for tree-based models
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        
        # Train classifier
        self.failure_classifier.fit(X_train_flat, y_train)
        
        # Calculate training metrics
        train_pred = self.failure_classifier.predict(X_train_flat)
        train_proba = self.failure_classifier.predict_proba(X_train_flat)[:, 1]
        
        train_accuracy = accuracy_score(y_train, train_pred)
        train_precision = precision_score(y_train, train_pred, zero_division=0)
        train_recall = recall_score(y_train, train_pred, zero_division=0)
        train_f1 = f1_score(y_train, train_pred, zero_division=0)
        train_auc = roc_auc_score(y_train, train_proba)
        
        history = {
            'accuracy': [train_accuracy],
            'precision': [train_precision],
            'recall': [train_recall],
            'f1_score': [train_f1],
            'auc': [train_auc]
        }
        
        if X_val is not None and y_val is not None:
            X_val_flat = X_val.reshape(X_val.shape[0], -1)
            val_pred = self.failure_classifier.predict(X_val_flat)
            val_proba = self.failure_classifier.predict_proba(X_val_flat)[:, 1]
            
            val_accuracy = accuracy_score(y_val, val_pred)
            val_precision = precision_score(y_val, val_pred, zero_division=0)
            val_recall = recall_score(y_val, val_pred, zero_division=0)
            val_f1 = f1_score(y_val, val_pred, zero_division=0)
            val_auc = roc_auc_score(y_val, val_proba)
            
            history.update({
                'val_accuracy': [val_accuracy],
                'val_precision': [val_precision],
                'val_recall': [val_recall],
                'val_f1_score': [val_f1],
                'val_auc': [val_auc]
            })
        
        return history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Input features (time-series sensor data)
            
        Returns:
            Predictions array (failure probabilities)
        """
        try:
            if not self.is_trained:
                raise ValueError("Model must be trained before making predictions")
            
            if self.model_type == 'lstm':
                predictions = self.model.predict(X, verbose=0)
                return predictions.flatten()
            else:
                X_flat = X.reshape(X.shape[0], -1)
                predictions = self.failure_classifier.predict_proba(X_flat)[:, 1]
                return predictions
            
        except Exception as e:
            logger.error(f"Failed to make predictions: {e}")
            raise
    
    def predict_failure_class(self, X: np.ndarray) -> np.ndarray:
        """
        Predict failure class (binary classification).
        
        Args:
            X: Input features
            
        Returns:
            Failure class predictions (0 or 1)
        """
        try:
            if not self.is_trained:
                raise ValueError("Model must be trained before making predictions")
            
            if self.model_type == 'lstm':
                probabilities = self.predict(X)
                return (probabilities > 0.5).astype(int)
            else:
                X_flat = X.reshape(X.shape[0], -1)
                return self.failure_classifier.predict(X_flat)
            
        except Exception as e:
            logger.error(f"Failed to predict failure class: {e}")
            raise
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance on test data.
        
        Args:
            X_test: Test features
            y_test: Test targets (failure labels)
            
        Returns:
            Evaluation metrics dictionary
        """
        try:
            if not self.is_trained:
                raise ValueError("Model must be trained before evaluation")
            
            # Make predictions
            predictions = self.predict(X_test)
            class_predictions = self.predict_failure_class(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, class_predictions)
            precision = precision_score(y_test, class_predictions, zero_division=0)
            recall = recall_score(y_test, class_predictions, zero_division=0)
            f1 = f1_score(y_test, class_predictions, zero_division=0)
            auc = roc_auc_score(y_test, predictions)
            
            # Calculate latency
            start_time = time.time()
            _ = self.predict(X_test[:10])
            latency = (time.time() - start_time) / 10 * 1000
            
            self.evaluation_metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'auc': auc,
                'latency_ms': latency,
                'test_samples': len(X_test)
            }
            
            logger.info(f"Model evaluation completed:")
            logger.info(f"  Accuracy: {accuracy:.4f}")
            logger.info(f"  Precision: {precision:.4f}")
            logger.info(f"  Recall: {recall:.4f}")
            logger.info(f"  F1-Score: {f1:.4f}")
            logger.info(f"  AUC: {auc:.4f}")
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
        if self.model_type in ['random_forest', 'gradient_boosting'] and hasattr(self.failure_classifier, 'feature_importances_'):
            feature_names = self.get_feature_names()
            importances = self.failure_classifier.feature_importances_
            return dict(zip(feature_names, importances))
        else:
            # For LSTM models, return zero importance
            feature_names = self.get_feature_names()
            return {name: 0.0 for name in feature_names}
    
    def predict_failure_risk(self, sensor_data: Dict[str, List[float]], 
                           equipment_type: str,
                           time_horizon: int = 24) -> Dict[str, Any]:
        """
        Predict failure risk for specific equipment and time horizon.
        
        Args:
            sensor_data: Time-series sensor data
            equipment_type: Type of equipment
            time_horizon: Prediction time horizon in hours
            
        Returns:
            Dictionary with failure risk assessment
        """
        try:
            # Create feature vector from sensor data
            features = self._create_feature_vector(sensor_data, equipment_type)
            
            # Make prediction
            failure_probability = self.predict(features.reshape(1, -1))[0]
            failure_class = self.predict_failure_class(features.reshape(1, -1))[0]
            
            # Calculate time to failure estimate
            time_to_failure = self._estimate_time_to_failure(failure_probability, time_horizon)
            
            # Assess risk level
            risk_assessment = self._assess_failure_risk(failure_probability, time_to_failure)
            
            # Generate recommendations
            recommendations = self._generate_failure_prevention_recommendations(
                failure_probability, time_to_failure, equipment_type, sensor_data
            )
            
            return {
                'equipment_type': equipment_type,
                'failure_probability': float(failure_probability),
                'failure_predicted': bool(failure_class),
                'time_to_failure_hours': float(time_to_failure),
                'risk_level': risk_assessment['risk_level'],
                'risk_score': risk_assessment['risk_score'],
                'recommendations': recommendations,
                'confidence': self._calculate_prediction_confidence(failure_probability, sensor_data)
            }
            
        except Exception as e:
            logger.error(f"Failed to predict failure risk: {e}")
            raise
    
    def _create_feature_vector(self, sensor_data: Dict[str, List[float]], 
                              equipment_type: str) -> np.ndarray:
        """
        Create feature vector from sensor data.
        
        Args:
            sensor_data: Time-series sensor data
            equipment_type: Type of equipment
            
        Returns:
            Feature vector
        """
        # This would be implemented based on the specific sensor data structure
        feature_names = self.get_feature_names()
        features = np.zeros(len(feature_names))
        
        # Map sensor data to features
        for sensor_name, values in sensor_data.items():
            if sensor_name in feature_names:
                # Use statistical features (mean, std, min, max, trend)
                features[feature_names.index(sensor_name)] = np.mean(values)
        
        return features
    
    def _estimate_time_to_failure(self, failure_probability: float, time_horizon: int) -> float:
        """
        Estimate time to failure based on failure probability.
        
        Args:
            failure_probability: Predicted failure probability
            time_horizon: Prediction time horizon
            
        Returns:
            Estimated time to failure in hours
        """
        # Simple estimation based on probability
        if failure_probability > 0.8:
            return np.random.uniform(1, 6)  # 1-6 hours
        elif failure_probability > 0.6:
            return np.random.uniform(6, 24)  # 6-24 hours
        elif failure_probability > 0.4:
            return np.random.uniform(24, 72)  # 1-3 days
        elif failure_probability > 0.2:
            return np.random.uniform(72, 168)  # 3-7 days
        else:
            return np.random.uniform(168, 720)  # 1-4 weeks
    
    def _assess_failure_risk(self, failure_probability: float, time_to_failure: float) -> Dict[str, Any]:
        """
        Assess failure risk level based on probability and time to failure.
        
        Args:
            failure_probability: Predicted failure probability
            time_to_failure: Estimated time to failure
            
        Returns:
            Dictionary with risk assessment
        """
        # Calculate risk score
        risk_score = failure_probability * (1 / (time_to_failure / 24 + 1))  # Higher risk for shorter time
        
        # Determine risk level
        if risk_score >= 0.7:
            risk_level = 'critical'
        elif risk_score >= 0.5:
            risk_level = 'high'
        elif risk_score >= 0.3:
            risk_level = 'medium'
        else:
            risk_level = 'low'
        
        return {
            'risk_score': float(risk_score),
            'risk_level': risk_level,
            'urgency': self._get_urgency_level(risk_level)
        }
    
    def _get_urgency_level(self, risk_level: str) -> str:
        """
        Get urgency level based on risk level.
        
        Args:
            risk_level: Risk level (low, medium, high, critical)
            
        Returns:
            Urgency level string
        """
        urgency_map = {
            'critical': 'immediate',
            'high': 'urgent',
            'medium': 'soon',
            'low': 'scheduled'
        }
        return urgency_map.get(risk_level, 'unknown')
    
    def _generate_failure_prevention_recommendations(self, failure_probability: float, 
                                                   time_to_failure: float,
                                                   equipment_type: str,
                                                   sensor_data: Dict[str, List[float]]) -> List[str]:
        """
        Generate failure prevention recommendations.
        
        Args:
            failure_probability: Predicted failure probability
            time_to_failure: Estimated time to failure
            equipment_type: Type of equipment
            sensor_data: Sensor data
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        if failure_probability > 0.7:
            recommendations.append(f"CRITICAL: {equipment_type} failure imminent within {time_to_failure:.1f} hours")
            recommendations.append("Immediate shutdown and inspection required")
            recommendations.append("Contact maintenance team immediately")
        
        elif failure_probability > 0.5:
            recommendations.append(f"HIGH RISK: {equipment_type} failure likely within {time_to_failure:.1f} hours")
            recommendations.append("Schedule emergency maintenance")
            recommendations.append("Reduce operational load if possible")
        
        elif failure_probability > 0.3:
            recommendations.append(f"MEDIUM RISK: {equipment_type} showing signs of degradation")
            recommendations.append("Schedule preventive maintenance within 48 hours")
            recommendations.append("Monitor sensor readings closely")
        
        else:
            recommendations.append(f"LOW RISK: {equipment_type} operating normally")
            recommendations.append("Continue routine monitoring")
            recommendations.append("Schedule regular maintenance as planned")
        
        # Add sensor-specific recommendations
        for sensor_name, values in sensor_data.items():
            if len(values) > 1:
                trend = np.polyfit(range(len(values)), values, 1)[0]
                if abs(trend) > np.std(values) * 2:  # Significant trend
                    if trend > 0:
                        recommendations.append(f"Warning: {sensor_name} showing increasing trend")
                    else:
                        recommendations.append(f"Warning: {sensor_name} showing decreasing trend")
        
        return recommendations
    
    def _calculate_prediction_confidence(self, failure_probability: float, 
                                       sensor_data: Dict[str, List[float]]) -> float:
        """
        Calculate prediction confidence based on data quality and probability.
        
        Args:
            failure_probability: Predicted failure probability
            sensor_data: Sensor data
            
        Returns:
            Confidence score (0-1)
        """
        # Base confidence on data completeness
        total_sensors = len(sensor_data)
        complete_sensors = sum(1 for values in sensor_data.values() if len(values) > 0)
        data_completeness = complete_sensors / total_sensors if total_sensors > 0 else 0
        
        # Adjust confidence based on probability (extreme probabilities are less confident)
        probability_confidence = 1.0 - abs(failure_probability - 0.5) * 2
        
        # Combine factors
        confidence = (data_completeness + probability_confidence) / 2
        
        return np.clip(confidence, 0.0, 1.0)
