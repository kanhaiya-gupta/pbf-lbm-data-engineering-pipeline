"""
Equipment Health Monitor Model

This module implements a model for monitoring equipment health and performance
for PBF-LB/M systems including laser, recoater, chamber, and gas systems.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time

from ..base_model import BaseModel

logger = logging.getLogger(__name__)


class EquipmentHealthMonitor(BaseModel):
    """
    Model for monitoring equipment health and performance in PBF-LB/M systems.
    
    This model monitors:
    - Laser system health (power stability, beam quality, cooling efficiency)
    - Recoater system health (movement accuracy, wear patterns)
    - Chamber system health (pressure, temperature, atmosphere)
    - Gas system health (flow rates, purity, pressure)
    - Overall system performance degradation
    """
    
    def __init__(self, config_manager=None):
        """
        Initialize the equipment health monitor.
        
        Args:
            config_manager: Configuration manager instance
        """
        super().__init__('equipment_health_monitor', config_manager)
        self.model_type = self.model_info.get('algorithm', 'lstm')
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.health_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        
        logger.info(f"Initialized EquipmentHealthMonitor with algorithm: {self.model_type}")
    
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
            elif algorithm == 'isolation_forest':
                return self.anomaly_detector
            elif algorithm == 'random_forest':
                return self.health_classifier
            else:
                raise ValueError(f"Unsupported algorithm: {algorithm}")
                
        except Exception as e:
            logger.error(f"Failed to build model: {e}")
            raise
    
    def _build_lstm_model(self) -> tf.keras.Model:
        """Build LSTM model for time-series health monitoring."""
        model = Sequential()
        
        # Input layer
        input_shape = self.architecture.get('input_shape', [24, 10])
        model.add(LSTM(64, return_sequences=True, input_shape=tuple(input_shape[1:])))
        model.add(Dropout(0.2))
        
        # Hidden LSTM layers
        model.add(LSTM(32, return_sequences=False))
        model.add(Dropout(0.2))
        
        # Dense layers
        model.add(Dense(16, activation='relu'))
        model.add(Dropout(0.1))
        
        # Output layer (health score 0-1)
        model.add(Dense(1, activation='sigmoid'))
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=self.training_config.get('learning_rate', 0.001)),
            loss='mse',
            metrics=['mae', 'mse']
        )
        
        return model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            X_train: Training features (time-series sensor data)
            y_train: Training targets (health scores or anomaly labels)
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
            elif self.model_type == 'isolation_forest':
                history = self._train_anomaly_detector(X_train, y_train)
            else:
                history = self._train_health_classifier(X_train, y_train, X_val, y_val)
            
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
            epochs=self.training_config.get('epochs', 100),
            validation_data=(X_val, y_val) if X_val is not None else None,
            callbacks=callbacks,
            verbose=1
        )
        
        return history.history
    
    def _train_anomaly_detector(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """Train anomaly detection model."""
        # Flatten time-series data for isolation forest
        X_flat = X_train.reshape(X_train.shape[0], -1)
        
        # Train anomaly detector
        self.anomaly_detector.fit(X_flat)
        
        # Calculate training metrics
        train_scores = self.anomaly_detector.decision_function(X_flat)
        train_predictions = self.anomaly_detector.predict(X_flat)
        
        # Convert predictions to binary (1 for normal, -1 for anomaly)
        train_predictions_binary = (train_predictions == 1).astype(int)
        
        if len(np.unique(y_train)) > 1:  # If we have labeled data
            accuracy = accuracy_score(y_train, train_predictions_binary)
            precision = precision_score(y_train, train_predictions_binary, zero_division=0)
            recall = recall_score(y_train, train_predictions_binary, zero_division=0)
            f1 = f1_score(y_train, train_predictions_binary, zero_division=0)
        else:
            accuracy = precision = recall = f1 = 0.0
        
        return {
            'accuracy': [accuracy],
            'precision': [precision],
            'recall': [recall],
            'f1_score': [f1],
            'anomaly_ratio': [np.mean(train_predictions == -1)]
        }
    
    def _train_health_classifier(self, X_train: np.ndarray, y_train: np.ndarray, 
                                X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Train health classification model."""
        # Flatten time-series data for random forest
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        
        # Train classifier
        self.health_classifier.fit(X_train_flat, y_train)
        
        # Calculate training metrics
        train_pred = self.health_classifier.predict(X_train_flat)
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
        
        if X_val is not None and y_val is not None:
            X_val_flat = X_val.reshape(X_val.shape[0], -1)
            val_pred = self.health_classifier.predict(X_val_flat)
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
            X: Input features (time-series sensor data)
            
        Returns:
            Predictions array (health scores or anomaly predictions)
        """
        try:
            if not self.is_trained:
                raise ValueError("Model must be trained before making predictions")
            
            if self.model_type == 'lstm':
                predictions = self.model.predict(X, verbose=0)
                return predictions.flatten()
            elif self.model_type == 'isolation_forest':
                X_flat = X.reshape(X.shape[0], -1)
                predictions = self.anomaly_detector.predict(X_flat)
                return predictions
            else:  # random_forest
                X_flat = X.reshape(X.shape[0], -1)
                predictions = self.health_classifier.predict(X_flat)
                return predictions
            
        except Exception as e:
            logger.error(f"Failed to make predictions: {e}")
            raise
    
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
            
            if self.model_type == 'lstm':
                # For regression (health scores)
                mse = np.mean((y_test - predictions) ** 2)
                mae = np.mean(np.abs(y_test - predictions))
                r2 = 1 - (np.sum((y_test - predictions) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2))
                
                metrics = {
                    'mse': mse,
                    'mae': mae,
                    'r2': r2
                }
            else:
                # For classification (anomaly detection or health classification)
                if self.model_type == 'isolation_forest':
                    predictions_binary = (predictions == 1).astype(int)
                else:
                    predictions_binary = predictions
                
                accuracy = accuracy_score(y_test, predictions_binary)
                precision = precision_score(y_test, predictions_binary, average='weighted', zero_division=0)
                recall = recall_score(y_test, predictions_binary, average='weighted', zero_division=0)
                f1 = f1_score(y_test, predictions_binary, average='weighted', zero_division=0)
                
                metrics = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1
                }
            
            # Calculate latency
            start_time = time.time()
            _ = self.predict(X_test[:10])
            latency = (time.time() - start_time) / 10 * 1000
            
            metrics['latency_ms'] = latency
            
            self.evaluation_metrics = metrics
            
            logger.info(f"Model evaluation completed:")
            for metric, value in metrics.items():
                if metric != 'latency_ms':
                    logger.info(f"  {metric}: {value:.4f}")
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
        if self.model_type == 'random_forest' and hasattr(self.health_classifier, 'feature_importances_'):
            feature_names = self.get_feature_names()
            importances = self.health_classifier.feature_importances_
            return dict(zip(feature_names, importances))
        else:
            # For other models, return zero importance
            feature_names = self.get_feature_names()
            return {name: 0.0 for name in feature_names}
    
    def monitor_equipment_health(self, sensor_data: Dict[str, List[float]], 
                                equipment_type: str) -> Dict[str, Any]:
        """
        Monitor equipment health based on sensor data.
        
        Args:
            sensor_data: Time-series sensor data for different equipment
            equipment_type: Type of equipment (laser, recoater, chamber, gas_system)
            
        Returns:
            Dictionary with health assessment and recommendations
        """
        try:
            # Convert sensor data to feature vector
            features = self._create_feature_vector(sensor_data, equipment_type)
            
            # Make prediction
            if self.model_type == 'lstm':
                health_score = self.predict(features.reshape(1, -1))[0]
                health_status = self._score_to_status(health_score)
            else:
                prediction = self.predict(features.reshape(1, -1))[0]
                health_status = self._prediction_to_status(prediction)
                health_score = 1.0 if health_status == 'healthy' else 0.0
            
            # Analyze trends and anomalies
            trend_analysis = self._analyze_trends(sensor_data)
            anomaly_detection = self._detect_anomalies(sensor_data)
            
            # Generate maintenance recommendations
            recommendations = self._generate_maintenance_recommendations(
                health_status, health_score, trend_analysis, anomaly_detection, equipment_type
            )
            
            return {
                'equipment_type': equipment_type,
                'health_score': float(health_score),
                'health_status': health_status,
                'trend_analysis': trend_analysis,
                'anomaly_detection': anomaly_detection,
                'maintenance_recommendations': recommendations,
                'risk_assessment': self._assess_risk(health_score, trend_analysis)
            }
            
        except Exception as e:
            logger.error(f"Failed to monitor equipment health: {e}")
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
        # For now, return a placeholder
        feature_names = self.get_feature_names()
        features = np.zeros(len(feature_names))
        
        # Map sensor data to features
        for sensor_name, values in sensor_data.items():
            if sensor_name in feature_names:
                # Use statistical features (mean, std, min, max)
                features[feature_names.index(sensor_name)] = np.mean(values)
        
        return features
    
    def _score_to_status(self, score: float) -> str:
        """
        Convert health score to status.
        
        Args:
            score: Health score (0-1)
            
        Returns:
            Health status string
        """
        if score >= 0.9:
            return 'excellent'
        elif score >= 0.8:
            return 'good'
        elif score >= 0.6:
            return 'fair'
        elif score >= 0.4:
            return 'poor'
        else:
            return 'critical'
    
    def _prediction_to_status(self, prediction: int) -> str:
        """
        Convert prediction to status.
        
        Args:
            prediction: Model prediction
            
        Returns:
            Health status string
        """
        if prediction == 1:
            return 'healthy'
        else:
            return 'anomalous'
    
    def _analyze_trends(self, sensor_data: Dict[str, List[float]]) -> Dict[str, Any]:
        """
        Analyze trends in sensor data.
        
        Args:
            sensor_data: Time-series sensor data
            
        Returns:
            Dictionary with trend analysis
        """
        trends = {}
        
        for sensor_name, values in sensor_data.items():
            if len(values) > 1:
                # Calculate trend (slope of linear regression)
                x = np.arange(len(values))
                slope = np.polyfit(x, values, 1)[0]
                
                if slope > 0.1:
                    trend = 'increasing'
                elif slope < -0.1:
                    trend = 'decreasing'
                else:
                    trend = 'stable'
                
                trends[sensor_name] = {
                    'trend': trend,
                    'slope': float(slope),
                    'variance': float(np.var(values)),
                    'mean': float(np.mean(values))
                }
        
        return trends
    
    def _detect_anomalies(self, sensor_data: Dict[str, List[float]]) -> Dict[str, Any]:
        """
        Detect anomalies in sensor data.
        
        Args:
            sensor_data: Time-series sensor data
            
        Returns:
            Dictionary with anomaly detection results
        """
        anomalies = {}
        
        for sensor_name, values in sensor_data.items():
            if len(values) > 3:
                # Simple anomaly detection using IQR method
                q1 = np.percentile(values, 25)
                q3 = np.percentile(values, 75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                anomaly_count = np.sum((values < lower_bound) | (values > upper_bound))
                anomaly_ratio = anomaly_count / len(values)
                
                anomalies[sensor_name] = {
                    'anomaly_count': int(anomaly_count),
                    'anomaly_ratio': float(anomaly_ratio),
                    'has_anomalies': anomaly_ratio > 0.1
                }
        
        return anomalies
    
    def _generate_maintenance_recommendations(self, health_status: str, health_score: float,
                                            trend_analysis: Dict[str, Any], 
                                            anomaly_detection: Dict[str, Any],
                                            equipment_type: str) -> List[str]:
        """
        Generate maintenance recommendations based on health assessment.
        
        Args:
            health_status: Equipment health status
            health_score: Health score
            trend_analysis: Trend analysis results
            anomaly_detection: Anomaly detection results
            equipment_type: Type of equipment
            
        Returns:
            List of maintenance recommendations
        """
        recommendations = []
        
        if health_status in ['critical', 'poor']:
            recommendations.append(f"URGENT: {equipment_type} requires immediate attention")
            recommendations.append("Schedule maintenance as soon as possible")
        
        elif health_status == 'fair':
            recommendations.append(f"Monitor {equipment_type} closely")
            recommendations.append("Schedule preventive maintenance within 2 weeks")
        
        # Check for concerning trends
        for sensor_name, trend_info in trend_analysis.items():
            if trend_info['trend'] == 'increasing' and trend_info['slope'] > 0.5:
                recommendations.append(f"Warning: {sensor_name} shows concerning upward trend")
            elif trend_info['trend'] == 'decreasing' and trend_info['slope'] < -0.5:
                recommendations.append(f"Warning: {sensor_name} shows concerning downward trend")
        
        # Check for anomalies
        for sensor_name, anomaly_info in anomaly_detection.items():
            if anomaly_info['has_anomalies']:
                recommendations.append(f"Anomalies detected in {sensor_name} - investigate further")
        
        return recommendations
    
    def _assess_risk(self, health_score: float, trend_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess risk level based on health score and trends.
        
        Args:
            health_score: Health score
            trend_analysis: Trend analysis results
            
        Returns:
            Dictionary with risk assessment
        """
        # Calculate risk score
        risk_score = 1.0 - health_score
        
        # Adjust risk based on trends
        for trend_info in trend_analysis.values():
            if trend_info['trend'] in ['increasing', 'decreasing']:
                risk_score += 0.1
        
        risk_score = min(risk_score, 1.0)
        
        # Determine risk level
        if risk_score >= 0.8:
            risk_level = 'high'
        elif risk_score >= 0.5:
            risk_level = 'medium'
        else:
            risk_level = 'low'
        
        return {
            'risk_score': float(risk_score),
            'risk_level': risk_level,
            'recommended_action': self._get_risk_action(risk_level)
        }
    
    def _get_risk_action(self, risk_level: str) -> str:
        """
        Get recommended action based on risk level.
        
        Args:
            risk_level: Risk level (low, medium, high)
            
        Returns:
            Recommended action string
        """
        actions = {
            'low': 'Continue monitoring',
            'medium': 'Schedule preventive maintenance',
            'high': 'Immediate maintenance required'
        }
        return actions.get(risk_level, 'Unknown risk level')
