"""
Real-time Defect Predictor Model

This module implements an LSTM-based model for real-time defect prediction
based on ISPM sensor data during PBF-LB/M processes.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
import time

from ..base_model import BaseModel

logger = logging.getLogger(__name__)


class RealTimeDefectPredictor(BaseModel):
    """
    LSTM-based real-time defect predictor for PBF-LB/M processes.
    
    This model predicts defect types (NONE, MINOR, MAJOR) based on ISPM sensor data
    including laser power, scan speed, temperature, melt pool size, and other process parameters.
    """
    
    def __init__(self, config_manager=None):
        """
        Initialize the real-time defect predictor.
        
        Args:
            config_manager: Configuration manager instance
        """
        super().__init__('real_time_defect_predictor', config_manager)
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.sequence_length = self.data_config.get('preprocessing', {}).get('sequence_length', 24)
        self.prediction_horizon = self.data_config.get('preprocessing', {}).get('prediction_horizon', 1)
        
        logger.info(f"Initialized RealTimeDefectPredictor with sequence length: {self.sequence_length}")
    
    def build_model(self) -> tf.keras.Model:
        """
        Build the LSTM model architecture based on configuration.
        
        Returns:
            Compiled Keras model
        """
        try:
            # Get architecture configuration
            arch_config = self.architecture
            input_shape = arch_config.get('input_shape', [24, 10])
            layers_config = arch_config.get('layers', [])
            
            # Build model
            model = Sequential()
            
            # Add layers based on configuration
            for i, layer_config in enumerate(layers_config):
                layer_type = layer_config.get('type')
                
                if layer_type == 'LSTM':
                    units = layer_config.get('units', 64)
                    return_sequences = layer_config.get('return_sequences', True)
                    dropout = layer_config.get('dropout', 0.2)
                    recurrent_dropout = layer_config.get('recurrent_dropout', 0.2)
                    
                    if i == 0:
                        # First LSTM layer with input shape
                        model.add(LSTM(
                            units=units,
                            return_sequences=return_sequences,
                            dropout=dropout,
                            recurrent_dropout=recurrent_dropout,
                            input_shape=tuple(input_shape[1:])  # Remove batch dimension
                        ))
                    else:
                        model.add(LSTM(
                            units=units,
                            return_sequences=return_sequences,
                            dropout=dropout,
                            recurrent_dropout=recurrent_dropout
                        ))
                
                elif layer_type == 'Dense':
                    units = layer_config.get('units', 16)
                    activation = layer_config.get('activation', 'relu')
                    dropout = layer_config.get('dropout', 0.3)
                    
                    model.add(Dense(units=units, activation=activation))
                    if dropout > 0:
                        model.add(Dropout(dropout))
            
            # Compile model
            optimizer = arch_config.get('optimizer', 'adam')
            loss_function = arch_config.get('loss_function', 'categorical_crossentropy')
            metrics = arch_config.get('metrics', ['accuracy'])
            
            model.compile(
                optimizer=optimizer,
                loss=loss_function,
                metrics=metrics
            )
            
            self.model = model
            logger.info("LSTM model built successfully")
            return model
            
        except Exception as e:
            logger.error(f"Failed to build model: {e}")
            raise
    
    def prepare_data(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Prepare data for training/prediction.
        
        Args:
            X: Input features
            y: Target labels (optional)
            
        Returns:
            Tuple of (prepared_X, prepared_y)
        """
        try:
            # Normalize features
            if not hasattr(self.scaler, 'mean_'):
                X_scaled = self.scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
            else:
                X_scaled = self.scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
            
            # Prepare targets if provided
            y_prepared = None
            if y is not None:
                if len(y.shape) == 1:
                    # Encode labels if not already encoded
                    if not hasattr(self.label_encoder, 'classes_'):
                        y_encoded = self.label_encoder.fit_transform(y)
                    else:
                        y_encoded = self.label_encoder.transform(y)
                    
                    # Convert to categorical
                    y_prepared = tf.keras.utils.to_categorical(y_encoded, num_classes=3)
                else:
                    y_prepared = y
            
            return X_scaled, y_prepared
            
        except Exception as e:
            logger.error(f"Failed to prepare data: {e}")
            raise
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Train the LSTM model.
        
        Args:
            X_train: Training features with shape (samples, sequence_length, features)
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            
        Returns:
            Training history dictionary
        """
        try:
            # Build model if not already built
            if self.model is None:
                self.build_model()
            
            # Prepare data
            X_train_prep, y_train_prep = self.prepare_data(X_train, y_train)
            X_val_prep, y_val_prep = None, None
            
            if X_val is not None and y_val is not None:
                X_val_prep, y_val_prep = self.prepare_data(X_val, y_val)
            
            # Get training configuration
            train_config = self.training_config
            batch_size = train_config.get('batch_size', 32)
            epochs = train_config.get('epochs', 100)
            learning_rate = train_config.get('learning_rate', 0.001)
            
            # Set up callbacks
            callbacks = []
            
            # Early stopping
            early_stopping_config = train_config.get('early_stopping', {})
            if early_stopping_config.get('enabled', True):
                callbacks.append(EarlyStopping(
                    monitor=early_stopping_config.get('monitor', 'val_loss'),
                    patience=early_stopping_config.get('patience', 10),
                    mode=early_stopping_config.get('mode', 'min'),
                    restore_best_weights=early_stopping_config.get('restore_best_weights', True),
                    min_delta=early_stopping_config.get('min_delta', 0.001)
                ))
            
            # Learning rate reduction
            lr_schedule_config = train_config.get('learning_rate_schedule', {})
            if lr_schedule_config.get('enabled', True):
                callbacks.append(ReduceLROnPlateau(
                    monitor=lr_schedule_config.get('monitor', 'val_loss'),
                    factor=lr_schedule_config.get('factor', 0.5),
                    patience=lr_schedule_config.get('patience', 5),
                    min_lr=lr_schedule_config.get('min_lr', 0.0001)
                ))
            
            # Model checkpoint
            callbacks.append(ModelCheckpoint(
                filepath='best_model.h5',
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=False
            ))
            
            # Train model
            start_time = time.time()
            
            history = self.model.fit(
                X_train_prep, y_train_prep,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(X_val_prep, y_val_prep) if X_val_prep is not None else None,
                callbacks=callbacks,
                verbose=1
            )
            
            training_time = time.time() - start_time
            
            # Store training history
            self.training_history = {
                'history': history.history,
                'training_time': training_time,
                'epochs_trained': len(history.history['loss']),
                'final_loss': history.history['loss'][-1],
                'final_val_loss': history.history.get('val_loss', [None])[-1],
                'final_accuracy': history.history['accuracy'][-1],
                'final_val_accuracy': history.history.get('val_accuracy', [None])[-1]
            }
            
            self.is_trained = True
            
            logger.info(f"Model training completed in {training_time:.2f} seconds")
            logger.info(f"Final training accuracy: {self.training_history['final_accuracy']:.4f}")
            
            return self.training_history
            
        except Exception as e:
            logger.error(f"Failed to train model: {e}")
            raise
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Input features with shape (samples, sequence_length, features)
            
        Returns:
            Predictions array
        """
        try:
            if not self.is_trained:
                raise ValueError("Model must be trained before making predictions")
            
            # Prepare data
            X_prep, _ = self.prepare_data(X)
            
            # Make predictions
            predictions = self.model.predict(X_prep, verbose=0)
            
            # Convert probabilities to class predictions
            class_predictions = np.argmax(predictions, axis=1)
            
            # Decode labels if encoder is available
            if hasattr(self.label_encoder, 'classes_'):
                class_predictions = self.label_encoder.inverse_transform(class_predictions)
            
            return class_predictions
            
        except Exception as e:
            logger.error(f"Failed to make predictions: {e}")
            raise
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            X: Input features with shape (samples, sequence_length, features)
            
        Returns:
            Prediction probabilities array
        """
        try:
            if not self.is_trained:
                raise ValueError("Model must be trained before making predictions")
            
            # Prepare data
            X_prep, _ = self.prepare_data(X)
            
            # Get probabilities
            probabilities = self.model.predict(X_prep, verbose=0)
            
            return probabilities
            
        except Exception as e:
            logger.error(f"Failed to get prediction probabilities: {e}")
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
            
            # Prepare data
            X_test_prep, y_test_prep = self.prepare_data(X_test, y_test)
            
            # Make predictions
            predictions = self.predict(X_test)
            probabilities = self.predict_proba(X_test)
            
            # Calculate metrics
            if len(y_test.shape) == 1:
                # Convert string labels to numeric for metrics calculation
                if hasattr(self.label_encoder, 'classes_'):
                    y_test_numeric = self.label_encoder.transform(y_test)
                else:
                    y_test_numeric = y_test
            else:
                y_test_numeric = np.argmax(y_test, axis=1)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test_numeric, predictions)
            precision = precision_score(y_test_numeric, predictions, average='macro', zero_division=0)
            recall = recall_score(y_test_numeric, predictions, average='macro', zero_division=0)
            f1 = f1_score(y_test_numeric, predictions, average='macro', zero_division=0)
            
            # Confusion matrix
            cm = confusion_matrix(y_test_numeric, predictions)
            
            # Calculate latency
            start_time = time.time()
            _ = self.predict(X_test[:10])  # Predict on small sample
            latency = (time.time() - start_time) / 10 * 1000  # Average latency in milliseconds
            
            # Store evaluation metrics
            self.evaluation_metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'confusion_matrix': cm.tolist(),
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
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance (placeholder for LSTM models).
        
        Returns:
            Feature importance dictionary
        """
        # LSTM models don't have direct feature importance
        # This could be implemented using SHAP or other methods
        feature_names = self.get_feature_names()
        return {name: 0.0 for name in feature_names}
    
    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get detailed model summary.
        
        Returns:
            Model summary dictionary
        """
        summary = super().get_model_summary()
        
        if self.model is not None:
            summary.update({
                'total_params': self.model.count_params(),
                'trainable_params': sum([tf.keras.backend.count_params(w) for w in self.model.trainable_weights]),
                'non_trainable_params': sum([tf.keras.backend.count_params(w) for w in self.model.non_trainable_weights]),
                'sequence_length': self.sequence_length,
                'prediction_horizon': self.prediction_horizon
            })
        
        if self.training_history:
            summary.update({
                'training_time': self.training_history.get('training_time'),
                'epochs_trained': self.training_history.get('epochs_trained'),
                'final_accuracy': self.training_history.get('final_accuracy')
            })
        
        return summary
