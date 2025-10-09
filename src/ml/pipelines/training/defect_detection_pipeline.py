"""
Defect Detection Training Pipeline

This module implements the training pipeline for defect detection models in PBF-LB/M processes.
It handles data ingestion, preprocessing, feature engineering, model training, and evaluation
for real-time defect prediction models.
"""

import logging
import time
from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import mlflow
import mlflow.tensorflow

from .base_training_pipeline import BaseTrainingPipeline
from ...models.defect_detection import RealTimeDefectPredictor, ImageDefectClassifier, DefectSeverityAssessor, RootCauseAnalyzer
from ...features.sensor_features import PyrometerFeatures, CameraFeatures, AccelerometerFeatures, TemperatureFeatures
from ...features.temporal_features import TimeSeriesFeatures, LagFeatures, RollingFeatures, FrequencyFeatures
from ...utils.evaluation import ClassificationMetrics, TimeSeriesMetrics, CustomMetrics

logger = logging.getLogger(__name__)


class DefectDetectionTrainingPipeline(BaseTrainingPipeline):
    """
    Training pipeline for defect detection models.
    
    This pipeline handles the complete training workflow for defect detection models including:
    - Real-time defect prediction
    - Image-based defect classification
    - Defect severity assessment
    - Root cause analysis
    """
    
    def __init__(self, config_manager=None):
        """
        Initialize the defect detection training pipeline.
        
        Args:
            config_manager: Configuration manager instance
        """
        super().__init__("defect_detection_pipeline", config_manager)
        
        # Initialize feature engineers
        self.feature_engineers = {
            'pyrometer': PyrometerFeatures(self.config_manager),
            'camera': CameraFeatures(self.config_manager),
            'accelerometer': AccelerometerFeatures(self.config_manager),
            'temperature': TemperatureFeatures(self.config_manager),
            'time_series': TimeSeriesFeatures(self.config_manager),
            'lag': LagFeatures(self.config_manager),
            'rolling': RollingFeatures(self.config_manager),
            'frequency': FrequencyFeatures(self.config_manager)
        }
        
        # Initialize models
        self.models = {
            'real_time_predictor': RealTimeDefectPredictor(self.config_manager),
            'image_classifier': ImageDefectClassifier(self.config_manager),
            'severity_assessor': DefectSeverityAssessor(self.config_manager),
            'root_cause_analyzer': RootCauseAnalyzer(self.config_manager)
        }
        
        # Initialize evaluators
        self.evaluators = {
            'classification': ClassificationMetrics(),
            'time_series': TimeSeriesMetrics(),
            'custom': CustomMetrics()
        }
        
        logger.info("Initialized DefectDetectionTrainingPipeline")
    
    def _execute_model_training_stage(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute model training stage for defect detection models.
        
        Args:
            config: Stage configuration
            
        Returns:
            Stage execution results
        """
        logger.info("Executing defect detection model training stage")
        
        # Get split data
        split_data = self.stage_results.get('data_splitting', {})
        train_data = split_data.get('train_data')
        validation_data = split_data.get('validation_data')
        
        if train_data is None or validation_data is None:
            raise ValueError("No training or validation data available")
        
        # Get training parameters
        training_params = config.get('training_params', {})
        batch_size = training_params.get('batch_size', 32)
        epochs = training_params.get('epochs', 100)
        learning_rate = training_params.get('learning_rate', 0.001)
        
        # Prepare data for training
        X_train, y_train = self._prepare_training_data(train_data)
        X_val, y_val = self._prepare_training_data(validation_data)
        
        # Train multiple models
        trained_models = {}
        training_results = {}
        
        for model_name, model in self.models.items():
            logger.info(f"Training {model_name}")
            
            try:
                # Train the model
                training_start_time = time.time()
                
                if model_name == 'real_time_predictor':
                    # Train LSTM-based real-time predictor
                    trained_model, history = self._train_real_time_predictor(
                        model, X_train, y_train, X_val, y_val, training_params
                    )
                elif model_name == 'image_classifier':
                    # Train CNN-based image classifier
                    trained_model, history = self._train_image_classifier(
                        model, X_train, y_train, X_val, y_val, training_params
                    )
                elif model_name == 'severity_assessor':
                    # Train severity assessment model
                    trained_model, history = self._train_severity_assessor(
                        model, X_train, y_train, X_val, y_val, training_params
                    )
                elif model_name == 'root_cause_analyzer':
                    # Train root cause analysis model
                    trained_model, history = self._train_root_cause_analyzer(
                        model, X_train, y_train, X_val, y_val, training_params
                    )
                
                training_duration = time.time() - training_start_time
                
                # Store results
                trained_models[model_name] = trained_model
                training_results[model_name] = {
                    'model': trained_model,
                    'history': history,
                    'training_duration': training_duration,
                    'training_params': training_params
                }
                
                # Log training metrics to MLflow
                self._log_training_metrics(model_name, history, training_duration)
                
                logger.info(f"Successfully trained {model_name} in {training_duration:.2f} seconds")
                
            except Exception as e:
                logger.error(f"Failed to train {model_name}: {e}")
                raise
        
        return {
            'trained_models': trained_models,
            'training_results': training_results,
            'training_params': training_params
        }
    
    def _prepare_training_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data for model training.
        
        Args:
            data: Input data DataFrame
            
        Returns:
            Tuple of (X, y) arrays
        """
        # Separate features and target
        target_columns = ['defect_type', 'defect_severity', 'defect_probability']
        feature_columns = [col for col in data.columns if col not in target_columns]
        
        X = data[feature_columns].values
        y = data[target_columns[0]].values  # Primary target: defect_type
        
        # Handle categorical target
        if y.dtype == 'object':
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y)
        
        return X, y
    
    def _train_real_time_predictor(self, model, X_train, y_train, X_val, y_val, training_params):
        """
        Train the real-time defect predictor model.
        
        Args:
            model: Model instance
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            training_params: Training parameters
            
        Returns:
            Tuple of (trained_model, training_history)
        """
        # Reshape data for LSTM (samples, timesteps, features)
        if len(X_train.shape) == 2:
            # Add timestep dimension
            X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
            X_val = X_val.reshape(X_val.shape[0], 1, X_val.shape[1])
        
        # Build model architecture
        model.build_model(
            input_shape=X_train.shape[1:],
            num_classes=len(np.unique(y_train))
        )
        
        # Compile model
        model.compile_model(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train model
        history = model.train(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=training_params.get('epochs', 100),
            batch_size=training_params.get('batch_size', 32),
            callbacks=self._get_training_callbacks(training_params)
        )
        
        return model.model, history
    
    def _train_image_classifier(self, model, X_train, y_train, X_val, y_val, training_params):
        """
        Train the image defect classifier model.
        
        Args:
            model: Model instance
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            training_params: Training parameters
            
        Returns:
            Tuple of (trained_model, training_history)
        """
        # Reshape data for CNN (samples, height, width, channels)
        if len(X_train.shape) == 2:
            # Assume square images, add channel dimension
            img_size = int(np.sqrt(X_train.shape[1]))
            X_train = X_train.reshape(X_train.shape[0], img_size, img_size, 1)
            X_val = X_val.reshape(X_val.shape[0], img_size, img_size, 1)
        
        # Build model architecture
        model.build_model(
            input_shape=X_train.shape[1:],
            num_classes=len(np.unique(y_train))
        )
        
        # Compile model
        model.compile_model(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train model
        history = model.train(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=training_params.get('epochs', 100),
            batch_size=training_params.get('batch_size', 32),
            callbacks=self._get_training_callbacks(training_params)
        )
        
        return model.model, history
    
    def _train_severity_assessor(self, model, X_train, y_train, X_val, y_val, training_params):
        """
        Train the defect severity assessor model.
        
        Args:
            model: Model instance
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            training_params: Training parameters
            
        Returns:
            Tuple of (trained_model, training_history)
        """
        # Build model architecture
        model.build_model(
            input_shape=X_train.shape[1:],
            num_classes=len(np.unique(y_train))
        )
        
        # Compile model
        model.compile_model(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train model
        history = model.train(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=training_params.get('epochs', 100),
            batch_size=training_params.get('batch_size', 32),
            callbacks=self._get_training_callbacks(training_params)
        )
        
        return model.model, history
    
    def _train_root_cause_analyzer(self, model, X_train, y_train, X_val, y_val, training_params):
        """
        Train the root cause analyzer model.
        
        Args:
            model: Model instance
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            training_params: Training parameters
            
        Returns:
            Tuple of (trained_model, training_history)
        """
        # Build model architecture
        model.build_model(
            input_shape=X_train.shape[1:],
            num_classes=len(np.unique(y_train))
        )
        
        # Compile model
        model.compile_model(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train model
        history = model.train(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=training_params.get('epochs', 100),
            batch_size=training_params.get('batch_size', 32),
            callbacks=self._get_training_callbacks(training_params)
        )
        
        return model.model, history
    
    def _get_training_callbacks(self, training_params):
        """
        Get training callbacks based on configuration.
        
        Args:
            training_params: Training parameters
            
        Returns:
            List of Keras callbacks
        """
        callbacks = []
        
        # Early stopping
        early_stopping_config = training_params.get('early_stopping', {})
        if early_stopping_config:
            callbacks.append(tf.keras.callbacks.EarlyStopping(
                monitor=early_stopping_config.get('monitor', 'val_loss'),
                patience=early_stopping_config.get('patience', 10),
                restore_best_weights=early_stopping_config.get('restore_best_weights', True)
            ))
        
        # Model checkpoint
        callbacks.append(tf.keras.callbacks.ModelCheckpoint(
            filepath='best_model.h5',
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False
        ))
        
        # Reduce learning rate on plateau
        callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7
        ))
        
        return callbacks
    
    def _log_training_metrics(self, model_name: str, history, training_duration: float):
        """
        Log training metrics to MLflow.
        
        Args:
            model_name: Name of the model
            history: Training history
            training_duration: Training duration in seconds
        """
        # Log training duration
        mlflow.log_metric(f"{model_name}_training_duration", training_duration)
        
        # Log training history metrics
        if hasattr(history, 'history'):
            for epoch, (loss, val_loss) in enumerate(zip(history.history['loss'], history.history['val_loss'])):
                mlflow.log_metric(f"{model_name}_train_loss", loss, step=epoch)
                mlflow.log_metric(f"{model_name}_val_loss", val_loss, step=epoch)
            
            if 'accuracy' in history.history:
                for epoch, (acc, val_acc) in enumerate(zip(history.history['accuracy'], history.history['val_accuracy'])):
                    mlflow.log_metric(f"{model_name}_train_accuracy", acc, step=epoch)
                    mlflow.log_metric(f"{model_name}_val_accuracy", val_acc, step=epoch)
        
        # Log final metrics
        if hasattr(history, 'history'):
            final_train_loss = history.history['loss'][-1]
            final_val_loss = history.history['val_loss'][-1]
            mlflow.log_metric(f"{model_name}_final_train_loss", final_train_loss)
            mlflow.log_metric(f"{model_name}_final_val_loss", final_val_loss)
            
            if 'accuracy' in history.history:
                final_train_acc = history.history['accuracy'][-1]
                final_val_acc = history.history['val_accuracy'][-1]
                mlflow.log_metric(f"{model_name}_final_train_accuracy", final_train_acc)
                mlflow.log_metric(f"{model_name}_final_val_accuracy", final_val_acc)
    
    def _execute_feature_engineering_stage(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute feature engineering stage with defect detection specific features.
        
        Args:
            config: Stage configuration
            
        Returns:
            Stage execution results
        """
        logger.info("Executing defect detection feature engineering stage")
        
        # Get preprocessed data
        preprocessed_data = self.stage_results.get('data_preprocessing', {}).get('preprocessed_data')
        
        if preprocessed_data is None:
            raise ValueError("No preprocessed data available for feature engineering")
        
        # Apply sensor-specific feature engineering
        engineered_features = preprocessed_data.copy()
        
        # Pyrometer features
        if 'pyrometer' in preprocessed_data.columns:
            pyrometer_data = preprocessed_data[['pyrometer']]
            pyrometer_features = self.feature_engineers['pyrometer'].extract_all_features(pyrometer_data)
            engineered_features = pd.concat([engineered_features, pyrometer_features], axis=1)
        
        # Camera features
        if 'camera' in preprocessed_data.columns:
            camera_data = preprocessed_data[['camera']]
            camera_features = self.feature_engineers['camera'].extract_all_features(camera_data)
            engineered_features = pd.concat([engineered_features, camera_features], axis=1)
        
        # Accelerometer features
        if 'accelerometer' in preprocessed_data.columns:
            accel_data = preprocessed_data[['accelerometer']]
            accel_features = self.feature_engineers['accelerometer'].extract_all_features(accel_data)
            engineered_features = pd.concat([engineered_features, accel_features], axis=1)
        
        # Temperature features
        if 'temperature' in preprocessed_data.columns:
            temp_data = preprocessed_data[['temperature']]
            temp_features = self.feature_engineers['temperature'].extract_all_features(temp_data)
            engineered_features = pd.concat([engineered_features, temp_features], axis=1)
        
        # Temporal features
        temporal_features = self.feature_engineers['time_series'].extract_all_features(engineered_features)
        engineered_features = pd.concat([engineered_features, temporal_features], axis=1)
        
        # Lag features
        lag_features = self.feature_engineers['lag'].extract_all_features(engineered_features)
        engineered_features = pd.concat([engineered_features, lag_features], axis=1)
        
        # Rolling features
        rolling_features = self.feature_engineers['rolling'].extract_all_features(engineered_features)
        engineered_features = pd.concat([engineered_features, rolling_features], axis=1)
        
        # Frequency features
        frequency_features = self.feature_engineers['frequency'].extract_all_features(engineered_features)
        engineered_features = pd.concat([engineered_features, frequency_features], axis=1)
        
        # Apply transformations from config
        transformations = config.get('transformations', [])
        for transformation in transformations:
            transformation_type = transformation.get('type', 'unknown')
            
            if transformation_type == "temporal_features":
                engineered_features = self._apply_temporal_features(engineered_features, transformation)
            elif transformation_type == "statistical_features":
                engineered_features = self._apply_statistical_features(engineered_features, transformation)
            elif transformation_type == "frequency_features":
                engineered_features = self._apply_frequency_features(engineered_features, transformation)
            elif transformation_type == "cross_features":
                engineered_features = self._apply_cross_features(engineered_features, transformation)
        
        # Log feature engineering metrics
        mlflow.log_metric("total_features", len(engineered_features.columns))
        mlflow.log_metric("feature_engineering_duration", time.time() - time.time())
        
        return {
            'engineered_features': engineered_features,
            'transformations_applied': transformations,
            'feature_count': len(engineered_features.columns),
            'sensor_features_applied': ['pyrometer', 'camera', 'accelerometer', 'temperature'],
            'temporal_features_applied': ['time_series', 'lag', 'rolling', 'frequency']
        }
    
    def _apply_temporal_features(self, data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """
        Apply temporal feature transformations for defect detection.
        
        Args:
            data: Input data
            config: Transformation configuration
            
        Returns:
            Data with temporal features
        """
        lags = config.get('lags', [1, 2, 3, 6, 12, 24])
        windows = config.get('windows', [6, 12, 24])
        
        # Apply lag features
        for lag in lags:
            for col in data.select_dtypes(include=[np.number]).columns:
                data[f'{col}_lag_{lag}'] = data[col].shift(lag)
        
        # Apply rolling window features
        for window in windows:
            for col in data.select_dtypes(include=[np.number]).columns:
                data[f'{col}_rolling_mean_{window}'] = data[col].rolling(window=window).mean()
                data[f'{col}_rolling_std_{window}'] = data[col].rolling(window=window).std()
                data[f'{col}_rolling_max_{window}'] = data[col].rolling(window=window).max()
                data[f'{col}_rolling_min_{window}'] = data[col].rolling(window=window).min()
        
        return data
    
    def _apply_statistical_features(self, data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """
        Apply statistical feature transformations for defect detection.
        
        Args:
            data: Input data
            config: Transformation configuration
            
        Returns:
            Data with statistical features
        """
        aggregations = config.get('aggregations', ['mean', 'std', 'min', 'max', 'median'])
        windows = config.get('windows', [5, 10, 20])
        
        for window in windows:
            for col in data.select_dtypes(include=[np.number]).columns:
                for agg in aggregations:
                    if agg == 'mean':
                        data[f'{col}_rolling_mean_{window}'] = data[col].rolling(window=window).mean()
                    elif agg == 'std':
                        data[f'{col}_rolling_std_{window}'] = data[col].rolling(window=window).std()
                    elif agg == 'min':
                        data[f'{col}_rolling_min_{window}'] = data[col].rolling(window=window).min()
                    elif agg == 'max':
                        data[f'{col}_rolling_max_{window}'] = data[col].rolling(window=window).max()
                    elif agg == 'median':
                        data[f'{col}_rolling_median_{window}'] = data[col].rolling(window=window).median()
        
        return data
    
    def _apply_frequency_features(self, data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """
        Apply frequency feature transformations for defect detection.
        
        Args:
            data: Input data
            config: Transformation configuration
            
        Returns:
            Data with frequency features
        """
        methods = config.get('methods', ['fft', 'spectral_centroid', 'spectral_rolloff'])
        windows = config.get('windows', [10, 20])
        
        for window in windows:
            for col in data.select_dtypes(include=[np.number]).columns:
                # Apply FFT-based features
                if 'fft' in methods:
                    fft_features = self.feature_engineers['frequency'].extract_spectral_features(
                        data[[col]].rolling(window=window)
                    )
                    data = pd.concat([data, fft_features], axis=1)
        
        return data
    
    def _apply_cross_features(self, data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """
        Apply cross feature transformations for defect detection.
        
        Args:
            data: Input data
            config: Transformation configuration
            
        Returns:
            Data with cross features
        """
        interactions = config.get('interactions', ['temperature*laser_power', 'scan_speed*hatch_spacing'])
        
        for interaction in interactions:
            if '*' in interaction:
                col1, col2 = interaction.split('*')
                if col1 in data.columns and col2 in data.columns:
                    data[f'{col1}_{col2}_interaction'] = data[col1] * data[col2]
                    data[f'{col1}_{col2}_ratio'] = data[col1] / (data[col2] + 1e-6)
        
        return data
