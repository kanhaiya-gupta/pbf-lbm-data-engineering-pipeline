"""
Streaming Inference Pipeline

This module implements the streaming inference pipeline for PBF-LB/M processes.
It handles continuous data streams, real-time feature engineering, and model inference
for live monitoring and control systems.
"""

import logging
import time
from typing import Dict, Any, Optional, List, Tuple, AsyncGenerator
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime, timedelta
import asyncio
import json
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import mlflow
import mlflow.tensorflow

from ...models.base_model import BaseModel
from ...features.process_features import LaserParameterFeatures, BuildParameterFeatures, MaterialFeatures
from ...features.sensor_features import PyrometerFeatures, CameraFeatures, AccelerometerFeatures, TemperatureFeatures
from ...features.temporal_features import TimeSeriesFeatures, LagFeatures, RollingFeatures, FrequencyFeatures
from ...utils.preprocessing import DataCleaner, FeatureScaler, OutlierDetector, DataValidator
from ...utils.evaluation import RegressionMetrics, ClassificationMetrics, TimeSeriesMetrics, CustomMetrics

logger = logging.getLogger(__name__)


class StreamingInferencePipeline:
    """
    Streaming inference pipeline for PBF-LB/M processes.
    
    This pipeline handles continuous data streams and real-time inference for:
    - Live process monitoring
    - Real-time control systems
    - Continuous quality assessment
    - Stream-based maintenance predictions
    """
    
    def __init__(self, config_manager=None, window_size: int = 100, buffer_size: int = 1000):
        """
        Initialize the streaming inference pipeline.
        
        Args:
            config_manager: Configuration manager instance
            window_size: Size of sliding window for feature extraction
            buffer_size: Size of data buffer for streaming
        """
        self.config_manager = config_manager
        self.pipeline_name = "streaming_inference"
        self.window_size = window_size
        self.buffer_size = buffer_size
        
        # Initialize feature engineers
        self.feature_engineers = {
            'laser_parameter': LaserParameterFeatures(self.config_manager),
            'build_parameter': BuildParameterFeatures(self.config_manager),
            'material': MaterialFeatures(self.config_manager),
            'pyrometer': PyrometerFeatures(self.config_manager),
            'camera': CameraFeatures(self.config_manager),
            'accelerometer': AccelerometerFeatures(self.config_manager),
            'temperature': TemperatureFeatures(self.config_manager),
            'time_series': TimeSeriesFeatures(self.config_manager),
            'lag': LagFeatures(self.config_manager),
            'rolling': RollingFeatures(self.config_manager),
            'frequency': FrequencyFeatures(self.config_manager)
        }
        
        # Initialize preprocessing components
        self.preprocessors = {
            'data_cleaner': DataCleaner(),
            'feature_scaler': FeatureScaler(),
            'outlier_detector': OutlierDetector(),
            'data_validator': DataValidator()
        }
        
        # Initialize evaluators
        self.evaluators = {
            'regression': RegressionMetrics(),
            'classification': ClassificationMetrics(),
            'time_series': TimeSeriesMetrics(),
            'custom': CustomMetrics()
        }
        
        # Model cache for loaded models
        self.model_cache = {}
        
        # Streaming data buffer
        self.data_buffer = deque(maxlen=buffer_size)
        self.feature_buffer = deque(maxlen=buffer_size)
        self.prediction_buffer = deque(maxlen=buffer_size)
        
        # Streaming metrics
        self.streaming_metrics = {
            'total_streams_processed': 0,
            'total_records_processed': 0,
            'successful_streams': 0,
            'failed_streams': 0,
            'average_stream_latency': 0.0,
            'last_stream_time': None,
            'buffer_utilization': 0.0
        }
        
        # Stream processing state
        self.is_streaming = False
        self.stream_tasks = []
        
        logger.info("Initialized StreamingInferencePipeline")
    
    async def start_streaming(self, 
                             data_stream: AsyncGenerator[Dict[str, Any], None], 
                             model_configs: List[Dict[str, Any]],
                             output_stream: Optional[AsyncGenerator[Dict[str, Any], None]] = None) -> None:
        """
        Start streaming inference processing.
        
        Args:
            data_stream: Async generator for input data
            model_configs: List of model configurations to use
            output_stream: Optional async generator for output data
        """
        self.is_streaming = True
        logger.info("Starting streaming inference")
        
        try:
            async for data_point in data_stream:
                if not self.is_streaming:
                    break
                
                # Process single data point
                result = await self.process_stream_data_point(data_point, model_configs)
                
                # Add to buffer
                self.data_buffer.append(data_point)
                if 'features' in result:
                    self.feature_buffer.append(result['features'])
                if 'predictions' in result:
                    self.prediction_buffer.append(result['predictions'])
                
                # Send to output stream if provided
                if output_stream:
                    await output_stream.asend(result)
                
                # Update metrics
                self._update_streaming_metrics(True, 1)
                
        except Exception as e:
            logger.error(f"Streaming inference failed: {e}")
            self._update_streaming_metrics(False, 0)
            raise
        
        finally:
            self.is_streaming = False
            logger.info("Streaming inference stopped")
    
    async def process_stream_data_point(self, 
                                       data_point: Dict[str, Any], 
                                       model_configs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process a single data point from the stream.
        
        Args:
            data_point: Single data point from stream
            model_configs: List of model configurations
            
        Returns:
            Processing results for the data point
        """
        start_time = time.time()
        
        try:
            # Validate data point
            validated_data = await self._validate_stream_data(data_point)
            
            # Preprocess data point
            preprocessed_data = await self._preprocess_stream_data(validated_data)
            
            # Extract features with sliding window
            features = await self._extract_stream_features(preprocessed_data)
            
            # Generate predictions
            predictions = await self._generate_stream_predictions(features, model_configs)
            
            # Post-process predictions
            processed_predictions = await self._post_process_stream_predictions(predictions)
            
            return {
                'data_point': validated_data,
                'features': features,
                'predictions': processed_predictions,
                'metadata': {
                    'processing_time': time.time() - start_time,
                    'timestamp': datetime.now().isoformat(),
                    'window_size': self.window_size,
                    'buffer_size': len(self.data_buffer)
                },
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"Stream data point processing failed: {e}")
            return {
                'data_point': data_point,
                'error': str(e),
                'metadata': {
                    'processing_time': time.time() - start_time,
                    'timestamp': datetime.now().isoformat()
                },
                'status': 'error'
            }
    
    async def _validate_stream_data(self, data_point: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate stream data point.
        
        Args:
            data_point: Input data point
            
        Returns:
            Validated data point
        """
        # Check required fields
        required_fields = ['timestamp', 'sensor_data']
        for field in required_fields:
            if field not in data_point:
                raise ValueError(f"Missing required field: {field}")
        
        # Validate timestamp
        if not isinstance(data_point['timestamp'], (str, datetime)):
            raise ValueError("Invalid timestamp format")
        
        # Validate sensor data
        if not isinstance(data_point['sensor_data'], dict):
            raise ValueError("Sensor data must be a dictionary")
        
        return data_point
    
    async def _preprocess_stream_data(self, data_point: Dict[str, Any]) -> pd.DataFrame:
        """
        Preprocess stream data point.
        
        Args:
            data_point: Validated data point
            
        Returns:
            Preprocessed DataFrame
        """
        # Create DataFrame from data point
        df = pd.DataFrame([data_point['sensor_data']])
        df['timestamp'] = pd.to_datetime(data_point['timestamp'])
        
        # Clean data
        cleaned_data = self.preprocessors['data_cleaner'].clean_data(df)
        
        # Detect outliers
        outlier_info = self.preprocessors['outlier_detector'].detect_outliers(cleaned_data)
        if outlier_info['outlier_count'] > 0:
            logger.warning(f"Detected {outlier_info['outlier_count']} outliers in stream data")
            cleaned_data = self.preprocessors['outlier_detector'].handle_outliers(cleaned_data)
        
        # Validate data quality
        validation_results = self.preprocessors['data_validator'].validate_data(cleaned_data)
        if not validation_results['is_valid']:
            logger.warning(f"Stream data validation failed: {validation_results['errors']}")
        
        return cleaned_data
    
    async def _extract_stream_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features from stream data with sliding window.
        
        Args:
            data: Preprocessed stream data
            
        Returns:
            DataFrame with extracted features
        """
        features = data.copy()
        
        # Add to sliding window buffer
        if len(self.data_buffer) >= self.window_size:
            # Create sliding window DataFrame
            window_data = pd.concat(list(self.data_buffer)[-self.window_size:], ignore_index=True)
            
            # Extract temporal features from window
            if 'timestamp' in window_data.columns:
                time_features = self.feature_engineers['time_series'].extract_all_features(window_data[['timestamp']])
                features = pd.concat([features, time_features], axis=1)
            
            # Extract lag features from window
            if any(col in window_data.columns for col in ['sensor_value', 'measurement']):
                lag_data = window_data[['sensor_value', 'measurement']].fillna(0)
                lag_features = self.feature_engineers['lag'].extract_all_features(lag_data)
                features = pd.concat([features, lag_features], axis=1)
            
            # Extract rolling features from window
            if any(col in window_data.columns for col in ['rolling_mean', 'rolling_std']):
                rolling_data = window_data[['rolling_mean', 'rolling_std']].fillna(0)
                rolling_features = self.feature_engineers['rolling'].extract_all_features(rolling_data)
                features = pd.concat([features, rolling_features], axis=1)
            
            # Extract frequency features from window
            if any(col in window_data.columns for col in ['frequency', 'spectral']):
                freq_data = window_data[['frequency', 'spectral']].fillna(0)
                freq_features = self.feature_engineers['frequency'].extract_all_features(freq_data)
                features = pd.concat([features, freq_features], axis=1)
        
        # Extract current point features
        if any(col in data.columns for col in ['laser_power', 'scan_speed', 'layer_height']):
            process_data = data[['laser_power', 'scan_speed', 'layer_height']].fillna(0)
            process_features = self.feature_engineers['laser_parameter'].extract_all_features(process_data)
            features = pd.concat([features, process_features], axis=1)
        
        if any(col in data.columns for col in ['temperature', 'vibration', 'pressure']):
            sensor_data = data[['temperature', 'vibration', 'pressure']].fillna(0)
            sensor_features = self.feature_engineers['pyrometer'].extract_all_features(sensor_data)
            features = pd.concat([features, sensor_features], axis=1)
        
        return features
    
    async def _generate_stream_predictions(self, 
                                          features: pd.DataFrame, 
                                          model_configs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate predictions for stream data.
        
        Args:
            features: Feature DataFrame
            model_configs: List of model configurations
            
        Returns:
            Dictionary containing predictions from all models
        """
        predictions = {}
        
        # Process models in parallel
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for model_config in model_configs:
                future = executor.submit(self._predict_stream_with_model, features, model_config)
                futures.append((model_config['model_name'], future))
            
            # Collect results
            for model_name, future in futures:
                try:
                    result = future.result(timeout=2.0)  # 2 second timeout for streaming
                    predictions[model_name] = result
                except Exception as e:
                    logger.error(f"Stream prediction failed for {model_name}: {e}")
                    predictions[model_name] = {'error': str(e)}
        
        return predictions
    
    def _predict_stream_with_model(self, features: pd.DataFrame, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate stream prediction with a specific model.
        
        Args:
            features: Feature DataFrame
            model_config: Model configuration
            
        Returns:
            Model prediction results
        """
        model_name = model_config['model_name']
        model_version = model_config.get('version', 'latest')
        
        try:
            # Load model if not cached
            if model_name not in self.model_cache:
                self.model_cache[model_name] = self._load_model(model_name, model_version)
            
            model = self.model_cache[model_name]
            
            # Prepare features for model
            X = features.select_dtypes(include=[np.number]).values
            
            # Generate prediction
            if hasattr(model, 'predict'):
                prediction = model.predict(X)
            elif hasattr(model, 'predict_proba'):
                prediction = model.predict_proba(X)
            else:
                raise ValueError(f"Model {model_name} does not support prediction")
            
            # Calculate confidence/uncertainty if available
            confidence = None
            if hasattr(model, 'predict_proba'):
                confidence = np.max(model.predict_proba(X), axis=1)
            elif hasattr(model, 'decision_function'):
                confidence = np.abs(model.decision_function(X))
            
            return {
                'prediction': prediction.tolist() if hasattr(prediction, 'tolist') else prediction,
                'confidence': confidence.tolist() if confidence is not None and hasattr(confidence, 'tolist') else confidence,
                'model_name': model_name,
                'model_version': model_version,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Stream model prediction failed for {model_name}: {e}")
            return {'error': str(e)}
    
    def _load_model(self, model_name: str, version: str) -> Any:
        """
        Load a model from MLflow or local storage.
        
        Args:
            model_name: Name of the model
            version: Model version
            
        Returns:
            Loaded model
        """
        try:
            # Try to load from MLflow
            if version == 'latest':
                model = mlflow.tensorflow.load_model(f"models:/{model_name}/latest")
            else:
                model = mlflow.tensorflow.load_model(f"models:/{model_name}/{version}")
            
            logger.info(f"Loaded model {model_name} version {version} from MLflow")
            return model
            
        except Exception as e:
            logger.warning(f"Failed to load model from MLflow: {e}")
            # Fallback to local model loading
            return self._load_local_model(model_name, version)
    
    def _load_local_model(self, model_name: str, version: str) -> Any:
        """
        Load a model from local storage.
        
        Args:
            model_name: Name of the model
            version: Model version
            
        Returns:
            Loaded model
        """
        # This would implement local model loading logic
        # For now, return a mock model
        class MockModel:
            def predict(self, X):
                return np.random.random(X.shape[0])
            
            def predict_proba(self, X):
                prob = np.random.random(X.shape[0])
                return np.column_stack([1-prob, prob])
        
        return MockModel()
    
    async def _post_process_stream_predictions(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Post-process stream predictions.
        
        Args:
            predictions: Raw predictions from models
            
        Returns:
            Processed predictions
        """
        processed_predictions = {}
        
        for model_name, prediction in predictions.items():
            if 'error' in prediction:
                processed_predictions[model_name] = prediction
                continue
            
            # Apply post-processing based on model type
            if 'confidence' in prediction and prediction['confidence'] is not None:
                # Add confidence interpretation
                confidence = prediction['confidence']
                if isinstance(confidence, list) and len(confidence) > 0:
                    avg_confidence = np.mean(confidence)
                    prediction['confidence_level'] = self._interpret_confidence(avg_confidence)
            
            # Add prediction interpretation
            if 'prediction' in prediction:
                prediction['interpretation'] = self._interpret_prediction(model_name, prediction['prediction'])
            
            # Add stream-specific metadata
            prediction['stream_metadata'] = {
                'window_size': self.window_size,
                'buffer_utilization': len(self.data_buffer) / self.buffer_size,
                'processing_timestamp': datetime.now().isoformat()
            }
            
            processed_predictions[model_name] = prediction
        
        return processed_predictions
    
    def _interpret_confidence(self, confidence: float) -> str:
        """
        Interpret confidence score.
        
        Args:
            confidence: Confidence score (0-1)
            
        Returns:
            Confidence level description
        """
        if confidence >= 0.9:
            return "Very High"
        elif confidence >= 0.8:
            return "High"
        elif confidence >= 0.7:
            return "Medium"
        elif confidence >= 0.6:
            return "Low"
        else:
            return "Very Low"
    
    def _interpret_prediction(self, model_name: str, prediction: Any) -> str:
        """
        Interpret prediction based on model type.
        
        Args:
            model_name: Name of the model
            prediction: Model prediction
            
        Returns:
            Prediction interpretation
        """
        if 'defect' in model_name.lower():
            if isinstance(prediction, list) and len(prediction) > 0:
                if prediction[0] > 0.5:
                    return "Defect detected"
                else:
                    return "No defect detected"
        
        elif 'quality' in model_name.lower():
            if isinstance(prediction, list) and len(prediction) > 0:
                score = prediction[0]
                if score >= 0.8:
                    return "High quality"
                elif score >= 0.6:
                    return "Medium quality"
                else:
                    return "Low quality"
        
        elif 'maintenance' in model_name.lower():
            if isinstance(prediction, list) and len(prediction) > 0:
                if prediction[0] > 0.7:
                    return "Maintenance required"
                else:
                    return "No maintenance needed"
        
        return "Prediction generated"
    
    def _update_streaming_metrics(self, success: bool, record_count: int):
        """
        Update streaming metrics.
        
        Args:
            success: Whether processing was successful
            record_count: Number of records processed
        """
        self.streaming_metrics['total_streams_processed'] += 1
        if success:
            self.streaming_metrics['successful_streams'] += 1
        else:
            self.streaming_metrics['failed_streams'] += 1
        
        self.streaming_metrics['total_records_processed'] += record_count
        self.streaming_metrics['buffer_utilization'] = len(self.data_buffer) / self.buffer_size
        self.streaming_metrics['last_stream_time'] = datetime.now().isoformat()
    
    def stop_streaming(self):
        """
        Stop streaming inference.
        """
        self.is_streaming = False
        logger.info("Streaming inference stop requested")
    
    def get_streaming_metrics(self) -> Dict[str, Any]:
        """
        Get current streaming metrics.
        
        Returns:
            Dictionary containing streaming metrics
        """
        return self.streaming_metrics.copy()
    
    def get_buffer_status(self) -> Dict[str, Any]:
        """
        Get current buffer status.
        
        Returns:
            Dictionary containing buffer status
        """
        return {
            'data_buffer_size': len(self.data_buffer),
            'feature_buffer_size': len(self.feature_buffer),
            'prediction_buffer_size': len(self.prediction_buffer),
            'buffer_utilization': len(self.data_buffer) / self.buffer_size,
            'window_size': self.window_size,
            'max_buffer_size': self.buffer_size
        }
    
    def clear_buffers(self):
        """
        Clear all buffers.
        """
        self.data_buffer.clear()
        self.feature_buffer.clear()
        self.prediction_buffer.clear()
        logger.info("All buffers cleared")
    
    def clear_model_cache(self):
        """
        Clear the model cache.
        """
        self.model_cache.clear()
        logger.info("Model cache cleared")
    
    def get_cached_models(self) -> List[str]:
        """
        Get list of cached model names.
        
        Returns:
            List of cached model names
        """
        return list(self.model_cache.keys())
