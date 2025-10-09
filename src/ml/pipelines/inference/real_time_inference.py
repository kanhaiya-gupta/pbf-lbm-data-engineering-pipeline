"""
Real-time Inference Pipeline

This module implements the real-time inference pipeline for PBF-LB/M processes.
It handles real-time data ingestion, preprocessing, feature engineering, and model inference
for process optimization, defect detection, quality assessment, and maintenance predictions.
"""

import logging
import time
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime, timedelta
import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
import mlflow
import mlflow.tensorflow

from ...models.base_model import BaseModel
from ...features.process_features import LaserParameterFeatures, BuildParameterFeatures, MaterialFeatures
from ...features.sensor_features import PyrometerFeatures, CameraFeatures, AccelerometerFeatures, TemperatureFeatures
from ...features.image_features import CTScanFeatures, PowderBedFeatures, DefectImageFeatures, SurfaceTextureFeatures
from ...features.temporal_features import TimeSeriesFeatures, LagFeatures, RollingFeatures, FrequencyFeatures
from ...utils.preprocessing import DataCleaner, FeatureScaler, OutlierDetector, DataValidator
from ...utils.evaluation import RegressionMetrics, ClassificationMetrics, TimeSeriesMetrics, CustomMetrics

logger = logging.getLogger(__name__)


class RealTimeInferencePipeline:
    """
    Real-time inference pipeline for PBF-LB/M processes.
    
    This pipeline handles real-time data processing and model inference for:
    - Process optimization
    - Defect detection
    - Quality assessment
    - Predictive maintenance
    """
    
    def __init__(self, config_manager=None):
        """
        Initialize the real-time inference pipeline.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config_manager = config_manager
        self.pipeline_name = "real_time_inference"
        
        # Initialize feature engineers
        self.feature_engineers = {
            'laser_parameter': LaserParameterFeatures(self.config_manager),
            'build_parameter': BuildParameterFeatures(self.config_manager),
            'material': MaterialFeatures(self.config_manager),
            'pyrometer': PyrometerFeatures(self.config_manager),
            'camera': CameraFeatures(self.config_manager),
            'accelerometer': AccelerometerFeatures(self.config_manager),
            'temperature': TemperatureFeatures(self.config_manager),
            'ct_scan': CTScanFeatures(self.config_manager),
            'powder_bed': PowderBedFeatures(self.config_manager),
            'defect_image': DefectImageFeatures(self.config_manager),
            'surface_texture': SurfaceTextureFeatures(self.config_manager),
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
        
        # Inference metrics
        self.inference_metrics = {
            'total_inferences': 0,
            'successful_inferences': 0,
            'failed_inferences': 0,
            'average_latency': 0.0,
            'last_inference_time': None
        }
        
        logger.info("Initialized RealTimeInferencePipeline")
    
    async def process_real_time_data(self, data: Dict[str, Any], model_configs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process real-time data and generate predictions.
        
        Args:
            data: Real-time input data
            model_configs: List of model configurations to use
            
        Returns:
            Dictionary containing predictions and metadata
        """
        start_time = time.time()
        
        try:
            # Validate input data
            validated_data = await self._validate_input_data(data)
            
            # Preprocess data
            preprocessed_data = await self._preprocess_data(validated_data)
            
            # Extract features
            features = await self._extract_features(preprocessed_data)
            
            # Generate predictions
            predictions = await self._generate_predictions(features, model_configs)
            
            # Post-process predictions
            processed_predictions = await self._post_process_predictions(predictions)
            
            # Update metrics
            self._update_inference_metrics(start_time, True)
            
            return {
                'predictions': processed_predictions,
                'metadata': {
                    'inference_time': time.time() - start_time,
                    'timestamp': datetime.now().isoformat(),
                    'model_configs_used': model_configs,
                    'feature_count': len(features.columns) if hasattr(features, 'columns') else 0
                },
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"Real-time inference failed: {e}")
            self._update_inference_metrics(start_time, False)
            
            return {
                'predictions': {},
                'metadata': {
                    'inference_time': time.time() - start_time,
                    'timestamp': datetime.now().isoformat(),
                    'error': str(e)
                },
                'status': 'error'
            }
    
    async def _validate_input_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate input data for real-time inference.
        
        Args:
            data: Input data dictionary
            
        Returns:
            Validated data dictionary
        """
        # Check required fields
        required_fields = ['timestamp', 'sensor_data', 'process_data']
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Missing required field: {field}")
        
        # Validate timestamp
        if not isinstance(data['timestamp'], (str, datetime)):
            raise ValueError("Invalid timestamp format")
        
        # Validate sensor data
        if not isinstance(data['sensor_data'], dict):
            raise ValueError("Sensor data must be a dictionary")
        
        # Validate process data
        if not isinstance(data['process_data'], dict):
            raise ValueError("Process data must be a dictionary")
        
        return data
    
    async def _preprocess_data(self, data: Dict[str, Any]) -> pd.DataFrame:
        """
        Preprocess real-time data for inference.
        
        Args:
            data: Validated input data
            
        Returns:
            Preprocessed DataFrame
        """
        # Combine sensor and process data
        combined_data = {**data['sensor_data'], **data['process_data']}
        
        # Create DataFrame
        df = pd.DataFrame([combined_data])
        
        # Add timestamp
        df['timestamp'] = pd.to_datetime(data['timestamp'])
        
        # Clean data
        cleaned_data = self.preprocessors['data_cleaner'].clean_data(df)
        
        # Detect and handle outliers
        outlier_info = self.preprocessors['outlier_detector'].detect_outliers(cleaned_data)
        if outlier_info['outlier_count'] > 0:
            logger.warning(f"Detected {outlier_info['outlier_count']} outliers in real-time data")
            # Handle outliers (e.g., cap values)
            cleaned_data = self.preprocessors['outlier_detector'].handle_outliers(cleaned_data)
        
        # Validate data quality
        validation_results = self.preprocessors['data_validator'].validate_data(cleaned_data)
        if not validation_results['is_valid']:
            logger.warning(f"Data validation failed: {validation_results['errors']}")
        
        return cleaned_data
    
    async def _extract_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features from preprocessed data.
        
        Args:
            data: Preprocessed DataFrame
            
        Returns:
            DataFrame with extracted features
        """
        features = data.copy()
        
        # Extract process features
        if any(col in data.columns for col in ['laser_power', 'scan_speed', 'layer_height']):
            process_data = data[['laser_power', 'scan_speed', 'layer_height']].fillna(0)
            process_features = self.feature_engineers['laser_parameter'].extract_all_features(process_data)
            features = pd.concat([features, process_features], axis=1)
        
        # Extract sensor features
        if any(col in data.columns for col in ['temperature', 'vibration', 'pressure']):
            sensor_data = data[['temperature', 'vibration', 'pressure']].fillna(0)
            sensor_features = self.feature_engineers['pyrometer'].extract_all_features(sensor_data)
            features = pd.concat([features, sensor_features], axis=1)
        
        # Extract temporal features
        if 'timestamp' in data.columns:
            time_data = data[['timestamp']].fillna(0)
            time_features = self.feature_engineers['time_series'].extract_all_features(time_data)
            features = pd.concat([features, time_features], axis=1)
        
        # Extract lag features
        if any(col in data.columns for col in ['sensor_value', 'measurement']):
            lag_data = data[['sensor_value', 'measurement']].fillna(0)
            lag_features = self.feature_engineers['lag'].extract_all_features(lag_data)
            features = pd.concat([features, lag_features], axis=1)
        
        # Extract rolling features
        if any(col in data.columns for col in ['rolling_mean', 'rolling_std']):
            rolling_data = data[['rolling_mean', 'rolling_std']].fillna(0)
            rolling_features = self.feature_engineers['rolling'].extract_all_features(rolling_data)
            features = pd.concat([features, rolling_features], axis=1)
        
        # Extract frequency features
        if any(col in data.columns for col in ['frequency', 'spectral']):
            freq_data = data[['frequency', 'spectral']].fillna(0)
            freq_features = self.feature_engineers['frequency'].extract_all_features(freq_data)
            features = pd.concat([features, freq_features], axis=1)
        
        return features
    
    async def _generate_predictions(self, features: pd.DataFrame, model_configs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate predictions using specified models.
        
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
                future = executor.submit(self._predict_with_model, features, model_config)
                futures.append((model_config['model_name'], future))
            
            # Collect results
            for model_name, future in futures:
                try:
                    result = future.result(timeout=5.0)  # 5 second timeout
                    predictions[model_name] = result
                except Exception as e:
                    logger.error(f"Prediction failed for {model_name}: {e}")
                    predictions[model_name] = {'error': str(e)}
        
        return predictions
    
    def _predict_with_model(self, features: pd.DataFrame, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate prediction with a specific model.
        
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
            logger.error(f"Model prediction failed for {model_name}: {e}")
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
    
    async def _post_process_predictions(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Post-process predictions for output.
        
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
    
    def _update_inference_metrics(self, start_time: float, success: bool):
        """
        Update inference metrics.
        
        Args:
            start_time: Inference start time
            success: Whether inference was successful
        """
        inference_time = time.time() - start_time
        
        self.inference_metrics['total_inferences'] += 1
        if success:
            self.inference_metrics['successful_inferences'] += 1
        else:
            self.inference_metrics['failed_inferences'] += 1
        
        # Update average latency
        total_successful = self.inference_metrics['successful_inferences']
        if total_successful > 0:
            current_avg = self.inference_metrics['average_latency']
            self.inference_metrics['average_latency'] = (
                (current_avg * (total_successful - 1) + inference_time) / total_successful
            )
        
        self.inference_metrics['last_inference_time'] = datetime.now().isoformat()
    
    def get_inference_metrics(self) -> Dict[str, Any]:
        """
        Get current inference metrics.
        
        Returns:
            Dictionary containing inference metrics
        """
        return self.inference_metrics.copy()
    
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
