"""
Batch Inference Pipeline

This module implements the batch inference pipeline for PBF-LB/M processes.
It handles batch data processing, model inference, and result aggregation
for large-scale analysis and reporting.
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
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import mlflow
import mlflow.tensorflow
from pathlib import Path
import pickle

from ...models.base_model import BaseModel
from ...features.process_features import LaserParameterFeatures, BuildParameterFeatures, MaterialFeatures
from ...features.sensor_features import PyrometerFeatures, CameraFeatures, AccelerometerFeatures, TemperatureFeatures
from ...features.image_features import CTScanFeatures, PowderBedFeatures, DefectImageFeatures, SurfaceTextureFeatures
from ...features.temporal_features import TimeSeriesFeatures, LagFeatures, RollingFeatures, FrequencyFeatures
from ...utils.preprocessing import DataCleaner, FeatureScaler, OutlierDetector, DataValidator
from ...utils.evaluation import RegressionMetrics, ClassificationMetrics, TimeSeriesMetrics, CustomMetrics

logger = logging.getLogger(__name__)


class BatchInferencePipeline:
    """
    Batch inference pipeline for PBF-LB/M processes.
    
    This pipeline handles batch data processing and model inference for:
    - Historical data analysis
    - Large-scale predictions
    - Batch reporting
    - Model evaluation
    """
    
    def __init__(self, config_manager=None):
        """
        Initialize the batch inference pipeline.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config_manager = config_manager
        self.pipeline_name = "batch_inference"
        
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
        
        # Batch processing metrics
        self.batch_metrics = {
            'total_batches_processed': 0,
            'total_records_processed': 0,
            'successful_batches': 0,
            'failed_batches': 0,
            'average_batch_processing_time': 0.0,
            'last_batch_time': None
        }
        
        logger.info("Initialized BatchInferencePipeline")
    
    async def process_batch_data(self, 
                                data_source: str, 
                                model_configs: List[Dict[str, Any]], 
                                batch_size: int = 1000,
                                output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Process batch data and generate predictions.
        
        Args:
            data_source: Path to data source or data dictionary
            model_configs: List of model configurations to use
            batch_size: Size of each batch for processing
            output_path: Optional path to save results
            
        Returns:
            Dictionary containing batch processing results
        """
        start_time = time.time()
        
        try:
            # Load batch data
            batch_data = await self._load_batch_data(data_source)
            
            # Validate batch data
            validated_data = await self._validate_batch_data(batch_data)
            
            # Process data in batches
            batch_results = await self._process_data_batches(validated_data, model_configs, batch_size)
            
            # Aggregate results
            aggregated_results = await self._aggregate_batch_results(batch_results)
            
            # Save results if output path specified
            if output_path:
                await self._save_batch_results(aggregated_results, output_path)
            
            # Update metrics
            self._update_batch_metrics(start_time, True, len(validated_data))
            
            return {
                'results': aggregated_results,
                'metadata': {
                    'processing_time': time.time() - start_time,
                    'total_records': len(validated_data),
                    'batch_size': batch_size,
                    'model_configs_used': model_configs,
                    'output_path': output_path
                },
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"Batch inference failed: {e}")
            self._update_batch_metrics(start_time, False, 0)
            
            return {
                'results': {},
                'metadata': {
                    'processing_time': time.time() - start_time,
                    'error': str(e)
                },
                'status': 'error'
            }
    
    async def _load_batch_data(self, data_source: str) -> pd.DataFrame:
        """
        Load batch data from various sources.
        
        Args:
            data_source: Path to data source or data dictionary
            
        Returns:
            Loaded DataFrame
        """
        if isinstance(data_source, str):
            # Load from file
            if data_source.endswith('.csv'):
                return pd.read_csv(data_source)
            elif data_source.endswith('.parquet'):
                return pd.read_parquet(data_source)
            elif data_source.endswith('.json'):
                return pd.read_json(data_source)
            elif data_source.endswith('.pkl'):
                with open(data_source, 'rb') as f:
                    return pickle.load(f)
            else:
                raise ValueError(f"Unsupported file format: {data_source}")
        
        elif isinstance(data_source, dict):
            # Load from dictionary
            return pd.DataFrame(data_source)
        
        elif isinstance(data_source, pd.DataFrame):
            # Already a DataFrame
            return data_source
        
        else:
            raise ValueError(f"Unsupported data source type: {type(data_source)}")
    
    async def _validate_batch_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Validate batch data for processing.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Validated DataFrame
        """
        # Check if data is empty
        if data.empty:
            raise ValueError("Batch data is empty")
        
        # Check required columns
        required_columns = ['timestamp']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            logger.warning(f"Missing required columns: {missing_columns}")
        
        # Validate timestamp column
        if 'timestamp' in data.columns:
            try:
                data['timestamp'] = pd.to_datetime(data['timestamp'])
            except Exception as e:
                logger.warning(f"Failed to parse timestamp column: {e}")
        
        # Check for duplicate records
        if data.duplicated().any():
            logger.warning(f"Found {data.duplicated().sum()} duplicate records")
            data = data.drop_duplicates()
        
        return data
    
    async def _process_data_batches(self, 
                                   data: pd.DataFrame, 
                                   model_configs: List[Dict[str, Any]], 
                                   batch_size: int) -> List[Dict[str, Any]]:
        """
        Process data in batches.
        
        Args:
            data: Input DataFrame
            model_configs: List of model configurations
            batch_size: Size of each batch
            
        Returns:
            List of batch results
        """
        batch_results = []
        total_batches = (len(data) + batch_size - 1) // batch_size
        
        logger.info(f"Processing {len(data)} records in {total_batches} batches of size {batch_size}")
        
        for i in range(0, len(data), batch_size):
            batch_data = data.iloc[i:i+batch_size]
            batch_num = i // batch_size + 1
            
            logger.info(f"Processing batch {batch_num}/{total_batches}")
            
            try:
                # Process single batch
                batch_result = await self._process_single_batch(batch_data, model_configs)
                batch_result['batch_number'] = batch_num
                batch_result['batch_size'] = len(batch_data)
                batch_results.append(batch_result)
                
            except Exception as e:
                logger.error(f"Failed to process batch {batch_num}: {e}")
                batch_results.append({
                    'batch_number': batch_num,
                    'batch_size': len(batch_data),
                    'error': str(e),
                    'status': 'failed'
                })
        
        return batch_results
    
    async def _process_single_batch(self, 
                                   batch_data: pd.DataFrame, 
                                   model_configs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process a single batch of data.
        
        Args:
            batch_data: Single batch DataFrame
            model_configs: List of model configurations
            
        Returns:
            Batch processing results
        """
        start_time = time.time()
        
        try:
            # Preprocess batch data
            preprocessed_data = await self._preprocess_batch_data(batch_data)
            
            # Extract features
            features = await self._extract_batch_features(preprocessed_data)
            
            # Generate predictions
            predictions = await self._generate_batch_predictions(features, model_configs)
            
            # Post-process predictions
            processed_predictions = await self._post_process_batch_predictions(predictions)
            
            return {
                'predictions': processed_predictions,
                'processing_time': time.time() - start_time,
                'record_count': len(batch_data),
                'feature_count': len(features.columns) if hasattr(features, 'columns') else 0,
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"Single batch processing failed: {e}")
            return {
                'error': str(e),
                'processing_time': time.time() - start_time,
                'record_count': len(batch_data),
                'status': 'failed'
            }
    
    async def _preprocess_batch_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess batch data.
        
        Args:
            data: Input batch DataFrame
            
        Returns:
            Preprocessed DataFrame
        """
        # Clean data
        cleaned_data = self.preprocessors['data_cleaner'].clean_data(data)
        
        # Detect and handle outliers
        outlier_info = self.preprocessors['outlier_detector'].detect_outliers(cleaned_data)
        if outlier_info['outlier_count'] > 0:
            logger.info(f"Detected {outlier_info['outlier_count']} outliers in batch data")
            cleaned_data = self.preprocessors['outlier_detector'].handle_outliers(cleaned_data)
        
        # Validate data quality
        validation_results = self.preprocessors['data_validator'].validate_data(cleaned_data)
        if not validation_results['is_valid']:
            logger.warning(f"Batch data validation failed: {validation_results['errors']}")
        
        return cleaned_data
    
    async def _extract_batch_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features from batch data.
        
        Args:
            data: Preprocessed batch DataFrame
            
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
    
    async def _generate_batch_predictions(self, 
                                         features: pd.DataFrame, 
                                         model_configs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate predictions for batch data.
        
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
                future = executor.submit(self._predict_batch_with_model, features, model_config)
                futures.append((model_config['model_name'], future))
            
            # Collect results
            for model_name, future in futures:
                try:
                    result = future.result(timeout=30.0)  # 30 second timeout for batch
                    predictions[model_name] = result
                except Exception as e:
                    logger.error(f"Batch prediction failed for {model_name}: {e}")
                    predictions[model_name] = {'error': str(e)}
        
        return predictions
    
    def _predict_batch_with_model(self, features: pd.DataFrame, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate batch predictions with a specific model.
        
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
            
            # Generate predictions
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
                'predictions': prediction.tolist() if hasattr(prediction, 'tolist') else prediction,
                'confidence': confidence.tolist() if confidence is not None and hasattr(confidence, 'tolist') else confidence,
                'model_name': model_name,
                'model_version': model_version,
                'record_count': len(X),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Batch model prediction failed for {model_name}: {e}")
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
    
    async def _post_process_batch_predictions(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Post-process batch predictions.
        
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
                # Add confidence statistics
                confidence = prediction['confidence']
                if isinstance(confidence, list) and len(confidence) > 0:
                    prediction['confidence_stats'] = {
                        'mean': np.mean(confidence),
                        'std': np.std(confidence),
                        'min': np.min(confidence),
                        'max': np.max(confidence)
                    }
            
            # Add prediction statistics
            if 'predictions' in prediction:
                pred_values = prediction['predictions']
                if isinstance(pred_values, list) and len(pred_values) > 0:
                    prediction['prediction_stats'] = {
                        'mean': np.mean(pred_values),
                        'std': np.std(pred_values),
                        'min': np.min(pred_values),
                        'max': np.max(pred_values)
                    }
            
            processed_predictions[model_name] = prediction
        
        return processed_predictions
    
    async def _aggregate_batch_results(self, batch_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate results from all batches.
        
        Args:
            batch_results: List of batch results
            
        Returns:
            Aggregated results
        """
        aggregated_results = {
            'total_batches': len(batch_results),
            'successful_batches': sum(1 for result in batch_results if result.get('status') == 'success'),
            'failed_batches': sum(1 for result in batch_results if result.get('status') == 'failed'),
            'total_records': sum(result.get('record_count', 0) for result in batch_results),
            'total_processing_time': sum(result.get('processing_time', 0) for result in batch_results),
            'model_results': {}
        }
        
        # Aggregate model results
        model_names = set()
        for result in batch_results:
            if 'predictions' in result:
                for model_name in result['predictions'].keys():
                    model_names.add(model_name)
        
        for model_name in model_names:
            model_predictions = []
            model_confidences = []
            
            for result in batch_results:
                if 'predictions' in result and model_name in result['predictions']:
                    pred_data = result['predictions'][model_name]
                    if 'predictions' in pred_data:
                        model_predictions.extend(pred_data['predictions'])
                    if 'confidence' in pred_data and pred_data['confidence'] is not None:
                        model_confidences.extend(pred_data['confidence'])
            
            aggregated_results['model_results'][model_name] = {
                'total_predictions': len(model_predictions),
                'predictions': model_predictions,
                'confidence': model_confidences if model_confidences else None
            }
        
        return aggregated_results
    
    async def _save_batch_results(self, results: Dict[str, Any], output_path: str):
        """
        Save batch results to file.
        
        Args:
            results: Batch results to save
            output_path: Path to save results
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if output_path.suffix == '.json':
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
        elif output_path.suffix == '.pkl':
            with open(output_path, 'wb') as f:
                pickle.dump(results, f)
        else:
            # Default to JSON
            with open(output_path.with_suffix('.json'), 'w') as f:
                json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Batch results saved to {output_path}")
    
    def _update_batch_metrics(self, start_time: float, success: bool, record_count: int):
        """
        Update batch processing metrics.
        
        Args:
            start_time: Processing start time
            success: Whether processing was successful
            record_count: Number of records processed
        """
        processing_time = time.time() - start_time
        
        self.batch_metrics['total_batches_processed'] += 1
        if success:
            self.batch_metrics['successful_batches'] += 1
        else:
            self.batch_metrics['failed_batches'] += 1
        
        self.batch_metrics['total_records_processed'] += record_count
        
        # Update average processing time
        total_successful = self.batch_metrics['successful_batches']
        if total_successful > 0:
            current_avg = self.batch_metrics['average_batch_processing_time']
            self.batch_metrics['average_batch_processing_time'] = (
                (current_avg * (total_successful - 1) + processing_time) / total_successful
            )
        
        self.batch_metrics['last_batch_time'] = datetime.now().isoformat()
    
    def get_batch_metrics(self) -> Dict[str, Any]:
        """
        Get current batch processing metrics.
        
        Returns:
            Dictionary containing batch metrics
        """
        return self.batch_metrics.copy()
    
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
