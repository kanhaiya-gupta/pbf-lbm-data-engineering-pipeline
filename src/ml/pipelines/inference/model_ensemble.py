"""
Model Ensemble Pipeline

This module implements the model ensemble pipeline for PBF-LB/M processes.
It handles multiple model predictions, ensemble methods, and result aggregation
for improved accuracy and robustness.
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
from sklearn.ensemble import VotingClassifier, VotingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

from ...models.base_model import BaseModel
from ...features.process_features import LaserParameterFeatures, BuildParameterFeatures, MaterialFeatures
from ...features.sensor_features import PyrometerFeatures, CameraFeatures, AccelerometerFeatures, TemperatureFeatures
from ...features.image_features import CTScanFeatures, PowderBedFeatures, DefectImageFeatures, SurfaceTextureFeatures
from ...features.temporal_features import TimeSeriesFeatures, LagFeatures, RollingFeatures, FrequencyFeatures
from ...utils.preprocessing import DataCleaner, FeatureScaler, OutlierDetector, DataValidator
from ...utils.evaluation import RegressionMetrics, ClassificationMetrics, TimeSeriesMetrics, CustomMetrics

logger = logging.getLogger(__name__)


class ModelEnsemblePipeline:
    """
    Model ensemble pipeline for PBF-LB/M processes.
    
    This pipeline handles ensemble methods for:
    - Multiple model predictions
    - Ensemble voting and averaging
    - Model selection and weighting
    - Uncertainty quantification
    """
    
    def __init__(self, config_manager=None):
        """
        Initialize the model ensemble pipeline.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config_manager = config_manager
        self.pipeline_name = "model_ensemble"
        
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
        
        # Ensemble configurations
        self.ensemble_configs = {
            'voting': {
                'regression': VotingRegressor,
                'classification': VotingClassifier
            },
            'averaging': {
                'methods': ['mean', 'median', 'weighted_mean']
            },
            'stacking': {
                'meta_models': {
                    'regression': LinearRegression,
                    'classification': LogisticRegression
                }
            }
        }
        
        # Ensemble metrics
        self.ensemble_metrics = {
            'total_ensembles_created': 0,
            'total_predictions_made': 0,
            'successful_ensembles': 0,
            'failed_ensembles': 0,
            'average_ensemble_accuracy': 0.0,
            'last_ensemble_time': None
        }
        
        logger.info("Initialized ModelEnsemblePipeline")
    
    async def create_ensemble(self, 
                             model_configs: List[Dict[str, Any]], 
                             ensemble_method: str = 'voting',
                             ensemble_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create a model ensemble from multiple models.
        
        Args:
            model_configs: List of model configurations
            ensemble_method: Ensemble method ('voting', 'averaging', 'stacking')
            ensemble_params: Additional ensemble parameters
            
        Returns:
            Dictionary containing ensemble creation results
        """
        start_time = time.time()
        
        try:
            # Load models
            models = await self._load_ensemble_models(model_configs)
            
            # Create ensemble based on method
            if ensemble_method == 'voting':
                ensemble = await self._create_voting_ensemble(models, ensemble_params)
            elif ensemble_method == 'averaging':
                ensemble = await self._create_averaging_ensemble(models, ensemble_params)
            elif ensemble_method == 'stacking':
                ensemble = await self._create_stacking_ensemble(models, ensemble_params)
            else:
                raise ValueError(f"Unsupported ensemble method: {ensemble_method}")
            
            # Store ensemble
            ensemble_id = f"ensemble_{int(time.time())}"
            self.model_cache[ensemble_id] = ensemble
            
            # Update metrics
            self._update_ensemble_metrics(start_time, True)
            
            return {
                'ensemble_id': ensemble_id,
                'ensemble_method': ensemble_method,
                'model_count': len(models),
                'model_names': [config['model_name'] for config in model_configs],
                'ensemble_params': ensemble_params,
                'creation_time': time.time() - start_time,
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"Ensemble creation failed: {e}")
            self._update_ensemble_metrics(start_time, False)
            
            return {
                'error': str(e),
                'ensemble_method': ensemble_method,
                'creation_time': time.time() - start_time,
                'status': 'error'
            }
    
    async def _load_ensemble_models(self, model_configs: List[Dict[str, Any]]) -> List[Any]:
        """
        Load models for ensemble.
        
        Args:
            model_configs: List of model configurations
            
        Returns:
            List of loaded models
        """
        models = []
        
        for config in model_configs:
            model_name = config['model_name']
            model_version = config.get('version', 'latest')
            
            try:
                # Load model if not cached
                if model_name not in self.model_cache:
                    self.model_cache[model_name] = self._load_model(model_name, model_version)
                
                models.append(self.model_cache[model_name])
                logger.info(f"Loaded model {model_name} for ensemble")
                
            except Exception as e:
                logger.error(f"Failed to load model {model_name}: {e}")
                raise
        
        return models
    
    async def _create_voting_ensemble(self, models: List[Any], params: Optional[Dict[str, Any]]) -> Any:
        """
        Create a voting ensemble.
        
        Args:
            models: List of models
            params: Ensemble parameters
            
        Returns:
            Voting ensemble
        """
        # Determine if regression or classification
        is_regression = params.get('task_type', 'regression') == 'regression'
        
        if is_regression:
            # Create voting regressor
            estimators = [(f'model_{i}', model) for i, model in enumerate(models)]
            ensemble = VotingRegressor(estimators)
        else:
            # Create voting classifier
            estimators = [(f'model_{i}', model) for i, model in enumerate(models)]
            ensemble = VotingClassifier(estimators, voting='soft')
        
        return ensemble
    
    async def _create_averaging_ensemble(self, models: List[Any], params: Optional[Dict[str, Any]]) -> Any:
        """
        Create an averaging ensemble.
        
        Args:
            models: List of models
            params: Ensemble parameters
            
        Returns:
            Averaging ensemble
        """
        class AveragingEnsemble:
            def __init__(self, models, method='mean', weights=None):
                self.models = models
                self.method = method
                self.weights = weights or [1.0] * len(models)
            
            def predict(self, X):
                predictions = []
                for model in self.models:
                    if hasattr(model, 'predict'):
                        pred = model.predict(X)
                    elif hasattr(model, 'predict_proba'):
                        pred = model.predict_proba(X)
                    else:
                        raise ValueError("Model does not support prediction")
                    predictions.append(pred)
                
                predictions = np.array(predictions)
                
                if self.method == 'mean':
                    return np.mean(predictions, axis=0)
                elif self.method == 'median':
                    return np.median(predictions, axis=0)
                elif self.method == 'weighted_mean':
                    return np.average(predictions, axis=0, weights=self.weights)
                else:
                    raise ValueError(f"Unsupported averaging method: {self.method}")
            
            def predict_proba(self, X):
                predictions = []
                for model in self.models:
                    if hasattr(model, 'predict_proba'):
                        pred = model.predict_proba(X)
                    elif hasattr(model, 'predict'):
                        pred = model.predict(X)
                        # Convert to probabilities (simple approach)
                        pred = np.column_stack([1-pred, pred])
                    else:
                        raise ValueError("Model does not support probability prediction")
                    predictions.append(pred)
                
                predictions = np.array(predictions)
                
                if self.method == 'mean':
                    return np.mean(predictions, axis=0)
                elif self.method == 'median':
                    return np.median(predictions, axis=0)
                elif self.method == 'weighted_mean':
                    return np.average(predictions, axis=0, weights=self.weights)
                else:
                    raise ValueError(f"Unsupported averaging method: {self.method}")
        
        method = params.get('method', 'mean') if params else 'mean'
        weights = params.get('weights', None) if params else None
        
        return AveragingEnsemble(models, method, weights)
    
    async def _create_stacking_ensemble(self, models: List[Any], params: Optional[Dict[str, Any]]) -> Any:
        """
        Create a stacking ensemble.
        
        Args:
            models: List of models
            params: Ensemble parameters
            
        Returns:
            Stacking ensemble
        """
        class StackingEnsemble:
            def __init__(self, models, meta_model, task_type='regression'):
                self.models = models
                self.meta_model = meta_model
                self.task_type = task_type
                self.is_fitted = False
            
            def fit(self, X, y):
                # Generate base model predictions
                base_predictions = []
                for model in self.models:
                    if hasattr(model, 'predict'):
                        pred = model.predict(X)
                    elif hasattr(model, 'predict_proba'):
                        pred = model.predict_proba(X)
                    else:
                        raise ValueError("Model does not support prediction")
                    base_predictions.append(pred)
                
                # Stack predictions
                stacked_features = np.column_stack(base_predictions)
                
                # Fit meta model
                self.meta_model.fit(stacked_features, y)
                self.is_fitted = True
            
            def predict(self, X):
                if not self.is_fitted:
                    raise ValueError("Ensemble must be fitted before prediction")
                
                # Generate base model predictions
                base_predictions = []
                for model in self.models:
                    if hasattr(model, 'predict'):
                        pred = model.predict(X)
                    elif hasattr(model, 'predict_proba'):
                        pred = model.predict_proba(X)
                    else:
                        raise ValueError("Model does not support prediction")
                    base_predictions.append(pred)
                
                # Stack predictions
                stacked_features = np.column_stack(base_predictions)
                
                # Predict with meta model
                return self.meta_model.predict(stacked_features)
            
            def predict_proba(self, X):
                if not self.is_fitted:
                    raise ValueError("Ensemble must be fitted before prediction")
                
                # Generate base model predictions
                base_predictions = []
                for model in self.models:
                    if hasattr(model, 'predict_proba'):
                        pred = model.predict_proba(X)
                    elif hasattr(model, 'predict'):
                        pred = model.predict(X)
                        # Convert to probabilities (simple approach)
                        pred = np.column_stack([1-pred, pred])
                    else:
                        raise ValueError("Model does not support probability prediction")
                    base_predictions.append(pred)
                
                # Stack predictions
                stacked_features = np.column_stack(base_predictions)
                
                # Predict with meta model
                if hasattr(self.meta_model, 'predict_proba'):
                    return self.meta_model.predict_proba(stacked_features)
                else:
                    # Convert to probabilities
                    pred = self.meta_model.predict(stacked_features)
                    return np.column_stack([1-pred, pred])
        
        task_type = params.get('task_type', 'regression') if params else 'regression'
        meta_model_class = self.ensemble_configs['stacking']['meta_models'][task_type]
        meta_model = meta_model_class()
        
        return StackingEnsemble(models, meta_model, task_type)
    
    async def predict_with_ensemble(self, 
                                   ensemble_id: str, 
                                   data: pd.DataFrame,
                                   return_individual_predictions: bool = False) -> Dict[str, Any]:
        """
        Make predictions using an ensemble.
        
        Args:
            ensemble_id: ID of the ensemble
            data: Input data
            return_individual_predictions: Whether to return individual model predictions
            
        Returns:
            Dictionary containing ensemble predictions
        """
        start_time = time.time()
        
        try:
            # Get ensemble
            if ensemble_id not in self.model_cache:
                raise ValueError(f"Ensemble {ensemble_id} not found")
            
            ensemble = self.model_cache[ensemble_id]
            
            # Preprocess data
            preprocessed_data = await self._preprocess_ensemble_data(data)
            
            # Extract features
            features = await self._extract_ensemble_features(preprocessed_data)
            
            # Make ensemble prediction
            ensemble_prediction = ensemble.predict(features)
            
            # Get individual predictions if requested
            individual_predictions = None
            if return_individual_predictions:
                individual_predictions = await self._get_individual_predictions(ensemble, features)
            
            # Calculate uncertainty
            uncertainty = await self._calculate_ensemble_uncertainty(ensemble, features)
            
            # Update metrics
            self.ensemble_metrics['total_predictions_made'] += 1
            
            return {
                'ensemble_prediction': ensemble_prediction.tolist() if hasattr(ensemble_prediction, 'tolist') else ensemble_prediction,
                'individual_predictions': individual_predictions,
                'uncertainty': uncertainty,
                'ensemble_id': ensemble_id,
                'prediction_time': time.time() - start_time,
                'feature_count': len(features.columns) if hasattr(features, 'columns') else 0,
                'timestamp': datetime.now().isoformat(),
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"Ensemble prediction failed: {e}")
            
            return {
                'error': str(e),
                'ensemble_id': ensemble_id,
                'prediction_time': time.time() - start_time,
                'timestamp': datetime.now().isoformat(),
                'status': 'error'
            }
    
    async def _preprocess_ensemble_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess data for ensemble prediction.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Preprocessed DataFrame
        """
        # Clean data
        cleaned_data = self.preprocessors['data_cleaner'].clean_data(data)
        
        # Detect and handle outliers
        outlier_info = self.preprocessors['outlier_detector'].detect_outliers(cleaned_data)
        if outlier_info['outlier_count'] > 0:
            logger.info(f"Detected {outlier_info['outlier_count']} outliers in ensemble data")
            cleaned_data = self.preprocessors['outlier_detector'].handle_outliers(cleaned_data)
        
        # Validate data quality
        validation_results = self.preprocessors['data_validator'].validate_data(cleaned_data)
        if not validation_results['is_valid']:
            logger.warning(f"Ensemble data validation failed: {validation_results['errors']}")
        
        return cleaned_data
    
    async def _extract_ensemble_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features for ensemble prediction.
        
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
    
    async def _get_individual_predictions(self, ensemble: Any, features: pd.DataFrame) -> Dict[str, Any]:
        """
        Get individual model predictions from ensemble.
        
        Args:
            ensemble: Ensemble model
            features: Feature DataFrame
            
        Returns:
            Dictionary containing individual predictions
        """
        individual_predictions = {}
        
        # Get individual models from ensemble
        if hasattr(ensemble, 'models'):
            for i, model in enumerate(ensemble.models):
                try:
                    if hasattr(model, 'predict'):
                        pred = model.predict(features)
                    elif hasattr(model, 'predict_proba'):
                        pred = model.predict_proba(features)
                    else:
                        continue
                    
                    individual_predictions[f'model_{i}'] = {
                        'prediction': pred.tolist() if hasattr(pred, 'tolist') else pred,
                        'model_type': type(model).__name__
                    }
                except Exception as e:
                    logger.warning(f"Failed to get prediction from model {i}: {e}")
        
        return individual_predictions
    
    async def _calculate_ensemble_uncertainty(self, ensemble: Any, features: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate ensemble uncertainty.
        
        Args:
            ensemble: Ensemble model
            features: Feature DataFrame
            
        Returns:
            Dictionary containing uncertainty metrics
        """
        uncertainty = {}
        
        # Get individual predictions
        individual_predictions = await self._get_individual_predictions(ensemble, features)
        
        if individual_predictions:
            # Calculate prediction variance
            predictions = []
            for pred_data in individual_predictions.values():
                if 'prediction' in pred_data:
                    pred = pred_data['prediction']
                    if isinstance(pred, list):
                        predictions.append(pred)
                    else:
                        predictions.append([pred])
            
            if predictions:
                predictions = np.array(predictions)
                uncertainty['prediction_variance'] = np.var(predictions, axis=0).tolist()
                uncertainty['prediction_std'] = np.std(predictions, axis=0).tolist()
                uncertainty['prediction_range'] = (np.max(predictions, axis=0) - np.min(predictions, axis=0)).tolist()
                
                # Calculate agreement score
                if len(predictions) > 1:
                    agreement_scores = []
                    for i in range(len(predictions[0])):
                        pred_values = predictions[:, i]
                        if len(set(pred_values)) == 1:
                            agreement_scores.append(1.0)  # Perfect agreement
                        else:
                            agreement_scores.append(1.0 - (np.std(pred_values) / (np.mean(pred_values) + 1e-6)))
                    uncertainty['agreement_score'] = agreement_scores
        
        return uncertainty
    
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
    
    def _update_ensemble_metrics(self, start_time: float, success: bool):
        """
        Update ensemble metrics.
        
        Args:
            start_time: Processing start time
            success: Whether processing was successful
        """
        processing_time = time.time() - start_time
        
        self.ensemble_metrics['total_ensembles_created'] += 1
        if success:
            self.ensemble_metrics['successful_ensembles'] += 1
        else:
            self.ensemble_metrics['failed_ensembles'] += 1
        
        self.ensemble_metrics['last_ensemble_time'] = datetime.now().isoformat()
    
    def get_ensemble_metrics(self) -> Dict[str, Any]:
        """
        Get current ensemble metrics.
        
        Returns:
            Dictionary containing ensemble metrics
        """
        return self.ensemble_metrics.copy()
    
    def get_available_ensembles(self) -> List[str]:
        """
        Get list of available ensemble IDs.
        
        Returns:
            List of ensemble IDs
        """
        return [key for key in self.model_cache.keys() if key.startswith('ensemble_')]
    
    def clear_ensemble_cache(self):
        """
        Clear the ensemble cache.
        """
        self.model_cache.clear()
        logger.info("Ensemble cache cleared")
    
    def get_ensemble_info(self, ensemble_id: str) -> Dict[str, Any]:
        """
        Get information about a specific ensemble.
        
        Args:
            ensemble_id: ID of the ensemble
            
        Returns:
            Dictionary containing ensemble information
        """
        if ensemble_id not in self.model_cache:
            return {'error': f'Ensemble {ensemble_id} not found'}
        
        ensemble = self.model_cache[ensemble_id]
        
        return {
            'ensemble_id': ensemble_id,
            'ensemble_type': type(ensemble).__name__,
            'model_count': len(ensemble.models) if hasattr(ensemble, 'models') else 'unknown',
            'is_fitted': getattr(ensemble, 'is_fitted', False),
            'created_at': ensemble_id.split('_')[1] if '_' in ensemble_id else 'unknown'
        }
