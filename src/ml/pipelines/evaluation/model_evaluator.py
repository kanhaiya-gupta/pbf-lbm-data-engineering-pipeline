"""
Model Evaluation Pipeline

This module implements the model evaluation pipeline for PBF-LB/M processes.
It handles comprehensive model evaluation, performance analysis, and validation
for all ML models in the system.
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
from sklearn.model_selection import cross_val_score, validation_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

from ...models.base_model import BaseModel
from ...features.process_features import LaserParameterFeatures, BuildParameterFeatures, MaterialFeatures
from ...features.sensor_features import PyrometerFeatures, CameraFeatures, AccelerometerFeatures, TemperatureFeatures
from ...features.image_features import CTScanFeatures, PowderBedFeatures, DefectImageFeatures, SurfaceTextureFeatures
from ...features.temporal_features import TimeSeriesFeatures, LagFeatures, RollingFeatures, FrequencyFeatures
from ...utils.preprocessing import DataCleaner, FeatureScaler, OutlierDetector, DataValidator
from ...utils.evaluation import RegressionMetrics, ClassificationMetrics, TimeSeriesMetrics, CustomMetrics
from ...utils.visualization import ModelVisualizer, FeatureVisualizer, PredictionVisualizer, PerformanceVisualizer

logger = logging.getLogger(__name__)


class ModelEvaluatorPipeline:
    """
    Model evaluation pipeline for PBF-LB/M processes.
    
    This pipeline handles comprehensive model evaluation for:
    - Performance metrics calculation
    - Cross-validation
    - Model comparison
    - Performance visualization
    - Model validation
    """
    
    def __init__(self, config_manager=None):
        """
        Initialize the model evaluation pipeline.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config_manager = config_manager
        self.pipeline_name = "model_evaluator"
        
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
        
        # Initialize visualizers
        self.visualizers = {
            'model': ModelVisualizer(),
            'feature': FeatureVisualizer(),
            'prediction': PredictionVisualizer(),
            'performance': PerformanceVisualizer()
        }
        
        # Model cache for loaded models
        self.model_cache = {}
        
        # Evaluation metrics
        self.evaluation_metrics = {
            'total_models_evaluated': 0,
            'successful_evaluations': 0,
            'failed_evaluations': 0,
            'average_evaluation_time': 0.0,
            'last_evaluation_time': None
        }
        
        logger.info("Initialized ModelEvaluatorPipeline")
    
    async def evaluate_model(self, 
                            model_config: Dict[str, Any], 
                            test_data: pd.DataFrame,
                            evaluation_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Evaluate a single model comprehensively.
        
        Args:
            model_config: Model configuration
            test_data: Test dataset
            evaluation_config: Evaluation configuration
            
        Returns:
            Dictionary containing evaluation results
        """
        start_time = time.time()
        
        try:
            # Load model
            model = await self._load_model_for_evaluation(model_config)
            
            # Preprocess test data
            preprocessed_data = await self._preprocess_evaluation_data(test_data)
            
            # Extract features
            features = await self._extract_evaluation_features(preprocessed_data)
            
            # Prepare data for evaluation
            X_test, y_test = await self._prepare_evaluation_data(features, test_data)
            
            # Generate predictions
            predictions = await self._generate_evaluation_predictions(model, X_test)
            
            # Calculate performance metrics
            performance_metrics = await self._calculate_performance_metrics(y_test, predictions, model_config)
            
            # Generate evaluation visualizations
            visualizations = await self._generate_evaluation_visualizations(y_test, predictions, model_config)
            
            # Perform cross-validation if requested
            cv_results = None
            if evaluation_config and evaluation_config.get('cross_validation', False):
                cv_results = await self._perform_cross_validation(model, X_test, y_test, evaluation_config)
            
            # Calculate model interpretability metrics
            interpretability_metrics = await self._calculate_interpretability_metrics(model, X_test, model_config)
            
            # Update metrics
            self._update_evaluation_metrics(start_time, True)
            
            return {
                'model_name': model_config['model_name'],
                'model_version': model_config.get('version', 'latest'),
                'performance_metrics': performance_metrics,
                'cross_validation_results': cv_results,
                'interpretability_metrics': interpretability_metrics,
                'visualizations': visualizations,
                'evaluation_config': evaluation_config,
                'evaluation_time': time.time() - start_time,
                'test_data_size': len(test_data),
                'feature_count': len(features.columns) if hasattr(features, 'columns') else 0,
                'timestamp': datetime.now().isoformat(),
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"Model evaluation failed: {e}")
            self._update_evaluation_metrics(start_time, False)
            
            return {
                'model_name': model_config.get('model_name', 'unknown'),
                'error': str(e),
                'evaluation_time': time.time() - start_time,
                'timestamp': datetime.now().isoformat(),
                'status': 'error'
            }
    
    async def compare_models(self, 
                           model_configs: List[Dict[str, Any]], 
                           test_data: pd.DataFrame,
                           comparison_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Compare multiple models.
        
        Args:
            model_configs: List of model configurations
            test_data: Test dataset
            comparison_config: Comparison configuration
            
        Returns:
            Dictionary containing comparison results
        """
        start_time = time.time()
        
        try:
            # Evaluate each model
            model_results = []
            for model_config in model_configs:
                result = await self.evaluate_model(model_config, test_data, comparison_config)
                model_results.append(result)
            
            # Generate comparison metrics
            comparison_metrics = await self._generate_comparison_metrics(model_results)
            
            # Generate comparison visualizations
            comparison_visualizations = await self._generate_comparison_visualizations(model_results)
            
            # Rank models
            model_ranking = await self._rank_models(model_results, comparison_config)
            
            return {
                'model_results': model_results,
                'comparison_metrics': comparison_metrics,
                'comparison_visualizations': comparison_visualizations,
                'model_ranking': model_ranking,
                'comparison_config': comparison_config,
                'comparison_time': time.time() - start_time,
                'model_count': len(model_configs),
                'test_data_size': len(test_data),
                'timestamp': datetime.now().isoformat(),
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"Model comparison failed: {e}")
            
            return {
                'error': str(e),
                'comparison_time': time.time() - start_time,
                'model_count': len(model_configs),
                'timestamp': datetime.now().isoformat(),
                'status': 'error'
            }
    
    async def _load_model_for_evaluation(self, model_config: Dict[str, Any]) -> Any:
        """
        Load model for evaluation.
        
        Args:
            model_config: Model configuration
            
        Returns:
            Loaded model
        """
        model_name = model_config['model_name']
        model_version = model_config.get('version', 'latest')
        
        try:
            # Load model if not cached
            if model_name not in self.model_cache:
                self.model_cache[model_name] = self._load_model(model_name, model_version)
            
            return self.model_cache[model_name]
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise
    
    async def _preprocess_evaluation_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess data for evaluation.
        
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
            logger.info(f"Detected {outlier_info['outlier_count']} outliers in evaluation data")
            cleaned_data = self.preprocessors['outlier_detector'].handle_outliers(cleaned_data)
        
        # Validate data quality
        validation_results = self.preprocessors['data_validator'].validate_data(cleaned_data)
        if not validation_results['is_valid']:
            logger.warning(f"Evaluation data validation failed: {validation_results['errors']}")
        
        return cleaned_data
    
    async def _extract_evaluation_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features for evaluation.
        
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
    
    async def _prepare_evaluation_data(self, features: pd.DataFrame, test_data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for evaluation.
        
        Args:
            features: Feature DataFrame
            test_data: Test DataFrame
            
        Returns:
            Tuple of (X, y) arrays
        """
        # Separate features and target
        target_columns = ['quality_score', 'defect_probability', 'health_status', 'failure_probability']
        feature_columns = [col for col in features.columns if col not in target_columns]
        
        X = features[feature_columns].values
        y = test_data[target_columns[0]].values if target_columns[0] in test_data.columns else np.zeros(len(features))
        
        return X, y
    
    async def _generate_evaluation_predictions(self, model: Any, X_test: np.ndarray) -> Dict[str, Any]:
        """
        Generate predictions for evaluation.
        
        Args:
            model: Model instance
            X_test: Test features
            
        Returns:
            Dictionary containing predictions
        """
        predictions = {}
        
        try:
            # Generate predictions
            if hasattr(model, 'predict'):
                y_pred = model.predict(X_test)
                predictions['predictions'] = y_pred.tolist() if hasattr(y_pred, 'tolist') else y_pred
            
            # Generate probabilities if available
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test)
                predictions['probabilities'] = y_proba.tolist() if hasattr(y_proba, 'tolist') else y_proba
            
            # Generate decision function if available
            if hasattr(model, 'decision_function'):
                y_decision = model.decision_function(X_test)
                predictions['decision_function'] = y_decision.tolist() if hasattr(y_decision, 'tolist') else y_decision
            
        except Exception as e:
            logger.error(f"Failed to generate predictions: {e}")
            predictions['error'] = str(e)
        
        return predictions
    
    async def _calculate_performance_metrics(self, 
                                           y_true: np.ndarray, 
                                           predictions: Dict[str, Any], 
                                           model_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate performance metrics.
        
        Args:
            y_true: True labels
            predictions: Model predictions
            model_config: Model configuration
            
        Returns:
            Dictionary containing performance metrics
        """
        metrics = {}
        
        if 'predictions' not in predictions:
            return {'error': 'No predictions available'}
        
        y_pred = predictions['predictions']
        
        # Determine task type
        task_type = model_config.get('task_type', 'regression')
        
        if task_type == 'regression':
            # Regression metrics
            metrics['mse'] = mean_squared_error(y_true, y_pred)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            metrics['mae'] = mean_absolute_error(y_true, y_pred)
            metrics['r2'] = r2_score(y_true, y_pred)
            
            # Custom manufacturing metrics
            custom_metrics = self.evaluators['custom'].calculate_manufacturing_metrics(y_true, y_pred)
            metrics.update(custom_metrics)
            
        elif task_type == 'classification':
            # Classification metrics
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
            metrics['precision'] = precision_score(y_true, y_pred, average='weighted')
            metrics['recall'] = recall_score(y_true, y_pred, average='weighted')
            metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted')
            
            # ROC AUC if probabilities available
            if 'probabilities' in predictions:
                y_proba = predictions['probabilities']
                if len(y_proba.shape) == 2 and y_proba.shape[1] == 2:
                    metrics['roc_auc'] = roc_auc_score(y_true, y_proba[:, 1])
            
            # Custom manufacturing metrics
            custom_metrics = self.evaluators['custom'].calculate_manufacturing_metrics(y_true, y_pred)
            metrics.update(custom_metrics)
        
        elif task_type == 'time_series':
            # Time series metrics
            time_series_metrics = self.evaluators['time_series'].calculate_time_series_metrics(y_true, y_pred)
            metrics.update(time_series_metrics)
        
        return metrics
    
    async def _generate_evaluation_visualizations(self, 
                                                 y_true: np.ndarray, 
                                                 predictions: Dict[str, Any], 
                                                 model_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate evaluation visualizations.
        
        Args:
            y_true: True labels
            predictions: Model predictions
            model_config: Model configuration
            
        Returns:
            Dictionary containing visualization paths
        """
        visualizations = {}
        
        if 'predictions' not in predictions:
            return {'error': 'No predictions available for visualization'}
        
        y_pred = predictions['predictions']
        
        try:
            # Generate prediction vs actual plot
            pred_vs_actual_path = self.visualizers['prediction'].plot_prediction_vs_actual(y_true, y_pred)
            visualizations['prediction_vs_actual'] = pred_vs_actual_path
            
            # Generate residual plot
            residual_path = self.visualizers['prediction'].plot_residuals(y_true, y_pred)
            visualizations['residuals'] = residual_path
            
            # Generate confusion matrix for classification
            if model_config.get('task_type') == 'classification':
                confusion_matrix_path = self.visualizers['prediction'].plot_confusion_matrix(y_true, y_pred)
                visualizations['confusion_matrix'] = confusion_matrix_path
            
            # Generate ROC curve for binary classification
            if (model_config.get('task_type') == 'classification' and 
                'probabilities' in predictions and 
                len(predictions['probabilities'][0]) == 2):
                roc_curve_path = self.visualizers['prediction'].plot_roc_curve(y_true, predictions['probabilities'])
                visualizations['roc_curve'] = roc_curve_path
            
        except Exception as e:
            logger.error(f"Failed to generate visualizations: {e}")
            visualizations['error'] = str(e)
        
        return visualizations
    
    async def _perform_cross_validation(self, 
                                       model: Any, 
                                       X: np.ndarray, 
                                       y: np.ndarray, 
                                       evaluation_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform cross-validation.
        
        Args:
            model: Model instance
            X: Features
            y: Targets
            evaluation_config: Evaluation configuration
            
        Returns:
            Dictionary containing cross-validation results
        """
        cv_config = evaluation_config.get('cross_validation', {})
        cv_folds = cv_config.get('folds', 5)
        cv_scoring = cv_config.get('scoring', 'accuracy')
        
        try:
            # Perform cross-validation
            cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring=cv_scoring)
            
            return {
                'cv_scores': cv_scores.tolist(),
                'cv_mean': np.mean(cv_scores),
                'cv_std': np.std(cv_scores),
                'cv_folds': cv_folds,
                'cv_scoring': cv_scoring
            }
            
        except Exception as e:
            logger.error(f"Cross-validation failed: {e}")
            return {'error': str(e)}
    
    async def _calculate_interpretability_metrics(self, 
                                                 model: Any, 
                                                 X: np.ndarray, 
                                                 model_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate model interpretability metrics.
        
        Args:
            model: Model instance
            X: Features
            model_config: Model configuration
            
        Returns:
            Dictionary containing interpretability metrics
        """
        interpretability = {}
        
        try:
            # Feature importance if available
            if hasattr(model, 'feature_importances_'):
                interpretability['feature_importance'] = model.feature_importances_.tolist()
            
            # Model complexity
            if hasattr(model, 'n_estimators'):
                interpretability['n_estimators'] = model.n_estimators
            
            if hasattr(model, 'max_depth'):
                interpretability['max_depth'] = model.max_depth
            
            # Model size (approximate)
            if hasattr(model, 'coef_'):
                interpretability['n_features'] = len(model.coef_)
            
        except Exception as e:
            logger.error(f"Failed to calculate interpretability metrics: {e}")
            interpretability['error'] = str(e)
        
        return interpretability
    
    async def _generate_comparison_metrics(self, model_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate comparison metrics for multiple models.
        
        Args:
            model_results: List of model evaluation results
            
        Returns:
            Dictionary containing comparison metrics
        """
        comparison_metrics = {}
        
        # Extract performance metrics for comparison
        performance_data = []
        for result in model_results:
            if 'performance_metrics' in result:
                performance_data.append({
                    'model_name': result['model_name'],
                    'metrics': result['performance_metrics']
                })
        
        if not performance_data:
            return {'error': 'No performance metrics available for comparison'}
        
        # Calculate comparison statistics
        comparison_metrics['model_count'] = len(performance_data)
        comparison_metrics['best_models'] = {}
        
        # Find best model for each metric
        for metric_name in performance_data[0]['metrics'].keys():
            if metric_name != 'error':
                metric_values = []
                for data in performance_data:
                    if metric_name in data['metrics']:
                        metric_values.append((data['model_name'], data['metrics'][metric_name]))
                
                if metric_values:
                    # Sort by metric value (higher is better for most metrics)
                    metric_values.sort(key=lambda x: x[1], reverse=True)
                    comparison_metrics['best_models'][metric_name] = {
                        'best_model': metric_values[0][0],
                        'best_value': metric_values[0][1],
                        'all_values': metric_values
                    }
        
        return comparison_metrics
    
    async def _generate_comparison_visualizations(self, model_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate comparison visualizations.
        
        Args:
            model_results: List of model evaluation results
            
        Returns:
            Dictionary containing comparison visualization paths
        """
        visualizations = {}
        
        try:
            # Generate model comparison plot
            comparison_plot_path = self.visualizers['performance'].plot_model_comparison(model_results)
            visualizations['model_comparison'] = comparison_plot_path
            
            # Generate performance distribution plot
            performance_dist_path = self.visualizers['performance'].plot_performance_distribution(model_results)
            visualizations['performance_distribution'] = performance_dist_path
            
        except Exception as e:
            logger.error(f"Failed to generate comparison visualizations: {e}")
            visualizations['error'] = str(e)
        
        return visualizations
    
    async def _rank_models(self, 
                          model_results: List[Dict[str, Any]], 
                          comparison_config: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Rank models based on performance.
        
        Args:
            model_results: List of model evaluation results
            comparison_config: Comparison configuration
            
        Returns:
            List of ranked models
        """
        ranking_config = comparison_config.get('ranking', {}) if comparison_config else {}
        ranking_metric = ranking_config.get('primary_metric', 'accuracy')
        ranking_weights = ranking_config.get('weights', {})
        
        # Calculate ranking scores
        ranking_scores = []
        for result in model_results:
            if 'performance_metrics' in result:
                metrics = result['performance_metrics']
                score = 0.0
                
                # Calculate weighted score
                for metric, weight in ranking_weights.items():
                    if metric in metrics:
                        score += metrics[metric] * weight
                
                # If no weights specified, use primary metric
                if not ranking_weights and ranking_metric in metrics:
                    score = metrics[ranking_metric]
                
                ranking_scores.append({
                    'model_name': result['model_name'],
                    'score': score,
                    'metrics': metrics
                })
        
        # Sort by score
        ranking_scores.sort(key=lambda x: x['score'], reverse=True)
        
        # Add rank
        for i, model in enumerate(ranking_scores):
            model['rank'] = i + 1
        
        return ranking_scores
    
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
    
    def _update_evaluation_metrics(self, start_time: float, success: bool):
        """
        Update evaluation metrics.
        
        Args:
            start_time: Evaluation start time
            success: Whether evaluation was successful
        """
        evaluation_time = time.time() - start_time
        
        self.evaluation_metrics['total_models_evaluated'] += 1
        if success:
            self.evaluation_metrics['successful_evaluations'] += 1
        else:
            self.evaluation_metrics['failed_evaluations'] += 1
        
        # Update average evaluation time
        total_successful = self.evaluation_metrics['successful_evaluations']
        if total_successful > 0:
            current_avg = self.evaluation_metrics['average_evaluation_time']
            self.evaluation_metrics['average_evaluation_time'] = (
                (current_avg * (total_successful - 1) + evaluation_time) / total_successful
            )
        
        self.evaluation_metrics['last_evaluation_time'] = datetime.now().isoformat()
    
    def get_evaluation_metrics(self) -> Dict[str, Any]:
        """
        Get current evaluation metrics.
        
        Returns:
            Dictionary containing evaluation metrics
        """
        return self.evaluation_metrics.copy()
    
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
