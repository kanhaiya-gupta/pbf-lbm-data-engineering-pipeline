"""
Performance Analysis Pipeline

This module implements the performance analysis pipeline for PBF-LB/M processes.
It handles comprehensive performance analysis, monitoring, and reporting
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


class PerformanceAnalyzerPipeline:
    """
    Performance analysis pipeline for PBF-LB/M processes.
    
    This pipeline handles comprehensive performance analysis for:
    - Model performance monitoring
    - Performance trend analysis
    - Performance comparison
    - Performance reporting
    """
    
    def __init__(self, config_manager=None):
        """
        Initialize the performance analysis pipeline.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config_manager = config_manager
        self.pipeline_name = "performance_analyzer"
        
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
        
        # Performance analysis metrics
        self.performance_metrics = {
            'total_analyses_performed': 0,
            'successful_analyses': 0,
            'failed_analyses': 0,
            'average_analysis_time': 0.0,
            'last_analysis_time': None
        }
        
        logger.info("Initialized PerformanceAnalyzerPipeline")
    
    async def analyze_model_performance(self, 
                                       model_config: Dict[str, Any], 
                                       test_data: pd.DataFrame,
                                       performance_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyze model performance comprehensively.
        
        Args:
            model_config: Model configuration
            test_data: Test dataset
            performance_config: Performance analysis configuration
            
        Returns:
            Dictionary containing performance analysis results
        """
        start_time = time.time()
        
        try:
            # Load model
            model = await self._load_model_for_analysis(model_config)
            
            # Preprocess test data
            preprocessed_data = await self._preprocess_analysis_data(test_data)
            
            # Extract features
            features = await self._extract_analysis_features(preprocessed_data)
            
            # Prepare data for analysis
            X_test, y_test = await self._prepare_analysis_data(features, test_data)
            
            # Generate predictions
            predictions = await self._generate_analysis_predictions(model, X_test)
            
            # Calculate performance metrics
            performance_metrics = await self._calculate_performance_metrics(y_test, predictions, model_config)
            
            # Analyze performance trends
            performance_trends = await self._analyze_performance_trends(y_test, predictions, performance_config)
            
            # Generate performance visualizations
            performance_visualizations = await self._generate_performance_visualizations(y_test, predictions, model_config)
            
            # Generate performance report
            performance_report = await self._generate_performance_report(performance_metrics, performance_trends, model_config)
            
            # Update metrics
            self._update_performance_metrics(start_time, True)
            
            return {
                'model_name': model_config['model_name'],
                'model_version': model_config.get('version', 'latest'),
                'performance_metrics': performance_metrics,
                'performance_trends': performance_trends,
                'performance_visualizations': performance_visualizations,
                'performance_report': performance_report,
                'performance_config': performance_config,
                'analysis_time': time.time() - start_time,
                'test_data_size': len(test_data),
                'feature_count': len(features.columns) if hasattr(features, 'columns') else 0,
                'timestamp': datetime.now().isoformat(),
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"Performance analysis failed: {e}")
            self._update_performance_metrics(start_time, False)
            
            return {
                'model_name': model_config.get('model_name', 'unknown'),
                'error': str(e),
                'analysis_time': time.time() - start_time,
                'timestamp': datetime.now().isoformat(),
                'status': 'error'
            }
    
    async def compare_model_performance(self, 
                                       model_configs: List[Dict[str, Any]], 
                                       test_data: pd.DataFrame,
                                       comparison_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Compare performance of multiple models.
        
        Args:
            model_configs: List of model configurations
            test_data: Test dataset
            comparison_config: Comparison configuration
            
        Returns:
            Dictionary containing performance comparison results
        """
        start_time = time.time()
        
        try:
            # Analyze each model
            model_analyses = []
            for model_config in model_configs:
                analysis = await self.analyze_model_performance(model_config, test_data, comparison_config)
                model_analyses.append(analysis)
            
            # Generate comparison metrics
            comparison_metrics = await self._generate_comparison_metrics(model_analyses)
            
            # Generate comparison visualizations
            comparison_visualizations = await self._generate_comparison_visualizations(model_analyses)
            
            # Generate comparison report
            comparison_report = await self._generate_comparison_report(model_analyses, comparison_metrics)
            
            return {
                'model_analyses': model_analyses,
                'comparison_metrics': comparison_metrics,
                'comparison_visualizations': comparison_visualizations,
                'comparison_report': comparison_report,
                'comparison_config': comparison_config,
                'comparison_time': time.time() - start_time,
                'model_count': len(model_configs),
                'test_data_size': len(test_data),
                'timestamp': datetime.now().isoformat(),
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"Performance comparison failed: {e}")
            
            return {
                'error': str(e),
                'comparison_time': time.time() - start_time,
                'model_count': len(model_configs),
                'timestamp': datetime.now().isoformat(),
                'status': 'error'
            }
    
    async def monitor_model_performance(self, 
                                       model_config: Dict[str, Any], 
                                       monitoring_data: pd.DataFrame,
                                       monitoring_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Monitor model performance over time.
        
        Args:
            model_config: Model configuration
            monitoring_data: Monitoring dataset
            monitoring_config: Monitoring configuration
            
        Returns:
            Dictionary containing monitoring results
        """
        start_time = time.time()
        
        try:
            # Load model
            model = await self._load_model_for_analysis(model_config)
            
            # Preprocess monitoring data
            preprocessed_data = await self._preprocess_analysis_data(monitoring_data)
            
            # Extract features
            features = await self._extract_analysis_features(preprocessed_data)
            
            # Prepare data for monitoring
            X_monitor, y_monitor = await self._prepare_analysis_data(features, monitoring_data)
            
            # Generate predictions
            predictions = await self._generate_analysis_predictions(model, X_monitor)
            
            # Calculate monitoring metrics
            monitoring_metrics = await self._calculate_monitoring_metrics(y_monitor, predictions, monitoring_config)
            
            # Detect performance drift
            drift_detection = await self._detect_performance_drift(monitoring_metrics, monitoring_config)
            
            # Generate monitoring visualizations
            monitoring_visualizations = await self._generate_monitoring_visualizations(monitoring_metrics, drift_detection)
            
            # Generate monitoring report
            monitoring_report = await self._generate_monitoring_report(monitoring_metrics, drift_detection, model_config)
            
            # Update metrics
            self._update_performance_metrics(start_time, True)
            
            return {
                'model_name': model_config['model_name'],
                'model_version': model_config.get('version', 'latest'),
                'monitoring_metrics': monitoring_metrics,
                'drift_detection': drift_detection,
                'monitoring_visualizations': monitoring_visualizations,
                'monitoring_report': monitoring_report,
                'monitoring_config': monitoring_config,
                'monitoring_time': time.time() - start_time,
                'monitoring_data_size': len(monitoring_data),
                'feature_count': len(features.columns) if hasattr(features, 'columns') else 0,
                'timestamp': datetime.now().isoformat(),
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"Performance monitoring failed: {e}")
            self._update_performance_metrics(start_time, False)
            
            return {
                'model_name': model_config.get('model_name', 'unknown'),
                'error': str(e),
                'monitoring_time': time.time() - start_time,
                'timestamp': datetime.now().isoformat(),
                'status': 'error'
            }
    
    async def _load_model_for_analysis(self, model_config: Dict[str, Any]) -> Any:
        """
        Load model for performance analysis.
        
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
    
    async def _preprocess_analysis_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess data for performance analysis.
        
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
            logger.info(f"Detected {outlier_info['outlier_count']} outliers in analysis data")
            cleaned_data = self.preprocessors['outlier_detector'].handle_outliers(cleaned_data)
        
        # Validate data quality
        validation_results = self.preprocessors['data_validator'].validate_data(cleaned_data)
        if not validation_results['is_valid']:
            logger.warning(f"Analysis data validation failed: {validation_results['errors']}")
        
        return cleaned_data
    
    async def _extract_analysis_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features for performance analysis.
        
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
    
    async def _prepare_analysis_data(self, features: pd.DataFrame, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for performance analysis.
        
        Args:
            features: Feature DataFrame
            data: Original DataFrame
            
        Returns:
            Tuple of (X, y) arrays
        """
        # Separate features and target
        target_columns = ['quality_score', 'defect_probability', 'health_status', 'failure_probability']
        feature_columns = [col for col in features.columns if col not in target_columns]
        
        X = features[feature_columns].values
        y = data[target_columns[0]].values if target_columns[0] in data.columns else np.zeros(len(features))
        
        return X, y
    
    async def _generate_analysis_predictions(self, model: Any, X: np.ndarray) -> Dict[str, Any]:
        """
        Generate predictions for performance analysis.
        
        Args:
            model: Model instance
            X: Features
            
        Returns:
            Dictionary containing predictions
        """
        predictions = {}
        
        try:
            # Generate predictions
            if hasattr(model, 'predict'):
                y_pred = model.predict(X)
                predictions['predictions'] = y_pred.tolist() if hasattr(y_pred, 'tolist') else y_pred
            
            # Generate probabilities if available
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X)
                predictions['probabilities'] = y_proba.tolist() if hasattr(y_proba, 'tolist') else y_proba
            
            # Generate decision function if available
            if hasattr(model, 'decision_function'):
                y_decision = model.decision_function(X)
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
    
    async def _analyze_performance_trends(self, 
                                         y_true: np.ndarray, 
                                         predictions: Dict[str, Any], 
                                         performance_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze performance trends.
        
        Args:
            y_true: True labels
            predictions: Model predictions
            performance_config: Performance configuration
            
        Returns:
            Dictionary containing trend analysis
        """
        trends = {}
        
        if 'predictions' not in predictions:
            return {'error': 'No predictions available for trend analysis'}
        
        y_pred = predictions['predictions']
        
        try:
            # Calculate performance over time windows
            window_size = performance_config.get('window_size', 100) if performance_config else 100
            
            if len(y_true) > window_size:
                # Calculate rolling performance metrics
                rolling_accuracy = []
                rolling_mse = []
                
                for i in range(window_size, len(y_true)):
                    window_y_true = y_true[i-window_size:i]
                    window_y_pred = y_pred[i-window_size:i]
                    
                    # Calculate metrics for this window
                    window_accuracy = accuracy_score(window_y_true, window_y_pred)
                    window_mse = mean_squared_error(window_y_true, window_y_pred)
                    
                    rolling_accuracy.append(window_accuracy)
                    rolling_mse.append(window_mse)
                
                # Analyze trends
                trends['rolling_accuracy'] = rolling_accuracy
                trends['rolling_mse'] = rolling_mse
                trends['accuracy_trend'] = self._calculate_trend(rolling_accuracy)
                trends['mse_trend'] = self._calculate_trend(rolling_mse)
                
                # Detect performance degradation
                trends['performance_degradation'] = self._detect_performance_degradation(rolling_accuracy, rolling_mse)
            
        except Exception as e:
            logger.error(f"Failed to analyze performance trends: {e}")
            trends['error'] = str(e)
        
        return trends
    
    def _calculate_trend(self, values: List[float]) -> str:
        """
        Calculate trend direction.
        
        Args:
            values: List of values
            
        Returns:
            Trend direction ('increasing', 'decreasing', 'stable')
        """
        if len(values) < 2:
            return 'stable'
        
        # Calculate slope
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        if slope > 0.01:
            return 'increasing'
        elif slope < -0.01:
            return 'decreasing'
        else:
            return 'stable'
    
    def _detect_performance_degradation(self, accuracy_values: List[float], mse_values: List[float]) -> bool:
        """
        Detect performance degradation.
        
        Args:
            accuracy_values: List of accuracy values
            mse_values: List of MSE values
            
        Returns:
            True if performance degradation detected
        """
        if len(accuracy_values) < 2 or len(mse_values) < 2:
            return False
        
        # Check for significant decrease in accuracy
        accuracy_decrease = accuracy_values[0] - accuracy_values[-1]
        accuracy_threshold = 0.05  # 5% decrease threshold
        
        # Check for significant increase in MSE
        mse_increase = mse_values[-1] - mse_values[0]
        mse_threshold = 0.1  # 10% increase threshold
        
        return accuracy_decrease > accuracy_threshold or mse_increase > mse_threshold
    
    async def _generate_performance_visualizations(self, 
                                                  y_true: np.ndarray, 
                                                  predictions: Dict[str, Any], 
                                                  model_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate performance visualizations.
        
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
            logger.error(f"Failed to generate performance visualizations: {e}")
            visualizations['error'] = str(e)
        
        return visualizations
    
    async def _generate_performance_report(self, 
                                          performance_metrics: Dict[str, Any], 
                                          performance_trends: Dict[str, Any], 
                                          model_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate performance report.
        
        Args:
            performance_metrics: Performance metrics
            performance_trends: Performance trends
            model_config: Model configuration
            
        Returns:
            Dictionary containing performance report
        """
        report = {
            'model_name': model_config['model_name'],
            'model_version': model_config.get('version', 'latest'),
            'analysis_timestamp': datetime.now().isoformat(),
            'performance_summary': {},
            'trend_analysis': {},
            'recommendations': []
        }
        
        try:
            # Performance summary
            if 'error' not in performance_metrics:
                report['performance_summary'] = {
                    'overall_performance': 'Good' if performance_metrics.get('accuracy', 0) > 0.8 else 'Needs Improvement',
                    'key_metrics': performance_metrics,
                    'performance_grade': self._calculate_performance_grade(performance_metrics)
                }
            
            # Trend analysis
            if 'error' not in performance_trends:
                report['trend_analysis'] = {
                    'accuracy_trend': performance_trends.get('accuracy_trend', 'unknown'),
                    'mse_trend': performance_trends.get('mse_trend', 'unknown'),
                    'performance_degradation': performance_trends.get('performance_degradation', False)
                }
            
            # Generate recommendations
            report['recommendations'] = self._generate_recommendations(performance_metrics, performance_trends)
            
        except Exception as e:
            logger.error(f"Failed to generate performance report: {e}")
            report['error'] = str(e)
        
        return report
    
    def _calculate_performance_grade(self, metrics: Dict[str, Any]) -> str:
        """
        Calculate performance grade.
        
        Args:
            metrics: Performance metrics
            
        Returns:
            Performance grade (A, B, C, D, F)
        """
        # Use accuracy for classification, R2 for regression
        if 'accuracy' in metrics:
            score = metrics['accuracy']
        elif 'r2' in metrics:
            score = metrics['r2']
        else:
            return 'F'
        
        if score >= 0.9:
            return 'A'
        elif score >= 0.8:
            return 'B'
        elif score >= 0.7:
            return 'C'
        elif score >= 0.6:
            return 'D'
        else:
            return 'F'
    
    def _generate_recommendations(self, 
                                 performance_metrics: Dict[str, Any], 
                                 performance_trends: Dict[str, Any]) -> List[str]:
        """
        Generate performance recommendations.
        
        Args:
            performance_metrics: Performance metrics
            performance_trends: Performance trends
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Check performance metrics
        if 'accuracy' in performance_metrics and performance_metrics['accuracy'] < 0.8:
            recommendations.append("Consider retraining the model with more data or different features")
        
        if 'r2' in performance_metrics and performance_metrics['r2'] < 0.7:
            recommendations.append("Model explains less than 70% of variance. Consider feature engineering")
        
        # Check trends
        if performance_trends.get('performance_degradation', False):
            recommendations.append("Performance degradation detected. Monitor model closely and consider retraining")
        
        if performance_trends.get('accuracy_trend') == 'decreasing':
            recommendations.append("Accuracy trend is decreasing. Investigate data drift or model staleness")
        
        if performance_trends.get('mse_trend') == 'increasing':
            recommendations.append("MSE trend is increasing. Consider model retraining or hyperparameter tuning")
        
        return recommendations
    
    async def _generate_comparison_metrics(self, model_analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate comparison metrics for multiple models.
        
        Args:
            model_analyses: List of model analysis results
            
        Returns:
            Dictionary containing comparison metrics
        """
        comparison_metrics = {}
        
        # Extract performance metrics for comparison
        performance_data = []
        for analysis in model_analyses:
            if 'performance_metrics' in analysis:
                performance_data.append({
                    'model_name': analysis['model_name'],
                    'metrics': analysis['performance_metrics']
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
    
    async def _generate_comparison_visualizations(self, model_analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate comparison visualizations.
        
        Args:
            model_analyses: List of model analysis results
            
        Returns:
            Dictionary containing comparison visualization paths
        """
        visualizations = {}
        
        try:
            # Generate model comparison plot
            comparison_plot_path = self.visualizers['performance'].plot_model_comparison(model_analyses)
            visualizations['model_comparison'] = comparison_plot_path
            
            # Generate performance distribution plot
            performance_dist_path = self.visualizers['performance'].plot_performance_distribution(model_analyses)
            visualizations['performance_distribution'] = performance_dist_path
            
        except Exception as e:
            logger.error(f"Failed to generate comparison visualizations: {e}")
            visualizations['error'] = str(e)
        
        return visualizations
    
    async def _generate_comparison_report(self, 
                                         model_analyses: List[Dict[str, Any]], 
                                         comparison_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comparison report.
        
        Args:
            model_analyses: List of model analysis results
            comparison_metrics: Comparison metrics
            
        Returns:
            Dictionary containing comparison report
        """
        report = {
            'comparison_timestamp': datetime.now().isoformat(),
            'model_count': len(model_analyses),
            'comparison_summary': {},
            'best_models': comparison_metrics.get('best_models', {}),
            'recommendations': []
        }
        
        try:
            # Generate comparison summary
            report['comparison_summary'] = {
                'total_models_compared': len(model_analyses),
                'successful_analyses': sum(1 for analysis in model_analyses if analysis.get('status') == 'success'),
                'failed_analyses': sum(1 for analysis in model_analyses if analysis.get('status') == 'error')
            }
            
            # Generate recommendations
            if comparison_metrics.get('best_models'):
                best_models = comparison_metrics['best_models']
                if 'accuracy' in best_models:
                    report['recommendations'].append(f"Best accuracy model: {best_models['accuracy']['best_model']}")
                if 'r2' in best_models:
                    report['recommendations'].append(f"Best R2 model: {best_models['r2']['best_model']}")
            
        except Exception as e:
            logger.error(f"Failed to generate comparison report: {e}")
            report['error'] = str(e)
        
        return report
    
    async def _calculate_monitoring_metrics(self, 
                                           y_true: np.ndarray, 
                                           predictions: Dict[str, Any], 
                                           monitoring_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate monitoring metrics.
        
        Args:
            y_true: True labels
            predictions: Model predictions
            monitoring_config: Monitoring configuration
            
        Returns:
            Dictionary containing monitoring metrics
        """
        metrics = {}
        
        if 'predictions' not in predictions:
            return {'error': 'No predictions available for monitoring'}
        
        y_pred = predictions['predictions']
        
        try:
            # Calculate basic performance metrics
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
            metrics['mse'] = mean_squared_error(y_true, y_pred)
            metrics['mae'] = mean_absolute_error(y_true, y_pred)
            
            # Calculate monitoring-specific metrics
            metrics['prediction_count'] = len(y_pred)
            metrics['monitoring_timestamp'] = datetime.now().isoformat()
            
            # Calculate confidence metrics if available
            if 'probabilities' in predictions:
                y_proba = predictions['probabilities']
                if len(y_proba.shape) == 2:
                    confidence_scores = np.max(y_proba, axis=1)
                    metrics['average_confidence'] = np.mean(confidence_scores)
                    metrics['confidence_std'] = np.std(confidence_scores)
                    metrics['low_confidence_count'] = np.sum(confidence_scores < 0.5)
            
        except Exception as e:
            logger.error(f"Failed to calculate monitoring metrics: {e}")
            metrics['error'] = str(e)
        
        return metrics
    
    async def _detect_performance_drift(self, 
                                       monitoring_metrics: Dict[str, Any], 
                                       monitoring_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect performance drift.
        
        Args:
            monitoring_metrics: Monitoring metrics
            monitoring_config: Monitoring configuration
            
        Returns:
            Dictionary containing drift detection results
        """
        drift_detection = {}
        
        try:
            # Get drift thresholds
            accuracy_threshold = monitoring_config.get('accuracy_threshold', 0.05)
            mse_threshold = monitoring_config.get('mse_threshold', 0.1)
            
            # Check for drift
            if 'accuracy' in monitoring_metrics:
                baseline_accuracy = monitoring_config.get('baseline_accuracy', 0.8)
                accuracy_drift = abs(monitoring_metrics['accuracy'] - baseline_accuracy)
                drift_detection['accuracy_drift'] = accuracy_drift > accuracy_threshold
                drift_detection['accuracy_drift_magnitude'] = accuracy_drift
            
            if 'mse' in monitoring_metrics:
                baseline_mse = monitoring_config.get('baseline_mse', 0.1)
                mse_drift = abs(monitoring_metrics['mse'] - baseline_mse)
                drift_detection['mse_drift'] = mse_drift > mse_threshold
                drift_detection['mse_drift_magnitude'] = mse_drift
            
            # Overall drift detection
            drift_detection['drift_detected'] = (
                drift_detection.get('accuracy_drift', False) or 
                drift_detection.get('mse_drift', False)
            )
            
        except Exception as e:
            logger.error(f"Failed to detect performance drift: {e}")
            drift_detection['error'] = str(e)
        
        return drift_detection
    
    async def _generate_monitoring_visualizations(self, 
                                                 monitoring_metrics: Dict[str, Any], 
                                                 drift_detection: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate monitoring visualizations.
        
        Args:
            monitoring_metrics: Monitoring metrics
            drift_detection: Drift detection results
            
        Returns:
            Dictionary containing monitoring visualization paths
        """
        visualizations = {}
        
        try:
            # Generate monitoring dashboard
            monitoring_dashboard_path = self.visualizers['performance'].plot_monitoring_dashboard(monitoring_metrics, drift_detection)
            visualizations['monitoring_dashboard'] = monitoring_dashboard_path
            
            # Generate drift detection plot
            if drift_detection.get('drift_detected', False):
                drift_plot_path = self.visualizers['performance'].plot_drift_detection(monitoring_metrics, drift_detection)
                visualizations['drift_detection'] = drift_plot_path
            
        except Exception as e:
            logger.error(f"Failed to generate monitoring visualizations: {e}")
            visualizations['error'] = str(e)
        
        return visualizations
    
    async def _generate_monitoring_report(self, 
                                         monitoring_metrics: Dict[str, Any], 
                                         drift_detection: Dict[str, Any], 
                                         model_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate monitoring report.
        
        Args:
            monitoring_metrics: Monitoring metrics
            drift_detection: Drift detection results
            model_config: Model configuration
            
        Returns:
            Dictionary containing monitoring report
        """
        report = {
            'model_name': model_config['model_name'],
            'model_version': model_config.get('version', 'latest'),
            'monitoring_timestamp': datetime.now().isoformat(),
            'monitoring_summary': {},
            'drift_analysis': drift_detection,
            'recommendations': []
        }
        
        try:
            # Monitoring summary
            if 'error' not in monitoring_metrics:
                report['monitoring_summary'] = {
                    'current_accuracy': monitoring_metrics.get('accuracy', 'N/A'),
                    'current_mse': monitoring_metrics.get('mse', 'N/A'),
                    'prediction_count': monitoring_metrics.get('prediction_count', 0),
                    'average_confidence': monitoring_metrics.get('average_confidence', 'N/A')
                }
            
            # Generate recommendations
            if drift_detection.get('drift_detected', False):
                report['recommendations'].append("Performance drift detected. Consider model retraining or data investigation")
            
            if monitoring_metrics.get('low_confidence_count', 0) > 0:
                report['recommendations'].append("High number of low-confidence predictions. Review model performance")
            
        except Exception as e:
            logger.error(f"Failed to generate monitoring report: {e}")
            report['error'] = str(e)
        
        return report
    
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
    
    def _update_performance_metrics(self, start_time: float, success: bool):
        """
        Update performance analysis metrics.
        
        Args:
            start_time: Analysis start time
            success: Whether analysis was successful
        """
        analysis_time = time.time() - start_time
        
        self.performance_metrics['total_analyses_performed'] += 1
        if success:
            self.performance_metrics['successful_analyses'] += 1
        else:
            self.performance_metrics['failed_analyses'] += 1
        
        # Update average analysis time
        total_successful = self.performance_metrics['successful_analyses']
        if total_successful > 0:
            current_avg = self.performance_metrics['average_analysis_time']
            self.performance_metrics['average_analysis_time'] = (
                (current_avg * (total_successful - 1) + analysis_time) / total_successful
            )
        
        self.performance_metrics['last_analysis_time'] = datetime.now().isoformat()
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get current performance analysis metrics.
        
        Returns:
            Dictionary containing performance metrics
        """
        return self.performance_metrics.copy()
    
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
