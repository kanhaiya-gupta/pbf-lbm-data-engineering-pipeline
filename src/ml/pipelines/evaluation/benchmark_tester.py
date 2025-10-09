"""
Benchmark Testing Pipeline

This module implements the benchmark testing pipeline for PBF-LB/M processes.
It handles comprehensive benchmark testing, performance comparison, and validation
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


class BenchmarkTesterPipeline:
    """
    Benchmark testing pipeline for PBF-LB/M processes.
    
    This pipeline handles comprehensive benchmark testing for:
    - Model performance benchmarking
    - Benchmark comparison
    - Performance validation
    - Benchmark reporting
    """
    
    def __init__(self, config_manager=None):
        """
        Initialize the benchmark testing pipeline.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config_manager = config_manager
        self.pipeline_name = "benchmark_tester"
        
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
        
        # Benchmark testing metrics
        self.benchmark_metrics = {
            'total_benchmarks_run': 0,
            'successful_benchmarks': 0,
            'failed_benchmarks': 0,
            'average_benchmark_time': 0.0,
            'last_benchmark_time': None
        }
        
        logger.info("Initialized BenchmarkTesterPipeline")
    
    async def run_benchmark_test(self, 
                                model_config: Dict[str, Any], 
                                benchmark_data: pd.DataFrame,
                                benchmark_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run comprehensive benchmark test.
        
        Args:
            model_config: Model configuration
            benchmark_data: Benchmark dataset
            benchmark_config: Benchmark configuration
            
        Returns:
            Dictionary containing benchmark test results
        """
        start_time = time.time()
        
        try:
            # Load model
            model = await self._load_model_for_benchmark(model_config)
            
            # Preprocess benchmark data
            preprocessed_data = await self._preprocess_benchmark_data(benchmark_data)
            
            # Extract features
            features = await self._extract_benchmark_features(preprocessed_data)
            
            # Prepare data for benchmarking
            X_benchmark, y_benchmark = await self._prepare_benchmark_data(features, benchmark_data)
            
            # Generate predictions
            predictions = await self._generate_benchmark_predictions(model, X_benchmark)
            
            # Calculate benchmark metrics
            benchmark_metrics = await self._calculate_benchmark_metrics(y_benchmark, predictions, benchmark_config)
            
            # Perform benchmark validation
            benchmark_validation = await self._validate_benchmark_results(benchmark_metrics, benchmark_config)
            
            # Generate benchmark visualizations
            benchmark_visualizations = await self._generate_benchmark_visualizations(y_benchmark, predictions, benchmark_config)
            
            # Generate benchmark report
            benchmark_report = await self._generate_benchmark_report(benchmark_metrics, benchmark_validation, model_config)
            
            # Update metrics
            self._update_benchmark_metrics(start_time, True)
            
            return {
                'model_name': model_config['model_name'],
                'model_version': model_config.get('version', 'latest'),
                'benchmark_metrics': benchmark_metrics,
                'benchmark_validation': benchmark_validation,
                'benchmark_visualizations': benchmark_visualizations,
                'benchmark_report': benchmark_report,
                'benchmark_config': benchmark_config,
                'benchmark_time': time.time() - start_time,
                'benchmark_data_size': len(benchmark_data),
                'feature_count': len(features.columns) if hasattr(features, 'columns') else 0,
                'timestamp': datetime.now().isoformat(),
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"Benchmark test failed: {e}")
            self._update_benchmark_metrics(start_time, False)
            
            return {
                'model_name': model_config.get('model_name', 'unknown'),
                'error': str(e),
                'benchmark_time': time.time() - start_time,
                'timestamp': datetime.now().isoformat(),
                'status': 'error'
            }
    
    async def compare_benchmark_results(self, 
                                       benchmark_results: List[Dict[str, Any]], 
                                       comparison_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Compare benchmark results from multiple models.
        
        Args:
            benchmark_results: List of benchmark results
            comparison_config: Comparison configuration
            
        Returns:
            Dictionary containing benchmark comparison results
        """
        start_time = time.time()
        
        try:
            # Generate comparison metrics
            comparison_metrics = await self._generate_benchmark_comparison_metrics(benchmark_results)
            
            # Generate comparison visualizations
            comparison_visualizations = await self._generate_benchmark_comparison_visualizations(benchmark_results)
            
            # Generate comparison report
            comparison_report = await self._generate_benchmark_comparison_report(benchmark_results, comparison_metrics)
            
            return {
                'benchmark_results': benchmark_results,
                'comparison_metrics': comparison_metrics,
                'comparison_visualizations': comparison_visualizations,
                'comparison_report': comparison_report,
                'comparison_config': comparison_config,
                'comparison_time': time.time() - start_time,
                'model_count': len(benchmark_results),
                'timestamp': datetime.now().isoformat(),
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"Benchmark comparison failed: {e}")
            
            return {
                'error': str(e),
                'comparison_time': time.time() - start_time,
                'model_count': len(benchmark_results),
                'timestamp': datetime.now().isoformat(),
                'status': 'error'
            }
    
    async def run_benchmark_suite(self, 
                                 model_configs: List[Dict[str, Any]], 
                                 benchmark_data: pd.DataFrame,
                                 suite_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run benchmark test suite for multiple models.
        
        Args:
            model_configs: List of model configurations
            benchmark_data: Benchmark dataset
            suite_config: Suite configuration
            
        Returns:
            Dictionary containing benchmark suite results
        """
        start_time = time.time()
        
        try:
            # Run benchmark tests for all models
            benchmark_results = []
            for model_config in model_configs:
                result = await self.run_benchmark_test(model_config, benchmark_data, suite_config)
                benchmark_results.append(result)
            
            # Compare benchmark results
            comparison_results = await self.compare_benchmark_results(benchmark_results, suite_config)
            
            # Generate suite report
            suite_report = await self._generate_benchmark_suite_report(benchmark_results, comparison_results, suite_config)
            
            return {
                'benchmark_results': benchmark_results,
                'comparison_results': comparison_results,
                'suite_report': suite_report,
                'suite_config': suite_config,
                'suite_time': time.time() - start_time,
                'model_count': len(model_configs),
                'benchmark_data_size': len(benchmark_data),
                'timestamp': datetime.now().isoformat(),
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"Benchmark suite failed: {e}")
            
            return {
                'error': str(e),
                'suite_time': time.time() - start_time,
                'model_count': len(model_configs),
                'timestamp': datetime.now().isoformat(),
                'status': 'error'
            }
    
    async def _load_model_for_benchmark(self, model_config: Dict[str, Any]) -> Any:
        """
        Load model for benchmark testing.
        
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
    
    async def _preprocess_benchmark_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess data for benchmark testing.
        
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
            logger.info(f"Detected {outlier_info['outlier_count']} outliers in benchmark data")
            cleaned_data = self.preprocessors['outlier_detector'].handle_outliers(cleaned_data)
        
        # Validate data quality
        validation_results = self.preprocessors['data_validator'].validate_data(cleaned_data)
        if not validation_results['is_valid']:
            logger.warning(f"Benchmark data validation failed: {validation_results['errors']}")
        
        return cleaned_data
    
    async def _extract_benchmark_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features for benchmark testing.
        
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
    
    async def _prepare_benchmark_data(self, features: pd.DataFrame, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for benchmark testing.
        
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
    
    async def _generate_benchmark_predictions(self, model: Any, X: np.ndarray) -> Dict[str, Any]:
        """
        Generate predictions for benchmark testing.
        
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
            logger.error(f"Failed to generate benchmark predictions: {e}")
            predictions['error'] = str(e)
        
        return predictions
    
    async def _calculate_benchmark_metrics(self, 
                                         y_true: np.ndarray, 
                                         predictions: Dict[str, Any], 
                                         benchmark_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate benchmark metrics.
        
        Args:
            y_true: True labels
            predictions: Model predictions
            benchmark_config: Benchmark configuration
            
        Returns:
            Dictionary containing benchmark metrics
        """
        metrics = {}
        
        if 'predictions' not in predictions:
            return {'error': 'No predictions available for benchmarking'}
        
        y_pred = predictions['predictions']
        
        try:
            # Calculate basic performance metrics
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
            metrics['precision'] = precision_score(y_true, y_pred, average='weighted')
            metrics['recall'] = recall_score(y_true, y_pred, average='weighted')
            metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted')
            metrics['mse'] = mean_squared_error(y_true, y_pred)
            metrics['mae'] = mean_absolute_error(y_true, y_pred)
            metrics['r2'] = r2_score(y_true, y_pred)
            
            # Calculate benchmark-specific metrics
            metrics['benchmark_timestamp'] = datetime.now().isoformat()
            metrics['benchmark_data_size'] = len(y_true)
            
            # Calculate confidence metrics if available
            if 'probabilities' in predictions:
                y_proba = predictions['probabilities']
                if len(y_proba.shape) == 2:
                    confidence_scores = np.max(y_proba, axis=1)
                    metrics['average_confidence'] = np.mean(confidence_scores)
                    metrics['confidence_std'] = np.std(confidence_scores)
                    metrics['low_confidence_count'] = np.sum(confidence_scores < 0.5)
            
            # Calculate manufacturing-specific metrics
            custom_metrics = self.evaluators['custom'].calculate_manufacturing_metrics(y_true, y_pred)
            metrics.update(custom_metrics)
            
        except Exception as e:
            logger.error(f"Failed to calculate benchmark metrics: {e}")
            metrics['error'] = str(e)
        
        return metrics
    
    async def _validate_benchmark_results(self, 
                                         benchmark_metrics: Dict[str, Any], 
                                         benchmark_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate benchmark results.
        
        Args:
            benchmark_metrics: Benchmark metrics
            benchmark_config: Benchmark configuration
            
        Returns:
            Dictionary containing validation results
        """
        validation = {}
        
        try:
            # Get validation thresholds
            accuracy_threshold = benchmark_config.get('accuracy_threshold', 0.8)
            mse_threshold = benchmark_config.get('mse_threshold', 0.1)
            r2_threshold = benchmark_config.get('r2_threshold', 0.7)
            
            # Validate performance metrics
            if 'accuracy' in benchmark_metrics:
                validation['accuracy_valid'] = benchmark_metrics['accuracy'] >= accuracy_threshold
                validation['accuracy_score'] = benchmark_metrics['accuracy']
                validation['accuracy_threshold'] = accuracy_threshold
            
            if 'mse' in benchmark_metrics:
                validation['mse_valid'] = benchmark_metrics['mse'] <= mse_threshold
                validation['mse_score'] = benchmark_metrics['mse']
                validation['mse_threshold'] = mse_threshold
            
            if 'r2' in benchmark_metrics:
                validation['r2_valid'] = benchmark_metrics['r2'] >= r2_threshold
                validation['r2_score'] = benchmark_metrics['r2']
                validation['r2_threshold'] = r2_threshold
            
            # Overall validation
            validation['overall_valid'] = all([
                validation.get('accuracy_valid', True),
                validation.get('mse_valid', True),
                validation.get('r2_valid', True)
            ])
            
            # Validation summary
            validation['validation_summary'] = {
                'passed_checks': sum([
                    validation.get('accuracy_valid', False),
                    validation.get('mse_valid', False),
                    validation.get('r2_valid', False)
                ]),
                'total_checks': 3,
                'validation_status': 'PASSED' if validation['overall_valid'] else 'FAILED'
            }
            
        except Exception as e:
            logger.error(f"Failed to validate benchmark results: {e}")
            validation['error'] = str(e)
        
        return validation
    
    async def _generate_benchmark_visualizations(self, 
                                                y_true: np.ndarray, 
                                                predictions: Dict[str, Any], 
                                                benchmark_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate benchmark visualizations.
        
        Args:
            y_true: True labels
            predictions: Model predictions
            benchmark_config: Benchmark configuration
            
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
            if benchmark_config.get('task_type') == 'classification':
                confusion_matrix_path = self.visualizers['prediction'].plot_confusion_matrix(y_true, y_pred)
                visualizations['confusion_matrix'] = confusion_matrix_path
            
            # Generate ROC curve for binary classification
            if (benchmark_config.get('task_type') == 'classification' and 
                'probabilities' in predictions and 
                len(predictions['probabilities'][0]) == 2):
                roc_curve_path = self.visualizers['prediction'].plot_roc_curve(y_true, predictions['probabilities'])
                visualizations['roc_curve'] = roc_curve_path
            
            # Generate benchmark dashboard
            benchmark_dashboard_path = self.visualizers['performance'].plot_benchmark_dashboard(y_true, y_pred, benchmark_config)
            visualizations['benchmark_dashboard'] = benchmark_dashboard_path
            
        except Exception as e:
            logger.error(f"Failed to generate benchmark visualizations: {e}")
            visualizations['error'] = str(e)
        
        return visualizations
    
    async def _generate_benchmark_report(self, 
                                        benchmark_metrics: Dict[str, Any], 
                                        benchmark_validation: Dict[str, Any], 
                                        model_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate benchmark report.
        
        Args:
            benchmark_metrics: Benchmark metrics
            benchmark_validation: Benchmark validation results
            model_config: Model configuration
            
        Returns:
            Dictionary containing benchmark report
        """
        report = {
            'model_name': model_config['model_name'],
            'model_version': model_config.get('version', 'latest'),
            'benchmark_timestamp': datetime.now().isoformat(),
            'benchmark_summary': {},
            'validation_results': benchmark_validation,
            'recommendations': []
        }
        
        try:
            # Benchmark summary
            if 'error' not in benchmark_metrics:
                report['benchmark_summary'] = {
                    'overall_performance': 'Good' if benchmark_metrics.get('accuracy', 0) > 0.8 else 'Needs Improvement',
                    'key_metrics': benchmark_metrics,
                    'performance_grade': self._calculate_benchmark_grade(benchmark_metrics)
                }
            
            # Generate recommendations
            if not benchmark_validation.get('overall_valid', False):
                report['recommendations'].append("Benchmark validation failed. Review model performance and thresholds")
            
            if benchmark_metrics.get('low_confidence_count', 0) > 0:
                report['recommendations'].append("High number of low-confidence predictions. Consider model retraining")
            
            if benchmark_metrics.get('accuracy', 0) < 0.8:
                report['recommendations'].append("Accuracy below 80%. Consider feature engineering or model tuning")
            
        except Exception as e:
            logger.error(f"Failed to generate benchmark report: {e}")
            report['error'] = str(e)
        
        return report
    
    def _calculate_benchmark_grade(self, metrics: Dict[str, Any]) -> str:
        """
        Calculate benchmark grade.
        
        Args:
            metrics: Benchmark metrics
            
        Returns:
            Benchmark grade (A, B, C, D, F)
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
    
    async def _generate_benchmark_comparison_metrics(self, benchmark_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate benchmark comparison metrics.
        
        Args:
            benchmark_results: List of benchmark results
            
        Returns:
            Dictionary containing comparison metrics
        """
        comparison_metrics = {}
        
        # Extract benchmark metrics for comparison
        benchmark_data = []
        for result in benchmark_results:
            if 'benchmark_metrics' in result:
                benchmark_data.append({
                    'model_name': result['model_name'],
                    'metrics': result['benchmark_metrics']
                })
        
        if not benchmark_data:
            return {'error': 'No benchmark metrics available for comparison'}
        
        # Calculate comparison statistics
        comparison_metrics['model_count'] = len(benchmark_data)
        comparison_metrics['best_models'] = {}
        
        # Find best model for each metric
        for metric_name in benchmark_data[0]['metrics'].keys():
            if metric_name != 'error':
                metric_values = []
                for data in benchmark_data:
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
    
    async def _generate_benchmark_comparison_visualizations(self, benchmark_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate benchmark comparison visualizations.
        
        Args:
            benchmark_results: List of benchmark results
            
        Returns:
            Dictionary containing comparison visualization paths
        """
        visualizations = {}
        
        try:
            # Generate benchmark comparison plot
            comparison_plot_path = self.visualizers['performance'].plot_benchmark_comparison(benchmark_results)
            visualizations['benchmark_comparison'] = comparison_plot_path
            
            # Generate performance distribution plot
            performance_dist_path = self.visualizers['performance'].plot_benchmark_distribution(benchmark_results)
            visualizations['benchmark_distribution'] = performance_dist_path
            
        except Exception as e:
            logger.error(f"Failed to generate benchmark comparison visualizations: {e}")
            visualizations['error'] = str(e)
        
        return visualizations
    
    async def _generate_benchmark_comparison_report(self, 
                                                   benchmark_results: List[Dict[str, Any]], 
                                                   comparison_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate benchmark comparison report.
        
        Args:
            benchmark_results: List of benchmark results
            comparison_metrics: Comparison metrics
            
        Returns:
            Dictionary containing comparison report
        """
        report = {
            'comparison_timestamp': datetime.now().isoformat(),
            'model_count': len(benchmark_results),
            'comparison_summary': {},
            'best_models': comparison_metrics.get('best_models', {}),
            'recommendations': []
        }
        
        try:
            # Generate comparison summary
            report['comparison_summary'] = {
                'total_models_compared': len(benchmark_results),
                'successful_benchmarks': sum(1 for result in benchmark_results if result.get('status') == 'success'),
                'failed_benchmarks': sum(1 for result in benchmark_results if result.get('status') == 'error')
            }
            
            # Generate recommendations
            if comparison_metrics.get('best_models'):
                best_models = comparison_metrics['best_models']
                if 'accuracy' in best_models:
                    report['recommendations'].append(f"Best accuracy model: {best_models['accuracy']['best_model']}")
                if 'r2' in best_models:
                    report['recommendations'].append(f"Best R2 model: {best_models['r2']['best_model']}")
            
        except Exception as e:
            logger.error(f"Failed to generate benchmark comparison report: {e}")
            report['error'] = str(e)
        
        return report
    
    async def _generate_benchmark_suite_report(self, 
                                              benchmark_results: List[Dict[str, Any]], 
                                              comparison_results: Dict[str, Any], 
                                              suite_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate benchmark suite report.
        
        Args:
            benchmark_results: List of benchmark results
            comparison_results: Comparison results
            suite_config: Suite configuration
            
        Returns:
            Dictionary containing suite report
        """
        report = {
            'suite_timestamp': datetime.now().isoformat(),
            'model_count': len(benchmark_results),
            'suite_summary': {},
            'comparison_summary': comparison_results.get('comparison_report', {}),
            'recommendations': []
        }
        
        try:
            # Generate suite summary
            report['suite_summary'] = {
                'total_models_tested': len(benchmark_results),
                'successful_benchmarks': sum(1 for result in benchmark_results if result.get('status') == 'success'),
                'failed_benchmarks': sum(1 for result in benchmark_results if result.get('status') == 'error'),
                'suite_config': suite_config
            }
            
            # Generate recommendations
            if comparison_results.get('comparison_metrics', {}).get('best_models'):
                best_models = comparison_results['comparison_metrics']['best_models']
                if 'accuracy' in best_models:
                    report['recommendations'].append(f"Recommended model for accuracy: {best_models['accuracy']['best_model']}")
                if 'r2' in best_models:
                    report['recommendations'].append(f"Recommended model for R2: {best_models['r2']['best_model']}")
            
        except Exception as e:
            logger.error(f"Failed to generate benchmark suite report: {e}")
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
    
    def _update_benchmark_metrics(self, start_time: float, success: bool):
        """
        Update benchmark testing metrics.
        
        Args:
            start_time: Benchmark start time
            success: Whether benchmark was successful
        """
        benchmark_time = time.time() - start_time
        
        self.benchmark_metrics['total_benchmarks_run'] += 1
        if success:
            self.benchmark_metrics['successful_benchmarks'] += 1
        else:
            self.benchmark_metrics['failed_benchmarks'] += 1
        
        # Update average benchmark time
        total_successful = self.benchmark_metrics['successful_benchmarks']
        if total_successful > 0:
            current_avg = self.benchmark_metrics['average_benchmark_time']
            self.benchmark_metrics['average_benchmark_time'] = (
                (current_avg * (total_successful - 1) + benchmark_time) / total_successful
            )
        
        self.benchmark_metrics['last_benchmark_time'] = datetime.now().isoformat()
    
    def get_benchmark_metrics(self) -> Dict[str, Any]:
        """
        Get current benchmark testing metrics.
        
        Returns:
            Dictionary containing benchmark metrics
        """
        return self.benchmark_metrics.copy()
    
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
