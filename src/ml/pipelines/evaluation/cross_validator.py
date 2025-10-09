"""
Cross Validation Pipeline

This module implements the cross validation pipeline for PBF-LB/M processes.
It handles comprehensive cross-validation, model selection, and hyperparameter tuning
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
from sklearn.model_selection import (
    KFold, StratifiedKFold, TimeSeriesSplit, 
    cross_val_score, cross_validate, 
    GridSearchCV, RandomizedSearchCV,
    validation_curve, learning_curve
)
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


class CrossValidatorPipeline:
    """
    Cross validation pipeline for PBF-LB/M processes.
    
    This pipeline handles comprehensive cross-validation for:
    - Model validation and selection
    - Hyperparameter tuning
    - Performance estimation
    - Overfitting detection
    """
    
    def __init__(self, config_manager=None):
        """
        Initialize the cross validation pipeline.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config_manager = config_manager
        self.pipeline_name = "cross_validator"
        
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
        
        # Cross validation metrics
        self.cv_metrics = {
            'total_cv_runs': 0,
            'successful_cv_runs': 0,
            'failed_cv_runs': 0,
            'average_cv_time': 0.0,
            'last_cv_time': None
        }
        
        logger.info("Initialized CrossValidatorPipeline")
    
    async def perform_cross_validation(self, 
                                      model_config: Dict[str, Any], 
                                      data: pd.DataFrame,
                                      cv_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform comprehensive cross-validation.
        
        Args:
            model_config: Model configuration
            data: Dataset for cross-validation
            cv_config: Cross-validation configuration
            
        Returns:
            Dictionary containing cross-validation results
        """
        start_time = time.time()
        
        try:
            # Load model
            model = await self._load_model_for_cv(model_config)
            
            # Preprocess data
            preprocessed_data = await self._preprocess_cv_data(data)
            
            # Extract features
            features = await self._extract_cv_features(preprocessed_data)
            
            # Prepare data for cross-validation
            X, y = await self._prepare_cv_data(features, data)
            
            # Perform cross-validation
            cv_results = await self._execute_cross_validation(model, X, y, cv_config)
            
            # Generate cross-validation visualizations
            cv_visualizations = await self._generate_cv_visualizations(cv_results, cv_config)
            
            # Analyze cross-validation results
            cv_analysis = await self._analyze_cv_results(cv_results, cv_config)
            
            # Update metrics
            self._update_cv_metrics(start_time, True)
            
            return {
                'model_name': model_config['model_name'],
                'model_version': model_config.get('version', 'latest'),
                'cv_results': cv_results,
                'cv_analysis': cv_analysis,
                'cv_visualizations': cv_visualizations,
                'cv_config': cv_config,
                'cv_time': time.time() - start_time,
                'data_size': len(data),
                'feature_count': len(features.columns) if hasattr(features, 'columns') else 0,
                'timestamp': datetime.now().isoformat(),
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"Cross-validation failed: {e}")
            self._update_cv_metrics(start_time, False)
            
            return {
                'model_name': model_config.get('model_name', 'unknown'),
                'error': str(e),
                'cv_time': time.time() - start_time,
                'timestamp': datetime.now().isoformat(),
                'status': 'error'
            }
    
    async def perform_hyperparameter_tuning(self, 
                                           model_config: Dict[str, Any], 
                                           data: pd.DataFrame,
                                           tuning_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning using cross-validation.
        
        Args:
            model_config: Model configuration
            data: Dataset for tuning
            tuning_config: Hyperparameter tuning configuration
            
        Returns:
            Dictionary containing tuning results
        """
        start_time = time.time()
        
        try:
            # Load model
            model = await self._load_model_for_cv(model_config)
            
            # Preprocess data
            preprocessed_data = await self._preprocess_cv_data(data)
            
            # Extract features
            features = await self._extract_cv_features(preprocessed_data)
            
            # Prepare data for tuning
            X, y = await self._prepare_cv_data(features, data)
            
            # Perform hyperparameter tuning
            tuning_results = await self._execute_hyperparameter_tuning(model, X, y, tuning_config)
            
            # Generate tuning visualizations
            tuning_visualizations = await self._generate_tuning_visualizations(tuning_results, tuning_config)
            
            # Analyze tuning results
            tuning_analysis = await self._analyze_tuning_results(tuning_results, tuning_config)
            
            # Update metrics
            self._update_cv_metrics(start_time, True)
            
            return {
                'model_name': model_config['model_name'],
                'model_version': model_config.get('version', 'latest'),
                'tuning_results': tuning_results,
                'tuning_analysis': tuning_analysis,
                'tuning_visualizations': tuning_visualizations,
                'tuning_config': tuning_config,
                'tuning_time': time.time() - start_time,
                'data_size': len(data),
                'feature_count': len(features.columns) if hasattr(features, 'columns') else 0,
                'timestamp': datetime.now().isoformat(),
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"Hyperparameter tuning failed: {e}")
            self._update_cv_metrics(start_time, False)
            
            return {
                'model_name': model_config.get('model_name', 'unknown'),
                'error': str(e),
                'tuning_time': time.time() - start_time,
                'timestamp': datetime.now().isoformat(),
                'status': 'error'
            }
    
    async def perform_learning_curve_analysis(self, 
                                             model_config: Dict[str, Any], 
                                             data: pd.DataFrame,
                                             learning_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform learning curve analysis.
        
        Args:
            model_config: Model configuration
            data: Dataset for analysis
            learning_config: Learning curve configuration
            
        Returns:
            Dictionary containing learning curve results
        """
        start_time = time.time()
        
        try:
            # Load model
            model = await self._load_model_for_cv(model_config)
            
            # Preprocess data
            preprocessed_data = await self._preprocess_cv_data(data)
            
            # Extract features
            features = await self._extract_cv_features(preprocessed_data)
            
            # Prepare data for analysis
            X, y = await self._prepare_cv_data(features, data)
            
            # Perform learning curve analysis
            learning_results = await self._execute_learning_curve_analysis(model, X, y, learning_config)
            
            # Generate learning curve visualizations
            learning_visualizations = await self._generate_learning_visualizations(learning_results, learning_config)
            
            # Analyze learning curve results
            learning_analysis = await self._analyze_learning_results(learning_results, learning_config)
            
            # Update metrics
            self._update_cv_metrics(start_time, True)
            
            return {
                'model_name': model_config['model_name'],
                'model_version': model_config.get('version', 'latest'),
                'learning_results': learning_results,
                'learning_analysis': learning_analysis,
                'learning_visualizations': learning_visualizations,
                'learning_config': learning_config,
                'learning_time': time.time() - start_time,
                'data_size': len(data),
                'feature_count': len(features.columns) if hasattr(features, 'columns') else 0,
                'timestamp': datetime.now().isoformat(),
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"Learning curve analysis failed: {e}")
            self._update_cv_metrics(start_time, False)
            
            return {
                'model_name': model_config.get('model_name', 'unknown'),
                'error': str(e),
                'learning_time': time.time() - start_time,
                'timestamp': datetime.now().isoformat(),
                'status': 'error'
            }
    
    async def _load_model_for_cv(self, model_config: Dict[str, Any]) -> Any:
        """
        Load model for cross-validation.
        
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
    
    async def _preprocess_cv_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess data for cross-validation.
        
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
            logger.info(f"Detected {outlier_info['outlier_count']} outliers in CV data")
            cleaned_data = self.preprocessors['outlier_detector'].handle_outliers(cleaned_data)
        
        # Validate data quality
        validation_results = self.preprocessors['data_validator'].validate_data(cleaned_data)
        if not validation_results['is_valid']:
            logger.warning(f"CV data validation failed: {validation_results['errors']}")
        
        return cleaned_data
    
    async def _extract_cv_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features for cross-validation.
        
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
    
    async def _prepare_cv_data(self, features: pd.DataFrame, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for cross-validation.
        
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
    
    async def _execute_cross_validation(self, 
                                       model: Any, 
                                       X: np.ndarray, 
                                       y: np.ndarray, 
                                       cv_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute cross-validation.
        
        Args:
            model: Model instance
            X: Features
            y: Targets
            cv_config: Cross-validation configuration
            
        Returns:
            Dictionary containing cross-validation results
        """
        cv_type = cv_config.get('type', 'kfold')
        cv_folds = cv_config.get('folds', 5)
        cv_scoring = cv_config.get('scoring', 'accuracy')
        cv_metrics = cv_config.get('metrics', [cv_scoring])
        
        try:
            # Create cross-validation strategy
            if cv_type == 'kfold':
                cv_strategy = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
            elif cv_type == 'stratified_kfold':
                cv_strategy = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            elif cv_type == 'time_series_split':
                cv_strategy = TimeSeriesSplit(n_splits=cv_folds)
            else:
                raise ValueError(f"Unsupported CV type: {cv_type}")
            
            # Perform cross-validation
            cv_results = cross_validate(
                model, X, y, 
                cv=cv_strategy, 
                scoring=cv_metrics,
                return_train_score=True,
                return_estimator=True
            )
            
            return {
                'cv_type': cv_type,
                'cv_folds': cv_folds,
                'cv_scoring': cv_scoring,
                'cv_metrics': cv_metrics,
                'test_scores': cv_results['test_score'].tolist(),
                'train_scores': cv_results['train_score'].tolist(),
                'fit_times': cv_results['fit_time'].tolist(),
                'score_times': cv_results['score_time'].tolist(),
                'estimators': cv_results['estimator']
            }
            
        except Exception as e:
            logger.error(f"Cross-validation execution failed: {e}")
            return {'error': str(e)}
    
    async def _execute_hyperparameter_tuning(self, 
                                            model: Any, 
                                            X: np.ndarray, 
                                            y: np.ndarray, 
                                            tuning_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute hyperparameter tuning.
        
        Args:
            model: Model instance
            X: Features
            y: Targets
            tuning_config: Tuning configuration
            
        Returns:
            Dictionary containing tuning results
        """
        tuning_method = tuning_config.get('method', 'grid_search')
        param_grid = tuning_config.get('param_grid', {})
        cv_folds = tuning_config.get('cv_folds', 5)
        scoring = tuning_config.get('scoring', 'accuracy')
        n_iter = tuning_config.get('n_iter', 100)
        
        try:
            # Create cross-validation strategy
            cv_strategy = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
            
            # Perform hyperparameter tuning
            if tuning_method == 'grid_search':
                search = GridSearchCV(
                    model, param_grid, 
                    cv=cv_strategy, 
                    scoring=scoring,
                    n_jobs=-1
                )
            elif tuning_method == 'randomized_search':
                search = RandomizedSearchCV(
                    model, param_grid, 
                    cv=cv_strategy, 
                    scoring=scoring,
                    n_iter=n_iter,
                    n_jobs=-1,
                    random_state=42
                )
            else:
                raise ValueError(f"Unsupported tuning method: {tuning_method}")
            
            # Fit the search
            search.fit(X, y)
            
            return {
                'tuning_method': tuning_method,
                'best_params': search.best_params_,
                'best_score': search.best_score_,
                'best_estimator': search.best_estimator_,
                'cv_results': search.cv_results_,
                'param_grid': param_grid,
                'cv_folds': cv_folds,
                'scoring': scoring
            }
            
        except Exception as e:
            logger.error(f"Hyperparameter tuning execution failed: {e}")
            return {'error': str(e)}
    
    async def _execute_learning_curve_analysis(self, 
                                              model: Any, 
                                              X: np.ndarray, 
                                              y: np.ndarray, 
                                              learning_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute learning curve analysis.
        
        Args:
            model: Model instance
            X: Features
            y: Targets
            learning_config: Learning curve configuration
            
        Returns:
            Dictionary containing learning curve results
        """
        train_sizes = learning_config.get('train_sizes', np.linspace(0.1, 1.0, 10))
        cv_folds = learning_config.get('cv_folds', 5)
        scoring = learning_config.get('scoring', 'accuracy')
        
        try:
            # Create cross-validation strategy
            cv_strategy = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
            
            # Perform learning curve analysis
            train_sizes_abs, train_scores, val_scores = learning_curve(
                model, X, y,
                train_sizes=train_sizes,
                cv=cv_strategy,
                scoring=scoring,
                n_jobs=-1
            )
            
            return {
                'train_sizes': train_sizes_abs.tolist(),
                'train_scores': train_scores.tolist(),
                'val_scores': val_scores.tolist(),
                'cv_folds': cv_folds,
                'scoring': scoring
            }
            
        except Exception as e:
            logger.error(f"Learning curve analysis execution failed: {e}")
            return {'error': str(e)}
    
    async def _generate_cv_visualizations(self, 
                                         cv_results: Dict[str, Any], 
                                         cv_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate cross-validation visualizations.
        
        Args:
            cv_results: Cross-validation results
            cv_config: Cross-validation configuration
            
        Returns:
            Dictionary containing visualization paths
        """
        visualizations = {}
        
        try:
            # Generate CV scores plot
            cv_scores_path = self.visualizers['performance'].plot_cv_scores(cv_results)
            visualizations['cv_scores'] = cv_scores_path
            
            # Generate CV distribution plot
            cv_dist_path = self.visualizers['performance'].plot_cv_distribution(cv_results)
            visualizations['cv_distribution'] = cv_dist_path
            
        except Exception as e:
            logger.error(f"Failed to generate CV visualizations: {e}")
            visualizations['error'] = str(e)
        
        return visualizations
    
    async def _generate_tuning_visualizations(self, 
                                             tuning_results: Dict[str, Any], 
                                             tuning_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate hyperparameter tuning visualizations.
        
        Args:
            tuning_results: Tuning results
            tuning_config: Tuning configuration
            
        Returns:
            Dictionary containing visualization paths
        """
        visualizations = {}
        
        try:
            # Generate parameter importance plot
            param_importance_path = self.visualizers['performance'].plot_parameter_importance(tuning_results)
            visualizations['parameter_importance'] = param_importance_path
            
            # Generate tuning progress plot
            tuning_progress_path = self.visualizers['performance'].plot_tuning_progress(tuning_results)
            visualizations['tuning_progress'] = tuning_progress_path
            
        except Exception as e:
            logger.error(f"Failed to generate tuning visualizations: {e}")
            visualizations['error'] = str(e)
        
        return visualizations
    
    async def _generate_learning_visualizations(self, 
                                               learning_results: Dict[str, Any], 
                                               learning_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate learning curve visualizations.
        
        Args:
            learning_results: Learning curve results
            learning_config: Learning curve configuration
            
        Returns:
            Dictionary containing visualization paths
        """
        visualizations = {}
        
        try:
            # Generate learning curve plot
            learning_curve_path = self.visualizers['performance'].plot_learning_curve(learning_results)
            visualizations['learning_curve'] = learning_curve_path
            
            # Generate bias-variance analysis plot
            bias_variance_path = self.visualizers['performance'].plot_bias_variance_analysis(learning_results)
            visualizations['bias_variance'] = bias_variance_path
            
        except Exception as e:
            logger.error(f"Failed to generate learning visualizations: {e}")
            visualizations['error'] = str(e)
        
        return visualizations
    
    async def _analyze_cv_results(self, 
                                 cv_results: Dict[str, Any], 
                                 cv_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze cross-validation results.
        
        Args:
            cv_results: Cross-validation results
            cv_config: Cross-validation configuration
            
        Returns:
            Dictionary containing analysis results
        """
        analysis = {}
        
        try:
            if 'test_scores' in cv_results:
                test_scores = np.array(cv_results['test_scores'])
                train_scores = np.array(cv_results['train_scores'])
                
                # Calculate statistics
                analysis['test_score_mean'] = np.mean(test_scores)
                analysis['test_score_std'] = np.std(test_scores)
                analysis['test_score_min'] = np.min(test_scores)
                analysis['test_score_max'] = np.max(test_scores)
                
                analysis['train_score_mean'] = np.mean(train_scores)
                analysis['train_score_std'] = np.std(train_scores)
                analysis['train_score_min'] = np.min(train_scores)
                analysis['train_score_max'] = np.max(train_scores)
                
                # Calculate overfitting indicator
                analysis['overfitting_indicator'] = analysis['train_score_mean'] - analysis['test_score_mean']
                
                # Calculate stability indicator
                analysis['stability_indicator'] = analysis['test_score_std']
                
        except Exception as e:
            logger.error(f"Failed to analyze CV results: {e}")
            analysis['error'] = str(e)
        
        return analysis
    
    async def _analyze_tuning_results(self, 
                                     tuning_results: Dict[str, Any], 
                                     tuning_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze hyperparameter tuning results.
        
        Args:
            tuning_results: Tuning results
            tuning_config: Tuning configuration
            
        Returns:
            Dictionary containing analysis results
        """
        analysis = {}
        
        try:
            if 'best_params' in tuning_results:
                analysis['best_params'] = tuning_results['best_params']
                analysis['best_score'] = tuning_results['best_score']
                
                # Analyze parameter importance
                if 'cv_results' in tuning_results:
                    cv_results = tuning_results['cv_results']
                    param_names = [key for key in cv_results.keys() if key.startswith('param_')]
                    
                    param_importance = {}
                    for param_name in param_names:
                        param_values = cv_results[param_name]
                        param_scores = cv_results['mean_test_score']
                        
                        # Calculate parameter importance (simplified)
                        param_importance[param_name] = np.std(param_scores)
                    
                    analysis['parameter_importance'] = param_importance
                
        except Exception as e:
            logger.error(f"Failed to analyze tuning results: {e}")
            analysis['error'] = str(e)
        
        return analysis
    
    async def _analyze_learning_results(self, 
                                       learning_results: Dict[str, Any], 
                                       learning_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze learning curve results.
        
        Args:
            learning_results: Learning curve results
            learning_config: Learning curve configuration
            
        Returns:
            Dictionary containing analysis results
        """
        analysis = {}
        
        try:
            if 'train_scores' in learning_results:
                train_scores = np.array(learning_results['train_scores'])
                val_scores = np.array(learning_results['val_scores'])
                
                # Calculate learning curve statistics
                analysis['final_train_score'] = np.mean(train_scores[-1])
                analysis['final_val_score'] = np.mean(val_scores[-1])
                analysis['learning_gap'] = analysis['final_train_score'] - analysis['final_val_score']
                
                # Calculate learning rate
                if len(train_scores) > 1:
                    learning_rate = np.mean(np.diff(train_scores, axis=0))
                    analysis['learning_rate'] = learning_rate.tolist()
                
                # Detect overfitting
                if analysis['learning_gap'] > 0.1:
                    analysis['overfitting_detected'] = True
                else:
                    analysis['overfitting_detected'] = False
                
        except Exception as e:
            logger.error(f"Failed to analyze learning results: {e}")
            analysis['error'] = str(e)
        
        return analysis
    
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
    
    def _update_cv_metrics(self, start_time: float, success: bool):
        """
        Update cross-validation metrics.
        
        Args:
            start_time: CV start time
            success: Whether CV was successful
        """
        cv_time = time.time() - start_time
        
        self.cv_metrics['total_cv_runs'] += 1
        if success:
            self.cv_metrics['successful_cv_runs'] += 1
        else:
            self.cv_metrics['failed_cv_runs'] += 1
        
        # Update average CV time
        total_successful = self.cv_metrics['successful_cv_runs']
        if total_successful > 0:
            current_avg = self.cv_metrics['average_cv_time']
            self.cv_metrics['average_cv_time'] = (
                (current_avg * (total_successful - 1) + cv_time) / total_successful
            )
        
        self.cv_metrics['last_cv_time'] = datetime.now().isoformat()
    
    def get_cv_metrics(self) -> Dict[str, Any]:
        """
        Get current cross-validation metrics.
        
        Returns:
            Dictionary containing CV metrics
        """
        return self.cv_metrics.copy()
    
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
