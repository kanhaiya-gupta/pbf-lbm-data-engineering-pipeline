"""
Base Training Pipeline Class

This module provides the base class for all training pipelines in the PBF-LB/M system.
It provides common functionality for data loading, preprocessing, training, and evaluation.
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, List, Tuple
import numpy as np
import pandas as pd
from pathlib import Path
import mlflow
import mlflow.sklearn
import mlflow.tensorflow
from datetime import datetime
import json

from ...config import ConfigManager
from ...utils.data_loaders import ProcessDataLoader, SensorDataLoader, ImageDataLoader, MultiSourceLoader
from ...utils.preprocessing import DataCleaner, FeatureScaler, OutlierDetector, DataValidator
from ...utils.evaluation import RegressionMetrics, ClassificationMetrics, TimeSeriesMetrics, CustomMetrics

logger = logging.getLogger(__name__)


class BaseTrainingPipeline(ABC):
    """
    Base class for all training pipelines in the PBF-LB/M system.
    
    This class provides common functionality for:
    - Data ingestion and preprocessing
    - Feature engineering
    - Model training and evaluation
    - Model registration and artifact management
    - Pipeline monitoring and logging
    """
    
    def __init__(self, pipeline_name: str, config_manager: Optional[ConfigManager] = None):
        """
        Initialize the base training pipeline.
        
        Args:
            pipeline_name: Name of the pipeline (should match YAML config file name)
            config_manager: Configuration manager instance
        """
        self.pipeline_name = pipeline_name
        self.config_manager = config_manager or ConfigManager()
        self.config = self._load_pipeline_config()
        self.stages = self.config.get('stages', [])
        self.metadata = self.config.get('pipeline', {}).get('metadata', {})
        
        # Initialize components
        self.data_loaders = self._initialize_data_loaders()
        self.preprocessors = self._initialize_preprocessors()
        self.evaluators = self._initialize_evaluators()
        
        # Pipeline state
        self.current_stage = None
        self.stage_results = {}
        self.pipeline_start_time = None
        self.pipeline_end_time = None
        
        logger.info(f"Initialized {pipeline_name} training pipeline")
    
    def _load_pipeline_config(self) -> Dict[str, Any]:
        """Load pipeline configuration from YAML."""
        try:
            return self.config_manager.load_pipeline_config(f"training/{self.pipeline_name}")
        except Exception as e:
            logger.error(f"Failed to load pipeline config: {e}")
            raise
    
    def _initialize_data_loaders(self) -> Dict[str, Any]:
        """Initialize data loaders based on configuration."""
        loaders = {}
        
        # Process data loader
        loaders['process'] = ProcessDataLoader()
        
        # Sensor data loader
        loaders['sensor'] = SensorDataLoader()
        
        # Image data loader
        loaders['image'] = ImageDataLoader()
        
        # Multi-source loader
        loaders['multi_source'] = MultiSourceLoader()
        
        return loaders
    
    def _initialize_preprocessors(self) -> Dict[str, Any]:
        """Initialize preprocessors based on configuration."""
        preprocessors = {}
        
        # Data cleaner
        preprocessors['cleaner'] = DataCleaner()
        
        # Feature scaler
        preprocessors['scaler'] = FeatureScaler()
        
        # Outlier detector
        preprocessors['outlier_detector'] = OutlierDetector()
        
        # Data validator
        preprocessors['validator'] = DataValidator()
        
        return preprocessors
    
    def _initialize_evaluators(self) -> Dict[str, Any]:
        """Initialize evaluators based on configuration."""
        evaluators = {}
        
        # Regression metrics
        evaluators['regression'] = RegressionMetrics()
        
        # Classification metrics
        evaluators['classification'] = ClassificationMetrics()
        
        # Time series metrics
        evaluators['time_series'] = TimeSeriesMetrics()
        
        # Custom metrics
        evaluators['custom'] = CustomMetrics()
        
        return evaluators
    
    def run(self) -> Dict[str, Any]:
        """
        Run the complete training pipeline.
        
        Returns:
            Dictionary containing pipeline results and metadata
        """
        logger.info(f"Starting {self.pipeline_name} training pipeline")
        self.pipeline_start_time = time.time()
        
        try:
            # Initialize MLflow experiment
            self._setup_mlflow_experiment()
            
            with mlflow.start_run(run_name=f"{self.pipeline_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                # Log pipeline metadata
                self._log_pipeline_metadata()
                
                # Execute pipeline stages
                for stage in self.stages:
                    self._execute_stage(stage)
                
                # Finalize pipeline
                self._finalize_pipeline()
                
                logger.info(f"Successfully completed {self.pipeline_name} training pipeline")
                
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            self._handle_pipeline_failure(e)
            raise
        
        finally:
            self.pipeline_end_time = time.time()
            self._log_pipeline_metrics()
        
        return self._get_pipeline_results()
    
    def _setup_mlflow_experiment(self):
        """Setup MLflow experiment for the pipeline."""
        experiment_name = self.config.get('environment', {}).get('env_vars', {}).get('MLFLOW_EXPERIMENT_NAME', f"{self.pipeline_name}_experiments")
        
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                mlflow.create_experiment(experiment_name)
            mlflow.set_experiment(experiment_name)
        except Exception as e:
            logger.warning(f"Failed to setup MLflow experiment: {e}")
    
    def _log_pipeline_metadata(self):
        """Log pipeline metadata to MLflow."""
        mlflow.log_param("pipeline_name", self.pipeline_name)
        mlflow.log_param("pipeline_version", self.metadata.get('version', '1.0.0'))
        mlflow.log_param("pipeline_description", self.metadata.get('description', ''))
        mlflow.log_param("environment", self.config_manager.environment)
        
        # Log stage information
        stage_names = [stage.get('name', 'unknown') for stage in self.stages]
        mlflow.log_param("pipeline_stages", json.dumps(stage_names))
    
    def _execute_stage(self, stage: Dict[str, Any]):
        """
        Execute a single pipeline stage.
        
        Args:
            stage: Stage configuration dictionary
        """
        stage_name = stage.get('name', 'unknown')
        stage_type = stage.get('type', 'unknown')
        stage_config = stage.get('config', {})
        
        logger.info(f"Executing stage: {stage_name} (type: {stage_type})")
        self.current_stage = stage_name
        
        stage_start_time = time.time()
        
        try:
            # Execute stage based on type
            if stage_type == "data_loader":
                result = self._execute_data_ingestion_stage(stage_config)
            elif stage_type == "data_preprocessor":
                result = self._execute_preprocessing_stage(stage_config)
            elif stage_type == "feature_processor":
                result = self._execute_feature_engineering_stage(stage_config)
            elif stage_type == "data_splitter":
                result = self._execute_data_splitting_stage(stage_config)
            elif stage_type == "model_trainer":
                result = self._execute_model_training_stage(stage_config)
            elif stage_type == "model_evaluator":
                result = self._execute_model_evaluation_stage(stage_config)
            elif stage_type == "model_validator":
                result = self._execute_model_validation_stage(stage_config)
            elif stage_type == "model_registry":
                result = self._execute_model_registration_stage(stage_config)
            else:
                logger.warning(f"Unknown stage type: {stage_type}")
                result = {}
            
            # Store stage result
            self.stage_results[stage_name] = result
            
            # Log stage metrics
            stage_duration = time.time() - stage_start_time
            mlflow.log_metric(f"stage_{stage_name}_duration", stage_duration)
            
            logger.info(f"Completed stage: {stage_name} in {stage_duration:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Stage {stage_name} failed: {e}")
            raise
    
    def _execute_data_ingestion_stage(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute data ingestion stage.
        
        Args:
            config: Stage configuration
            
        Returns:
            Stage execution results
        """
        logger.info("Executing data ingestion stage")
        
        # Load data from multiple sources
        data_sources = config.get('sources', [])
        loaded_data = {}
        
        for source in data_sources:
            source_type = source.get('type', 'unknown')
            source_name = source.get('name', source_type)
            
            try:
                if source_type == "kafka":
                    data = self._load_kafka_data(source)
                elif source_type == "postgresql":
                    data = self._load_postgresql_data(source)
                elif source_type == "mongodb":
                    data = self._load_mongodb_data(source)
                elif source_type == "s3":
                    data = self._load_s3_data(source)
                else:
                    logger.warning(f"Unknown data source type: {source_type}")
                    continue
                
                loaded_data[source_name] = data
                logger.info(f"Loaded {len(data)} records from {source_name}")
                
            except Exception as e:
                logger.error(f"Failed to load data from {source_name}: {e}")
                raise
        
        return {
            'loaded_data': loaded_data,
            'data_sources': [s.get('name', s.get('type')) for s in data_sources],
            'total_records': sum(len(data) for data in loaded_data.values())
        }
    
    def _execute_preprocessing_stage(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute data preprocessing stage.
        
        Args:
            config: Stage configuration
            
        Returns:
            Stage execution results
        """
        logger.info("Executing data preprocessing stage")
        
        # Get data from previous stage
        raw_data = self.stage_results.get('data_ingestion', {}).get('loaded_data', {})
        
        if not raw_data:
            raise ValueError("No data available for preprocessing")
        
        # Combine data from all sources
        combined_data = pd.concat(raw_data.values(), ignore_index=True)
        
        # Apply preprocessing steps
        cleaning_config = config.get('cleaning', {})
        normalization_config = config.get('normalization', {})
        feature_selection_config = config.get('feature_selection', {})
        
        # Data cleaning
        if cleaning_config.get('remove_duplicates', False):
            combined_data = self.preprocessors['cleaner'].remove_duplicates(combined_data)
        
        if cleaning_config.get('handle_missing_values'):
            method = cleaning_config['handle_missing_values']
            combined_data = self.preprocessors['cleaner'].handle_missing_values(combined_data, method=method)
        
        if cleaning_config.get('outlier_detection'):
            method = cleaning_config['outlier_detection']
            threshold = cleaning_config.get('outlier_threshold', 3.0)
            combined_data = self.preprocessors['outlier_detector'].detect_and_handle_outliers(
                combined_data, method=method, threshold=threshold
            )
        
        # Feature normalization
        if normalization_config.get('method'):
            method = normalization_config['method']
            fit_on_train = normalization_config.get('fit_on_train', True)
            combined_data = self.preprocessors['scaler'].fit_transform(combined_data, method=method)
        
        # Feature selection
        if feature_selection_config.get('enabled', False):
            method = feature_selection_config.get('method', 'mutual_information')
            n_features = feature_selection_config.get('n_features', 20)
            # Feature selection would be implemented here
        
        return {
            'preprocessed_data': combined_data,
            'cleaning_applied': cleaning_config,
            'normalization_applied': normalization_config,
            'feature_selection_applied': feature_selection_config
        }
    
    def _execute_feature_engineering_stage(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute feature engineering stage.
        
        Args:
            config: Stage configuration
            
        Returns:
            Stage execution results
        """
        logger.info("Executing feature engineering stage")
        
        # Get preprocessed data
        preprocessed_data = self.stage_results.get('data_preprocessing', {}).get('preprocessed_data')
        
        if preprocessed_data is None:
            raise ValueError("No preprocessed data available for feature engineering")
        
        # Apply feature engineering transformations
        transformations = config.get('transformations', [])
        engineered_features = preprocessed_data.copy()
        
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
        
        return {
            'engineered_features': engineered_features,
            'transformations_applied': transformations,
            'feature_count': len(engineered_features.columns)
        }
    
    def _execute_data_splitting_stage(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute data splitting stage.
        
        Args:
            config: Stage configuration
            
        Returns:
            Stage execution results
        """
        logger.info("Executing data splitting stage")
        
        # Get engineered features
        engineered_features = self.stage_results.get('feature_engineering', {}).get('engineered_features')
        
        if engineered_features is None:
            raise ValueError("No engineered features available for data splitting")
        
        # Split data
        splits = config.get('splits', {'train': 0.7, 'validation': 0.2, 'test': 0.1})
        strategy = config.get('strategy', 'time_series_split')
        random_state = config.get('random_state', 42)
        target_column = config.get('target_column')
        
        if strategy == "time_series_split":
            train_size = int(len(engineered_features) * splits['train'])
            val_size = int(len(engineered_features) * splits['validation'])
            
            train_data = engineered_features[:train_size]
            val_data = engineered_features[train_size:train_size + val_size]
            test_data = engineered_features[train_size + val_size:]
        else:
            # Random split
            from sklearn.model_selection import train_test_split
            
            train_data, temp_data = train_test_split(
                engineered_features, 
                test_size=1-splits['train'], 
                random_state=random_state,
                stratify=engineered_features[target_column] if target_column else None
            )
            
            val_data, test_data = train_test_split(
                temp_data,
                test_size=splits['test']/(splits['validation'] + splits['test']),
                random_state=random_state,
                stratify=temp_data[target_column] if target_column else None
            )
        
        return {
            'train_data': train_data,
            'validation_data': val_data,
            'test_data': test_data,
            'split_ratios': splits,
            'split_strategy': strategy
        }
    
    @abstractmethod
    def _execute_model_training_stage(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute model training stage. Must be implemented by subclasses.
        
        Args:
            config: Stage configuration
            
        Returns:
            Stage execution results
        """
        pass
    
    def _execute_model_evaluation_stage(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute model evaluation stage.
        
        Args:
            config: Stage configuration
            
        Returns:
            Stage execution results
        """
        logger.info("Executing model evaluation stage")
        
        # Get trained model and test data
        trained_model = self.stage_results.get('model_training', {}).get('trained_model')
        test_data = self.stage_results.get('data_splitting', {}).get('test_data')
        
        if trained_model is None or test_data is None:
            raise ValueError("No trained model or test data available for evaluation")
        
        # Evaluate model
        metrics = config.get('metrics', ['accuracy', 'precision', 'recall', 'f1'])
        evaluation_results = {}
        
        for metric in metrics:
            if metric == "accuracy":
                evaluation_results[metric] = self.evaluators['classification'].calculate_accuracy(trained_model, test_data)
            elif metric == "precision":
                evaluation_results[metric] = self.evaluators['classification'].calculate_precision(trained_model, test_data)
            elif metric == "recall":
                evaluation_results[metric] = self.evaluators['classification'].calculate_recall(trained_model, test_data)
            elif metric == "f1":
                evaluation_results[metric] = self.evaluators['classification'].calculate_f1_score(trained_model, test_data)
        
        # Log metrics to MLflow
        for metric_name, metric_value in evaluation_results.items():
            mlflow.log_metric(metric_name, metric_value)
        
        return {
            'evaluation_results': evaluation_results,
            'metrics_calculated': metrics
        }
    
    def _execute_model_validation_stage(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute model validation stage.
        
        Args:
            config: Stage configuration
            
        Returns:
            Stage execution results
        """
        logger.info("Executing model validation stage")
        
        # Get evaluation results
        evaluation_results = self.stage_results.get('model_evaluation', {}).get('evaluation_results', {})
        
        # Check requirements
        requirements = config.get('requirements', {})
        validation_results = {}
        
        for requirement, threshold in requirements.items():
            if requirement.endswith('_threshold'):
                metric_name = requirement.replace('_threshold', '')
                if metric_name in evaluation_results:
                    validation_results[requirement] = evaluation_results[metric_name] >= threshold
                else:
                    validation_results[requirement] = False
        
        # Overall validation
        validation_passed = all(validation_results.values())
        validation_results['overall_validation'] = validation_passed
        
        # Log validation results
        mlflow.log_param("validation_passed", validation_passed)
        for validation_name, validation_result in validation_results.items():
            mlflow.log_param(f"validation_{validation_name}", validation_result)
        
        return {
            'validation_results': validation_results,
            'validation_passed': validation_passed
        }
    
    def _execute_model_registration_stage(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute model registration stage.
        
        Args:
            config: Stage configuration
            
        Returns:
            Stage execution results
        """
        logger.info("Executing model registration stage")
        
        # Get trained model
        trained_model = self.stage_results.get('model_training', {}).get('trained_model')
        
        if trained_model is None:
            raise ValueError("No trained model available for registration")
        
        # Register model in MLflow
        registry_config = config.get('registry', 'mlflow')
        stage = config.get('stage', 'staging')
        metadata = config.get('metadata', {})
        tags = config.get('tags', {})
        
        # Log model
        if hasattr(trained_model, 'predict'):
            mlflow.sklearn.log_model(trained_model, "model")
        else:
            mlflow.tensorflow.log_model(trained_model, "model")
        
        # Log metadata and tags
        for key, value in metadata.items():
            mlflow.log_param(f"metadata_{key}", value)
        
        for key, value in tags.items():
            mlflow.set_tag(key, value)
        
        return {
            'registered_model': trained_model,
            'registry': registry_config,
            'stage': stage,
            'metadata': metadata,
            'tags': tags
        }
    
    def _finalize_pipeline(self):
        """Finalize pipeline execution."""
        logger.info("Finalizing pipeline execution")
        
        # Log final pipeline metrics
        total_duration = time.time() - self.pipeline_start_time
        mlflow.log_metric("total_pipeline_duration", total_duration)
        
        # Log stage durations
        for stage_name, stage_result in self.stage_results.items():
            if 'duration' in stage_result:
                mlflow.log_metric(f"stage_{stage_name}_duration", stage_result['duration'])
    
    def _handle_pipeline_failure(self, error: Exception):
        """Handle pipeline failure."""
        logger.error(f"Pipeline failed: {error}")
        
        # Log failure metrics
        mlflow.log_param("pipeline_failed", True)
        mlflow.log_param("failure_error", str(error))
        
        # Cleanup if needed
        # This would include cleanup of temporary files, resources, etc.
    
    def _log_pipeline_metrics(self):
        """Log pipeline execution metrics."""
        if self.pipeline_start_time and self.pipeline_end_time:
            total_duration = self.pipeline_end_time - self.pipeline_start_time
            logger.info(f"Pipeline total duration: {total_duration:.2f} seconds")
    
    def _get_pipeline_results(self) -> Dict[str, Any]:
        """Get comprehensive pipeline results."""
        return {
            'pipeline_name': self.pipeline_name,
            'pipeline_status': 'completed',
            'stage_results': self.stage_results,
            'total_duration': self.pipeline_end_time - self.pipeline_start_time if self.pipeline_end_time else None,
            'metadata': self.metadata
        }
    
    # Helper methods for data loading
    def _load_kafka_data(self, source_config: Dict[str, Any]) -> pd.DataFrame:
        """Load data from Kafka source."""
        # Implementation would depend on Kafka client
        logger.info(f"Loading data from Kafka topic: {source_config.get('topic')}")
        # Return mock data for now
        return pd.DataFrame()
    
    def _load_postgresql_data(self, source_config: Dict[str, Any]) -> pd.DataFrame:
        """Load data from PostgreSQL source."""
        # Implementation would use SQLAlchemy or similar
        logger.info(f"Loading data from PostgreSQL table: {source_config.get('table')}")
        # Return mock data for now
        return pd.DataFrame()
    
    def _load_mongodb_data(self, source_config: Dict[str, Any]) -> pd.DataFrame:
        """Load data from MongoDB source."""
        # Implementation would use pymongo
        logger.info(f"Loading data from MongoDB collection: {source_config.get('collection')}")
        # Return mock data for now
        return pd.DataFrame()
    
    def _load_s3_data(self, source_config: Dict[str, Any]) -> pd.DataFrame:
        """Load data from S3 source."""
        # Implementation would use boto3
        logger.info(f"Loading data from S3: {source_config.get('bucket')}/{source_config.get('key')}")
        # Return mock data for now
        return pd.DataFrame()
    
    # Helper methods for feature engineering
    def _apply_temporal_features(self, data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Apply temporal feature transformations."""
        # Implementation would use temporal feature engineering
        return data
    
    def _apply_statistical_features(self, data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Apply statistical feature transformations."""
        # Implementation would use statistical feature engineering
        return data
    
    def _apply_frequency_features(self, data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Apply frequency feature transformations."""
        # Implementation would use frequency feature engineering
        return data
    
    def _apply_cross_features(self, data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Apply cross feature transformations."""
        # Implementation would use cross feature engineering
        return data
