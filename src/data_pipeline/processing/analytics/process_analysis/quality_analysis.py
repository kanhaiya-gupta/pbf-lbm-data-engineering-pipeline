"""
Quality Analysis for PBF-LB/M Systems

This module provides specialized quality analysis capabilities for PBF-LB/M
additive manufacturing systems, including quality prediction, quality control,
and quality optimization.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import logging
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import warnings

logger = logging.getLogger(__name__)


@dataclass
class QualityAnalysisConfig:
    """Configuration for quality analysis."""
    
    # Model parameters
    model_type: str = "random_forest"  # "random_forest", "gradient_boosting"
    test_size: float = 0.2
    random_seed: Optional[int] = None
    
    # Quality thresholds
    quality_threshold: float = 0.8
    defect_threshold: float = 0.1
    
    # Analysis parameters
    confidence_level: float = 0.95


@dataclass
class QualityAnalysisResult:
    """Result of quality analysis."""
    
    success: bool
    method: str
    quality_metrics: Dict[str, float]
    quality_predictions: np.ndarray
    quality_classifications: np.ndarray
    model_performance: Dict[str, float]
    analysis_time: float = 0.0
    error_message: Optional[str] = None


class QualityAnalyzer:
    """
    Quality analyzer for PBF-LB/M systems.
    
    This class provides specialized quality analysis capabilities including
    quality prediction, quality control, and quality optimization for
    PBF-LB/M additive manufacturing.
    """
    
    def __init__(self, config: QualityAnalysisConfig = None):
        """Initialize the quality analyzer."""
        self.config = config or QualityAnalysisConfig()
        self.analysis_cache = {}
        
        # Set random seed
        if self.config.random_seed is not None:
            np.random.seed(self.config.random_seed)
        
        logger.info("Quality Analyzer initialized")
    
    def analyze_quality_prediction(
        self,
        process_data: pd.DataFrame,
        quality_target: str,
        feature_names: List[str] = None
    ) -> QualityAnalysisResult:
        """
        Perform quality prediction analysis.
        
        Args:
            process_data: DataFrame containing process data
            quality_target: Name of quality target variable
            feature_names: List of feature names (optional)
            
        Returns:
            QualityAnalysisResult: Quality prediction analysis results
        """
        try:
            start_time = datetime.now()
            
            if feature_names is None:
                feature_names = [col for col in process_data.columns if col != quality_target]
            
            # Prepare data
            X = process_data[feature_names].values
            y = process_data[quality_target].values
            
            # Handle missing values
            valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
            X = X[valid_mask]
            y = y[valid_mask]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.config.test_size, random_state=self.config.random_seed
            )
            
            # Select model
            if self.config.model_type == "random_forest":
                model = RandomForestRegressor(random_state=self.config.random_seed)
            elif self.config.model_type == "gradient_boosting":
                model = GradientBoostingRegressor(random_state=self.config.random_seed)
            else:
                raise ValueError(f"Unsupported model type: {self.config.model_type}")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate model performance
            model_performance = {
                'r2_score': r2_score(y_test, y_pred),
                'mse': mean_squared_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
            }
            
            # Calculate quality metrics
            quality_metrics = {
                'mean_quality': np.mean(y_pred),
                'std_quality': np.std(y_pred),
                'min_quality': np.min(y_pred),
                'max_quality': np.max(y_pred),
                'quality_range': np.max(y_pred) - np.min(y_pred)
            }
            
            # Classify quality
            quality_classifications = self._classify_quality(y_pred)
            
            # Calculate analysis time
            analysis_time = (datetime.now() - start_time).total_seconds()
            
            # Create result
            result = QualityAnalysisResult(
                success=True,
                method="QualityPrediction",
                quality_metrics=quality_metrics,
                quality_predictions=y_pred,
                quality_classifications=quality_classifications,
                model_performance=model_performance,
                analysis_time=analysis_time
            )
            
            # Cache result
            self._cache_result("quality_prediction", result)
            
            logger.info(f"Quality prediction analysis completed: {analysis_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Error in quality prediction analysis: {e}")
            return QualityAnalysisResult(
                success=False,
                method="QualityPrediction",
                quality_metrics={},
                quality_predictions=np.array([]),
                quality_classifications=np.array([]),
                model_performance={},
                error_message=str(e)
            )
    
    def _classify_quality(self, quality_values: np.ndarray) -> np.ndarray:
        """Classify quality values into categories."""
        classifications = np.zeros(len(quality_values), dtype=int)
        
        # High quality: >= quality_threshold
        classifications[quality_values >= self.config.quality_threshold] = 2
        
        # Medium quality: defect_threshold <= quality < quality_threshold
        medium_mask = (quality_values >= self.config.defect_threshold) & (quality_values < self.config.quality_threshold)
        classifications[medium_mask] = 1
        
        # Low quality (defects): < defect_threshold
        classifications[quality_values < self.config.defect_threshold] = 0
        
        return classifications
    
    def _cache_result(self, method: str, result: QualityAnalysisResult):
        """Cache analysis result."""
        cache_key = f"{method}_{hash(str(result.quality_metrics))}"
        self.analysis_cache[cache_key] = result
    
    def get_cached_result(self, method: str) -> Optional[QualityAnalysisResult]:
        """Get cached analysis result."""
        cache_key = f"{method}_default"
        return self.analysis_cache.get(cache_key)
    
    def clear_cache(self):
        """Clear analysis cache."""
        self.analysis_cache.clear()
    
    def get_analysis_statistics(self) -> Dict[str, Any]:
        """Get analysis statistics."""
        return {
            'cache_size': len(self.analysis_cache),
            'config': {
                'model_type': self.config.model_type,
                'test_size': self.config.test_size,
                'quality_threshold': self.config.quality_threshold,
                'defect_threshold': self.config.defect_threshold
            }
        }


class QualityPredictor(QualityAnalyzer):
    """Specialized quality predictor."""
    
    def __init__(self, config: QualityAnalysisConfig = None):
        super().__init__(config)
        self.method_name = "QualityPredictor"
    
    def predict(self, process_data: pd.DataFrame, quality_target: str, 
                feature_names: List[str] = None) -> QualityAnalysisResult:
        """Predict quality from process data."""
        return self.analyze_quality_prediction(process_data, quality_target, feature_names)
