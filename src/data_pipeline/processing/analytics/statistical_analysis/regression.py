"""
Regression Analysis for PBF-LB/M Process Data

This module provides comprehensive regression analysis capabilities including
linear regression, polynomial regression, and advanced regression methods
for PBF-LB/M process data analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import logging
from datetime import datetime
from scipy import stats
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score, GridSearchCV
import warnings

logger = logging.getLogger(__name__)


@dataclass
class RegressionConfig:
    """Configuration for regression analysis."""
    
    # Model parameters
    model_type: str = "linear"  # "linear", "polynomial", "ridge", "lasso", "elastic_net"
    polynomial_degree: int = 2
    regularization_alpha: float = 1.0
    l1_ratio: float = 0.5  # For elastic net
    
    # Cross-validation parameters
    cv_folds: int = 5
    scoring: str = "r2"
    
    # Analysis parameters
    confidence_level: float = 0.95
    significance_level: float = 0.05
    random_seed: Optional[int] = None


@dataclass
class RegressionResult:
    """Result of regression analysis."""
    
    success: bool
    method: str
    feature_names: List[str]
    target_name: str
    model_coefficients: Dict[str, float]
    model_metrics: Dict[str, float]
    predictions: np.ndarray
    residuals: np.ndarray
    confidence_intervals: Dict[str, Tuple[float, float]]
    analysis_time: float = 0.0
    error_message: Optional[str] = None


class RegressionAnalyzer:
    """
    Regression analyzer for PBF-LB/M process data.
    
    This class provides comprehensive regression analysis capabilities including
    linear regression, polynomial regression, and regularized regression methods
    for understanding relationships in PBF-LB/M process data.
    """
    
    def __init__(self, config: RegressionConfig = None):
        """Initialize the regression analyzer."""
        self.config = config or RegressionConfig()
        self.analysis_cache = {}
        
        # Set random seed
        if self.config.random_seed is not None:
            np.random.seed(self.config.random_seed)
        
        logger.info("Regression Analyzer initialized")
    
    def analyze_linear_regression(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_names: List[str] = None
    ) -> RegressionResult:
        """
        Perform linear regression analysis.
        
        Args:
            X: Input features as DataFrame
            y: Target variable as Series
            feature_names: List of feature names (optional)
            
        Returns:
            RegressionResult: Linear regression analysis results
        """
        try:
            start_time = datetime.now()
            
            if feature_names is None:
                feature_names = list(X.columns)
            
            # Prepare data
            X_data = X[feature_names].values
            y_data = y.values
            
            # Handle missing values
            X_data, y_data = self._handle_missing_values(X_data, y_data)
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_data)
            
            # Fit linear regression model
            model = LinearRegression()
            model.fit(X_scaled, y_data)
            
            # Make predictions
            predictions = model.predict(X_scaled)
            residuals = y_data - predictions
            
            # Calculate model metrics
            model_metrics = self._calculate_model_metrics(y_data, predictions)
            
            # Calculate confidence intervals
            confidence_intervals = self._calculate_confidence_intervals(
                X_scaled, y_data, predictions, residuals
            )
            
            # Create coefficients dictionary
            model_coefficients = {
                'intercept': model.intercept_,
                **{name: coef for name, coef in zip(feature_names, model.coef_)}
            }
            
            # Calculate analysis time
            analysis_time = (datetime.now() - start_time).total_seconds()
            
            # Create result
            result = RegressionResult(
                success=True,
                method="LinearRegression",
                feature_names=feature_names,
                target_name=y.name if hasattr(y, 'name') else 'target',
                model_coefficients=model_coefficients,
                model_metrics=model_metrics,
                predictions=predictions,
                residuals=residuals,
                confidence_intervals=confidence_intervals,
                analysis_time=analysis_time
            )
            
            # Cache result
            self._cache_result("linear_regression", result)
            
            logger.info(f"Linear regression analysis completed: {analysis_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Error in linear regression analysis: {e}")
            return RegressionResult(
                success=False,
                method="LinearRegression",
                feature_names=feature_names or [],
                target_name=y.name if hasattr(y, 'name') else 'target',
                model_coefficients={},
                model_metrics={},
                predictions=np.array([]),
                residuals=np.array([]),
                confidence_intervals={},
                error_message=str(e)
            )
    
    def analyze_polynomial_regression(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_names: List[str] = None,
        degree: int = None
    ) -> RegressionResult:
        """
        Perform polynomial regression analysis.
        
        Args:
            X: Input features as DataFrame
            y: Target variable as Series
            feature_names: List of feature names (optional)
            degree: Polynomial degree (optional)
            
        Returns:
            RegressionResult: Polynomial regression analysis results
        """
        try:
            start_time = datetime.now()
            
            if feature_names is None:
                feature_names = list(X.columns)
            
            if degree is None:
                degree = self.config.polynomial_degree
            
            # Prepare data
            X_data = X[feature_names].values
            y_data = y.values
            
            # Handle missing values
            X_data, y_data = self._handle_missing_values(X_data, y_data)
            
            # Create polynomial features
            poly_features = PolynomialFeatures(degree=degree, include_bias=False)
            X_poly = poly_features.fit_transform(X_data)
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_poly)
            
            # Fit polynomial regression model
            model = LinearRegression()
            model.fit(X_scaled, y_data)
            
            # Make predictions
            predictions = model.predict(X_scaled)
            residuals = y_data - predictions
            
            # Calculate model metrics
            model_metrics = self._calculate_model_metrics(y_data, predictions)
            
            # Calculate confidence intervals
            confidence_intervals = self._calculate_confidence_intervals(
                X_scaled, y_data, predictions, residuals
            )
            
            # Create coefficients dictionary
            poly_feature_names = poly_features.get_feature_names_out(feature_names)
            model_coefficients = {
                'intercept': model.intercept_,
                **{name: coef for name, coef in zip(poly_feature_names, model.coef_)}
            }
            
            # Calculate analysis time
            analysis_time = (datetime.now() - start_time).total_seconds()
            
            # Create result
            result = RegressionResult(
                success=True,
                method="PolynomialRegression",
                feature_names=feature_names,
                target_name=y.name if hasattr(y, 'name') else 'target',
                model_coefficients=model_coefficients,
                model_metrics=model_metrics,
                predictions=predictions,
                residuals=residuals,
                confidence_intervals=confidence_intervals,
                analysis_time=analysis_time
            )
            
            # Cache result
            self._cache_result("polynomial_regression", result)
            
            logger.info(f"Polynomial regression analysis completed: {analysis_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Error in polynomial regression analysis: {e}")
            return RegressionResult(
                success=False,
                method="PolynomialRegression",
                feature_names=feature_names or [],
                target_name=y.name if hasattr(y, 'name') else 'target',
                model_coefficients={},
                model_metrics={},
                predictions=np.array([]),
                residuals=np.array([]),
                confidence_intervals={},
                error_message=str(e)
            )
    
    def analyze_regularized_regression(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_names: List[str] = None,
        model_type: str = None
    ) -> RegressionResult:
        """
        Perform regularized regression analysis.
        
        Args:
            X: Input features as DataFrame
            y: Target variable as Series
            feature_names: List of feature names (optional)
            model_type: Regularization type ("ridge", "lasso", "elastic_net")
            
        Returns:
            RegressionResult: Regularized regression analysis results
        """
        try:
            start_time = datetime.now()
            
            if feature_names is None:
                feature_names = list(X.columns)
            
            if model_type is None:
                model_type = self.config.model_type
            
            # Prepare data
            X_data = X[feature_names].values
            y_data = y.values
            
            # Handle missing values
            X_data, y_data = self._handle_missing_values(X_data, y_data)
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_data)
            
            # Select model
            if model_type == "ridge":
                model = Ridge(alpha=self.config.regularization_alpha, random_state=self.config.random_seed)
            elif model_type == "lasso":
                model = Lasso(alpha=self.config.regularization_alpha, random_state=self.config.random_seed)
            elif model_type == "elastic_net":
                model = ElasticNet(alpha=self.config.regularization_alpha, l1_ratio=self.config.l1_ratio, 
                                 random_state=self.config.random_seed)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            # Fit model
            model.fit(X_scaled, y_data)
            
            # Make predictions
            predictions = model.predict(X_scaled)
            residuals = y_data - predictions
            
            # Calculate model metrics
            model_metrics = self._calculate_model_metrics(y_data, predictions)
            
            # Calculate confidence intervals
            confidence_intervals = self._calculate_confidence_intervals(
                X_scaled, y_data, predictions, residuals
            )
            
            # Create coefficients dictionary
            model_coefficients = {
                'intercept': model.intercept_,
                **{name: coef for name, coef in zip(feature_names, model.coef_)}
            }
            
            # Calculate analysis time
            analysis_time = (datetime.now() - start_time).total_seconds()
            
            # Create result
            result = RegressionResult(
                success=True,
                method=f"RegularizedRegression_{model_type}",
                feature_names=feature_names,
                target_name=y.name if hasattr(y, 'name') else 'target',
                model_coefficients=model_coefficients,
                model_metrics=model_metrics,
                predictions=predictions,
                residuals=residuals,
                confidence_intervals=confidence_intervals,
                analysis_time=analysis_time
            )
            
            # Cache result
            self._cache_result("regularized_regression", result)
            
            logger.info(f"Regularized regression analysis completed: {analysis_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Error in regularized regression analysis: {e}")
            return RegressionResult(
                success=False,
                method=f"RegularizedRegression_{model_type}",
                feature_names=feature_names or [],
                target_name=y.name if hasattr(y, 'name') else 'target',
                model_coefficients={},
                model_metrics={},
                predictions=np.array([]),
                residuals=np.array([]),
                confidence_intervals={},
                error_message=str(e)
            )
    
    def _handle_missing_values(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Handle missing values in data."""
        # Remove rows with missing values
        valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        return X[valid_mask], y[valid_mask]
    
    def _calculate_model_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate model performance metrics."""
        return {
            'r2_score': r2_score(y_true, y_pred),
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        }
    
    def _calculate_confidence_intervals(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        predictions: np.ndarray, 
        residuals: np.ndarray
    ) -> Dict[str, Tuple[float, float]]:
        """Calculate confidence intervals for predictions."""
        # Calculate standard error
        n = len(y)
        p = X.shape[1]
        mse = np.mean(residuals**2)
        
        # Calculate confidence interval for mean prediction
        se_mean = np.sqrt(mse * (1/n + np.sum((X - np.mean(X, axis=0))**2, axis=1) / np.sum((X - np.mean(X, axis=0))**2)))
        t_value = stats.t.ppf(1 - (1 - self.config.confidence_level) / 2, n - p - 1)
        
        ci_lower = predictions - t_value * se_mean
        ci_upper = predictions + t_value * se_mean
        
        return {
            'confidence_interval': (np.mean(ci_lower), np.mean(ci_upper)),
            'prediction_interval': (np.min(ci_lower), np.max(ci_upper))
        }
    
    def _cache_result(self, method: str, result: RegressionResult):
        """Cache analysis result."""
        cache_key = f"{method}_{hash(str(result.feature_names))}"
        self.analysis_cache[cache_key] = result
    
    def get_cached_result(self, method: str, feature_names: List[str]) -> Optional[RegressionResult]:
        """Get cached analysis result."""
        cache_key = f"{method}_{hash(str(feature_names))}"
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
                'polynomial_degree': self.config.polynomial_degree,
                'regularization_alpha': self.config.regularization_alpha,
                'cv_folds': self.config.cv_folds,
                'scoring': self.config.scoring
            }
        }


class LinearRegression(RegressionAnalyzer):
    """Specialized linear regression analyzer."""
    
    def __init__(self, config: RegressionConfig = None):
        super().__init__(config)
        self.method_name = "LinearRegression"
    
    def analyze(self, X: pd.DataFrame, y: pd.Series, feature_names: List[str] = None) -> RegressionResult:
        """Perform linear regression analysis."""
        return self.analyze_linear_regression(X, y, feature_names)


class PolynomialRegression(RegressionAnalyzer):
    """Specialized polynomial regression analyzer."""
    
    def __init__(self, config: RegressionConfig = None):
        super().__init__(config)
        self.method_name = "PolynomialRegression"
    
    def analyze(self, X: pd.DataFrame, y: pd.Series, feature_names: List[str] = None, 
                degree: int = None) -> RegressionResult:
        """Perform polynomial regression analysis."""
        return self.analyze_polynomial_regression(X, y, feature_names, degree)
