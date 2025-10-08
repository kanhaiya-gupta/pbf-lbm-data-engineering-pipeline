"""
Time Series Analysis for PBF-LB/M Process Data

This module provides comprehensive time series analysis capabilities including
trend analysis, seasonality detection, and time series forecasting for
PBF-LB/M process monitoring data.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import logging
from datetime import datetime, timedelta
from scipy import stats
from scipy.signal import find_peaks, periodogram
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import warnings

logger = logging.getLogger(__name__)


@dataclass
class TimeSeriesConfig:
    """Configuration for time series analysis."""
    
    # Trend analysis parameters
    trend_method: str = "linear"  # "linear", "polynomial", "exponential"
    polynomial_degree: int = 2
    
    # Seasonality parameters
    seasonality_detection: bool = True
    min_periods: int = 2
    max_periods: int = 100
    
    # Forecasting parameters
    forecast_horizon: int = 10
    confidence_level: float = 0.95
    
    # Analysis parameters
    significance_level: float = 0.05
    random_seed: Optional[int] = None


@dataclass
class TimeSeriesResult:
    """Result of time series analysis."""
    
    success: bool
    method: str
    time_series_data: pd.Series
    trend_analysis: Dict[str, Any]
    seasonality_analysis: Dict[str, Any]
    forecasting_results: Dict[str, Any]
    analysis_time: float = 0.0
    error_message: Optional[str] = None


class TimeSeriesAnalyzer:
    """
    Time series analyzer for PBF-LB/M process data.
    
    This class provides comprehensive time series analysis capabilities including
    trend analysis, seasonality detection, and time series forecasting for
    understanding temporal patterns in PBF-LB/M process data.
    """
    
    def __init__(self, config: TimeSeriesConfig = None):
        """Initialize the time series analyzer."""
        self.config = config or TimeSeriesConfig()
        self.analysis_cache = {}
        
        # Set random seed
        if self.config.random_seed is not None:
            np.random.seed(self.config.random_seed)
        
        logger.info("Time Series Analyzer initialized")
    
    def analyze_trend(
        self,
        time_series: pd.Series,
        method: str = None
    ) -> TimeSeriesResult:
        """
        Perform trend analysis on time series data.
        
        Args:
            time_series: Input time series data
            method: Trend analysis method (optional)
            
        Returns:
            TimeSeriesResult: Trend analysis results
        """
        try:
            start_time = datetime.now()
            
            if method is None:
                method = self.config.trend_method
            
            # Prepare data
            y = time_series.values
            x = np.arange(len(y))
            
            # Perform trend analysis
            if method == "linear":
                trend_result = self._analyze_linear_trend(x, y)
            elif method == "polynomial":
                trend_result = self._analyze_polynomial_trend(x, y)
            elif method == "exponential":
                trend_result = self._analyze_exponential_trend(x, y)
            else:
                raise ValueError(f"Unsupported trend method: {method}")
            
            # Calculate analysis time
            analysis_time = (datetime.now() - start_time).total_seconds()
            
            # Create result
            result = TimeSeriesResult(
                success=True,
                method=f"Trend_{method}",
                time_series_data=time_series,
                trend_analysis=trend_result,
                seasonality_analysis={},
                forecasting_results={},
                analysis_time=analysis_time
            )
            
            # Cache result
            self._cache_result("trend", result)
            
            logger.info(f"Trend analysis completed: {analysis_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Error in trend analysis: {e}")
            return TimeSeriesResult(
                success=False,
                method=f"Trend_{method}",
                time_series_data=time_series,
                trend_analysis={},
                seasonality_analysis={},
                forecasting_results={},
                error_message=str(e)
            )
    
    def analyze_seasonality(
        self,
        time_series: pd.Series
    ) -> TimeSeriesResult:
        """
        Perform seasonality analysis on time series data.
        
        Args:
            time_series: Input time series data
            
        Returns:
            TimeSeriesResult: Seasonality analysis results
        """
        try:
            start_time = datetime.now()
            
            # Prepare data
            y = time_series.values
            
            # Detect seasonality
            seasonality_result = self._detect_seasonality(y)
            
            # Calculate analysis time
            analysis_time = (datetime.now() - start_time).total_seconds()
            
            # Create result
            result = TimeSeriesResult(
                success=True,
                method="Seasonality",
                time_series_data=time_series,
                trend_analysis={},
                seasonality_analysis=seasonality_result,
                forecasting_results={},
                analysis_time=analysis_time
            )
            
            # Cache result
            self._cache_result("seasonality", result)
            
            logger.info(f"Seasonality analysis completed: {analysis_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Error in seasonality analysis: {e}")
            return TimeSeriesResult(
                success=False,
                method="Seasonality",
                time_series_data=time_series,
                trend_analysis={},
                seasonality_analysis={},
                forecasting_results={},
                error_message=str(e)
            )
    
    def analyze_forecasting(
        self,
        time_series: pd.Series,
        forecast_horizon: int = None
    ) -> TimeSeriesResult:
        """
        Perform time series forecasting.
        
        Args:
            time_series: Input time series data
            forecast_horizon: Number of periods to forecast (optional)
            
        Returns:
            TimeSeriesResult: Forecasting analysis results
        """
        try:
            start_time = datetime.now()
            
            if forecast_horizon is None:
                forecast_horizon = self.config.forecast_horizon
            
            # Prepare data
            y = time_series.values
            
            # Perform forecasting
            forecast_result = self._perform_forecasting(y, forecast_horizon)
            
            # Calculate analysis time
            analysis_time = (datetime.now() - start_time).total_seconds()
            
            # Create result
            result = TimeSeriesResult(
                success=True,
                method="Forecasting",
                time_series_data=time_series,
                trend_analysis={},
                seasonality_analysis={},
                forecasting_results=forecast_result,
                analysis_time=analysis_time
            )
            
            # Cache result
            self._cache_result("forecasting", result)
            
            logger.info(f"Forecasting analysis completed: {analysis_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Error in forecasting analysis: {e}")
            return TimeSeriesResult(
                success=False,
                method="Forecasting",
                time_series_data=time_series,
                trend_analysis={},
                seasonality_analysis={},
                forecasting_results={},
                error_message=str(e)
            )
    
    def _analyze_linear_trend(self, x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Analyze linear trend."""
        # Fit linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        # Calculate trend line
        trend_line = slope * x + intercept
        
        # Calculate residuals
        residuals = y - trend_line
        
        # Calculate trend statistics
        trend_stats = {
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_value ** 2,
            'p_value': p_value,
            'std_error': std_err,
            'trend_line': trend_line,
            'residuals': residuals,
            'trend_direction': 'increasing' if slope > 0 else 'decreasing' if slope < 0 else 'stable'
        }
        
        return trend_stats
    
    def _analyze_polynomial_trend(self, x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Analyze polynomial trend."""
        # Create polynomial features
        poly_features = PolynomialFeatures(degree=self.config.polynomial_degree)
        x_poly = poly_features.fit_transform(x.reshape(-1, 1))
        
        # Fit polynomial regression
        model = LinearRegression()
        model.fit(x_poly, y)
        
        # Calculate trend line
        trend_line = model.predict(x_poly)
        
        # Calculate residuals
        residuals = y - trend_line
        
        # Calculate R-squared
        r_squared = model.score(x_poly, y)
        
        # Calculate trend statistics
        trend_stats = {
            'coefficients': model.coef_,
            'intercept': model.intercept_,
            'r_squared': r_squared,
            'trend_line': trend_line,
            'residuals': residuals,
            'polynomial_degree': self.config.polynomial_degree
        }
        
        return trend_stats
    
    def _analyze_exponential_trend(self, x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Analyze exponential trend."""
        # Ensure positive values for exponential fitting
        y_positive = np.abs(y) + 1e-10
        
        # Fit exponential trend: y = a * exp(b * x)
        # Linearize: ln(y) = ln(a) + b * x
        log_y = np.log(y_positive)
        
        # Fit linear regression to log-transformed data
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, log_y)
        
        # Convert back to exponential parameters
        a = np.exp(intercept)
        b = slope
        
        # Calculate trend line
        trend_line = a * np.exp(b * x)
        
        # Calculate residuals
        residuals = y - trend_line
        
        # Calculate trend statistics
        trend_stats = {
            'a': a,
            'b': b,
            'r_squared': r_value ** 2,
            'p_value': p_value,
            'std_error': std_err,
            'trend_line': trend_line,
            'residuals': residuals,
            'trend_direction': 'exponential_growth' if b > 0 else 'exponential_decay' if b < 0 else 'stable'
        }
        
        return trend_stats
    
    def _detect_seasonality(self, y: np.ndarray) -> Dict[str, Any]:
        """Detect seasonality in time series."""
        # Calculate periodogram
        frequencies, power = periodogram(y)
        
        # Find dominant frequencies
        peaks, properties = find_peaks(power, height=np.mean(power))
        
        # Calculate periods
        periods = 1 / frequencies[peaks]
        
        # Filter periods within reasonable range
        valid_periods = periods[(periods >= self.config.min_periods) & 
                               (periods <= self.config.max_periods)]
        
        # Calculate seasonality strength
        seasonality_strength = np.max(power[peaks]) / np.mean(power)
        
        # Detect if seasonality is significant
        is_seasonal = seasonality_strength > 2.0 and len(valid_periods) > 0
        
        seasonality_result = {
            'is_seasonal': is_seasonal,
            'seasonality_strength': seasonality_strength,
            'dominant_periods': valid_periods,
            'frequencies': frequencies,
            'power_spectrum': power,
            'peaks': peaks
        }
        
        return seasonality_result
    
    def _perform_forecasting(self, y: np.ndarray, forecast_horizon: int) -> Dict[str, Any]:
        """Perform time series forecasting."""
        # Simple linear trend forecasting
        x = np.arange(len(y))
        
        # Fit linear trend
        slope, intercept, _, _, _ = stats.linregress(x, y)
        
        # Generate forecast
        future_x = np.arange(len(y), len(y) + forecast_horizon)
        forecast_values = slope * future_x + intercept
        
        # Calculate confidence intervals
        residuals = y - (slope * x + intercept)
        residual_std = np.std(residuals)
        
        # 95% confidence interval
        confidence_interval = 1.96 * residual_std
        
        forecast_result = {
            'forecast_values': forecast_values,
            'forecast_horizon': forecast_horizon,
            'confidence_interval': confidence_interval,
            'lower_bound': forecast_values - confidence_interval,
            'upper_bound': forecast_values + confidence_interval,
            'trend_slope': slope,
            'trend_intercept': intercept
        }
        
        return forecast_result
    
    def _cache_result(self, method: str, result: TimeSeriesResult):
        """Cache analysis result."""
        cache_key = f"{method}_{hash(str(result.time_series_data.index))}"
        self.analysis_cache[cache_key] = result
    
    def get_cached_result(self, method: str, time_series: pd.Series) -> Optional[TimeSeriesResult]:
        """Get cached analysis result."""
        cache_key = f"{method}_{hash(str(time_series.index))}"
        return self.analysis_cache.get(cache_key)
    
    def clear_cache(self):
        """Clear analysis cache."""
        self.analysis_cache.clear()
    
    def get_analysis_statistics(self) -> Dict[str, Any]:
        """Get analysis statistics."""
        return {
            'cache_size': len(self.analysis_cache),
            'config': {
                'trend_method': self.config.trend_method,
                'polynomial_degree': self.config.polynomial_degree,
                'seasonality_detection': self.config.seasonality_detection,
                'forecast_horizon': self.config.forecast_horizon,
                'confidence_level': self.config.confidence_level
            }
        }


class TrendAnalyzer(TimeSeriesAnalyzer):
    """Specialized trend analyzer."""
    
    def __init__(self, config: TimeSeriesConfig = None):
        super().__init__(config)
        self.method_name = "Trend"
    
    def analyze(self, time_series: pd.Series, method: str = None) -> TimeSeriesResult:
        """Perform trend analysis."""
        return self.analyze_trend(time_series, method)


class SeasonalityAnalyzer(TimeSeriesAnalyzer):
    """Specialized seasonality analyzer."""
    
    def __init__(self, config: TimeSeriesConfig = None):
        super().__init__(config)
        self.method_name = "Seasonality"
    
    def analyze(self, time_series: pd.Series) -> TimeSeriesResult:
        """Perform seasonality analysis."""
        return self.analyze_seasonality(time_series)
