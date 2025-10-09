"""
Time Series Feature Engineering

This module extracts and engineers features from time series data for PBF-LB/M processes.
Integrates with YAML configuration for feature definitions and validation rules.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import logging
from ..config.config_manager import ConfigManager

logger = logging.getLogger(__name__)


class TimeSeriesFeatures:
    """
    Feature engineering for time series data in PBF-LB/M processes.
    
    Extracts features from temporal patterns, trends, seasonality, and cycles
    based on YAML configuration.
    """
    
    def __init__(self, config_manager: Optional[ConfigManager] = None):
        """
        Initialize time series feature engineering.
        
        Args:
            config_manager: Configuration manager for loading feature definitions
        """
        self.config_manager = config_manager or ConfigManager()
        self.feature_config = self._load_feature_config()
        self.feature_definitions = self.feature_config.get('feature_definitions', {})
        self.validation_rules = self.feature_config.get('validation_rules', {})
        
    def _load_feature_config(self) -> Dict[str, Any]:
        """Load time series feature configuration from YAML."""
        try:
            return self.config_manager.load_feature_config('temporal_features/time_series_features')
        except Exception as e:
            logger.warning(f"Could not load time series config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for time series features."""
        return {
            'feature_definitions': {
                'trend_features': {
                    'aggregations': ['slope', 'intercept', 'r_squared'],
                    'derived': ['trend_strength', 'trend_direction']
                },
                'seasonality_features': {
                    'aggregations': ['seasonal_amplitude', 'seasonal_phase'],
                    'derived': ['seasonal_strength', 'seasonal_period']
                },
                'cyclical_features': {
                    'aggregations': ['cycle_amplitude', 'cycle_frequency'],
                    'derived': ['cycle_strength', 'cycle_period']
                }
            },
            'validation_rules': {
                'frequency_range': [0.001, 1000],  # Hz
                'period_range': [0.001, 10000],  # seconds
                'amplitude_range': [0, 1000]
            }
        }
    
    def extract_trend_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract trend features from time series data.
        
        Args:
            data: DataFrame with time series data
            
        Returns:
            DataFrame with trend features
        """
        features = pd.DataFrame(index=data.index)
        
        # Get numeric columns for trend analysis
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col in data.columns:
                series = data[col]
                
                # Linear trend features
                x = np.arange(len(series))
                valid_idx = ~series.isna()
                
                if valid_idx.sum() > 2:  # Need at least 3 points for trend
                    x_valid = x[valid_idx]
                    y_valid = series[valid_idx]
                    
                    # Linear regression
                    coeffs = np.polyfit(x_valid, y_valid, 1)
                    slope, intercept = coeffs
                    
                    features[f'{col}_trend_slope'] = slope
                    features[f'{col}_trend_intercept'] = intercept
                    
                    # R-squared
                    y_pred = slope * x_valid + intercept
                    ss_res = np.sum((y_valid - y_pred) ** 2)
                    ss_tot = np.sum((y_valid - np.mean(y_valid)) ** 2)
                    r_squared = 1 - (ss_res / (ss_tot + 1e-6))
                    features[f'{col}_trend_r_squared'] = r_squared
                    
                    # Trend strength
                    features[f'{col}_trend_strength'] = abs(slope) * r_squared
                    
                    # Trend direction
                    features[f'{col}_trend_direction'] = np.sign(slope)
                    
                    # Trend categories
                    features[f'{col}_trend_category'] = pd.cut(features[f'{col}_trend_slope'], 
                                                           bins=[-np.inf, -0.1, 0.1, np.inf], 
                                                           labels=['decreasing', 'stable', 'increasing'])
        
        return features
    
    def extract_seasonality_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract seasonality features from time series data.
        
        Args:
            data: DataFrame with time series data
            
        Returns:
            DataFrame with seasonality features
        """
        features = pd.DataFrame(index=data.index)
        
        # Get numeric columns for seasonality analysis
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col in data.columns:
                series = data[col]
                
                # Remove trend first
                x = np.arange(len(series))
                valid_idx = ~series.isna()
                
                if valid_idx.sum() > 10:  # Need sufficient data for seasonality
                    x_valid = x[valid_idx]
                    y_valid = series[valid_idx]
                    
                    # Detrend
                    coeffs = np.polyfit(x_valid, y_valid, 1)
                    trend = coeffs[0] * x_valid + coeffs[1]
                    detrended = y_valid - trend
                    
                    # Seasonal decomposition (simplified)
                    # Assume seasonal period based on data length
                    n = len(detrended)
                    seasonal_periods = [n//4, n//2, n//3, n//6]  # Common seasonal periods
                    
                    for period in seasonal_periods:
                        if period > 2 and period < n//2:
                            # Calculate seasonal component
                            seasonal = np.zeros_like(detrended)
                            for i in range(period):
                                indices = np.arange(i, n, period)
                                if len(indices) > 1:
                                    seasonal[indices] = np.mean(detrended[indices])
                            
                            # Seasonal amplitude
                            seasonal_amplitude = np.std(seasonal)
                            features[f'{col}_seasonal_amplitude_{period}'] = seasonal_amplitude
                            
                            # Seasonal strength
                            total_variance = np.var(detrended)
                            seasonal_strength = seasonal_amplitude / (np.sqrt(total_variance) + 1e-6)
                            features[f'{col}_seasonal_strength_{period}'] = seasonal_strength
                            
                            # Seasonal phase (simplified)
                            seasonal_phase = np.argmax(seasonal[:period])
                            features[f'{col}_seasonal_phase_{period}'] = seasonal_phase
        
        return features
    
    def extract_cyclical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract cyclical features from time series data.
        
        Args:
            data: DataFrame with time series data
            
        Returns:
            DataFrame with cyclical features
        """
        features = pd.DataFrame(index=data.index)
        
        # Get numeric columns for cyclical analysis
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col in data.columns:
                series = data[col]
                
                valid_idx = ~series.isna()
                if valid_idx.sum() > 20:  # Need sufficient data for cycle analysis
                    y_valid = series[valid_idx]
                    
                    # Autocorrelation analysis
                    autocorr = np.correlate(y_valid, y_valid, mode='full')
                    autocorr = autocorr[autocorr.size // 2:]
                    autocorr = autocorr / autocorr[0]  # Normalize
                    
                    # Find peaks in autocorrelation (potential cycle periods)
                    from scipy.signal import find_peaks
                    peaks, _ = find_peaks(autocorr[1:], height=0.1, distance=5)
                    
                    if len(peaks) > 0:
                        # Primary cycle period
                        primary_cycle = peaks[0] + 1
                        features[f'{col}_cycle_period'] = primary_cycle
                        
                        # Cycle amplitude
                        cycle_amplitude = autocorr[primary_cycle]
                        features[f'{col}_cycle_amplitude'] = cycle_amplitude
                        
                        # Cycle frequency
                        cycle_frequency = 1 / (primary_cycle + 1e-6)
                        features[f'{col}_cycle_frequency'] = cycle_frequency
                        
                        # Cycle strength
                        cycle_strength = cycle_amplitude
                        features[f'{col}_cycle_strength'] = cycle_strength
                        
                        # Cycle categories
                        features[f'{col}_cycle_category'] = pd.cut(features[f'{col}_cycle_strength'], 
                                                               bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0], 
                                                               labels=['weak', 'moderate', 'strong', 'very_strong', 'dominant'])
        
        return features
    
    def extract_stationarity_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract stationarity features from time series data.
        
        Args:
            data: DataFrame with time series data
            
        Returns:
            DataFrame with stationarity features
        """
        features = pd.DataFrame(index=data.index)
        
        # Get numeric columns for stationarity analysis
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col in data.columns:
                series = data[col]
                
                valid_idx = ~series.isna()
                if valid_idx.sum() > 10:
                    y_valid = series[valid_idx]
                    
                    # Rolling statistics for stationarity test
                    window_size = min(20, len(y_valid) // 4)
                    if window_size > 2:
                        rolling_mean = pd.Series(y_valid).rolling(window=window_size).mean()
                        rolling_std = pd.Series(y_valid).rolling(window=window_size).std()
                        
                        # Mean stationarity
                        mean_stationarity = 1 / (rolling_mean.std() + 1e-6)
                        features[f'{col}_mean_stationarity'] = mean_stationarity
                        
                        # Variance stationarity
                        variance_stationarity = 1 / (rolling_std.std() + 1e-6)
                        features[f'{col}_variance_stationarity'] = variance_stationarity
                        
                        # Overall stationarity
                        overall_stationarity = (mean_stationarity + variance_stationarity) / 2
                        features[f'{col}_overall_stationarity'] = overall_stationarity
                        
                        # Stationarity categories
                        features[f'{col}_stationarity_category'] = pd.cut(overall_stationarity, 
                                                                       bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0], 
                                                                       labels=['non_stationary', 'weak_stationary', 'moderate_stationary', 'strong_stationary', 'very_stationary'])
        
        return features
    
    def extract_volatility_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract volatility features from time series data.
        
        Args:
            data: DataFrame with time series data
            
        Returns:
            DataFrame with volatility features
        """
        features = pd.DataFrame(index=data.index)
        
        # Get numeric columns for volatility analysis
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col in data.columns:
                series = data[col]
                
                # Rolling volatility
                for window in [5, 10, 20]:
                    if len(series) > window:
                        rolling_vol = series.rolling(window=window).std()
                        features[f'{col}_volatility_{window}'] = rolling_vol
                        features[f'{col}_volatility_{window}_mean'] = rolling_vol.rolling(window=10).mean()
                        
                        # Volatility categories
                        features[f'{col}_volatility_{window}_category'] = pd.cut(rolling_vol, 
                                                                              bins=[0, 0.1, 0.3, 0.5, 0.7, 1.0], 
                                                                              labels=['low', 'moderate', 'medium', 'high', 'very_high'])
                
                # GARCH-like volatility (simplified)
                returns = series.pct_change().dropna()
                if len(returns) > 10:
                    # Exponentially weighted volatility
                    ew_vol = returns.ewm(span=10).std()
                    features[f'{col}_ew_volatility'] = ew_vol
                    
                    # Volatility clustering
                    vol_clustering = returns.rolling(window=10).apply(lambda x: (x > 0).sum() / len(x))
                    features[f'{col}_volatility_clustering'] = vol_clustering
        
        return features
    
    def extract_autocorrelation_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract autocorrelation features from time series data.
        
        Args:
            data: DataFrame with time series data
            
        Returns:
            DataFrame with autocorrelation features
        """
        features = pd.DataFrame(index=data.index)
        
        # Get numeric columns for autocorrelation analysis
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col in data.columns:
                series = data[col]
                
                valid_idx = ~series.isna()
                if valid_idx.sum() > 20:
                    y_valid = series[valid_idx]
                    
                    # Calculate autocorrelation for different lags
                    for lag in [1, 2, 5, 10]:
                        if len(y_valid) > lag:
                            autocorr = np.corrcoef(y_valid[:-lag], y_valid[lag:])[0, 1]
                            if not np.isnan(autocorr):
                                features[f'{col}_autocorr_lag_{lag}'] = autocorr
                    
                    # First significant autocorrelation
                    autocorrs = []
                    for lag in range(1, min(20, len(y_valid)//2)):
                        if len(y_valid) > lag:
                            autocorr = np.corrcoef(y_valid[:-lag], y_valid[lag:])[0, 1]
                            if not np.isnan(autocorr):
                                autocorrs.append(autocorr)
                    
                    if len(autocorrs) > 0:
                        features[f'{col}_max_autocorr'] = max(autocorrs)
                        features[f'{col}_min_autocorr'] = min(autocorrs)
                        features[f'{col}_autocorr_range'] = max(autocorrs) - min(autocorrs)
        
        return features
    
    def extract_all_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract all time series features.
        
        Args:
            data: DataFrame with time series data
            
        Returns:
            DataFrame with all extracted features
        """
        logger.info("Extracting time series features...")
        
        # Validate input data
        self._validate_input_data(data)
        
        # Extract different feature groups
        feature_groups = [
            self.extract_trend_features(data),
            self.extract_seasonality_features(data),
            self.extract_cyclical_features(data),
            self.extract_stationarity_features(data),
            self.extract_volatility_features(data),
            self.extract_autocorrelation_features(data)
        ]
        
        # Combine all features
        all_features = pd.concat(feature_groups, axis=1)
        
        # Clean and validate features
        all_features = self._clean_features(all_features)
        
        logger.info(f"Extracted {len(all_features.columns)} time series features")
        return all_features
    
    def _validate_input_data(self, data: pd.DataFrame) -> None:
        """Validate input data against configuration rules."""
        validation_rules = self.validation_rules
        
        # Validate frequency data
        if 'frequency' in data.columns:
            freq_range = validation_rules.get('frequency_range', [0.001, 1000])
            invalid_freq = (data['frequency'] < freq_range[0]) | (data['frequency'] > freq_range[1])
            if invalid_freq.any():
                logger.warning(f"Found {invalid_freq.sum()} invalid frequency values")
        
        # Validate period data
        if 'period' in data.columns:
            period_range = validation_rules.get('period_range', [0.001, 10000])
            invalid_period = (data['period'] < period_range[0]) | (data['period'] > period_range[1])
            if invalid_period.any():
                logger.warning(f"Found {invalid_period.sum()} invalid period values")
        
        # Validate amplitude data
        if 'amplitude' in data.columns:
            amplitude_range = validation_rules.get('amplitude_range', [0, 1000])
            invalid_amplitude = (data['amplitude'] < amplitude_range[0]) | (data['amplitude'] > amplitude_range[1])
            if invalid_amplitude.any():
                logger.warning(f"Found {invalid_amplitude.sum()} invalid amplitude values")
    
    def _clean_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate extracted features."""
        # Remove infinite values
        features = features.replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN values with forward fill then backward fill
        features = features.fillna(method='ffill').fillna(method='bfill')
        
        # Remove features with all NaN values
        features = features.dropna(axis=1, how='all')
        
        return features
    
    def get_feature_importance(self, features: pd.DataFrame, target: pd.Series) -> Dict[str, float]:
        """
        Calculate feature importance for time series data.
        
        Args:
            features: Extracted features
            target: Target variable
            
        Returns:
            Dictionary of feature importance scores
        """
        from sklearn.ensemble import RandomForestRegressor
        
        # Remove NaN values
        valid_idx = ~(features.isna().any(axis=1) | target.isna())
        X = features[valid_idx]
        y = target[valid_idx]
        
        if len(X) == 0:
            logger.warning("No valid data for feature importance calculation")
            return {}
        
        # Train random forest for feature importance
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)
        
        # Get feature importance
        importance_scores = dict(zip(features.columns, rf.feature_importances_))
        
        return importance_scores
    
    def get_feature_summary(self, features: pd.DataFrame) -> Dict[str, Any]:
        """
        Get summary statistics for extracted features.
        
        Args:
            features: Extracted features
            
        Returns:
            Dictionary with feature summary statistics
        """
        summary = {
            'total_features': len(features.columns),
            'feature_names': list(features.columns),
            'data_shape': features.shape,
            'missing_values': features.isna().sum().to_dict(),
            'feature_types': features.dtypes.to_dict(),
            'basic_stats': features.describe().to_dict()
        }
        
        return summary
