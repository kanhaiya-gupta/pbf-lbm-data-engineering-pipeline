"""
Rolling Feature Engineering

This module extracts and engineers rolling window features from time series data for PBF-LB/M processes.
Integrates with YAML configuration for feature definitions and validation rules.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import logging
from ..config.config_manager import ConfigManager

logger = logging.getLogger(__name__)


class RollingFeatures:
    """
    Feature engineering for rolling window features in PBF-LB/M processes.
    
    Extracts features from rolling statistics, moving averages, and window-based aggregations
    based on YAML configuration.
    """
    
    def __init__(self, config_manager: Optional[ConfigManager] = None):
        """
        Initialize rolling feature engineering.
        
        Args:
            config_manager: Configuration manager for loading feature definitions
        """
        self.config_manager = config_manager or ConfigManager()
        self.feature_config = self._load_feature_config()
        self.feature_definitions = self.feature_config.get('feature_definitions', {})
        self.validation_rules = self.feature_config.get('validation_rules', {})
        
    def _load_feature_config(self) -> Dict[str, Any]:
        """Load rolling feature configuration from YAML."""
        try:
            return self.config_manager.load_feature_config('temporal_features/rolling_features')
        except Exception as e:
            logger.warning(f"Could not load rolling config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for rolling features."""
        return {
            'feature_definitions': {
                'rolling_statistics': {
                    'windows': [5, 10, 20, 50],
                    'statistics': ['mean', 'std', 'min', 'max', 'median']
                },
                'rolling_moments': {
                    'windows': [10, 20, 50],
                    'moments': ['skewness', 'kurtosis', 'variance']
                },
                'rolling_percentiles': {
                    'windows': [10, 20, 50],
                    'percentiles': [25, 50, 75, 90, 95]
                }
            },
            'validation_rules': {
                'max_window': 100,
                'min_window': 2,
                'max_percentile': 99
            }
        }
    
    def extract_rolling_statistics(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract rolling statistical features.
        
        Args:
            data: DataFrame with time series data
            
        Returns:
            DataFrame with rolling statistical features
        """
        features = pd.DataFrame(index=data.index)
        
        # Get rolling statistics configuration
        stats_config = self.feature_definitions.get('rolling_statistics', {})
        windows = stats_config.get('windows', [5, 10, 20, 50])
        statistics = stats_config.get('statistics', ['mean', 'std', 'min', 'max', 'median'])
        
        # Get numeric columns for rolling analysis
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col in data.columns:
                series = data[col]
                
                # Create rolling features for each window
                for window in windows:
                    if len(series) > window:
                        rolling = series.rolling(window=window)
                        
                        # Rolling mean
                        if 'mean' in statistics:
                            features[f'{col}_rolling_mean_{window}'] = rolling.mean()
                        
                        # Rolling standard deviation
                        if 'std' in statistics:
                            features[f'{col}_rolling_std_{window}'] = rolling.std()
                        
                        # Rolling minimum
                        if 'min' in statistics:
                            features[f'{col}_rolling_min_{window}'] = rolling.min()
                        
                        # Rolling maximum
                        if 'max' in statistics:
                            features[f'{col}_rolling_max_{window}'] = rolling.max()
                        
                        # Rolling median
                        if 'median' in statistics:
                            features[f'{col}_rolling_median_{window}'] = rolling.median()
                        
                        # Rolling range
                        features[f'{col}_rolling_range_{window}'] = rolling.max() - rolling.min()
                        
                        # Rolling coefficient of variation
                        rolling_mean = rolling.mean()
                        rolling_std = rolling.std()
                        features[f'{col}_rolling_cv_{window}'] = rolling_std / (rolling_mean + 1e-6)
        
        return features
    
    def extract_rolling_moments(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract rolling moment features.
        
        Args:
            data: DataFrame with time series data
            
        Returns:
            DataFrame with rolling moment features
        """
        features = pd.DataFrame(index=data.index)
        
        # Get rolling moments configuration
        moments_config = self.feature_definitions.get('rolling_moments', {})
        windows = moments_config.get('windows', [10, 20, 50])
        moments = moments_config.get('moments', ['skewness', 'kurtosis', 'variance'])
        
        # Get numeric columns for rolling moments analysis
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col in data.columns:
                series = data[col]
                
                # Create rolling moment features for each window
                for window in windows:
                    if len(series) > window:
                        rolling = series.rolling(window=window)
                        
                        # Rolling skewness
                        if 'skewness' in moments:
                            features[f'{col}_rolling_skewness_{window}'] = rolling.skew()
                        
                        # Rolling kurtosis
                        if 'kurtosis' in moments:
                            features[f'{col}_rolling_kurtosis_{window}'] = rolling.kurt()
                        
                        # Rolling variance
                        if 'variance' in moments:
                            features[f'{col}_rolling_variance_{window}'] = rolling.var()
                        
                        # Rolling standard error
                        rolling_std = rolling.std()
                        features[f'{col}_rolling_se_{window}'] = rolling_std / np.sqrt(window)
        
        return features
    
    def extract_rolling_percentiles(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract rolling percentile features.
        
        Args:
            data: DataFrame with time series data
            
        Returns:
            DataFrame with rolling percentile features
        """
        features = pd.DataFrame(index=data.index)
        
        # Get rolling percentiles configuration
        percentiles_config = self.feature_definitions.get('rolling_percentiles', {})
        windows = percentiles_config.get('windows', [10, 20, 50])
        percentiles = percentiles_config.get('percentiles', [25, 50, 75, 90, 95])
        
        # Get numeric columns for rolling percentiles analysis
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col in data.columns:
                series = data[col]
                
                # Create rolling percentile features for each window
                for window in windows:
                    if len(series) > window:
                        rolling = series.rolling(window=window)
                        
                        # Rolling percentiles
                        for percentile in percentiles:
                            features[f'{col}_rolling_p{percentile}_{window}'] = rolling.quantile(percentile / 100)
                        
                        # Interquartile range
                        p25 = rolling.quantile(0.25)
                        p75 = rolling.quantile(0.75)
                        features[f'{col}_rolling_iqr_{window}'] = p75 - p25
                        
                        # Percentile range (90th - 10th percentile)
                        p10 = rolling.quantile(0.10)
                        p90 = rolling.quantile(0.90)
                        features[f'{col}_rolling_pr_{window}'] = p90 - p10
        
        return features
    
    def extract_rolling_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract rolling technical indicators.
        
        Args:
            data: DataFrame with time series data
            
        Returns:
            DataFrame with rolling technical indicators
        """
        features = pd.DataFrame(index=data.index)
        
        # Get numeric columns for technical indicators
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col in data.columns:
                series = data[col]
                
                # Simple Moving Average (SMA)
                for window in [5, 10, 20, 50]:
                    if len(series) > window:
                        features[f'{col}_sma_{window}'] = series.rolling(window=window).mean()
                
                # Exponential Moving Average (EMA)
                for span in [5, 10, 20, 50]:
                    if len(series) > span:
                        features[f'{col}_ema_{span}'] = series.ewm(span=span).mean()
                
                # Bollinger Bands
                for window in [20, 50]:
                    if len(series) > window:
                        rolling_mean = series.rolling(window=window).mean()
                        rolling_std = series.rolling(window=window).std()
                        
                        # Upper and lower bands
                        features[f'{col}_bb_upper_{window}'] = rolling_mean + (2 * rolling_std)
                        features[f'{col}_bb_lower_{window}'] = rolling_mean - (2 * rolling_std)
                        features[f'{col}_bb_middle_{window}'] = rolling_mean
                        
                        # Bollinger Band width
                        features[f'{col}_bb_width_{window}'] = features[f'{col}_bb_upper_{window}'] - features[f'{col}_bb_lower_{window}']
                        
                        # Bollinger Band position
                        features[f'{col}_bb_position_{window}'] = (series - features[f'{col}_bb_lower_{window}']) / (features[f'{col}_bb_width_{window}'] + 1e-6)
                
                # Relative Strength Index (RSI)
                for window in [14, 21]:
                    if len(series) > window:
                        delta = series.diff()
                        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
                        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
                        rs = gain / (loss + 1e-6)
                        features[f'{col}_rsi_{window}'] = 100 - (100 / (1 + rs))
                
                # Moving Average Convergence Divergence (MACD)
                if len(series) > 26:
                    ema_12 = series.ewm(span=12).mean()
                    ema_26 = series.ewm(span=26).mean()
                    features[f'{col}_macd'] = ema_12 - ema_26
                    features[f'{col}_macd_signal'] = features[f'{col}_macd'].ewm(span=9).mean()
                    features[f'{col}_macd_histogram'] = features[f'{col}_macd'] - features[f'{col}_macd_signal']
        
        return features
    
    def extract_rolling_volatility_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract rolling volatility features.
        
        Args:
            data: DataFrame with time series data
            
        Returns:
            DataFrame with rolling volatility features
        """
        features = pd.DataFrame(index=data.index)
        
        # Get numeric columns for volatility analysis
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col in data.columns:
                series = data[col]
                
                # Rolling volatility (standard deviation)
                for window in [5, 10, 20, 50]:
                    if len(series) > window:
                        features[f'{col}_rolling_volatility_{window}'] = series.rolling(window=window).std()
                
                # Rolling volatility of returns
                returns = series.pct_change()
                for window in [5, 10, 20, 50]:
                    if len(returns) > window:
                        features[f'{col}_rolling_return_volatility_{window}'] = returns.rolling(window=window).std()
                
                # Exponentially weighted volatility
                for span in [10, 20, 50]:
                    if len(returns) > span:
                        features[f'{col}_ew_volatility_{span}'] = returns.ewm(span=span).std()
                
                # Parkinson volatility (high-low range)
                if 'high' in data.columns and 'low' in data.columns:
                    high = data['high']
                    low = data['low']
                    parkinson_vol = np.sqrt(0.25 * np.log(high / low) ** 2)
                    for window in [5, 10, 20, 50]:
                        if len(parkinson_vol) > window:
                            features[f'{col}_parkinson_volatility_{window}'] = parkinson_vol.rolling(window=window).mean()
        
        return features
    
    def extract_rolling_correlation_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract rolling correlation features.
        
        Args:
            data: DataFrame with time series data
            
        Returns:
            DataFrame with rolling correlation features
        """
        features = pd.DataFrame(index=data.index)
        
        # Get numeric columns for correlation analysis
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        # Create rolling correlation features between pairs of variables
        for i, col1 in enumerate(numeric_cols):
            for j, col2 in enumerate(numeric_cols):
                if i < j:  # Avoid duplicates
                    series1 = data[col1]
                    series2 = data[col2]
                    
                    # Rolling correlation
                    for window in [10, 20, 50]:
                        if len(series1) > window and len(series2) > window:
                            rolling_corr = series1.rolling(window=window).corr(series2)
                            features[f'{col1}_{col2}_rolling_corr_{window}'] = rolling_corr
                            
                            # Rolling correlation statistics
                            if rolling_corr.notna().sum() > 0:
                                features[f'{col1}_{col2}_rolling_corr_{window}_mean'] = rolling_corr.rolling(window=10).mean()
                                features[f'{col1}_{col2}_rolling_corr_{window}_std'] = rolling_corr.rolling(window=10).std()
        
        return features
    
    def extract_rolling_anomaly_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract rolling anomaly detection features.
        
        Args:
            data: DataFrame with time series data
            
        Returns:
            DataFrame with rolling anomaly features
        """
        features = pd.DataFrame(index=data.index)
        
        # Get numeric columns for anomaly analysis
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col in data.columns:
                series = data[col]
                
                # Rolling z-score
                for window in [10, 20, 50]:
                    if len(series) > window:
                        rolling_mean = series.rolling(window=window).mean()
                        rolling_std = series.rolling(window=window).std()
                        z_score = (series - rolling_mean) / (rolling_std + 1e-6)
                        features[f'{col}_rolling_zscore_{window}'] = z_score
                        
                        # Anomaly detection (z-score > 3)
                        features[f'{col}_rolling_anomaly_{window}'] = (np.abs(z_score) > 3).astype(int)
                
                # Rolling IQR-based anomaly detection
                for window in [10, 20, 50]:
                    if len(series) > window:
                        rolling_q25 = series.rolling(window=window).quantile(0.25)
                        rolling_q75 = series.rolling(window=window).quantile(0.75)
                        iqr = rolling_q75 - rolling_q25
                        lower_bound = rolling_q25 - 1.5 * iqr
                        upper_bound = rolling_q75 + 1.5 * iqr
                        
                        features[f'{col}_rolling_iqr_anomaly_{window}'] = ((series < lower_bound) | (series > upper_bound)).astype(int)
        
        return features
    
    def extract_all_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract all rolling features.
        
        Args:
            data: DataFrame with time series data
            
        Returns:
            DataFrame with all extracted features
        """
        logger.info("Extracting rolling features...")
        
        # Validate input data
        self._validate_input_data(data)
        
        # Extract different feature groups
        feature_groups = [
            self.extract_rolling_statistics(data),
            self.extract_rolling_moments(data),
            self.extract_rolling_percentiles(data),
            self.extract_rolling_technical_indicators(data),
            self.extract_rolling_volatility_features(data),
            self.extract_rolling_correlation_features(data),
            self.extract_rolling_anomaly_features(data)
        ]
        
        # Combine all features
        all_features = pd.concat(feature_groups, axis=1)
        
        # Clean and validate features
        all_features = self._clean_features(all_features)
        
        logger.info(f"Extracted {len(all_features.columns)} rolling features")
        return all_features
    
    def _validate_input_data(self, data: pd.DataFrame) -> None:
        """Validate input data against configuration rules."""
        validation_rules = self.validation_rules
        
        # Validate window parameters
        max_window = validation_rules.get('max_window', 100)
        min_window = validation_rules.get('min_window', 2)
        
        if len(data) < min_window:
            logger.warning(f"Data length {len(data)} is less than minimum window {min_window}")
        
        if len(data) > max_window:
            logger.info(f"Data length {len(data)} is greater than maximum window {max_window}")
        
        # Validate percentile parameters
        max_percentile = validation_rules.get('max_percentile', 99)
        
        if max_percentile > 100:
            logger.warning(f"Maximum percentile {max_percentile} is greater than 100")
    
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
        Calculate feature importance for rolling features.
        
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
