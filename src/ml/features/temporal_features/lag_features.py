"""
Lag Feature Engineering

This module extracts and engineers lag features from time series data for PBF-LB/M processes.
Integrates with YAML configuration for feature definitions and validation rules.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import logging
from ..config.config_manager import ConfigManager

logger = logging.getLogger(__name__)


class LagFeatures:
    """
    Feature engineering for lag features in PBF-LB/M processes.
    
    Extracts features from time-lagged values, differences, and temporal relationships
    based on YAML configuration.
    """
    
    def __init__(self, config_manager: Optional[ConfigManager] = None):
        """
        Initialize lag feature engineering.
        
        Args:
            config_manager: Configuration manager for loading feature definitions
        """
        self.config_manager = config_manager or ConfigManager()
        self.feature_config = self._load_feature_config()
        self.feature_definitions = self.feature_config.get('feature_definitions', {})
        self.validation_rules = self.feature_config.get('validation_rules', {})
        
    def _load_feature_config(self) -> Dict[str, Any]:
        """Load lag feature configuration from YAML."""
        try:
            return self.config_manager.load_feature_config('temporal_features/lag_features')
        except Exception as e:
            logger.warning(f"Could not load lag config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for lag features."""
        return {
            'feature_definitions': {
                'lag_features': {
                    'lags': [1, 2, 3, 5, 10, 20],
                    'aggregations': ['mean', 'std', 'min', 'max']
                },
                'difference_features': {
                    'differences': [1, 2, 3, 5, 10],
                    'derived': ['acceleration', 'jerk']
                },
                'ratio_features': {
                    'ratios': [1, 2, 3, 5, 10],
                    'derived': ['growth_rate', 'change_rate']
                }
            },
            'validation_rules': {
                'max_lag': 100,
                'min_lag': 1,
                'max_difference': 50
            }
        }
    
    def extract_lag_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract lag features from time series data.
        
        Args:
            data: DataFrame with time series data
            
        Returns:
            DataFrame with lag features
        """
        features = pd.DataFrame(index=data.index)
        
        # Get lag configuration
        lag_config = self.feature_definitions.get('lag_features', {})
        lags = lag_config.get('lags', [1, 2, 3, 5, 10, 20])
        
        # Get numeric columns for lag analysis
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col in data.columns:
                series = data[col]
                
                # Create lag features
                for lag in lags:
                    if len(series) > lag:
                        lag_feature = series.shift(lag)
                        features[f'{col}_lag_{lag}'] = lag_feature
                        
                        # Lag statistics
                        if lag_feature.notna().sum() > 0:
                            features[f'{col}_lag_{lag}_mean'] = lag_feature.rolling(window=10).mean()
                            features[f'{col}_lag_{lag}_std'] = lag_feature.rolling(window=10).std()
                            features[f'{col}_lag_{lag}_min'] = lag_feature.rolling(window=10).min()
                            features[f'{col}_lag_{lag}_max'] = lag_feature.rolling(window=10).max()
        
        return features
    
    def extract_difference_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract difference features from time series data.
        
        Args:
            data: DataFrame with time series data
            
        Returns:
            DataFrame with difference features
        """
        features = pd.DataFrame(index=data.index)
        
        # Get difference configuration
        diff_config = self.feature_definitions.get('difference_features', {})
        differences = diff_config.get('differences', [1, 2, 3, 5, 10])
        
        # Get numeric columns for difference analysis
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col in data.columns:
                series = data[col]
                
                # Create difference features
                for diff in differences:
                    if len(series) > diff:
                        diff_feature = series.diff(diff)
                        features[f'{col}_diff_{diff}'] = diff_feature
                        
                        # Difference statistics
                        if diff_feature.notna().sum() > 0:
                            features[f'{col}_diff_{diff}_mean'] = diff_feature.rolling(window=10).mean()
                            features[f'{col}_diff_{diff}_std'] = diff_feature.rolling(window=10).std()
                            features[f'{col}_diff_{diff}_min'] = diff_feature.rolling(window=10).min()
                            features[f'{col}_diff_{diff}_max'] = diff_feature.rolling(window=10).max()
                
                # Acceleration (second difference)
                if len(series) > 2:
                    acceleration = series.diff().diff()
                    features[f'{col}_acceleration'] = acceleration
                    features[f'{col}_acceleration_mean'] = acceleration.rolling(window=10).mean()
                    features[f'{col}_acceleration_std'] = acceleration.rolling(window=10).std()
                
                # Jerk (third difference)
                if len(series) > 3:
                    jerk = series.diff().diff().diff()
                    features[f'{col}_jerk'] = jerk
                    features[f'{col}_jerk_mean'] = jerk.rolling(window=10).mean()
                    features[f'{col}_jerk_std'] = jerk.rolling(window=10).std()
        
        return features
    
    def extract_ratio_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract ratio features from time series data.
        
        Args:
            data: DataFrame with time series data
            
        Returns:
            DataFrame with ratio features
        """
        features = pd.DataFrame(index=data.index)
        
        # Get ratio configuration
        ratio_config = self.feature_definitions.get('ratio_features', {})
        ratios = ratio_config.get('ratios', [1, 2, 3, 5, 10])
        
        # Get numeric columns for ratio analysis
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col in data.columns:
                series = data[col]
                
                # Create ratio features
                for ratio in ratios:
                    if len(series) > ratio:
                        lag_series = series.shift(ratio)
                        ratio_feature = series / (lag_series + 1e-6)  # Avoid division by zero
                        features[f'{col}_ratio_{ratio}'] = ratio_feature
                        
                        # Ratio statistics
                        if ratio_feature.notna().sum() > 0:
                            features[f'{col}_ratio_{ratio}_mean'] = ratio_feature.rolling(window=10).mean()
                            features[f'{col}_ratio_{ratio}_std'] = ratio_feature.rolling(window=10).std()
                            features[f'{col}_ratio_{ratio}_min'] = ratio_feature.rolling(window=10).min()
                            features[f'{col}_ratio_{ratio}_max'] = ratio_feature.rolling(window=10).max()
                
                # Growth rate (percentage change)
                if len(series) > 1:
                    growth_rate = series.pct_change()
                    features[f'{col}_growth_rate'] = growth_rate
                    features[f'{col}_growth_rate_mean'] = growth_rate.rolling(window=10).mean()
                    features[f'{col}_growth_rate_std'] = growth_rate.rolling(window=10).std()
                
                # Change rate (absolute change)
                if len(series) > 1:
                    change_rate = series.diff()
                    features[f'{col}_change_rate'] = change_rate
                    features[f'{col}_change_rate_mean'] = change_rate.rolling(window=10).mean()
                    features[f'{col}_change_rate_std'] = change_rate.rolling(window=10).std()
        
        return features
    
    def extract_cross_lag_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract cross-lag features between different variables.
        
        Args:
            data: DataFrame with time series data
            
        Returns:
            DataFrame with cross-lag features
        """
        features = pd.DataFrame(index=data.index)
        
        # Get numeric columns for cross-lag analysis
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        # Create cross-lag features between pairs of variables
        for i, col1 in enumerate(numeric_cols):
            for j, col2 in enumerate(numeric_cols):
                if i < j:  # Avoid duplicates
                    series1 = data[col1]
                    series2 = data[col2]
                    
                    # Cross-correlation at different lags
                    for lag in [1, 2, 5, 10]:
                        if len(series1) > lag and len(series2) > lag:
                            # Lag series2 by lag periods
                            lagged_series2 = series2.shift(lag)
                            
                            # Cross-correlation
                            cross_corr = series1.corr(lagged_series2)
                            if not np.isnan(cross_corr):
                                features[f'{col1}_{col2}_cross_corr_lag_{lag}'] = cross_corr
                            
                            # Cross-ratio
                            cross_ratio = series1 / (lagged_series2 + 1e-6)
                            features[f'{col1}_{col2}_cross_ratio_lag_{lag}'] = cross_ratio
                            features[f'{col1}_{col2}_cross_ratio_lag_{lag}_mean'] = cross_ratio.rolling(window=10).mean()
        
        return features
    
    def extract_lead_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract lead features (future values) from time series data.
        
        Args:
            data: DataFrame with time series data
            
        Returns:
            DataFrame with lead features
        """
        features = pd.DataFrame(index=data.index)
        
        # Get numeric columns for lead analysis
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        # Create lead features
        leads = [1, 2, 3, 5, 10]
        
        for col in numeric_cols:
            if col in data.columns:
                series = data[col]
                
                for lead in leads:
                    if len(series) > lead:
                        lead_feature = series.shift(-lead)  # Negative shift for future values
                        features[f'{col}_lead_{lead}'] = lead_feature
                        
                        # Lead statistics
                        if lead_feature.notna().sum() > 0:
                            features[f'{col}_lead_{lead}_mean'] = lead_feature.rolling(window=10).mean()
                            features[f'{col}_lead_{lead}_std'] = lead_feature.rolling(window=10).std()
        
        return features
    
    def extract_temporal_interaction_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract temporal interaction features.
        
        Args:
            data: DataFrame with time series data
            
        Returns:
            DataFrame with temporal interaction features
        """
        features = pd.DataFrame(index=data.index)
        
        # Get numeric columns for interaction analysis
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        # Create interaction features
        for i, col1 in enumerate(numeric_cols):
            for j, col2 in enumerate(numeric_cols):
                if i < j:  # Avoid duplicates
                    series1 = data[col1]
                    series2 = data[col2]
                    
                    # Current interaction
                    interaction = series1 * series2
                    features[f'{col1}_{col2}_interaction'] = interaction
                    features[f'{col1}_{col2}_interaction_mean'] = interaction.rolling(window=10).mean()
                    
                    # Lagged interaction
                    for lag in [1, 2, 5]:
                        if len(series1) > lag:
                            lagged_series1 = series1.shift(lag)
                            lagged_interaction = lagged_series1 * series2
                            features[f'{col1}_{col2}_interaction_lag_{lag}'] = lagged_interaction
                            features[f'{col1}_{col2}_interaction_lag_{lag}_mean'] = lagged_interaction.rolling(window=10).mean()
        
        return features
    
    def extract_all_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract all lag features.
        
        Args:
            data: DataFrame with time series data
            
        Returns:
            DataFrame with all extracted features
        """
        logger.info("Extracting lag features...")
        
        # Validate input data
        self._validate_input_data(data)
        
        # Extract different feature groups
        feature_groups = [
            self.extract_lag_features(data),
            self.extract_difference_features(data),
            self.extract_ratio_features(data),
            self.extract_cross_lag_features(data),
            self.extract_lead_features(data),
            self.extract_temporal_interaction_features(data)
        ]
        
        # Combine all features
        all_features = pd.concat(feature_groups, axis=1)
        
        # Clean and validate features
        all_features = self._clean_features(all_features)
        
        logger.info(f"Extracted {len(all_features.columns)} lag features")
        return all_features
    
    def _validate_input_data(self, data: pd.DataFrame) -> None:
        """Validate input data against configuration rules."""
        validation_rules = self.validation_rules
        
        # Validate lag parameters
        max_lag = validation_rules.get('max_lag', 100)
        min_lag = validation_rules.get('min_lag', 1)
        
        if len(data) < min_lag:
            logger.warning(f"Data length {len(data)} is less than minimum lag {min_lag}")
        
        if len(data) > max_lag:
            logger.info(f"Data length {len(data)} is greater than maximum lag {max_lag}")
        
        # Validate difference parameters
        max_difference = validation_rules.get('max_difference', 50)
        
        if len(data) < max_difference:
            logger.warning(f"Data length {len(data)} is less than maximum difference {max_difference}")
    
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
        Calculate feature importance for lag features.
        
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
