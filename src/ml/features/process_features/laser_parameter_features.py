"""
Laser Parameter Feature Engineering

This module extracts and engineers features from laser parameters for PBF-LB/M processes.
Integrates with YAML configuration for feature definitions and validation rules.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import logging
from ..config.config_manager import ConfigManager

logger = logging.getLogger(__name__)


class LaserParameterFeatures:
    """
    Feature engineering for laser parameters in PBF-LB/M processes.
    
    Extracts features from laser power, speed, spot size, and other parameters
    based on YAML configuration definitions.
    """
    
    def __init__(self, config_manager: Optional[ConfigManager] = None):
        """
        Initialize laser parameter feature engineering.
        
        Args:
            config_manager: Configuration manager for loading feature definitions
        """
        self.config_manager = config_manager or ConfigManager()
        self.feature_config = self._load_feature_config()
        self.feature_definitions = self.feature_config.get('feature_definitions', {})
        self.validation_rules = self.feature_config.get('validation_rules', {})
        
    def _load_feature_config(self) -> Dict[str, Any]:
        """Load laser parameter feature configuration from YAML."""
        try:
            return self.config_manager.load_feature_config('process_features/laser_parameter_features')
        except Exception as e:
            logger.warning(f"Could not load laser parameter config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for laser parameter features."""
        return {
            'feature_definitions': {
                'power_features': {
                    'aggregations': ['mean', 'std', 'min', 'max', 'range'],
                    'temporal': ['trend', 'volatility'],
                    'derived': ['power_density', 'energy_per_layer']
                },
                'speed_features': {
                    'aggregations': ['mean', 'std', 'min', 'max'],
                    'temporal': ['acceleration', 'deceleration'],
                    'derived': ['scan_time', 'exposure_time']
                },
                'spot_features': {
                    'aggregations': ['mean', 'std'],
                    'derived': ['spot_area', 'overlap_ratio']
                }
            },
            'validation_rules': {
                'power_range': [0, 1000],  # Watts
                'speed_range': [0, 10000],  # mm/s
                'spot_range': [0.01, 1.0]   # mm
            }
        }
    
    def extract_power_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract laser power features.
        
        Args:
            data: DataFrame with laser power data
            
        Returns:
            DataFrame with power features
        """
        features = pd.DataFrame(index=data.index)
        
        # Basic aggregations
        if 'laser_power' in data.columns:
            power_col = data['laser_power']
            features['power_mean'] = power_col.rolling(window=10).mean()
            features['power_std'] = power_col.rolling(window=10).std()
            features['power_min'] = power_col.rolling(window=10).min()
            features['power_max'] = power_col.rolling(window=10).max()
            features['power_range'] = features['power_max'] - features['power_min']
            
            # Temporal features
            features['power_trend'] = power_col.diff().rolling(window=5).mean()
            features['power_volatility'] = power_col.rolling(window=10).std()
            
            # Derived features
            if 'scan_speed' in data.columns:
                features['power_density'] = power_col / (np.pi * (data.get('spot_size', 0.1) / 2) ** 2)
                features['energy_per_layer'] = power_col / data['scan_speed']
        
        return features
    
    def extract_speed_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract laser scan speed features.
        
        Args:
            data: DataFrame with scan speed data
            
        Returns:
            DataFrame with speed features
        """
        features = pd.DataFrame(index=data.index)
        
        if 'scan_speed' in data.columns:
            speed_col = data['scan_speed']
            features['speed_mean'] = speed_col.rolling(window=10).mean()
            features['speed_std'] = speed_col.rolling(window=10).std()
            features['speed_min'] = speed_col.rolling(window=10).min()
            features['speed_max'] = speed_col.rolling(window=10).max()
            
            # Temporal features
            features['speed_acceleration'] = speed_col.diff().diff()
            features['speed_deceleration'] = -features['speed_acceleration'].clip(lower=0)
            
            # Derived features
            if 'layer_height' in data.columns:
                features['scan_time'] = data['layer_height'] / speed_col
            if 'laser_power' in data.columns:
                features['exposure_time'] = data['laser_power'] / speed_col
        
        return features
    
    def extract_spot_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract laser spot size features.
        
        Args:
            data: DataFrame with spot size data
            
        Returns:
            DataFrame with spot features
        """
        features = pd.DataFrame(index=data.index)
        
        if 'spot_size' in data.columns:
            spot_col = data['spot_size']
            features['spot_mean'] = spot_col.rolling(window=10).mean()
            features['spot_std'] = spot_col.rolling(window=10).std()
            
            # Derived features
            features['spot_area'] = np.pi * (spot_col / 2) ** 2
            if 'hatch_spacing' in data.columns:
                features['overlap_ratio'] = 1 - (data['hatch_spacing'] / spot_col)
        
        return features
    
    def extract_hatch_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract hatch pattern features.
        
        Args:
            data: DataFrame with hatch pattern data
            
        Returns:
            DataFrame with hatch features
        """
        features = pd.DataFrame(index=data.index)
        
        if 'hatch_spacing' in data.columns:
            hatch_col = data['hatch_spacing']
            features['hatch_spacing_mean'] = hatch_col.rolling(window=10).mean()
            features['hatch_spacing_std'] = hatch_col.rolling(window=10).std()
            
            # Derived features
            if 'spot_size' in data.columns:
                features['overlap_percentage'] = (1 - hatch_col / data['spot_size']) * 100
                features['energy_density'] = data.get('laser_power', 100) / (hatch_col * data.get('scan_speed', 1000))
        
        return features
    
    def extract_layer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract layer-specific features.
        
        Args:
            data: DataFrame with layer data
            
        Returns:
            DataFrame with layer features
        """
        features = pd.DataFrame(index=data.index)
        
        if 'layer_height' in data.columns:
            layer_col = data['layer_height']
            features['layer_height_mean'] = layer_col.rolling(window=10).mean()
            features['layer_height_std'] = layer_col.rolling(window=10).std()
            
            # Layer progression features
            features['layer_number'] = data.get('layer_number', range(len(data)))
            features['build_progress'] = features['layer_number'] / features['layer_number'].max()
            
            # Derived features
            if 'laser_power' in data.columns and 'scan_speed' in data.columns:
                features['energy_per_layer'] = (data['laser_power'] * layer_col) / data['scan_speed']
        
        return features
    
    def extract_interaction_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract interaction features between laser parameters.
        
        Args:
            data: DataFrame with laser parameter data
            
        Returns:
            DataFrame with interaction features
        """
        features = pd.DataFrame(index=data.index)
        
        # Power-Speed interactions
        if 'laser_power' in data.columns and 'scan_speed' in data.columns:
            features['power_speed_ratio'] = data['laser_power'] / data['scan_speed']
            features['power_speed_product'] = data['laser_power'] * data['scan_speed']
        
        # Power-Spot interactions
        if 'laser_power' in data.columns and 'spot_size' in data.columns:
            features['power_spot_ratio'] = data['laser_power'] / data['spot_size']
            features['power_density'] = data['laser_power'] / (np.pi * (data['spot_size'] / 2) ** 2)
        
        # Speed-Hatch interactions
        if 'scan_speed' in data.columns and 'hatch_spacing' in data.columns:
            features['speed_hatch_ratio'] = data['scan_speed'] / data['hatch_spacing']
            features['exposure_time'] = data['hatch_spacing'] / data['scan_speed']
        
        return features
    
    def extract_all_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract all laser parameter features.
        
        Args:
            data: DataFrame with laser parameter data
            
        Returns:
            DataFrame with all extracted features
        """
        logger.info("Extracting laser parameter features...")
        
        # Validate input data
        self._validate_input_data(data)
        
        # Extract different feature groups
        feature_groups = [
            self.extract_power_features(data),
            self.extract_speed_features(data),
            self.extract_spot_features(data),
            self.extract_hatch_features(data),
            self.extract_layer_features(data),
            self.extract_interaction_features(data)
        ]
        
        # Combine all features
        all_features = pd.concat(feature_groups, axis=1)
        
        # Clean and validate features
        all_features = self._clean_features(all_features)
        
        logger.info(f"Extracted {len(all_features.columns)} laser parameter features")
        return all_features
    
    def _validate_input_data(self, data: pd.DataFrame) -> None:
        """Validate input data against configuration rules."""
        validation_rules = self.validation_rules
        
        if 'laser_power' in data.columns:
            power_range = validation_rules.get('power_range', [0, 1000])
            invalid_power = (data['laser_power'] < power_range[0]) | (data['laser_power'] > power_range[1])
            if invalid_power.any():
                logger.warning(f"Found {invalid_power.sum()} invalid laser power values")
        
        if 'scan_speed' in data.columns:
            speed_range = validation_rules.get('speed_range', [0, 10000])
            invalid_speed = (data['scan_speed'] < speed_range[0]) | (data['scan_speed'] > speed_range[1])
            if invalid_speed.any():
                logger.warning(f"Found {invalid_speed.sum()} invalid scan speed values")
    
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
        Calculate feature importance for laser parameters.
        
        Args:
            features: Extracted features
            target: Target variable
            
        Returns:
            Dictionary of feature importance scores
        """
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import mean_squared_error
        
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
