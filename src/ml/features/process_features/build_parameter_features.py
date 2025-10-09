"""
Build Parameter Feature Engineering

This module extracts and engineers features from build parameters for PBF-LB/M processes.
Integrates with YAML configuration for feature definitions and validation rules.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import logging
from ..config.config_manager import ConfigManager

logger = logging.getLogger(__name__)


class BuildParameterFeatures:
    """
    Feature engineering for build parameters in PBF-LB/M processes.
    
    Extracts features from build orientation, support structures, scan patterns,
    and other build-specific parameters based on YAML configuration.
    """
    
    def __init__(self, config_manager: Optional[ConfigManager] = None):
        """
        Initialize build parameter feature engineering.
        
        Args:
            config_manager: Configuration manager for loading feature definitions
        """
        self.config_manager = config_manager or ConfigManager()
        self.feature_config = self._load_feature_config()
        self.feature_definitions = self.feature_config.get('feature_definitions', {})
        self.validation_rules = self.feature_config.get('validation_rules', {})
        
    def _load_feature_config(self) -> Dict[str, Any]:
        """Load build parameter feature configuration from YAML."""
        try:
            return self.config_manager.load_feature_config('process_features/build_parameter_features')
        except Exception as e:
            logger.warning(f"Could not load build parameter config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for build parameter features."""
        return {
            'feature_definitions': {
                'orientation_features': {
                    'aggregations': ['mean', 'std'],
                    'derived': ['surface_area_ratio', 'support_volume_ratio']
                },
                'support_features': {
                    'aggregations': ['count', 'total_volume', 'contact_area'],
                    'derived': ['support_density', 'removal_difficulty']
                },
                'scan_features': {
                    'aggregations': ['pattern_complexity', 'scan_length'],
                    'derived': ['build_time_estimate', 'energy_consumption']
                }
            },
            'validation_rules': {
                'orientation_range': [0, 90],  # degrees
                'support_volume_range': [0, 1.0],  # ratio
                'scan_speed_range': [0, 10000]  # mm/s
            }
        }
    
    def extract_orientation_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract build orientation features.
        
        Args:
            data: DataFrame with orientation data
            
        Returns:
            DataFrame with orientation features
        """
        features = pd.DataFrame(index=data.index)
        
        if 'build_orientation_x' in data.columns:
            features['orientation_x_mean'] = data['build_orientation_x'].rolling(window=10).mean()
            features['orientation_x_std'] = data['build_orientation_x'].rolling(window=10).std()
        
        if 'build_orientation_y' in data.columns:
            features['orientation_y_mean'] = data['build_orientation_y'].rolling(window=10).mean()
            features['orientation_y_std'] = data['build_orientation_y'].rolling(window=10).std()
        
        if 'build_orientation_z' in data.columns:
            features['orientation_z_mean'] = data['build_orientation_z'].rolling(window=10).mean()
            features['orientation_z_std'] = data['build_orientation_z'].rolling(window=10).std()
        
        # Combined orientation features
        orientation_cols = [col for col in data.columns if 'orientation' in col]
        if len(orientation_cols) >= 2:
            # Calculate orientation vector magnitude
            orientation_data = data[orientation_cols]
            features['orientation_magnitude'] = np.sqrt((orientation_data ** 2).sum(axis=1))
            
            # Calculate orientation stability
            features['orientation_stability'] = features['orientation_magnitude'].rolling(window=10).std()
        
        # Derived features
        if 'part_volume' in data.columns and 'part_surface_area' in data.columns:
            features['surface_area_ratio'] = data['part_surface_area'] / data['part_volume']
        
        return features
    
    def extract_support_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract support structure features.
        
        Args:
            data: DataFrame with support structure data
            
        Returns:
            DataFrame with support features
        """
        features = pd.DataFrame(index=data.index)
        
        if 'support_count' in data.columns:
            features['support_count_mean'] = data['support_count'].rolling(window=10).mean()
            features['support_count_std'] = data['support_count'].rolling(window=10).std()
        
        if 'support_volume' in data.columns:
            features['support_volume_mean'] = data['support_volume'].rolling(window=10).mean()
            features['support_volume_std'] = data['support_volume'].rolling(window=10).std()
            
            # Support volume ratio
            if 'part_volume' in data.columns:
                features['support_volume_ratio'] = data['support_volume'] / data['part_volume']
        
        if 'support_contact_area' in data.columns:
            features['support_contact_area_mean'] = data['support_contact_area'].rolling(window=10).mean()
            features['support_contact_area_std'] = data['support_contact_area'].rolling(window=10).std()
        
        # Derived support features
        if 'support_count' in data.columns and 'part_volume' in data.columns:
            features['support_density'] = data['support_count'] / data['part_volume']
        
        if 'support_height' in data.columns and 'support_angle' in data.columns:
            features['removal_difficulty'] = data['support_height'] * np.sin(np.radians(data['support_angle']))
        
        return features
    
    def extract_scan_pattern_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract scan pattern features.
        
        Args:
            data: DataFrame with scan pattern data
            
        Returns:
            DataFrame with scan pattern features
        """
        features = pd.DataFrame(index=data.index)
        
        if 'scan_pattern_type' in data.columns:
            # Encode scan pattern types
            pattern_mapping = {'contour': 1, 'hatch': 2, 'spiral': 3, 'zigzag': 4}
            features['scan_pattern_encoded'] = data['scan_pattern_type'].map(pattern_mapping)
        
        if 'scan_length' in data.columns:
            features['scan_length_mean'] = data['scan_length'].rolling(window=10).mean()
            features['scan_length_std'] = data['scan_length'].rolling(window=10).std()
        
        if 'scan_direction_changes' in data.columns:
            features['direction_changes_mean'] = data['scan_direction_changes'].rolling(window=10).mean()
            features['direction_changes_std'] = data['scan_direction_changes'].rolling(window=10).std()
        
        # Pattern complexity features
        if 'scan_length' in data.columns and 'layer_area' in data.columns:
            features['pattern_complexity'] = data['scan_length'] / data['layer_area']
        
        # Build time estimation
        if 'scan_length' in data.columns and 'scan_speed' in data.columns:
            features['build_time_estimate'] = data['scan_length'] / data['scan_speed']
        
        return features
    
    def extract_layer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract layer-specific build features.
        
        Args:
            data: DataFrame with layer data
            
        Returns:
            DataFrame with layer features
        """
        features = pd.DataFrame(index=data.index)
        
        if 'layer_number' in data.columns:
            features['layer_number'] = data['layer_number']
            features['build_progress'] = data['layer_number'] / data['layer_number'].max()
        
        if 'layer_thickness' in data.columns:
            features['layer_thickness_mean'] = data['layer_thickness'].rolling(window=10).mean()
            features['layer_thickness_std'] = data['layer_thickness'].rolling(window=10).std()
        
        if 'layer_area' in data.columns:
            features['layer_area_mean'] = data['layer_area'].rolling(window=10).mean()
            features['layer_area_std'] = data['layer_area'].rolling(window=10).std()
            
            # Layer area progression
            features['layer_area_trend'] = data['layer_area'].diff().rolling(window=5).mean()
        
        if 'layer_perimeter' in data.columns:
            features['layer_perimeter_mean'] = data['layer_perimeter'].rolling(window=10).mean()
            features['layer_perimeter_std'] = data['layer_perimeter'].rolling(window=10).std()
        
        # Layer complexity features
        if 'layer_area' in data.columns and 'layer_perimeter' in data.columns:
            features['layer_complexity'] = data['layer_perimeter'] / np.sqrt(data['layer_area'])
        
        return features
    
    def extract_geometry_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract part geometry features.
        
        Args:
            data: DataFrame with geometry data
            
        Returns:
            DataFrame with geometry features
        """
        features = pd.DataFrame(index=data.index)
        
        # Basic geometry features
        geometry_cols = ['part_volume', 'part_surface_area', 'part_height', 'part_width', 'part_depth']
        for col in geometry_cols:
            if col in data.columns:
                features[f'{col}_mean'] = data[col].rolling(window=10).mean()
                features[f'{col}_std'] = data[col].rolling(window=10).std()
        
        # Derived geometry features
        if 'part_volume' in data.columns and 'part_surface_area' in data.columns:
            features['surface_to_volume_ratio'] = data['part_surface_area'] / data['part_volume']
        
        if 'part_height' in data.columns and 'part_width' in data.columns and 'part_depth' in data.columns:
            features['aspect_ratio'] = data['part_height'] / np.maximum(data['part_width'], data['part_depth'])
            features['compactness'] = data['part_volume'] / (data['part_height'] * data['part_width'] * data['part_depth'])
        
        # Overhang features
        if 'overhang_angle' in data.columns:
            features['overhang_angle_mean'] = data['overhang_angle'].rolling(window=10).mean()
            features['overhang_angle_std'] = data['overhang_angle'].rolling(window=10).std()
            
            # Critical overhang detection
            features['critical_overhang'] = (data['overhang_angle'] > 45).astype(int)
        
        if 'overhang_area' in data.columns:
            features['overhang_area_mean'] = data['overhang_area'].rolling(window=10).mean()
            features['overhang_area_std'] = data['overhang_area'].rolling(window=10).std()
        
        return features
    
    def extract_energy_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract energy consumption features.
        
        Args:
            data: DataFrame with energy data
            
        Returns:
            DataFrame with energy features
        """
        features = pd.DataFrame(index=data.index)
        
        if 'laser_power' in data.columns and 'scan_speed' in data.columns and 'scan_length' in data.columns:
            # Energy per layer
            features['energy_per_layer'] = (data['laser_power'] * data['scan_length']) / data['scan_speed']
            
            # Total energy consumption
            features['total_energy'] = features['energy_per_layer'].cumsum()
        
        if 'build_time' in data.columns:
            features['build_time_mean'] = data['build_time'].rolling(window=10).mean()
            features['build_time_std'] = data['build_time'].rolling(window=10).std()
        
        # Energy efficiency features
        if 'part_volume' in data.columns and 'total_energy' in features.columns:
            features['energy_efficiency'] = data['part_volume'] / features['total_energy']
        
        return features
    
    def extract_all_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract all build parameter features.
        
        Args:
            data: DataFrame with build parameter data
            
        Returns:
            DataFrame with all extracted features
        """
        logger.info("Extracting build parameter features...")
        
        # Validate input data
        self._validate_input_data(data)
        
        # Extract different feature groups
        feature_groups = [
            self.extract_orientation_features(data),
            self.extract_support_features(data),
            self.extract_scan_pattern_features(data),
            self.extract_layer_features(data),
            self.extract_geometry_features(data),
            self.extract_energy_features(data)
        ]
        
        # Combine all features
        all_features = pd.concat(feature_groups, axis=1)
        
        # Clean and validate features
        all_features = self._clean_features(all_features)
        
        logger.info(f"Extracted {len(all_features.columns)} build parameter features")
        return all_features
    
    def _validate_input_data(self, data: pd.DataFrame) -> None:
        """Validate input data against configuration rules."""
        validation_rules = self.validation_rules
        
        # Validate orientation data
        orientation_cols = [col for col in data.columns if 'orientation' in col]
        for col in orientation_cols:
            orientation_range = validation_rules.get('orientation_range', [0, 90])
            invalid_values = (data[col] < orientation_range[0]) | (data[col] > orientation_range[1])
            if invalid_values.any():
                logger.warning(f"Found {invalid_values.sum()} invalid {col} values")
        
        # Validate support volume
        if 'support_volume' in data.columns:
            support_range = validation_rules.get('support_volume_range', [0, 1.0])
            invalid_support = (data['support_volume'] < support_range[0]) | (data['support_volume'] > support_range[1])
            if invalid_support.any():
                logger.warning(f"Found {invalid_support.sum()} invalid support volume values")
    
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
        Calculate feature importance for build parameters.
        
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
