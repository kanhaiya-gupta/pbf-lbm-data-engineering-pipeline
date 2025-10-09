"""
Surface Texture Feature Engineering

This module extracts and engineers features from surface texture image data for PBF-LB/M processes.
Integrates with YAML configuration for feature definitions and validation rules.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import logging
from ..config.config_manager import ConfigManager

logger = logging.getLogger(__name__)


class SurfaceTextureFeatures:
    """
    Feature engineering for surface texture image data in PBF-LB/M processes.
    
    Extracts features from surface roughness, texture patterns, and finish quality
    based on YAML configuration.
    """
    
    def __init__(self, config_manager: Optional[ConfigManager] = None):
        """
        Initialize surface texture feature engineering.
        
        Args:
            config_manager: Configuration manager for loading feature definitions
        """
        self.config_manager = config_manager or ConfigManager()
        self.feature_config = self._load_feature_config()
        self.feature_definitions = self.feature_config.get('feature_definitions', {})
        self.validation_rules = self.feature_config.get('validation_rules', {})
        
    def _load_feature_config(self) -> Dict[str, Any]:
        """Load surface texture feature configuration from YAML."""
        try:
            return self.config_manager.load_feature_config('image_features/surface_texture_features')
        except Exception as e:
            logger.warning(f"Could not load surface texture config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for surface texture features."""
        return {
            'feature_definitions': {
                'roughness_features': {
                    'aggregations': ['Ra', 'Rq', 'Rz', 'Rt'],
                    'derived': ['roughness_uniformity', 'surface_quality']
                },
                'texture_features': {
                    'aggregations': ['texture_energy', 'texture_contrast'],
                    'derived': ['pattern_regularity', 'surface_characteristics']
                },
                'finish_features': {
                    'aggregations': ['finish_quality', 'gloss_level'],
                    'derived': ['surface_appearance', 'aesthetic_quality']
                }
            },
            'validation_rules': {
                'roughness_range': [0, 100],  # micrometers
                'texture_energy_range': [0, 1],
                'gloss_range': [0, 100]  # gloss units
            }
        }
    
    def extract_roughness_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract surface roughness features.
        
        Args:
            data: DataFrame with roughness data
            
        Returns:
            DataFrame with roughness features
        """
        features = pd.DataFrame(index=data.index)
        
        # Ra (Average Roughness) features
        if 'Ra' in data.columns:
            ra_col = data['Ra']
            features['Ra_mean'] = ra_col.rolling(window=10).mean()
            features['Ra_std'] = ra_col.rolling(window=10).std()
            features['Ra_min'] = ra_col.rolling(window=10).min()
            features['Ra_max'] = ra_col.rolling(window=10).max()
            
            # Ra categories
            features['Ra_category'] = pd.cut(ra_col, 
                                          bins=[0, 0.4, 0.8, 1.6, 3.2, 6.3, 12.5, 25, 50, 100], 
                                          labels=['N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'N7', 'N8', 'N9'])
        
        # Rq (Root Mean Square Roughness) features
        if 'Rq' in data.columns:
            features['Rq_mean'] = data['Rq'].rolling(window=10).mean()
            features['Rq_std'] = data['Rq'].rolling(window=10).std()
            features['Rq_min'] = data['Rq'].rolling(window=10).min()
            features['Rq_max'] = data['Rq'].rolling(window=10).max()
        
        # Rz (Average Peak-to-Valley Height) features
        if 'Rz' in data.columns:
            features['Rz_mean'] = data['Rz'].rolling(window=10).mean()
            features['Rz_std'] = data['Rz'].rolling(window=10).std()
            features['Rz_min'] = data['Rz'].rolling(window=10).min()
            features['Rz_max'] = data['Rz'].rolling(window=10).max()
        
        # Rt (Total Roughness) features
        if 'Rt' in data.columns:
            features['Rt_mean'] = data['Rt'].rolling(window=10).mean()
            features['Rt_std'] = data['Rt'].rolling(window=10).std()
            features['Rt_min'] = data['Rt'].rolling(window=10).min()
            features['Rt_max'] = data['Rt'].rolling(window=10).max()
        
        # Roughness uniformity
        roughness_cols = [col for col in data.columns if col in ['Ra', 'Rq', 'Rz', 'Rt']]
        if len(roughness_cols) > 1:
            roughness_data = data[roughness_cols]
            features['roughness_uniformity'] = 1 / (roughness_data.std(axis=1) + 1e-6)
            features['roughness_uniformity_mean'] = features['roughness_uniformity'].rolling(window=10).mean()
        
        # Overall roughness trend
        if 'Ra' in data.columns:
            features['roughness_trend'] = data['Ra'].diff().rolling(window=5).mean()
            features['roughness_volatility'] = data['Ra'].rolling(window=10).std()
        
        return features
    
    def extract_texture_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract texture pattern features.
        
        Args:
            data: DataFrame with texture data
            
        Returns:
            DataFrame with texture features
        """
        features = pd.DataFrame(index=data.index)
        
        # Texture energy features
        if 'texture_energy' in data.columns:
            energy_col = data['texture_energy']
            features['texture_energy_mean'] = energy_col.rolling(window=10).mean()
            features['texture_energy_std'] = energy_col.rolling(window=10).std()
            features['texture_energy_min'] = energy_col.rolling(window=10).min()
            features['texture_energy_max'] = energy_col.rolling(window=10).max()
            
            # Energy categories
            features['energy_category'] = pd.cut(energy_col, 
                                              bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0], 
                                              labels=['low', 'moderate', 'medium', 'high', 'very_high'])
        
        # Texture contrast features
        if 'texture_contrast' in data.columns:
            features['texture_contrast_mean'] = data['texture_contrast'].rolling(window=10).mean()
            features['texture_contrast_std'] = data['texture_contrast'].rolling(window=10).std()
            features['texture_contrast_min'] = data['texture_contrast'].rolling(window=10).min()
            features['texture_contrast_max'] = data['texture_contrast'].rolling(window=10).max()
        
        # Texture homogeneity features
        if 'texture_homogeneity' in data.columns:
            features['texture_homogeneity_mean'] = data['texture_homogeneity'].rolling(window=10).mean()
            features['texture_homogeneity_std'] = data['texture_homogeneity'].rolling(window=10).std()
            
            # Homogeneity categories
            features['homogeneity_category'] = pd.cut(data['texture_homogeneity'], 
                                                   bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0], 
                                                   labels=['very_heterogeneous', 'heterogeneous', 'moderate', 'homogeneous', 'very_homogeneous'])
        
        # Texture entropy features
        if 'texture_entropy' in data.columns:
            features['texture_entropy_mean'] = data['texture_entropy'].rolling(window=10).mean()
            features['texture_entropy_std'] = data['texture_entropy'].rolling(window=10).std()
        
        # Pattern regularity features
        if 'pattern_regularity' in data.columns:
            features['pattern_regularity_mean'] = data['pattern_regularity'].rolling(window=10).mean()
            features['pattern_regularity_std'] = data['pattern_regularity'].rolling(window=10).std()
            
            # Regularity categories
            features['regularity_category'] = pd.cut(data['pattern_regularity'], 
                                                  bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0], 
                                                  labels=['irregular', 'low_regular', 'moderate', 'regular', 'very_regular'])
        
        # Texture directionality
        if 'texture_direction' in data.columns:
            features['texture_direction_mean'] = data['texture_direction'].rolling(window=10).mean()
            features['texture_direction_std'] = data['texture_direction'].rolling(window=10).std()
        
        return features
    
    def extract_finish_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract surface finish features.
        
        Args:
            data: DataFrame with finish data
            
        Returns:
            DataFrame with finish features
        """
        features = pd.DataFrame(index=data.index)
        
        # Finish quality features
        if 'finish_quality' in data.columns:
            quality_col = data['finish_quality']
            features['finish_quality_mean'] = quality_col.rolling(window=10).mean()
            features['finish_quality_std'] = quality_col.rolling(window=10).std()
            features['finish_quality_min'] = quality_col.rolling(window=10).min()
            features['finish_quality_max'] = quality_col.rolling(window=10).max()
            
            # Quality categories
            features['finish_quality_category'] = pd.cut(quality_col, 
                                                      bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0], 
                                                      labels=['poor', 'fair', 'good', 'very_good', 'excellent'])
        
        # Gloss level features
        if 'gloss_level' in data.columns:
            gloss_col = data['gloss_level']
            features['gloss_level_mean'] = gloss_col.rolling(window=10).mean()
            features['gloss_level_std'] = gloss_col.rolling(window=10).std()
            features['gloss_level_min'] = gloss_col.rolling(window=10).min()
            features['gloss_level_max'] = gloss_col.rolling(window=10).max()
            
            # Gloss categories
            features['gloss_category'] = pd.cut(gloss_col, 
                                             bins=[0, 10, 30, 60, 80, 100], 
                                             labels=['matte', 'low_gloss', 'medium_gloss', 'high_gloss', 'mirror'])
        
        # Surface appearance features
        if 'surface_appearance' in data.columns:
            features['surface_appearance_mean'] = data['surface_appearance'].rolling(window=10).mean()
            features['surface_appearance_std'] = data['surface_appearance'].rolling(window=10).std()
        
        # Aesthetic quality features
        if 'aesthetic_quality' in data.columns:
            features['aesthetic_quality_mean'] = data['aesthetic_quality'].rolling(window=10).mean()
            features['aesthetic_quality_std'] = data['aesthetic_quality'].rolling(window=10).std()
            
            # Aesthetic categories
            features['aesthetic_category'] = pd.cut(data['aesthetic_quality'], 
                                                 bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0], 
                                                 labels=['poor', 'fair', 'good', 'very_good', 'excellent'])
        
        return features
    
    def extract_waviness_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract surface waviness features.
        
        Args:
            data: DataFrame with waviness data
            
        Returns:
            DataFrame with waviness features
        """
        features = pd.DataFrame(index=data.index)
        
        # Wa (Average Waviness) features
        if 'Wa' in data.columns:
            features['Wa_mean'] = data['Wa'].rolling(window=10).mean()
            features['Wa_std'] = data['Wa'].rolling(window=10).std()
            features['Wa_min'] = data['Wa'].rolling(window=10).min()
            features['Wa_max'] = data['Wa'].rolling(window=10).max()
        
        # Wq (Root Mean Square Waviness) features
        if 'Wq' in data.columns:
            features['Wq_mean'] = data['Wq'].rolling(window=10).mean()
            features['Wq_std'] = data['Wq'].rolling(window=10).std()
            features['Wq_min'] = data['Wq'].rolling(window=10).min()
            features['Wq_max'] = data['Wq'].rolling(window=10).max()
        
        # Wavelength features
        if 'wavelength' in data.columns:
            features['wavelength_mean'] = data['wavelength'].rolling(window=10).mean()
            features['wavelength_std'] = data['wavelength'].rolling(window=10).std()
            
            # Wavelength categories
            features['wavelength_category'] = pd.cut(data['wavelength'], 
                                                  bins=[0, 0.5, 1.0, 2.0, 5.0, 10.0], 
                                                  labels=['very_short', 'short', 'medium', 'long', 'very_long'])
        
        # Waviness ratio
        if 'Wa' in data.columns and 'Ra' in data.columns:
            features['waviness_ratio'] = data['Wa'] / (data['Ra'] + 1e-6)
            features['waviness_ratio_mean'] = features['waviness_ratio'].rolling(window=10).mean()
        
        return features
    
    def extract_lay_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract surface lay (direction) features.
        
        Args:
            data: DataFrame with lay data
            
        Returns:
            DataFrame with lay features
        """
        features = pd.DataFrame(index=data.index)
        
        # Lay direction features
        if 'lay_direction' in data.columns:
            features['lay_direction_mean'] = data['lay_direction'].rolling(window=10).mean()
            features['lay_direction_std'] = data['lay_direction'].rolling(window=10).std()
            
            # Direction categories
            features['lay_direction_category'] = pd.cut(data['lay_direction'], 
                                                     bins=[0, 45, 90, 135, 180], 
                                                     labels=['horizontal', 'diagonal_up', 'vertical', 'diagonal_down'])
        
        # Lay consistency features
        if 'lay_consistency' in data.columns:
            features['lay_consistency_mean'] = data['lay_consistency'].rolling(window=10).mean()
            features['lay_consistency_std'] = data['lay_consistency'].rolling(window=10).std()
            
            # Consistency categories
            features['lay_consistency_category'] = pd.cut(data['lay_consistency'], 
                                                       bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0], 
                                                       labels=['inconsistent', 'low_consistent', 'moderate', 'consistent', 'very_consistent'])
        
        # Lay angle features
        if 'lay_angle' in data.columns:
            features['lay_angle_mean'] = data['lay_angle'].rolling(window=10).mean()
            features['lay_angle_std'] = data['lay_angle'].rolling(window=10).std()
        
        return features
    
    def extract_quality_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract overall surface quality features.
        
        Args:
            data: DataFrame with quality data
            
        Returns:
            DataFrame with quality features
        """
        features = pd.DataFrame(index=data.index)
        
        # Overall surface quality score
        quality_components = []
        
        # Roughness component (inverted)
        if 'Ra' in data.columns:
            roughness_quality = 1 / (data['Ra'] + 1e-6)
            quality_components.append(roughness_quality)
        
        # Texture uniformity component
        if 'texture_homogeneity' in data.columns:
            quality_components.append(data['texture_homogeneity'])
        
        # Finish quality component
        if 'finish_quality' in data.columns:
            quality_components.append(data['finish_quality'])
        
        # Gloss level component
        if 'gloss_level' in data.columns:
            gloss_quality = data['gloss_level'] / 100  # Normalize to 0-1
            quality_components.append(gloss_quality)
        
        if len(quality_components) > 1:
            features['overall_surface_quality'] = np.mean(quality_components, axis=0)
            features['overall_surface_quality_mean'] = features['overall_surface_quality'].rolling(window=10).mean()
            
            # Quality categories
            features['surface_quality_category'] = pd.cut(features['overall_surface_quality'], 
                                                       bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0], 
                                                       labels=['poor', 'fair', 'good', 'very_good', 'excellent'])
        
        # Quality consistency
        if 'overall_surface_quality' in features.columns:
            features['quality_consistency'] = 1 / (features['overall_surface_quality'].rolling(window=10).std() + 1e-6)
        
        # Quality trend
        if 'overall_surface_quality' in features.columns:
            features['quality_trend'] = features['overall_surface_quality'].diff().rolling(window=5).mean()
        
        return features
    
    def extract_all_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract all surface texture features.
        
        Args:
            data: DataFrame with surface texture data
            
        Returns:
            DataFrame with all extracted features
        """
        logger.info("Extracting surface texture features...")
        
        # Validate input data
        self._validate_input_data(data)
        
        # Extract different feature groups
        feature_groups = [
            self.extract_roughness_features(data),
            self.extract_texture_features(data),
            self.extract_finish_features(data),
            self.extract_waviness_features(data),
            self.extract_lay_features(data),
            self.extract_quality_features(data)
        ]
        
        # Combine all features
        all_features = pd.concat(feature_groups, axis=1)
        
        # Clean and validate features
        all_features = self._clean_features(all_features)
        
        logger.info(f"Extracted {len(all_features.columns)} surface texture features")
        return all_features
    
    def _validate_input_data(self, data: pd.DataFrame) -> None:
        """Validate input data against configuration rules."""
        validation_rules = self.validation_rules
        
        # Validate roughness data
        if 'Ra' in data.columns:
            roughness_range = validation_rules.get('roughness_range', [0, 100])
            invalid_roughness = (data['Ra'] < roughness_range[0]) | (data['Ra'] > roughness_range[1])
            if invalid_roughness.any():
                logger.warning(f"Found {invalid_roughness.sum()} invalid roughness values")
        
        # Validate texture energy data
        if 'texture_energy' in data.columns:
            energy_range = validation_rules.get('texture_energy_range', [0, 1])
            invalid_energy = (data['texture_energy'] < energy_range[0]) | (data['texture_energy'] > energy_range[1])
            if invalid_energy.any():
                logger.warning(f"Found {invalid_energy.sum()} invalid texture energy values")
        
        # Validate gloss data
        if 'gloss_level' in data.columns:
            gloss_range = validation_rules.get('gloss_range', [0, 100])
            invalid_gloss = (data['gloss_level'] < gloss_range[0]) | (data['gloss_level'] > gloss_range[1])
            if invalid_gloss.any():
                logger.warning(f"Found {invalid_gloss.sum()} invalid gloss values")
    
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
        Calculate feature importance for surface texture data.
        
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
