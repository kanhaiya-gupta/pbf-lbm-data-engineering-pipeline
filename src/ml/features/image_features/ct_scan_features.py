"""
CT Scan Feature Engineering

This module extracts and engineers features from CT scan data for PBF-LB/M processes.
Integrates with YAML configuration for feature definitions and validation rules.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import logging
from ..config.config_manager import ConfigManager

logger = logging.getLogger(__name__)


class CTScanFeatures:
    """
    Feature engineering for CT scan data in PBF-LB/M processes.
    
    Extracts features from 3D volume data, density distributions,
    and internal structure analysis based on YAML configuration.
    """
    
    def __init__(self, config_manager: Optional[ConfigManager] = None):
        """
        Initialize CT scan feature engineering.
        
        Args:
            config_manager: Configuration manager for loading feature definitions
        """
        self.config_manager = config_manager or ConfigManager()
        self.feature_config = self._load_feature_config()
        self.feature_definitions = self.feature_config.get('feature_definitions', {})
        self.validation_rules = self.feature_config.get('validation_rules', {})
        
    def _load_feature_config(self) -> Dict[str, Any]:
        """Load CT scan feature configuration from YAML."""
        try:
            return self.config_manager.load_feature_config('image_features/ct_scan_features')
        except Exception as e:
            logger.warning(f"Could not load CT scan config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for CT scan features."""
        return {
            'feature_definitions': {
                'density_features': {
                    'aggregations': ['mean', 'std', 'min', 'max', 'histogram'],
                    'derived': ['density_uniformity', 'porosity_estimation']
                },
                'volume_features': {
                    'aggregations': ['volume', 'surface_area', 'sphericity'],
                    'derived': ['volume_fraction', 'connectivity']
                },
                'defect_features': {
                    'aggregations': ['defect_count', 'defect_size', 'defect_density'],
                    'derived': ['defect_severity', 'critical_defect_ratio']
                }
            },
            'validation_rules': {
                'density_range': [0, 100],  # HU (Hounsfield Units)
                'volume_range': [0, 1000000],  # mm³
                'resolution_range': [0.01, 1.0]  # mm
            }
        }
    
    def extract_density_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract density and Hounsfield Unit features.
        
        Args:
            data: DataFrame with CT scan density data
            
        Returns:
            DataFrame with density features
        """
        features = pd.DataFrame(index=data.index)
        
        # Basic density features
        if 'density_mean' in data.columns:
            density_col = data['density_mean']
            features['density_mean'] = density_col.rolling(window=10).mean()
            features['density_std'] = density_col.rolling(window=10).std()
            features['density_min'] = density_col.rolling(window=10).min()
            features['density_max'] = density_col.rolling(window=10).max()
            features['density_range'] = features['density_max'] - features['density_min']
            
            # Density categories
            features['density_category'] = pd.cut(density_col, 
                                               bins=[-1000, -500, 0, 100, 500, 1000, 3000], 
                                               labels=['air', 'fat', 'water', 'soft_tissue', 'bone', 'metal'])
        
        # Density distribution features
        if 'density_histogram' in data.columns:
            # Parse histogram data (assuming it's a string representation)
            features['density_histogram_mean'] = data['density_histogram'].rolling(window=10).mean()
            features['density_histogram_std'] = data['density_histogram'].rolling(window=10).std()
        
        # Density uniformity
        if 'density_std' in features.columns:
            features['density_uniformity'] = 1 / (features['density_std'] + 1e-6)
        
        # Porosity estimation
        if 'density_mean' in data.columns:
            # Simplified porosity calculation (assuming material density = 1000 HU)
            material_density = 1000  # HU
            features['porosity_estimation'] = 1 - (data['density_mean'] / material_density)
            features['porosity_estimation'] = features['porosity_estimation'].clip(0, 1)
            features['porosity_mean'] = features['porosity_estimation'].rolling(window=10).mean()
        
        # Density gradient features
        if 'density_gradient' in data.columns:
            features['density_gradient_mean'] = data['density_gradient'].rolling(window=10).mean()
            features['density_gradient_std'] = data['density_gradient'].rolling(window=10).std()
            features['density_gradient_max'] = data['density_gradient'].rolling(window=10).max()
        
        return features
    
    def extract_volume_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract volume and geometric features.
        
        Args:
            data: DataFrame with volume data
            
        Returns:
            DataFrame with volume features
        """
        features = pd.DataFrame(index=data.index)
        
        # Basic volume features
        if 'volume' in data.columns:
            volume_col = data['volume']
            features['volume_mean'] = volume_col.rolling(window=10).mean()
            features['volume_std'] = volume_col.rolling(window=10).std()
            features['volume_min'] = volume_col.rolling(window=10).min()
            features['volume_max'] = volume_col.rolling(window=10).max()
            
            # Volume categories
            features['volume_category'] = pd.cut(volume_col, 
                                              bins=[0, 100, 1000, 10000, 100000, 1000000], 
                                              labels=['tiny', 'small', 'medium', 'large', 'very_large'])
        
        # Surface area features
        if 'surface_area' in data.columns:
            features['surface_area_mean'] = data['surface_area'].rolling(window=10).mean()
            features['surface_area_std'] = data['surface_area'].rolling(window=10).std()
            
            # Surface-to-volume ratio
            if 'volume' in data.columns:
                features['surface_volume_ratio'] = data['surface_area'] / (data['volume'] + 1e-6)
                features['surface_volume_ratio_mean'] = features['surface_volume_ratio'].rolling(window=10).mean()
        
        # Sphericity features
        if 'sphericity' in data.columns:
            features['sphericity_mean'] = data['sphericity'].rolling(window=10).mean()
            features['sphericity_std'] = data['sphericity'].rolling(window=10).std()
            
            # Shape categories
            features['shape_category'] = pd.cut(data['sphericity'], 
                                             bins=[0, 0.3, 0.6, 0.8, 0.9, 1.0], 
                                             labels=['irregular', 'elongated', 'moderate', 'round', 'spherical'])
        
        # Volume fraction features
        if 'volume' in data.columns and 'total_volume' in data.columns:
            features['volume_fraction'] = data['volume'] / data['total_volume']
            features['volume_fraction_mean'] = features['volume_fraction'].rolling(window=10).mean()
        
        # Connectivity features
        if 'connectivity' in data.columns:
            features['connectivity_mean'] = data['connectivity'].rolling(window=10).mean()
            features['connectivity_std'] = data['connectivity'].rolling(window=10).std()
            
            # Connectivity categories
            features['connectivity_category'] = pd.cut(data['connectivity'], 
                                                    bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0], 
                                                    labels=['isolated', 'low', 'moderate', 'high', 'fully_connected'])
        
        return features
    
    def extract_defect_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract defect and flaw features.
        
        Args:
            data: DataFrame with defect data
            
        Returns:
            DataFrame with defect features
        """
        features = pd.DataFrame(index=data.index)
        
        # Defect count features
        if 'defect_count' in data.columns:
            defect_count = data['defect_count']
            features['defect_count_mean'] = defect_count.rolling(window=10).mean()
            features['defect_count_std'] = defect_count.rolling(window=10).std()
            features['defect_count_max'] = defect_count.rolling(window=10).max()
            
            # Defect categories
            features['defect_category'] = pd.cut(defect_count, 
                                              bins=[0, 1, 5, 10, 20, 100], 
                                              labels=['none', 'few', 'moderate', 'many', 'excessive'])
        
        # Defect size features
        if 'defect_size_mean' in data.columns:
            features['defect_size_mean'] = data['defect_size_mean'].rolling(window=10).mean()
            features['defect_size_std'] = data['defect_size_mean'].rolling(window=10).std()
            features['defect_size_max'] = data['defect_size_mean'].rolling(window=10).max()
            
            # Size categories
            features['defect_size_category'] = pd.cut(data['defect_size_mean'], 
                                                   bins=[0, 0.1, 0.5, 1.0, 2.0, 10], 
                                                   labels=['micro', 'small', 'medium', 'large', 'very_large'])
        
        # Defect density features
        if 'defect_density' in data.columns:
            features['defect_density_mean'] = data['defect_density'].rolling(window=10).mean()
            features['defect_density_std'] = data['defect_density'].rolling(window=10).std()
            
            # Density categories
            features['defect_density_category'] = pd.cut(data['defect_density'], 
                                                       bins=[0, 0.01, 0.05, 0.1, 0.2, 1.0], 
                                                       labels=['minimal', 'low', 'moderate', 'high', 'severe'])
        
        # Defect severity estimation
        if 'defect_count' in data.columns and 'defect_size_mean' in data.columns:
            features['defect_severity'] = data['defect_count'] * data['defect_size_mean']
            features['defect_severity_mean'] = features['defect_severity'].rolling(window=10).mean()
        
        # Critical defect ratio
        if 'critical_defect_count' in data.columns and 'defect_count' in data.columns:
            features['critical_defect_ratio'] = data['critical_defect_count'] / (data['defect_count'] + 1e-6)
            features['critical_defect_ratio_mean'] = features['critical_defect_ratio'].rolling(window=10).mean()
        
        return features
    
    def extract_morphology_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract morphological and shape features.
        
        Args:
            data: DataFrame with morphology data
            
        Returns:
            DataFrame with morphology features
        """
        features = pd.DataFrame(index=data.index)
        
        # Aspect ratio features
        if 'aspect_ratio' in data.columns:
            features['aspect_ratio_mean'] = data['aspect_ratio'].rolling(window=10).mean()
            features['aspect_ratio_std'] = data['aspect_ratio'].rolling(window=10).std()
            
            # Shape categories
            features['aspect_category'] = pd.cut(data['aspect_ratio'], 
                                              bins=[0, 0.5, 0.8, 1.2, 2.0, 10], 
                                              labels=['elongated', 'rectangular', 'square', 'wide', 'very_wide'])
        
        # Compactness features
        if 'compactness' in data.columns:
            features['compactness_mean'] = data['compactness'].rolling(window=10).mean()
            features['compactness_std'] = data['compactness'].rolling(window=10).std()
        
        # Convexity features
        if 'convexity' in data.columns:
            features['convexity_mean'] = data['convexity'].rolling(window=10).mean()
            features['convexity_std'] = data['convexity'].rolling(window=10).std()
            
            # Convexity categories
            features['convexity_category'] = pd.cut(data['convexity'], 
                                                 bins=[0, 0.5, 0.7, 0.8, 0.9, 1.0], 
                                                 labels=['concave', 'low_convex', 'moderate', 'high_convex', 'fully_convex'])
        
        # Solidity features
        if 'solidity' in data.columns:
            features['solidity_mean'] = data['solidity'].rolling(window=10).mean()
            features['solidity_std'] = data['solidity'].rolling(window=10).std()
        
        # Eccentricity features
        if 'eccentricity' in data.columns:
            features['eccentricity_mean'] = data['eccentricity'].rolling(window=10).mean()
            features['eccentricity_std'] = data['eccentricity'].rolling(window=10).std()
        
        return features
    
    def extract_texture_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract texture and pattern features.
        
        Args:
            data: DataFrame with texture data
            
        Returns:
            DataFrame with texture features
        """
        features = pd.DataFrame(index=data.index)
        
        # Texture energy features
        if 'texture_energy' in data.columns:
            features['texture_energy_mean'] = data['texture_energy'].rolling(window=10).mean()
            features['texture_energy_std'] = data['texture_energy'].rolling(window=10).std()
        
        # Texture contrast features
        if 'texture_contrast' in data.columns:
            features['texture_contrast_mean'] = data['texture_contrast'].rolling(window=10).mean()
            features['texture_contrast_std'] = data['texture_contrast'].rolling(window=10).std()
        
        # Texture homogeneity features
        if 'texture_homogeneity' in data.columns:
            features['texture_homogeneity_mean'] = data['texture_homogeneity'].rolling(window=10).mean()
            features['texture_homogeneity_std'] = data['texture_homogeneity'].rolling(window=10).std()
        
        # Texture entropy features
        if 'texture_entropy' in data.columns:
            features['texture_entropy_mean'] = data['texture_entropy'].rolling(window=10).mean()
            features['texture_entropy_std'] = data['texture_entropy'].rolling(window=10).std()
        
        # Surface roughness estimation
        if 'texture_energy' in data.columns and 'texture_contrast' in data.columns:
            features['surface_roughness'] = data['texture_energy'] * data['texture_contrast']
            features['surface_roughness_mean'] = features['surface_roughness'].rolling(window=10).mean()
        
        return features
    
    def extract_quality_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract quality assessment features.
        
        Args:
            data: DataFrame with quality data
            
        Returns:
            DataFrame with quality features
        """
        features = pd.DataFrame(index=data.index)
        
        # Overall quality score
        quality_components = []
        
        # Density uniformity component
        if 'density_uniformity' in features.columns:
            quality_components.append(features['density_uniformity'])
        
        # Defect severity component (inverted)
        if 'defect_severity' in features.columns:
            defect_quality = 1 / (features['defect_severity'] + 1e-6)
            quality_components.append(defect_quality)
        
        # Porosity component (inverted)
        if 'porosity_estimation' in features.columns:
            porosity_quality = 1 - features['porosity_estimation']
            quality_components.append(porosity_quality)
        
        if len(quality_components) > 1:
            features['overall_quality'] = np.mean(quality_components, axis=0)
            features['overall_quality_mean'] = features['overall_quality'].rolling(window=10).mean()
            
            # Quality categories
            features['quality_category'] = pd.cut(features['overall_quality'], 
                                               bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0], 
                                               labels=['poor', 'fair', 'good', 'very_good', 'excellent'])
        
        # Quality consistency
        if 'overall_quality' in features.columns:
            features['quality_consistency'] = 1 / (features['overall_quality'].rolling(window=10).std() + 1e-6)
        
        return features
    
    def extract_all_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract all CT scan features.
        
        Args:
            data: DataFrame with CT scan data
            
        Returns:
            DataFrame with all extracted features
        """
        logger.info("Extracting CT scan features...")
        
        # Validate input data
        self._validate_input_data(data)
        
        # Extract different feature groups
        feature_groups = [
            self.extract_density_features(data),
            self.extract_volume_features(data),
            self.extract_defect_features(data),
            self.extract_morphology_features(data),
            self.extract_texture_features(data),
            self.extract_quality_features(data)
        ]
        
        # Combine all features
        all_features = pd.concat(feature_groups, axis=1)
        
        # Clean and validate features
        all_features = self._clean_features(all_features)
        
        logger.info(f"Extracted {len(all_features.columns)} CT scan features")
        return all_features
    
    def _validate_input_data(self, data: pd.DataFrame) -> None:
        """Validate input data against configuration rules."""
        validation_rules = self.validation_rules
        
        # Validate density data
        if 'density_mean' in data.columns:
            density_range = validation_rules.get('density_range', [0, 100])
            invalid_density = (data['density_mean'] < density_range[0]) | (data['density_mean'] > density_range[1])
            if invalid_density.any():
                logger.warning(f"Found {invalid_density.sum()} invalid density values")
        
        # Validate volume data
        if 'volume' in data.columns:
            volume_range = validation_rules.get('volume_range', [0, 1000000])
            invalid_volume = (data['volume'] < volume_range[0]) | (data['volume'] > volume_range[1])
            if invalid_volume.any():
                logger.warning(f"Found {invalid_volume.sum()} invalid volume values")
        
        # Validate resolution data
        if 'resolution' in data.columns:
            resolution_range = validation_rules.get('resolution_range', [0.01, 1.0])
            invalid_resolution = (data['resolution'] < resolution_range[0]) | (data['resolution'] > resolution_range[1])
            if invalid_resolution.any():
                logger.warning(f"Found {invalid_resolution.sum()} invalid resolution values")
    
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
        Calculate feature importance for CT scan data.
        
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
