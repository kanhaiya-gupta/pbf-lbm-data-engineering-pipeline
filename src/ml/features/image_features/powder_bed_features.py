"""
Powder Bed Feature Engineering

This module extracts and engineers features from powder bed image data for PBF-LB/M processes.
Integrates with YAML configuration for feature definitions and validation rules.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import logging
from ..config.config_manager import ConfigManager

logger = logging.getLogger(__name__)


class PowderBedFeatures:
    """
    Feature engineering for powder bed image data in PBF-LB/M processes.
    
    Extracts features from powder distribution, layer quality, and bed uniformity
    based on YAML configuration.
    """
    
    def __init__(self, config_manager: Optional[ConfigManager] = None):
        """
        Initialize powder bed feature engineering.
        
        Args:
            config_manager: Configuration manager for loading feature definitions
        """
        self.config_manager = config_manager or ConfigManager()
        self.feature_config = self._load_feature_config()
        self.feature_definitions = self.feature_config.get('feature_definitions', {})
        self.validation_rules = self.feature_config.get('validation_rules', {})
        
    def _load_feature_config(self) -> Dict[str, Any]:
        """Load powder bed feature configuration from YAML."""
        try:
            return self.config_manager.load_feature_config('image_features/powder_bed_features')
        except Exception as e:
            logger.warning(f"Could not load powder bed config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for powder bed features."""
        return {
            'feature_definitions': {
                'distribution_features': {
                    'aggregations': ['uniformity', 'coverage', 'density'],
                    'derived': ['powder_quality', 'layer_completeness']
                },
                'particle_features': {
                    'aggregations': ['size_distribution', 'shape_factor'],
                    'derived': ['flowability', 'packing_density']
                },
                'defect_features': {
                    'aggregations': ['void_count', 'agglomeration', 'contamination'],
                    'derived': ['defect_severity', 'quality_score']
                }
            },
            'validation_rules': {
                'coverage_range': [0, 100],  # percentage
                'particle_size_range': [0.01, 100],  # micrometers
                'uniformity_range': [0, 1]
            }
        }
    
    def extract_distribution_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract powder distribution features.
        
        Args:
            data: DataFrame with powder distribution data
            
        Returns:
            DataFrame with distribution features
        """
        features = pd.DataFrame(index=data.index)
        
        # Coverage features
        if 'coverage' in data.columns:
            coverage_col = data['coverage']
            features['coverage_mean'] = coverage_col.rolling(window=10).mean()
            features['coverage_std'] = coverage_col.rolling(window=10).std()
            features['coverage_min'] = coverage_col.rolling(window=10).min()
            features['coverage_max'] = coverage_col.rolling(window=10).max()
            
            # Coverage categories
            features['coverage_category'] = pd.cut(coverage_col, 
                                                bins=[0, 50, 70, 85, 95, 100], 
                                                labels=['poor', 'fair', 'good', 'very_good', 'excellent'])
        
        # Uniformity features
        if 'uniformity' in data.columns:
            uniformity_col = data['uniformity']
            features['uniformity_mean'] = uniformity_col.rolling(window=10).mean()
            features['uniformity_std'] = uniformity_col.rolling(window=10).std()
            
            # Uniformity categories
            features['uniformity_category'] = pd.cut(uniformity_col, 
                                                  bins=[0, 0.3, 0.5, 0.7, 0.8, 1.0], 
                                                  labels=['very_poor', 'poor', 'fair', 'good', 'excellent'])
        
        # Density features
        if 'powder_density' in data.columns:
            features['powder_density_mean'] = data['powder_density'].rolling(window=10).mean()
            features['powder_density_std'] = data['powder_density'].rolling(window=10).std()
            features['powder_density_min'] = data['powder_density'].rolling(window=10).min()
            features['powder_density_max'] = data['powder_density'].rolling(window=10).max()
        
        # Spatial distribution features
        if 'spatial_variance' in data.columns:
            features['spatial_variance_mean'] = data['spatial_variance'].rolling(window=10).mean()
            features['spatial_variance_std'] = data['spatial_variance'].rolling(window=10).std()
        
        # Layer completeness
        if 'coverage' in data.columns and 'target_coverage' in data.columns:
            features['layer_completeness'] = data['coverage'] / data['target_coverage']
            features['layer_completeness_mean'] = features['layer_completeness'].rolling(window=10).mean()
            
            # Completeness categories
            features['completeness_category'] = pd.cut(features['layer_completeness'], 
                                                    bins=[0, 0.7, 0.8, 0.9, 0.95, 1.0], 
                                                    labels=['incomplete', 'poor', 'fair', 'good', 'complete'])
        
        return features
    
    def extract_particle_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract particle size and shape features.
        
        Args:
            data: DataFrame with particle data
            
        Returns:
            DataFrame with particle features
        """
        features = pd.DataFrame(index=data.index)
        
        # Particle size features
        if 'particle_size_mean' in data.columns:
            features['particle_size_mean'] = data['particle_size_mean'].rolling(window=10).mean()
            features['particle_size_std'] = data['particle_size_mean'].rolling(window=10).std()
            features['particle_size_min'] = data['particle_size_mean'].rolling(window=10).min()
            features['particle_size_max'] = data['particle_size_mean'].rolling(window=10).max()
            
            # Size categories
            features['particle_size_category'] = pd.cut(data['particle_size_mean'], 
                                                     bins=[0, 10, 25, 50, 100, 200], 
                                                     labels=['very_fine', 'fine', 'medium', 'coarse', 'very_coarse'])
        
        # Size distribution features
        if 'size_distribution_width' in data.columns:
            features['size_distribution_width_mean'] = data['size_distribution_width'].rolling(window=10).mean()
            features['size_distribution_width_std'] = data['size_distribution_width'].rolling(window=10).std()
            
            # Distribution categories
            features['distribution_category'] = pd.cut(data['size_distribution_width'], 
                                                    bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0], 
                                                    labels=['narrow', 'moderate', 'wide', 'very_wide', 'extremely_wide'])
        
        # Shape factor features
        if 'shape_factor' in data.columns:
            features['shape_factor_mean'] = data['shape_factor'].rolling(window=10).mean()
            features['shape_factor_std'] = data['shape_factor'].rolling(window=10).std()
            
            # Shape categories
            features['shape_category'] = pd.cut(data['shape_factor'], 
                                             bins=[0, 0.3, 0.5, 0.7, 0.8, 1.0], 
                                             labels=['irregular', 'angular', 'sub_angular', 'sub_rounded', 'rounded'])
        
        # Flowability estimation
        if 'particle_size_mean' in data.columns and 'shape_factor' in data.columns:
            # Simplified flowability calculation
            features['flowability'] = data['shape_factor'] / (data['particle_size_mean'] + 1e-6)
            features['flowability_mean'] = features['flowability'].rolling(window=10).mean()
            
            # Flowability categories
            features['flowability_category'] = pd.cut(features['flowability'], 
                                                   bins=[0, 0.1, 0.3, 0.5, 0.7, 1.0], 
                                                   labels=['poor', 'fair', 'good', 'very_good', 'excellent'])
        
        # Packing density estimation
        if 'particle_size_mean' in data.columns and 'powder_density' in data.columns:
            # Simplified packing density calculation
            features['packing_density'] = data['powder_density'] / (data['particle_size_mean'] + 1e-6)
            features['packing_density_mean'] = features['packing_density'].rolling(window=10).mean()
        
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
        
        # Void count features
        if 'void_count' in data.columns:
            void_count = data['void_count']
            features['void_count_mean'] = void_count.rolling(window=10).mean()
            features['void_count_std'] = void_count.rolling(window=10).std()
            features['void_count_max'] = void_count.rolling(window=10).max()
            
            # Void categories
            features['void_category'] = pd.cut(void_count, 
                                            bins=[0, 1, 5, 10, 20, 100], 
                                            labels=['none', 'few', 'moderate', 'many', 'excessive'])
        
        # Void size features
        if 'void_size_mean' in data.columns:
            features['void_size_mean'] = data['void_size_mean'].rolling(window=10).mean()
            features['void_size_std'] = data['void_size_mean'].rolling(window=10).std()
            features['void_size_max'] = data['void_size_mean'].rolling(window=10).max()
        
        # Agglomeration features
        if 'agglomeration_count' in data.columns:
            features['agglomeration_count_mean'] = data['agglomeration_count'].rolling(window=10).mean()
            features['agglomeration_count_std'] = data['agglomeration_count'].rolling(window=10).std()
            
            # Agglomeration categories
            features['agglomeration_category'] = pd.cut(data['agglomeration_count'], 
                                                     bins=[0, 1, 3, 5, 10, 50], 
                                                     labels=['none', 'minimal', 'low', 'moderate', 'high'])
        
        # Contamination features
        if 'contamination_level' in data.columns:
            features['contamination_level_mean'] = data['contamination_level'].rolling(window=10).mean()
            features['contamination_level_std'] = data['contamination_level'].rolling(window=10).std()
            
            # Contamination categories
            features['contamination_category'] = pd.cut(data['contamination_level'], 
                                                     bins=[0, 0.01, 0.05, 0.1, 0.2, 1.0], 
                                                     labels=['clean', 'minimal', 'low', 'moderate', 'high'])
        
        # Defect severity estimation
        if 'void_count' in data.columns and 'void_size_mean' in data.columns:
            features['defect_severity'] = data['void_count'] * data['void_size_mean']
            features['defect_severity_mean'] = features['defect_severity'].rolling(window=10).mean()
        
        # Overall defect score
        defect_components = []
        if 'void_count' in data.columns:
            defect_components.append(data['void_count'])
        if 'agglomeration_count' in data.columns:
            defect_components.append(data['agglomeration_count'])
        if 'contamination_level' in data.columns:
            defect_components.append(data['contamination_level'] * 100)  # Scale contamination
        
        if len(defect_components) > 1:
            features['overall_defect_score'] = np.mean(defect_components, axis=0)
            features['overall_defect_score_mean'] = features['overall_defect_score'].rolling(window=10).mean()
            
            # Defect score categories
            features['defect_score_category'] = pd.cut(features['overall_defect_score'], 
                                                     bins=[0, 1, 3, 5, 10, 100], 
                                                     labels=['excellent', 'good', 'fair', 'poor', 'very_poor'])
        
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
        
        # Surface roughness estimation
        if 'texture_energy' in data.columns and 'texture_contrast' in data.columns:
            features['surface_roughness'] = data['texture_energy'] * data['texture_contrast']
            features['surface_roughness_mean'] = features['surface_roughness'].rolling(window=10).mean()
            
            # Roughness categories
            features['roughness_category'] = pd.cut(features['surface_roughness'], 
                                                 bins=[0, 0.1, 0.3, 0.5, 0.7, 1.0], 
                                                 labels=['smooth', 'slightly_rough', 'moderate', 'rough', 'very_rough'])
        
        return features
    
    def extract_quality_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract overall quality assessment features.
        
        Args:
            data: DataFrame with quality data
            
        Returns:
            DataFrame with quality features
        """
        features = pd.DataFrame(index=data.index)
        
        # Overall quality score
        quality_components = []
        
        # Coverage component
        if 'coverage' in data.columns:
            coverage_quality = data['coverage'] / 100  # Normalize to 0-1
            quality_components.append(coverage_quality)
        
        # Uniformity component
        if 'uniformity' in data.columns:
            quality_components.append(data['uniformity'])
        
        # Defect component (inverted)
        if 'overall_defect_score' in features.columns:
            defect_quality = 1 / (features['overall_defect_score'] + 1e-6)
            quality_components.append(defect_quality)
        
        # Flowability component
        if 'flowability' in features.columns:
            quality_components.append(features['flowability'])
        
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
        
        # Quality trend
        if 'overall_quality' in features.columns:
            features['quality_trend'] = features['overall_quality'].diff().rolling(window=5).mean()
        
        return features
    
    def extract_all_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract all powder bed features.
        
        Args:
            data: DataFrame with powder bed data
            
        Returns:
            DataFrame with all extracted features
        """
        logger.info("Extracting powder bed features...")
        
        # Validate input data
        self._validate_input_data(data)
        
        # Extract different feature groups
        feature_groups = [
            self.extract_distribution_features(data),
            self.extract_particle_features(data),
            self.extract_defect_features(data),
            self.extract_texture_features(data),
            self.extract_quality_features(data)
        ]
        
        # Combine all features
        all_features = pd.concat(feature_groups, axis=1)
        
        # Clean and validate features
        all_features = self._clean_features(all_features)
        
        logger.info(f"Extracted {len(all_features.columns)} powder bed features")
        return all_features
    
    def _validate_input_data(self, data: pd.DataFrame) -> None:
        """Validate input data against configuration rules."""
        validation_rules = self.validation_rules
        
        # Validate coverage data
        if 'coverage' in data.columns:
            coverage_range = validation_rules.get('coverage_range', [0, 100])
            invalid_coverage = (data['coverage'] < coverage_range[0]) | (data['coverage'] > coverage_range[1])
            if invalid_coverage.any():
                logger.warning(f"Found {invalid_coverage.sum()} invalid coverage values")
        
        # Validate particle size data
        if 'particle_size_mean' in data.columns:
            size_range = validation_rules.get('particle_size_range', [0.01, 100])
            invalid_size = (data['particle_size_mean'] < size_range[0]) | (data['particle_size_mean'] > size_range[1])
            if invalid_size.any():
                logger.warning(f"Found {invalid_size.sum()} invalid particle size values")
        
        # Validate uniformity data
        if 'uniformity' in data.columns:
            uniformity_range = validation_rules.get('uniformity_range', [0, 1])
            invalid_uniformity = (data['uniformity'] < uniformity_range[0]) | (data['uniformity'] > uniformity_range[1])
            if invalid_uniformity.any():
                logger.warning(f"Found {invalid_uniformity.sum()} invalid uniformity values")
    
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
        Calculate feature importance for powder bed data.
        
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
