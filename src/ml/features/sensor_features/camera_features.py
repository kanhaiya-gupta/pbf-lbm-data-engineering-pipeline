"""
Camera Feature Engineering

This module extracts and engineers features from camera sensor data for PBF-LB/M processes.
Integrates with YAML configuration for feature definitions and validation rules.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import logging
from ..config.config_manager import ConfigManager

logger = logging.getLogger(__name__)


class CameraFeatures:
    """
    Feature engineering for camera sensor data in PBF-LB/M processes.
    
    Extracts features from image data, visual inspection, and optical measurements
    based on YAML configuration.
    """
    
    def __init__(self, config_manager: Optional[ConfigManager] = None):
        """
        Initialize camera feature engineering.
        
        Args:
            config_manager: Configuration manager for loading feature definitions
        """
        self.config_manager = config_manager or ConfigManager()
        self.feature_config = self._load_feature_config()
        self.feature_definitions = self.feature_config.get('feature_definitions', {})
        self.validation_rules = self.feature_config.get('validation_rules', {})
        
    def _load_feature_config(self) -> Dict[str, Any]:
        """Load camera feature configuration from YAML."""
        try:
            return self.config_manager.load_feature_config('sensor_features/camera_features')
        except Exception as e:
            logger.warning(f"Could not load camera config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for camera features."""
        return {
            'feature_definitions': {
                'image_features': {
                    'aggregations': ['brightness', 'contrast', 'sharpness'],
                    'derived': ['image_quality', 'noise_level']
                },
                'geometric_features': {
                    'aggregations': ['area', 'perimeter', 'aspect_ratio'],
                    'derived': ['shape_complexity', 'symmetry']
                },
                'texture_features': {
                    'aggregations': ['texture_energy', 'texture_contrast'],
                    'derived': ['surface_roughness', 'pattern_regularity']
                }
            },
            'validation_rules': {
                'brightness_range': [0, 255],
                'contrast_range': [0, 100],
                'resolution_range': [100, 10000]
            }
        }
    
    def extract_brightness_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract brightness and illumination features.
        
        Args:
            data: DataFrame with camera brightness data
            
        Returns:
            DataFrame with brightness features
        """
        features = pd.DataFrame(index=data.index)
        
        # Basic brightness features
        if 'brightness' in data.columns:
            brightness_col = data['brightness']
            features['brightness_mean'] = brightness_col.rolling(window=10).mean()
            features['brightness_std'] = brightness_col.rolling(window=10).std()
            features['brightness_min'] = brightness_col.rolling(window=10).min()
            features['brightness_max'] = brightness_col.rolling(window=10).max()
            features['brightness_range'] = features['brightness_max'] - features['brightness_min']
            
            # Brightness categories
            features['brightness_category'] = pd.cut(brightness_col, 
                                                  bins=[0, 50, 100, 150, 200, 255], 
                                                  labels=['very_dark', 'dark', 'medium', 'bright', 'very_bright'])
        
        # Multi-channel brightness features
        brightness_cols = [col for col in data.columns if 'brightness_' in col.lower()]
        if len(brightness_cols) > 1:
            brightness_data = data[brightness_cols]
            features['multi_brightness_mean'] = brightness_data.mean(axis=1)
            features['multi_brightness_std'] = brightness_data.std(axis=1)
            
            # Brightness uniformity
            features['brightness_uniformity'] = 1 / (features['multi_brightness_std'] + 1e-6)
        
        # RGB channel features
        rgb_channels = ['red', 'green', 'blue']
        for channel in rgb_channels:
            if f'{channel}_intensity' in data.columns:
                features[f'{channel}_intensity_mean'] = data[f'{channel}_intensity'].rolling(window=10).mean()
                features[f'{channel}_intensity_std'] = data[f'{channel}_intensity'].rolling(window=10).std()
        
        return features
    
    def extract_contrast_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract contrast and edge features.
        
        Args:
            data: DataFrame with camera contrast data
            
        Returns:
            DataFrame with contrast features
        """
        features = pd.DataFrame(index=data.index)
        
        # Basic contrast features
        if 'contrast' in data.columns:
            contrast_col = data['contrast']
            features['contrast_mean'] = contrast_col.rolling(window=10).mean()
            features['contrast_std'] = contrast_col.rolling(window=10).std()
            features['contrast_min'] = contrast_col.rolling(window=10).min()
            features['contrast_max'] = contrast_col.rolling(window=10).max()
            
            # Contrast categories
            features['contrast_category'] = pd.cut(contrast_col, 
                                                bins=[0, 20, 40, 60, 80, 100], 
                                                labels=['very_low', 'low', 'medium', 'high', 'very_high'])
        
        # Edge detection features
        if 'edge_density' in data.columns:
            features['edge_density_mean'] = data['edge_density'].rolling(window=10).mean()
            features['edge_density_std'] = data['edge_density'].rolling(window=10).std()
            
            # Edge categories
            features['edge_category'] = pd.cut(data['edge_density'], 
                                            bins=[0, 0.1, 0.3, 0.5, 0.7, 1.0], 
                                            labels=['smooth', 'low_detail', 'medium_detail', 'high_detail', 'very_detailed'])
        
        # Gradient features
        if 'gradient_magnitude' in data.columns:
            features['gradient_magnitude_mean'] = data['gradient_magnitude'].rolling(window=10).mean()
            features['gradient_magnitude_std'] = data['gradient_magnitude'].rolling(window=10).std()
        
        return features
    
    def extract_sharpness_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract sharpness and focus features.
        
        Args:
            data: DataFrame with camera sharpness data
            
        Returns:
            DataFrame with sharpness features
        """
        features = pd.DataFrame(index=data.index)
        
        # Basic sharpness features
        if 'sharpness' in data.columns:
            sharpness_col = data['sharpness']
            features['sharpness_mean'] = sharpness_col.rolling(window=10).mean()
            features['sharpness_std'] = sharpness_col.rolling(window=10).std()
            features['sharpness_min'] = sharpness_col.rolling(window=10).min()
            features['sharpness_max'] = sharpness_col.rolling(window=10).max()
            
            # Sharpness categories
            features['sharpness_category'] = pd.cut(sharpness_col, 
                                                 bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0], 
                                                 labels=['very_blurry', 'blurry', 'medium', 'sharp', 'very_sharp'])
        
        # Focus quality features
        if 'focus_score' in data.columns:
            features['focus_score_mean'] = data['focus_score'].rolling(window=10).mean()
            features['focus_score_std'] = data['focus_score'].rolling(window=10).std()
            
            # Focus stability
            features['focus_stability'] = 1 / (features['focus_score_std'] + 1e-6)
        
        # Blur detection features
        if 'blur_level' in data.columns:
            features['blur_level_mean'] = data['blur_level'].rolling(window=10).mean()
            features['blur_level_std'] = data['blur_level'].rolling(window=10).std()
            
            # Blur categories
            features['blur_category'] = pd.cut(data['blur_level'], 
                                            bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0], 
                                            labels=['sharp', 'slight_blur', 'moderate_blur', 'heavy_blur', 'very_blurry'])
        
        return features
    
    def extract_geometric_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract geometric and shape features.
        
        Args:
            data: DataFrame with geometric data
            
        Returns:
            DataFrame with geometric features
        """
        features = pd.DataFrame(index=data.index)
        
        # Area features
        if 'area' in data.columns:
            features['area_mean'] = data['area'].rolling(window=10).mean()
            features['area_std'] = data['area'].rolling(window=10).std()
            features['area_min'] = data['area'].rolling(window=10).min()
            features['area_max'] = data['area'].rolling(window=10).max()
        
        # Perimeter features
        if 'perimeter' in data.columns:
            features['perimeter_mean'] = data['perimeter'].rolling(window=10).mean()
            features['perimeter_std'] = data['perimeter'].rolling(window=10).std()
        
        # Aspect ratio features
        if 'aspect_ratio' in data.columns:
            features['aspect_ratio_mean'] = data['aspect_ratio'].rolling(window=10).mean()
            features['aspect_ratio_std'] = data['aspect_ratio'].rolling(window=10).std()
            
            # Shape categories
            features['shape_category'] = pd.cut(data['aspect_ratio'], 
                                             bins=[0, 0.5, 0.8, 1.2, 2.0, 10], 
                                             labels=['elongated', 'rectangular', 'square', 'wide', 'very_wide'])
        
        # Derived geometric features
        if 'area' in data.columns and 'perimeter' in data.columns:
            # Compactness (circularity)
            features['compactness'] = (4 * np.pi * data['area']) / (data['perimeter'] ** 2)
            features['compactness_mean'] = features['compactness'].rolling(window=10).mean()
        
        # Bounding box features
        bbox_cols = ['bbox_width', 'bbox_height', 'bbox_x', 'bbox_y']
        for col in bbox_cols:
            if col in data.columns:
                features[f'{col}_mean'] = data[col].rolling(window=10).mean()
                features[f'{col}_std'] = data[col].rolling(window=10).std()
        
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
        
        # Surface roughness estimation
        if 'texture_energy' in data.columns and 'texture_contrast' in data.columns:
            features['surface_roughness'] = data['texture_energy'] * data['texture_contrast']
            features['surface_roughness_mean'] = features['surface_roughness'].rolling(window=10).mean()
        
        # Pattern regularity features
        if 'pattern_regularity' in data.columns:
            features['pattern_regularity_mean'] = data['pattern_regularity'].rolling(window=10).mean()
            features['pattern_regularity_std'] = data['pattern_regularity'].rolling(window=10).std()
            
            # Pattern categories
            features['pattern_category'] = pd.cut(data['pattern_regularity'], 
                                               bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0], 
                                               labels=['irregular', 'low_regular', 'medium_regular', 'high_regular', 'very_regular'])
        
        return features
    
    def extract_quality_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract image quality and noise features.
        
        Args:
            data: DataFrame with quality data
            
        Returns:
            DataFrame with quality features
        """
        features = pd.DataFrame(index=data.index)
        
        # Noise level features
        if 'noise_level' in data.columns:
            features['noise_level_mean'] = data['noise_level'].rolling(window=10).mean()
            features['noise_level_std'] = data['noise_level'].rolling(window=10).std()
            
            # Noise categories
            features['noise_category'] = pd.cut(data['noise_level'], 
                                             bins=[0, 0.1, 0.3, 0.5, 0.7, 1.0], 
                                             labels=['clean', 'low_noise', 'medium_noise', 'high_noise', 'very_noisy'])
        
        # Signal-to-noise ratio
        if 'signal_strength' in data.columns and 'noise_level' in data.columns:
            features['snr'] = data['signal_strength'] / (data['noise_level'] + 1e-6)
            features['snr_mean'] = features['snr'].rolling(window=10).mean()
            features['snr_std'] = features['snr'].rolling(window=10).std()
        
        # Overall image quality
        quality_components = []
        if 'brightness' in data.columns:
            # Normalize brightness to 0-1 scale
            brightness_norm = (data['brightness'] - data['brightness'].min()) / (data['brightness'].max() - data['brightness'].min())
            quality_components.append(brightness_norm)
        
        if 'contrast' in data.columns:
            # Normalize contrast to 0-1 scale
            contrast_norm = data['contrast'] / 100
            quality_components.append(contrast_norm)
        
        if 'sharpness' in data.columns:
            quality_components.append(data['sharpness'])
        
        if len(quality_components) > 1:
            features['overall_quality'] = np.mean(quality_components, axis=0)
            features['overall_quality_mean'] = features['overall_quality'].rolling(window=10).mean()
            
            # Quality categories
            features['quality_category'] = pd.cut(features['overall_quality'], 
                                               bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0], 
                                               labels=['poor', 'fair', 'good', 'very_good', 'excellent'])
        
        return features
    
    def extract_motion_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract motion and stability features.
        
        Args:
            data: DataFrame with motion data
            
        Returns:
            DataFrame with motion features
        """
        features = pd.DataFrame(index=data.index)
        
        # Motion detection features
        if 'motion_detected' in data.columns:
            features['motion_detected_mean'] = data['motion_detected'].rolling(window=10).mean()
            features['motion_detected_std'] = data['motion_detected'].rolling(window=10).std()
        
        # Motion magnitude features
        if 'motion_magnitude' in data.columns:
            features['motion_magnitude_mean'] = data['motion_magnitude'].rolling(window=10).mean()
            features['motion_magnitude_std'] = data['motion_magnitude'].rolling(window=10).std()
            
            # Motion categories
            features['motion_category'] = pd.cut(data['motion_magnitude'], 
                                              bins=[0, 0.1, 0.3, 0.5, 0.7, 1.0], 
                                              labels=['stable', 'slight_motion', 'moderate_motion', 'high_motion', 'very_high_motion'])
        
        # Camera stability features
        if 'camera_stability' in data.columns:
            features['camera_stability_mean'] = data['camera_stability'].rolling(window=10).mean()
            features['camera_stability_std'] = data['camera_stability'].rolling(window=10).std()
        
        return features
    
    def extract_all_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract all camera features.
        
        Args:
            data: DataFrame with camera data
            
        Returns:
            DataFrame with all extracted features
        """
        logger.info("Extracting camera features...")
        
        # Validate input data
        self._validate_input_data(data)
        
        # Extract different feature groups
        feature_groups = [
            self.extract_brightness_features(data),
            self.extract_contrast_features(data),
            self.extract_sharpness_features(data),
            self.extract_geometric_features(data),
            self.extract_texture_features(data),
            self.extract_quality_features(data),
            self.extract_motion_features(data)
        ]
        
        # Combine all features
        all_features = pd.concat(feature_groups, axis=1)
        
        # Clean and validate features
        all_features = self._clean_features(all_features)
        
        logger.info(f"Extracted {len(all_features.columns)} camera features")
        return all_features
    
    def _validate_input_data(self, data: pd.DataFrame) -> None:
        """Validate input data against configuration rules."""
        validation_rules = self.validation_rules
        
        # Validate brightness data
        if 'brightness' in data.columns:
            brightness_range = validation_rules.get('brightness_range', [0, 255])
            invalid_brightness = (data['brightness'] < brightness_range[0]) | (data['brightness'] > brightness_range[1])
            if invalid_brightness.any():
                logger.warning(f"Found {invalid_brightness.sum()} invalid brightness values")
        
        # Validate contrast data
        if 'contrast' in data.columns:
            contrast_range = validation_rules.get('contrast_range', [0, 100])
            invalid_contrast = (data['contrast'] < contrast_range[0]) | (data['contrast'] > contrast_range[1])
            if invalid_contrast.any():
                logger.warning(f"Found {invalid_contrast.sum()} invalid contrast values")
        
        # Validate resolution data
        if 'resolution' in data.columns:
            resolution_range = validation_rules.get('resolution_range', [100, 10000])
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
        Calculate feature importance for camera data.
        
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
