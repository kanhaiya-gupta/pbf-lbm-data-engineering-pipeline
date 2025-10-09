"""
Defect Image Feature Engineering

This module extracts and engineers features from defect image data for PBF-LB/M processes.
Integrates with YAML configuration for feature definitions and validation rules.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import logging
from ..config.config_manager import ConfigManager

logger = logging.getLogger(__name__)


class DefectImageFeatures:
    """
    Feature engineering for defect image data in PBF-LB/M processes.
    
    Extracts features from defect detection, classification, and severity assessment
    based on YAML configuration.
    """
    
    def __init__(self, config_manager: Optional[ConfigManager] = None):
        """
        Initialize defect image feature engineering.
        
        Args:
            config_manager: Configuration manager for loading feature definitions
        """
        self.config_manager = config_manager or ConfigManager()
        self.feature_config = self._load_feature_config()
        self.feature_definitions = self.feature_config.get('feature_definitions', {})
        self.validation_rules = self.feature_config.get('validation_rules', {})
        
    def _load_feature_config(self) -> Dict[str, Any]:
        """Load defect image feature configuration from YAML."""
        try:
            return self.config_manager.load_feature_config('image_features/defect_image_features')
        except Exception as e:
            logger.warning(f"Could not load defect image config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for defect image features."""
        return {
            'feature_definitions': {
                'detection_features': {
                    'aggregations': ['defect_count', 'detection_confidence'],
                    'derived': ['defect_density', 'detection_accuracy']
                },
                'classification_features': {
                    'aggregations': ['defect_type', 'severity_level'],
                    'derived': ['type_distribution', 'severity_distribution']
                },
                'morphology_features': {
                    'aggregations': ['area', 'perimeter', 'aspect_ratio'],
                    'derived': ['shape_complexity', 'defect_characteristics']
                }
            },
            'validation_rules': {
                'confidence_range': [0, 1],
                'severity_range': [0, 10],
                'area_range': [0, 10000]  # pixels
            }
        }
    
    def extract_detection_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract defect detection features.
        
        Args:
            data: DataFrame with defect detection data
            
        Returns:
            DataFrame with detection features
        """
        features = pd.DataFrame(index=data.index)
        
        # Defect count features
        if 'defect_count' in data.columns:
            defect_count = data['defect_count']
            features['defect_count_mean'] = defect_count.rolling(window=10).mean()
            features['defect_count_std'] = defect_count.rolling(window=10).std()
            features['defect_count_max'] = defect_count.rolling(window=10).max()
            features['defect_count_min'] = defect_count.rolling(window=10).min()
            
            # Defect count categories
            features['defect_count_category'] = pd.cut(defect_count, 
                                                     bins=[0, 1, 5, 10, 20, 100], 
                                                     labels=['none', 'few', 'moderate', 'many', 'excessive'])
        
        # Detection confidence features
        if 'detection_confidence' in data.columns:
            confidence_col = data['detection_confidence']
            features['confidence_mean'] = confidence_col.rolling(window=10).mean()
            features['confidence_std'] = confidence_col.rolling(window=10).std()
            features['confidence_min'] = confidence_col.rolling(window=10).min()
            features['confidence_max'] = confidence_col.rolling(window=10).max()
            
            # Confidence categories
            features['confidence_category'] = pd.cut(confidence_col, 
                                                  bins=[0, 0.3, 0.5, 0.7, 0.8, 1.0], 
                                                  labels=['low', 'fair', 'good', 'high', 'very_high'])
        
        # Detection accuracy features
        if 'detection_accuracy' in data.columns:
            features['accuracy_mean'] = data['detection_accuracy'].rolling(window=10).mean()
            features['accuracy_std'] = data['detection_accuracy'].rolling(window=10).std()
            
            # Accuracy categories
            features['accuracy_category'] = pd.cut(data['detection_accuracy'], 
                                                bins=[0, 0.5, 0.7, 0.8, 0.9, 1.0], 
                                                labels=['poor', 'fair', 'good', 'very_good', 'excellent'])
        
        # Defect density features
        if 'defect_count' in data.columns and 'image_area' in data.columns:
            features['defect_density'] = data['defect_count'] / data['image_area']
            features['defect_density_mean'] = features['defect_density'].rolling(window=10).mean()
            features['defect_density_std'] = features['defect_density'].rolling(window=10).std()
            
            # Density categories
            features['density_category'] = pd.cut(features['defect_density'], 
                                               bins=[0, 0.001, 0.01, 0.1, 1.0, 10], 
                                               labels=['minimal', 'low', 'moderate', 'high', 'very_high'])
        
        return features
    
    def extract_classification_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract defect classification features.
        
        Args:
            data: DataFrame with defect classification data
            
        Returns:
            DataFrame with classification features
        """
        features = pd.DataFrame(index=data.index)
        
        # Defect type features
        defect_types = ['pore', 'crack', 'inclusion', 'surface_roughness', 'dimensional_error']
        for defect_type in defect_types:
            if f'{defect_type}_count' in data.columns:
                features[f'{defect_type}_count_mean'] = data[f'{defect_type}_count'].rolling(window=10).mean()
                features[f'{defect_type}_count_std'] = data[f'{defect_type}_count'].rolling(window=10).std()
                features[f'{defect_type}_count_max'] = data[f'{defect_type}_count'].rolling(window=10).max()
        
        # Severity level features
        if 'severity_level' in data.columns:
            severity_col = data['severity_level']
            features['severity_mean'] = severity_col.rolling(window=10).mean()
            features['severity_std'] = severity_col.rolling(window=10).std()
            features['severity_max'] = severity_col.rolling(window=10).max()
            features['severity_min'] = severity_col.rolling(window=10).min()
            
            # Severity categories
            features['severity_category'] = pd.cut(severity_col, 
                                                bins=[0, 2, 4, 6, 8, 10], 
                                                labels=['minimal', 'low', 'moderate', 'high', 'critical'])
        
        # Type distribution features
        type_cols = [col for col in data.columns if col.endswith('_count')]
        if len(type_cols) > 1:
            type_data = data[type_cols]
            total_defects = type_data.sum(axis=1)
            
            for col in type_cols:
                defect_type = col.replace('_count', '')
                features[f'{defect_type}_ratio'] = type_data[col] / (total_defects + 1e-6)
                features[f'{defect_type}_ratio_mean'] = features[f'{defect_type}_ratio'].rolling(window=10).mean()
        
        # Severity distribution features
        severity_levels = ['low_severity', 'medium_severity', 'high_severity', 'critical_severity']
        for severity in severity_levels:
            if f'{severity}_count' in data.columns:
                features[f'{severity}_count_mean'] = data[f'{severity}_count'].rolling(window=10).mean()
                features[f'{severity}_count_std'] = data[f'{severity}_count'].rolling(window=10).std()
        
        return features
    
    def extract_morphology_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract defect morphology features.
        
        Args:
            data: DataFrame with morphology data
            
        Returns:
            DataFrame with morphology features
        """
        features = pd.DataFrame(index=data.index)
        
        # Area features
        if 'defect_area' in data.columns:
            area_col = data['defect_area']
            features['area_mean'] = area_col.rolling(window=10).mean()
            features['area_std'] = area_col.rolling(window=10).std()
            features['area_max'] = area_col.rolling(window=10).max()
            features['area_min'] = area_col.rolling(window=10).min()
            
            # Area categories
            features['area_category'] = pd.cut(area_col, 
                                            bins=[0, 10, 50, 100, 500, 10000], 
                                            labels=['tiny', 'small', 'medium', 'large', 'very_large'])
        
        # Perimeter features
        if 'defect_perimeter' in data.columns:
            features['perimeter_mean'] = data['defect_perimeter'].rolling(window=10).mean()
            features['perimeter_std'] = data['defect_perimeter'].rolling(window=10).std()
            features['perimeter_max'] = data['defect_perimeter'].rolling(window=10).max()
        
        # Aspect ratio features
        if 'aspect_ratio' in data.columns:
            features['aspect_ratio_mean'] = data['aspect_ratio'].rolling(window=10).mean()
            features['aspect_ratio_std'] = data['aspect_ratio'].rolling(window=10).std()
            
            # Shape categories
            features['shape_category'] = pd.cut(data['aspect_ratio'], 
                                             bins=[0, 0.5, 0.8, 1.2, 2.0, 10], 
                                             labels=['elongated', 'rectangular', 'square', 'wide', 'very_wide'])
        
        # Compactness features
        if 'defect_area' in data.columns and 'defect_perimeter' in data.columns:
            features['compactness'] = (4 * np.pi * data['defect_area']) / (data['defect_perimeter'] ** 2)
            features['compactness_mean'] = features['compactness'].rolling(window=10).mean()
            features['compactness_std'] = features['compactness'].rolling(window=10).std()
        
        # Shape complexity features
        if 'defect_area' in data.columns and 'defect_perimeter' in data.columns:
            features['shape_complexity'] = data['defect_perimeter'] / np.sqrt(data['defect_area'])
            features['shape_complexity_mean'] = features['shape_complexity'].rolling(window=10).mean()
            
            # Complexity categories
            features['complexity_category'] = pd.cut(features['shape_complexity'], 
                                                  bins=[0, 2, 4, 6, 8, 20], 
                                                  labels=['simple', 'moderate', 'complex', 'very_complex', 'extremely_complex'])
        
        # Circularity features
        if 'compactness' in features.columns:
            features['circularity'] = features['compactness']
            features['circularity_mean'] = features['circularity'].rolling(window=10).mean()
            
            # Circularity categories
            features['circularity_category'] = pd.cut(features['circularity'], 
                                                   bins=[0, 0.3, 0.5, 0.7, 0.8, 1.0], 
                                                   labels=['irregular', 'angular', 'moderate', 'round', 'circular'])
        
        return features
    
    def extract_severity_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract defect severity assessment features.
        
        Args:
            data: DataFrame with severity data
            
        Returns:
            DataFrame with severity features
        """
        features = pd.DataFrame(index=data.index)
        
        # Overall severity score
        severity_components = []
        
        # Size-based severity
        if 'defect_area' in data.columns:
            # Normalize area to 0-1 scale
            area_severity = data['defect_area'] / data['defect_area'].max()
            severity_components.append(area_severity)
        
        # Count-based severity
        if 'defect_count' in data.columns:
            # Normalize count to 0-1 scale
            count_severity = data['defect_count'] / data['defect_count'].max()
            severity_components.append(count_severity)
        
        # Type-based severity weights
        defect_type_weights = {
            'pore': 0.3,
            'crack': 0.8,
            'inclusion': 0.6,
            'surface_roughness': 0.2,
            'dimensional_error': 0.7
        }
        
        for defect_type, weight in defect_type_weights.items():
            if f'{defect_type}_count' in data.columns:
                type_severity = data[f'{defect_type}_count'] * weight
                severity_components.append(type_severity)
        
        if len(severity_components) > 1:
            features['overall_severity'] = np.mean(severity_components, axis=0)
            features['overall_severity_mean'] = features['overall_severity'].rolling(window=10).mean()
            
            # Severity categories
            features['overall_severity_category'] = pd.cut(features['overall_severity'], 
                                                         bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0], 
                                                         labels=['minimal', 'low', 'moderate', 'high', 'critical'])
        
        # Critical defect ratio
        if 'critical_defect_count' in data.columns and 'defect_count' in data.columns:
            features['critical_ratio'] = data['critical_defect_count'] / (data['defect_count'] + 1e-6)
            features['critical_ratio_mean'] = features['critical_ratio'].rolling(window=10).mean()
        
        # Severity trend
        if 'overall_severity' in features.columns:
            features['severity_trend'] = features['overall_severity'].diff().rolling(window=5).mean()
        
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
        
        # Overall quality score (inverted from defects)
        quality_components = []
        
        # Defect count component (inverted)
        if 'defect_count' in data.columns:
            defect_quality = 1 / (data['defect_count'] + 1e-6)
            quality_components.append(defect_quality)
        
        # Severity component (inverted)
        if 'overall_severity' in features.columns:
            severity_quality = 1 - features['overall_severity']
            quality_components.append(severity_quality)
        
        # Detection confidence component
        if 'detection_confidence' in data.columns:
            quality_components.append(data['detection_confidence'])
        
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
        Extract all defect image features.
        
        Args:
            data: DataFrame with defect image data
            
        Returns:
            DataFrame with all extracted features
        """
        logger.info("Extracting defect image features...")
        
        # Validate input data
        self._validate_input_data(data)
        
        # Extract different feature groups
        feature_groups = [
            self.extract_detection_features(data),
            self.extract_classification_features(data),
            self.extract_morphology_features(data),
            self.extract_severity_features(data),
            self.extract_quality_features(data)
        ]
        
        # Combine all features
        all_features = pd.concat(feature_groups, axis=1)
        
        # Clean and validate features
        all_features = self._clean_features(all_features)
        
        logger.info(f"Extracted {len(all_features.columns)} defect image features")
        return all_features
    
    def _validate_input_data(self, data: pd.DataFrame) -> None:
        """Validate input data against configuration rules."""
        validation_rules = self.validation_rules
        
        # Validate confidence data
        if 'detection_confidence' in data.columns:
            confidence_range = validation_rules.get('confidence_range', [0, 1])
            invalid_confidence = (data['detection_confidence'] < confidence_range[0]) | (data['detection_confidence'] > confidence_range[1])
            if invalid_confidence.any():
                logger.warning(f"Found {invalid_confidence.sum()} invalid confidence values")
        
        # Validate severity data
        if 'severity_level' in data.columns:
            severity_range = validation_rules.get('severity_range', [0, 10])
            invalid_severity = (data['severity_level'] < severity_range[0]) | (data['severity_level'] > severity_range[1])
            if invalid_severity.any():
                logger.warning(f"Found {invalid_severity.sum()} invalid severity values")
        
        # Validate area data
        if 'defect_area' in data.columns:
            area_range = validation_rules.get('area_range', [0, 10000])
            invalid_area = (data['defect_area'] < area_range[0]) | (data['defect_area'] > area_range[1])
            if invalid_area.any():
                logger.warning(f"Found {invalid_area.sum()} invalid area values")
    
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
        Calculate feature importance for defect image data.
        
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
