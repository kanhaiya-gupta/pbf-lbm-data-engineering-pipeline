"""
Material Feature Engineering

This module extracts and engineers features from material properties for PBF-LB/M processes.
Integrates with YAML configuration for feature definitions and validation rules.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import logging
from ..config.config_manager import ConfigManager

logger = logging.getLogger(__name__)


class MaterialFeatures:
    """
    Feature engineering for material properties in PBF-LB/M processes.
    
    Extracts features from material composition, properties, and behavior
    based on YAML configuration definitions.
    """
    
    def __init__(self, config_manager: Optional[ConfigManager] = None):
        """
        Initialize material feature engineering.
        
        Args:
            config_manager: Configuration manager for loading feature definitions
        """
        self.config_manager = config_manager or ConfigManager()
        self.feature_config = self._load_feature_config()
        self.feature_definitions = self.feature_config.get('feature_definitions', {})
        self.validation_rules = self.feature_config.get('validation_rules', {})
        
    def _load_feature_config(self) -> Dict[str, Any]:
        """Load material feature configuration from YAML."""
        try:
            return self.config_manager.load_feature_config('process_features/material_features')
        except Exception as e:
            logger.warning(f"Could not load material config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for material features."""
        return {
            'feature_definitions': {
                'composition_features': {
                    'aggregations': ['mean', 'std', 'min', 'max'],
                    'derived': ['alloy_ratio', 'impurity_level']
                },
                'thermal_features': {
                    'aggregations': ['mean', 'std'],
                    'derived': ['thermal_diffusivity', 'cooling_rate']
                },
                'mechanical_features': {
                    'aggregations': ['mean', 'std'],
                    'derived': ['strength_ratio', 'ductility_index']
                }
            },
            'validation_rules': {
                'composition_range': [0, 100],  # percentage
                'temperature_range': [20, 2000],  # Celsius
                'density_range': [1, 20]  # g/cm³
            }
        }
    
    def extract_composition_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract material composition features.
        
        Args:
            data: DataFrame with composition data
            
        Returns:
            DataFrame with composition features
        """
        features = pd.DataFrame(index=data.index)
        
        # Element composition features
        element_cols = [col for col in data.columns if col.startswith('element_')]
        for col in element_cols:
            element_name = col.replace('element_', '')
            features[f'{element_name}_content_mean'] = data[col].rolling(window=10).mean()
            features[f'{element_name}_content_std'] = data[col].rolling(window=10).std()
            features[f'{element_name}_content_min'] = data[col].rolling(window=10).min()
            features[f'{element_name}_content_max'] = data[col].rolling(window=10).max()
        
        # Alloy ratio features
        if 'element_Fe' in data.columns and 'element_C' in data.columns:
            features['Fe_C_ratio'] = data['element_Fe'] / (data['element_C'] + 1e-6)
        
        if 'element_Al' in data.columns and 'element_Si' in data.columns:
            features['Al_Si_ratio'] = data['element_Al'] / (data['element_Si'] + 1e-6)
        
        # Impurity level calculation
        impurity_elements = ['element_O', 'element_N', 'element_H', 'element_S', 'element_P']
        impurity_cols = [col for col in impurity_elements if col in data.columns]
        if impurity_cols:
            features['total_impurity_level'] = data[impurity_cols].sum(axis=1)
            features['impurity_level_mean'] = features['total_impurity_level'].rolling(window=10).mean()
        
        # Main alloy content
        if 'element_Fe' in data.columns:
            features['main_alloy_content'] = data['element_Fe']
        elif 'element_Al' in data.columns:
            features['main_alloy_content'] = data['element_Al']
        elif 'element_Ti' in data.columns:
            features['main_alloy_content'] = data['element_Ti']
        
        return features
    
    def extract_thermal_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract thermal property features.
        
        Args:
            data: DataFrame with thermal data
            
        Returns:
            DataFrame with thermal features
        """
        features = pd.DataFrame(index=data.index)
        
        # Basic thermal properties
        thermal_props = ['melting_point', 'boiling_point', 'thermal_conductivity', 'specific_heat']
        for prop in thermal_props:
            if prop in data.columns:
                features[f'{prop}_mean'] = data[prop].rolling(window=10).mean()
                features[f'{prop}_std'] = data[prop].rolling(window=10).std()
        
        # Temperature-dependent features
        if 'temperature' in data.columns:
            features['temperature_mean'] = data['temperature'].rolling(window=10).mean()
            features['temperature_std'] = data['temperature'].rolling(window=10).std()
            features['temperature_range'] = data['temperature'].rolling(window=10).max() - data['temperature'].rolling(window=10).min()
            
            # Temperature stability
            features['temperature_stability'] = 1 / (features['temperature_std'] + 1e-6)
        
        # Derived thermal features
        if 'thermal_conductivity' in data.columns and 'density' in data.columns and 'specific_heat' in data.columns:
            features['thermal_diffusivity'] = data['thermal_conductivity'] / (data['density'] * data['specific_heat'])
        
        if 'temperature' in data.columns and 'time' in data.columns:
            features['cooling_rate'] = data['temperature'].diff() / data['time'].diff()
            features['cooling_rate_mean'] = features['cooling_rate'].rolling(window=10).mean()
        
        # Phase transition features
        if 'melting_point' in data.columns and 'temperature' in data.columns:
            features['melting_ratio'] = data['temperature'] / data['melting_point']
            features['above_melting'] = (data['temperature'] > data['melting_point']).astype(int)
        
        return features
    
    def extract_mechanical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract mechanical property features.
        
        Args:
            data: DataFrame with mechanical data
            
        Returns:
            DataFrame with mechanical features
        """
        features = pd.DataFrame(index=data.index)
        
        # Basic mechanical properties
        mechanical_props = ['tensile_strength', 'yield_strength', 'elongation', 'hardness', 'modulus']
        for prop in mechanical_props:
            if prop in data.columns:
                features[f'{prop}_mean'] = data[prop].rolling(window=10).mean()
                features[f'{prop}_std'] = data[prop].rolling(window=10).std()
        
        # Strength ratio features
        if 'tensile_strength' in data.columns and 'yield_strength' in data.columns:
            features['strength_ratio'] = data['tensile_strength'] / (data['yield_strength'] + 1e-6)
            features['strength_ratio_mean'] = features['strength_ratio'].rolling(window=10).mean()
        
        # Ductility features
        if 'elongation' in data.columns:
            features['ductility_index'] = data['elongation'] / 100  # Convert percentage to ratio
            features['ductility_index_mean'] = features['ductility_index'].rolling(window=10).mean()
        
        # Hardness features
        if 'hardness' in data.columns:
            features['hardness_mean'] = data['hardness'].rolling(window=10).mean()
            features['hardness_std'] = data['hardness'].rolling(window=10).std()
            
            # Hardness categories
            features['hardness_category'] = pd.cut(data['hardness'], 
                                                 bins=[0, 200, 400, 600, 1000], 
                                                 labels=['soft', 'medium', 'hard', 'very_hard'])
        
        # Modulus features
        if 'modulus' in data.columns:
            features['modulus_mean'] = data['modulus'].rolling(window=10).mean()
            features['modulus_std'] = data['modulus'].rolling(window=10).std()
        
        return features
    
    def extract_physical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract physical property features.
        
        Args:
            data: DataFrame with physical data
            
        Returns:
            DataFrame with physical features
        """
        features = pd.DataFrame(index=data.index)
        
        # Basic physical properties
        physical_props = ['density', 'viscosity', 'surface_tension', 'electrical_conductivity']
        for prop in physical_props:
            if prop in data.columns:
                features[f'{prop}_mean'] = data[prop].rolling(window=10).mean()
                features[f'{prop}_std'] = data[prop].rolling(window=10).std()
        
        # Density features
        if 'density' in data.columns:
            features['density_mean'] = data['density'].rolling(window=10).mean()
            features['density_std'] = data['density'].rolling(window=10).std()
            
            # Density categories
            features['density_category'] = pd.cut(data['density'], 
                                                bins=[0, 3, 5, 8, 20], 
                                                labels=['light', 'medium', 'heavy', 'very_heavy'])
        
        # Viscosity features
        if 'viscosity' in data.columns:
            features['viscosity_mean'] = data['viscosity'].rolling(window=10).mean()
            features['viscosity_std'] = data['viscosity'].rolling(window=10).std()
            
            # Viscosity index
            features['viscosity_index'] = np.log10(data['viscosity'] + 1e-6)
        
        # Surface tension features
        if 'surface_tension' in data.columns:
            features['surface_tension_mean'] = data['surface_tension'].rolling(window=10).mean()
            features['surface_tension_std'] = data['surface_tension'].rolling(window=10).std()
        
        return features
    
    def extract_processing_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract material processing behavior features.
        
        Args:
            data: DataFrame with processing data
            
        Returns:
            DataFrame with processing features
        """
        features = pd.DataFrame(index=data.index)
        
        # Powder properties
        powder_props = ['powder_size', 'powder_flowability', 'powder_density', 'powder_porosity']
        for prop in powder_props:
            if prop in data.columns:
                features[f'{prop}_mean'] = data[prop].rolling(window=10).mean()
                features[f'{prop}_std'] = data[prop].rolling(window=10).std()
        
        # Powder size features
        if 'powder_size' in data.columns:
            features['powder_size_mean'] = data['powder_size'].rolling(window=10).mean()
            features['powder_size_std'] = data['powder_size'].rolling(window=10).std()
            
            # Size distribution features
            features['size_uniformity'] = 1 / (features['powder_size_std'] + 1e-6)
        
        # Flowability features
        if 'powder_flowability' in data.columns:
            features['flowability_mean'] = data['powder_flowability'].rolling(window=10).mean()
            features['flowability_std'] = data['powder_flowability'].rolling(window=10).std()
            
            # Flowability categories
            features['flowability_category'] = pd.cut(data['powder_flowability'], 
                                                    bins=[0, 20, 40, 60, 100], 
                                                    labels=['poor', 'fair', 'good', 'excellent'])
        
        # Porosity features
        if 'powder_porosity' in data.columns:
            features['porosity_mean'] = data['powder_porosity'].rolling(window=10).mean()
            features['porosity_std'] = data['powder_porosity'].rolling(window=10).std()
            
            # Porosity impact
            features['porosity_impact'] = data['powder_porosity'] * 0.1  # Simplified impact factor
        
        return features
    
    def extract_interaction_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract interaction features between material properties.
        
        Args:
            data: DataFrame with material data
            
        Returns:
            DataFrame with interaction features
        """
        features = pd.DataFrame(index=data.index)
        
        # Thermal-mechanical interactions
        if 'thermal_conductivity' in data.columns and 'tensile_strength' in data.columns:
            features['thermal_strength_ratio'] = data['thermal_conductivity'] / (data['tensile_strength'] + 1e-6)
        
        if 'melting_point' in data.columns and 'hardness' in data.columns:
            features['melting_hardness_ratio'] = data['melting_point'] / (data['hardness'] + 1e-6)
        
        # Composition-property interactions
        if 'element_C' in data.columns and 'hardness' in data.columns:
            features['carbon_hardness_ratio'] = data['element_C'] / (data['hardness'] + 1e-6)
        
        if 'element_Fe' in data.columns and 'density' in data.columns:
            features['iron_density_ratio'] = data['element_Fe'] / (data['density'] + 1e-6)
        
        # Processing-property interactions
        if 'powder_size' in data.columns and 'surface_tension' in data.columns:
            features['size_tension_ratio'] = data['powder_size'] / (data['surface_tension'] + 1e-6)
        
        if 'viscosity' in data.columns and 'flowability' in data.columns:
            features['viscosity_flow_ratio'] = data['viscosity'] / (data['flowability'] + 1e-6)
        
        return features
    
    def extract_all_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract all material features.
        
        Args:
            data: DataFrame with material data
            
        Returns:
            DataFrame with all extracted features
        """
        logger.info("Extracting material features...")
        
        # Validate input data
        self._validate_input_data(data)
        
        # Extract different feature groups
        feature_groups = [
            self.extract_composition_features(data),
            self.extract_thermal_features(data),
            self.extract_mechanical_features(data),
            self.extract_physical_features(data),
            self.extract_processing_features(data),
            self.extract_interaction_features(data)
        ]
        
        # Combine all features
        all_features = pd.concat(feature_groups, axis=1)
        
        # Clean and validate features
        all_features = self._clean_features(all_features)
        
        logger.info(f"Extracted {len(all_features.columns)} material features")
        return all_features
    
    def _validate_input_data(self, data: pd.DataFrame) -> None:
        """Validate input data against configuration rules."""
        validation_rules = self.validation_rules
        
        # Validate composition data
        element_cols = [col for col in data.columns if col.startswith('element_')]
        for col in element_cols:
            composition_range = validation_rules.get('composition_range', [0, 100])
            invalid_values = (data[col] < composition_range[0]) | (data[col] > composition_range[1])
            if invalid_values.any():
                logger.warning(f"Found {invalid_values.sum()} invalid {col} values")
        
        # Validate temperature data
        if 'temperature' in data.columns:
            temp_range = validation_rules.get('temperature_range', [20, 2000])
            invalid_temp = (data['temperature'] < temp_range[0]) | (data['temperature'] > temp_range[1])
            if invalid_temp.any():
                logger.warning(f"Found {invalid_temp.sum()} invalid temperature values")
        
        # Validate density data
        if 'density' in data.columns:
            density_range = validation_rules.get('density_range', [1, 20])
            invalid_density = (data['density'] < density_range[0]) | (data['density'] > density_range[1])
            if invalid_density.any():
                logger.warning(f"Found {invalid_density.sum()} invalid density values")
    
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
        Calculate feature importance for material properties.
        
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
