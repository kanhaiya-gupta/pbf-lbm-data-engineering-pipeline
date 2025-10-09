"""
Environmental Feature Engineering

This module extracts and engineers features from environmental conditions for PBF-LB/M processes.
Integrates with YAML configuration for feature definitions and validation rules.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import logging
from ..config.config_manager import ConfigManager

logger = logging.getLogger(__name__)


class EnvironmentalFeatures:
    """
    Feature engineering for environmental conditions in PBF-LB/M processes.
    
    Extracts features from temperature, humidity, pressure, gas composition,
    and other environmental factors based on YAML configuration.
    """
    
    def __init__(self, config_manager: Optional[ConfigManager] = None):
        """
        Initialize environmental feature engineering.
        
        Args:
            config_manager: Configuration manager for loading feature definitions
        """
        self.config_manager = config_manager or ConfigManager()
        self.feature_config = self._load_feature_config()
        self.feature_definitions = self.feature_config.get('feature_definitions', {})
        self.validation_rules = self.feature_config.get('validation_rules', {})
        
    def _load_feature_config(self) -> Dict[str, Any]:
        """Load environmental feature configuration from YAML."""
        try:
            return self.config_manager.load_feature_config('process_features/environmental_features')
        except Exception as e:
            logger.warning(f"Could not load environmental config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for environmental features."""
        return {
            'feature_definitions': {
                'temperature_features': {
                    'aggregations': ['mean', 'std', 'min', 'max', 'range'],
                    'temporal': ['trend', 'volatility', 'stability']
                },
                'humidity_features': {
                    'aggregations': ['mean', 'std', 'min', 'max'],
                    'derived': ['dew_point', 'humidity_index']
                },
                'pressure_features': {
                    'aggregations': ['mean', 'std', 'min', 'max'],
                    'derived': ['pressure_gradient', 'stability_index']
                },
                'gas_features': {
                    'aggregations': ['concentration', 'purity', 'flow_rate'],
                    'derived': ['gas_quality', 'contamination_level']
                }
            },
            'validation_rules': {
                'temperature_range': [-50, 200],  # Celsius
                'humidity_range': [0, 100],  # percentage
                'pressure_range': [0.1, 2.0],  # bar
                'oxygen_range': [0, 21]  # percentage
            }
        }
    
    def extract_temperature_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract temperature-related features.
        
        Args:
            data: DataFrame with temperature data
            
        Returns:
            DataFrame with temperature features
        """
        features = pd.DataFrame(index=data.index)
        
        # Basic temperature features
        temp_cols = [col for col in data.columns if 'temperature' in col.lower()]
        for col in temp_cols:
            temp_name = col.replace('temperature_', '').replace('_temperature', '')
            features[f'{temp_name}_temp_mean'] = data[col].rolling(window=10).mean()
            features[f'{temp_name}_temp_std'] = data[col].rolling(window=10).std()
            features[f'{temp_name}_temp_min'] = data[col].rolling(window=10).min()
            features[f'{temp_name}_temp_max'] = data[col].rolling(window=10).max()
            features[f'{temp_name}_temp_range'] = features[f'{temp_name}_temp_max'] - features[f'{temp_name}_temp_min']
            
            # Temporal features
            features[f'{temp_name}_temp_trend'] = data[col].diff().rolling(window=5).mean()
            features[f'{temp_name}_temp_volatility'] = data[col].rolling(window=10).std()
            features[f'{temp_name}_temp_stability'] = 1 / (features[f'{temp_name}_temp_volatility'] + 1e-6)
        
        # Chamber temperature features
        if 'chamber_temperature' in data.columns:
            features['chamber_temp_mean'] = data['chamber_temperature'].rolling(window=10).mean()
            features['chamber_temp_std'] = data['chamber_temperature'].rolling(window=10).std()
            
            # Temperature zones
            features['chamber_temp_zone'] = pd.cut(data['chamber_temperature'], 
                                                bins=[0, 50, 100, 150, 200], 
                                                labels=['cold', 'cool', 'warm', 'hot'])
        
        # Build plate temperature
        if 'build_plate_temperature' in data.columns:
            features['plate_temp_mean'] = data['build_plate_temperature'].rolling(window=10).mean()
            features['plate_temp_std'] = data['build_plate_temperature'].rolling(window=10).std()
            
            # Temperature difference with chamber
            if 'chamber_temperature' in data.columns:
                features['temp_difference'] = data['chamber_temperature'] - data['build_plate_temperature']
                features['temp_difference_mean'] = features['temp_difference'].rolling(window=10).mean()
        
        return features
    
    def extract_humidity_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract humidity-related features.
        
        Args:
            data: DataFrame with humidity data
            
        Returns:
            DataFrame with humidity features
        """
        features = pd.DataFrame(index=data.index)
        
        if 'humidity' in data.columns:
            features['humidity_mean'] = data['humidity'].rolling(window=10).mean()
            features['humidity_std'] = data['humidity'].rolling(window=10).std()
            features['humidity_min'] = data['humidity'].rolling(window=10).min()
            features['humidity_max'] = data['humidity'].rolling(window=10).max()
            
            # Humidity categories
            features['humidity_category'] = pd.cut(data['humidity'], 
                                                bins=[0, 30, 50, 70, 100], 
                                                labels=['dry', 'comfortable', 'humid', 'very_humid'])
            
            # Dew point calculation (simplified)
            if 'chamber_temperature' in data.columns:
                # Magnus formula approximation
                features['dew_point'] = data['chamber_temperature'] - ((100 - data['humidity']) / 5)
                features['dew_point_mean'] = features['dew_point'].rolling(window=10).mean()
            
            # Humidity index
            features['humidity_index'] = data['humidity'] / 100
            features['humidity_index_mean'] = features['humidity_index'].rolling(window=10).mean()
        
        return features
    
    def extract_pressure_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract pressure-related features.
        
        Args:
            data: DataFrame with pressure data
            
        Returns:
            DataFrame with pressure features
        """
        features = pd.DataFrame(index=data.index)
        
        if 'chamber_pressure' in data.columns:
            features['pressure_mean'] = data['chamber_pressure'].rolling(window=10).mean()
            features['pressure_std'] = data['chamber_pressure'].rolling(window=10).std()
            features['pressure_min'] = data['chamber_pressure'].rolling(window=10).min()
            features['pressure_max'] = data['chamber_pressure'].rolling(window=10).max()
            
            # Pressure gradient
            features['pressure_gradient'] = data['chamber_pressure'].diff()
            features['pressure_gradient_mean'] = features['pressure_gradient'].rolling(window=10).mean()
            
            # Pressure stability
            features['pressure_stability'] = 1 / (features['pressure_std'] + 1e-6)
            
            # Pressure categories
            features['pressure_category'] = pd.cut(data['chamber_pressure'], 
                                                bins=[0, 0.5, 1.0, 1.5, 2.0], 
                                                labels=['low', 'normal', 'high', 'very_high'])
        
        # Atmospheric pressure comparison
        if 'atmospheric_pressure' in data.columns and 'chamber_pressure' in data.columns:
            features['pressure_difference'] = data['chamber_pressure'] - data['atmospheric_pressure']
            features['pressure_ratio'] = data['chamber_pressure'] / (data['atmospheric_pressure'] + 1e-6)
        
        return features
    
    def extract_gas_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract gas composition and flow features.
        
        Args:
            data: DataFrame with gas data
            
        Returns:
            DataFrame with gas features
        """
        features = pd.DataFrame(index=data.index)
        
        # Gas composition features
        gas_cols = [col for col in data.columns if any(gas in col.lower() for gas in ['oxygen', 'nitrogen', 'argon', 'helium'])]
        for col in gas_cols:
            gas_name = col.replace('_concentration', '').replace('_percentage', '')
            features[f'{gas_name}_concentration_mean'] = data[col].rolling(window=10).mean()
            features[f'{gas_name}_concentration_std'] = data[col].rolling(window=10).std()
            features[f'{gas_name}_concentration_min'] = data[col].rolling(window=10).min()
            features[f'{gas_name}_concentration_max'] = data[col].rolling(window=10).max()
        
        # Oxygen content features
        if 'oxygen_concentration' in data.columns:
            features['oxygen_mean'] = data['oxygen_concentration'].rolling(window=10).mean()
            features['oxygen_std'] = data['oxygen_concentration'].rolling(window=10).std()
            
            # Oxygen categories
            features['oxygen_category'] = pd.cut(data['oxygen_concentration'], 
                                              bins=[0, 0.1, 1, 5, 21], 
                                              labels=['inert', 'low', 'moderate', 'high'])
            
            # Contamination level
            features['contamination_level'] = data['oxygen_concentration'] / 21  # Normalize to atmospheric
            features['contamination_level_mean'] = features['contamination_level'].rolling(window=10).mean()
        
        # Gas flow features
        if 'gas_flow_rate' in data.columns:
            features['flow_rate_mean'] = data['gas_flow_rate'].rolling(window=10).mean()
            features['flow_rate_std'] = data['gas_flow_rate'].rolling(window=10).std()
            
            # Flow rate categories
            features['flow_rate_category'] = pd.cut(data['gas_flow_rate'], 
                                                  bins=[0, 10, 20, 50, 100], 
                                                  labels=['low', 'normal', 'high', 'very_high'])
        
        # Gas purity calculation
        inert_gases = ['nitrogen_concentration', 'argon_concentration', 'helium_concentration']
        inert_cols = [col for col in inert_gases if col in data.columns]
        if inert_cols:
            features['inert_gas_total'] = data[inert_cols].sum(axis=1)
            features['gas_purity'] = features['inert_gas_total'] / 100
            features['gas_purity_mean'] = features['gas_purity'].rolling(window=10).mean()
        
        return features
    
    def extract_vibration_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract vibration and mechanical disturbance features.
        
        Args:
            data: DataFrame with vibration data
            
        Returns:
            DataFrame with vibration features
        """
        features = pd.DataFrame(index=data.index)
        
        # Vibration amplitude features
        vibration_cols = [col for col in data.columns if 'vibration' in col.lower()]
        for col in vibration_cols:
            vib_name = col.replace('vibration_', '').replace('_vibration', '')
            features[f'{vib_name}_vib_mean'] = data[col].rolling(window=10).mean()
            features[f'{vib_name}_vib_std'] = data[col].rolling(window=10).std()
            features[f'{vib_name}_vib_max'] = data[col].rolling(window=10).max()
            
            # Vibration intensity
            features[f'{vib_name}_vib_intensity'] = np.sqrt(data[col] ** 2)
            features[f'{vib_name}_vib_intensity_mean'] = features[f'{vib_name}_vib_intensity'].rolling(window=10).mean()
        
        # Overall vibration level
        if len(vibration_cols) > 1:
            vibration_data = data[vibration_cols]
            features['overall_vibration'] = np.sqrt((vibration_data ** 2).sum(axis=1))
            features['overall_vibration_mean'] = features['overall_vibration'].rolling(window=10).mean()
        
        return features
    
    def extract_contamination_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract contamination and cleanliness features.
        
        Args:
            data: DataFrame with contamination data
            
        Returns:
            DataFrame with contamination features
        """
        features = pd.DataFrame(index=data.index)
        
        # Particle count features
        if 'particle_count' in data.columns:
            features['particle_count_mean'] = data['particle_count'].rolling(window=10).mean()
            features['particle_count_std'] = data['particle_count'].rolling(window=10).std()
            
            # Particle categories
            features['particle_category'] = pd.cut(data['particle_count'], 
                                                bins=[0, 100, 1000, 10000, 100000], 
                                                labels=['clean', 'low', 'moderate', 'high'])
        
        # Contamination level
        if 'contamination_level' in data.columns:
            features['contamination_mean'] = data['contamination_level'].rolling(window=10).mean()
            features['contamination_std'] = data['contamination_level'].rolling(window=10).std()
            
            # Contamination trend
            features['contamination_trend'] = data['contamination_level'].diff().rolling(window=5).mean()
        
        # Cleanliness index
        if 'particle_count' in data.columns:
            features['cleanliness_index'] = 1 / (data['particle_count'] + 1)
            features['cleanliness_index_mean'] = features['cleanliness_index'].rolling(window=10).mean()
        
        return features
    
    def extract_interaction_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract interaction features between environmental factors.
        
        Args:
            data: DataFrame with environmental data
            
        Returns:
            DataFrame with interaction features
        """
        features = pd.DataFrame(index=data.index)
        
        # Temperature-humidity interactions
        if 'chamber_temperature' in data.columns and 'humidity' in data.columns:
            features['temp_humidity_ratio'] = data['chamber_temperature'] / (data['humidity'] + 1e-6)
            features['comfort_index'] = data['chamber_temperature'] * (1 - data['humidity'] / 100)
        
        # Pressure-temperature interactions
        if 'chamber_pressure' in data.columns and 'chamber_temperature' in data.columns:
            features['pressure_temp_ratio'] = data['chamber_pressure'] / (data['chamber_temperature'] + 1e-6)
        
        # Gas-flow interactions
        if 'gas_flow_rate' in data.columns and 'oxygen_concentration' in data.columns:
            features['flow_oxygen_ratio'] = data['gas_flow_rate'] / (data['oxygen_concentration'] + 1e-6)
        
        # Environmental stability index
        stability_cols = []
        if 'chamber_temperature' in data.columns:
            stability_cols.append('chamber_temperature')
        if 'humidity' in data.columns:
            stability_cols.append('humidity')
        if 'chamber_pressure' in data.columns:
            stability_cols.append('chamber_pressure')
        
        if len(stability_cols) > 1:
            stability_data = data[stability_cols]
            features['environmental_stability'] = 1 / (stability_data.rolling(window=10).std().sum(axis=1) + 1e-6)
        
        return features
    
    def extract_all_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract all environmental features.
        
        Args:
            data: DataFrame with environmental data
            
        Returns:
            DataFrame with all extracted features
        """
        logger.info("Extracting environmental features...")
        
        # Validate input data
        self._validate_input_data(data)
        
        # Extract different feature groups
        feature_groups = [
            self.extract_temperature_features(data),
            self.extract_humidity_features(data),
            self.extract_pressure_features(data),
            self.extract_gas_features(data),
            self.extract_vibration_features(data),
            self.extract_contamination_features(data),
            self.extract_interaction_features(data)
        ]
        
        # Combine all features
        all_features = pd.concat(feature_groups, axis=1)
        
        # Clean and validate features
        all_features = self._clean_features(all_features)
        
        logger.info(f"Extracted {len(all_features.columns)} environmental features")
        return all_features
    
    def _validate_input_data(self, data: pd.DataFrame) -> None:
        """Validate input data against configuration rules."""
        validation_rules = self.validation_rules
        
        # Validate temperature data
        if 'chamber_temperature' in data.columns:
            temp_range = validation_rules.get('temperature_range', [-50, 200])
            invalid_temp = (data['chamber_temperature'] < temp_range[0]) | (data['chamber_temperature'] > temp_range[1])
            if invalid_temp.any():
                logger.warning(f"Found {invalid_temp.sum()} invalid chamber temperature values")
        
        # Validate humidity data
        if 'humidity' in data.columns:
            humidity_range = validation_rules.get('humidity_range', [0, 100])
            invalid_humidity = (data['humidity'] < humidity_range[0]) | (data['humidity'] > humidity_range[1])
            if invalid_humidity.any():
                logger.warning(f"Found {invalid_humidity.sum()} invalid humidity values")
        
        # Validate pressure data
        if 'chamber_pressure' in data.columns:
            pressure_range = validation_rules.get('pressure_range', [0.1, 2.0])
            invalid_pressure = (data['chamber_pressure'] < pressure_range[0]) | (data['chamber_pressure'] > pressure_range[1])
            if invalid_pressure.any():
                logger.warning(f"Found {invalid_pressure.sum()} invalid chamber pressure values")
        
        # Validate oxygen data
        if 'oxygen_concentration' in data.columns:
            oxygen_range = validation_rules.get('oxygen_range', [0, 21])
            invalid_oxygen = (data['oxygen_concentration'] < oxygen_range[0]) | (data['oxygen_concentration'] > oxygen_range[1])
            if invalid_oxygen.any():
                logger.warning(f"Found {invalid_oxygen.sum()} invalid oxygen concentration values")
    
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
        Calculate feature importance for environmental factors.
        
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
