"""
Temperature Feature Engineering

This module extracts and engineers features from temperature sensor data for PBF-LB/M processes.
Integrates with YAML configuration for feature definitions and validation rules.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import logging
from ..config.config_manager import ConfigManager

logger = logging.getLogger(__name__)


class TemperatureFeatures:
    """
    Feature engineering for temperature sensor data in PBF-LB/M processes.
    
    Extracts features from temperature measurements, thermal gradients,
    and heat distribution patterns based on YAML configuration.
    """
    
    def __init__(self, config_manager: Optional[ConfigManager] = None):
        """
        Initialize temperature feature engineering.
        
        Args:
            config_manager: Configuration manager for loading feature definitions
        """
        self.config_manager = config_manager or ConfigManager()
        self.feature_config = self._load_feature_config()
        self.feature_definitions = self.feature_config.get('feature_definitions', {})
        self.validation_rules = self.feature_config.get('validation_rules', {})
        
    def _load_feature_config(self) -> Dict[str, Any]:
        """Load temperature feature configuration from YAML."""
        try:
            return self.config_manager.load_feature_config('sensor_features/temperature_features')
        except Exception as e:
            logger.warning(f"Could not load temperature config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for temperature features."""
        return {
            'feature_definitions': {
                'temperature_features': {
                    'aggregations': ['mean', 'std', 'min', 'max', 'range'],
                    'temporal': ['trend', 'volatility', 'stability']
                },
                'gradient_features': {
                    'aggregations': ['spatial_gradient', 'temporal_gradient'],
                    'derived': ['thermal_conductivity', 'heat_flux']
                },
                'cycle_features': {
                    'aggregations': ['cycle_amplitude', 'cycle_frequency'],
                    'derived': ['thermal_cycling', 'fatigue_estimation']
                }
            },
            'validation_rules': {
                'temperature_range': [-50, 3000],  # Celsius
                'gradient_range': [-1000, 1000],  # K/m
                'sampling_rate_range': [0.1, 1000]  # Hz
            }
        }
    
    def extract_basic_temperature_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract basic temperature measurement features.
        
        Args:
            data: DataFrame with temperature data
            
        Returns:
            DataFrame with basic temperature features
        """
        features = pd.DataFrame(index=data.index)
        
        # Single temperature sensor features
        if 'temperature' in data.columns:
            temp_col = data['temperature']
            features['temp_mean'] = temp_col.rolling(window=10).mean()
            features['temp_std'] = temp_col.rolling(window=10).std()
            features['temp_min'] = temp_col.rolling(window=10).min()
            features['temp_max'] = temp_col.rolling(window=10).max()
            features['temp_range'] = features['temp_max'] - features['temp_min']
            
            # Temporal features
            features['temp_trend'] = temp_col.diff().rolling(window=5).mean()
            features['temp_volatility'] = temp_col.rolling(window=10).std()
            features['temp_stability'] = 1 / (features['temp_volatility'] + 1e-6)
            
            # Temperature categories
            features['temp_category'] = pd.cut(temp_col, 
                                            bins=[-50, 0, 50, 100, 500, 1000, 3000], 
                                            labels=['freezing', 'cold', 'cool', 'warm', 'hot', 'very_hot'])
        
        # Multi-sensor temperature features
        temp_cols = [col for col in data.columns if 'temp_' in col.lower() and col != 'temperature']
        if len(temp_cols) > 1:
            temp_data = data[temp_cols]
            features['multi_temp_mean'] = temp_data.mean(axis=1)
            features['multi_temp_std'] = temp_data.std(axis=1)
            features['multi_temp_max'] = temp_data.max(axis=1)
            features['multi_temp_min'] = temp_data.min(axis=1)
            
            # Temperature uniformity
            features['temp_uniformity'] = 1 / (features['multi_temp_std'] + 1e-6)
            
            # Hot spot detection
            features['hot_spot_detected'] = (features['multi_temp_max'] > 
                                           features['multi_temp_mean'] + 2 * features['multi_temp_std']).astype(int)
            
            # Cold spot detection
            features['cold_spot_detected'] = (features['multi_temp_min'] < 
                                            features['multi_temp_mean'] - 2 * features['multi_temp_std']).astype(int)
        
        return features
    
    def extract_thermal_gradient_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract thermal gradient features.
        
        Args:
            data: DataFrame with spatial temperature data
            
        Returns:
            DataFrame with thermal gradient features
        """
        features = pd.DataFrame(index=data.index)
        
        # Spatial temperature gradients
        spatial_temp_cols = [col for col in data.columns if 'temp_' in col.lower()]
        if len(spatial_temp_cols) >= 2:
            temp_data = data[spatial_temp_cols]
            
            # Calculate gradients between adjacent points
            for i in range(len(spatial_temp_cols) - 1):
                col1, col2 = spatial_temp_cols[i], spatial_temp_cols[i + 1]
                features[f'spatial_gradient_{i}_{i+1}'] = temp_data[col2] - temp_data[col1]
                features[f'spatial_gradient_{i}_{i+1}_mean'] = features[f'spatial_gradient_{i}_{i+1}'].rolling(window=10).mean()
                features[f'spatial_gradient_{i}_{i+1}_std'] = features[f'spatial_gradient_{i}_{i+1}'].rolling(window=10).std()
            
            # Overall thermal gradient
            features['overall_gradient'] = temp_data.max(axis=1) - temp_data.min(axis=1)
            features['overall_gradient_mean'] = features['overall_gradient'].rolling(window=10).mean()
            features['overall_gradient_std'] = features['overall_gradient'].rolling(window=10).std()
            
            # Gradient stability
            features['gradient_stability'] = 1 / (features['overall_gradient_std'] + 1e-6)
            
            # Gradient categories
            features['gradient_category'] = pd.cut(features['overall_gradient'], 
                                                bins=[0, 10, 50, 100, 500, 1000], 
                                                labels=['minimal', 'low', 'moderate', 'high', 'extreme'])
        
        # Time-based thermal gradients
        if 'temperature' in data.columns:
            features['temporal_gradient'] = data['temperature'].diff()
            features['temporal_gradient_mean'] = features['temporal_gradient'].rolling(window=10).mean()
            features['temporal_gradient_std'] = features['temporal_gradient'].rolling(window=10).std()
            
            # Heating rate
            features['heating_rate'] = features['temporal_gradient'].clip(lower=0)
            features['heating_rate_mean'] = features['heating_rate'].rolling(window=10).mean()
            
            # Cooling rate
            features['cooling_rate'] = -features['temporal_gradient'].clip(upper=0)
            features['cooling_rate_mean'] = features['cooling_rate'].rolling(window=10).mean()
        
        return features
    
    def extract_heat_flux_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract heat flux and energy features.
        
        Args:
            data: DataFrame with temperature and spatial data
            
        Returns:
            DataFrame with heat flux features
        """
        features = pd.DataFrame(index=data.index)
        
        # Heat flux estimation (simplified)
        if 'temperature' in data.columns and 'distance' in data.columns:
            # Simplified heat flux calculation
            features['heat_flux'] = data['temperature'] / (data['distance'] + 1e-6)
            features['heat_flux_mean'] = features['heat_flux'].rolling(window=10).mean()
            features['heat_flux_std'] = features['heat_flux'].rolling(window=10).std()
        
        # Energy density features
        if 'temperature' in data.columns and 'area' in data.columns:
            features['energy_density'] = data['temperature'] * data['area']
            features['energy_density_mean'] = features['energy_density'].rolling(window=10).mean()
        
        # Thermal energy accumulation
        if 'temperature' in data.columns and 'time' in data.columns:
            features['thermal_energy'] = data['temperature'].cumsum()
            features['thermal_energy_rate'] = data['temperature'].diff() / data['time'].diff()
        
        # Heat capacity estimation
        if 'temperature' in data.columns and 'energy_input' in data.columns:
            features['heat_capacity'] = data['energy_input'] / (data['temperature'].diff() + 1e-6)
            features['heat_capacity_mean'] = features['heat_capacity'].rolling(window=10).mean()
        
        return features
    
    def extract_thermal_cycling_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract thermal cycling and fatigue features.
        
        Args:
            data: DataFrame with temperature data
            
        Returns:
            DataFrame with thermal cycling features
        """
        features = pd.DataFrame(index=data.index)
        
        # Temperature cycling detection
        if 'temperature' in data.columns:
            temp_col = data['temperature']
            
            # Find temperature peaks and valleys
            temp_diff = temp_col.diff()
            peaks = (temp_diff > 0) & (temp_diff.shift(-1) < 0)
            valleys = (temp_diff < 0) & (temp_diff.shift(-1) > 0)
            
            features['peak_count'] = peaks.rolling(window=100).sum()
            features['valley_count'] = valleys.rolling(window=100).sum()
            features['cycle_count'] = (features['peak_count'] + features['valley_count']) / 2
            
            # Cycle amplitude
            if peaks.any() and valleys.any():
                peak_temps = temp_col[peaks]
                valley_temps = temp_col[valleys]
                if len(peak_temps) > 0 and len(valley_temps) > 0:
                    features['cycle_amplitude'] = peak_temps.rolling(window=10).mean() - valley_temps.rolling(window=10).mean()
            
            # Cycle frequency
            features['cycle_frequency'] = features['cycle_count'] / 100  # cycles per 100 samples
        
        # Thermal fatigue estimation
        if 'cycle_amplitude' in features.columns:
            # Simplified fatigue estimation based on cycle amplitude
            features['fatigue_estimation'] = features['cycle_amplitude'] * features['cycle_frequency']
            features['fatigue_estimation_mean'] = features['fatigue_estimation'].rolling(window=10).mean()
        
        # Temperature range features
        if 'temperature' in data.columns:
            temp_col = data['temperature']
            features['temp_range_10'] = temp_col.rolling(window=10).max() - temp_col.rolling(window=10).min()
            features['temp_range_50'] = temp_col.rolling(window=50).max() - temp_col.rolling(window=50).min()
            features['temp_range_100'] = temp_col.rolling(window=100).max() - temp_col.rolling(window=100).min()
        
        return features
    
    def extract_phase_transition_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract phase transition and material state features.
        
        Args:
            data: DataFrame with temperature data
            
        Returns:
            DataFrame with phase transition features
        """
        features = pd.DataFrame(index=data.index)
        
        # Melting point features
        if 'melting_point' in data.columns and 'temperature' in data.columns:
            features['melting_ratio'] = data['temperature'] / data['melting_point']
            features['above_melting'] = (data['temperature'] > data['melting_point']).astype(int)
            features['melting_ratio_mean'] = features['melting_ratio'].rolling(window=10).mean()
        
        # Boiling point features
        if 'boiling_point' in data.columns and 'temperature' in data.columns:
            features['boiling_ratio'] = data['temperature'] / data['boiling_point']
            features['above_boiling'] = (data['temperature'] > data['boiling_point']).astype(int)
        
        # Phase transition detection
        if 'temperature' in data.columns:
            temp_col = data['temperature']
            temp_diff = temp_col.diff()
            
            # Detect rapid temperature changes (potential phase transitions)
            features['rapid_temp_change'] = (np.abs(temp_diff) > 2 * temp_diff.rolling(window=20).std()).astype(int)
            features['rapid_heating'] = (temp_diff > 2 * temp_diff.rolling(window=20).std()).astype(int)
            features['rapid_cooling'] = (temp_diff < -2 * temp_diff.rolling(window=20).std()).astype(int)
        
        # Material state features
        if 'temperature' in data.columns:
            temp_col = data['temperature']
            features['material_state'] = pd.cut(temp_col, 
                                             bins=[-50, 0, 100, 500, 1000, 2000, 3000], 
                                             labels=['solid_cold', 'solid', 'warm_solid', 'hot_solid', 'molten', 'vapor'])
        
        return features
    
    def extract_anomaly_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract temperature anomaly detection features.
        
        Args:
            data: DataFrame with temperature data
            
        Returns:
            DataFrame with anomaly features
        """
        features = pd.DataFrame(index=data.index)
        
        # Temperature anomalies
        if 'temperature' in data.columns:
            temp_col = data['temperature']
            temp_mean = temp_col.rolling(window=50).mean()
            temp_std = temp_col.rolling(window=50).std()
            
            # Z-score based anomaly detection
            features['temp_zscore'] = (temp_col - temp_mean) / (temp_std + 1e-6)
            features['temp_anomaly'] = (np.abs(features['temp_zscore']) > 3).astype(int)
            
            # Temperature spike detection
            features['temp_spike'] = (temp_col.diff() > 2 * temp_std).astype(int)
            
            # Sudden drop detection
            features['temp_drop'] = (temp_col.diff() < -2 * temp_std).astype(int)
            
            # Temperature plateau detection
            features['temp_plateau'] = (temp_col.rolling(window=10).std() < 0.1 * temp_std).astype(int)
        
        # Gradient anomalies
        if 'overall_gradient' in features.columns:
            gradient = features['overall_gradient']
            gradient_mean = gradient.rolling(window=50).mean()
            gradient_std = gradient.rolling(window=50).std()
            
            features['gradient_zscore'] = (gradient - gradient_mean) / (gradient_std + 1e-6)
            features['gradient_anomaly'] = (np.abs(features['gradient_zscore']) > 3).astype(int)
        
        # Combined anomaly score
        anomaly_cols = [col for col in features.columns if 'anomaly' in col]
        if len(anomaly_cols) > 1:
            features['combined_anomaly_score'] = features[anomaly_cols].sum(axis=1)
            features['anomaly_detected'] = (features['combined_anomaly_score'] > 0).astype(int)
        
        return features
    
    def extract_all_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract all temperature features.
        
        Args:
            data: DataFrame with temperature data
            
        Returns:
            DataFrame with all extracted features
        """
        logger.info("Extracting temperature features...")
        
        # Validate input data
        self._validate_input_data(data)
        
        # Extract different feature groups
        feature_groups = [
            self.extract_basic_temperature_features(data),
            self.extract_thermal_gradient_features(data),
            self.extract_heat_flux_features(data),
            self.extract_thermal_cycling_features(data),
            self.extract_phase_transition_features(data),
            self.extract_anomaly_features(data)
        ]
        
        # Combine all features
        all_features = pd.concat(feature_groups, axis=1)
        
        # Clean and validate features
        all_features = self._clean_features(all_features)
        
        logger.info(f"Extracted {len(all_features.columns)} temperature features")
        return all_features
    
    def _validate_input_data(self, data: pd.DataFrame) -> None:
        """Validate input data against configuration rules."""
        validation_rules = self.validation_rules
        
        # Validate temperature data
        temp_cols = [col for col in data.columns if 'temp' in col.lower()]
        for col in temp_cols:
            temp_range = validation_rules.get('temperature_range', [-50, 3000])
            invalid_temp = (data[col] < temp_range[0]) | (data[col] > temp_range[1])
            if invalid_temp.any():
                logger.warning(f"Found {invalid_temp.sum()} invalid {col} values")
        
        # Validate gradient data
        if 'gradient' in data.columns:
            gradient_range = validation_rules.get('gradient_range', [-1000, 1000])
            invalid_gradient = (data['gradient'] < gradient_range[0]) | (data['gradient'] > gradient_range[1])
            if invalid_gradient.any():
                logger.warning(f"Found {invalid_gradient.sum()} invalid gradient values")
        
        # Validate sampling rate
        if 'sampling_rate' in data.columns:
            sampling_range = validation_rules.get('sampling_rate_range', [0.1, 1000])
            invalid_sampling = (data['sampling_rate'] < sampling_range[0]) | (data['sampling_rate'] > sampling_range[1])
            if invalid_sampling.any():
                logger.warning(f"Found {invalid_sampling.sum()} invalid sampling rate values")
    
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
        Calculate feature importance for temperature data.
        
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
