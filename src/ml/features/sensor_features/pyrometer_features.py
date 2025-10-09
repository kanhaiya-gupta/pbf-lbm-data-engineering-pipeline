"""
Pyrometer Feature Engineering

This module extracts and engineers features from pyrometer sensor data for PBF-LB/M processes.
Integrates with YAML configuration for feature definitions and validation rules.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import logging
from ..config.config_manager import ConfigManager

logger = logging.getLogger(__name__)


class PyrometerFeatures:
    """
    Feature engineering for pyrometer sensor data in PBF-LB/M processes.
    
    Extracts features from temperature measurements, thermal gradients,
    and heat distribution patterns based on YAML configuration.
    """
    
    def __init__(self, config_manager: Optional[ConfigManager] = None):
        """
        Initialize pyrometer feature engineering.
        
        Args:
            config_manager: Configuration manager for loading feature definitions
        """
        self.config_manager = config_manager or ConfigManager()
        self.feature_config = self._load_feature_config()
        self.feature_definitions = self.feature_config.get('feature_definitions', {})
        self.validation_rules = self.feature_config.get('validation_rules', {})
        
    def _load_feature_config(self) -> Dict[str, Any]:
        """Load pyrometer feature configuration from YAML."""
        try:
            return self.config_manager.load_feature_config('sensor_features/pyrometer_features')
        except Exception as e:
            logger.warning(f"Could not load pyrometer config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for pyrometer features."""
        return {
            'feature_definitions': {
                'temperature_features': {
                    'aggregations': ['mean', 'std', 'min', 'max', 'range'],
                    'temporal': ['trend', 'volatility', 'stability'],
                    'derived': ['thermal_gradient', 'heat_flux']
                },
                'spatial_features': {
                    'aggregations': ['spatial_mean', 'spatial_std'],
                    'derived': ['temperature_uniformity', 'hot_spot_detection']
                },
                'spectral_features': {
                    'aggregations': ['wavelength_peak', 'spectral_width'],
                    'derived': ['emissivity_estimate', 'material_identification']
                }
            },
            'validation_rules': {
                'temperature_range': [20, 3000],  # Celsius
                'wavelength_range': [0.5, 20],  # micrometers
                'emissivity_range': [0.1, 1.0]
            }
        }
    
    def extract_temperature_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract temperature measurement features.
        
        Args:
            data: DataFrame with pyrometer temperature data
            
        Returns:
            DataFrame with temperature features
        """
        features = pd.DataFrame(index=data.index)
        
        # Basic temperature features
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
                                            bins=[0, 100, 500, 1000, 2000, 3000], 
                                            labels=['cold', 'warm', 'hot', 'very_hot', 'extreme'])
        
        # Multi-point temperature features
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
            features['hot_spot_detected'] = (features['multi_temp_max'] > features['multi_temp_mean'] + 2 * features['multi_temp_std']).astype(int)
        
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
                features[f'gradient_{i}_{i+1}'] = temp_data[col2] - temp_data[col1]
                features[f'gradient_{i}_{i+1}_mean'] = features[f'gradient_{i}_{i+1}'].rolling(window=10).mean()
                features[f'gradient_{i}_{i+1}_std'] = features[f'gradient_{i}_{i+1}'].rolling(window=10).std()
            
            # Overall thermal gradient
            features['overall_gradient'] = temp_data.max(axis=1) - temp_data.min(axis=1)
            features['overall_gradient_mean'] = features['overall_gradient'].rolling(window=10).mean()
            
            # Gradient stability
            features['gradient_stability'] = 1 / (features['overall_gradient'].rolling(window=10).std() + 1e-6)
        
        # Time-based thermal gradients
        if 'temperature' in data.columns:
            features['temporal_gradient'] = data['temperature'].diff()
            features['temporal_gradient_mean'] = features['temporal_gradient'].rolling(window=10).mean()
            features['temporal_gradient_std'] = features['temporal_gradient'].rolling(window=10).std()
            
            # Cooling rate
            features['cooling_rate'] = -features['temporal_gradient']  # Negative gradient = cooling
            features['cooling_rate_mean'] = features['cooling_rate'].rolling(window=10).mean()
        
        return features
    
    def extract_heat_flux_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract heat flux and energy features.
        
        Args:
            data: DataFrame with pyrometer data
            
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
        
        return features
    
    def extract_spectral_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract spectral and emissivity features.
        
        Args:
            data: DataFrame with spectral data
            
        Returns:
            DataFrame with spectral features
        """
        features = pd.DataFrame(index=data.index)
        
        # Wavelength features
        if 'wavelength' in data.columns:
            features['wavelength_mean'] = data['wavelength'].rolling(window=10).mean()
            features['wavelength_std'] = data['wavelength'].rolling(window=10).std()
            features['wavelength_peak'] = data['wavelength'].rolling(window=10).max()
        
        # Intensity features
        if 'intensity' in data.columns:
            features['intensity_mean'] = data['intensity'].rolling(window=10).mean()
            features['intensity_std'] = data['intensity'].rolling(window=10).std()
            features['intensity_max'] = data['intensity'].rolling(window=10).max()
            
            # Spectral width (simplified)
            features['spectral_width'] = features['intensity_std'] * 2
        
        # Emissivity estimation
        if 'temperature' in data.columns and 'intensity' in data.columns:
            # Simplified emissivity calculation (Stefan-Boltzmann law)
            features['emissivity_estimate'] = data['intensity'] / (5.67e-8 * (data['temperature'] + 273.15) ** 4)
            features['emissivity_estimate'] = features['emissivity_estimate'].clip(0.1, 1.0)
            features['emissivity_mean'] = features['emissivity_estimate'].rolling(window=10).mean()
        
        # Material identification features
        if 'wavelength' in data.columns and 'intensity' in data.columns:
            # Peak wavelength ratio for material identification
            features['peak_ratio'] = data['wavelength'] / (data['intensity'] + 1e-6)
            features['peak_ratio_mean'] = features['peak_ratio'].rolling(window=10).mean()
        
        return features
    
    def extract_calibration_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract calibration and accuracy features.
        
        Args:
            data: DataFrame with calibration data
            
        Returns:
            DataFrame with calibration features
        """
        features = pd.DataFrame(index=data.index)
        
        # Calibration drift features
        if 'reference_temperature' in data.columns and 'measured_temperature' in data.columns:
            features['calibration_error'] = data['measured_temperature'] - data['reference_temperature']
            features['calibration_error_mean'] = features['calibration_error'].rolling(window=10).mean()
            features['calibration_error_std'] = features['calibration_error'].rolling(window=10).std()
            
            # Calibration accuracy
            features['calibration_accuracy'] = 1 / (features['calibration_error_std'] + 1e-6)
        
        # Signal quality features
        if 'signal_strength' in data.columns:
            features['signal_strength_mean'] = data['signal_strength'].rolling(window=10).mean()
            features['signal_strength_std'] = data['signal_strength'].rolling(window=10).std()
            
            # Signal quality categories
            features['signal_quality'] = pd.cut(data['signal_strength'], 
                                             bins=[0, 0.3, 0.7, 1.0], 
                                             labels=['poor', 'good', 'excellent'])
        
        # Noise level features
        if 'noise_level' in data.columns:
            features['noise_level_mean'] = data['noise_level'].rolling(window=10).mean()
            features['noise_level_std'] = data['noise_level'].rolling(window=10).std()
            
            # Signal-to-noise ratio
            if 'signal_strength' in data.columns:
                features['snr'] = data['signal_strength'] / (data['noise_level'] + 1e-6)
                features['snr_mean'] = features['snr'].rolling(window=10).mean()
        
        return features
    
    def extract_anomaly_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract anomaly detection features.
        
        Args:
            data: DataFrame with pyrometer data
            
        Returns:
            DataFrame with anomaly features
        """
        features = pd.DataFrame(index=data.index)
        
        # Temperature anomalies
        if 'temperature' in data.columns:
            temp_col = data['temperature']
            temp_mean = temp_col.rolling(window=20).mean()
            temp_std = temp_col.rolling(window=20).std()
            
            # Z-score based anomaly detection
            features['temp_zscore'] = (temp_col - temp_mean) / (temp_std + 1e-6)
            features['temp_anomaly'] = (np.abs(features['temp_zscore']) > 3).astype(int)
            
            # Temperature spike detection
            features['temp_spike'] = (temp_col.diff() > 2 * temp_std).astype(int)
            
            # Sudden drop detection
            features['temp_drop'] = (temp_col.diff() < -2 * temp_std).astype(int)
        
        # Intensity anomalies
        if 'intensity' in data.columns:
            intensity_col = data['intensity']
            intensity_mean = intensity_col.rolling(window=20).mean()
            intensity_std = intensity_col.rolling(window=20).std()
            
            features['intensity_zscore'] = (intensity_col - intensity_mean) / (intensity_std + 1e-6)
            features['intensity_anomaly'] = (np.abs(features['intensity_zscore']) > 3).astype(int)
        
        # Combined anomaly score
        anomaly_cols = [col for col in features.columns if 'anomaly' in col]
        if len(anomaly_cols) > 1:
            features['combined_anomaly_score'] = features[anomaly_cols].sum(axis=1)
            features['anomaly_detected'] = (features['combined_anomaly_score'] > 0).astype(int)
        
        return features
    
    def extract_all_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract all pyrometer features.
        
        Args:
            data: DataFrame with pyrometer data
            
        Returns:
            DataFrame with all extracted features
        """
        logger.info("Extracting pyrometer features...")
        
        # Validate input data
        self._validate_input_data(data)
        
        # Extract different feature groups
        feature_groups = [
            self.extract_temperature_features(data),
            self.extract_thermal_gradient_features(data),
            self.extract_heat_flux_features(data),
            self.extract_spectral_features(data),
            self.extract_calibration_features(data),
            self.extract_anomaly_features(data)
        ]
        
        # Combine all features
        all_features = pd.concat(feature_groups, axis=1)
        
        # Clean and validate features
        all_features = self._clean_features(all_features)
        
        logger.info(f"Extracted {len(all_features.columns)} pyrometer features")
        return all_features
    
    def _validate_input_data(self, data: pd.DataFrame) -> None:
        """Validate input data against configuration rules."""
        validation_rules = self.validation_rules
        
        # Validate temperature data
        if 'temperature' in data.columns:
            temp_range = validation_rules.get('temperature_range', [20, 3000])
            invalid_temp = (data['temperature'] < temp_range[0]) | (data['temperature'] > temp_range[1])
            if invalid_temp.any():
                logger.warning(f"Found {invalid_temp.sum()} invalid temperature values")
        
        # Validate wavelength data
        if 'wavelength' in data.columns:
            wavelength_range = validation_rules.get('wavelength_range', [0.5, 20])
            invalid_wavelength = (data['wavelength'] < wavelength_range[0]) | (data['wavelength'] > wavelength_range[1])
            if invalid_wavelength.any():
                logger.warning(f"Found {invalid_wavelength.sum()} invalid wavelength values")
        
        # Validate emissivity data
        if 'emissivity' in data.columns:
            emissivity_range = validation_rules.get('emissivity_range', [0.1, 1.0])
            invalid_emissivity = (data['emissivity'] < emissivity_range[0]) | (data['emissivity'] > emissivity_range[1])
            if invalid_emissivity.any():
                logger.warning(f"Found {invalid_emissivity.sum()} invalid emissivity values")
    
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
        Calculate feature importance for pyrometer data.
        
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
