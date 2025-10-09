"""
Accelerometer Feature Engineering

This module extracts and engineers features from accelerometer sensor data for PBF-LB/M processes.
Integrates with YAML configuration for feature definitions and validation rules.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import logging
from ..config.config_manager import ConfigManager

logger = logging.getLogger(__name__)


class AccelerometerFeatures:
    """
    Feature engineering for accelerometer sensor data in PBF-LB/M processes.
    
    Extracts features from vibration, acceleration, and motion data
    based on YAML configuration.
    """
    
    def __init__(self, config_manager: Optional[ConfigManager] = None):
        """
        Initialize accelerometer feature engineering.
        
        Args:
            config_manager: Configuration manager for loading feature definitions
        """
        self.config_manager = config_manager or ConfigManager()
        self.feature_config = self._load_feature_config()
        self.feature_definitions = self.feature_config.get('feature_definitions', {})
        self.validation_rules = self.feature_config.get('validation_rules', {})
        
    def _load_feature_config(self) -> Dict[str, Any]:
        """Load accelerometer feature configuration from YAML."""
        try:
            return self.config_manager.load_feature_config('sensor_features/accelerometer_features')
        except Exception as e:
            logger.warning(f"Could not load accelerometer config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for accelerometer features."""
        return {
            'feature_definitions': {
                'acceleration_features': {
                    'aggregations': ['mean', 'std', 'min', 'max', 'rms'],
                    'temporal': ['trend', 'volatility', 'stability']
                },
                'vibration_features': {
                    'aggregations': ['frequency_domain', 'spectral_centroid'],
                    'derived': ['vibration_intensity', 'resonance_detection']
                },
                'motion_features': {
                    'aggregations': ['velocity', 'displacement'],
                    'derived': ['motion_pattern', 'stability_index']
                }
            },
            'validation_rules': {
                'acceleration_range': [-50, 50],  # m/s²
                'frequency_range': [0, 1000],  # Hz
                'sampling_rate_range': [1, 10000]  # Hz
            }
        }
    
    def extract_acceleration_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract acceleration measurement features.
        
        Args:
            data: DataFrame with accelerometer data
            
        Returns:
            DataFrame with acceleration features
        """
        features = pd.DataFrame(index=data.index)
        
        # X-axis acceleration features
        if 'accel_x' in data.columns:
            accel_x = data['accel_x']
            features['accel_x_mean'] = accel_x.rolling(window=10).mean()
            features['accel_x_std'] = accel_x.rolling(window=10).std()
            features['accel_x_min'] = accel_x.rolling(window=10).min()
            features['accel_x_max'] = accel_x.rolling(window=10).max()
            features['accel_x_rms'] = np.sqrt((accel_x ** 2).rolling(window=10).mean())
            
            # Temporal features
            features['accel_x_trend'] = accel_x.diff().rolling(window=5).mean()
            features['accel_x_volatility'] = accel_x.rolling(window=10).std()
            features['accel_x_stability'] = 1 / (features['accel_x_volatility'] + 1e-6)
        
        # Y-axis acceleration features
        if 'accel_y' in data.columns:
            accel_y = data['accel_y']
            features['accel_y_mean'] = accel_y.rolling(window=10).mean()
            features['accel_y_std'] = accel_y.rolling(window=10).std()
            features['accel_y_min'] = accel_y.rolling(window=10).min()
            features['accel_y_max'] = accel_y.rolling(window=10).max()
            features['accel_y_rms'] = np.sqrt((accel_y ** 2).rolling(window=10).mean())
            
            # Temporal features
            features['accel_y_trend'] = accel_y.diff().rolling(window=5).mean()
            features['accel_y_volatility'] = accel_y.rolling(window=10).std()
            features['accel_y_stability'] = 1 / (features['accel_y_volatility'] + 1e-6)
        
        # Z-axis acceleration features
        if 'accel_z' in data.columns:
            accel_z = data['accel_z']
            features['accel_z_mean'] = accel_z.rolling(window=10).mean()
            features['accel_z_std'] = accel_z.rolling(window=10).std()
            features['accel_z_min'] = accel_z.rolling(window=10).min()
            features['accel_z_max'] = accel_z.rolling(window=10).max()
            features['accel_z_rms'] = np.sqrt((accel_z ** 2).rolling(window=10).mean())
            
            # Temporal features
            features['accel_z_trend'] = accel_z.diff().rolling(window=5).mean()
            features['accel_z_volatility'] = accel_z.rolling(window=10).std()
            features['accel_z_stability'] = 1 / (features['accel_z_volatility'] + 1e-6)
        
        # Combined acceleration features
        accel_cols = [col for col in data.columns if col.startswith('accel_')]
        if len(accel_cols) >= 2:
            accel_data = data[accel_cols]
            
            # Magnitude of acceleration vector
            features['accel_magnitude'] = np.sqrt((accel_data ** 2).sum(axis=1))
            features['accel_magnitude_mean'] = features['accel_magnitude'].rolling(window=10).mean()
            features['accel_magnitude_std'] = features['accel_magnitude'].rolling(window=10).std()
            features['accel_magnitude_rms'] = np.sqrt((features['accel_magnitude'] ** 2).rolling(window=10).mean())
            
            # Overall acceleration stability
            features['accel_stability'] = 1 / (features['accel_magnitude_std'] + 1e-6)
        
        return features
    
    def extract_vibration_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract vibration and frequency features.
        
        Args:
            data: DataFrame with vibration data
            
        Returns:
            DataFrame with vibration features
        """
        features = pd.DataFrame(index=data.index)
        
        # Vibration intensity features
        if 'vibration_intensity' in data.columns:
            vib_intensity = data['vibration_intensity']
            features['vib_intensity_mean'] = vib_intensity.rolling(window=10).mean()
            features['vib_intensity_std'] = vib_intensity.rolling(window=10).std()
            features['vib_intensity_max'] = vib_intensity.rolling(window=10).max()
            features['vib_intensity_rms'] = np.sqrt((vib_intensity ** 2).rolling(window=10).mean())
            
            # Vibration categories
            features['vibration_category'] = pd.cut(vib_intensity, 
                                                 bins=[0, 0.1, 0.5, 1.0, 2.0, 10], 
                                                 labels=['minimal', 'low', 'moderate', 'high', 'severe'])
        
        # Frequency domain features
        if 'dominant_frequency' in data.columns:
            features['dominant_freq_mean'] = data['dominant_frequency'].rolling(window=10).mean()
            features['dominant_freq_std'] = data['dominant_frequency'].rolling(window=10).std()
            
            # Frequency categories
            features['frequency_category'] = pd.cut(data['dominant_frequency'], 
                                                 bins=[0, 10, 50, 100, 500, 1000], 
                                                 labels=['very_low', 'low', 'medium', 'high', 'very_high'])
        
        # Spectral features
        if 'spectral_centroid' in data.columns:
            features['spectral_centroid_mean'] = data['spectral_centroid'].rolling(window=10).mean()
            features['spectral_centroid_std'] = data['spectral_centroid'].rolling(window=10).std()
        
        if 'spectral_bandwidth' in data.columns:
            features['spectral_bandwidth_mean'] = data['spectral_bandwidth'].rolling(window=10).mean()
            features['spectral_bandwidth_std'] = data['spectral_bandwidth'].rolling(window=10).std()
        
        # Resonance detection
        if 'resonance_frequency' in data.columns:
            features['resonance_detected'] = (data['resonance_frequency'] > 0).astype(int)
            features['resonance_freq_mean'] = data['resonance_frequency'].rolling(window=10).mean()
        
        # Harmonic features
        harmonic_cols = [col for col in data.columns if 'harmonic_' in col.lower()]
        if len(harmonic_cols) > 0:
            harmonic_data = data[harmonic_cols]
            features['harmonic_strength'] = harmonic_data.sum(axis=1)
            features['harmonic_strength_mean'] = features['harmonic_strength'].rolling(window=10).mean()
        
        return features
    
    def extract_motion_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract motion and velocity features.
        
        Args:
            data: DataFrame with motion data
            
        Returns:
            DataFrame with motion features
        """
        features = pd.DataFrame(index=data.index)
        
        # Velocity features (integrated from acceleration)
        if 'velocity_x' in data.columns:
            features['velocity_x_mean'] = data['velocity_x'].rolling(window=10).mean()
            features['velocity_x_std'] = data['velocity_x'].rolling(window=10).std()
            features['velocity_x_max'] = data['velocity_x'].rolling(window=10).max()
        
        if 'velocity_y' in data.columns:
            features['velocity_y_mean'] = data['velocity_y'].rolling(window=10).mean()
            features['velocity_y_std'] = data['velocity_y'].rolling(window=10).std()
            features['velocity_y_max'] = data['velocity_y'].rolling(window=10).max()
        
        if 'velocity_z' in data.columns:
            features['velocity_z_mean'] = data['velocity_z'].rolling(window=10).mean()
            features['velocity_z_std'] = data['velocity_z'].rolling(window=10).std()
            features['velocity_z_max'] = data['velocity_z'].rolling(window=10).max()
        
        # Combined velocity features
        velocity_cols = [col for col in data.columns if col.startswith('velocity_')]
        if len(velocity_cols) >= 2:
            velocity_data = data[velocity_cols]
            features['velocity_magnitude'] = np.sqrt((velocity_data ** 2).sum(axis=1))
            features['velocity_magnitude_mean'] = features['velocity_magnitude'].rolling(window=10).mean()
            features['velocity_magnitude_std'] = features['velocity_magnitude'].rolling(window=10).std()
        
        # Displacement features
        if 'displacement_x' in data.columns:
            features['displacement_x_mean'] = data['displacement_x'].rolling(window=10).mean()
            features['displacement_x_std'] = data['displacement_x'].rolling(window=10).std()
            features['displacement_x_max'] = data['displacement_x'].rolling(window=10).max()
        
        if 'displacement_y' in data.columns:
            features['displacement_y_mean'] = data['displacement_y'].rolling(window=10).mean()
            features['displacement_y_std'] = data['displacement_y'].rolling(window=10).std()
            features['displacement_y_max'] = data['displacement_y'].rolling(window=10).max()
        
        if 'displacement_z' in data.columns:
            features['displacement_z_mean'] = data['displacement_z'].rolling(window=10).mean()
            features['displacement_z_std'] = data['displacement_z'].rolling(window=10).std()
            features['displacement_z_max'] = data['displacement_z'].rolling(window=10).max()
        
        # Combined displacement features
        displacement_cols = [col for col in data.columns if col.startswith('displacement_')]
        if len(displacement_cols) >= 2:
            displacement_data = data[displacement_cols]
            features['displacement_magnitude'] = np.sqrt((displacement_data ** 2).sum(axis=1))
            features['displacement_magnitude_mean'] = features['displacement_magnitude'].rolling(window=10).mean()
            features['displacement_magnitude_std'] = features['displacement_magnitude'].rolling(window=10).std()
        
        return features
    
    def extract_stability_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract stability and balance features.
        
        Args:
            data: DataFrame with stability data
            
        Returns:
            DataFrame with stability features
        """
        features = pd.DataFrame(index=data.index)
        
        # Overall stability index
        stability_components = []
        
        # Acceleration stability
        if 'accel_magnitude' in data.columns:
            accel_stability = 1 / (data['accel_magnitude'].rolling(window=10).std() + 1e-6)
            stability_components.append(accel_stability)
        
        # Velocity stability
        if 'velocity_magnitude' in data.columns:
            vel_stability = 1 / (data['velocity_magnitude'].rolling(window=10).std() + 1e-6)
            stability_components.append(vel_stability)
        
        # Vibration stability
        if 'vibration_intensity' in data.columns:
            vib_stability = 1 / (data['vibration_intensity'].rolling(window=10).std() + 1e-6)
            stability_components.append(vib_stability)
        
        if len(stability_components) > 1:
            features['overall_stability'] = np.mean(stability_components, axis=0)
            features['overall_stability_mean'] = features['overall_stability'].rolling(window=10).mean()
            
            # Stability categories
            features['stability_category'] = pd.cut(features['overall_stability'], 
                                                 bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0], 
                                                 labels=['unstable', 'poor', 'fair', 'good', 'excellent'])
        
        # Motion pattern detection
        if 'accel_magnitude' in data.columns:
            # Detect periodic motion
            features['periodic_motion'] = (data['accel_magnitude'].rolling(window=20).std() > 
                                         data['accel_magnitude'].rolling(window=100).std()).astype(int)
            
            # Detect sudden movements
            features['sudden_movement'] = (data['accel_magnitude'].diff() > 
                                        2 * data['accel_magnitude'].rolling(window=20).std()).astype(int)
        
        return features
    
    def extract_anomaly_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract anomaly detection features.
        
        Args:
            data: DataFrame with accelerometer data
            
        Returns:
            DataFrame with anomaly features
        """
        features = pd.DataFrame(index=data.index)
        
        # Acceleration anomalies
        if 'accel_magnitude' in data.columns:
            accel_mag = data['accel_magnitude']
            accel_mean = accel_mag.rolling(window=50).mean()
            accel_std = accel_mag.rolling(window=50).std()
            
            # Z-score based anomaly detection
            features['accel_zscore'] = (accel_mag - accel_mean) / (accel_std + 1e-6)
            features['accel_anomaly'] = (np.abs(features['accel_zscore']) > 3).astype(int)
            
            # Spike detection
            features['accel_spike'] = (accel_mag.diff() > 2 * accel_std).astype(int)
        
        # Vibration anomalies
        if 'vibration_intensity' in data.columns:
            vib_intensity = data['vibration_intensity']
            vib_mean = vib_intensity.rolling(window=50).mean()
            vib_std = vib_intensity.rolling(window=50).std()
            
            features['vib_zscore'] = (vib_intensity - vib_mean) / (vib_std + 1e-6)
            features['vib_anomaly'] = (np.abs(features['vib_zscore']) > 3).astype(int)
        
        # Frequency anomalies
        if 'dominant_frequency' in data.columns:
            freq = data['dominant_frequency']
            freq_mean = freq.rolling(window=50).mean()
            freq_std = freq.rolling(window=50).std()
            
            features['freq_zscore'] = (freq - freq_mean) / (freq_std + 1e-6)
            features['freq_anomaly'] = (np.abs(features['freq_zscore']) > 3).astype(int)
        
        # Combined anomaly score
        anomaly_cols = [col for col in features.columns if 'anomaly' in col]
        if len(anomaly_cols) > 1:
            features['combined_anomaly_score'] = features[anomaly_cols].sum(axis=1)
            features['anomaly_detected'] = (features['combined_anomaly_score'] > 0).astype(int)
        
        return features
    
    def extract_all_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract all accelerometer features.
        
        Args:
            data: DataFrame with accelerometer data
            
        Returns:
            DataFrame with all extracted features
        """
        logger.info("Extracting accelerometer features...")
        
        # Validate input data
        self._validate_input_data(data)
        
        # Extract different feature groups
        feature_groups = [
            self.extract_acceleration_features(data),
            self.extract_vibration_features(data),
            self.extract_motion_features(data),
            self.extract_stability_features(data),
            self.extract_anomaly_features(data)
        ]
        
        # Combine all features
        all_features = pd.concat(feature_groups, axis=1)
        
        # Clean and validate features
        all_features = self._clean_features(all_features)
        
        logger.info(f"Extracted {len(all_features.columns)} accelerometer features")
        return all_features
    
    def _validate_input_data(self, data: pd.DataFrame) -> None:
        """Validate input data against configuration rules."""
        validation_rules = self.validation_rules
        
        # Validate acceleration data
        accel_cols = [col for col in data.columns if col.startswith('accel_')]
        for col in accel_cols:
            accel_range = validation_rules.get('acceleration_range', [-50, 50])
            invalid_accel = (data[col] < accel_range[0]) | (data[col] > accel_range[1])
            if invalid_accel.any():
                logger.warning(f"Found {invalid_accel.sum()} invalid {col} values")
        
        # Validate frequency data
        if 'dominant_frequency' in data.columns:
            freq_range = validation_rules.get('frequency_range', [0, 1000])
            invalid_freq = (data['dominant_frequency'] < freq_range[0]) | (data['dominant_frequency'] > freq_range[1])
            if invalid_freq.any():
                logger.warning(f"Found {invalid_freq.sum()} invalid frequency values")
        
        # Validate sampling rate
        if 'sampling_rate' in data.columns:
            sampling_range = validation_rules.get('sampling_rate_range', [1, 10000])
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
        Calculate feature importance for accelerometer data.
        
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
