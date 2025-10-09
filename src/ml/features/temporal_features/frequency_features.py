"""
Frequency Feature Engineering

This module extracts and engineers frequency domain features from time series data for PBF-LB/M processes.
Integrates with YAML configuration for feature definitions and validation rules.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import logging
from ..config.config_manager import ConfigManager

logger = logging.getLogger(__name__)


class FrequencyFeatures:
    """
    Feature engineering for frequency domain features in PBF-LB/M processes.
    
    Extracts features from frequency analysis, spectral characteristics, and harmonic patterns
    based on YAML configuration.
    """
    
    def __init__(self, config_manager: Optional[ConfigManager] = None):
        """
        Initialize frequency feature engineering.
        
        Args:
            config_manager: Configuration manager for loading feature definitions
        """
        self.config_manager = config_manager or ConfigManager()
        self.feature_config = self._load_feature_config()
        self.feature_definitions = self.feature_config.get('feature_definitions', {})
        self.validation_rules = self.feature_config.get('validation_rules', {})
        
    def _load_feature_config(self) -> Dict[str, Any]:
        """Load frequency feature configuration from YAML."""
        try:
            return self.config_manager.load_feature_config('temporal_features/frequency_features')
        except Exception as e:
            logger.warning(f"Could not load frequency config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for frequency features."""
        return {
            'feature_definitions': {
                'spectral_features': {
                    'frequencies': ['dominant', 'peak', 'centroid'],
                    'derived': ['spectral_energy', 'spectral_entropy']
                },
                'harmonic_features': {
                    'harmonics': [1, 2, 3, 5, 10],
                    'derived': ['harmonic_ratio', 'harmonic_strength']
                },
                'power_features': {
                    'bands': ['low', 'medium', 'high'],
                    'derived': ['power_ratio', 'bandwidth']
                }
            },
            'validation_rules': {
                'frequency_range': [0.001, 1000],  # Hz
                'sampling_rate_range': [1, 10000],  # Hz
                'window_size_range': [32, 2048]
            }
        }
    
    def extract_spectral_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract spectral features from time series data.
        
        Args:
            data: DataFrame with time series data
            
        Returns:
            DataFrame with spectral features
        """
        features = pd.DataFrame(index=data.index)
        
        # Get numeric columns for spectral analysis
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col in data.columns:
                series = data[col]
                
                # Remove NaN values
                series_clean = series.dropna()
                
                if len(series_clean) > 32:  # Minimum window size for FFT
                    # Apply FFT
                    fft = np.fft.fft(series_clean)
                    freqs = np.fft.fftfreq(len(series_clean))
                    power_spectrum = np.abs(fft) ** 2
                    
                    # Dominant frequency
                    dominant_freq_idx = np.argmax(power_spectrum[1:len(power_spectrum)//2]) + 1
                    dominant_freq = freqs[dominant_freq_idx]
                    features[f'{col}_dominant_frequency'] = dominant_freq
                    
                    # Peak frequency
                    peaks, _ = self._find_peaks(power_spectrum[1:len(power_spectrum)//2])
                    if len(peaks) > 0:
                        peak_freq = freqs[peaks[0] + 1]
                        features[f'{col}_peak_frequency'] = peak_freq
                    
                    # Spectral centroid
                    spectral_centroid = np.sum(freqs * power_spectrum) / (np.sum(power_spectrum) + 1e-6)
                    features[f'{col}_spectral_centroid'] = spectral_centroid
                    
                    # Spectral energy
                    spectral_energy = np.sum(power_spectrum)
                    features[f'{col}_spectral_energy'] = spectral_energy
                    
                    # Spectral entropy
                    normalized_spectrum = power_spectrum / (np.sum(power_spectrum) + 1e-6)
                    spectral_entropy = -np.sum(normalized_spectrum * np.log(normalized_spectrum + 1e-6))
                    features[f'{col}_spectral_entropy'] = spectral_entropy
                    
                    # Spectral bandwidth
                    spectral_bandwidth = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * power_spectrum) / (np.sum(power_spectrum) + 1e-6))
                    features[f'{col}_spectral_bandwidth'] = spectral_bandwidth
                    
                    # Spectral rolloff
                    cumulative_energy = np.cumsum(power_spectrum)
                    total_energy = cumulative_energy[-1]
                    rolloff_threshold = 0.85 * total_energy
                    rolloff_idx = np.where(cumulative_energy >= rolloff_threshold)[0]
                    if len(rolloff_idx) > 0:
                        spectral_rolloff = freqs[rolloff_idx[0]]
                        features[f'{col}_spectral_rolloff'] = spectral_rolloff
        
        return features
    
    def extract_harmonic_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract harmonic features from time series data.
        
        Args:
            data: DataFrame with time series data
            
        Returns:
            DataFrame with harmonic features
        """
        features = pd.DataFrame(index=data.index)
        
        # Get harmonic configuration
        harmonic_config = self.feature_definitions.get('harmonic_features', {})
        harmonics = harmonic_config.get('harmonics', [1, 2, 3, 5, 10])
        
        # Get numeric columns for harmonic analysis
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col in data.columns:
                series = data[col]
                
                # Remove NaN values
                series_clean = series.dropna()
                
                if len(series_clean) > 32:
                    # Apply FFT
                    fft = np.fft.fft(series_clean)
                    freqs = np.fft.fftfreq(len(series_clean))
                    power_spectrum = np.abs(fft) ** 2
                    
                    # Find fundamental frequency
                    fundamental_idx = np.argmax(power_spectrum[1:len(power_spectrum)//2]) + 1
                    fundamental_freq = freqs[fundamental_idx]
                    
                    # Harmonic analysis
                    harmonic_strengths = []
                    for harmonic in harmonics:
                        harmonic_freq = fundamental_freq * harmonic
                        harmonic_idx = np.argmin(np.abs(freqs - harmonic_freq))
                        harmonic_strength = power_spectrum[harmonic_idx]
                        features[f'{col}_harmonic_{harmonic}_strength'] = harmonic_strength
                        harmonic_strengths.append(harmonic_strength)
                    
                    # Harmonic ratio
                    if len(harmonic_strengths) > 1:
                        fundamental_strength = harmonic_strengths[0]
                        harmonic_ratio = np.sum(harmonic_strengths[1:]) / (fundamental_strength + 1e-6)
                        features[f'{col}_harmonic_ratio'] = harmonic_ratio
                    
                    # Total harmonic strength
                    total_harmonic_strength = np.sum(harmonic_strengths)
                    features[f'{col}_total_harmonic_strength'] = total_harmonic_strength
                    
                    # Harmonic distortion
                    if len(harmonic_strengths) > 1:
                        fundamental_strength = harmonic_strengths[0]
                        harmonic_distortion = np.sqrt(np.sum([h**2 for h in harmonic_strengths[1:]])) / (fundamental_strength + 1e-6)
                        features[f'{col}_harmonic_distortion'] = harmonic_distortion
        
        return features
    
    def extract_power_spectral_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract power spectral features from time series data.
        
        Args:
            data: DataFrame with time series data
            
        Returns:
            DataFrame with power spectral features
        """
        features = pd.DataFrame(index=data.index)
        
        # Get power band configuration
        power_config = self.feature_definitions.get('power_features', {})
        bands = power_config.get('bands', ['low', 'medium', 'high'])
        
        # Get numeric columns for power analysis
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col in data.columns:
                series = data[col]
                
                # Remove NaN values
                series_clean = series.dropna()
                
                if len(series_clean) > 32:
                    # Apply FFT
                    fft = np.fft.fft(series_clean)
                    freqs = np.fft.fftfreq(len(series_clean))
                    power_spectrum = np.abs(fft) ** 2
                    
                    # Define frequency bands
                    nyquist = 0.5
                    band_limits = {
                        'low': (0, nyquist * 0.25),
                        'medium': (nyquist * 0.25, nyquist * 0.75),
                        'high': (nyquist * 0.75, nyquist)
                    }
                    
                    # Calculate power in each band
                    band_powers = {}
                    for band_name, (low_freq, high_freq) in band_limits.items():
                        band_mask = (freqs >= low_freq) & (freqs <= high_freq)
                        band_power = np.sum(power_spectrum[band_mask])
                        band_powers[band_name] = band_power
                        features[f'{col}_power_{band_name}'] = band_power
                    
                    # Power ratios
                    if len(band_powers) > 1:
                        total_power = sum(band_powers.values())
                        for band_name, band_power in band_powers.items():
                            features[f'{col}_power_ratio_{band_name}'] = band_power / (total_power + 1e-6)
                    
                    # Bandwidth
                    spectral_centroid = np.sum(freqs * power_spectrum) / (np.sum(power_spectrum) + 1e-6)
                    bandwidth = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * power_spectrum) / (np.sum(power_spectrum) + 1e-6))
                    features[f'{col}_bandwidth'] = bandwidth
                    
                    # Power spectral density
                    psd = power_spectrum / (np.sum(power_spectrum) + 1e-6)
                    features[f'{col}_psd_mean'] = np.mean(psd)
                    features[f'{col}_psd_std'] = np.std(psd)
        
        return features
    
    def extract_wavelet_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract wavelet features from time series data.
        
        Args:
            data: DataFrame with time series data
            
        Returns:
            DataFrame with wavelet features
        """
        features = pd.DataFrame(index=data.index)
        
        # Get numeric columns for wavelet analysis
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col in data.columns:
                series = data[col]
                
                # Remove NaN values
                series_clean = series.dropna()
                
                if len(series_clean) > 16:  # Minimum for wavelet analysis
                    # Simple wavelet-like features using moving averages
                    # This is a simplified implementation
                    
                    # Wavelet energy at different scales
                    scales = [2, 4, 8, 16]
                    for scale in scales:
                        if len(series_clean) > scale:
                            # Downsample by scale
                            downsampled = series_clean[::scale]
                            wavelet_energy = np.sum(downsampled ** 2)
                            features[f'{col}_wavelet_energy_scale_{scale}'] = wavelet_energy
                            
                            # Wavelet variance
                            wavelet_variance = np.var(downsampled)
                            features[f'{col}_wavelet_variance_scale_{scale}'] = wavelet_variance
                    
                    # Wavelet entropy
                    total_energy = np.sum(series_clean ** 2)
                    if total_energy > 0:
                        normalized_energy = (series_clean ** 2) / total_energy
                        wavelet_entropy = -np.sum(normalized_energy * np.log(normalized_energy + 1e-6))
                        features[f'{col}_wavelet_entropy'] = wavelet_entropy
        
        return features
    
    def extract_frequency_domain_statistics(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract frequency domain statistical features.
        
        Args:
            data: DataFrame with time series data
            
        Returns:
            DataFrame with frequency domain statistical features
        """
        features = pd.DataFrame(index=data.index)
        
        # Get numeric columns for frequency domain analysis
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col in data.columns:
                series = data[col]
                
                # Remove NaN values
                series_clean = series.dropna()
                
                if len(series_clean) > 32:
                    # Apply FFT
                    fft = np.fft.fft(series_clean)
                    freqs = np.fft.fftfreq(len(series_clean))
                    power_spectrum = np.abs(fft) ** 2
                    
                    # Frequency domain statistics
                    features[f'{col}_freq_domain_mean'] = np.mean(power_spectrum)
                    features[f'{col}_freq_domain_std'] = np.std(power_spectrum)
                    features[f'{col}_freq_domain_min'] = np.min(power_spectrum)
                    features[f'{col}_freq_domain_max'] = np.max(power_spectrum)
                    features[f'{col}_freq_domain_median'] = np.median(power_spectrum)
                    
                    # Frequency domain skewness and kurtosis
                    features[f'{col}_freq_domain_skewness'] = self._calculate_skewness(power_spectrum)
                    features[f'{col}_freq_domain_kurtosis'] = self._calculate_kurtosis(power_spectrum)
                    
                    # Frequency domain percentiles
                    features[f'{col}_freq_domain_p25'] = np.percentile(power_spectrum, 25)
                    features[f'{col}_freq_domain_p75'] = np.percentile(power_spectrum, 75)
                    features[f'{col}_freq_domain_p90'] = np.percentile(power_spectrum, 90)
                    features[f'{col}_freq_domain_p95'] = np.percentile(power_spectrum, 95)
        
        return features
    
    def extract_all_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract all frequency features.
        
        Args:
            data: DataFrame with time series data
            
        Returns:
            DataFrame with all extracted features
        """
        logger.info("Extracting frequency features...")
        
        # Validate input data
        self._validate_input_data(data)
        
        # Extract different feature groups
        feature_groups = [
            self.extract_spectral_features(data),
            self.extract_harmonic_features(data),
            self.extract_power_spectral_features(data),
            self.extract_wavelet_features(data),
            self.extract_frequency_domain_statistics(data)
        ]
        
        # Combine all features
        all_features = pd.concat(feature_groups, axis=1)
        
        # Clean and validate features
        all_features = self._clean_features(all_features)
        
        logger.info(f"Extracted {len(all_features.columns)} frequency features")
        return all_features
    
    def _find_peaks(self, signal, height=None, distance=None):
        """Simple peak finding function."""
        peaks = []
        if len(signal) < 3:
            return peaks, {}
        
        for i in range(1, len(signal) - 1):
            if signal[i] > signal[i-1] and signal[i] > signal[i+1]:
                if height is None or signal[i] > height:
                    if distance is None or len(peaks) == 0 or i - peaks[-1] > distance:
                        peaks.append(i)
        
        return peaks, {}
    
    def _calculate_skewness(self, data):
        """Calculate skewness of data."""
        if len(data) < 3:
            return 0
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        skewness = np.mean(((data - mean) / std) ** 3)
        return skewness
    
    def _calculate_kurtosis(self, data):
        """Calculate kurtosis of data."""
        if len(data) < 4:
            return 0
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        kurtosis = np.mean(((data - mean) / std) ** 4) - 3
        return kurtosis
    
    def _validate_input_data(self, data: pd.DataFrame) -> None:
        """Validate input data against configuration rules."""
        validation_rules = self.validation_rules
        
        # Validate frequency data
        if 'frequency' in data.columns:
            freq_range = validation_rules.get('frequency_range', [0.001, 1000])
            invalid_freq = (data['frequency'] < freq_range[0]) | (data['frequency'] > freq_range[1])
            if invalid_freq.any():
                logger.warning(f"Found {invalid_freq.sum()} invalid frequency values")
        
        # Validate sampling rate data
        if 'sampling_rate' in data.columns:
            sampling_range = validation_rules.get('sampling_rate_range', [1, 10000])
            invalid_sampling = (data['sampling_rate'] < sampling_range[0]) | (data['sampling_rate'] > sampling_range[1])
            if invalid_sampling.any():
                logger.warning(f"Found {invalid_sampling.sum()} invalid sampling rate values")
        
        # Validate window size data
        if 'window_size' in data.columns:
            window_range = validation_rules.get('window_size_range', [32, 2048])
            invalid_window = (data['window_size'] < window_range[0]) | (data['window_size'] > window_range[1])
            if invalid_window.any():
                logger.warning(f"Found {invalid_window.sum()} invalid window size values")
    
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
        Calculate feature importance for frequency features.
        
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
