"""
Sensor Analysis for PBF-LB/M Systems

This module provides specialized sensor analysis capabilities for PBF-LB/M
additive manufacturing systems, including ISPM sensor analysis, CT sensor
analysis, and multi-sensor data fusion.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import logging
from datetime import datetime
from scipy import signal
from scipy.stats import zscore
import warnings

logger = logging.getLogger(__name__)


@dataclass
class SensorAnalysisConfig:
    """Configuration for sensor analysis."""
    
    # Signal processing parameters
    sampling_rate: float = 1000.0  # Hz
    filter_type: str = "butterworth"  # "butterworth", "chebyshev", "ellip"
    filter_order: int = 4
    cutoff_frequency: float = 100.0  # Hz
    
    # Anomaly detection parameters
    anomaly_threshold: float = 3.0  # Standard deviations
    window_size: int = 100
    
    # Analysis parameters
    confidence_level: float = 0.95


@dataclass
class SensorAnalysisResult:
    """Result of sensor analysis."""
    
    success: bool
    method: str
    sensor_data: pd.DataFrame
    processed_data: pd.DataFrame
    anomaly_detection: Dict[str, Any]
    signal_statistics: Dict[str, float]
    analysis_time: float = 0.0
    error_message: Optional[str] = None


class SensorAnalyzer:
    """
    Sensor analyzer for PBF-LB/M systems.
    
    This class provides specialized sensor analysis capabilities including
    signal processing, anomaly detection, and multi-sensor data fusion
    for PBF-LB/M additive manufacturing.
    """
    
    def __init__(self, config: SensorAnalysisConfig = None):
        """Initialize the sensor analyzer."""
        self.config = config or SensorAnalysisConfig()
        self.analysis_cache = {}
        
        logger.info("Sensor Analyzer initialized")
    
    def analyze_sensor_data(
        self,
        sensor_data: pd.DataFrame,
        sensor_columns: List[str] = None
    ) -> SensorAnalysisResult:
        """
        Perform comprehensive sensor data analysis.
        
        Args:
            sensor_data: DataFrame containing sensor data
            sensor_columns: List of sensor column names (optional)
            
        Returns:
            SensorAnalysisResult: Sensor analysis results
        """
        try:
            start_time = datetime.now()
            
            if sensor_columns is None:
                sensor_columns = list(sensor_data.columns)
            
            # Process sensor data
            processed_data = self._process_sensor_signals(sensor_data, sensor_columns)
            
            # Detect anomalies
            anomaly_detection = self._detect_anomalies(processed_data, sensor_columns)
            
            # Calculate signal statistics
            signal_statistics = self._calculate_signal_statistics(processed_data, sensor_columns)
            
            # Calculate analysis time
            analysis_time = (datetime.now() - start_time).total_seconds()
            
            # Create result
            result = SensorAnalysisResult(
                success=True,
                method="SensorAnalysis",
                sensor_data=sensor_data,
                processed_data=processed_data,
                anomaly_detection=anomaly_detection,
                signal_statistics=signal_statistics,
                analysis_time=analysis_time
            )
            
            # Cache result
            self._cache_result("sensor_analysis", result)
            
            logger.info(f"Sensor analysis completed: {analysis_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Error in sensor analysis: {e}")
            return SensorAnalysisResult(
                success=False,
                method="SensorAnalysis",
                sensor_data=sensor_data,
                processed_data=pd.DataFrame(),
                anomaly_detection={},
                signal_statistics={},
                error_message=str(e)
            )
    
    def _process_sensor_signals(self, sensor_data: pd.DataFrame, sensor_columns: List[str]) -> pd.DataFrame:
        """Process sensor signals with filtering and normalization."""
        processed_data = sensor_data.copy()
        
        for column in sensor_columns:
            if column in sensor_data.columns:
                signal_data = sensor_data[column].values
                
                # Remove NaN values
                valid_mask = ~np.isnan(signal_data)
                if not np.any(valid_mask):
                    continue
                
                signal_data = signal_data[valid_mask]
                
                # Apply low-pass filter
                filtered_signal = self._apply_lowpass_filter(signal_data)
                
                # Normalize signal
                normalized_signal = self._normalize_signal(filtered_signal)
                
                # Update processed data
                processed_data.loc[valid_mask, f"{column}_processed"] = normalized_signal
        
        return processed_data
    
    def _apply_lowpass_filter(self, signal_data: np.ndarray) -> np.ndarray:
        """Apply low-pass filter to signal data."""
        try:
            # Design filter
            nyquist = self.config.sampling_rate / 2
            normal_cutoff = self.config.cutoff_frequency / nyquist
            
            if self.config.filter_type == "butterworth":
                b, a = signal.butter(self.config.filter_order, normal_cutoff, btype='low')
            elif self.config.filter_type == "chebyshev":
                b, a = signal.cheby1(self.config.filter_order, 1, normal_cutoff, btype='low')
            elif self.config.filter_type == "ellip":
                b, a = signal.ellip(self.config.filter_order, 1, 40, normal_cutoff, btype='low')
            else:
                return signal_data  # No filtering
            
            # Apply filter
            filtered_signal = signal.filtfilt(b, a, signal_data)
            return filtered_signal
            
        except Exception as e:
            logger.warning(f"Filtering failed: {e}")
            return signal_data
    
    def _normalize_signal(self, signal_data: np.ndarray) -> np.ndarray:
        """Normalize signal data."""
        # Z-score normalization
        normalized_signal = zscore(signal_data)
        return normalized_signal
    
    def _detect_anomalies(self, processed_data: pd.DataFrame, sensor_columns: List[str]) -> Dict[str, Any]:
        """Detect anomalies in sensor data."""
        anomaly_detection = {}
        
        for column in sensor_columns:
            processed_column = f"{column}_processed"
            if processed_column in processed_data.columns:
                signal_data = processed_data[processed_column].values
                valid_mask = ~np.isnan(signal_data)
                
                if np.any(valid_mask):
                    valid_signal = signal_data[valid_mask]
                    
                    # Calculate z-scores
                    z_scores = np.abs(zscore(valid_signal))
                    
                    # Detect anomalies
                    anomaly_mask = z_scores > self.config.anomaly_threshold
                    anomalies = np.where(anomaly_mask)[0]
                    
                    anomaly_detection[column] = {
                        'anomaly_count': len(anomalies),
                        'anomaly_percentage': len(anomalies) / len(valid_signal) * 100,
                        'max_z_score': np.max(z_scores),
                        'anomaly_indices': anomalies
                    }
        
        return anomaly_detection
    
    def _calculate_signal_statistics(self, processed_data: pd.DataFrame, sensor_columns: List[str]) -> Dict[str, float]:
        """Calculate signal statistics."""
        signal_statistics = {}
        
        for column in sensor_columns:
            processed_column = f"{column}_processed"
            if processed_column in processed_data.columns:
                signal_data = processed_data[processed_column].values
                valid_mask = ~np.isnan(signal_data)
                
                if np.any(valid_mask):
                    valid_signal = signal_data[valid_mask]
                    
                    signal_statistics[f"{column}_mean"] = np.mean(valid_signal)
                    signal_statistics[f"{column}_std"] = np.std(valid_signal)
                    signal_statistics[f"{column}_min"] = np.min(valid_signal)
                    signal_statistics[f"{column}_max"] = np.max(valid_signal)
                    signal_statistics[f"{column}_rms"] = np.sqrt(np.mean(valid_signal**2))
        
        return signal_statistics
    
    def _cache_result(self, method: str, result: SensorAnalysisResult):
        """Cache analysis result."""
        cache_key = f"{method}_{hash(str(result.sensor_data.columns))}"
        self.analysis_cache[cache_key] = result
    
    def get_cached_result(self, method: str, sensor_columns: List[str]) -> Optional[SensorAnalysisResult]:
        """Get cached analysis result."""
        cache_key = f"{method}_{hash(str(sensor_columns))}"
        return self.analysis_cache.get(cache_key)
    
    def clear_cache(self):
        """Clear analysis cache."""
        self.analysis_cache.clear()
    
    def get_analysis_statistics(self) -> Dict[str, Any]:
        """Get analysis statistics."""
        return {
            'cache_size': len(self.analysis_cache),
            'config': {
                'sampling_rate': self.config.sampling_rate,
                'filter_type': self.config.filter_type,
                'filter_order': self.config.filter_order,
                'cutoff_frequency': self.config.cutoff_frequency,
                'anomaly_threshold': self.config.anomaly_threshold
            }
        }


class ISPMAnalyzer(SensorAnalyzer):
    """Specialized ISPM sensor analyzer."""
    
    def __init__(self, config: SensorAnalysisConfig = None):
        super().__init__(config)
        self.method_name = "ISPMAnalyzer"
    
    def analyze_ispm_data(self, ispm_data: pd.DataFrame) -> SensorAnalysisResult:
        """Analyze ISPM sensor data."""
        return self.analyze_sensor_data(ispm_data)


class CTSensorAnalyzer(SensorAnalyzer):
    """Specialized CT sensor analyzer."""
    
    def __init__(self, config: SensorAnalysisConfig = None):
        super().__init__(config)
        self.method_name = "CTSensorAnalyzer"
    
    def analyze_ct_data(self, ct_data: pd.DataFrame) -> SensorAnalysisResult:
        """Analyze CT sensor data."""
        return self.analyze_sensor_data(ct_data)
