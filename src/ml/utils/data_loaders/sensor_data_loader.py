"""
Sensor Data Loader

This module implements utilities for loading sensor data from ISPM
(In-Situ Process Monitoring) systems in PBF-LB/M manufacturing.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Union, Tuple
from pathlib import Path
import json
import yaml
from datetime import datetime, timedelta
import h5py

logger = logging.getLogger(__name__)


class SensorDataLoader:
    """
    Utility class for loading sensor data from ISPM systems.
    
    This class handles:
    - Loading sensor data from various file formats
    - Loading from time-series databases
    - Sensor data validation and calibration
    - Time synchronization across multiple sensors
    - Data aggregation and resampling
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the sensor data loader.
        
        Args:
            config: Configuration dictionary with sensor settings
        """
        self.config = config or {}
        self.supported_formats = ['csv', 'json', 'hdf5', 'parquet', 'npy']
        
        # Define sensor types and their expected data ranges
        self.sensor_types = {
            'pyrometer': {
                'temperature_range': (500, 3000),  # °C
                'sampling_rate': 1000,  # Hz
                'units': '°C'
            },
            'camera': {
                'intensity_range': (0, 255),  # 8-bit
                'sampling_rate': 30,  # FPS
                'units': 'pixel_intensity'
            },
            'accelerometer': {
                'acceleration_range': (-50, 50),  # m/s²
                'sampling_rate': 1000,  # Hz
                'units': 'm/s²'
            },
            'temperature_sensor': {
                'temperature_range': (0, 500),  # °C
                'sampling_rate': 10,  # Hz
                'units': '°C'
            },
            'pressure_sensor': {
                'pressure_range': (0, 10),  # bar
                'sampling_rate': 100,  # Hz
                'units': 'bar'
            },
            'flow_sensor': {
                'flow_range': (0, 100),  # L/min
                'sampling_rate': 10,  # Hz
                'units': 'L/min'
            }
        }
        
        logger.info("Initialized SensorDataLoader")
    
    def load_from_file(self, file_path: Union[str, Path], 
                      sensor_type: str,
                      file_format: Optional[str] = None) -> pd.DataFrame:
        """
        Load sensor data from a file.
        
        Args:
            file_path: Path to the sensor data file
            sensor_type: Type of sensor (pyrometer, camera, accelerometer, etc.)
            file_format: File format (csv, json, hdf5, parquet, npy). If None, inferred from extension
            
        Returns:
            DataFrame with sensor data
            
        Raises:
            ValueError: If sensor type or file format is not supported
            FileNotFoundError: If file does not exist
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if sensor_type not in self.sensor_types:
            raise ValueError(f"Unsupported sensor type: {sensor_type}")
        
        # Determine file format
        if file_format is None:
            file_format = file_path.suffix.lower().lstrip('.')
        
        if file_format not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {file_format}")
        
        try:
            if file_format == 'csv':
                return self._load_csv(file_path, sensor_type)
            elif file_format == 'json':
                return self._load_json(file_path, sensor_type)
            elif file_format == 'hdf5':
                return self._load_hdf5(file_path, sensor_type)
            elif file_format == 'parquet':
                return self._load_parquet(file_path, sensor_type)
            elif file_format == 'npy':
                return self._load_npy(file_path, sensor_type)
        except Exception as e:
            logger.error(f"Failed to load sensor data from {file_path}: {e}")
            raise
    
    def _load_csv(self, file_path: Path, sensor_type: str) -> pd.DataFrame:
        """Load sensor data from CSV file."""
        df = pd.read_csv(file_path)
        return self._validate_and_convert_sensor_data(df, sensor_type)
    
    def _load_json(self, file_path: Path, sensor_type: str) -> pd.DataFrame:
        """Load sensor data from JSON file."""
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Handle different JSON structures
        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, dict):
            if 'sensor_data' in data:
                df = pd.DataFrame(data['sensor_data'])
            elif 'data' in data:
                df = pd.DataFrame(data['data'])
            else:
                df = pd.DataFrame([data])
        else:
            raise ValueError("Invalid JSON structure for sensor data")
        
        return self._validate_and_convert_sensor_data(df, sensor_type)
    
    def _load_hdf5(self, file_path: Path, sensor_type: str) -> pd.DataFrame:
        """Load sensor data from HDF5 file."""
        try:
            with h5py.File(file_path, 'r') as f:
                # Try to find sensor data in common locations
                data_paths = ['/sensor_data', '/data', f'/{sensor_type}', '/']
                
                data = None
                for path in data_paths:
                    if path in f:
                        data = f[path][:]
                        break
                
                if data is None:
                    # If no specific path found, try to load the first dataset
                    keys = list(f.keys())
                    if keys:
                        data = f[keys[0]][:]
                    else:
                        raise ValueError("No data found in HDF5 file")
                
                # Convert to DataFrame
                if data.ndim == 1:
                    df = pd.DataFrame({sensor_type: data})
                elif data.ndim == 2:
                    df = pd.DataFrame(data)
                else:
                    raise ValueError(f"Unsupported data dimensions: {data.ndim}")
                
                return self._validate_and_convert_sensor_data(df, sensor_type)
                
        except ImportError:
            raise ImportError("h5py is required for HDF5 support. Install with: pip install h5py")
    
    def _load_parquet(self, file_path: Path, sensor_type: str) -> pd.DataFrame:
        """Load sensor data from Parquet file."""
        df = pd.read_parquet(file_path)
        return self._validate_and_convert_sensor_data(df, sensor_type)
    
    def _load_npy(self, file_path: Path, sensor_type: str) -> pd.DataFrame:
        """Load sensor data from NumPy file."""
        data = np.load(file_path)
        
        # Convert to DataFrame
        if data.ndim == 1:
            df = pd.DataFrame({sensor_type: data})
        elif data.ndim == 2:
            df = pd.DataFrame(data)
        else:
            raise ValueError(f"Unsupported data dimensions: {data.ndim}")
        
        return self._validate_and_convert_sensor_data(df, sensor_type)
    
    def _validate_and_convert_sensor_data(self, df: pd.DataFrame, sensor_type: str) -> pd.DataFrame:
        """
        Validate and convert sensor data to appropriate types.
        
        Args:
            df: Raw DataFrame
            sensor_type: Type of sensor
            
        Returns:
            Validated and converted DataFrame
        """
        sensor_config = self.sensor_types[sensor_type]
        
        # Ensure timestamp column exists
        if 'timestamp' not in df.columns:
            if 'time' in df.columns:
                df = df.rename(columns={'time': 'timestamp'})
            else:
                # Create timestamp column if missing
                df['timestamp'] = pd.date_range(
                    start=datetime.now() - timedelta(seconds=len(df)/sensor_config['sampling_rate']),
                    periods=len(df),
                    freq=f"{1/sensor_config['sampling_rate']*1000}ms"
                )
        
        # Convert timestamp to datetime
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        except (ValueError, TypeError) as e:
            logger.warning(f"Failed to convert timestamp to datetime: {e}")
        
        # Validate sensor-specific data
        self._validate_sensor_ranges(df, sensor_type, sensor_config)
        
        # Remove outliers if configured
        if self.config.get('remove_outliers', False):
            df = self._remove_outliers(df, sensor_type, sensor_config)
        
        return df
    
    def _validate_sensor_ranges(self, df: pd.DataFrame, sensor_type: str, sensor_config: Dict[str, Any]):
        """
        Validate that sensor data is within expected ranges.
        
        Args:
            df: DataFrame to validate
            sensor_type: Type of sensor
            sensor_config: Sensor configuration
        """
        # Get the main data column (excluding timestamp)
        data_columns = [col for col in df.columns if col != 'timestamp']
        
        for col in data_columns:
            if col in sensor_config:
                range_key = f"{col}_range"
                if range_key in sensor_config:
                    min_val, max_val = sensor_config[range_key]
                    invalid_count = ((df[col] < min_val) | (df[col] > max_val)).sum()
                    if invalid_count > 0:
                        logger.warning(f"{invalid_count} values of {col} are outside range [{min_val}, {max_val}]")
            else:
                # Use default range for the sensor type
                range_key = f"{sensor_type}_range"
                if range_key in sensor_config:
                    min_val, max_val = sensor_config[range_key]
                    invalid_count = ((df[col] < min_val) | (df[col] > max_val)).sum()
                    if invalid_count > 0:
                        logger.warning(f"{invalid_count} values of {col} are outside range [{min_val}, {max_val}]")
    
    def _remove_outliers(self, df: pd.DataFrame, sensor_type: str, sensor_config: Dict[str, Any]) -> pd.DataFrame:
        """
        Remove outliers from sensor data.
        
        Args:
            df: DataFrame with sensor data
            sensor_type: Type of sensor
            sensor_config: Sensor configuration
            
        Returns:
            DataFrame with outliers removed
        """
        data_columns = [col for col in df.columns if col != 'timestamp']
        
        for col in data_columns:
            if col in df.columns:
                # Use IQR method to detect outliers
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Remove outliers
                outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
                outlier_count = outlier_mask.sum()
                
                if outlier_count > 0:
                    logger.info(f"Removing {outlier_count} outliers from {col}")
                    df = df[~outlier_mask]
        
        return df
    
    def load_time_series_data(self, start_time: datetime, 
                             end_time: datetime,
                             sensor_types: List[str],
                             sampling_rate: Optional[int] = None) -> Dict[str, pd.DataFrame]:
        """
        Load time-series sensor data for multiple sensors.
        
        Args:
            start_time: Start time for data retrieval
            end_time: End time for data retrieval
            sensor_types: List of sensor types to retrieve
            sampling_rate: Override sampling rate (Hz)
            
        Returns:
            Dictionary mapping sensor types to DataFrames
        """
        sensor_data = {}
        
        for sensor_type in sensor_types:
            if sensor_type not in self.sensor_types:
                logger.warning(f"Unknown sensor type: {sensor_type}")
                continue
            
            sensor_config = self.sensor_types[sensor_type]
            rate = sampling_rate or sensor_config['sampling_rate']
            
            # Calculate number of samples
            duration = (end_time - start_time).total_seconds()
            num_samples = int(duration * rate)
            
            # Generate time series
            timestamps = pd.date_range(start=start_time, end=end_time, periods=num_samples)
            
            # Generate sample data (in real implementation, this would come from database)
            if sensor_type == 'pyrometer':
                data = np.random.normal(2000, 200, num_samples)  # Temperature in °C
            elif sensor_type == 'camera':
                data = np.random.randint(0, 256, num_samples)  # Pixel intensity
            elif sensor_type == 'accelerometer':
                data = np.random.normal(0, 5, num_samples)  # Acceleration in m/s²
            elif sensor_type == 'temperature_sensor':
                data = np.random.normal(100, 20, num_samples)  # Temperature in °C
            elif sensor_type == 'pressure_sensor':
                data = np.random.normal(1, 0.1, num_samples)  # Pressure in bar
            elif sensor_type == 'flow_sensor':
                data = np.random.normal(10, 2, num_samples)  # Flow in L/min
            else:
                data = np.random.normal(0, 1, num_samples)  # Default
            
            df = pd.DataFrame({
                'timestamp': timestamps,
                sensor_type: data
            })
            
            sensor_data[sensor_type] = self._validate_and_convert_sensor_data(df, sensor_type)
        
        return sensor_data
    
    def synchronize_sensor_data(self, sensor_data: Dict[str, pd.DataFrame], 
                              reference_sensor: str = 'pyrometer') -> Dict[str, pd.DataFrame]:
        """
        Synchronize sensor data to a common time base.
        
        Args:
            sensor_data: Dictionary of sensor DataFrames
            reference_sensor: Reference sensor for time synchronization
            
        Returns:
            Dictionary of synchronized sensor DataFrames
        """
        if reference_sensor not in sensor_data:
            raise ValueError(f"Reference sensor {reference_sensor} not found in data")
        
        reference_df = sensor_data[reference_sensor]
        reference_timestamps = reference_df['timestamp']
        
        synchronized_data = {reference_sensor: reference_df}
        
        for sensor_type, df in sensor_data.items():
            if sensor_type == reference_sensor:
                continue
            
            # Interpolate data to reference timestamps
            df_sync = df.set_index('timestamp')
            df_sync = df_sync.reindex(reference_timestamps, method='linear')
            df_sync = df_sync.reset_index()
            df_sync = df_sync.rename(columns={'index': 'timestamp'})
            
            synchronized_data[sensor_type] = df_sync
        
        return synchronized_data
    
    def resample_sensor_data(self, df: pd.DataFrame, 
                           target_frequency: str = '1S',
                           aggregation_method: str = 'mean') -> pd.DataFrame:
        """
        Resample sensor data to a different frequency.
        
        Args:
            df: DataFrame with sensor data
            target_frequency: Target frequency (e.g., '1S', '100ms', '1min')
            aggregation_method: Aggregation method (mean, median, max, min, sum)
            
        Returns:
            Resampled DataFrame
        """
        if 'timestamp' not in df.columns:
            raise ValueError("DataFrame must have a 'timestamp' column")
        
        df_resampled = df.set_index('timestamp')
        
        # Resample data
        if aggregation_method == 'mean':
            df_resampled = df_resampled.resample(target_frequency).mean()
        elif aggregation_method == 'median':
            df_resampled = df_resampled.resample(target_frequency).median()
        elif aggregation_method == 'max':
            df_resampled = df_resampled.resample(target_frequency).max()
        elif aggregation_method == 'min':
            df_resampled = df_resampled.resample(target_frequency).min()
        elif aggregation_method == 'sum':
            df_resampled = df_resampled.resample(target_frequency).sum()
        else:
            raise ValueError(f"Unsupported aggregation method: {aggregation_method}")
        
        # Remove NaN values
        df_resampled = df_resampled.dropna()
        
        return df_resampled.reset_index()
    
    def load_calibration_data(self, sensor_id: str) -> Dict[str, Any]:
        """
        Load calibration data for a specific sensor.
        
        Args:
            sensor_id: Unique sensor identifier
            
        Returns:
            Dictionary with calibration parameters
        """
        # This would typically load from a calibration database
        # For now, return default calibration data
        
        calibration_data = {
            'sensor_id': sensor_id,
            'calibration_date': datetime.now(),
            'calibration_factor': 1.0,
            'offset': 0.0,
            'linearity_error': 0.01,
            'repeatability': 0.005,
            'drift_rate': 0.001,
            'temperature_coefficient': 0.0001,
            'valid_until': datetime.now() + timedelta(days=365)
        }
        
        return calibration_data
    
    def apply_calibration(self, df: pd.DataFrame, 
                         sensor_id: str,
                         calibration_data: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Apply calibration corrections to sensor data.
        
        Args:
            df: DataFrame with sensor data
            sensor_id: Sensor identifier
            calibration_data: Calibration parameters (if None, loaded automatically)
            
        Returns:
            DataFrame with calibrated data
        """
        if calibration_data is None:
            calibration_data = self.load_calibration_data(sensor_id)
        
        df_calibrated = df.copy()
        
        # Apply calibration corrections
        data_columns = [col for col in df.columns if col != 'timestamp']
        
        for col in data_columns:
            if col in df_calibrated.columns:
                # Apply linear calibration: y = factor * x + offset
                factor = calibration_data.get('calibration_factor', 1.0)
                offset = calibration_data.get('offset', 0.0)
                
                df_calibrated[col] = factor * df_calibrated[col] + offset
                
                # Apply temperature compensation if available
                if 'temperature_coefficient' in calibration_data and 'temperature' in df_calibrated.columns:
                    temp_coeff = calibration_data['temperature_coefficient']
                    ref_temp = 25.0  # Reference temperature
                    temp_compensation = temp_coeff * (df_calibrated['temperature'] - ref_temp)
                    df_calibrated[col] = df_calibrated[col] + temp_compensation
        
        return df_calibrated
    
    def get_sensor_metadata(self, sensor_id: str) -> Dict[str, Any]:
        """
        Get metadata for a specific sensor.
        
        Args:
            sensor_id: Unique sensor identifier
            
        Returns:
            Dictionary with sensor metadata
        """
        # This would typically load from a sensor database
        # For now, return default metadata
        
        metadata = {
            'sensor_id': sensor_id,
            'sensor_type': 'pyrometer',  # Default type
            'manufacturer': 'Unknown',
            'model': 'Unknown',
            'serial_number': 'Unknown',
            'installation_date': datetime.now(),
            'location': 'Unknown',
            'sampling_rate': 1000,
            'resolution': 0.1,
            'accuracy': 0.5,
            'range_min': 500,
            'range_max': 3000,
            'units': '°C',
            'status': 'active',
            'last_calibration': datetime.now(),
            'next_calibration': datetime.now() + timedelta(days=365)
        }
        
        return metadata
    
    def get_data_summary(self, df: pd.DataFrame, sensor_type: str) -> Dict[str, Any]:
        """
        Get summary statistics for sensor data.
        
        Args:
            df: DataFrame with sensor data
            sensor_type: Type of sensor
            
        Returns:
            Dictionary with summary statistics
        """
        summary = {
            'sensor_type': sensor_type,
            'total_records': len(df),
            'columns': list(df.columns),
            'data_types': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'time_range': {},
            'statistics': {}
        }
        
        # Time range information
        if 'timestamp' in df.columns:
            summary['time_range'] = {
                'start': df['timestamp'].min(),
                'end': df['timestamp'].max(),
                'duration': df['timestamp'].max() - df['timestamp'].min(),
                'sampling_rate': self._calculate_sampling_rate(df['timestamp'])
            }
        
        # Statistical summary for numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 0:
            summary['statistics'] = df[numeric_columns].describe().to_dict()
            
            # Add sensor-specific statistics
            for col in numeric_columns:
                if col != 'timestamp':
                    summary['statistics'][col].update({
                        'variance': df[col].var(),
                        'skewness': df[col].skew(),
                        'kurtosis': df[col].kurtosis(),
                        'outlier_count': self._count_outliers(df[col])
                    })
        
        return summary
    
    def _calculate_sampling_rate(self, timestamps: pd.Series) -> float:
        """Calculate sampling rate from timestamps."""
        if len(timestamps) < 2:
            return 0.0
        
        time_diffs = timestamps.diff().dropna()
        median_interval = time_diffs.median()
        
        if pd.isna(median_interval):
            return 0.0
        
        # Convert to Hz
        sampling_rate = 1.0 / median_interval.total_seconds()
        return sampling_rate
    
    def _count_outliers(self, data: pd.Series) -> int:
        """Count outliers using IQR method."""
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = ((data < lower_bound) | (data > upper_bound)).sum()
        return int(outliers)
