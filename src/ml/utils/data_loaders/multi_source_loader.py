"""
Multi-Source Data Loader

This module implements utilities for loading and combining data from multiple
sources in PBF-LB/M manufacturing processes.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Union, Tuple
from pathlib import Path
import json
import yaml
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from .process_data_loader import ProcessDataLoader
from .sensor_data_loader import SensorDataLoader
from .image_data_loader import ImageDataLoader

logger = logging.getLogger(__name__)


class MultiSourceLoader:
    """
    Utility class for loading and combining data from multiple sources.
    
    This class handles:
    - Loading data from multiple sources simultaneously
    - Data synchronization and alignment
    - Data fusion and integration
    - Time-based data correlation
    - Cross-source data validation
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the multi-source data loader.
        
        Args:
            config: Configuration dictionary with data source settings
        """
        self.config = config or {}
        self.process_loader = ProcessDataLoader(config.get('process', {}))
        self.sensor_loader = SensorDataLoader(config.get('sensor', {}))
        self.image_loader = ImageDataLoader(config.get('image', {}))
        
        # Data source configurations
        self.data_sources = {
            'process': {
                'loader': self.process_loader,
                'time_column': 'timestamp',
                'priority': 1
            },
            'sensor': {
                'loader': self.sensor_loader,
                'time_column': 'timestamp',
                'priority': 2
            },
            'image': {
                'loader': self.image_loader,
                'time_column': 'timestamp',
                'priority': 3
            }
        }
        
        # Thread lock for thread-safe operations
        self._lock = threading.Lock()
        
        logger.info("Initialized MultiSourceLoader")
    
    def load_multi_source_data(self, 
                              data_sources: Dict[str, Dict[str, Any]],
                              time_range: Optional[Tuple[datetime, datetime]] = None,
                              synchronization_method: str = 'interpolation') -> Dict[str, pd.DataFrame]:
        """
        Load data from multiple sources.
        
        Args:
            data_sources: Dictionary mapping source names to their configurations
            time_range: Time range for data loading (start_time, end_time)
            synchronization_method: Method for data synchronization ('interpolation', 'nearest', 'forward_fill')
            
        Returns:
            Dictionary mapping source names to DataFrames
        """
        loaded_data = {}
        
        # Load data from each source
        for source_name, source_config in data_sources.items():
            try:
                if source_name not in self.data_sources:
                    logger.warning(f"Unknown data source: {source_name}")
                    continue
                
                source_loader = self.data_sources[source_name]['loader']
                
                # Load data based on source type
                if source_name == 'process':
                    data = self._load_process_data(source_loader, source_config, time_range)
                elif source_name == 'sensor':
                    data = self._load_sensor_data(source_loader, source_config, time_range)
                elif source_name == 'image':
                    data = self._load_image_data(source_loader, source_config, time_range)
                else:
                    logger.warning(f"Unsupported source type: {source_name}")
                    continue
                
                if data is not None and not data.empty:
                    loaded_data[source_name] = data
                    logger.info(f"Loaded {len(data)} records from {source_name}")
                else:
                    logger.warning(f"No data loaded from {source_name}")
                    
            except Exception as e:
                logger.error(f"Failed to load data from {source_name}: {e}")
                continue
        
        # Synchronize data if multiple sources
        if len(loaded_data) > 1:
            synchronized_data = self._synchronize_data(loaded_data, synchronization_method)
            return synchronized_data
        
        return loaded_data
    
    def _load_process_data(self, loader: ProcessDataLoader, 
                          config: Dict[str, Any], 
                          time_range: Optional[Tuple[datetime, datetime]]) -> pd.DataFrame:
        """Load process data."""
        if 'file_path' in config:
            return loader.load_from_file(config['file_path'])
        elif 'connection_string' in config:
            return loader.load_from_database(
                config['connection_string'],
                config.get('query', '{}'),
                config.get('database_type', 'mongodb')
            )
        elif 'batch_ids' in config:
            return loader.load_batch_data(config['batch_ids'])
        elif 'material_type' in config:
            return loader.load_material_specific_data(config['material_type'])
        elif time_range is not None:
            return loader.load_time_series_data(
                time_range[0], time_range[1],
                config.get('parameters'),
                config.get('aggregation', 'mean'),
                config.get('interval', '1min')
            )
        else:
            logger.warning("No valid configuration found for process data loading")
            return pd.DataFrame()
    
    def _load_sensor_data(self, loader: SensorDataLoader, 
                         config: Dict[str, Any], 
                         time_range: Optional[Tuple[datetime, datetime]]) -> pd.DataFrame:
        """Load sensor data."""
        if 'file_path' in config:
            return loader.load_from_file(
                config['file_path'],
                config.get('sensor_type', 'pyrometer')
            )
        elif time_range is not None and 'sensor_types' in config:
            sensor_data = loader.load_time_series_data(
                time_range[0], time_range[1],
                config['sensor_types'],
                config.get('sampling_rate')
            )
            # Combine sensor data
            if sensor_data:
                return self._combine_sensor_data(sensor_data)
        else:
            logger.warning("No valid configuration found for sensor data loading")
            return pd.DataFrame()
    
    def _load_image_data(self, loader: ImageDataLoader, 
                        config: Dict[str, Any], 
                        time_range: Optional[Tuple[datetime, datetime]]) -> pd.DataFrame:
        """Load image data."""
        if 'directory_path' in config:
            images, paths = loader.load_from_directory(
                config['directory_path'],
                config.get('image_type', 'defect_image'),
                config.get('file_pattern', '*'),
                config.get('recursive', True),
                config.get('target_size'),
                config.get('normalize', True)
            )
            
            # Convert to DataFrame
            if images:
                return self._images_to_dataframe(images, paths, config.get('image_type', 'defect_image'))
        elif 'dataset_path' in config:
            dataset_info = loader.load_image_dataset(
                config['dataset_path'],
                config.get('image_type', 'defect_image'),
                config.get('target_size'),
                config.get('normalize', True)
            )
            
            # Convert to DataFrame
            if dataset_info['images']:
                return self._dataset_to_dataframe(dataset_info)
        else:
            logger.warning("No valid configuration found for image data loading")
            return pd.DataFrame()
    
    def _combine_sensor_data(self, sensor_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Combine data from multiple sensors."""
        if not sensor_data:
            return pd.DataFrame()
        
        # Use the first sensor as reference
        reference_sensor = list(sensor_data.keys())[0]
        combined_df = sensor_data[reference_sensor].copy()
        
        # Add data from other sensors
        for sensor_type, df in sensor_data.items():
            if sensor_type == reference_sensor:
                continue
            
            # Merge on timestamp
            combined_df = pd.merge(combined_df, df, on='timestamp', how='outer', suffixes=('', f'_{sensor_type}'))
        
        # Sort by timestamp
        combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)
        
        return combined_df
    
    def _images_to_dataframe(self, images: List[np.ndarray], 
                           paths: List[Path], 
                           image_type: str) -> pd.DataFrame:
        """Convert images to DataFrame format."""
        data = []
        
        for i, (image, path) in enumerate(zip(images, paths)):
            # Flatten image for storage
            image_flat = image.flatten()
            
            row_data = {
                'image_id': i,
                'file_path': str(path),
                'image_type': image_type,
                'timestamp': datetime.now(),  # Default timestamp
                'image_shape': image.shape,
                'image_data': image_flat
            }
            
            # Add image statistics
            row_data.update({
                'min_value': float(image.min()),
                'max_value': float(image.max()),
                'mean_value': float(image.mean()),
                'std_value': float(image.std())
            })
            
            data.append(row_data)
        
        return pd.DataFrame(data)
    
    def _dataset_to_dataframe(self, dataset_info: Dict[str, Any]) -> pd.DataFrame:
        """Convert dataset info to DataFrame format."""
        data = []
        
        for i, (image, label, metadata) in enumerate(zip(
            dataset_info['images'], 
            dataset_info['labels'], 
            dataset_info['metadata']
        )):
            # Flatten image for storage
            image_flat = image.flatten()
            
            row_data = {
                'image_id': i,
                'label': label,
                'image_type': dataset_info['image_type'],
                'timestamp': datetime.now(),  # Default timestamp
                'image_shape': image.shape,
                'image_data': image_flat
            }
            
            # Add image statistics
            row_data.update({
                'min_value': float(image.min()),
                'max_value': float(image.max()),
                'mean_value': float(image.mean()),
                'std_value': float(image.std())
            })
            
            # Add metadata
            row_data.update(metadata)
            
            data.append(row_data)
        
        return pd.DataFrame(data)
    
    def _synchronize_data(self, data_dict: Dict[str, pd.DataFrame], 
                         method: str = 'interpolation') -> Dict[str, pd.DataFrame]:
        """
        Synchronize data from multiple sources to a common time base.
        
        Args:
            data_dict: Dictionary of DataFrames from different sources
            method: Synchronization method
            
        Returns:
            Dictionary of synchronized DataFrames
        """
        if len(data_dict) < 2:
            return data_dict
        
        # Find common time range
        time_columns = {}
        for source_name, df in data_dict.items():
            time_col = self.data_sources[source_name]['time_column']
            if time_col in df.columns:
                time_columns[source_name] = time_col
        
        if not time_columns:
            logger.warning("No timestamp columns found for synchronization")
            return data_dict
        
        # Get time range
        all_times = []
        for source_name, time_col in time_columns.items():
            times = pd.to_datetime(data_dict[source_name][time_col])
            all_times.extend(times.tolist())
        
        if not all_times:
            logger.warning("No timestamps found for synchronization")
            return data_dict
        
        # Create common time index
        min_time = min(all_times)
        max_time = max(all_times)
        
        # Use the highest frequency source as reference
        reference_source = self._find_reference_source(data_dict, time_columns)
        reference_df = data_dict[reference_source]
        reference_time_col = time_columns[reference_source]
        
        # Create common time index
        common_times = pd.to_datetime(reference_df[reference_time_col])
        
        synchronized_data = {reference_source: reference_df}
        
        # Synchronize other sources
        for source_name, df in data_dict.items():
            if source_name == reference_source:
                continue
            
            time_col = time_columns.get(source_name)
            if time_col is None:
                logger.warning(f"No timestamp column found for {source_name}")
                synchronized_data[source_name] = df
                continue
            
            # Set timestamp as index
            df_sync = df.set_index(time_col)
            
            # Reindex to common time base
            if method == 'interpolation':
                df_sync = df_sync.reindex(common_times, method='linear')
            elif method == 'nearest':
                df_sync = df_sync.reindex(common_times, method='nearest')
            elif method == 'forward_fill':
                df_sync = df_sync.reindex(common_times, method='ffill')
            else:
                logger.warning(f"Unknown synchronization method: {method}")
                df_sync = df_sync.reindex(common_times, method='linear')
            
            # Reset index
            df_sync = df_sync.reset_index()
            df_sync = df_sync.rename(columns={'index': time_col})
            
            synchronized_data[source_name] = df_sync
        
        return synchronized_data
    
    def _find_reference_source(self, data_dict: Dict[str, pd.DataFrame], 
                              time_columns: Dict[str, str]) -> str:
        """Find the source with the highest sampling rate to use as reference."""
        max_samples = 0
        reference_source = None
        
        for source_name, time_col in time_columns.items():
            df = data_dict[source_name]
            if time_col in df.columns:
                num_samples = len(df)
                if num_samples > max_samples:
                    max_samples = num_samples
                    reference_source = source_name
        
        return reference_source or list(data_dict.keys())[0]
    
    def load_parallel_data(self, 
                          data_sources: Dict[str, Dict[str, Any]],
                          max_workers: int = 4) -> Dict[str, pd.DataFrame]:
        """
        Load data from multiple sources in parallel.
        
        Args:
            data_sources: Dictionary mapping source names to their configurations
            max_workers: Maximum number of worker threads
            
        Returns:
            Dictionary mapping source names to DataFrames
        """
        loaded_data = {}
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all loading tasks
            future_to_source = {}
            
            for source_name, source_config in data_sources.items():
                future = executor.submit(self._load_single_source, source_name, source_config)
                future_to_source[future] = source_name
            
            # Collect results
            for future in as_completed(future_to_source):
                source_name = future_to_source[future]
                try:
                    data = future.result()
                    if data is not None and not data.empty:
                        loaded_data[source_name] = data
                        logger.info(f"Loaded {len(data)} records from {source_name}")
                    else:
                        logger.warning(f"No data loaded from {source_name}")
                except Exception as e:
                    logger.error(f"Failed to load data from {source_name}: {e}")
        
        return loaded_data
    
    def _load_single_source(self, source_name: str, source_config: Dict[str, Any]) -> pd.DataFrame:
        """Load data from a single source (for parallel loading)."""
        try:
            if source_name not in self.data_sources:
                logger.warning(f"Unknown data source: {source_name}")
                return pd.DataFrame()
            
            source_loader = self.data_sources[source_name]['loader']
            
            # Load data based on source type
            if source_name == 'process':
                return self._load_process_data(source_loader, source_config, None)
            elif source_name == 'sensor':
                return self._load_sensor_data(source_loader, source_config, None)
            elif source_name == 'image':
                return self._load_image_data(source_loader, source_config, None)
            else:
                logger.warning(f"Unsupported source type: {source_name}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error loading data from {source_name}: {e}")
            return pd.DataFrame()
    
    def fuse_data(self, data_dict: Dict[str, pd.DataFrame], 
                  fusion_method: str = 'concatenate') -> pd.DataFrame:
        """
        Fuse data from multiple sources into a single DataFrame.
        
        Args:
            data_dict: Dictionary of DataFrames from different sources
            fusion_method: Method for data fusion ('concatenate', 'merge', 'join')
            
        Returns:
            Fused DataFrame
        """
        if not data_dict:
            return pd.DataFrame()
        
        if len(data_dict) == 1:
            return list(data_dict.values())[0]
        
        if fusion_method == 'concatenate':
            return self._concatenate_data(data_dict)
        elif fusion_method == 'merge':
            return self._merge_data(data_dict)
        elif fusion_method == 'join':
            return self._join_data(data_dict)
        else:
            logger.warning(f"Unknown fusion method: {fusion_method}")
            return self._concatenate_data(data_dict)
    
    def _concatenate_data(self, data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Concatenate data from multiple sources."""
        # Add source column to each DataFrame
        dataframes = []
        for source_name, df in data_dict.items():
            df_copy = df.copy()
            df_copy['data_source'] = source_name
            dataframes.append(df_copy)
        
        # Concatenate all DataFrames
        fused_df = pd.concat(dataframes, ignore_index=True, sort=False)
        
        return fused_df
    
    def _merge_data(self, data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Merge data from multiple sources on common columns."""
        # Find common columns
        all_columns = set()
        for df in data_dict.values():
            all_columns.update(df.columns)
        
        # Find timestamp columns
        timestamp_columns = []
        for source_name, df in data_dict.items():
            time_col = self.data_sources[source_name]['time_column']
            if time_col in df.columns:
                timestamp_columns.append(time_col)
        
        if not timestamp_columns:
            logger.warning("No timestamp columns found for merging")
            return self._concatenate_data(data_dict)
        
        # Use the first timestamp column as reference
        reference_time_col = timestamp_columns[0]
        merged_df = None
        
        for source_name, df in data_dict.items():
            time_col = self.data_sources[source_name]['time_column']
            
            if time_col in df.columns:
                # Convert timestamp to datetime
                df_copy = df.copy()
                df_copy[time_col] = pd.to_datetime(df_copy[time_col])
                
                if merged_df is None:
                    merged_df = df_copy
                else:
                    # Merge on timestamp
                    merged_df = pd.merge(merged_df, df_copy, left_on=reference_time_col, right_on=time_col, how='outer', suffixes=('', f'_{source_name}'))
        
        return merged_df if merged_df is not None else pd.DataFrame()
    
    def _join_data(self, data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Join data from multiple sources."""
        # Find common index columns
        common_columns = set(list(data_dict.values())[0].columns)
        for df in data_dict.values():
            common_columns = common_columns.intersection(set(df.columns))
        
        if not common_columns:
            logger.warning("No common columns found for joining")
            return self._concatenate_data(data_dict)
        
        # Use the first common column as index
        index_col = list(common_columns)[0]
        joined_df = None
        
        for source_name, df in data_dict.items():
            df_copy = df.set_index(index_col)
            
            if joined_df is None:
                joined_df = df_copy
            else:
                joined_df = joined_df.join(df_copy, how='outer', rsuffix=f'_{source_name}')
        
        return joined_df.reset_index() if joined_df is not None else pd.DataFrame()
    
    def validate_cross_source_data(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Validate data consistency across multiple sources.
        
        Args:
            data_dict: Dictionary of DataFrames from different sources
            
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'overall_status': 'valid',
            'source_validation': {},
            'cross_source_validation': {},
            'issues': []
        }
        
        # Validate each source individually
        for source_name, df in data_dict.items():
            source_validation = self._validate_single_source(df, source_name)
            validation_results['source_validation'][source_name] = source_validation
            
            if source_validation['status'] != 'valid':
                validation_results['overall_status'] = 'invalid'
                validation_results['issues'].extend(source_validation['issues'])
        
        # Cross-source validation
        if len(data_dict) > 1:
            cross_validation = self._validate_cross_source_consistency(data_dict)
            validation_results['cross_source_validation'] = cross_validation
            
            if cross_validation['status'] != 'valid':
                validation_results['overall_status'] = 'invalid'
                validation_results['issues'].extend(cross_validation['issues'])
        
        return validation_results
    
    def _validate_single_source(self, df: pd.DataFrame, source_name: str) -> Dict[str, Any]:
        """Validate data from a single source."""
        validation = {
            'status': 'valid',
            'issues': [],
            'statistics': {}
        }
        
        # Check for empty DataFrame
        if df.empty:
            validation['status'] = 'invalid'
            validation['issues'].append(f"Empty DataFrame for {source_name}")
            return validation
        
        # Check for missing values
        missing_values = df.isnull().sum()
        if missing_values.sum() > 0:
            validation['issues'].append(f"Missing values found in {source_name}: {missing_values.to_dict()}")
        
        # Check for duplicate rows
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            validation['issues'].append(f"{duplicates} duplicate rows found in {source_name}")
        
        # Check timestamp consistency
        time_col = self.data_sources[source_name]['time_column']
        if time_col in df.columns:
            timestamps = pd.to_datetime(df[time_col])
            if not timestamps.is_monotonic_increasing:
                validation['issues'].append(f"Non-monotonic timestamps in {source_name}")
        
        # Add statistics
        validation['statistics'] = {
            'total_records': len(df),
            'columns': list(df.columns),
            'missing_values': missing_values.to_dict(),
            'duplicates': int(duplicates)
        }
        
        if validation['issues']:
            validation['status'] = 'warning'
        
        return validation
    
    def _validate_cross_source_consistency(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Validate consistency across multiple sources."""
        validation = {
            'status': 'valid',
            'issues': [],
            'consistency_metrics': {}
        }
        
        # Check time range consistency
        time_ranges = {}
        for source_name, df in data_dict.items():
            time_col = self.data_sources[source_name]['time_column']
            if time_col in df.columns:
                timestamps = pd.to_datetime(df[time_col])
                time_ranges[source_name] = {
                    'start': timestamps.min(),
                    'end': timestamps.max(),
                    'count': len(timestamps)
                }
        
        if len(time_ranges) > 1:
            # Check for overlapping time ranges
            time_overlaps = self._check_time_overlaps(time_ranges)
            validation['consistency_metrics']['time_overlaps'] = time_overlaps
            
            if not time_overlaps['has_overlap']:
                validation['issues'].append("No time overlap between data sources")
                validation['status'] = 'warning'
        
        # Check data volume consistency
        record_counts = {name: len(df) for name, df in data_dict.items()}
        validation['consistency_metrics']['record_counts'] = record_counts
        
        # Check for significant volume differences
        if len(record_counts) > 1:
            max_count = max(record_counts.values())
            min_count = min(record_counts.values())
            volume_ratio = max_count / min_count if min_count > 0 else float('inf')
            
            if volume_ratio > 10:  # More than 10x difference
                validation['issues'].append(f"Significant volume difference between sources (ratio: {volume_ratio:.2f})")
                validation['status'] = 'warning'
        
        return validation
    
    def _check_time_overlaps(self, time_ranges: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Check for time overlaps between sources."""
        sources = list(time_ranges.keys())
        overlaps = []
        
        for i in range(len(sources)):
            for j in range(i + 1, len(sources)):
                source1, source2 = sources[i], sources[j]
                range1 = time_ranges[source1]
                range2 = time_ranges[source2]
                
                # Check for overlap
                overlap_start = max(range1['start'], range2['start'])
                overlap_end = min(range1['end'], range2['end'])
                
                if overlap_start <= overlap_end:
                    overlap_duration = overlap_end - overlap_start
                    overlaps.append({
                        'source1': source1,
                        'source2': source2,
                        'overlap_start': overlap_start,
                        'overlap_end': overlap_end,
                        'overlap_duration': overlap_duration
                    })
        
        return {
            'has_overlap': len(overlaps) > 0,
            'overlaps': overlaps
        }
    
    def get_data_summary(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Get summary statistics for multi-source data.
        
        Args:
            data_dict: Dictionary of DataFrames from different sources
            
        Returns:
            Dictionary with summary statistics
        """
        summary = {
            'total_sources': len(data_dict),
            'source_summaries': {},
            'combined_statistics': {},
            'data_quality': {}
        }
        
        # Individual source summaries
        for source_name, df in data_dict.items():
            if source_name == 'process':
                source_summary = self.process_loader.get_data_summary(df)
            elif source_name == 'sensor':
                source_summary = self.sensor_loader.get_data_summary(df, 'multi_sensor')
            else:
                # Generic summary
                source_summary = {
                    'total_records': len(df),
                    'columns': list(df.columns),
                    'data_types': df.dtypes.to_dict(),
                    'missing_values': df.isnull().sum().to_dict()
                }
            
            summary['source_summaries'][source_name] = source_summary
        
        # Combined statistics
        total_records = sum(len(df) for df in data_dict.values())
        all_columns = set()
        for df in data_dict.values():
            all_columns.update(df.columns)
        
        summary['combined_statistics'] = {
            'total_records': total_records,
            'unique_columns': len(all_columns),
            'common_columns': len(set.intersection(*[set(df.columns) for df in data_dict.values()])) if data_dict else 0
        }
        
        # Data quality assessment
        validation_results = self.validate_cross_source_data(data_dict)
        summary['data_quality'] = {
            'overall_status': validation_results['overall_status'],
            'issue_count': len(validation_results['issues']),
            'issues': validation_results['issues']
        }
        
        return summary
