"""
Data Cleaner

This module implements utilities for cleaning and preprocessing data
in PBF-LB/M manufacturing processes.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Union, Tuple
from datetime import datetime, timedelta
import re

logger = logging.getLogger(__name__)


class DataCleaner:
    """
    Utility class for cleaning and preprocessing data.
    
    This class handles:
    - Missing value imputation
    - Data type conversion and validation
    - String cleaning and normalization
    - Date/time parsing and validation
    - Data deduplication
    - Data format standardization
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the data cleaner.
        
        Args:
            config: Configuration dictionary with cleaning settings
        """
        self.config = config or {}
        self.default_imputation_methods = {
            'numeric': 'median',
            'categorical': 'mode',
            'datetime': 'forward_fill',
            'text': 'empty_string'
        }
        
        # Data quality thresholds
        self.quality_thresholds = {
            'missing_threshold': 0.5,  # 50% missing values threshold
            'duplicate_threshold': 0.1,  # 10% duplicate threshold
            'outlier_threshold': 0.05,  # 5% outlier threshold
            'inconsistency_threshold': 0.1  # 10% inconsistency threshold
        }
        
        logger.info("Initialized DataCleaner")
    
    def clean_dataframe(self, df: pd.DataFrame, 
                       cleaning_config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Clean a DataFrame using specified configuration.
        
        Args:
            df: Input DataFrame
            cleaning_config: Configuration for cleaning operations
            
        Returns:
            Cleaned DataFrame
        """
        if df.empty:
            logger.warning("Input DataFrame is empty")
            return df
        
        cleaning_config = cleaning_config or self.config
        cleaned_df = df.copy()
        
        # Apply cleaning steps
        if cleaning_config.get('remove_duplicates', True):
            cleaned_df = self.remove_duplicates(cleaned_df)
        
        if cleaning_config.get('handle_missing_values', True):
            cleaned_df = self.handle_missing_values(cleaned_df, cleaning_config.get('missing_value_strategy'))
        
        if cleaning_config.get('convert_data_types', True):
            cleaned_df = self.convert_data_types(cleaned_df, cleaning_config.get('data_type_mapping'))
        
        if cleaning_config.get('clean_strings', True):
            cleaned_df = self.clean_string_columns(cleaned_df, cleaning_config.get('string_cleaning_config'))
        
        if cleaning_config.get('validate_data', True):
            cleaned_df = self.validate_data(cleaned_df, cleaning_config.get('validation_rules'))
        
        if cleaning_config.get('standardize_format', True):
            cleaned_df = self.standardize_format(cleaned_df, cleaning_config.get('format_standards'))
        
        logger.info(f"Data cleaning completed. Original shape: {df.shape}, Cleaned shape: {cleaned_df.shape}")
        
        return cleaned_df
    
    def remove_duplicates(self, df: pd.DataFrame, 
                         subset: Optional[List[str]] = None,
                         keep: str = 'first') -> pd.DataFrame:
        """
        Remove duplicate rows from DataFrame.
        
        Args:
            df: Input DataFrame
            subset: Columns to consider for duplicate detection
            keep: Which duplicate to keep ('first', 'last', False)
            
        Returns:
            DataFrame with duplicates removed
        """
        original_count = len(df)
        
        if subset is None:
            # Remove exact duplicates
            cleaned_df = df.drop_duplicates(keep=keep)
        else:
            # Remove duplicates based on subset of columns
            cleaned_df = df.drop_duplicates(subset=subset, keep=keep)
        
        removed_count = original_count - len(cleaned_df)
        
        if removed_count > 0:
            logger.info(f"Removed {removed_count} duplicate rows")
        
        return cleaned_df
    
    def handle_missing_values(self, df: pd.DataFrame, 
                            strategy: Optional[Dict[str, str]] = None) -> pd.DataFrame:
        """
        Handle missing values in DataFrame.
        
        Args:
            df: Input DataFrame
            strategy: Dictionary mapping column names to imputation strategies
            
        Returns:
            DataFrame with missing values handled
        """
        cleaned_df = df.copy()
        missing_summary = {}
        
        for column in cleaned_df.columns:
            missing_count = cleaned_df[column].isnull().sum()
            missing_percentage = missing_count / len(cleaned_df)
            
            if missing_count > 0:
                missing_summary[column] = {
                    'count': missing_count,
                    'percentage': missing_percentage
                }
                
                # Determine imputation strategy
                if strategy and column in strategy:
                    imputation_method = strategy[column]
                else:
                    imputation_method = self._get_default_imputation_method(cleaned_df[column])
                
                # Apply imputation
                if missing_percentage < self.quality_thresholds['missing_threshold']:
                    cleaned_df[column] = self._impute_missing_values(
                        cleaned_df[column], imputation_method
                    )
                else:
                    logger.warning(f"High missing value percentage ({missing_percentage:.2%}) in column {column}")
                    # For high missing values, consider dropping the column
                    if missing_percentage > 0.8:
                        logger.warning(f"Dropping column {column} due to high missing values")
                        cleaned_df = cleaned_df.drop(column, axis=1)
                    else:
                        # Still impute but log warning
                        cleaned_df[column] = self._impute_missing_values(
                            cleaned_df[column], imputation_method
                        )
        
        if missing_summary:
            logger.info(f"Missing value summary: {missing_summary}")
        
        return cleaned_df
    
    def _get_default_imputation_method(self, series: pd.Series) -> str:
        """Get default imputation method based on data type."""
        if pd.api.types.is_numeric_dtype(series):
            return self.default_imputation_methods['numeric']
        elif pd.api.types.is_datetime64_any_dtype(series):
            return self.default_imputation_methods['datetime']
        elif pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series):
            return self.default_imputation_methods['categorical']
        else:
            return self.default_imputation_methods['numeric']
    
    def _impute_missing_values(self, series: pd.Series, method: str) -> pd.Series:
        """Impute missing values using specified method."""
        if method == 'mean':
            return series.fillna(series.mean())
        elif method == 'median':
            return series.fillna(series.median())
        elif method == 'mode':
            mode_value = series.mode()
            return series.fillna(mode_value[0] if len(mode_value) > 0 else series.iloc[0])
        elif method == 'forward_fill':
            return series.fillna(method='ffill')
        elif method == 'backward_fill':
            return series.fillna(method='bfill')
        elif method == 'interpolate':
            return series.interpolate()
        elif method == 'drop':
            return series.dropna()
        elif method == 'empty_string':
            return series.fillna('')
        elif method == 'zero':
            return series.fillna(0)
        else:
            logger.warning(f"Unknown imputation method: {method}, using median")
            return series.fillna(series.median())
    
    def convert_data_types(self, df: pd.DataFrame, 
                          type_mapping: Optional[Dict[str, str]] = None) -> pd.DataFrame:
        """
        Convert data types of DataFrame columns.
        
        Args:
            df: Input DataFrame
            type_mapping: Dictionary mapping column names to target data types
            
        Returns:
            DataFrame with converted data types
        """
        cleaned_df = df.copy()
        
        if type_mapping is None:
            # Auto-detect and convert data types
            type_mapping = self._auto_detect_data_types(cleaned_df)
        
        for column, target_type in type_mapping.items():
            if column in cleaned_df.columns:
                try:
                    if target_type == 'numeric':
                        cleaned_df[column] = pd.to_numeric(cleaned_df[column], errors='coerce')
                    elif target_type == 'datetime':
                        cleaned_df[column] = pd.to_datetime(cleaned_df[column], errors='coerce')
                    elif target_type == 'category':
                        cleaned_df[column] = cleaned_df[column].astype('category')
                    elif target_type == 'string':
                        cleaned_df[column] = cleaned_df[column].astype('string')
                    elif target_type == 'boolean':
                        cleaned_df[column] = cleaned_df[column].astype('boolean')
                    else:
                        cleaned_df[column] = cleaned_df[column].astype(target_type)
                    
                    logger.info(f"Converted column {column} to {target_type}")
                    
                except Exception as e:
                    logger.warning(f"Failed to convert column {column} to {target_type}: {e}")
        
        return cleaned_df
    
    def _auto_detect_data_types(self, df: pd.DataFrame) -> Dict[str, str]:
        """Auto-detect appropriate data types for columns."""
        type_mapping = {}
        
        for column in df.columns:
            if df[column].dtype == 'object':
                # Try to detect the best type for object columns
                sample_values = df[column].dropna().head(100)
                
                if len(sample_values) == 0:
                    continue
                
                # Check if it's numeric
                if self._is_numeric(sample_values):
                    type_mapping[column] = 'numeric'
                # Check if it's datetime
                elif self._is_datetime(sample_values):
                    type_mapping[column] = 'datetime'
                # Check if it's boolean
                elif self._is_boolean(sample_values):
                    type_mapping[column] = 'boolean'
                # Check if it's categorical
                elif self._is_categorical(sample_values):
                    type_mapping[column] = 'category'
                else:
                    type_mapping[column] = 'string'
        
        return type_mapping
    
    def _is_numeric(self, series: pd.Series) -> bool:
        """Check if series contains numeric data."""
        try:
            pd.to_numeric(series, errors='raise')
            return True
        except (ValueError, TypeError):
            return False
    
    def _is_datetime(self, series: pd.Series) -> bool:
        """Check if series contains datetime data."""
        try:
            pd.to_datetime(series, errors='raise')
            return True
        except (ValueError, TypeError):
            return False
    
    def _is_boolean(self, series: pd.Series) -> bool:
        """Check if series contains boolean data."""
        unique_values = set(series.str.lower().unique())
        boolean_values = {'true', 'false', '1', '0', 'yes', 'no', 'y', 'n'}
        return unique_values.issubset(boolean_values)
    
    def _is_categorical(self, series: pd.Series) -> bool:
        """Check if series is categorical (low cardinality)."""
        unique_count = series.nunique()
        total_count = len(series)
        return unique_count / total_count < 0.1 and unique_count < 50
    
    def clean_string_columns(self, df: pd.DataFrame, 
                           cleaning_config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Clean string columns in DataFrame.
        
        Args:
            df: Input DataFrame
            cleaning_config: Configuration for string cleaning
            
        Returns:
            DataFrame with cleaned string columns
        """
        cleaned_df = df.copy()
        cleaning_config = cleaning_config or {}
        
        string_columns = cleaned_df.select_dtypes(include=['object', 'string']).columns
        
        for column in string_columns:
            if column in cleaned_df.columns:
                # Remove leading/trailing whitespace
                if cleaning_config.get('strip_whitespace', True):
                    cleaned_df[column] = cleaned_df[column].astype(str).str.strip()
                
                # Convert to lowercase
                if cleaning_config.get('lowercase', False):
                    cleaned_df[column] = cleaned_df[column].str.lower()
                
                # Remove special characters
                if cleaning_config.get('remove_special_chars', False):
                    cleaned_df[column] = cleaned_df[column].str.replace(r'[^a-zA-Z0-9\s]', '', regex=True)
                
                # Replace multiple spaces with single space
                if cleaning_config.get('normalize_spaces', True):
                    cleaned_df[column] = cleaned_df[column].str.replace(r'\s+', ' ', regex=True)
                
                # Handle null values
                if cleaning_config.get('handle_null_strings', True):
                    null_strings = ['null', 'none', 'nan', 'n/a', 'na', '']
                    cleaned_df[column] = cleaned_df[column].replace(null_strings, np.nan)
                
                # Remove duplicates in string values
                if cleaning_config.get('remove_duplicate_strings', False):
                    cleaned_df[column] = cleaned_df[column].drop_duplicates()
        
        return cleaned_df
    
    def validate_data(self, df: pd.DataFrame, 
                     validation_rules: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Validate data against specified rules.
        
        Args:
            df: Input DataFrame
            validation_rules: Dictionary of validation rules
            
        Returns:
            DataFrame with validation applied
        """
        cleaned_df = df.copy()
        validation_rules = validation_rules or {}
        
        for column, rules in validation_rules.items():
            if column not in cleaned_df.columns:
                continue
            
            # Range validation
            if 'min_value' in rules or 'max_value' in rules:
                min_val = rules.get('min_value')
                max_val = rules.get('max_value')
                
                if min_val is not None:
                    invalid_mask = cleaned_df[column] < min_val
                    if invalid_mask.any():
                        logger.warning(f"Found {invalid_mask.sum()} values below minimum in column {column}")
                        if rules.get('clip_to_range', True):
                            cleaned_df.loc[invalid_mask, column] = min_val
                
                if max_val is not None:
                    invalid_mask = cleaned_df[column] > max_val
                    if invalid_mask.any():
                        logger.warning(f"Found {invalid_mask.sum()} values above maximum in column {column}")
                        if rules.get('clip_to_range', True):
                            cleaned_df.loc[invalid_mask, column] = max_val
            
            # Pattern validation
            if 'pattern' in rules:
                pattern = rules['pattern']
                invalid_mask = ~cleaned_df[column].astype(str).str.match(pattern, na=False)
                if invalid_mask.any():
                    logger.warning(f"Found {invalid_mask.sum()} values not matching pattern in column {column}")
                    if rules.get('remove_invalid', False):
                        cleaned_df = cleaned_df[~invalid_mask]
            
            # Value validation
            if 'allowed_values' in rules:
                allowed_values = rules['allowed_values']
                invalid_mask = ~cleaned_df[column].isin(allowed_values)
                if invalid_mask.any():
                    logger.warning(f"Found {invalid_mask.sum()} invalid values in column {column}")
                    if rules.get('remove_invalid', False):
                        cleaned_df = cleaned_df[~invalid_mask]
                    elif rules.get('replace_invalid', False):
                        cleaned_df.loc[invalid_mask, column] = rules.get('default_value', np.nan)
        
        return cleaned_df
    
    def standardize_format(self, df: pd.DataFrame, 
                          format_standards: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Standardize data format according to specified standards.
        
        Args:
            df: Input DataFrame
            format_standards: Dictionary of format standards
            
        Returns:
            DataFrame with standardized format
        """
        cleaned_df = df.copy()
        format_standards = format_standards or {}
        
        for column, standards in format_standards.items():
            if column not in cleaned_df.columns:
                continue
            
            # Date format standardization
            if 'date_format' in standards:
                try:
                    cleaned_df[column] = pd.to_datetime(cleaned_df[column], format=standards['date_format'])
                except Exception as e:
                    logger.warning(f"Failed to standardize date format for column {column}: {e}")
            
            # Number format standardization
            if 'number_format' in standards:
                try:
                    if standards['number_format'] == 'integer':
                        cleaned_df[column] = pd.to_numeric(cleaned_df[column], errors='coerce').astype('Int64')
                    elif standards['number_format'] == 'float':
                        cleaned_df[column] = pd.to_numeric(cleaned_df[column], errors='coerce')
                except Exception as e:
                    logger.warning(f"Failed to standardize number format for column {column}: {e}")
            
            # String format standardization
            if 'string_format' in standards:
                if standards['string_format'] == 'uppercase':
                    cleaned_df[column] = cleaned_df[column].str.upper()
                elif standards['string_format'] == 'lowercase':
                    cleaned_df[column] = cleaned_df[column].str.lower()
                elif standards['string_format'] == 'title':
                    cleaned_df[column] = cleaned_df[column].str.title()
        
        return cleaned_df
    
    def get_data_quality_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate a data quality report for the DataFrame.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with data quality metrics
        """
        report = {
            'overall_quality_score': 0.0,
            'dimensions': {
                'rows': len(df),
                'columns': len(df.columns)
            },
            'missing_values': {},
            'data_types': {},
            'duplicates': {},
            'outliers': {},
            'inconsistencies': {},
            'recommendations': []
        }
        
        # Missing values analysis
        missing_values = df.isnull().sum()
        missing_percentage = (missing_values / len(df)) * 100
        
        report['missing_values'] = {
            'total_missing': missing_values.sum(),
            'missing_percentage': (missing_values.sum() / (len(df) * len(df.columns))) * 100,
            'columns_with_missing': missing_percentage[missing_percentage > 0].to_dict()
        }
        
        # Data types analysis
        report['data_types'] = df.dtypes.to_dict()
        
        # Duplicates analysis
        duplicate_count = df.duplicated().sum()
        report['duplicates'] = {
            'count': duplicate_count,
            'percentage': (duplicate_count / len(df)) * 100
        }
        
        # Outliers analysis (for numeric columns)
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        outlier_summary = {}
        
        for column in numeric_columns:
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = ((df[column] < lower_bound) | (df[column] > upper_bound)).sum()
            outlier_percentage = (outliers / len(df)) * 100
            
            outlier_summary[column] = {
                'count': outliers,
                'percentage': outlier_percentage
            }
        
        report['outliers'] = outlier_summary
        
        # Calculate overall quality score
        quality_factors = []
        
        # Missing values factor
        missing_factor = 1 - (report['missing_values']['missing_percentage'] / 100)
        quality_factors.append(missing_factor)
        
        # Duplicates factor
        duplicate_factor = 1 - (report['duplicates']['percentage'] / 100)
        quality_factors.append(duplicate_factor)
        
        # Outliers factor
        if outlier_summary:
            avg_outlier_percentage = np.mean([col['percentage'] for col in outlier_summary.values()])
            outlier_factor = 1 - (avg_outlier_percentage / 100)
            quality_factors.append(outlier_factor)
        
        # Calculate overall score
        if quality_factors:
            report['overall_quality_score'] = np.mean(quality_factors)
        
        # Generate recommendations
        if report['missing_values']['missing_percentage'] > 20:
            report['recommendations'].append("High missing value percentage detected. Consider imputation or data collection improvement.")
        
        if report['duplicates']['percentage'] > 10:
            report['recommendations'].append("High duplicate percentage detected. Consider removing duplicates.")
        
        if outlier_summary:
            high_outlier_columns = [col for col, stats in outlier_summary.items() if stats['percentage'] > 10]
            if high_outlier_columns:
                report['recommendations'].append(f"High outlier percentage detected in columns: {high_outlier_columns}")
        
        if report['overall_quality_score'] < 0.7:
            report['recommendations'].append("Overall data quality is below acceptable threshold. Consider data cleaning.")
        
        return report
    
    def clean_time_series_data(self, df: pd.DataFrame, 
                              time_column: str = 'timestamp',
                              frequency: Optional[str] = None) -> pd.DataFrame:
        """
        Clean time series data.
        
        Args:
            df: Input DataFrame
            time_column: Name of the time column
            frequency: Expected frequency of the time series
            
        Returns:
            Cleaned time series DataFrame
        """
        if time_column not in df.columns:
            logger.warning(f"Time column {time_column} not found in DataFrame")
            return df
        
        cleaned_df = df.copy()
        
        # Convert time column to datetime
        cleaned_df[time_column] = pd.to_datetime(cleaned_df[time_column], errors='coerce')
        
        # Remove rows with invalid timestamps
        invalid_timestamps = cleaned_df[time_column].isnull()
        if invalid_timestamps.any():
            logger.warning(f"Removing {invalid_timestamps.sum()} rows with invalid timestamps")
            cleaned_df = cleaned_df[~invalid_timestamps]
        
        # Sort by time
        cleaned_df = cleaned_df.sort_values(time_column).reset_index(drop=True)
        
        # Remove duplicate timestamps
        duplicate_timestamps = cleaned_df[time_column].duplicated()
        if duplicate_timestamps.any():
            logger.warning(f"Removing {duplicate_timestamps.sum()} rows with duplicate timestamps")
            cleaned_df = cleaned_df[~duplicate_timestamps]
        
        # Handle missing timestamps if frequency is specified
        if frequency is not None:
            cleaned_df = self._handle_missing_timestamps(cleaned_df, time_column, frequency)
        
        return cleaned_df
    
    def _handle_missing_timestamps(self, df: pd.DataFrame, 
                                  time_column: str, 
                                  frequency: str) -> pd.DataFrame:
        """Handle missing timestamps in time series data."""
        # Create complete time range
        start_time = df[time_column].min()
        end_time = df[time_column].max()
        complete_time_range = pd.date_range(start=start_time, end=end_time, freq=frequency)
        
        # Set time column as index
        df_indexed = df.set_index(time_column)
        
        # Reindex to complete time range
        df_complete = df_indexed.reindex(complete_time_range)
        
        # Reset index
        df_complete = df_complete.reset_index()
        df_complete = df_complete.rename(columns={'index': time_column})
        
        return df_complete
    
    def clean_categorical_data(self, df: pd.DataFrame, 
                              categorical_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Clean categorical data.
        
        Args:
            df: Input DataFrame
            categorical_columns: List of categorical column names
            
        Returns:
            DataFrame with cleaned categorical data
        """
        cleaned_df = df.copy()
        
        if categorical_columns is None:
            categorical_columns = cleaned_df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        for column in categorical_columns:
            if column in cleaned_df.columns:
                # Convert to string and strip whitespace
                cleaned_df[column] = cleaned_df[column].astype(str).str.strip()
                
                # Handle case sensitivity
                cleaned_df[column] = cleaned_df[column].str.lower()
                
                # Handle common variations
                cleaned_df[column] = cleaned_df[column].replace({
                    'yes': 'y',
                    'no': 'n',
                    'true': 't',
                    'false': 'f',
                    'male': 'm',
                    'female': 'f'
                })
                
                # Remove empty strings and replace with NaN
                cleaned_df[column] = cleaned_df[column].replace('', np.nan)
                
                # Convert back to category
                cleaned_df[column] = cleaned_df[column].astype('category')
        
        return cleaned_df
