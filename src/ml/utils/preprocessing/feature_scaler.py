"""
Feature Scaler

This module implements utilities for feature scaling and normalization
in PBF-LB/M manufacturing processes.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Union, Tuple
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, 
    Normalizer, PowerTransformer, QuantileTransformer
)
from sklearn.base import BaseEstimator, TransformerMixin
import joblib
from pathlib import Path

logger = logging.getLogger(__name__)


class FeatureScaler:
    """
    Utility class for feature scaling and normalization.
    
    This class handles:
    - Standard scaling (z-score normalization)
    - Min-max scaling
    - Robust scaling
    - Power transformation
    - Quantile transformation
    - Custom scaling methods
    - Scaling for different data types
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the feature scaler.
        
        Args:
            config: Configuration dictionary with scaling settings
        """
        self.config = config or {}
        self.scalers = {}
        self.scaling_methods = {
            'standard': StandardScaler,
            'minmax': MinMaxScaler,
            'robust': RobustScaler,
            'normalizer': Normalizer,
            'power': PowerTransformer,
            'quantile': QuantileTransformer
        }
        
        # Default scaling parameters
        self.default_params = {
            'standard': {},
            'minmax': {'feature_range': (0, 1)},
            'robust': {},
            'normalizer': {'norm': 'l2'},
            'power': {'method': 'yeo-johnson'},
            'quantile': {'output_distribution': 'uniform'}
        }
        
        logger.info("Initialized FeatureScaler")
    
    def fit_scalers(self, df: pd.DataFrame, 
                   scaling_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Fit scalers for specified columns.
        
        Args:
            df: Input DataFrame
            scaling_config: Configuration for scaling operations
            
        Returns:
            Dictionary with fitted scalers
        """
        scaling_config = scaling_config or self.config
        fitted_scalers = {}
        
        for column, config in scaling_config.items():
            if column not in df.columns:
                logger.warning(f"Column {column} not found in DataFrame")
                continue
            
            method = config.get('method', 'standard')
            params = config.get('params', self.default_params.get(method, {}))
            
            if method not in self.scaling_methods:
                logger.warning(f"Unknown scaling method: {method}")
                continue
            
            try:
                # Create scaler
                scaler_class = self.scaling_methods[method]
                scaler = scaler_class(**params)
                
                # Fit scaler
                column_data = df[column].values.reshape(-1, 1)
                scaler.fit(column_data)
                
                fitted_scalers[column] = {
                    'scaler': scaler,
                    'method': method,
                    'params': params
                }
                
                logger.info(f"Fitted {method} scaler for column {column}")
                
            except Exception as e:
                logger.error(f"Failed to fit scaler for column {column}: {e}")
                continue
        
        self.scalers = fitted_scalers
        return fitted_scalers
    
    def transform_data(self, df: pd.DataFrame, 
                      fitted_scalers: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Transform data using fitted scalers.
        
        Args:
            df: Input DataFrame
            fitted_scalers: Dictionary of fitted scalers
            
        Returns:
            Transformed DataFrame
        """
        if fitted_scalers is None:
            fitted_scalers = self.scalers
        
        if not fitted_scalers:
            logger.warning("No fitted scalers found. Call fit_scalers first.")
            return df
        
        transformed_df = df.copy()
        
        for column, scaler_info in fitted_scalers.items():
            if column not in transformed_df.columns:
                continue
            
            try:
                scaler = scaler_info['scaler']
                column_data = transformed_df[column].values.reshape(-1, 1)
                
                # Transform data
                transformed_data = scaler.transform(column_data)
                
                # Update DataFrame
                transformed_df[column] = transformed_data.flatten()
                
                logger.info(f"Transformed column {column} using {scaler_info['method']} scaler")
                
            except Exception as e:
                logger.error(f"Failed to transform column {column}: {e}")
                continue
        
        return transformed_df
    
    def fit_transform_data(self, df: pd.DataFrame, 
                          scaling_config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Fit scalers and transform data in one step.
        
        Args:
            df: Input DataFrame
            scaling_config: Configuration for scaling operations
            
        Returns:
            Transformed DataFrame
        """
        # Fit scalers
        fitted_scalers = self.fit_scalers(df, scaling_config)
        
        # Transform data
        transformed_df = self.transform_data(df, fitted_scalers)
        
        return transformed_df
    
    def inverse_transform_data(self, df: pd.DataFrame, 
                              fitted_scalers: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Inverse transform data using fitted scalers.
        
        Args:
            df: Input DataFrame
            fitted_scalers: Dictionary of fitted scalers
            
        Returns:
            Inverse transformed DataFrame
        """
        if fitted_scalers is None:
            fitted_scalers = self.scalers
        
        if not fitted_scalers:
            logger.warning("No fitted scalers found. Call fit_scalers first.")
            return df
        
        inverse_transformed_df = df.copy()
        
        for column, scaler_info in fitted_scalers.items():
            if column not in inverse_transformed_df.columns:
                continue
            
            try:
                scaler = scaler_info['scaler']
                
                # Check if scaler supports inverse transform
                if not hasattr(scaler, 'inverse_transform'):
                    logger.warning(f"Scaler for column {column} does not support inverse transform")
                    continue
                
                column_data = inverse_transformed_df[column].values.reshape(-1, 1)
                
                # Inverse transform data
                inverse_transformed_data = scaler.inverse_transform(column_data)
                
                # Update DataFrame
                inverse_transformed_df[column] = inverse_transformed_data.flatten()
                
                logger.info(f"Inverse transformed column {column}")
                
            except Exception as e:
                logger.error(f"Failed to inverse transform column {column}: {e}")
                continue
        
        return inverse_transformed_df
    
    def scale_numeric_features(self, df: pd.DataFrame, 
                              method: str = 'standard',
                              exclude_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Scale all numeric features in DataFrame.
        
        Args:
            df: Input DataFrame
            method: Scaling method to use
            exclude_columns: Columns to exclude from scaling
            
        Returns:
            DataFrame with scaled numeric features
        """
        exclude_columns = exclude_columns or []
        
        # Get numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_columns = [col for col in numeric_columns if col not in exclude_columns]
        
        if not numeric_columns:
            logger.warning("No numeric columns found for scaling")
            return df
        
        # Create scaling configuration
        scaling_config = {col: {'method': method} for col in numeric_columns}
        
        # Fit and transform
        scaled_df = self.fit_transform_data(df, scaling_config)
        
        logger.info(f"Scaled {len(numeric_columns)} numeric columns using {method} method")
        
        return scaled_df
    
    def scale_categorical_features(self, df: pd.DataFrame, 
                                 method: str = 'onehot',
                                 exclude_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Scale categorical features in DataFrame.
        
        Args:
            df: Input DataFrame
            method: Scaling method to use ('onehot', 'label', 'target')
            exclude_columns: Columns to exclude from scaling
            
        Returns:
            DataFrame with scaled categorical features
        """
        exclude_columns = exclude_columns or []
        
        # Get categorical columns
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        categorical_columns = [col for col in categorical_columns if col not in exclude_columns]
        
        if not categorical_columns:
            logger.warning("No categorical columns found for scaling")
            return df
        
        scaled_df = df.copy()
        
        for column in categorical_columns:
            if method == 'onehot':
                # One-hot encoding
                dummies = pd.get_dummies(df[column], prefix=column)
                scaled_df = pd.concat([scaled_df, dummies], axis=1)
                scaled_df = scaled_df.drop(column, axis=1)
                
            elif method == 'label':
                # Label encoding
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                scaled_df[column] = le.fit_transform(df[column].astype(str))
                
            elif method == 'target':
                # Target encoding (mean encoding)
                target_column = self.config.get('target_column')
                if target_column and target_column in df.columns:
                    target_means = df.groupby(column)[target_column].mean()
                    scaled_df[column] = df[column].map(target_means)
                else:
                    logger.warning(f"Target column not specified for target encoding of {column}")
                    continue
        
        logger.info(f"Scaled {len(categorical_columns)} categorical columns using {method} method")
        
        return scaled_df
    
    def scale_time_series_features(self, df: pd.DataFrame, 
                                  time_column: str = 'timestamp',
                                  method: str = 'standard',
                                  window_size: int = 100) -> pd.DataFrame:
        """
        Scale time series features using rolling window statistics.
        
        Args:
            df: Input DataFrame
            time_column: Name of the time column
            method: Scaling method ('standard', 'minmax', 'robust')
            window_size: Size of the rolling window
            
        Returns:
            DataFrame with scaled time series features
        """
        if time_column not in df.columns:
            logger.warning(f"Time column {time_column} not found")
            return df
        
        scaled_df = df.copy()
        
        # Sort by time
        scaled_df = scaled_df.sort_values(time_column).reset_index(drop=True)
        
        # Get numeric columns (excluding time column)
        numeric_columns = scaled_df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_columns = [col for col in numeric_columns if col != time_column]
        
        for column in numeric_columns:
            if method == 'standard':
                # Rolling z-score normalization
                rolling_mean = scaled_df[column].rolling(window=window_size, min_periods=1).mean()
                rolling_std = scaled_df[column].rolling(window=window_size, min_periods=1).std()
                scaled_df[column] = (scaled_df[column] - rolling_mean) / rolling_std
                
            elif method == 'minmax':
                # Rolling min-max scaling
                rolling_min = scaled_df[column].rolling(window=window_size, min_periods=1).min()
                rolling_max = scaled_df[column].rolling(window=window_size, min_periods=1).max()
                scaled_df[column] = (scaled_df[column] - rolling_min) / (rolling_max - rolling_min)
                
            elif method == 'robust':
                # Rolling robust scaling
                rolling_median = scaled_df[column].rolling(window=window_size, min_periods=1).median()
                rolling_mad = scaled_df[column].rolling(window=window_size, min_periods=1).apply(
                    lambda x: np.median(np.abs(x - np.median(x)))
                )
                scaled_df[column] = (scaled_df[column] - rolling_median) / rolling_mad
        
        logger.info(f"Scaled {len(numeric_columns)} time series features using {method} method")
        
        return scaled_df
    
    def scale_image_features(self, images: List[np.ndarray], 
                           method: str = 'standard',
                           per_channel: bool = True) -> List[np.ndarray]:
        """
        Scale image features.
        
        Args:
            images: List of image arrays
            method: Scaling method to use
            per_channel: Whether to scale per channel or globally
            
        Returns:
            List of scaled image arrays
        """
        if not images:
            return images
        
        scaled_images = []
        
        for image in images:
            if method == 'standard':
                if per_channel and len(image.shape) == 3:
                    # Scale per channel
                    scaled_image = np.zeros_like(image)
                    for i in range(image.shape[2]):
                        channel = image[:, :, i]
                        scaled_image[:, :, i] = (channel - np.mean(channel)) / np.std(channel)
                else:
                    # Scale globally
                    scaled_image = (image - np.mean(image)) / np.std(image)
                    
            elif method == 'minmax':
                if per_channel and len(image.shape) == 3:
                    # Scale per channel
                    scaled_image = np.zeros_like(image)
                    for i in range(image.shape[2]):
                        channel = image[:, :, i]
                        scaled_image[:, :, i] = (channel - np.min(channel)) / (np.max(channel) - np.min(channel))
                else:
                    # Scale globally
                    scaled_image = (image - np.min(image)) / (np.max(image) - np.min(image))
                    
            elif method == 'normalize':
                # L2 normalization
                if per_channel and len(image.shape) == 3:
                    scaled_image = np.zeros_like(image)
                    for i in range(image.shape[2]):
                        channel = image[:, :, i]
                        norm = np.linalg.norm(channel)
                        scaled_image[:, :, i] = channel / norm if norm > 0 else channel
                else:
                    norm = np.linalg.norm(image)
                    scaled_image = image / norm if norm > 0 else image
                    
            else:
                logger.warning(f"Unknown scaling method: {method}")
                scaled_image = image
            
            scaled_images.append(scaled_image)
        
        logger.info(f"Scaled {len(images)} images using {method} method")
        
        return scaled_images
    
    def save_scalers(self, file_path: Union[str, Path]):
        """
        Save fitted scalers to file.
        
        Args:
            file_path: Path to save scalers
        """
        if not self.scalers:
            logger.warning("No fitted scalers to save")
            return
        
        try:
            joblib.dump(self.scalers, file_path)
            logger.info(f"Saved scalers to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save scalers: {e}")
    
    def load_scalers(self, file_path: Union[str, Path]):
        """
        Load fitted scalers from file.
        
        Args:
            file_path: Path to load scalers from
        """
        try:
            self.scalers = joblib.load(file_path)
            logger.info(f"Loaded scalers from {file_path}")
        except Exception as e:
            logger.error(f"Failed to load scalers: {e}")
    
    def get_scaling_summary(self, df: pd.DataFrame, 
                           fitted_scalers: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Get summary of scaling operations.
        
        Args:
            df: Original DataFrame
            fitted_scalers: Dictionary of fitted scalers
            
        Returns:
            Dictionary with scaling summary
        """
        if fitted_scalers is None:
            fitted_scalers = self.scalers
        
        summary = {
            'total_columns': len(df.columns),
            'scaled_columns': len(fitted_scalers),
            'scaling_methods': {},
            'column_statistics': {}
        }
        
        for column, scaler_info in fitted_scalers.items():
            method = scaler_info['method']
            if method not in summary['scaling_methods']:
                summary['scaling_methods'][method] = 0
            summary['scaling_methods'][method] += 1
            
            # Get column statistics
            if column in df.columns:
                summary['column_statistics'][column] = {
                    'original_mean': float(df[column].mean()),
                    'original_std': float(df[column].std()),
                    'original_min': float(df[column].min()),
                    'original_max': float(df[column].max()),
                    'scaling_method': method
                }
        
        return summary


class CustomScaler(BaseEstimator, TransformerMixin):
    """
    Custom scaler for specific PBF-LB/M manufacturing features.
    """
    
    def __init__(self, method: str = 'log_transform', **kwargs):
        """
        Initialize custom scaler.
        
        Args:
            method: Scaling method to use
            **kwargs: Additional parameters
        """
        self.method = method
        self.params = kwargs
        self.is_fitted = False
    
    def fit(self, X, y=None):
        """Fit the scaler."""
        self.is_fitted = True
        return self
    
    def transform(self, X):
        """Transform the data."""
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted before transform")
        
        X_transformed = X.copy()
        
        if self.method == 'log_transform':
            # Log transformation with offset to handle zeros
            offset = self.params.get('offset', 1)
            X_transformed = np.log(X + offset)
            
        elif self.method == 'sqrt_transform':
            # Square root transformation
            X_transformed = np.sqrt(X)
            
        elif self.method == 'box_cox':
            # Box-Cox transformation
            from scipy.stats import boxcox
            X_transformed = boxcox(X + 1)[0]  # +1 to handle zeros
            
        elif self.method == 'yeo_johnson':
            # Yeo-Johnson transformation
            from sklearn.preprocessing import PowerTransformer
            pt = PowerTransformer(method='yeo-johnson')
            X_transformed = pt.fit_transform(X.reshape(-1, 1)).flatten()
            
        else:
            logger.warning(f"Unknown custom scaling method: {self.method}")
            X_transformed = X
        
        return X_transformed
    
    def inverse_transform(self, X):
        """Inverse transform the data."""
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted before inverse transform")
        
        X_inverse = X.copy()
        
        if self.method == 'log_transform':
            # Inverse log transformation
            offset = self.params.get('offset', 1)
            X_inverse = np.exp(X) - offset
            
        elif self.method == 'sqrt_transform':
            # Inverse square root transformation
            X_inverse = X ** 2
            
        else:
            logger.warning(f"Inverse transform not implemented for method: {self.method}")
            X_inverse = X
        
        return X_inverse
