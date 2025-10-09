"""
Outlier Detector

This module implements utilities for outlier detection and removal
in PBF-LB/M manufacturing processes.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Union, Tuple
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class OutlierDetector:
    """
    Utility class for outlier detection and removal.
    
    This class handles:
    - Statistical outlier detection (IQR, Z-score)
    - Machine learning-based outlier detection
    - Time series outlier detection
    - Multivariate outlier detection
    - Outlier visualization and analysis
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the outlier detector.
        
        Args:
            config: Configuration dictionary with outlier detection settings
        """
        self.config = config or {}
        
        # Outlier detection methods
        self.detection_methods = {
            'iqr': self._detect_iqr_outliers,
            'zscore': self._detect_zscore_outliers,
            'modified_zscore': self._detect_modified_zscore_outliers,
            'isolation_forest': self._detect_isolation_forest_outliers,
            'dbscan': self._detect_dbscan_outliers,
            'lof': self._detect_lof_outliers,
            'elliptic_envelope': self._detect_elliptic_envelope_outliers,
            'time_series': self._detect_time_series_outliers
        }
        
        # Default parameters for each method
        self.default_params = {
            'iqr': {'factor': 1.5},
            'zscore': {'threshold': 3.0},
            'modified_zscore': {'threshold': 3.5},
            'isolation_forest': {'contamination': 0.1, 'random_state': 42},
            'dbscan': {'eps': 0.5, 'min_samples': 5},
            'lof': {'n_neighbors': 20, 'contamination': 0.1},
            'elliptic_envelope': {'contamination': 0.1, 'random_state': 42},
            'time_series': {'window_size': 100, 'threshold': 3.0}
        }
        
        logger.info("Initialized OutlierDetector")
    
    def detect_outliers(self, df: pd.DataFrame, 
                       method: str = 'iqr',
                       columns: Optional[List[str]] = None,
                       params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Detect outliers in DataFrame.
        
        Args:
            df: Input DataFrame
            method: Outlier detection method
            columns: Columns to analyze (None for all numeric columns)
            params: Parameters for the detection method
            
        Returns:
            Dictionary with outlier detection results
        """
        if method not in self.detection_methods:
            raise ValueError(f"Unknown outlier detection method: {method}")
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not columns:
            logger.warning("No numeric columns found for outlier detection")
            return {}
        
        params = params or self.default_params.get(method, {})
        
        results = {
            'method': method,
            'parameters': params,
            'outlier_indices': [],
            'outlier_counts': {},
            'outlier_percentages': {},
            'column_results': {}
        }
        
        for column in columns:
            if column not in df.columns:
                continue
            
            try:
                column_data = df[column].dropna()
                if len(column_data) == 0:
                    continue
                
                # Detect outliers for this column
                outlier_indices = self.detection_methods[method](column_data, params)
                
                # Convert to DataFrame indices
                df_indices = column_data.index[outlier_indices].tolist()
                
                results['column_results'][column] = {
                    'outlier_indices': df_indices,
                    'outlier_count': len(df_indices),
                    'outlier_percentage': len(df_indices) / len(column_data) * 100,
                    'outlier_values': column_data.iloc[outlier_indices].tolist()
                }
                
                results['outlier_indices'].extend(df_indices)
                results['outlier_counts'][column] = len(df_indices)
                results['outlier_percentages'][column] = len(df_indices) / len(column_data) * 100
                
            except Exception as e:
                logger.error(f"Failed to detect outliers in column {column}: {e}")
                continue
        
        # Remove duplicate indices
        results['outlier_indices'] = list(set(results['outlier_indices']))
        
        logger.info(f"Detected outliers using {method} method")
        
        return results
    
    def _detect_iqr_outliers(self, data: pd.Series, params: Dict[str, Any]) -> np.ndarray:
        """Detect outliers using IQR method."""
        factor = params.get('factor', 1.5)
        
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        
        outlier_mask = (data < lower_bound) | (data > upper_bound)
        return np.where(outlier_mask)[0]
    
    def _detect_zscore_outliers(self, data: pd.Series, params: Dict[str, Any]) -> np.ndarray:
        """Detect outliers using Z-score method."""
        threshold = params.get('threshold', 3.0)
        
        z_scores = np.abs((data - data.mean()) / data.std())
        outlier_mask = z_scores > threshold
        
        return np.where(outlier_mask)[0]
    
    def _detect_modified_zscore_outliers(self, data: pd.Series, params: Dict[str, Any]) -> np.ndarray:
        """Detect outliers using modified Z-score method."""
        threshold = params.get('threshold', 3.5)
        
        median = data.median()
        mad = np.median(np.abs(data - median))
        
        if mad == 0:
            return np.array([])
        
        modified_z_scores = 0.6745 * (data - median) / mad
        outlier_mask = np.abs(modified_z_scores) > threshold
        
        return np.where(outlier_mask)[0]
    
    def _detect_isolation_forest_outliers(self, data: pd.Series, params: Dict[str, Any]) -> np.ndarray:
        """Detect outliers using Isolation Forest."""
        contamination = params.get('contamination', 0.1)
        random_state = params.get('random_state', 42)
        
        # Reshape data for sklearn
        X = data.values.reshape(-1, 1)
        
        # Fit Isolation Forest
        iso_forest = IsolationForest(contamination=contamination, random_state=random_state)
        outlier_labels = iso_forest.fit_predict(X)
        
        # Return indices of outliers (label = -1)
        return np.where(outlier_labels == -1)[0]
    
    def _detect_dbscan_outliers(self, data: pd.Series, params: Dict[str, Any]) -> np.ndarray:
        """Detect outliers using DBSCAN clustering."""
        eps = params.get('eps', 0.5)
        min_samples = params.get('min_samples', 5)
        
        # Reshape data for sklearn
        X = data.values.reshape(-1, 1)
        
        # Fit DBSCAN
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = dbscan.fit_predict(X)
        
        # Return indices of outliers (label = -1)
        return np.where(cluster_labels == -1)[0]
    
    def _detect_lof_outliers(self, data: pd.Series, params: Dict[str, Any]) -> np.ndarray:
        """Detect outliers using Local Outlier Factor."""
        n_neighbors = params.get('n_neighbors', 20)
        contamination = params.get('contamination', 0.1)
        
        # Reshape data for sklearn
        X = data.values.reshape(-1, 1)
        
        # Fit LOF
        lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
        outlier_labels = lof.fit_predict(X)
        
        # Return indices of outliers (label = -1)
        return np.where(outlier_labels == -1)[0]
    
    def _detect_elliptic_envelope_outliers(self, data: pd.Series, params: Dict[str, Any]) -> np.ndarray:
        """Detect outliers using Elliptic Envelope."""
        contamination = params.get('contamination', 0.1)
        random_state = params.get('random_state', 42)
        
        # Reshape data for sklearn
        X = data.values.reshape(-1, 1)
        
        # Fit Elliptic Envelope
        envelope = EllipticEnvelope(contamination=contamination, random_state=random_state)
        outlier_labels = envelope.fit_predict(X)
        
        # Return indices of outliers (label = -1)
        return np.where(outlier_labels == -1)[0]
    
    def _detect_time_series_outliers(self, data: pd.Series, params: Dict[str, Any]) -> np.ndarray:
        """Detect outliers in time series data."""
        window_size = params.get('window_size', 100)
        threshold = params.get('threshold', 3.0)
        
        # Calculate rolling statistics
        rolling_mean = data.rolling(window=window_size, min_periods=1).mean()
        rolling_std = data.rolling(window=window_size, min_periods=1).std()
        
        # Calculate z-scores
        z_scores = np.abs((data - rolling_mean) / rolling_std)
        outlier_mask = z_scores > threshold
        
        return np.where(outlier_mask)[0]
    
    def detect_multivariate_outliers(self, df: pd.DataFrame, 
                                   columns: Optional[List[str]] = None,
                                   method: str = 'isolation_forest',
                                   params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Detect multivariate outliers.
        
        Args:
            df: Input DataFrame
            columns: Columns to analyze (None for all numeric columns)
            method: Outlier detection method
            params: Parameters for the detection method
            
        Returns:
            Dictionary with multivariate outlier detection results
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(columns) < 2:
            logger.warning("Multivariate outlier detection requires at least 2 columns")
            return {}
        
        params = params or self.default_params.get(method, {})
        
        # Prepare data
        data = df[columns].dropna()
        if len(data) == 0:
            logger.warning("No valid data for multivariate outlier detection")
            return {}
        
        # Standardize data
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        
        results = {
            'method': method,
            'parameters': params,
            'outlier_indices': [],
            'outlier_count': 0,
            'outlier_percentage': 0.0,
            'columns_analyzed': columns
        }
        
        try:
            if method == 'isolation_forest':
                iso_forest = IsolationForest(**params)
                outlier_labels = iso_forest.fit_predict(data_scaled)
                
            elif method == 'dbscan':
                dbscan = DBSCAN(**params)
                outlier_labels = dbscan.fit_predict(data_scaled)
                
            elif method == 'lof':
                lof = LocalOutlierFactor(**params)
                outlier_labels = lof.fit_predict(data_scaled)
                
            elif method == 'elliptic_envelope':
                envelope = EllipticEnvelope(**params)
                outlier_labels = envelope.fit_predict(data_scaled)
                
            else:
                raise ValueError(f"Unknown multivariate outlier detection method: {method}")
            
            # Get outlier indices
            outlier_indices = np.where(outlier_labels == -1)[0]
            df_indices = data.index[outlier_indices].tolist()
            
            results['outlier_indices'] = df_indices
            results['outlier_count'] = len(df_indices)
            results['outlier_percentage'] = len(df_indices) / len(data) * 100
            
            logger.info(f"Detected {len(df_indices)} multivariate outliers using {method}")
            
        except Exception as e:
            logger.error(f"Failed to detect multivariate outliers: {e}")
        
        return results
    
    def remove_outliers(self, df: pd.DataFrame, 
                       outlier_indices: List[int],
                       method: str = 'drop') -> pd.DataFrame:
        """
        Remove outliers from DataFrame.
        
        Args:
            df: Input DataFrame
            outlier_indices: List of outlier indices to remove
            method: Removal method ('drop', 'cap', 'interpolate')
            
        Returns:
            DataFrame with outliers removed
        """
        if method == 'drop':
            # Drop rows with outliers
            cleaned_df = df.drop(index=outlier_indices)
            logger.info(f"Dropped {len(outlier_indices)} rows with outliers")
            
        elif method == 'cap':
            # Cap outliers to threshold values
            cleaned_df = df.copy()
            for column in cleaned_df.select_dtypes(include=[np.number]).columns:
                Q1 = cleaned_df[column].quantile(0.25)
                Q3 = cleaned_df[column].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                cleaned_df[column] = cleaned_df[column].clip(lower=lower_bound, upper=upper_bound)
            
            logger.info("Capped outliers to threshold values")
            
        elif method == 'interpolate':
            # Interpolate outlier values
            cleaned_df = df.copy()
            for column in cleaned_df.select_dtypes(include=[np.number]).columns:
                if column in cleaned_df.columns:
                    cleaned_df.loc[outlier_indices, column] = np.nan
                    cleaned_df[column] = cleaned_df[column].interpolate()
            
            logger.info("Interpolated outlier values")
            
        else:
            logger.warning(f"Unknown outlier removal method: {method}")
            cleaned_df = df.copy()
        
        return cleaned_df
    
    def visualize_outliers(self, df: pd.DataFrame, 
                          outlier_results: Dict[str, Any],
                          columns: Optional[List[str]] = None,
                          save_path: Optional[str] = None) -> None:
        """
        Visualize outliers in the data.
        
        Args:
            df: Input DataFrame
            outlier_results: Results from outlier detection
            columns: Columns to visualize (None for all numeric columns)
            save_path: Path to save the plot
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not columns:
            logger.warning("No numeric columns found for visualization")
            return
        
        # Create subplots
        n_cols = min(3, len(columns))
        n_rows = (len(columns) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()
        
        for i, column in enumerate(columns):
            if i >= len(axes):
                break
            
            ax = axes[i]
            
            # Plot data points
            ax.scatter(range(len(df)), df[column], alpha=0.6, label='Normal')
            
            # Highlight outliers
            if column in outlier_results.get('column_results', {}):
                outlier_indices = outlier_results['column_results'][column]['outlier_indices']
                if outlier_indices:
                    ax.scatter(outlier_indices, df.loc[outlier_indices, column], 
                             color='red', alpha=0.8, label='Outliers')
            
            ax.set_title(f'Outliers in {column}')
            ax.set_xlabel('Index')
            ax.set_ylabel(column)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(columns), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Outlier visualization saved to {save_path}")
        
        plt.show()
    
    def get_outlier_summary(self, outlier_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get summary of outlier detection results.
        
        Args:
            outlier_results: Results from outlier detection
            
        Returns:
            Dictionary with outlier summary
        """
        summary = {
            'method': outlier_results.get('method', 'unknown'),
            'total_outliers': len(outlier_results.get('outlier_indices', [])),
            'columns_analyzed': list(outlier_results.get('outlier_counts', {}).keys()),
            'outlier_statistics': {},
            'recommendations': []
        }
        
        # Calculate statistics for each column
        for column, count in outlier_results.get('outlier_counts', {}).items():
            percentage = outlier_results.get('outlier_percentages', {}).get(column, 0)
            summary['outlier_statistics'][column] = {
                'count': count,
                'percentage': percentage,
                'severity': self._get_outlier_severity(percentage)
            }
        
        # Generate recommendations
        total_percentage = np.mean(list(outlier_results.get('outlier_percentages', {}).values()))
        
        if total_percentage > 10:
            summary['recommendations'].append("High outlier percentage detected. Consider data quality investigation.")
        elif total_percentage > 5:
            summary['recommendations'].append("Moderate outlier percentage detected. Consider outlier treatment.")
        else:
            summary['recommendations'].append("Low outlier percentage detected. Data quality appears good.")
        
        # Method-specific recommendations
        method = outlier_results.get('method', '')
        if method == 'iqr':
            summary['recommendations'].append("IQR method used. Consider adjusting factor parameter for sensitivity.")
        elif method in ['isolation_forest', 'lof']:
            summary['recommendations'].append("ML-based method used. Consider adjusting contamination parameter.")
        
        return summary
    
    def _get_outlier_severity(self, percentage: float) -> str:
        """Get severity level based on outlier percentage."""
        if percentage > 10:
            return 'high'
        elif percentage > 5:
            return 'medium'
        else:
            return 'low'
    
    def compare_outlier_methods(self, df: pd.DataFrame, 
                               columns: Optional[List[str]] = None,
                               methods: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Compare different outlier detection methods.
        
        Args:
            df: Input DataFrame
            columns: Columns to analyze
            methods: List of methods to compare
            
        Returns:
            Dictionary with comparison results
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if methods is None:
            methods = ['iqr', 'zscore', 'isolation_forest', 'lof']
        
        comparison_results = {
            'methods_compared': methods,
            'columns_analyzed': columns,
            'results': {},
            'summary': {}
        }
        
        for method in methods:
            if method in self.detection_methods:
                try:
                    results = self.detect_outliers(df, method=method, columns=columns)
                    comparison_results['results'][method] = results
                    
                    # Calculate summary statistics
                    total_outliers = len(results.get('outlier_indices', []))
                    avg_percentage = np.mean(list(results.get('outlier_percentages', {}).values()))
                    
                    comparison_results['summary'][method] = {
                        'total_outliers': total_outliers,
                        'average_percentage': avg_percentage,
                        'columns_with_outliers': len([c for c, p in results.get('outlier_percentages', {}).items() if p > 0])
                    }
                    
                except Exception as e:
                    logger.error(f"Failed to run {method} outlier detection: {e}")
                    continue
        
        return comparison_results
