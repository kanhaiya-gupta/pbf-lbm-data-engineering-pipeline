"""
Time Series Metrics

This module implements evaluation metrics for time series models
in PBF-LB/M manufacturing processes.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Union, Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class TimeSeriesMetrics:
    """
    Utility class for time series model evaluation metrics.
    
    This class handles:
    - Standard time series metrics (MSE, MAE, MAPE, etc.)
    - Manufacturing-specific time series metrics
    - Time series error analysis
    - Trend and seasonality analysis
    - Forecasting accuracy metrics
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the time series metrics calculator.
        
        Args:
            config: Configuration dictionary with metric settings
        """
        self.config = config or {}
        
        # Standard time series metrics
        self.standard_metrics = {
            'mse': self._calculate_mse,
            'rmse': self._calculate_rmse,
            'mae': self._calculate_mae,
            'mape': self._calculate_mape,
            'smape': self._calculate_smape,
            'r2': self._calculate_r2,
            'theil_u': self._calculate_theil_u,
            'mean_absolute_scaled_error': self._calculate_mase,
            'symmetric_mape': self._calculate_smape,
            'mean_absolute_percentage_error': self._calculate_mape
        }
        
        # Manufacturing-specific time series metrics
        self.manufacturing_metrics = {
            'process_stability': self._calculate_process_stability,
            'trend_accuracy': self._calculate_trend_accuracy,
            'seasonality_accuracy': self._calculate_seasonality_accuracy,
            'anomaly_detection_accuracy': self._calculate_anomaly_detection_accuracy,
            'control_chart_performance': self._calculate_control_chart_performance,
            'production_forecast_accuracy': self._calculate_production_forecast_accuracy
        }
        
        logger.info("Initialized TimeSeriesMetrics")
    
    def calculate_metrics(self, y_true: Union[np.ndarray, pd.Series, List], 
                         y_pred: Union[np.ndarray, pd.Series, List],
                         y_naive: Optional[Union[np.ndarray, pd.Series, List]] = None,
                         metrics: Optional[List[str]] = None,
                         manufacturing_metrics: bool = False,
                         time_index: Optional[Union[np.ndarray, pd.Series, List]] = None) -> Dict[str, float]:
        """
        Calculate time series metrics.
        
        Args:
            y_true: True time series values
            y_pred: Predicted time series values
            y_naive: Naive forecast values (for MASE calculation)
            metrics: List of metrics to calculate (None for all standard metrics)
            manufacturing_metrics: Whether to include manufacturing-specific metrics
            time_index: Time index for trend and seasonality analysis
            
        Returns:
            Dictionary with calculated metrics
        """
        # Convert inputs to numpy arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        if y_naive is not None:
            y_naive = np.array(y_naive)
        
        if time_index is not None:
            time_index = np.array(time_index)
        
        # Validate inputs
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length")
        
        if len(y_true) == 0:
            raise ValueError("Input arrays cannot be empty")
        
        # Remove any NaN values
        valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true = y_true[valid_mask]
        y_pred = y_pred[valid_mask]
        
        if y_naive is not None:
            y_naive = y_naive[valid_mask]
        
        if time_index is not None:
            time_index = time_index[valid_mask]
        
        if len(y_true) == 0:
            raise ValueError("No valid values found after removing NaN")
        
        results = {}
        
        # Calculate standard metrics
        if metrics is None:
            metrics = list(self.standard_metrics.keys())
        
        for metric in metrics:
            if metric in self.standard_metrics:
                try:
                    if metric == 'mean_absolute_scaled_error' and y_naive is None:
                        logger.warning("Naive forecast required for MASE calculation")
                        results[metric] = np.nan
                        continue
                    
                    results[metric] = self.standard_metrics[metric](y_true, y_pred, y_naive)
                except Exception as e:
                    logger.warning(f"Failed to calculate {metric}: {e}")
                    results[metric] = np.nan
        
        # Calculate manufacturing-specific metrics
        if manufacturing_metrics:
            for metric_name, metric_func in self.manufacturing_metrics.items():
                try:
                    results[metric_name] = metric_func(y_true, y_pred, time_index)
                except Exception as e:
                    logger.warning(f"Failed to calculate {metric_name}: {e}")
                    results[metric_name] = np.nan
        
        return results
    
    def _calculate_mse(self, y_true: np.ndarray, y_pred: np.ndarray, y_naive: Optional[np.ndarray] = None) -> float:
        """Calculate Mean Squared Error."""
        return float(mean_squared_error(y_true, y_pred))
    
    def _calculate_rmse(self, y_true: np.ndarray, y_pred: np.ndarray, y_naive: Optional[np.ndarray] = None) -> float:
        """Calculate Root Mean Squared Error."""
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))
    
    def _calculate_mae(self, y_true: np.ndarray, y_pred: np.ndarray, y_naive: Optional[np.ndarray] = None) -> float:
        """Calculate Mean Absolute Error."""
        return float(mean_absolute_error(y_true, y_pred))
    
    def _calculate_mape(self, y_true: np.ndarray, y_pred: np.ndarray, y_naive: Optional[np.ndarray] = None) -> float:
        """Calculate Mean Absolute Percentage Error."""
        # Avoid division by zero
        mask = y_true != 0
        if not np.any(mask):
            return np.nan
        
        return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
    
    def _calculate_smape(self, y_true: np.ndarray, y_pred: np.ndarray, y_naive: Optional[np.ndarray] = None) -> float:
        """Calculate Symmetric Mean Absolute Percentage Error."""
        numerator = np.abs(y_true - y_pred)
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
        
        # Avoid division by zero
        mask = denominator != 0
        if not np.any(mask):
            return np.nan
        
        return float(np.mean(numerator[mask] / denominator[mask]) * 100)
    
    def _calculate_r2(self, y_true: np.ndarray, y_pred: np.ndarray, y_naive: Optional[np.ndarray] = None) -> float:
        """Calculate R-squared score."""
        return float(r2_score(y_true, y_pred))
    
    def _calculate_theil_u(self, y_true: np.ndarray, y_pred: np.ndarray, y_naive: Optional[np.ndarray] = None) -> float:
        """Calculate Theil's U statistic."""
        # Theil's U = RMSE of forecast / RMSE of naive forecast
        if y_naive is None:
            # Use naive forecast (previous value)
            y_naive = np.roll(y_true, 1)
            y_naive[0] = y_true[0]
        
        rmse_forecast = np.sqrt(mean_squared_error(y_true, y_pred))
        rmse_naive = np.sqrt(mean_squared_error(y_true, y_naive))
        
        if rmse_naive == 0:
            return np.nan
        
        return float(rmse_forecast / rmse_naive)
    
    def _calculate_mase(self, y_true: np.ndarray, y_pred: np.ndarray, y_naive: np.ndarray) -> float:
        """Calculate Mean Absolute Scaled Error."""
        mae_forecast = mean_absolute_error(y_true, y_pred)
        mae_naive = mean_absolute_error(y_true, y_naive)
        
        if mae_naive == 0:
            return np.nan
        
        return float(mae_forecast / mae_naive)
    
    def _calculate_process_stability(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                   time_index: Optional[np.ndarray] = None) -> float:
        """Calculate process stability based on prediction accuracy."""
        # Process stability is measured by consistent prediction accuracy
        errors = np.abs(y_true - y_pred)
        
        # Calculate coefficient of variation of errors
        if np.mean(errors) == 0:
            return 100.0  # Perfect stability
        
        cv_errors = np.std(errors) / np.mean(errors)
        
        # Convert to stability score (0-100)
        stability_score = max(0, 100 - cv_errors * 100)
        return float(stability_score)
    
    def _calculate_trend_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                time_index: Optional[np.ndarray] = None) -> float:
        """Calculate trend prediction accuracy."""
        if time_index is None:
            time_index = np.arange(len(y_true))
        
        # Calculate trends using linear regression
        from sklearn.linear_model import LinearRegression
        
        # True trend
        X_true = time_index.reshape(-1, 1)
        reg_true = LinearRegression().fit(X_true, y_true)
        trend_true = reg_true.coef_[0]
        
        # Predicted trend
        X_pred = time_index.reshape(-1, 1)
        reg_pred = LinearRegression().fit(X_pred, y_pred)
        trend_pred = reg_pred.coef_[0]
        
        # Calculate trend accuracy
        if trend_true == 0:
            trend_accuracy = 100.0 if trend_pred == 0 else 0.0
        else:
            trend_error = abs(trend_pred - trend_true) / abs(trend_true)
            trend_accuracy = max(0, 100 - trend_error * 100)
        
        return float(trend_accuracy)
    
    def _calculate_seasonality_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                      time_index: Optional[np.ndarray] = None) -> float:
        """Calculate seasonality prediction accuracy."""
        if time_index is None:
            time_index = np.arange(len(y_true))
        
        # Simple seasonality detection using autocorrelation
        def calculate_autocorr(series, lag):
            if len(series) <= lag:
                return 0
            return np.corrcoef(series[:-lag], series[lag:])[0, 1]
        
        # Calculate autocorrelation for different lags
        max_lag = min(20, len(y_true) // 4)
        autocorr_true = []
        autocorr_pred = []
        
        for lag in range(1, max_lag + 1):
            autocorr_true.append(calculate_autocorr(y_true, lag))
            autocorr_pred.append(calculate_autocorr(y_pred, lag))
        
        # Calculate seasonality accuracy
        autocorr_true = np.array(autocorr_true)
        autocorr_pred = np.array(autocorr_pred)
        
        # Remove NaN values
        valid_mask = ~(np.isnan(autocorr_true) | np.isnan(autocorr_pred))
        if not np.any(valid_mask):
            return 0.0
        
        autocorr_true = autocorr_true[valid_mask]
        autocorr_pred = autocorr_pred[valid_mask]
        
        # Calculate correlation between autocorrelations
        if len(autocorr_true) > 1:
            seasonality_corr = np.corrcoef(autocorr_true, autocorr_pred)[0, 1]
            if np.isnan(seasonality_corr):
                seasonality_corr = 0
            seasonality_accuracy = max(0, seasonality_corr * 100)
        else:
            seasonality_accuracy = 0.0
        
        return float(seasonality_accuracy)
    
    def _calculate_anomaly_detection_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                            time_index: Optional[np.ndarray] = None) -> float:
        """Calculate anomaly detection accuracy."""
        # Define anomalies as points where prediction error exceeds threshold
        errors = np.abs(y_true - y_pred)
        
        # Use IQR method to detect anomalies
        Q1 = np.percentile(errors, 25)
        Q3 = np.percentile(errors, 75)
        IQR = Q3 - Q1
        threshold = Q3 + 1.5 * IQR
        
        # Detect anomalies in true data
        true_anomalies = errors > threshold
        
        # For anomaly detection, we want to minimize false positives and false negatives
        # This is a simplified version - in practice, you'd have ground truth anomaly labels
        
        # Calculate accuracy based on error distribution
        if np.any(true_anomalies):
            # If there are anomalies, accuracy is based on how well we predict them
            anomaly_accuracy = 100 - np.mean(true_anomalies) * 100
        else:
            # If no anomalies, perfect accuracy
            anomaly_accuracy = 100.0
        
        return float(anomaly_accuracy)
    
    def _calculate_control_chart_performance(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                           time_index: Optional[np.ndarray] = None) -> float:
        """Calculate control chart performance."""
        # Control chart performance based on prediction accuracy within control limits
        errors = np.abs(y_true - y_pred)
        
        # Calculate control limits (3-sigma)
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        ucl = mean_error + 3 * std_error  # Upper control limit
        lcl = max(0, mean_error - 3 * std_error)  # Lower control limit
        
        # Count points within control limits
        within_limits = (errors >= lcl) & (errors <= ucl)
        control_chart_performance = np.mean(within_limits) * 100
        
        return float(control_chart_performance)
    
    def _calculate_production_forecast_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                              time_index: Optional[np.ndarray] = None) -> float:
        """Calculate production forecast accuracy."""
        # Production forecast accuracy combines multiple aspects
        mape = self._calculate_mape(y_true, y_pred)
        mae = self._calculate_mae(y_true, y_pred)
        
        # Normalize MAE by mean of true values
        normalized_mae = mae / np.mean(y_true) * 100
        
        # Combine MAPE and normalized MAE
        forecast_accuracy = 100 - (mape + normalized_mae) / 2
        
        return float(max(0, forecast_accuracy))
    
    def calculate_time_series_analysis(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                     time_index: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Perform detailed time series analysis.
        
        Args:
            y_true: True time series values
            y_pred: Predicted time series values
            time_index: Time index for analysis
            
        Returns:
            Dictionary with time series analysis results
        """
        if time_index is None:
            time_index = np.arange(len(y_true))
        
        analysis = {
            'basic_statistics': {},
            'error_analysis': {},
            'trend_analysis': {},
            'seasonality_analysis': {},
            'stability_analysis': {}
        }
        
        # Basic statistics
        analysis['basic_statistics'] = {
            'true_mean': float(np.mean(y_true)),
            'true_std': float(np.std(y_true)),
            'pred_mean': float(np.mean(y_pred)),
            'pred_std': float(np.std(y_pred)),
            'correlation': float(np.corrcoef(y_true, y_pred)[0, 1])
        }
        
        # Error analysis
        errors = y_pred - y_true
        analysis['error_analysis'] = {
            'mean_error': float(np.mean(errors)),
            'std_error': float(np.std(errors)),
            'mean_absolute_error': float(np.mean(np.abs(errors))),
            'max_error': float(np.max(np.abs(errors))),
            'error_autocorrelation': self._calculate_error_autocorrelation(errors)
        }
        
        # Trend analysis
        analysis['trend_analysis'] = self._analyze_trends(y_true, y_pred, time_index)
        
        # Seasonality analysis
        analysis['seasonality_analysis'] = self._analyze_seasonality(y_true, y_pred, time_index)
        
        # Stability analysis
        analysis['stability_analysis'] = self._analyze_stability(y_true, y_pred, time_index)
        
        return analysis
    
    def _calculate_error_autocorrelation(self, errors: np.ndarray) -> float:
        """Calculate autocorrelation of errors."""
        if len(errors) < 2:
            return 0.0
        
        # Calculate lag-1 autocorrelation
        if len(errors) > 1:
            autocorr = np.corrcoef(errors[:-1], errors[1:])[0, 1]
            return float(autocorr) if not np.isnan(autocorr) else 0.0
        return 0.0
    
    def _analyze_trends(self, y_true: np.ndarray, y_pred: np.ndarray, time_index: np.ndarray) -> Dict[str, Any]:
        """Analyze trends in the time series."""
        from sklearn.linear_model import LinearRegression
        
        # Fit linear trends
        X = time_index.reshape(-1, 1)
        reg_true = LinearRegression().fit(X, y_true)
        reg_pred = LinearRegression().fit(X, y_pred)
        
        trend_analysis = {
            'true_trend_slope': float(reg_true.coef_[0]),
            'pred_trend_slope': float(reg_pred.coef_[0]),
            'trend_difference': float(reg_pred.coef_[0] - reg_true.coef_[0]),
            'trend_accuracy': self._calculate_trend_accuracy(y_true, y_pred, time_index)
        }
        
        return trend_analysis
    
    def _analyze_seasonality(self, y_true: np.ndarray, y_pred: np.ndarray, time_index: np.ndarray) -> Dict[str, Any]:
        """Analyze seasonality in the time series."""
        # Simple seasonality analysis using autocorrelation
        def calculate_autocorr(series, lag):
            if len(series) <= lag:
                return 0
            corr = np.corrcoef(series[:-lag], series[lag:])[0, 1]
            return corr if not np.isnan(corr) else 0
        
        max_lag = min(20, len(y_true) // 4)
        autocorr_true = [calculate_autocorr(y_true, lag) for lag in range(1, max_lag + 1)]
        autocorr_pred = [calculate_autocorr(y_pred, lag) for lag in range(1, max_lag + 1)]
        
        seasonality_analysis = {
            'max_autocorr_true': float(np.max(autocorr_true)) if autocorr_true else 0.0,
            'max_autocorr_pred': float(np.max(autocorr_pred)) if autocorr_pred else 0.0,
            'seasonality_accuracy': self._calculate_seasonality_accuracy(y_true, y_pred, time_index)
        }
        
        return seasonality_analysis
    
    def _analyze_stability(self, y_true: np.ndarray, y_pred: np.ndarray, time_index: np.ndarray) -> Dict[str, Any]:
        """Analyze stability of the time series."""
        errors = np.abs(y_true - y_pred)
        
        # Calculate rolling statistics
        window_size = min(10, len(errors) // 4)
        if window_size > 1:
            rolling_mean = pd.Series(errors).rolling(window=window_size).mean()
            rolling_std = pd.Series(errors).rolling(window=window_size).std()
            
            stability_analysis = {
                'mean_error': float(np.mean(errors)),
                'std_error': float(np.std(errors)),
                'cv_error': float(np.std(errors) / np.mean(errors)) if np.mean(errors) > 0 else 0.0,
                'rolling_mean_std': float(rolling_std.std()) if not rolling_std.isna().all() else 0.0,
                'stability_score': self._calculate_process_stability(y_true, y_pred, time_index)
            }
        else:
            stability_analysis = {
                'mean_error': float(np.mean(errors)),
                'std_error': float(np.std(errors)),
                'cv_error': float(np.std(errors) / np.mean(errors)) if np.mean(errors) > 0 else 0.0,
                'rolling_mean_std': 0.0,
                'stability_score': self._calculate_process_stability(y_true, y_pred, time_index)
            }
        
        return stability_analysis
    
    def visualize_time_series(self, y_true: np.ndarray, y_pred: np.ndarray, 
                            time_index: Optional[np.ndarray] = None,
                            title: str = "Time Series Prediction",
                            save_path: Optional[str] = None) -> None:
        """
        Visualize time series predictions.
        
        Args:
            y_true: True time series values
            y_pred: Predicted time series values
            time_index: Time index for x-axis
            title: Plot title
            save_path: Path to save the plot
        """
        if time_index is None:
            time_index = np.arange(len(y_true))
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Time series plot
        axes[0, 0].plot(time_index, y_true, label='True', alpha=0.7)
        axes[0, 0].plot(time_index, y_pred, label='Predicted', alpha=0.7)
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Value')
        axes[0, 0].set_title('Time Series Prediction')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Scatter plot
        axes[0, 1].scatter(y_true, y_pred, alpha=0.6)
        axes[0, 1].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        axes[0, 1].set_xlabel('True Values')
        axes[0, 1].set_ylabel('Predicted Values')
        axes[0, 1].set_title('Predicted vs Actual')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Residuals plot
        residuals = y_pred - y_true
        axes[1, 0].plot(time_index, residuals, alpha=0.7)
        axes[1, 0].axhline(y=0, color='r', linestyle='--')
        axes[1, 0].set_xlabel('Time')
        axes[1, 0].set_ylabel('Residuals')
        axes[1, 0].set_title('Residuals Over Time')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Error distribution
        axes[1, 1].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
        axes[1, 1].set_xlabel('Residuals')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Error Distribution')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Time series visualization saved to {save_path}")
        
        plt.show()
    
    def generate_report(self, y_true: np.ndarray, y_pred: np.ndarray, 
                       y_naive: Optional[np.ndarray] = None,
                       time_index: Optional[np.ndarray] = None,
                       model_name: str = "Model",
                       manufacturing_metrics: bool = False) -> str:
        """
        Generate a comprehensive time series evaluation report.
        
        Args:
            y_true: True time series values
            y_pred: Predicted time series values
            y_naive: Naive forecast values
            time_index: Time index for analysis
            model_name: Name of the model
            manufacturing_metrics: Whether to include manufacturing-specific metrics
            
        Returns:
            Formatted evaluation report
        """
        # Calculate all metrics
        metrics = self.calculate_metrics(y_true, y_pred, y_naive, manufacturing_metrics=manufacturing_metrics, time_index=time_index)
        
        # Perform time series analysis
        analysis = self.calculate_time_series_analysis(y_true, y_pred, time_index)
        
        # Generate report
        report = []
        report.append("=" * 60)
        report.append(f"TIME SERIES MODEL EVALUATION REPORT")
        report.append(f"Model: {model_name}")
        report.append("=" * 60)
        report.append("")
        
        # Basic metrics
        report.append("BASIC METRICS:")
        report.append(f"  Mean Squared Error (MSE): {metrics.get('mse', 'N/A'):.6f}")
        report.append(f"  Root Mean Squared Error (RMSE): {metrics.get('rmse', 'N/A'):.6f}")
        report.append(f"  Mean Absolute Error (MAE): {metrics.get('mae', 'N/A'):.6f}")
        report.append(f"  Mean Absolute Percentage Error (MAPE): {metrics.get('mape', 'N/A'):.2f}%")
        report.append(f"  Symmetric MAPE (SMAPE): {metrics.get('smape', 'N/A'):.2f}%")
        report.append(f"  R-squared (R²): {metrics.get('r2', 'N/A'):.6f}")
        report.append(f"  Theil's U Statistic: {metrics.get('theil_u', 'N/A'):.6f}")
        
        if y_naive is not None:
            report.append(f"  Mean Absolute Scaled Error (MASE): {metrics.get('mean_absolute_scaled_error', 'N/A'):.6f}")
        
        report.append("")
        
        # Manufacturing metrics
        if manufacturing_metrics:
            report.append("MANUFACTURING METRICS:")
            if 'process_stability' in metrics:
                report.append(f"  Process Stability: {metrics['process_stability']:.2f}%")
            if 'trend_accuracy' in metrics:
                report.append(f"  Trend Accuracy: {metrics['trend_accuracy']:.2f}%")
            if 'seasonality_accuracy' in metrics:
                report.append(f"  Seasonality Accuracy: {metrics['seasonality_accuracy']:.2f}%")
            if 'anomaly_detection_accuracy' in metrics:
                report.append(f"  Anomaly Detection Accuracy: {metrics['anomaly_detection_accuracy']:.2f}%")
            if 'control_chart_performance' in metrics:
                report.append(f"  Control Chart Performance: {metrics['control_chart_performance']:.2f}%")
            if 'production_forecast_accuracy' in metrics:
                report.append(f"  Production Forecast Accuracy: {metrics['production_forecast_accuracy']:.2f}%")
            report.append("")
        
        # Time series analysis
        report.append("TIME SERIES ANALYSIS:")
        basic_stats = analysis['basic_statistics']
        report.append(f"  True Series Mean: {basic_stats['true_mean']:.6f}")
        report.append(f"  True Series Std: {basic_stats['true_std']:.6f}")
        report.append(f"  Predicted Series Mean: {basic_stats['pred_mean']:.6f}")
        report.append(f"  Predicted Series Std: {basic_stats['pred_std']:.6f}")
        report.append(f"  Correlation: {basic_stats['correlation']:.6f}")
        report.append("")
        
        # Error analysis
        error_analysis = analysis['error_analysis']
        report.append("ERROR ANALYSIS:")
        report.append(f"  Mean Error: {error_analysis['mean_error']:.6f}")
        report.append(f"  Error Standard Deviation: {error_analysis['std_error']:.6f}")
        report.append(f"  Mean Absolute Error: {error_analysis['mean_absolute_error']:.6f}")
        report.append(f"  Maximum Error: {error_analysis['max_error']:.6f}")
        report.append(f"  Error Autocorrelation: {error_analysis['error_autocorrelation']:.6f}")
        report.append("")
        
        # Trend analysis
        trend_analysis = analysis['trend_analysis']
        report.append("TREND ANALYSIS:")
        report.append(f"  True Trend Slope: {trend_analysis['true_trend_slope']:.6f}")
        report.append(f"  Predicted Trend Slope: {trend_analysis['pred_trend_slope']:.6f}")
        report.append(f"  Trend Difference: {trend_analysis['trend_difference']:.6f}")
        report.append(f"  Trend Accuracy: {trend_analysis['trend_accuracy']:.2f}%")
        report.append("")
        
        # Seasonality analysis
        seasonality_analysis = analysis['seasonality_analysis']
        report.append("SEASONALITY ANALYSIS:")
        report.append(f"  Max Autocorrelation (True): {seasonality_analysis['max_autocorr_true']:.6f}")
        report.append(f"  Max Autocorrelation (Predicted): {seasonality_analysis['max_autocorr_pred']:.6f}")
        report.append(f"  Seasonality Accuracy: {seasonality_analysis['seasonality_accuracy']:.2f}%")
        report.append("")
        
        # Stability analysis
        stability_analysis = analysis['stability_analysis']
        report.append("STABILITY ANALYSIS:")
        report.append(f"  Mean Error: {stability_analysis['mean_error']:.6f}")
        report.append(f"  Error Standard Deviation: {stability_analysis['std_error']:.6f}")
        report.append(f"  Coefficient of Variation: {stability_analysis['cv_error']:.6f}")
        report.append(f"  Rolling Mean Standard Deviation: {stability_analysis['rolling_mean_std']:.6f}")
        report.append(f"  Stability Score: {stability_analysis['stability_score']:.2f}%")
        report.append("")
        
        # Model performance assessment
        report.append("MODEL PERFORMANCE ASSESSMENT:")
        r2 = metrics.get('r2', 0)
        mape = metrics.get('mape', float('inf'))
        
        if r2 > 0.9:
            report.append("  R² Score: EXCELLENT (>0.9)")
        elif r2 > 0.8:
            report.append("  R² Score: GOOD (0.8-0.9)")
        elif r2 > 0.7:
            report.append("  R² Score: FAIR (0.7-0.8)")
        else:
            report.append("  R² Score: POOR (<0.7)")
        
        if mape < 5:
            report.append("  MAPE: EXCELLENT (<5%)")
        elif mape < 10:
            report.append("  MAPE: GOOD (5-10%)")
        elif mape < 20:
            report.append("  MAPE: FAIR (10-20%)")
        else:
            report.append("  MAPE: POOR (>20%)")
        
        if manufacturing_metrics and 'process_stability' in metrics:
            stability = metrics['process_stability']
            if stability > 90:
                report.append("  Process Stability: EXCELLENT (>90%)")
            elif stability > 80:
                report.append("  Process Stability: GOOD (80-90%)")
            elif stability > 70:
                report.append("  Process Stability: FAIR (70-80%)")
            else:
                report.append("  Process Stability: POOR (<70%)")
        
        report.append("")
        report.append("=" * 60)
        
        return "\n".join(report)
