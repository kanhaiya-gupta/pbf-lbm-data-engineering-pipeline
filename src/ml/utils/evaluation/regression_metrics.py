"""
Regression Metrics

This module implements evaluation metrics for regression models
in PBF-LB/M manufacturing processes.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Union, Tuple
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error, median_absolute_error,
    explained_variance_score, max_error
)
from sklearn.metrics import mean_squared_log_error
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class RegressionMetrics:
    """
    Utility class for regression model evaluation metrics.
    
    This class handles:
    - Standard regression metrics (MSE, MAE, R², etc.)
    - Manufacturing-specific metrics
    - Error analysis and visualization
    - Model comparison and benchmarking
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the regression metrics calculator.
        
        Args:
            config: Configuration dictionary with metric settings
        """
        self.config = config or {}
        
        # Standard regression metrics
        self.standard_metrics = {
            'mse': self._calculate_mse,
            'rmse': self._calculate_rmse,
            'mae': self._calculate_mae,
            'mape': self._calculate_mape,
            'smape': self._calculate_smape,
            'r2': self._calculate_r2,
            'adjusted_r2': self._calculate_adjusted_r2,
            'explained_variance': self._calculate_explained_variance,
            'median_ae': self._calculate_median_ae,
            'max_error': self._calculate_max_error,
            'msle': self._calculate_msle,
            'rmsle': self._calculate_rmsle
        }
        
        # Manufacturing-specific metrics
        self.manufacturing_metrics = {
            'tolerance_compliance': self._calculate_tolerance_compliance,
            'process_capability': self._calculate_process_capability,
            'defect_rate': self._calculate_defect_rate,
            'quality_score': self._calculate_quality_score,
            'dimensional_accuracy': self._calculate_dimensional_accuracy,
            'surface_roughness_accuracy': self._calculate_surface_roughness_accuracy
        }
        
        logger.info("Initialized RegressionMetrics")
    
    def calculate_metrics(self, y_true: Union[np.ndarray, pd.Series, List], 
                         y_pred: Union[np.ndarray, pd.Series, List],
                         metrics: Optional[List[str]] = None,
                         manufacturing_metrics: bool = False,
                         tolerance_specs: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        Calculate regression metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            metrics: List of metrics to calculate (None for all standard metrics)
            manufacturing_metrics: Whether to include manufacturing-specific metrics
            tolerance_specs: Tolerance specifications for manufacturing metrics
            
        Returns:
            Dictionary with calculated metrics
        """
        # Convert inputs to numpy arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Validate inputs
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length")
        
        if len(y_true) == 0:
            raise ValueError("Input arrays cannot be empty")
        
        # Remove any NaN values
        valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true = y_true[valid_mask]
        y_pred = y_pred[valid_mask]
        
        if len(y_true) == 0:
            raise ValueError("No valid values found after removing NaN")
        
        results = {}
        
        # Calculate standard metrics
        if metrics is None:
            metrics = list(self.standard_metrics.keys())
        
        for metric in metrics:
            if metric in self.standard_metrics:
                try:
                    results[metric] = self.standard_metrics[metric](y_true, y_pred)
                except Exception as e:
                    logger.warning(f"Failed to calculate {metric}: {e}")
                    results[metric] = np.nan
        
        # Calculate manufacturing-specific metrics
        if manufacturing_metrics:
            for metric_name, metric_func in self.manufacturing_metrics.items():
                try:
                    if metric_name in ['tolerance_compliance', 'process_capability', 'dimensional_accuracy']:
                        if tolerance_specs is None:
                            logger.warning(f"Tolerance specifications required for {metric_name}")
                            continue
                        results[metric_name] = metric_func(y_true, y_pred, tolerance_specs)
                    else:
                        results[metric_name] = metric_func(y_true, y_pred)
                except Exception as e:
                    logger.warning(f"Failed to calculate {metric_name}: {e}")
                    results[metric_name] = np.nan
        
        return results
    
    def _calculate_mse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Squared Error."""
        return float(mean_squared_error(y_true, y_pred))
    
    def _calculate_rmse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Root Mean Squared Error."""
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))
    
    def _calculate_mae(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Absolute Error."""
        return float(mean_absolute_error(y_true, y_pred))
    
    def _calculate_mape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Absolute Percentage Error."""
        # Avoid division by zero
        mask = y_true != 0
        if not np.any(mask):
            return np.nan
        
        return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
    
    def _calculate_smape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Symmetric Mean Absolute Percentage Error."""
        numerator = np.abs(y_true - y_pred)
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
        
        # Avoid division by zero
        mask = denominator != 0
        if not np.any(mask):
            return np.nan
        
        return float(np.mean(numerator[mask] / denominator[mask]) * 100)
    
    def _calculate_r2(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate R-squared score."""
        return float(r2_score(y_true, y_pred))
    
    def _calculate_adjusted_r2(self, y_true: np.ndarray, y_pred: np.ndarray, 
                              n_features: Optional[int] = None) -> float:
        """Calculate Adjusted R-squared score."""
        r2 = self._calculate_r2(y_true, y_pred)
        n = len(y_true)
        
        if n_features is None:
            n_features = 1  # Default assumption
        
        if n <= n_features + 1:
            return np.nan
        
        adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - n_features - 1)
        return float(adjusted_r2)
    
    def _calculate_explained_variance(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Explained Variance Score."""
        return float(explained_variance_score(y_true, y_pred))
    
    def _calculate_median_ae(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Median Absolute Error."""
        return float(median_absolute_error(y_true, y_pred))
    
    def _calculate_max_error(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Maximum Error."""
        return float(max_error(y_true, y_pred))
    
    def _calculate_msle(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Squared Logarithmic Error."""
        # Ensure positive values
        y_true = np.maximum(y_true, 1e-10)
        y_pred = np.maximum(y_pred, 1e-10)
        
        return float(mean_squared_log_error(y_true, y_pred))
    
    def _calculate_rmsle(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Root Mean Squared Logarithmic Error."""
        msle = self._calculate_msle(y_true, y_pred)
        return float(np.sqrt(msle))
    
    def _calculate_tolerance_compliance(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                      tolerance_specs: Dict[str, Any]) -> float:
        """Calculate tolerance compliance percentage."""
        tolerance = tolerance_specs.get('tolerance', 0.1)
        tolerance_type = tolerance_specs.get('type', 'absolute')
        
        if tolerance_type == 'absolute':
            errors = np.abs(y_true - y_pred)
            compliant = errors <= tolerance
        elif tolerance_type == 'relative':
            errors = np.abs(y_true - y_pred) / np.abs(y_true)
            compliant = errors <= tolerance
        else:
            raise ValueError(f"Unknown tolerance type: {tolerance_type}")
        
        return float(np.mean(compliant) * 100)
    
    def _calculate_process_capability(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                    tolerance_specs: Dict[str, Any]) -> float:
        """Calculate process capability index (Cp)."""
        tolerance = tolerance_specs.get('tolerance', 0.1)
        tolerance_type = tolerance_specs.get('type', 'absolute')
        
        if tolerance_type == 'absolute':
            errors = np.abs(y_true - y_pred)
            usl = tolerance  # Upper specification limit
            lsl = -tolerance  # Lower specification limit
        elif tolerance_type == 'relative':
            errors = (y_pred - y_true) / y_true
            usl = tolerance
            lsl = -tolerance
        else:
            raise ValueError(f"Unknown tolerance type: {tolerance_type}")
        
        # Calculate Cp
        sigma = np.std(errors)
        if sigma == 0:
            return np.inf
        
        cp = (usl - lsl) / (6 * sigma)
        return float(cp)
    
    def _calculate_defect_rate(self, y_true: np.ndarray, y_pred: np.ndarray, 
                              tolerance_specs: Optional[Dict[str, Any]] = None) -> float:
        """Calculate defect rate based on prediction errors."""
        if tolerance_specs is None:
            # Use default tolerance
            tolerance = 0.1
            tolerance_type = 'absolute'
        else:
            tolerance = tolerance_specs.get('tolerance', 0.1)
            tolerance_type = tolerance_specs.get('type', 'absolute')
        
        if tolerance_type == 'absolute':
            errors = np.abs(y_true - y_pred)
            defects = errors > tolerance
        elif tolerance_type == 'relative':
            errors = np.abs(y_true - y_pred) / np.abs(y_true)
            defects = errors > tolerance
        else:
            raise ValueError(f"Unknown tolerance type: {tolerance_type}")
        
        return float(np.mean(defects) * 100)
    
    def _calculate_quality_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate overall quality score."""
        # Combine multiple metrics into a quality score
        mae = self._calculate_mae(y_true, y_pred)
        r2 = self._calculate_r2(y_true, y_pred)
        
        # Normalize MAE (assuming max acceptable error is 10% of mean)
        max_acceptable_error = np.mean(y_true) * 0.1
        normalized_mae = min(mae / max_acceptable_error, 1.0)
        
        # Calculate quality score (0-100)
        quality_score = (r2 * 0.6 + (1 - normalized_mae) * 0.4) * 100
        return float(max(0, min(100, quality_score)))
    
    def _calculate_dimensional_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                      tolerance_specs: Dict[str, Any]) -> float:
        """Calculate dimensional accuracy for manufacturing."""
        tolerance = tolerance_specs.get('tolerance', 0.01)  # Default 0.01mm
        errors = np.abs(y_true - y_pred)
        
        # Calculate accuracy as percentage within tolerance
        within_tolerance = errors <= tolerance
        accuracy = np.mean(within_tolerance) * 100
        
        return float(accuracy)
    
    def _calculate_surface_roughness_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate surface roughness prediction accuracy."""
        # Surface roughness is typically measured in micrometers
        errors = np.abs(y_true - y_pred)
        
        # Calculate accuracy based on typical surface roughness tolerances
        tolerance = 0.5  # 0.5 micrometers
        within_tolerance = errors <= tolerance
        accuracy = np.mean(within_tolerance) * 100
        
        return float(accuracy)
    
    def calculate_error_analysis(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """
        Perform detailed error analysis.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary with error analysis results
        """
        errors = y_pred - y_true
        absolute_errors = np.abs(errors)
        
        analysis = {
            'error_statistics': {
                'mean_error': float(np.mean(errors)),
                'std_error': float(np.std(errors)),
                'median_error': float(np.median(errors)),
                'min_error': float(np.min(errors)),
                'max_error': float(np.max(errors)),
                'q25_error': float(np.percentile(errors, 25)),
                'q75_error': float(np.percentile(errors, 75))
            },
            'absolute_error_statistics': {
                'mean_absolute_error': float(np.mean(absolute_errors)),
                'std_absolute_error': float(np.std(absolute_errors)),
                'median_absolute_error': float(np.median(absolute_errors)),
                'min_absolute_error': float(np.min(absolute_errors)),
                'max_absolute_error': float(np.max(absolute_errors)),
                'q25_absolute_error': float(np.percentile(absolute_errors, 25)),
                'q75_absolute_error': float(np.percentile(absolute_errors, 75))
            },
            'error_distribution': {
                'positive_errors': int(np.sum(errors > 0)),
                'negative_errors': int(np.sum(errors < 0)),
                'zero_errors': int(np.sum(errors == 0)),
                'positive_error_percentage': float(np.mean(errors > 0) * 100),
                'negative_error_percentage': float(np.mean(errors < 0) * 100)
            },
            'outlier_analysis': {
                'iqr': float(np.percentile(absolute_errors, 75) - np.percentile(absolute_errors, 25)),
                'outlier_threshold': float(np.percentile(absolute_errors, 75) + 1.5 * (np.percentile(absolute_errors, 75) - np.percentile(absolute_errors, 25))),
                'outlier_count': int(np.sum(absolute_errors > (np.percentile(absolute_errors, 75) + 1.5 * (np.percentile(absolute_errors, 75) - np.percentile(absolute_errors, 25))))),
                'outlier_percentage': float(np.mean(absolute_errors > (np.percentile(absolute_errors, 75) + 1.5 * (np.percentile(absolute_errors, 75) - np.percentile(absolute_errors, 25)))) * 100)
            }
        }
        
        return analysis
    
    def compare_models(self, model_results: Dict[str, Dict[str, float]], 
                      metric: str = 'rmse') -> Dict[str, Any]:
        """
        Compare multiple models based on a specific metric.
        
        Args:
            model_results: Dictionary mapping model names to their metrics
            metric: Metric to use for comparison
            
        Returns:
            Dictionary with comparison results
        """
        if not model_results:
            return {}
        
        # Extract metric values
        metric_values = {}
        for model_name, metrics in model_results.items():
            if metric in metrics:
                metric_values[model_name] = metrics[metric]
            else:
                logger.warning(f"Metric {metric} not found for model {model_name}")
        
        if not metric_values:
            return {}
        
        # Find best and worst models
        if metric in ['mse', 'rmse', 'mae', 'mape', 'smape', 'msle', 'rmsle', 'max_error']:
            # Lower is better
            best_model = min(metric_values, key=metric_values.get)
            worst_model = max(metric_values, key=metric_values.get)
        else:
            # Higher is better (r2, explained_variance, etc.)
            best_model = max(metric_values, key=metric_values.get)
            worst_model = min(metric_values, key=metric_values.get)
        
        comparison = {
            'metric_used': metric,
            'best_model': best_model,
            'best_score': metric_values[best_model],
            'worst_model': worst_model,
            'worst_score': metric_values[worst_model],
            'model_rankings': sorted(metric_values.items(), key=lambda x: x[1], reverse=(metric not in ['mse', 'rmse', 'mae', 'mape', 'smape', 'msle', 'rmsle', 'max_error'])),
            'score_range': float(max(metric_values.values()) - min(metric_values.values())),
            'score_std': float(np.std(list(metric_values.values())))
        }
        
        return comparison
    
    def visualize_predictions(self, y_true: np.ndarray, y_pred: np.ndarray, 
                            title: str = "Prediction vs Actual",
                            save_path: Optional[str] = None) -> None:
        """
        Visualize predictions against actual values.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            title: Plot title
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Scatter plot: Predicted vs Actual
        axes[0, 0].scatter(y_true, y_pred, alpha=0.6)
        axes[0, 0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('Actual Values')
        axes[0, 0].set_ylabel('Predicted Values')
        axes[0, 0].set_title('Predicted vs Actual')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Residuals plot
        residuals = y_pred - y_true
        axes[0, 1].scatter(y_pred, residuals, alpha=0.6)
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_xlabel('Predicted Values')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title('Residuals vs Predicted')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Error distribution
        axes[1, 0].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
        axes[1, 0].set_xlabel('Residuals')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Error Distribution')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title('Q-Q Plot of Residuals')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Prediction visualization saved to {save_path}")
        
        plt.show()
    
    def generate_report(self, y_true: np.ndarray, y_pred: np.ndarray, 
                       model_name: str = "Model",
                       manufacturing_metrics: bool = False,
                       tolerance_specs: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate a comprehensive evaluation report.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            model_name: Name of the model
            manufacturing_metrics: Whether to include manufacturing-specific metrics
            tolerance_specs: Tolerance specifications for manufacturing metrics
            
        Returns:
            Formatted evaluation report
        """
        # Calculate all metrics
        metrics = self.calculate_metrics(y_true, y_pred, manufacturing_metrics=manufacturing_metrics, tolerance_specs=tolerance_specs)
        
        # Perform error analysis
        error_analysis = self.calculate_error_analysis(y_true, y_pred)
        
        # Generate report
        report = []
        report.append("=" * 60)
        report.append(f"REGRESSION MODEL EVALUATION REPORT")
        report.append(f"Model: {model_name}")
        report.append("=" * 60)
        report.append("")
        
        # Basic metrics
        report.append("BASIC METRICS:")
        report.append(f"  Mean Squared Error (MSE): {metrics.get('mse', 'N/A'):.6f}")
        report.append(f"  Root Mean Squared Error (RMSE): {metrics.get('rmse', 'N/A'):.6f}")
        report.append(f"  Mean Absolute Error (MAE): {metrics.get('mae', 'N/A'):.6f}")
        report.append(f"  Mean Absolute Percentage Error (MAPE): {metrics.get('mape', 'N/A'):.2f}%")
        report.append(f"  R-squared (R²): {metrics.get('r2', 'N/A'):.6f}")
        report.append(f"  Explained Variance: {metrics.get('explained_variance', 'N/A'):.6f}")
        report.append("")
        
        # Manufacturing metrics
        if manufacturing_metrics:
            report.append("MANUFACTURING METRICS:")
            if 'tolerance_compliance' in metrics:
                report.append(f"  Tolerance Compliance: {metrics['tolerance_compliance']:.2f}%")
            if 'process_capability' in metrics:
                report.append(f"  Process Capability (Cp): {metrics['process_capability']:.2f}")
            if 'defect_rate' in metrics:
                report.append(f"  Defect Rate: {metrics['defect_rate']:.2f}%")
            if 'quality_score' in metrics:
                report.append(f"  Quality Score: {metrics['quality_score']:.2f}")
            report.append("")
        
        # Error analysis
        report.append("ERROR ANALYSIS:")
        error_stats = error_analysis['error_statistics']
        report.append(f"  Mean Error: {error_stats['mean_error']:.6f}")
        report.append(f"  Error Standard Deviation: {error_stats['std_error']:.6f}")
        report.append(f"  Median Error: {error_stats['median_error']:.6f}")
        report.append(f"  Error Range: [{error_stats['min_error']:.6f}, {error_stats['max_error']:.6f}]")
        report.append("")
        
        # Error distribution
        error_dist = error_analysis['error_distribution']
        report.append("ERROR DISTRIBUTION:")
        report.append(f"  Positive Errors: {error_dist['positive_errors']} ({error_dist['positive_error_percentage']:.2f}%)")
        report.append(f"  Negative Errors: {error_dist['negative_errors']} ({error_dist['negative_error_percentage']:.2f}%)")
        report.append(f"  Zero Errors: {error_dist['zero_errors']}")
        report.append("")
        
        # Outlier analysis
        outlier_analysis = error_analysis['outlier_analysis']
        report.append("OUTLIER ANALYSIS:")
        report.append(f"  Outlier Count: {outlier_analysis['outlier_count']}")
        report.append(f"  Outlier Percentage: {outlier_analysis['outlier_percentage']:.2f}%")
        report.append(f"  Outlier Threshold: {outlier_analysis['outlier_threshold']:.6f}")
        report.append("")
        
        # Model performance assessment
        report.append("MODEL PERFORMANCE ASSESSMENT:")
        r2 = metrics.get('r2', 0)
        rmse = metrics.get('rmse', float('inf'))
        
        if r2 > 0.9:
            report.append("  R² Score: EXCELLENT (>0.9)")
        elif r2 > 0.8:
            report.append("  R² Score: GOOD (0.8-0.9)")
        elif r2 > 0.7:
            report.append("  R² Score: FAIR (0.7-0.8)")
        else:
            report.append("  R² Score: POOR (<0.7)")
        
        if manufacturing_metrics and 'tolerance_compliance' in metrics:
            compliance = metrics['tolerance_compliance']
            if compliance > 95:
                report.append("  Tolerance Compliance: EXCELLENT (>95%)")
            elif compliance > 90:
                report.append("  Tolerance Compliance: GOOD (90-95%)")
            elif compliance > 80:
                report.append("  Tolerance Compliance: FAIR (80-90%)")
            else:
                report.append("  Tolerance Compliance: POOR (<80%)")
        
        report.append("")
        report.append("=" * 60)
        
        return "\n".join(report)
