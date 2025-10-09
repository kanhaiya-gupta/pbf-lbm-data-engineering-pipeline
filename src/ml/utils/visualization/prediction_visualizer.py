"""
Prediction Visualizer

This module implements visualization utilities for prediction results
in PBF-LB/M manufacturing processes.
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional, List, Union, Tuple
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)


class PredictionVisualizer:
    """
    Utility class for prediction result visualization.
    
    This class handles:
    - Prediction accuracy visualization
    - Error analysis visualization
    - Prediction confidence visualization
    - Time series prediction visualization
    - Manufacturing-specific prediction visualization
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the prediction visualizer.
        
        Args:
            config: Configuration dictionary with visualization settings
        """
        self.config = config or {}
        
        # Default visualization settings
        self.default_settings = {
            'figure_size': (12, 8),
            'dpi': 300,
            'style': 'whitegrid',
            'color_palette': 'viridis',
            'font_size': 12,
            'title_size': 16,
            'label_size': 14
        }
        
        # Apply settings
        plt.style.use(self.default_settings['style'])
        sns.set_palette(self.default_settings['color_palette'])
        
        logger.info("Initialized PredictionVisualizer")
    
    def plot_prediction_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray, 
                               title: str = "Prediction Accuracy",
                               save_path: Optional[str] = None) -> None:
        """
        Plot prediction accuracy visualization.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            title: Plot title
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Scatter plot: Predicted vs Actual
        axes[0, 0].scatter(y_true, y_pred, alpha=0.6)
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        axes[0, 0].set_xlabel('True Values')
        axes[0, 0].set_ylabel('Predicted Values')
        axes[0, 0].set_title('Predicted vs Actual')
        axes[0, 0].legend()
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
            plt.savefig(save_path, dpi=self.default_settings['dpi'], bbox_inches='tight')
            logger.info(f"Prediction accuracy plot saved to {save_path}")
        
        plt.show()
    
    def plot_prediction_confidence(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                 confidence: np.ndarray,
                                 title: str = "Prediction Confidence",
                                 save_path: Optional[str] = None) -> None:
        """
        Plot prediction confidence visualization.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            confidence: Confidence scores
            title: Plot title
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Confidence vs Accuracy
        errors = np.abs(y_true - y_pred)
        axes[0, 0].scatter(confidence, errors, alpha=0.6)
        axes[0, 0].set_xlabel('Confidence Score')
        axes[0, 0].set_ylabel('Absolute Error')
        axes[0, 0].set_title('Confidence vs Accuracy')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Confidence distribution
        axes[0, 1].hist(confidence, bins=30, alpha=0.7, edgecolor='black')
        axes[0, 1].set_xlabel('Confidence Score')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Confidence Distribution')
        axes[0, 1].grid(True, alpha=0.3)
        
        # High confidence predictions
        high_conf_mask = confidence > np.percentile(confidence, 75)
        axes[1, 0].scatter(y_true[high_conf_mask], y_pred[high_conf_mask], 
                          alpha=0.6, label='High Confidence', color='green')
        axes[1, 0].scatter(y_true[~high_conf_mask], y_pred[~high_conf_mask], 
                          alpha=0.6, label='Low Confidence', color='red')
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        axes[1, 0].plot([min_val, max_val], [min_val, max_val], 'k--', lw=2)
        axes[1, 0].set_xlabel('True Values')
        axes[1, 0].set_ylabel('Predicted Values')
        axes[1, 0].set_title('High vs Low Confidence Predictions')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Confidence vs Error correlation
        from scipy.stats import pearsonr
        corr, p_value = pearsonr(confidence, errors)
        axes[1, 1].scatter(confidence, errors, alpha=0.6)
        axes[1, 1].set_xlabel('Confidence Score')
        axes[1, 1].set_ylabel('Absolute Error')
        axes[1, 1].set_title(f'Confidence-Error Correlation\nr={corr:.3f}, p={p_value:.3f}')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.default_settings['dpi'], bbox_inches='tight')
            logger.info(f"Prediction confidence plot saved to {save_path}")
        
        plt.show()
    
    def plot_time_series_predictions(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                   time_index: Optional[np.ndarray] = None,
                                   title: str = "Time Series Predictions",
                                   save_path: Optional[str] = None) -> None:
        """
        Plot time series predictions.
        
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
        
        # Prediction errors over time
        errors = y_pred - y_true
        axes[0, 1].plot(time_index, errors, alpha=0.7)
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_xlabel('Time')
        axes[0, 1].set_ylabel('Prediction Error')
        axes[0, 1].set_title('Prediction Errors Over Time')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Rolling error statistics
        window_size = min(50, len(errors) // 10)
        if window_size > 1:
            rolling_mean = pd.Series(errors).rolling(window=window_size).mean()
            rolling_std = pd.Series(errors).rolling(window=window_size).std()
            
            axes[1, 0].plot(time_index, rolling_mean, label='Rolling Mean', alpha=0.7)
            axes[1, 0].fill_between(time_index, 
                                   rolling_mean - rolling_std, 
                                   rolling_mean + rolling_std, 
                                   alpha=0.3, label='±1 Std')
            axes[1, 0].set_xlabel('Time')
            axes[1, 0].set_ylabel('Rolling Error')
            axes[1, 0].set_title('Rolling Error Statistics')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Error autocorrelation
        from scipy.stats import pearsonr
        if len(errors) > 1:
            autocorr = [pearsonr(errors[:-i], errors[i:])[0] for i in range(1, min(20, len(errors)))]
            axes[1, 1].plot(range(1, len(autocorr) + 1), autocorr, 'o-')
            axes[1, 1].set_xlabel('Lag')
            axes[1, 1].set_ylabel('Autocorrelation')
            axes[1, 1].set_title('Error Autocorrelation')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.default_settings['dpi'], bbox_inches='tight')
            logger.info(f"Time series predictions plot saved to {save_path}")
        
        plt.show()
    
    def plot_manufacturing_predictions(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                     prediction_type: str = "quality",
                                     specifications: Optional[Dict[str, Any]] = None,
                                     title: str = "Manufacturing Predictions",
                                     save_path: Optional[str] = None) -> None:
        """
        Plot manufacturing-specific predictions.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            prediction_type: Type of prediction (quality, dimensional, surface, etc.)
            specifications: Manufacturing specifications
            title: Plot title
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Prediction vs Actual with specifications
        axes[0, 0].scatter(y_true, y_pred, alpha=0.6)
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        # Add specification limits if available
        if specifications:
            tolerance = specifications.get('tolerance', 0.1)
            target = specifications.get('target', np.mean(y_true))
            
            axes[0, 0].axhline(y=target + tolerance, color='orange', linestyle='--', alpha=0.7, label='Upper Limit')
            axes[0, 0].axhline(y=target - tolerance, color='orange', linestyle='--', alpha=0.7, label='Lower Limit')
            axes[0, 0].axhline(y=target, color='green', linestyle='-', alpha=0.7, label='Target')
        
        axes[0, 0].set_xlabel('True Values')
        axes[0, 0].set_ylabel('Predicted Values')
        axes[0, 0].set_title(f'{prediction_type.title()} Prediction vs Actual')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Tolerance compliance
        if specifications:
            tolerance = specifications.get('tolerance', 0.1)
            target = specifications.get('target', np.mean(y_true))
            
            # Calculate compliance
            true_compliance = np.abs(y_true - target) <= tolerance
            pred_compliance = np.abs(y_pred - target) <= tolerance
            
            compliance_data = pd.DataFrame({
                'True': true_compliance,
                'Predicted': pred_compliance
            })
            
            compliance_counts = compliance_data.sum()
            axes[0, 1].bar(compliance_counts.index, compliance_counts.values, alpha=0.7)
            axes[0, 1].set_ylabel('Compliant Samples')
            axes[0, 1].set_title('Tolerance Compliance')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Error analysis by value range
        errors = np.abs(y_pred - y_true)
        value_ranges = pd.cut(y_true, bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
        error_by_range = pd.DataFrame({'Range': value_ranges, 'Error': errors}).groupby('Range')['Error'].mean()
        
        axes[1, 0].bar(error_by_range.index, error_by_range.values, alpha=0.7)
        axes[1, 0].set_xlabel('Value Range')
        axes[1, 0].set_ylabel('Mean Absolute Error')
        axes[1, 0].set_title('Error by Value Range')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Prediction quality metrics
        from sklearn.metrics import mean_absolute_error, r2_score
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        metrics_text = f'MAE: {mae:.4f}\nR²: {r2:.4f}'
        if specifications:
            tolerance = specifications.get('tolerance', 0.1)
            target = specifications.get('target', np.mean(y_true))
            compliance_rate = np.mean(np.abs(y_pred - target) <= tolerance) * 100
            metrics_text += f'\nCompliance: {compliance_rate:.1f}%'
        
        axes[1, 1].text(0.1, 0.5, metrics_text, transform=axes[1, 1].transAxes, 
                       fontsize=12, verticalalignment='center',
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].set_title('Prediction Quality Metrics')
        axes[1, 1].axis('off')
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.default_settings['dpi'], bbox_inches='tight')
            logger.info(f"Manufacturing predictions plot saved to {save_path}")
        
        plt.show()
    
    def plot_prediction_uncertainty(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                  uncertainty: np.ndarray,
                                  title: str = "Prediction Uncertainty",
                                  save_path: Optional[str] = None) -> None:
        """
        Plot prediction uncertainty visualization.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            uncertainty: Uncertainty estimates
            title: Plot title
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Uncertainty vs Error
        errors = np.abs(y_true - y_pred)
        axes[0, 0].scatter(uncertainty, errors, alpha=0.6)
        axes[0, 0].set_xlabel('Uncertainty')
        axes[0, 0].set_ylabel('Absolute Error')
        axes[0, 0].set_title('Uncertainty vs Error')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Uncertainty distribution
        axes[0, 1].hist(uncertainty, bins=30, alpha=0.7, edgecolor='black')
        axes[0, 1].set_xlabel('Uncertainty')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Uncertainty Distribution')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Predictions with uncertainty bands
        sorted_indices = np.argsort(y_pred)
        y_pred_sorted = y_pred[sorted_indices]
        y_true_sorted = y_true[sorted_indices]
        uncertainty_sorted = uncertainty[sorted_indices]
        
        axes[1, 0].scatter(y_pred_sorted, y_true_sorted, alpha=0.6, label='Predictions')
        axes[1, 0].fill_between(y_pred_sorted, 
                               y_pred_sorted - uncertainty_sorted, 
                               y_pred_sorted + uncertainty_sorted, 
                               alpha=0.3, label='Uncertainty Band')
        axes[1, 0].plot([y_pred_sorted.min(), y_pred_sorted.max()], 
                       [y_pred_sorted.min(), y_pred_sorted.max()], 'r--', lw=2, label='Perfect Prediction')
        axes[1, 0].set_xlabel('Predicted Values')
        axes[1, 0].set_ylabel('True Values')
        axes[1, 0].set_title('Predictions with Uncertainty Bands')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Uncertainty vs Prediction Value
        axes[1, 1].scatter(y_pred, uncertainty, alpha=0.6)
        axes[1, 1].set_xlabel('Predicted Values')
        axes[1, 1].set_ylabel('Uncertainty')
        axes[1, 1].set_title('Uncertainty vs Prediction Value')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.default_settings['dpi'], bbox_inches='tight')
            logger.info(f"Prediction uncertainty plot saved to {save_path}")
        
        plt.show()
    
    def plot_prediction_comparison(self, y_true: np.ndarray, 
                                 predictions: Dict[str, np.ndarray],
                                 title: str = "Prediction Comparison",
                                 save_path: Optional[str] = None) -> None:
        """
        Plot comparison of multiple prediction models.
        
        Args:
            y_true: True values
            predictions: Dictionary mapping model names to predictions
            title: Plot title
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # All predictions vs True
        axes[0, 0].scatter(y_true, y_true, alpha=0.6, label='Perfect Prediction', color='black')
        for model_name, y_pred in predictions.items():
            axes[0, 0].scatter(y_true, y_pred, alpha=0.6, label=model_name)
        
        axes[0, 0].set_xlabel('True Values')
        axes[0, 0].set_ylabel('Predicted Values')
        axes[0, 0].set_title('All Predictions vs True')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Error comparison
        errors = {}
        for model_name, y_pred in predictions.items():
            errors[model_name] = np.abs(y_pred - y_true)
        
        error_data = pd.DataFrame(errors)
        error_data.boxplot(ax=axes[0, 1])
        axes[0, 1].set_ylabel('Absolute Error')
        axes[0, 1].set_title('Error Distribution Comparison')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Model performance metrics
        from sklearn.metrics import mean_absolute_error, r2_score
        metrics_data = []
        for model_name, y_pred in predictions.items():
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            metrics_data.append({'Model': model_name, 'MAE': mae, 'R²': r2})
        
        metrics_df = pd.DataFrame(metrics_data)
        x_pos = np.arange(len(metrics_df))
        
        axes[1, 0].bar(x_pos, metrics_df['MAE'], alpha=0.7, label='MAE')
        axes[1, 0].set_xlabel('Models')
        axes[1, 0].set_ylabel('MAE')
        axes[1, 0].set_title('Mean Absolute Error Comparison')
        axes[1, 0].set_xticks(x_pos)
        axes[1, 0].set_xticklabels(metrics_df['Model'], rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].bar(x_pos, metrics_df['R²'], alpha=0.7, label='R²', color='orange')
        axes[1, 1].set_xlabel('Models')
        axes[1, 1].set_ylabel('R²')
        axes[1, 1].set_title('R² Score Comparison')
        axes[1, 1].set_xticks(x_pos)
        axes[1, 1].set_xticklabels(metrics_df['Model'], rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.default_settings['dpi'], bbox_inches='tight')
            logger.info(f"Prediction comparison plot saved to {save_path}")
        
        plt.show()
    
    def plot_prediction_errors_by_feature(self, X: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray,
                                        feature_names: Optional[List[str]] = None,
                                        title: str = "Prediction Errors by Feature",
                                        save_path: Optional[str] = None) -> None:
        """
        Plot prediction errors grouped by feature values.
        
        Args:
            X: Feature matrix
            y_true: True values
            y_pred: Predicted values
            feature_names: List of feature names
            title: Plot title
            save_path: Path to save the plot
        """
        errors = np.abs(y_pred - y_true)
        
        # Select top features by variance
        feature_vars = np.var(X, axis=0)
        top_features_idx = np.argsort(feature_vars)[-4:]  # Top 4 features
        
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, feature_idx in enumerate(top_features_idx):
            feature_values = X[:, feature_idx]
            feature_name = feature_names[feature_idx]
            
            # Create bins for feature values
            bins = pd.cut(feature_values, bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
            error_by_bin = pd.DataFrame({'Bin': bins, 'Error': errors}).groupby('Bin')['Error'].mean()
            
            axes[i].bar(error_by_bin.index, error_by_bin.values, alpha=0.7)
            axes[i].set_xlabel(feature_name)
            axes[i].set_ylabel('Mean Absolute Error')
            axes[i].set_title(f'Error by {feature_name}')
            axes[i].tick_params(axis='x', rotation=45)
            axes[i].grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.default_settings['dpi'], bbox_inches='tight')
            logger.info(f"Prediction errors by feature plot saved to {save_path}")
        
        plt.show()
    
    def create_prediction_dashboard(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                  confidence: Optional[np.ndarray] = None,
                                  uncertainty: Optional[np.ndarray] = None,
                                  time_index: Optional[np.ndarray] = None,
                                  save_path: Optional[str] = None) -> None:
        """
        Create comprehensive prediction dashboard.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            confidence: Confidence scores (optional)
            uncertainty: Uncertainty estimates (optional)
            time_index: Time index for time series (optional)
            save_path: Path to save the dashboard
        """
        try:
            # Create interactive dashboard using Plotly
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Prediction vs Actual', 'Error Analysis', 
                              'Confidence Analysis', 'Time Series (if available)'),
                specs=[[{"type": "scatter"}, {"type": "histogram"}],
                       [{"type": "scatter"}, {"type": "scatter"}]]
            )
            
            # Prediction vs Actual
            fig.add_trace(
                go.Scatter(x=y_true, y=y_pred, mode='markers', name='Predictions', opacity=0.6),
                row=1, col=1
            )
            
            # Perfect prediction line
            min_val = min(y_true.min(), y_pred.min())
            max_val = max(y_true.max(), y_pred.max())
            fig.add_trace(
                go.Scatter(x=[min_val, max_val], y=[min_val, max_val], 
                          mode='lines', name='Perfect Prediction', line=dict(dash='dash')),
                row=1, col=1
            )
            
            # Error analysis
            errors = y_pred - y_true
            fig.add_trace(
                go.Histogram(x=errors, name='Error Distribution'),
                row=1, col=2
            )
            
            # Confidence analysis
            if confidence is not None:
                fig.add_trace(
                    go.Scatter(x=confidence, y=np.abs(errors), mode='markers', 
                              name='Confidence vs Error', opacity=0.6),
                    row=2, col=1
                )
            
            # Time series
            if time_index is not None:
                fig.add_trace(
                    go.Scatter(x=time_index, y=y_true, mode='lines', name='True', opacity=0.7),
                    row=2, col=2
                )
                fig.add_trace(
                    go.Scatter(x=time_index, y=y_pred, mode='lines', name='Predicted', opacity=0.7),
                    row=2, col=2
                )
            
            fig.update_layout(height=800, showlegend=True, title_text="Prediction Analysis Dashboard")
            
            if save_path:
                fig.write_html(save_path)
                logger.info(f"Prediction dashboard saved to {save_path}")
            else:
                fig.show()
                
        except ImportError:
            logger.warning("Plotly not available for interactive dashboard")
        except Exception as e:
            logger.error(f"Failed to create prediction dashboard: {e}")
    
    def save_all_prediction_plots(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                confidence: Optional[np.ndarray] = None,
                                uncertainty: Optional[np.ndarray] = None,
                                time_index: Optional[np.ndarray] = None,
                                output_dir: str = "prediction_plots") -> None:
        """
        Save all prediction analysis plots to a directory.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            confidence: Confidence scores (optional)
            uncertainty: Uncertainty estimates (optional)
            time_index: Time index for time series (optional)
            output_dir: Output directory for plots
        """
        import os
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Save each plot
        plots = [
            ('prediction_accuracy', lambda: self.plot_prediction_accuracy(y_true, y_pred)),
        ]
        
        if confidence is not None:
            plots.append(('prediction_confidence', lambda: self.plot_prediction_confidence(y_true, y_pred, confidence)))
        
        if uncertainty is not None:
            plots.append(('prediction_uncertainty', lambda: self.plot_prediction_uncertainty(y_true, y_pred, uncertainty)))
        
        if time_index is not None:
            plots.append(('time_series_predictions', lambda: self.plot_time_series_predictions(y_true, y_pred, time_index)))
        
        for plot_name, plot_func in plots:
            save_path = os.path.join(output_dir, f"{plot_name}.png")
            
            try:
                plot_func()
                plt.savefig(save_path, dpi=self.default_settings['dpi'], bbox_inches='tight')
                plt.close()
                logger.info(f"Saved {plot_name} to {save_path}")
                
            except Exception as e:
                logger.error(f"Failed to save {plot_name}: {e}")
    
    def generate_prediction_report(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                 confidence: Optional[np.ndarray] = None,
                                 uncertainty: Optional[np.ndarray] = None,
                                 output_path: str = "prediction_report.html") -> None:
        """
        Generate HTML report with all prediction analysis.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            confidence: Confidence scores (optional)
            uncertainty: Uncertainty estimates (optional)
            output_path: Path to save the HTML report
        """
        try:
            # Calculate metrics
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
            
            mae = mean_absolute_error(y_true, y_pred)
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_true, y_pred)
            
            # Generate HTML report
            html_content = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Prediction Analysis Report</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; }
                    h1, h2 { color: #333; }
                    .metric { margin: 10px 0; }
                    .plot { margin: 20px 0; }
                    table { border-collapse: collapse; width: 100%; }
                    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                    th { background-color: #f2f2f2; }
                </style>
            </head>
            <body>
                <h1>Prediction Analysis Report</h1>
            """
            
            # Add prediction metrics
            html_content += "<h2>Prediction Metrics</h2>"
            html_content += f"<div class='metric'><strong>Mean Absolute Error:</strong> {mae:.4f}</div>"
            html_content += f"<div class='metric'><strong>Mean Squared Error:</strong> {mse:.4f}</div>"
            html_content += f"<div class='metric'><strong>Root Mean Squared Error:</strong> {rmse:.4f}</div>"
            html_content += f"<div class='metric'><strong>R² Score:</strong> {r2:.4f}</div>"
            
            # Add confidence analysis if available
            if confidence is not None:
                html_content += "<h2>Confidence Analysis</h2>"
                html_content += f"<div class='metric'><strong>Mean Confidence:</strong> {np.mean(confidence):.4f}</div>"
                html_content += f"<div class='metric'><strong>Confidence Std:</strong> {np.std(confidence):.4f}</div>"
            
            # Add uncertainty analysis if available
            if uncertainty is not None:
                html_content += "<h2>Uncertainty Analysis</h2>"
                html_content += f"<div class='metric'><strong>Mean Uncertainty:</strong> {np.mean(uncertainty):.4f}</div>"
                html_content += f"<div class='metric'><strong>Uncertainty Std:</strong> {np.std(uncertainty):.4f}</div>"
            
            html_content += """
            </body>
            </html>
            """
            
            # Save HTML report
            with open(output_path, 'w') as f:
                f.write(html_content)
            
            logger.info(f"Prediction report saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to generate prediction report: {e}")
