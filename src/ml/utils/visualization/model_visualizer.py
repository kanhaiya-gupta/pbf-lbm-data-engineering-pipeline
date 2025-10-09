"""
Model Visualizer

This module implements visualization utilities for ML model performance
in PBF-LB/M manufacturing processes.
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional, List, Union, Tuple
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)


class ModelVisualizer:
    """
    Utility class for ML model visualization.
    
    This class handles:
    - Model performance visualization
    - Training history visualization
    - Model comparison visualization
    - Feature importance visualization
    - Model architecture visualization
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the model visualizer.
        
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
        
        logger.info("Initialized ModelVisualizer")
    
    def plot_training_history(self, history: Dict[str, List[float]], 
                            title: str = "Training History",
                            save_path: Optional[str] = None) -> None:
        """
        Plot training history for neural networks.
        
        Args:
            history: Dictionary with training history data
            title: Plot title
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plot
        if 'loss' in history and 'val_loss' in history:
            axes[0, 0].plot(history['loss'], label='Training Loss')
            axes[0, 0].plot(history['val_loss'], label='Validation Loss')
            axes[0, 0].set_title('Model Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy plot
        if 'accuracy' in history and 'val_accuracy' in history:
            axes[0, 1].plot(history['accuracy'], label='Training Accuracy')
            axes[0, 1].plot(history['val_accuracy'], label='Validation Accuracy')
            axes[0, 1].set_title('Model Accuracy')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Accuracy')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Learning rate plot
        if 'lr' in history:
            axes[1, 0].plot(history['lr'])
            axes[1, 0].set_title('Learning Rate')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Additional metrics
        if 'mae' in history and 'val_mae' in history:
            axes[1, 1].plot(history['mae'], label='Training MAE')
            axes[1, 1].plot(history['val_mae'], label='Validation MAE')
            axes[1, 1].set_title('Mean Absolute Error')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('MAE')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.default_settings['dpi'], bbox_inches='tight')
            logger.info(f"Training history plot saved to {save_path}")
        
        plt.show()
    
    def plot_model_comparison(self, model_results: Dict[str, Dict[str, float]], 
                            metric: str = 'accuracy',
                            title: str = "Model Comparison",
                            save_path: Optional[str] = None) -> None:
        """
        Plot model comparison results.
        
        Args:
            model_results: Dictionary mapping model names to their metrics
            metric: Metric to compare
            title: Plot title
            save_path: Path to save the plot
        """
        # Extract metric values
        model_names = list(model_results.keys())
        metric_values = [model_results[model].get(metric, 0) for model in model_names]
        
        # Create bar plot
        plt.figure(figsize=(12, 8))
        bars = plt.bar(model_names, metric_values, alpha=0.7)
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.title(title, fontsize=16)
        plt.xlabel('Models', fontsize=14)
        plt.ylabel(metric.replace('_', ' ').title(), fontsize=14)
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=self.default_settings['dpi'], bbox_inches='tight')
            logger.info(f"Model comparison plot saved to {save_path}")
        
        plt.show()
    
    def plot_feature_importance(self, feature_names: List[str], 
                              importance_values: List[float],
                              title: str = "Feature Importance",
                              top_n: Optional[int] = None,
                              save_path: Optional[str] = None) -> None:
        """
        Plot feature importance.
        
        Args:
            feature_names: List of feature names
            importance_values: List of importance values
            title: Plot title
            top_n: Number of top features to show
            save_path: Path to save the plot
        """
        # Create DataFrame
        df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_values
        })
        
        # Sort by importance
        df = df.sort_values('importance', ascending=True)
        
        # Select top N features
        if top_n is not None:
            df = df.tail(top_n)
        
        # Create horizontal bar plot
        plt.figure(figsize=(10, max(6, len(df) * 0.3)))
        bars = plt.barh(df['feature'], df['importance'], alpha=0.7)
        
        # Add value labels
        for bar, value in zip(bars, df['importance']):
            plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{value:.3f}', ha='left', va='center')
        
        plt.title(title, fontsize=16)
        plt.xlabel('Importance', fontsize=14)
        plt.ylabel('Features', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=self.default_settings['dpi'], bbox_inches='tight')
            logger.info(f"Feature importance plot saved to {save_path}")
        
        plt.show()
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                            class_names: Optional[List[str]] = None,
                            title: str = "Confusion Matrix",
                            normalize: bool = False,
                            save_path: Optional[str] = None) -> None:
        """
        Plot confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: List of class names
            title: Plot title
            normalize: Whether to normalize the confusion matrix
            save_path: Path to save the plot
        """
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
        else:
            fmt = 'd'
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        
        plt.title(title, fontsize=16)
        plt.xlabel('Predicted Label', fontsize=14)
        plt.ylabel('True Label', fontsize=14)
        
        if save_path:
            plt.savefig(save_path, dpi=self.default_settings['dpi'], bbox_inches='tight')
            logger.info(f"Confusion matrix plot saved to {save_path}")
        
        plt.show()
    
    def plot_roc_curve(self, y_true: np.ndarray, y_prob: np.ndarray, 
                      class_names: Optional[List[str]] = None,
                      title: str = "ROC Curve",
                      save_path: Optional[str] = None) -> None:
        """
        Plot ROC curve.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            class_names: List of class names
            title: Plot title
            save_path: Path to save the plot
        """
        from sklearn.metrics import roc_curve, auc
        
        plt.figure(figsize=(10, 8))
        
        if len(np.unique(y_true)) == 2:
            # Binary classification
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, color='darkorange', lw=2,
                    label=f'ROC curve (AUC = {roc_auc:.2f})')
        else:
            # Multi-class classification
            from sklearn.preprocessing import label_binarize
            
            y_true_bin = label_binarize(y_true, classes=np.unique(y_true))
            n_classes = y_true_bin.shape[1]
            
            for i in range(n_classes):
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
                roc_auc = auc(fpr, tpr)
                
                class_name = class_names[i] if class_names else f'Class {i}'
                plt.plot(fpr, tpr, lw=2,
                        label=f'{class_name} (AUC = {roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=14)
        plt.ylabel('True Positive Rate', fontsize=14)
        plt.title(title, fontsize=16)
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=self.default_settings['dpi'], bbox_inches='tight')
            logger.info(f"ROC curve plot saved to {save_path}")
        
        plt.show()
    
    def plot_precision_recall_curve(self, y_true: np.ndarray, y_prob: np.ndarray, 
                                  class_names: Optional[List[str]] = None,
                                  title: str = "Precision-Recall Curve",
                                  save_path: Optional[str] = None) -> None:
        """
        Plot precision-recall curve.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            class_names: List of class names
            title: Plot title
            save_path: Path to save the plot
        """
        from sklearn.metrics import precision_recall_curve, average_precision_score
        
        plt.figure(figsize=(10, 8))
        
        if len(np.unique(y_true)) == 2:
            # Binary classification
            precision, recall, _ = precision_recall_curve(y_true, y_prob)
            avg_precision = average_precision_score(y_true, y_prob)
            
            plt.plot(recall, precision, color='darkorange', lw=2,
                    label=f'PR curve (AP = {avg_precision:.2f})')
        else:
            # Multi-class classification
            from sklearn.preprocessing import label_binarize
            
            y_true_bin = label_binarize(y_true, classes=np.unique(y_true))
            n_classes = y_true_bin.shape[1]
            
            for i in range(n_classes):
                precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_prob[:, i])
                avg_precision = average_precision_score(y_true_bin[:, i], y_prob[:, i])
                
                class_name = class_names[i] if class_names else f'Class {i}'
                plt.plot(recall, precision, lw=2,
                        label=f'{class_name} (AP = {avg_precision:.2f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=14)
        plt.ylabel('Precision', fontsize=14)
        plt.title(title, fontsize=16)
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=self.default_settings['dpi'], bbox_inches='tight')
            logger.info(f"Precision-recall curve plot saved to {save_path}")
        
        plt.show()
    
    def plot_residuals(self, y_true: np.ndarray, y_pred: np.ndarray, 
                      title: str = "Residuals Plot",
                      save_path: Optional[str] = None) -> None:
        """
        Plot residuals for regression models.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            title: Plot title
            save_path: Path to save the plot
        """
        residuals = y_pred - y_true
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Residuals vs Predicted
        axes[0, 0].scatter(y_pred, residuals, alpha=0.6)
        axes[0, 0].axhline(y=0, color='r', linestyle='--')
        axes[0, 0].set_xlabel('Predicted Values')
        axes[0, 0].set_ylabel('Residuals')
        axes[0, 0].set_title('Residuals vs Predicted')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Residuals vs True
        axes[0, 1].scatter(y_true, residuals, alpha=0.6)
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_xlabel('True Values')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title('Residuals vs True')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Residuals histogram
        axes[1, 0].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
        axes[1, 0].set_xlabel('Residuals')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Residuals Distribution')
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
            logger.info(f"Residuals plot saved to {save_path}")
        
        plt.show()
    
    def plot_prediction_vs_actual(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                title: str = "Prediction vs Actual",
                                save_path: Optional[str] = None) -> None:
        """
        Plot predicted vs actual values.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            title: Plot title
            save_path: Path to save the plot
        """
        plt.figure(figsize=(10, 8))
        
        # Scatter plot
        plt.scatter(y_true, y_pred, alpha=0.6)
        
        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        # Calculate R²
        from sklearn.metrics import r2_score
        r2 = r2_score(y_true, y_pred)
        
        plt.xlabel('True Values', fontsize=14)
        plt.ylabel('Predicted Values', fontsize=14)
        plt.title(f'{title} (R² = {r2:.3f})', fontsize=16)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=self.default_settings['dpi'], bbox_inches='tight')
            logger.info(f"Prediction vs actual plot saved to {save_path}")
        
        plt.show()
    
    def plot_feature_correlation(self, features: np.ndarray, 
                               feature_names: Optional[List[str]] = None,
                               title: str = "Feature Correlation Matrix",
                               save_path: Optional[str] = None) -> None:
        """
        Plot feature correlation matrix.
        
        Args:
            features: Feature matrix
            feature_names: List of feature names
            title: Plot title
            save_path: Path to save the plot
        """
        # Calculate correlation matrix
        if isinstance(features, pd.DataFrame):
            corr_matrix = features.corr()
        else:
            corr_matrix = np.corrcoef(features.T)
        
        # Create heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                   xticklabels=feature_names, yticklabels=feature_names)
        
        plt.title(title, fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.default_settings['dpi'], bbox_inches='tight')
            logger.info(f"Feature correlation plot saved to {save_path}")
        
        plt.show()
    
    def plot_learning_curves(self, train_sizes: List[int], 
                           train_scores: List[float],
                           val_scores: List[float],
                           title: str = "Learning Curves",
                           save_path: Optional[str] = None) -> None:
        """
        Plot learning curves.
        
        Args:
            train_sizes: List of training set sizes
            train_scores: List of training scores
            val_scores: List of validation scores
            title: Plot title
            save_path: Path to save the plot
        """
        plt.figure(figsize=(10, 8))
        
        plt.plot(train_sizes, train_scores, 'o-', label='Training Score')
        plt.plot(train_sizes, val_scores, 'o-', label='Validation Score')
        
        plt.xlabel('Training Set Size', fontsize=14)
        plt.ylabel('Score', fontsize=14)
        plt.title(title, fontsize=16)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=self.default_settings['dpi'], bbox_inches='tight')
            logger.info(f"Learning curves plot saved to {save_path}")
        
        plt.show()
    
    def plot_model_architecture(self, model, title: str = "Model Architecture",
                              save_path: Optional[str] = None) -> None:
        """
        Plot model architecture (for neural networks).
        
        Args:
            model: Neural network model
            title: Plot title
            save_path: Path to save the plot
        """
        try:
            import tensorflow as tf
            from tensorflow.keras.utils import plot_model
            
            # Save model architecture
            if save_path:
                plot_model(model, to_file=save_path, show_shapes=True, show_layer_names=True)
                logger.info(f"Model architecture saved to {save_path}")
            else:
                # Display in notebook
                plot_model(model, show_shapes=True, show_layer_names=True)
                
        except ImportError:
            logger.warning("TensorFlow not available for model architecture plotting")
        except Exception as e:
            logger.error(f"Failed to plot model architecture: {e}")
    
    def create_interactive_dashboard(self, model_results: Dict[str, Dict[str, float]], 
                                   save_path: Optional[str] = None) -> None:
        """
        Create interactive dashboard for model results.
        
        Args:
            model_results: Dictionary with model results
            save_path: Path to save the dashboard
        """
        try:
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Model Comparison', 'Feature Importance', 'Training History', 'Confusion Matrix'),
                specs=[[{"type": "bar"}, {"type": "bar"}],
                       [{"type": "scatter"}, {"type": "heatmap"}]]
            )
            
            # Model comparison
            model_names = list(model_results.keys())
            accuracy_scores = [model_results[model].get('accuracy', 0) for model in model_names]
            
            fig.add_trace(
                go.Bar(x=model_names, y=accuracy_scores, name='Accuracy'),
                row=1, col=1
            )
            
            # Add more traces as needed
            
            fig.update_layout(height=800, showlegend=False, title_text="Model Performance Dashboard")
            
            if save_path:
                fig.write_html(save_path)
                logger.info(f"Interactive dashboard saved to {save_path}")
            else:
                fig.show()
                
        except ImportError:
            logger.warning("Plotly not available for interactive dashboard")
        except Exception as e:
            logger.error(f"Failed to create interactive dashboard: {e}")
    
    def plot_hyperparameter_tuning(self, param_grid: Dict[str, List], 
                                 scores: List[float],
                                 title: str = "Hyperparameter Tuning Results",
                                 save_path: Optional[str] = None) -> None:
        """
        Plot hyperparameter tuning results.
        
        Args:
            param_grid: Dictionary with parameter grid
            scores: List of scores for each parameter combination
            title: Plot title
            save_path: Path to save the plot
        """
        # Create parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        # Create grid of parameter combinations
        from itertools import product
        param_combinations = list(product(*param_values))
        
        # Create DataFrame
        df = pd.DataFrame(param_combinations, columns=param_names)
        df['score'] = scores
        
        # Plot for each parameter
        n_params = len(param_names)
        fig, axes = plt.subplots(1, n_params, figsize=(5*n_params, 5))
        
        if n_params == 1:
            axes = [axes]
        
        for i, param in enumerate(param_names):
            # Group by parameter and calculate mean score
            param_scores = df.groupby(param)['score'].mean()
            
            axes[i].plot(param_scores.index, param_scores.values, 'o-')
            axes[i].set_xlabel(param)
            axes[i].set_ylabel('Score')
            axes[i].set_title(f'Score vs {param}')
            axes[i].grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.default_settings['dpi'], bbox_inches='tight')
            logger.info(f"Hyperparameter tuning plot saved to {save_path}")
        
        plt.show()
    
    def plot_model_interpretation(self, model, X: np.ndarray, 
                                feature_names: Optional[List[str]] = None,
                                title: str = "Model Interpretation",
                                save_path: Optional[str] = None) -> None:
        """
        Plot model interpretation using SHAP or LIME.
        
        Args:
            model: Trained model
            X: Feature matrix
            feature_names: List of feature names
            title: Plot title
            save_path: Path to save the plot
        """
        try:
            import shap
            
            # Create SHAP explainer
            explainer = shap.Explainer(model)
            shap_values = explainer(X)
            
            # Plot SHAP summary
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
            plt.title(title, fontsize=16)
            
            if save_path:
                plt.savefig(save_path, dpi=self.default_settings['dpi'], bbox_inches='tight')
                logger.info(f"Model interpretation plot saved to {save_path}")
            
            plt.show()
            
        except ImportError:
            logger.warning("SHAP not available for model interpretation")
        except Exception as e:
            logger.error(f"Failed to create model interpretation plot: {e}")
    
    def save_all_plots(self, plots: Dict[str, Any], 
                      output_dir: str = "plots") -> None:
        """
        Save all plots to a directory.
        
        Args:
            plots: Dictionary with plot data
            output_dir: Output directory for plots
        """
        import os
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Save each plot
        for plot_name, plot_data in plots.items():
            save_path = os.path.join(output_dir, f"{plot_name}.png")
            
            try:
                if 'type' in plot_data:
                    if plot_data['type'] == 'training_history':
                        self.plot_training_history(plot_data['data'], save_path=save_path)
                    elif plot_data['type'] == 'model_comparison':
                        self.plot_model_comparison(plot_data['data'], save_path=save_path)
                    # Add more plot types as needed
                    
                logger.info(f"Saved {plot_name} to {save_path}")
                
            except Exception as e:
                logger.error(f"Failed to save {plot_name}: {e}")
    
    def generate_model_report(self, model_results: Dict[str, Any], 
                            output_path: str = "model_report.html") -> None:
        """
        Generate HTML report with all model visualizations.
        
        Args:
            model_results: Dictionary with model results
            output_path: Path to save the HTML report
        """
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            import plotly.offline as pyo
            
            # Create HTML report
            html_content = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Model Performance Report</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; }
                    h1, h2 { color: #333; }
                    .metric { margin: 10px 0; }
                    .plot { margin: 20px 0; }
                </style>
            </head>
            <body>
                <h1>Model Performance Report</h1>
            """
            
            # Add model results
            for model_name, results in model_results.items():
                html_content += f"<h2>{model_name}</h2>"
                for metric, value in results.items():
                    html_content += f"<div class='metric'><strong>{metric}:</strong> {value:.4f}</div>"
            
            html_content += """
            </body>
            </html>
            """
            
            # Save HTML report
            with open(output_path, 'w') as f:
                f.write(html_content)
            
            logger.info(f"Model report saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to generate model report: {e}")
