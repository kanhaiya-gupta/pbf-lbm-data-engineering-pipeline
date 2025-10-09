"""
Feature Visualizer

This module implements visualization utilities for feature analysis
in PBF-LB/M manufacturing processes.
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional, List, Union, Tuple
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)


class FeatureVisualizer:
    """
    Utility class for feature analysis and visualization.
    
    This class handles:
    - Feature distribution visualization
    - Feature correlation analysis
    - Feature importance visualization
    - Dimensionality reduction visualization
    - Feature engineering visualization
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the feature visualizer.
        
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
        
        logger.info("Initialized FeatureVisualizer")
    
    def plot_feature_distributions(self, data: Union[np.ndarray, pd.DataFrame], 
                                 feature_names: Optional[List[str]] = None,
                                 title: str = "Feature Distributions",
                                 save_path: Optional[str] = None) -> None:
        """
        Plot distributions of all features.
        
        Args:
            data: Feature data
            feature_names: List of feature names
            title: Plot title
            save_path: Path to save the plot
        """
        if isinstance(data, pd.DataFrame):
            df = data
            feature_names = feature_names or df.columns.tolist()
        else:
            df = pd.DataFrame(data, columns=feature_names or [f'Feature_{i}' for i in range(data.shape[1])])
            feature_names = df.columns.tolist()
        
        # Calculate number of subplots
        n_features = len(feature_names)
        n_cols = min(4, n_features)
        n_rows = (n_features + n_cols - 1) // n_cols
        
        # Create subplots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()
        
        # Plot each feature
        for i, feature in enumerate(feature_names):
            if i < len(axes):
                # Check if feature is numeric
                if pd.api.types.is_numeric_dtype(df[feature]):
                    # Histogram for numeric features
                    axes[i].hist(df[feature].dropna(), bins=30, alpha=0.7, edgecolor='black')
                    axes[i].set_title(f'{feature}')
                    axes[i].set_xlabel('Value')
                    axes[i].set_ylabel('Frequency')
                else:
                    # Bar plot for categorical features
                    value_counts = df[feature].value_counts().head(10)
                    axes[i].bar(range(len(value_counts)), value_counts.values)
                    axes[i].set_title(f'{feature}')
                    axes[i].set_xlabel('Categories')
                    axes[i].set_ylabel('Count')
                    axes[i].set_xticks(range(len(value_counts)))
                    axes[i].set_xticklabels(value_counts.index, rotation=45)
                
                axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_features, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.default_settings['dpi'], bbox_inches='tight')
            logger.info(f"Feature distributions plot saved to {save_path}")
        
        plt.show()
    
    def plot_feature_correlation(self, data: Union[np.ndarray, pd.DataFrame], 
                               feature_names: Optional[List[str]] = None,
                               title: str = "Feature Correlation Matrix",
                               save_path: Optional[str] = None) -> None:
        """
        Plot feature correlation matrix.
        
        Args:
            data: Feature data
            feature_names: List of feature names
            title: Plot title
            save_path: Path to save the plot
        """
        if isinstance(data, pd.DataFrame):
            df = data
            feature_names = feature_names or df.columns.tolist()
        else:
            df = pd.DataFrame(data, columns=feature_names or [f'Feature_{i}' for i in range(data.shape[1])])
            feature_names = df.columns.tolist()
        
        # Calculate correlation matrix
        corr_matrix = df.corr()
        
        # Create heatmap
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        
        plt.title(title, fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.default_settings['dpi'], bbox_inches='tight')
            logger.info(f"Feature correlation plot saved to {save_path}")
        
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
    
    def plot_feature_vs_target(self, X: Union[np.ndarray, pd.DataFrame], 
                             y: Union[np.ndarray, pd.Series],
                             feature_names: Optional[List[str]] = None,
                             target_name: str = "Target",
                             title: str = "Features vs Target",
                             save_path: Optional[str] = None) -> None:
        """
        Plot features against target variable.
        
        Args:
            X: Feature data
            y: Target variable
            feature_names: List of feature names
            target_name: Name of target variable
            title: Plot title
            save_path: Path to save the plot
        """
        if isinstance(X, pd.DataFrame):
            df = X.copy()
            feature_names = feature_names or df.columns.tolist()
        else:
            df = pd.DataFrame(X, columns=feature_names or [f'Feature_{i}' for i in range(X.shape[1])])
            feature_names = df.columns.tolist()
        
        df[target_name] = y
        
        # Calculate number of subplots
        n_features = len(feature_names)
        n_cols = min(3, n_features)
        n_rows = (n_features + n_cols - 1) // n_cols
        
        # Create subplots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()
        
        # Plot each feature against target
        for i, feature in enumerate(feature_names):
            if i < len(axes):
                # Check if target is numeric
                if pd.api.types.is_numeric_dtype(df[target_name]):
                    # Scatter plot for numeric target
                    axes[i].scatter(df[feature], df[target_name], alpha=0.6)
                    axes[i].set_xlabel(feature)
                    axes[i].set_ylabel(target_name)
                    
                    # Add trend line
                    z = np.polyfit(df[feature].dropna(), df[target_name].dropna(), 1)
                    p = np.poly1d(z)
                    axes[i].plot(df[feature], p(df[feature]), "r--", alpha=0.8)
                else:
                    # Box plot for categorical target
                    sns.boxplot(data=df, x=target_name, y=feature, ax=axes[i])
                
                axes[i].set_title(f'{feature} vs {target_name}')
                axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_features, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.default_settings['dpi'], bbox_inches='tight')
            logger.info(f"Features vs target plot saved to {save_path}")
        
        plt.show()
    
    def plot_dimensionality_reduction(self, X: np.ndarray, 
                                    y: Optional[np.ndarray] = None,
                                    method: str = 'pca',
                                    n_components: int = 2,
                                    feature_names: Optional[List[str]] = None,
                                    title: str = "Dimensionality Reduction",
                                    save_path: Optional[str] = None) -> None:
        """
        Plot dimensionality reduction results.
        
        Args:
            X: Feature data
            y: Target variable (optional)
            method: Dimensionality reduction method ('pca', 'tsne')
            n_components: Number of components
            feature_names: List of feature names
            title: Plot title
            save_path: Path to save the plot
        """
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply dimensionality reduction
        if method.lower() == 'pca':
            reducer = PCA(n_components=n_components)
            X_reduced = reducer.fit_transform(X_scaled)
            
            # Get explained variance ratio
            explained_variance = reducer.explained_variance_ratio_
            
        elif method.lower() == 'tsne':
            reducer = TSNE(n_components=n_components, random_state=42)
            X_reduced = reducer.fit_transform(X_scaled)
            explained_variance = None
            
        else:
            raise ValueError(f"Unknown dimensionality reduction method: {method}")
        
        # Create plot
        plt.figure(figsize=(10, 8))
        
        if y is not None:
            # Color by target variable
            if len(np.unique(y)) <= 10:
                # Categorical target
                scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap='viridis', alpha=0.7)
                plt.colorbar(scatter)
            else:
                # Numeric target
                scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap='viridis', alpha=0.7)
                plt.colorbar(scatter)
        else:
            # No target variable
            plt.scatter(X_reduced[:, 0], X_reduced[:, 1], alpha=0.7)
        
        plt.xlabel(f'{method.upper()} Component 1', fontsize=14)
        plt.ylabel(f'{method.upper()} Component 2', fontsize=14)
        
        if explained_variance is not None:
            plt.title(f'{title}\nExplained Variance: {explained_variance[0]:.2%}, {explained_variance[1]:.2%}', fontsize=16)
        else:
            plt.title(title, fontsize=16)
        
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=self.default_settings['dpi'], bbox_inches='tight')
            logger.info(f"Dimensionality reduction plot saved to {save_path}")
        
        plt.show()
    
    def plot_feature_engineering(self, X_original: np.ndarray, 
                               X_engineered: np.ndarray,
                               feature_names_original: Optional[List[str]] = None,
                               feature_names_engineered: Optional[List[str]] = None,
                               title: str = "Feature Engineering Comparison",
                               save_path: Optional[str] = None) -> None:
        """
        Plot feature engineering comparison.
        
        Args:
            X_original: Original feature data
            X_engineered: Engineered feature data
            feature_names_original: Original feature names
            feature_names_engineered: Engineered feature names
            title: Plot title
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Original features distribution
        axes[0, 0].hist(X_original.flatten(), bins=50, alpha=0.7, label='Original')
        axes[0, 0].set_title('Original Features Distribution')
        axes[0, 0].set_xlabel('Value')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Engineered features distribution
        axes[0, 1].hist(X_engineered.flatten(), bins=50, alpha=0.7, label='Engineered', color='orange')
        axes[0, 1].set_title('Engineered Features Distribution')
        axes[0, 1].set_xlabel('Value')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Feature count comparison
        feature_counts = [X_original.shape[1], X_engineered.shape[1]]
        feature_labels = ['Original', 'Engineered']
        axes[1, 0].bar(feature_labels, feature_counts, alpha=0.7)
        axes[1, 0].set_title('Feature Count Comparison')
        axes[1, 0].set_ylabel('Number of Features')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Variance comparison
        original_variance = np.var(X_original, axis=0)
        engineered_variance = np.var(X_engineered, axis=0)
        
        axes[1, 1].scatter(original_variance, engineered_variance, alpha=0.7)
        axes[1, 1].plot([0, max(original_variance.max(), engineered_variance.max())], 
                       [0, max(original_variance.max(), engineered_variance.max())], 
                       'r--', alpha=0.8)
        axes[1, 1].set_xlabel('Original Features Variance')
        axes[1, 1].set_ylabel('Engineered Features Variance')
        axes[1, 1].set_title('Variance Comparison')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.default_settings['dpi'], bbox_inches='tight')
            logger.info(f"Feature engineering plot saved to {save_path}")
        
        plt.show()
    
    def plot_feature_selection(self, feature_names: List[str], 
                             selection_scores: List[float],
                             selection_method: str = "Feature Selection",
                             title: str = "Feature Selection Results",
                             save_path: Optional[str] = None) -> None:
        """
        Plot feature selection results.
        
        Args:
            feature_names: List of feature names
            selection_scores: List of selection scores
            selection_method: Name of selection method
            title: Plot title
            save_path: Path to save the plot
        """
        # Create DataFrame
        df = pd.DataFrame({
            'feature': feature_names,
            'score': selection_scores
        })
        
        # Sort by score
        df = df.sort_values('score', ascending=True)
        
        # Create horizontal bar plot
        plt.figure(figsize=(10, max(6, len(df) * 0.3)))
        bars = plt.barh(df['feature'], df['score'], alpha=0.7)
        
        # Add value labels
        for bar, value in zip(bars, df['score']):
            plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{value:.3f}', ha='left', va='center')
        
        plt.title(f'{title}\nMethod: {selection_method}', fontsize=16)
        plt.xlabel('Selection Score', fontsize=14)
        plt.ylabel('Features', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=self.default_settings['dpi'], bbox_inches='tight')
            logger.info(f"Feature selection plot saved to {save_path}")
        
        plt.show()
    
    def plot_feature_interactions(self, X: Union[np.ndarray, pd.DataFrame], 
                                feature_names: Optional[List[str]] = None,
                                title: str = "Feature Interactions",
                                save_path: Optional[str] = None) -> None:
        """
        Plot feature interactions.
        
        Args:
            X: Feature data
            feature_names: List of feature names
            title: Plot title
            save_path: Path to save the plot
        """
        if isinstance(X, pd.DataFrame):
            df = X
            feature_names = feature_names or df.columns.tolist()
        else:
            df = pd.DataFrame(X, columns=feature_names or [f'Feature_{i}' for i in range(X.shape[1])])
            feature_names = df.columns.tolist()
        
        # Calculate interaction matrix
        interaction_matrix = np.zeros((len(feature_names), len(feature_names)))
        
        for i, feature1 in enumerate(feature_names):
            for j, feature2 in enumerate(feature_names):
                if i != j:
                    # Calculate correlation between features
                    corr = df[feature1].corr(df[feature2])
                    interaction_matrix[i, j] = abs(corr) if not np.isnan(corr) else 0
        
        # Create heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(interaction_matrix, annot=True, cmap='viridis',
                   xticklabels=feature_names, yticklabels=feature_names)
        
        plt.title(title, fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.default_settings['dpi'], bbox_inches='tight')
            logger.info(f"Feature interactions plot saved to {save_path}")
        
        plt.show()
    
    def plot_feature_statistics(self, data: Union[np.ndarray, pd.DataFrame], 
                              feature_names: Optional[List[str]] = None,
                              title: str = "Feature Statistics",
                              save_path: Optional[str] = None) -> None:
        """
        Plot feature statistics summary.
        
        Args:
            data: Feature data
            feature_names: List of feature names
            title: Plot title
            save_path: Path to save the plot
        """
        if isinstance(data, pd.DataFrame):
            df = data
            feature_names = feature_names or df.columns.tolist()
        else:
            df = pd.DataFrame(data, columns=feature_names or [f'Feature_{i}' for i in range(data.shape[1])])
            feature_names = df.columns.tolist()
        
        # Calculate statistics
        stats = df.describe()
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Mean values
        axes[0, 0].bar(range(len(feature_names)), stats.loc['mean'])
        axes[0, 0].set_title('Mean Values')
        axes[0, 0].set_xlabel('Features')
        axes[0, 0].set_ylabel('Mean')
        axes[0, 0].set_xticks(range(len(feature_names)))
        axes[0, 0].set_xticklabels(feature_names, rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Standard deviation
        axes[0, 1].bar(range(len(feature_names)), stats.loc['std'])
        axes[0, 1].set_title('Standard Deviation')
        axes[0, 1].set_xlabel('Features')
        axes[0, 1].set_ylabel('Std')
        axes[0, 1].set_xticks(range(len(feature_names)))
        axes[0, 1].set_xticklabels(feature_names, rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Missing values
        missing_values = df.isnull().sum()
        axes[1, 0].bar(range(len(feature_names)), missing_values)
        axes[1, 0].set_title('Missing Values')
        axes[1, 0].set_xlabel('Features')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_xticks(range(len(feature_names)))
        axes[1, 0].set_xticklabels(feature_names, rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Data types
        data_types = df.dtypes.value_counts()
        axes[1, 1].pie(data_types.values, labels=data_types.index, autopct='%1.1f%%')
        axes[1, 1].set_title('Data Types Distribution')
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.default_settings['dpi'], bbox_inches='tight')
            logger.info(f"Feature statistics plot saved to {save_path}")
        
        plt.show()
    
    def create_feature_dashboard(self, data: Union[np.ndarray, pd.DataFrame], 
                               y: Optional[Union[np.ndarray, pd.Series]] = None,
                               feature_names: Optional[List[str]] = None,
                               save_path: Optional[str] = None) -> None:
        """
        Create comprehensive feature analysis dashboard.
        
        Args:
            data: Feature data
            y: Target variable (optional)
            feature_names: List of feature names
            save_path: Path to save the dashboard
        """
        try:
            # Create interactive dashboard using Plotly
            if isinstance(data, pd.DataFrame):
                df = data
                feature_names = feature_names or df.columns.tolist()
            else:
                df = pd.DataFrame(data, columns=feature_names or [f'Feature_{i}' for i in range(data.shape[1])])
                feature_names = df.columns.tolist()
            
            if y is not None:
                df['target'] = y
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Feature Distributions', 'Feature Correlations', 
                              'Feature vs Target', 'Feature Statistics'),
                specs=[[{"type": "histogram"}, {"type": "heatmap"}],
                       [{"type": "scatter"}, {"type": "bar"}]]
            )
            
            # Feature distributions
            for feature in feature_names[:5]:  # Show first 5 features
                fig.add_trace(
                    go.Histogram(x=df[feature], name=feature, opacity=0.7),
                    row=1, col=1
                )
            
            # Feature correlations
            corr_matrix = df[feature_names].corr()
            fig.add_trace(
                go.Heatmap(z=corr_matrix.values, x=corr_matrix.columns, y=corr_matrix.index),
                row=1, col=2
            )
            
            # Feature vs target (if available)
            if y is not None:
                for feature in feature_names[:3]:  # Show first 3 features
                    fig.add_trace(
                        go.Scatter(x=df[feature], y=df['target'], mode='markers', name=feature),
                        row=2, col=1
                    )
            
            # Feature statistics
            stats = df[feature_names].describe()
            fig.add_trace(
                go.Bar(x=feature_names, y=stats.loc['mean'], name='Mean'),
                row=2, col=2
            )
            
            fig.update_layout(height=800, showlegend=False, title_text="Feature Analysis Dashboard")
            
            if save_path:
                fig.write_html(save_path)
                logger.info(f"Feature dashboard saved to {save_path}")
            else:
                fig.show()
                
        except ImportError:
            logger.warning("Plotly not available for interactive dashboard")
        except Exception as e:
            logger.error(f"Failed to create feature dashboard: {e}")
    
    def save_all_feature_plots(self, data: Union[np.ndarray, pd.DataFrame], 
                             y: Optional[Union[np.ndarray, pd.Series]] = None,
                             feature_names: Optional[List[str]] = None,
                             output_dir: str = "feature_plots") -> None:
        """
        Save all feature analysis plots to a directory.
        
        Args:
            data: Feature data
            y: Target variable (optional)
            feature_names: List of feature names
            output_dir: Output directory for plots
        """
        import os
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Save each plot
        plots = [
            ('feature_distributions', lambda: self.plot_feature_distributions(data, feature_names)),
            ('feature_correlation', lambda: self.plot_feature_correlation(data, feature_names)),
            ('feature_statistics', lambda: self.plot_feature_statistics(data, feature_names)),
            ('feature_interactions', lambda: self.plot_feature_interactions(data, feature_names))
        ]
        
        if y is not None:
            plots.append(('feature_vs_target', lambda: self.plot_feature_vs_target(data, y, feature_names)))
        
        for plot_name, plot_func in plots:
            save_path = os.path.join(output_dir, f"{plot_name}.png")
            
            try:
                plot_func()
                plt.savefig(save_path, dpi=self.default_settings['dpi'], bbox_inches='tight')
                plt.close()
                logger.info(f"Saved {plot_name} to {save_path}")
                
            except Exception as e:
                logger.error(f"Failed to save {plot_name}: {e}")
    
    def generate_feature_report(self, data: Union[np.ndarray, pd.DataFrame], 
                              y: Optional[Union[np.ndarray, pd.Series]] = None,
                              feature_names: Optional[List[str]] = None,
                              output_path: str = "feature_report.html") -> None:
        """
        Generate HTML report with all feature analysis.
        
        Args:
            data: Feature data
            y: Target variable (optional)
            feature_names: List of feature names
            output_path: Path to save the HTML report
        """
        try:
            if isinstance(data, pd.DataFrame):
                df = data
                feature_names = feature_names or df.columns.tolist()
            else:
                df = pd.DataFrame(data, columns=feature_names or [f'Feature_{i}' for i in range(data.shape[1])])
                feature_names = df.columns.tolist()
            
            # Generate HTML report
            html_content = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Feature Analysis Report</title>
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
                <h1>Feature Analysis Report</h1>
            """
            
            # Add feature statistics
            html_content += "<h2>Feature Statistics</h2>"
            stats = df.describe()
            html_content += stats.to_html()
            
            # Add missing values
            html_content += "<h2>Missing Values</h2>"
            missing_values = df.isnull().sum()
            html_content += missing_values.to_frame('Missing Count').to_html()
            
            # Add data types
            html_content += "<h2>Data Types</h2>"
            data_types = df.dtypes.value_counts()
            html_content += data_types.to_frame('Count').to_html()
            
            html_content += """
            </body>
            </html>
            """
            
            # Save HTML report
            with open(output_path, 'w') as f:
                f.write(html_content)
            
            logger.info(f"Feature report saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to generate feature report: {e}")
