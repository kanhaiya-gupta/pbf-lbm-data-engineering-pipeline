"""
Performance Visualizer

This module implements visualization utilities for performance monitoring
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


class PerformanceVisualizer:
    """
    Utility class for performance monitoring visualization.
    
    This class handles:
    - Model performance monitoring
    - System performance visualization
    - Manufacturing KPIs visualization
    - Performance trend analysis
    - Performance comparison visualization
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the performance visualizer.
        
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
        
        logger.info("Initialized PerformanceVisualizer")
    
    def plot_performance_metrics(self, metrics_data: Dict[str, List[float]], 
                               time_index: Optional[List] = None,
                               title: str = "Performance Metrics",
                               save_path: Optional[str] = None) -> None:
        """
        Plot performance metrics over time.
        
        Args:
            metrics_data: Dictionary mapping metric names to their values over time
            time_index: Time index for x-axis
            title: Plot title
            save_path: Path to save the plot
        """
        if time_index is None:
            time_index = range(len(list(metrics_data.values())[0]))
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        # Plot each metric
        for i, (metric_name, values) in enumerate(metrics_data.items()):
            if i < len(axes):
                axes[i].plot(time_index, values, marker='o', linewidth=2, markersize=4)
                axes[i].set_title(metric_name.replace('_', ' ').title())
                axes[i].set_xlabel('Time')
                axes[i].set_ylabel('Value')
                axes[i].grid(True, alpha=0.3)
                
                # Add trend line
                z = np.polyfit(range(len(values)), values, 1)
                p = np.poly1d(z)
                axes[i].plot(time_index, p(range(len(values))), 'r--', alpha=0.8, label='Trend')
                axes[i].legend()
        
        # Hide unused subplots
        for i in range(len(metrics_data), len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.default_settings['dpi'], bbox_inches='tight')
            logger.info(f"Performance metrics plot saved to {save_path}")
        
        plt.show()
    
    def plot_manufacturing_kpis(self, kpi_data: Dict[str, Dict[str, float]], 
                              title: str = "Manufacturing KPIs",
                              save_path: Optional[str] = None) -> None:
        """
        Plot manufacturing KPIs.
        
        Args:
            kpi_data: Dictionary with KPI categories and their values
            title: Plot title
            save_path: Path to save the plot
        """
        # Extract KPI categories and values
        categories = list(kpi_data.keys())
        n_categories = len(categories)
        
        # Create subplots
        n_cols = min(3, n_categories)
        n_rows = (n_categories + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()
        
        # Plot each KPI category
        for i, (category, kpis) in enumerate(kpi_data.items()):
            if i < len(axes):
                kpi_names = list(kpis.keys())
                kpi_values = list(kpis.values())
                
                # Create bar plot
                bars = axes[i].bar(kpi_names, kpi_values, alpha=0.7)
                
                # Add value labels on bars
                for bar, value in zip(bars, kpi_values):
                    axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                                f'{value:.1f}', ha='center', va='bottom')
                
                axes[i].set_title(category.replace('_', ' ').title())
                axes[i].set_ylabel('Score')
                axes[i].tick_params(axis='x', rotation=45)
                axes[i].set_ylim(0, 100)
                axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_categories, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.default_settings['dpi'], bbox_inches='tight')
            logger.info(f"Manufacturing KPIs plot saved to {save_path}")
        
        plt.show()
    
    def plot_performance_trends(self, performance_data: pd.DataFrame, 
                              metrics: List[str],
                              title: str = "Performance Trends",
                              save_path: Optional[str] = None) -> None:
        """
        Plot performance trends over time.
        
        Args:
            performance_data: DataFrame with performance data
            metrics: List of metrics to plot
            title: Plot title
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        # Plot each metric
        for i, metric in enumerate(metrics):
            if i < len(axes) and metric in performance_data.columns:
                # Plot time series
                axes[i].plot(performance_data.index, performance_data[metric], 
                           marker='o', linewidth=2, markersize=4, label=metric)
                
                # Add moving average
                window_size = min(10, len(performance_data) // 4)
                if window_size > 1:
                    moving_avg = performance_data[metric].rolling(window=window_size).mean()
                    axes[i].plot(performance_data.index, moving_avg, 
                               'r--', linewidth=2, label=f'Moving Avg ({window_size})')
                
                # Add trend line
                x_numeric = np.arange(len(performance_data))
                z = np.polyfit(x_numeric, performance_data[metric], 1)
                p = np.poly1d(z)
                axes[i].plot(performance_data.index, p(x_numeric), 
                           'g--', alpha=0.8, label='Trend')
                
                axes[i].set_title(metric.replace('_', ' ').title())
                axes[i].set_xlabel('Time')
                axes[i].set_ylabel('Value')
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(metrics), len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.default_settings['dpi'], bbox_inches='tight')
            logger.info(f"Performance trends plot saved to {save_path}")
        
        plt.show()
    
    def plot_performance_comparison(self, comparison_data: Dict[str, Dict[str, float]], 
                                  title: str = "Performance Comparison",
                                  save_path: Optional[str] = None) -> None:
        """
        Plot performance comparison between different models/systems.
        
        Args:
            comparison_data: Dictionary mapping system names to their performance metrics
            title: Plot title
            save_path: Path to save the plot
        """
        # Extract data for plotting
        systems = list(comparison_data.keys())
        metrics = list(comparison_data[systems[0]].keys())
        
        # Create DataFrame
        df = pd.DataFrame(comparison_data).T
        
        # Create subplots
        n_metrics = len(metrics)
        n_cols = min(3, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()
        
        # Plot each metric
        for i, metric in enumerate(metrics):
            if i < len(axes):
                bars = axes[i].bar(systems, df[metric], alpha=0.7)
                
                # Add value labels on bars
                for bar, value in zip(bars, df[metric]):
                    axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                                f'{value:.2f}', ha='center', va='bottom')
                
                axes[i].set_title(metric.replace('_', ' ').title())
                axes[i].set_ylabel('Score')
                axes[i].tick_params(axis='x', rotation=45)
                axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_metrics, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.default_settings['dpi'], bbox_inches='tight')
            logger.info(f"Performance comparison plot saved to {save_path}")
        
        plt.show()
    
    def plot_performance_heatmap(self, performance_matrix: np.ndarray, 
                               row_labels: List[str],
                               col_labels: List[str],
                               title: str = "Performance Heatmap",
                               save_path: Optional[str] = None) -> None:
        """
        Plot performance heatmap.
        
        Args:
            performance_matrix: 2D array with performance data
            row_labels: Labels for rows
            col_labels: Labels for columns
            title: Plot title
            save_path: Path to save the plot
        """
        plt.figure(figsize=(12, 8))
        
        # Create heatmap
        sns.heatmap(performance_matrix, annot=True, fmt='.2f', cmap='viridis',
                   xticklabels=col_labels, yticklabels=row_labels)
        
        plt.title(title, fontsize=16)
        plt.xlabel('Metrics', fontsize=14)
        plt.ylabel('Systems/Models', fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.default_settings['dpi'], bbox_inches='tight')
            logger.info(f"Performance heatmap saved to {save_path}")
        
        plt.show()
    
    def plot_performance_distribution(self, performance_data: Dict[str, List[float]], 
                                    title: str = "Performance Distribution",
                                    save_path: Optional[str] = None) -> None:
        """
        Plot performance distribution for different systems/models.
        
        Args:
            performance_data: Dictionary mapping system names to their performance values
            title: Plot title
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Box plot
        data_for_box = [values for values in performance_data.values()]
        labels_for_box = list(performance_data.keys())
        
        axes[0, 0].boxplot(data_for_box, labels=labels_for_box)
        axes[0, 0].set_title('Performance Distribution (Box Plot)')
        axes[0, 0].set_ylabel('Performance Score')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Violin plot
        axes[0, 1].violinplot(data_for_box, labels=labels_for_box)
        axes[0, 1].set_title('Performance Distribution (Violin Plot)')
        axes[0, 1].set_ylabel('Performance Score')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Histogram
        for i, (system, values) in enumerate(performance_data.items()):
            axes[1, 0].hist(values, alpha=0.7, label=system, bins=20)
        axes[1, 0].set_title('Performance Distribution (Histogram)')
        axes[1, 0].set_xlabel('Performance Score')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Statistics summary
        stats_data = []
        for system, values in performance_data.items():
            stats_data.append({
                'System': system,
                'Mean': np.mean(values),
                'Std': np.std(values),
                'Min': np.min(values),
                'Max': np.max(values)
            })
        
        stats_df = pd.DataFrame(stats_data)
        
        # Create bar plot for mean performance
        axes[1, 1].bar(stats_df['System'], stats_df['Mean'], alpha=0.7)
        axes[1, 1].set_title('Mean Performance Comparison')
        axes[1, 1].set_ylabel('Mean Performance Score')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.default_settings['dpi'], bbox_inches='tight')
            logger.info(f"Performance distribution plot saved to {save_path}")
        
        plt.show()
    
    def plot_performance_correlation(self, performance_data: pd.DataFrame, 
                                   title: str = "Performance Correlation",
                                   save_path: Optional[str] = None) -> None:
        """
        Plot correlation between different performance metrics.
        
        Args:
            performance_data: DataFrame with performance data
            title: Plot title
            save_path: Path to save the plot
        """
        # Calculate correlation matrix
        corr_matrix = performance_data.corr()
        
        # Create heatmap
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        
        plt.title(title, fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.default_settings['dpi'], bbox_inches='tight')
            logger.info(f"Performance correlation plot saved to {save_path}")
        
        plt.show()
    
    def plot_performance_anomalies(self, performance_data: pd.DataFrame, 
                                 metrics: List[str],
                                 title: str = "Performance Anomalies",
                                 save_path: Optional[str] = None) -> None:
        """
        Plot performance anomalies detection.
        
        Args:
            performance_data: DataFrame with performance data
            metrics: List of metrics to analyze
            title: Plot title
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        # Analyze each metric
        for i, metric in enumerate(metrics):
            if i < len(axes) and metric in performance_data.columns:
                values = performance_data[metric]
                
                # Calculate anomaly threshold using IQR method
                Q1 = values.quantile(0.25)
                Q3 = values.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Identify anomalies
                anomalies = (values < lower_bound) | (values > upper_bound)
                
                # Plot time series
                axes[i].plot(performance_data.index, values, alpha=0.7, label='Normal')
                axes[i].scatter(performance_data.index[anomalies], values[anomalies], 
                              color='red', s=50, label='Anomalies', zorder=5)
                
                # Add threshold lines
                axes[i].axhline(y=lower_bound, color='orange', linestyle='--', alpha=0.7, label='Lower Bound')
                axes[i].axhline(y=upper_bound, color='orange', linestyle='--', alpha=0.7, label='Upper Bound')
                
                axes[i].set_title(f'{metric.replace("_", " ").title()} - Anomalies')
                axes[i].set_xlabel('Time')
                axes[i].set_ylabel('Value')
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(metrics), len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.default_settings['dpi'], bbox_inches='tight')
            logger.info(f"Performance anomalies plot saved to {save_path}")
        
        plt.show()
    
    def plot_performance_benchmarks(self, benchmark_data: Dict[str, Dict[str, float]], 
                                  title: str = "Performance Benchmarks",
                                  save_path: Optional[str] = None) -> None:
        """
        Plot performance benchmarks.
        
        Args:
            benchmark_data: Dictionary with benchmark data
            title: Plot title
            save_path: Path to save the plot
        """
        # Extract data
        systems = list(benchmark_data.keys())
        benchmarks = list(benchmark_data[systems[0]].keys())
        
        # Create DataFrame
        df = pd.DataFrame(benchmark_data).T
        
        # Create radar chart
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # Calculate angles for each benchmark
        angles = np.linspace(0, 2 * np.pi, len(benchmarks), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        # Plot each system
        for system in systems:
            values = df.loc[system].values.tolist()
            values += values[:1]  # Complete the circle
            
            ax.plot(angles, values, 'o-', linewidth=2, label=system)
            ax.fill(angles, values, alpha=0.25)
        
        # Add benchmark labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(benchmarks)
        ax.set_ylim(0, 100)
        ax.set_title(title, size=16, pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=self.default_settings['dpi'], bbox_inches='tight')
            logger.info(f"Performance benchmarks plot saved to {save_path}")
        
        plt.show()
    
    def create_performance_dashboard(self, performance_data: pd.DataFrame, 
                                   kpi_data: Optional[Dict[str, Dict[str, float]]] = None,
                                   save_path: Optional[str] = None) -> None:
        """
        Create comprehensive performance monitoring dashboard.
        
        Args:
            performance_data: DataFrame with performance data
            kpi_data: Dictionary with KPI data (optional)
            save_path: Path to save the dashboard
        """
        try:
            # Create interactive dashboard using Plotly
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Performance Trends', 'Performance Distribution', 
                              'Performance Correlation', 'Performance Anomalies'),
                specs=[[{"type": "scatter"}, {"type": "histogram"}],
                       [{"type": "heatmap"}, {"type": "scatter"}]]
            )
            
            # Performance trends
            for column in performance_data.columns[:3]:  # Show first 3 metrics
                fig.add_trace(
                    go.Scatter(x=performance_data.index, y=performance_data[column], 
                              name=column, mode='lines+markers'),
                    row=1, col=1
                )
            
            # Performance distribution
            for column in performance_data.columns[:3]:  # Show first 3 metrics
                fig.add_trace(
                    go.Histogram(x=performance_data[column], name=column, opacity=0.7),
                    row=1, col=2
                )
            
            # Performance correlation
            corr_matrix = performance_data.corr()
            fig.add_trace(
                go.Heatmap(z=corr_matrix.values, x=corr_matrix.columns, y=corr_matrix.index),
                row=2, col=1
            )
            
            # Performance anomalies (simplified)
            for column in performance_data.columns[:2]:  # Show first 2 metrics
                values = performance_data[column]
                Q1 = values.quantile(0.25)
                Q3 = values.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                anomalies = (values < lower_bound) | (values > upper_bound)
                
                fig.add_trace(
                    go.Scatter(x=performance_data.index, y=values, 
                              name=f'{column} (Normal)', mode='lines', opacity=0.7),
                    row=2, col=2
                )
                
                fig.add_trace(
                    go.Scatter(x=performance_data.index[anomalies], y=values[anomalies], 
                              name=f'{column} (Anomalies)', mode='markers', 
                              marker=dict(color='red', size=8)),
                    row=2, col=2
                )
            
            fig.update_layout(height=800, showlegend=True, title_text="Performance Monitoring Dashboard")
            
            if save_path:
                fig.write_html(save_path)
                logger.info(f"Performance dashboard saved to {save_path}")
            else:
                fig.show()
                
        except ImportError:
            logger.warning("Plotly not available for interactive dashboard")
        except Exception as e:
            logger.error(f"Failed to create performance dashboard: {e}")
    
    def save_all_performance_plots(self, performance_data: pd.DataFrame, 
                                 kpi_data: Optional[Dict[str, Dict[str, float]]] = None,
                                 output_dir: str = "performance_plots") -> None:
        """
        Save all performance monitoring plots to a directory.
        
        Args:
            performance_data: DataFrame with performance data
            kpi_data: Dictionary with KPI data (optional)
            output_dir: Output directory for plots
        """
        import os
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Save each plot
        plots = [
            ('performance_trends', lambda: self.plot_performance_trends(performance_data, list(performance_data.columns))),
            ('performance_correlation', lambda: self.plot_performance_correlation(performance_data)),
            ('performance_distribution', lambda: self.plot_performance_distribution(
                {col: performance_data[col].tolist() for col in performance_data.columns})),
            ('performance_anomalies', lambda: self.plot_performance_anomalies(performance_data, list(performance_data.columns)))
        ]
        
        if kpi_data is not None:
            plots.append(('manufacturing_kpis', lambda: self.plot_manufacturing_kpis(kpi_data)))
        
        for plot_name, plot_func in plots:
            save_path = os.path.join(output_dir, f"{plot_name}.png")
            
            try:
                plot_func()
                plt.savefig(save_path, dpi=self.default_settings['dpi'], bbox_inches='tight')
                plt.close()
                logger.info(f"Saved {plot_name} to {save_path}")
                
            except Exception as e:
                logger.error(f"Failed to save {plot_name}: {e}")
    
    def generate_performance_report(self, performance_data: pd.DataFrame, 
                                  kpi_data: Optional[Dict[str, Dict[str, float]]] = None,
                                  output_path: str = "performance_report.html") -> None:
        """
        Generate HTML report with all performance analysis.
        
        Args:
            performance_data: DataFrame with performance data
            kpi_data: Dictionary with KPI data (optional)
            output_path: Path to save the HTML report
        """
        try:
            # Calculate performance statistics
            stats = performance_data.describe()
            
            # Generate HTML report
            html_content = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Performance Monitoring Report</title>
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
                <h1>Performance Monitoring Report</h1>
            """
            
            # Add performance statistics
            html_content += "<h2>Performance Statistics</h2>"
            html_content += stats.to_html()
            
            # Add KPI data if available
            if kpi_data is not None:
                html_content += "<h2>Manufacturing KPIs</h2>"
                for category, kpis in kpi_data.items():
                    html_content += f"<h3>{category.replace('_', ' ').title()}</h3>"
                    for kpi, value in kpis.items():
                        html_content += f"<div class='metric'><strong>{kpi}:</strong> {value:.2f}</div>"
            
            # Add performance summary
            html_content += "<h2>Performance Summary</h2>"
            for column in performance_data.columns:
                mean_val = performance_data[column].mean()
                std_val = performance_data[column].std()
                min_val = performance_data[column].min()
                max_val = performance_data[column].max()
                
                html_content += f"<h3>{column.replace('_', ' ').title()}</h3>"
                html_content += f"<div class='metric'><strong>Mean:</strong> {mean_val:.4f}</div>"
                html_content += f"<div class='metric'><strong>Std:</strong> {std_val:.4f}</div>"
                html_content += f"<div class='metric'><strong>Min:</strong> {min_val:.4f}</div>"
                html_content += f"<div class='metric'><strong>Max:</strong> {max_val:.4f}</div>"
            
            html_content += """
            </body>
            </html>
            """
            
            # Save HTML report
            with open(output_path, 'w') as f:
                f.write(html_content)
            
            logger.info(f"Performance report saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to generate performance report: {e}")
