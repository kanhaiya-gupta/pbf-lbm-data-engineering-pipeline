"""
Visualization for PBF-LB/M Analytics

This module provides comprehensive visualization capabilities for PBF-LB/M
analytics results, including sensitivity analysis visualization, statistical
analysis visualization, and process analysis visualization.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import logging
from datetime import datetime
import warnings

logger = logging.getLogger(__name__)


@dataclass
class VisualizationConfig:
    """Configuration for visualization."""
    
    # Plot parameters
    figure_size: Tuple[int, int] = (12, 8)
    dpi: int = 300
    style: str = "whitegrid"  # "whitegrid", "darkgrid", "white", "dark"
    
    # Color parameters
    color_palette: str = "viridis"  # "viridis", "plasma", "inferno", "magma"
    alpha: float = 0.7
    
    # Output parameters
    output_directory: str = "plots"
    save_format: str = "png"  # "png", "pdf", "svg"
    
    # Analysis parameters
    confidence_level: float = 0.95


@dataclass
class VisualizationResult:
    """Result of visualization generation."""
    
    success: bool
    visualization_type: str
    plot_paths: List[str]
    generation_time: float
    error_message: Optional[str] = None


class AnalysisVisualizer:
    """
    Analysis visualizer for PBF-LB/M analytics.
    
    This class provides comprehensive visualization capabilities for
    analytics results including sensitivity analysis, statistical analysis,
    and process analysis visualization.
    """
    
    def __init__(self, config: VisualizationConfig = None):
        """Initialize the visualizer."""
        self.config = config or VisualizationConfig()
        
        # Set matplotlib style
        plt.style.use('seaborn-v0_8' if hasattr(plt.style, 'seaborn-v0_8') else 'seaborn')
        sns.set_palette(self.config.color_palette)
        
        # Create output directory
        import os
        os.makedirs(self.config.output_directory, exist_ok=True)
        
        logger.info("Analysis Visualizer initialized")
    
    def visualize_sensitivity_analysis(
        self,
        sensitivity_results: Dict[str, Any],
        plot_title: str = "Sensitivity Analysis"
    ) -> VisualizationResult:
        """
        Visualize sensitivity analysis results.
        
        Args:
            sensitivity_results: Dictionary containing sensitivity analysis results
            plot_title: Title for the plots
            
        Returns:
            VisualizationResult: Visualization generation result
        """
        try:
            start_time = datetime.now()
            plot_paths = []
            
            # Generate Sobol indices plot
            if 'sobol_analysis' in sensitivity_results:
                sobol_plot_path = self._plot_sobol_indices(sensitivity_results['sobol_analysis'], plot_title)
                plot_paths.append(sobol_plot_path)
            
            # Generate Morris screening plot
            if 'morris_analysis' in sensitivity_results:
                morris_plot_path = self._plot_morris_screening(sensitivity_results['morris_analysis'], plot_title)
                plot_paths.append(morris_plot_path)
            
            # Calculate generation time
            generation_time = (datetime.now() - start_time).total_seconds()
            
            # Create result
            result = VisualizationResult(
                success=True,
                visualization_type="SensitivityAnalysis",
                plot_paths=plot_paths,
                generation_time=generation_time
            )
            
            logger.info(f"Sensitivity analysis visualization completed: {generation_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Error in sensitivity analysis visualization: {e}")
            return VisualizationResult(
                success=False,
                visualization_type="SensitivityAnalysis",
                plot_paths=[],
                generation_time=0.0,
                error_message=str(e)
            )
    
    def visualize_statistical_analysis(
        self,
        statistical_results: Dict[str, Any],
        plot_title: str = "Statistical Analysis"
    ) -> VisualizationResult:
        """
        Visualize statistical analysis results.
        
        Args:
            statistical_results: Dictionary containing statistical analysis results
            plot_title: Title for the plots
            
        Returns:
            VisualizationResult: Visualization generation result
        """
        try:
            start_time = datetime.now()
            plot_paths = []
            
            # Generate PCA plot
            if 'pca_analysis' in statistical_results:
                pca_plot_path = self._plot_pca_results(statistical_results['pca_analysis'], plot_title)
                plot_paths.append(pca_plot_path)
            
            # Generate correlation heatmap
            if 'correlation_analysis' in statistical_results:
                corr_plot_path = self._plot_correlation_heatmap(statistical_results['correlation_analysis'], plot_title)
                plot_paths.append(corr_plot_path)
            
            # Calculate generation time
            generation_time = (datetime.now() - start_time).total_seconds()
            
            # Create result
            result = VisualizationResult(
                success=True,
                visualization_type="StatisticalAnalysis",
                plot_paths=plot_paths,
                generation_time=generation_time
            )
            
            logger.info(f"Statistical analysis visualization completed: {generation_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Error in statistical analysis visualization: {e}")
            return VisualizationResult(
                success=False,
                visualization_type="StatisticalAnalysis",
                plot_paths=[],
                generation_time=0.0,
                error_message=str(e)
            )
    
    def visualize_process_analysis(
        self,
        process_results: Dict[str, Any],
        plot_title: str = "Process Analysis"
    ) -> VisualizationResult:
        """
        Visualize process analysis results.
        
        Args:
            process_results: Dictionary containing process analysis results
            plot_title: Title for the plots
            
        Returns:
            VisualizationResult: Visualization generation result
        """
        try:
            start_time = datetime.now()
            plot_paths = []
            
            # Generate parameter importance plot
            if 'parameter_analysis' in process_results:
                param_plot_path = self._plot_parameter_importance(process_results['parameter_analysis'], plot_title)
                plot_paths.append(param_plot_path)
            
            # Generate quality prediction plot
            if 'quality_analysis' in process_results:
                quality_plot_path = self._plot_quality_predictions(process_results['quality_analysis'], plot_title)
                plot_paths.append(quality_plot_path)
            
            # Calculate generation time
            generation_time = (datetime.now() - start_time).total_seconds()
            
            # Create result
            result = VisualizationResult(
                success=True,
                visualization_type="ProcessAnalysis",
                plot_paths=plot_paths,
                generation_time=generation_time
            )
            
            logger.info(f"Process analysis visualization completed: {generation_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Error in process analysis visualization: {e}")
            return VisualizationResult(
                success=False,
                visualization_type="ProcessAnalysis",
                plot_paths=[],
                generation_time=0.0,
                error_message=str(e)
            )
    
    def _plot_sobol_indices(self, sobol_results: Any, plot_title: str) -> str:
        """Plot Sobol sensitivity indices."""
        fig, ax = plt.subplots(figsize=self.config.figure_size)
        
        # Extract parameter names and indices
        parameter_names = sobol_results.parameter_names
        sensitivity_indices = sobol_results.sensitivity_indices
        
        # Extract first-order indices
        s1_indices = [sensitivity_indices.get(f'S1_{name}', 0) for name in parameter_names]
        st_indices = [sensitivity_indices.get(f'ST_{name}', 0) for name in parameter_names]
        
        # Create bar plot
        x = np.arange(len(parameter_names))
        width = 0.35
        
        ax.bar(x - width/2, s1_indices, width, label='First-order (S1)', alpha=self.config.alpha)
        ax.bar(x + width/2, st_indices, width, label='Total-order (ST)', alpha=self.config.alpha)
        
        ax.set_xlabel('Parameters')
        ax.set_ylabel('Sensitivity Index')
        ax.set_title(f'{plot_title} - Sobol Indices')
        ax.set_xticks(x)
        ax.set_xticklabels(parameter_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = f"{self.config.output_directory}/sobol_indices_{timestamp}.{self.config.save_format}"
        plt.savefig(plot_path, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
        
        return plot_path
    
    def _plot_morris_screening(self, morris_results: Any, plot_title: str) -> str:
        """Plot Morris screening results."""
        fig, ax = plt.subplots(figsize=self.config.figure_size)
        
        # Extract parameter names and indices
        parameter_names = morris_results.parameter_names
        sensitivity_indices = morris_results.sensitivity_indices
        
        # Extract mu_star indices
        mu_star_indices = [sensitivity_indices.get(f'mu_star_{name}', 0) for name in parameter_names]
        
        # Create bar plot
        x = np.arange(len(parameter_names))
        ax.bar(x, mu_star_indices, alpha=self.config.alpha)
        
        ax.set_xlabel('Parameters')
        ax.set_ylabel('Mu* (Elementary Effects)')
        ax.set_title(f'{plot_title} - Morris Screening')
        ax.set_xticks(x)
        ax.set_xticklabels(parameter_names, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = f"{self.config.output_directory}/morris_screening_{timestamp}.{self.config.save_format}"
        plt.savefig(plot_path, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
        
        return plot_path
    
    def _plot_pca_results(self, pca_results: Any, plot_title: str) -> str:
        """Plot PCA results."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.config.figure_size)
        
        # Plot explained variance
        explained_variance = pca_results.explained_variance['explained_variance_ratio']
        cumulative_variance = pca_results.explained_variance['cumulative_variance_ratio']
        
        ax1.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=self.config.alpha)
        ax1.set_xlabel('Principal Component')
        ax1.set_ylabel('Explained Variance Ratio')
        ax1.set_title('Explained Variance by Component')
        ax1.grid(True, alpha=0.3)
        
        # Plot cumulative variance
        ax2.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'o-', alpha=self.config.alpha)
        ax2.set_xlabel('Principal Component')
        ax2.set_ylabel('Cumulative Explained Variance')
        ax2.set_title('Cumulative Explained Variance')
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(f'{plot_title} - PCA Results')
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = f"{self.config.output_directory}/pca_results_{timestamp}.{self.config.save_format}"
        plt.savefig(plot_path, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
        
        return plot_path
    
    def _plot_correlation_heatmap(self, correlation_results: Any, plot_title: str) -> str:
        """Plot correlation heatmap."""
        fig, ax = plt.subplots(figsize=self.config.figure_size)
        
        # Extract correlation matrix
        correlation_matrix = correlation_results.analysis_results['correlation_matrix']
        feature_names = correlation_results.analysis_results['feature_names']
        
        # Create heatmap
        sns.heatmap(correlation_matrix, 
                   xticklabels=feature_names, 
                   yticklabels=feature_names,
                   annot=True, 
                   cmap='coolwarm', 
                   center=0,
                   ax=ax)
        
        ax.set_title(f'{plot_title} - Correlation Heatmap')
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = f"{self.config.output_directory}/correlation_heatmap_{timestamp}.{self.config.save_format}"
        plt.savefig(plot_path, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
        
        return plot_path
    
    def _plot_parameter_importance(self, parameter_results: Any, plot_title: str) -> str:
        """Plot parameter importance."""
        fig, ax = plt.subplots(figsize=self.config.figure_size)
        
        # Extract parameter importance
        parameter_importance = parameter_results.parameter_importance
        parameter_names = list(parameter_importance.keys())
        importance_values = list(parameter_importance.values())
        
        # Create bar plot
        x = np.arange(len(parameter_names))
        ax.bar(x, importance_values, alpha=self.config.alpha)
        
        ax.set_xlabel('Parameters')
        ax.set_ylabel('Importance')
        ax.set_title(f'{plot_title} - Parameter Importance')
        ax.set_xticks(x)
        ax.set_xticklabels(parameter_names, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = f"{self.config.output_directory}/parameter_importance_{timestamp}.{self.config.save_format}"
        plt.savefig(plot_path, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
        
        return plot_path
    
    def _plot_quality_predictions(self, quality_results: Any, plot_title: str) -> str:
        """Plot quality predictions."""
        fig, ax = plt.subplots(figsize=self.config.figure_size)
        
        # Extract quality predictions
        quality_predictions = quality_results.quality_predictions
        
        # Create histogram
        ax.hist(quality_predictions, bins=30, alpha=self.config.alpha, edgecolor='black')
        
        ax.set_xlabel('Quality Value')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{plot_title} - Quality Predictions Distribution')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = f"{self.config.output_directory}/quality_predictions_{timestamp}.{self.config.save_format}"
        plt.savefig(plot_path, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
        
        return plot_path


class SensitivityVisualizer(AnalysisVisualizer):
    """Specialized sensitivity analysis visualizer."""
    
    def __init__(self, config: VisualizationConfig = None):
        super().__init__(config)
        self.visualization_type = "Sensitivity"
    
    def visualize(self, sensitivity_results: Dict[str, Any], plot_title: str = "Sensitivity Analysis") -> VisualizationResult:
        """Visualize sensitivity analysis results."""
        return self.visualize_sensitivity_analysis(sensitivity_results, plot_title)
