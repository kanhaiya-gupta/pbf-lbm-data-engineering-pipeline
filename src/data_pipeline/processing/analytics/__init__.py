"""
Analytics Module for PBF-LB/M Data Pipeline

This module provides advanced statistical analysis and sensitivity analysis capabilities
for PBF-LB/M (Powder Bed Fusion - Laser Beam/Metal) additive manufacturing research.
It extends the data pipeline with sophisticated analytics that process data through
the pipeline and provide systematic evaluation of process variables.

Key Features:
- Sensitivity Analysis: Sobol indices, Morris screening, design of experiments
- Statistical Analysis: Multivariate, time series, regression, nonparametric methods
- Process Analysis: Parameter analysis, quality analysis, sensor analysis, optimization
- Reporting: Automated report generation, visualization, documentation

Architecture:
- Sensitivity Analysis: Global and local sensitivity methods
- Statistical Analysis: Advanced statistical modeling and analysis
- Process Analysis: PBF-specific analysis tools
- Reporting: Analysis results and visualization
"""

# Sensitivity Analysis Components
from .sensitivity_analysis.global_analysis import GlobalSensitivityAnalyzer, SobolAnalyzer, MorrisAnalyzer
from .sensitivity_analysis.local_analysis import LocalSensitivityAnalyzer, DerivativeAnalyzer
from .sensitivity_analysis.doe import ExperimentalDesigner, FactorialDesign, ResponseSurfaceDesign
from .sensitivity_analysis.uncertainty import UncertaintyQuantifier, MonteCarloAnalyzer, BayesianAnalyzer

# Statistical Analysis Components
from .statistical_analysis.multivariate import MultivariateAnalyzer, PCAAnalyzer, ClusterAnalyzer
from .statistical_analysis.time_series import TimeSeriesAnalyzer, TrendAnalyzer, SeasonalityAnalyzer
from .statistical_analysis.regression import RegressionAnalyzer, LinearRegression, PolynomialRegression
from .statistical_analysis.nonparametric import NonparametricAnalyzer, KernelDensityAnalyzer

# Process Analysis Components
from .process_analysis.parameter_analysis import ParameterAnalyzer, ProcessParameterOptimizer
from .process_analysis.quality_analysis import QualityAnalyzer, QualityPredictor
from .process_analysis.sensor_analysis import SensorAnalyzer, ISPMAnalyzer, CTSensorAnalyzer
from .process_analysis.optimization import ProcessOptimizer, MultiObjectiveOptimizer

# Reporting Components
from .reporting.report_generators import AnalysisReportGenerator, SensitivityReportGenerator
from .reporting.visualization import AnalysisVisualizer, SensitivityVisualizer
from .reporting.documentation import AnalysisDocumentation, APIDocumentation

__all__ = [
    # Sensitivity Analysis
    'GlobalSensitivityAnalyzer',
    'SobolAnalyzer', 
    'MorrisAnalyzer',
    'LocalSensitivityAnalyzer',
    'DerivativeAnalyzer',
    'ExperimentalDesigner',
    'FactorialDesign',
    'ResponseSurfaceDesign',
    'UncertaintyQuantifier',
    'MonteCarloAnalyzer',
    'BayesianAnalyzer',
    
    # Statistical Analysis
    'MultivariateAnalyzer',
    'PCAAnalyzer',
    'ClusterAnalyzer',
    'TimeSeriesAnalyzer',
    'TrendAnalyzer',
    'SeasonalityAnalyzer',
    'RegressionAnalyzer',
    'LinearRegression',
    'PolynomialRegression',
    'NonparametricAnalyzer',
    'KernelDensityAnalyzer',
    
    # Process Analysis
    'ParameterAnalyzer',
    'ProcessParameterOptimizer',
    'QualityAnalyzer',
    'QualityPredictor',
    'SensorAnalyzer',
    'ISPMAnalyzer',
    'CTSensorAnalyzer',
    'ProcessOptimizer',
    'MultiObjectiveOptimizer',
    
    # Reporting
    'AnalysisReportGenerator',
    'SensitivityReportGenerator',
    'AnalysisVisualizer',
    'SensitivityVisualizer',
    'AnalysisDocumentation',
    'APIDocumentation',
]

# Version information
__version__ = "1.0.0"
__author__ = "PBF-LB/M Research Team"
__description__ = "Advanced analytics for PBF-LB/M data pipeline"
