"""
Analytics Module for PBF-LB/M Data Pipeline

This module provides advanced statistical analysis and sensitivity analysis capabilities
for PBF-LB/M (Powder Bed Fusion - Laser Beam/Metal) additive manufacturing research.

This is a top-level module that re-exports all analytics components from the processing
analytics submodule for easier access.
"""

# Re-export all analytics components from processing.analytics
from ..processing.analytics import *

# Additional convenience functions for creating analyzers
from typing import Optional, Dict, Any
import numpy as np

def create_sensitivity_analyzer(
    method: str = "sobol",
    config: Optional[Dict[str, Any]] = None
) -> 'GlobalSensitivityAnalyzer':
    """
    Create a sensitivity analyzer with the specified method.
    
    Args:
        method: Analysis method ('sobol', 'morris', 'local')
        config: Optional configuration dictionary
        
    Returns:
        Configured sensitivity analyzer
    """
    if method.lower() == "sobol":
        from ..processing.analytics.sensitivity_analysis.global_analysis import SobolAnalyzer
        return SobolAnalyzer(config or {})
    elif method.lower() == "morris":
        from ..processing.analytics.sensitivity_analysis.global_analysis import MorrisAnalyzer
        return MorrisAnalyzer(config or {})
    elif method.lower() == "local":
        from ..processing.analytics.sensitivity_analysis.local_analysis import LocalSensitivityAnalyzer
        return LocalSensitivityAnalyzer(config or {})
    else:
        from ..processing.analytics.sensitivity_analysis.global_analysis import GlobalSensitivityAnalyzer
        return GlobalSensitivityAnalyzer(config or {})


def create_uncertainty_quantifier(
    method: str = "monte_carlo",
    config: Optional[Dict[str, Any]] = None
) -> 'UncertaintyQuantifier':
    """
    Create an uncertainty quantifier with the specified method.
    
    Args:
        method: Quantification method ('monte_carlo', 'bayesian')
        config: Optional configuration dictionary
        
    Returns:
        Configured uncertainty quantifier
    """
    if method.lower() == "monte_carlo":
        from ..processing.analytics.sensitivity_analysis.uncertainty import MonteCarloAnalyzer
        return MonteCarloAnalyzer(config or {})
    elif method.lower() == "bayesian":
        from ..processing.analytics.sensitivity_analysis.uncertainty import BayesianAnalyzer
        return BayesianAnalyzer(config or {})
    else:
        from ..processing.analytics.sensitivity_analysis.uncertainty import UncertaintyQuantifier
        return UncertaintyQuantifier(config or {})


def create_statistical_analyzer(
    method: str = "multivariate",
    config: Optional[Dict[str, Any]] = None
) -> 'MultivariateAnalyzer':
    """
    Create a statistical analyzer with the specified method.
    
    Args:
        method: Analysis method ('multivariate', 'time_series', 'regression', 'nonparametric')
        config: Optional configuration dictionary
        
    Returns:
        Configured statistical analyzer
    """
    if method.lower() == "multivariate":
        from ..processing.analytics.statistical_analysis.multivariate import MultivariateAnalyzer
        return MultivariateAnalyzer(config or {})
    elif method.lower() == "time_series":
        from ..processing.analytics.statistical_analysis.time_series import TimeSeriesAnalyzer
        return TimeSeriesAnalyzer(config or {})
    elif method.lower() == "regression":
        from ..processing.analytics.statistical_analysis.regression import RegressionAnalyzer
        return RegressionAnalyzer(config or {})
    elif method.lower() == "nonparametric":
        from ..processing.analytics.statistical_analysis.nonparametric import NonparametricAnalyzer
        return NonparametricAnalyzer(config or {})
    else:
        from ..processing.analytics.statistical_analysis.multivariate import MultivariateAnalyzer
        return MultivariateAnalyzer(config or {})


def create_doe_analyzer(
    design_type: str = "factorial",
    config: Optional[Dict[str, Any]] = None
) -> 'ExperimentalDesigner':
    """
    Create a design of experiments analyzer.
    
    Args:
        design_type: Design type ('factorial', 'response_surface')
        config: Optional configuration dictionary
        
    Returns:
        Configured experimental designer
    """
    if design_type.lower() == "factorial":
        from ..processing.analytics.sensitivity_analysis.doe import FactorialDesign
        return FactorialDesign(config or {})
    elif design_type.lower() == "response_surface":
        from ..processing.analytics.sensitivity_analysis.doe import ResponseSurfaceDesign
        return ResponseSurfaceDesign(config or {})
    else:
        from ..processing.analytics.sensitivity_analysis.doe import ExperimentalDesigner
        return ExperimentalDesigner(config or {})


# Add convenience functions to __all__
__all__ = [
    # Re-export all from processing.analytics
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
    'ParameterAnalyzer',
    'ProcessParameterOptimizer',
    'QualityAnalyzer',
    'QualityPredictor',
    'SensorAnalyzer',
    'ISPMAnalyzer',
    'CTSensorAnalyzer',
    'ProcessOptimizer',
    'MultiObjectiveOptimizer',
    'AnalysisReportGenerator',
    'SensitivityReportGenerator',
    'AnalysisVisualizer',
    'SensitivityVisualizer',
    'AnalysisDocumentation',
    'APIDocumentation',
    
    # Convenience functions
    'create_sensitivity_analyzer',
    'create_uncertainty_quantifier',
    'create_statistical_analyzer',
    'create_doe_analyzer',
]

# Version information
__version__ = "1.0.0"
__author__ = "PBF-LB/M Research Team"
__description__ = "Advanced analytics for PBF-LB/M data pipeline"
