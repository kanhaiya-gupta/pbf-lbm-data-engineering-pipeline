"""
Process Parameter Analysis for PBF-LB/M Systems

This module provides specialized process parameter analysis capabilities
for PBF-LB/M additive manufacturing systems, including parameter optimization,
parameter interaction analysis, and process control optimization.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass
import logging
from datetime import datetime
from scipy.optimize import minimize, differential_evolution
from scipy.stats import pearsonr, spearmanr
import warnings

logger = logging.getLogger(__name__)


@dataclass
class ParameterAnalysisConfig:
    """Configuration for parameter analysis."""
    
    # Optimization parameters
    optimization_method: str = "differential_evolution"  # "minimize", "differential_evolution"
    max_iterations: int = 1000
    tolerance: float = 1e-6
    
    # Parameter interaction parameters
    interaction_threshold: float = 0.3
    correlation_method: str = "pearson"  # "pearson", "spearman"
    
    # Analysis parameters
    confidence_level: float = 0.95
    significance_level: float = 0.05
    random_seed: Optional[int] = None


@dataclass
class ParameterAnalysisResult:
    """Result of parameter analysis."""
    
    success: bool
    method: str
    parameter_names: List[str]
    optimal_parameters: Dict[str, float]
    optimal_value: float
    parameter_interactions: Dict[str, Dict[str, float]]
    parameter_importance: Dict[str, float]
    analysis_time: float = 0.0
    error_message: Optional[str] = None


class ParameterAnalyzer:
    """
    Process parameter analyzer for PBF-LB/M systems.
    
    This class provides specialized analysis capabilities for PBF-LB/M process
    parameters including optimization, interaction analysis, and importance
    ranking for process control and optimization.
    """
    
    def __init__(self, config: ParameterAnalysisConfig = None):
        """Initialize the parameter analyzer."""
        self.config = config or ParameterAnalysisConfig()
        self.analysis_cache = {}
        
        # Set random seed
        if self.config.random_seed is not None:
            np.random.seed(self.config.random_seed)
        
        logger.info("Parameter Analyzer initialized")
    
    def analyze_parameter_optimization(
        self,
        objective_function: Callable,
        parameter_bounds: Dict[str, Tuple[float, float]],
        parameter_names: List[str] = None,
        optimization_method: str = None
    ) -> ParameterAnalysisResult:
        """
        Perform parameter optimization analysis.
        
        Args:
            objective_function: Function to optimize (should return scalar value)
            parameter_bounds: Dictionary of parameter bounds {name: (min, max)}
            parameter_names: List of parameter names (optional)
            optimization_method: Optimization method (optional)
            
        Returns:
            ParameterAnalysisResult: Parameter optimization results
        """
        try:
            start_time = datetime.now()
            
            if parameter_names is None:
                parameter_names = list(parameter_bounds.keys())
            
            if optimization_method is None:
                optimization_method = self.config.optimization_method
            
            # Prepare bounds for optimization
            bounds = [parameter_bounds[name] for name in parameter_names]
            
            # Create wrapper function for optimization
            def wrapper(x):
                param_dict = {name: x[i] for i, name in enumerate(parameter_names)}
                return objective_function(param_dict)
            
            # Perform optimization
            if optimization_method == "differential_evolution":
                result = differential_evolution(
                    wrapper,
                    bounds,
                    maxiter=self.config.max_iterations,
                    tol=self.config.tolerance,
                    seed=self.config.random_seed
                )
            else:
                # Use scipy minimize
                x0 = [np.mean(bounds[i]) for i in range(len(bounds))]
                result = minimize(
                    wrapper,
                    x0,
                    bounds=bounds,
                    method='L-BFGS-B',
                    options={'maxiter': self.config.max_iterations}
                )
            
            # Extract optimal parameters
            optimal_parameters = {name: result.x[i] for i, name in enumerate(parameter_names)}
            optimal_value = result.fun
            
            # Calculate analysis time
            analysis_time = (datetime.now() - start_time).total_seconds()
            
            # Create result
            result_obj = ParameterAnalysisResult(
                success=result.success,
                method=f"ParameterOptimization_{optimization_method}",
                parameter_names=parameter_names,
                optimal_parameters=optimal_parameters,
                optimal_value=optimal_value,
                parameter_interactions={},
                parameter_importance={},
                analysis_time=analysis_time
            )
            
            # Cache result
            self._cache_result("parameter_optimization", result_obj)
            
            logger.info(f"Parameter optimization completed: {analysis_time:.2f}s")
            return result_obj
            
        except Exception as e:
            logger.error(f"Error in parameter optimization: {e}")
            return ParameterAnalysisResult(
                success=False,
                method=f"ParameterOptimization_{optimization_method}",
                parameter_names=parameter_names or [],
                optimal_parameters={},
                optimal_value=0.0,
                parameter_interactions={},
                parameter_importance={},
                error_message=str(e)
            )
    
    def analyze_parameter_interactions(
        self,
        process_data: pd.DataFrame,
        parameter_names: List[str] = None,
        target_variable: str = None
    ) -> ParameterAnalysisResult:
        """
        Analyze parameter interactions in process data.
        
        Args:
            process_data: DataFrame containing process data
            parameter_names: List of parameter names (optional)
            target_variable: Target variable name (optional)
            
        Returns:
            ParameterAnalysisResult: Parameter interaction analysis results
        """
        try:
            start_time = datetime.now()
            
            if parameter_names is None:
                parameter_names = [col for col in process_data.columns if col != target_variable]
            
            if target_variable is None:
                target_variable = process_data.columns[-1]  # Use last column as target
            
            # Prepare data
            X = process_data[parameter_names].values
            y = process_data[target_variable].values
            
            # Handle missing values
            valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
            X = X[valid_mask]
            y = y[valid_mask]
            
            # Calculate parameter interactions
            parameter_interactions = {}
            for i, param1 in enumerate(parameter_names):
                parameter_interactions[param1] = {}
                for j, param2 in enumerate(parameter_names):
                    if i != j:
                        if self.config.correlation_method == "pearson":
                            corr, p_value = pearsonr(X[:, i], X[:, j])
                        else:
                            corr, p_value = spearmanr(X[:, i], X[:, j])
                        
                        parameter_interactions[param1][param2] = corr
            
            # Calculate parameter importance (correlation with target)
            parameter_importance = {}
            for i, param in enumerate(parameter_names):
                if self.config.correlation_method == "pearson":
                    corr, p_value = pearsonr(X[:, i], y)
                else:
                    corr, p_value = spearmanr(X[:, i], y)
                
                parameter_importance[param] = abs(corr)
            
            # Calculate analysis time
            analysis_time = (datetime.now() - start_time).total_seconds()
            
            # Create result
            result = ParameterAnalysisResult(
                success=True,
                method="ParameterInteractions",
                parameter_names=parameter_names,
                optimal_parameters={},
                optimal_value=0.0,
                parameter_interactions=parameter_interactions,
                parameter_importance=parameter_importance,
                analysis_time=analysis_time
            )
            
            # Cache result
            self._cache_result("parameter_interactions", result)
            
            logger.info(f"Parameter interactions analysis completed: {analysis_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Error in parameter interactions analysis: {e}")
            return ParameterAnalysisResult(
                success=False,
                method="ParameterInteractions",
                parameter_names=parameter_names or [],
                optimal_parameters={},
                optimal_value=0.0,
                parameter_interactions={},
                parameter_importance={},
                error_message=str(e)
            )
    
    def analyze_parameter_sensitivity(
        self,
        objective_function: Callable,
        nominal_parameters: Dict[str, float],
        parameter_bounds: Dict[str, Tuple[float, float]],
        parameter_names: List[str] = None
    ) -> ParameterAnalysisResult:
        """
        Analyze parameter sensitivity around nominal point.
        
        Args:
            objective_function: Function to analyze
            nominal_parameters: Dictionary of nominal parameter values
            parameter_bounds: Dictionary of parameter bounds
            parameter_names: List of parameter names (optional)
            
        Returns:
            ParameterAnalysisResult: Parameter sensitivity analysis results
        """
        try:
            start_time = datetime.now()
            
            if parameter_names is None:
                parameter_names = list(nominal_parameters.keys())
            
            # Calculate nominal objective value
            nominal_value = objective_function(nominal_parameters)
            
            # Calculate parameter sensitivities
            parameter_importance = {}
            for param_name in parameter_names:
                # Create perturbed parameters
                perturbed_params = nominal_parameters.copy()
                
                # Calculate parameter range
                param_min, param_max = parameter_bounds[param_name]
                param_range = param_max - param_min
                perturbation = 0.01 * param_range  # 1% perturbation
                
                # Forward perturbation
                perturbed_params[param_name] = min(param_max, nominal_parameters[param_name] + perturbation)
                forward_value = objective_function(perturbed_params)
                
                # Backward perturbation
                perturbed_params[param_name] = max(param_min, nominal_parameters[param_name] - perturbation)
                backward_value = objective_function(perturbed_params)
                
                # Calculate sensitivity (finite difference)
                sensitivity = abs(forward_value - backward_value) / (2 * perturbation)
                parameter_importance[param_name] = sensitivity
            
            # Calculate analysis time
            analysis_time = (datetime.now() - start_time).total_seconds()
            
            # Create result
            result = ParameterAnalysisResult(
                success=True,
                method="ParameterSensitivity",
                parameter_names=parameter_names,
                optimal_parameters=nominal_parameters,
                optimal_value=nominal_value,
                parameter_interactions={},
                parameter_importance=parameter_importance,
                analysis_time=analysis_time
            )
            
            # Cache result
            self._cache_result("parameter_sensitivity", result)
            
            logger.info(f"Parameter sensitivity analysis completed: {analysis_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Error in parameter sensitivity analysis: {e}")
            return ParameterAnalysisResult(
                success=False,
                method="ParameterSensitivity",
                parameter_names=parameter_names or [],
                optimal_parameters={},
                optimal_value=0.0,
                parameter_interactions={},
                parameter_importance={},
                error_message=str(e)
            )
    
    def _cache_result(self, method: str, result: ParameterAnalysisResult):
        """Cache analysis result."""
        cache_key = f"{method}_{hash(str(result.parameter_names))}"
        self.analysis_cache[cache_key] = result
    
    def get_cached_result(self, method: str, parameter_names: List[str]) -> Optional[ParameterAnalysisResult]:
        """Get cached analysis result."""
        cache_key = f"{method}_{hash(str(parameter_names))}"
        return self.analysis_cache.get(cache_key)
    
    def clear_cache(self):
        """Clear analysis cache."""
        self.analysis_cache.clear()
    
    def get_analysis_statistics(self) -> Dict[str, Any]:
        """Get analysis statistics."""
        return {
            'cache_size': len(self.analysis_cache),
            'config': {
                'optimization_method': self.config.optimization_method,
                'max_iterations': self.config.max_iterations,
                'tolerance': self.config.tolerance,
                'interaction_threshold': self.config.interaction_threshold,
                'correlation_method': self.config.correlation_method
            }
        }


class ProcessParameterOptimizer(ParameterAnalyzer):
    """Specialized process parameter optimizer."""
    
    def __init__(self, config: ParameterAnalysisConfig = None):
        super().__init__(config)
        self.method_name = "ProcessParameterOptimizer"
    
    def optimize(self, objective_function: Callable, parameter_bounds: Dict[str, Tuple[float, float]], 
                 parameter_names: List[str] = None) -> ParameterAnalysisResult:
        """Optimize process parameters."""
        return self.analyze_parameter_optimization(objective_function, parameter_bounds, parameter_names)
