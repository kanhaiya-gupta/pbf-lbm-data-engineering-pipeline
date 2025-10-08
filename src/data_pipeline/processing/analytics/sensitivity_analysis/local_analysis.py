"""
Local Sensitivity Analysis for PBF-LB/M Process Parameters

This module provides local sensitivity analysis methods including derivative-based
analysis and local parameter perturbation for understanding local parameter
influences in PBF-LB/M processes.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass
import logging
from datetime import datetime
from scipy.optimize import approx_fprime
try:
    from scipy.misc import derivative
except ImportError:
    # For newer SciPy versions, use scipy.optimize.approx_fprime instead
    def derivative(func, x0, dx=1.0, n=1, args=(), order=3):
        """Fallback derivative function for newer SciPy versions."""
        if n == 1:
            return approx_fprime(x0, func, dx, *args)
        else:
            # For higher order derivatives, use finite differences
            import numpy as np
            if order == 1:
                return (func(x0 + dx, *args) - func(x0, *args)) / dx
            elif order == 2:
                return (func(x0 + dx, *args) - 2*func(x0, *args) + func(x0 - dx, *args)) / (dx**2)
            else:
                # Use central difference for higher orders
                return (func(x0 + dx, *args) - func(x0 - dx, *args)) / (2*dx)
import warnings

logger = logging.getLogger(__name__)


@dataclass
class LocalSensitivityConfig:
    """Configuration for local sensitivity analysis."""
    
    # Analysis parameters
    perturbation_size: float = 1e-6
    finite_difference_method: str = "central"  # "forward", "backward", "central"
    step_size: float = 1e-6
    
    # Numerical differentiation parameters
    num_diff_order: int = 1  # Order of numerical differentiation
    num_diff_accuracy: int = 2  # Accuracy of numerical differentiation
    
    # Analysis parameters
    confidence_level: float = 0.95
    significance_level: float = 0.05


@dataclass
class LocalSensitivityResult:
    """Result of local sensitivity analysis."""
    
    success: bool
    method: str
    parameter_names: List[str]
    nominal_point: Dict[str, float]
    sensitivity_gradients: Dict[str, float]
    sensitivity_elasticities: Dict[str, float]
    analysis_time: float
    error_message: Optional[str] = None


class LocalSensitivityAnalyzer:
    """
    Local sensitivity analyzer for PBF-LB/M process parameters.
    
    This class provides local sensitivity analysis capabilities including
    derivative-based analysis and local parameter perturbation for understanding
    parameter influences at specific operating points.
    """
    
    def __init__(self, config: LocalSensitivityConfig = None):
        """Initialize the local sensitivity analyzer."""
        self.config = config or LocalSensitivityConfig()
        self.analysis_cache = {}
        
        logger.info("Local Sensitivity Analyzer initialized")
    
    def analyze_derivatives(
        self,
        model_function: Callable,
        nominal_point: Dict[str, float],
        parameter_names: List[str] = None
    ) -> LocalSensitivityResult:
        """
        Perform derivative-based local sensitivity analysis.
        
        Args:
            model_function: Function that takes parameter array and returns output
            nominal_point: Dictionary of nominal parameter values {name: value}
            parameter_names: List of parameter names (optional)
            
        Returns:
            LocalSensitivityResult: Derivative analysis results
        """
        try:
            start_time = datetime.now()
            
            if parameter_names is None:
                parameter_names = list(nominal_point.keys())
            
            # Convert nominal point to array
            nominal_array = np.array([nominal_point[name] for name in parameter_names])
            
            # Calculate derivatives using finite differences
            derivatives = self._calculate_derivatives(model_function, nominal_array, parameter_names)
            
            # Calculate elasticities (relative sensitivities)
            elasticities = self._calculate_elasticities(derivatives, nominal_point, parameter_names)
            
            # Calculate analysis time
            analysis_time = (datetime.now() - start_time).total_seconds()
            
            # Create result
            result = LocalSensitivityResult(
                success=True,
                method="Derivatives",
                parameter_names=parameter_names,
                nominal_point=nominal_point,
                sensitivity_gradients=derivatives,
                sensitivity_elasticities=elasticities,
                analysis_time=analysis_time
            )
            
            # Cache result
            self._cache_result("derivatives", result)
            
            logger.info(f"Derivative analysis completed: {analysis_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Error in derivative analysis: {e}")
            return LocalSensitivityResult(
                success=False,
                method="Derivatives",
                parameter_names=parameter_names or [],
                nominal_point=nominal_point,
                sensitivity_gradients={},
                sensitivity_elasticities={},
                analysis_time=0.0,
                error_message=str(e)
            )
    
    def analyze_perturbation(
        self,
        model_function: Callable,
        nominal_point: Dict[str, float],
        parameter_names: List[str] = None,
        perturbation_size: float = None
    ) -> LocalSensitivityResult:
        """
        Perform perturbation-based local sensitivity analysis.
        
        Args:
            model_function: Function that takes parameter array and returns output
            nominal_point: Dictionary of nominal parameter values {name: value}
            parameter_names: List of parameter names (optional)
            perturbation_size: Size of parameter perturbation (optional)
            
        Returns:
            LocalSensitivityResult: Perturbation analysis results
        """
        try:
            start_time = datetime.now()
            
            if parameter_names is None:
                parameter_names = list(nominal_point.keys())
            
            if perturbation_size is None:
                perturbation_size = self.config.perturbation_size
            
            # Calculate nominal output
            nominal_array = np.array([nominal_point[name] for name in parameter_names])
            nominal_output = model_function(nominal_array)
            
            # Calculate sensitivities using perturbation
            sensitivities = {}
            elasticities = {}
            
            for i, name in enumerate(parameter_names):
                # Create perturbed parameter array
                perturbed_array = nominal_array.copy()
                perturbed_array[i] += perturbation_size
                
                # Calculate perturbed output
                perturbed_output = model_function(perturbed_array)
                
                # Calculate sensitivity (finite difference)
                sensitivity = (perturbed_output - nominal_output) / perturbation_size
                sensitivities[name] = sensitivity
                
                # Calculate elasticity (relative sensitivity)
                if nominal_output != 0:
                    elasticity = (sensitivity * nominal_point[name]) / nominal_output
                else:
                    elasticity = 0.0
                elasticities[name] = elasticity
            
            # Calculate analysis time
            analysis_time = (datetime.now() - start_time).total_seconds()
            
            # Create result
            result = LocalSensitivityResult(
                success=True,
                method="Perturbation",
                parameter_names=parameter_names,
                nominal_point=nominal_point,
                sensitivity_gradients=sensitivities,
                sensitivity_elasticities=elasticities,
                analysis_time=analysis_time
            )
            
            # Cache result
            self._cache_result("perturbation", result)
            
            logger.info(f"Perturbation analysis completed: {analysis_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Error in perturbation analysis: {e}")
            return LocalSensitivityResult(
                success=False,
                method="Perturbation",
                parameter_names=parameter_names or [],
                nominal_point=nominal_point,
                sensitivity_gradients={},
                sensitivity_elasticities={},
                analysis_time=0.0,
                error_message=str(e)
            )
    
    def analyze_central_differences(
        self,
        model_function: Callable,
        nominal_point: Dict[str, float],
        parameter_names: List[str] = None,
        step_size: float = None
    ) -> LocalSensitivityResult:
        """
        Perform central difference local sensitivity analysis.
        
        Args:
            model_function: Function that takes parameter array and returns output
            nominal_point: Dictionary of nominal parameter values {name: value}
            parameter_names: List of parameter names (optional)
            step_size: Step size for central differences (optional)
            
        Returns:
            LocalSensitivityResult: Central difference analysis results
        """
        try:
            start_time = datetime.now()
            
            if parameter_names is None:
                parameter_names = list(nominal_point.keys())
            
            if step_size is None:
                step_size = self.config.step_size
            
            # Calculate nominal output
            nominal_array = np.array([nominal_point[name] for name in parameter_names])
            nominal_output = model_function(nominal_array)
            
            # Calculate sensitivities using central differences
            sensitivities = {}
            elasticities = {}
            
            for i, name in enumerate(parameter_names):
                # Create forward and backward perturbed arrays
                forward_array = nominal_array.copy()
                forward_array[i] += step_size
                
                backward_array = nominal_array.copy()
                backward_array[i] -= step_size
                
                # Calculate perturbed outputs
                forward_output = model_function(forward_array)
                backward_output = model_function(backward_array)
                
                # Calculate sensitivity (central difference)
                sensitivity = (forward_output - backward_output) / (2 * step_size)
                sensitivities[name] = sensitivity
                
                # Calculate elasticity (relative sensitivity)
                if nominal_output != 0:
                    elasticity = (sensitivity * nominal_point[name]) / nominal_output
                else:
                    elasticity = 0.0
                elasticities[name] = elasticity
            
            # Calculate analysis time
            analysis_time = (datetime.now() - start_time).total_seconds()
            
            # Create result
            result = LocalSensitivityResult(
                success=True,
                method="CentralDifferences",
                parameter_names=parameter_names,
                nominal_point=nominal_point,
                sensitivity_gradients=sensitivities,
                sensitivity_elasticities=elasticities,
                analysis_time=analysis_time
            )
            
            # Cache result
            self._cache_result("central_differences", result)
            
            logger.info(f"Central difference analysis completed: {analysis_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Error in central difference analysis: {e}")
            return LocalSensitivityResult(
                success=False,
                method="CentralDifferences",
                parameter_names=parameter_names or [],
                nominal_point=nominal_point,
                sensitivity_gradients={},
                sensitivity_elasticities={},
                analysis_time=0.0,
                error_message=str(e)
            )
    
    def analyze_automatic_differentiation(
        self,
        model_function: Callable,
        nominal_point: Dict[str, float],
        parameter_names: List[str] = None
    ) -> LocalSensitivityResult:
        """
        Perform automatic differentiation local sensitivity analysis.
        
        Args:
            model_function: Function that takes parameter array and returns output
            nominal_point: Dictionary of nominal parameter values {name: value}
            parameter_names: List of parameter names (optional)
            
        Returns:
            LocalSensitivityResult: Automatic differentiation analysis results
        """
        try:
            start_time = datetime.now()
            
            if parameter_names is None:
                parameter_names = list(nominal_point.keys())
            
            # Convert nominal point to array
            nominal_array = np.array([nominal_point[name] for name in parameter_names])
            
            # Use scipy's automatic differentiation
            derivatives = approx_fprime(
                nominal_array,
                model_function,
                epsilon=self.config.step_size
            )
            
            # Convert to dictionary
            sensitivities = {name: derivatives[i] for i, name in enumerate(parameter_names)}
            
            # Calculate elasticities
            nominal_output = model_function(nominal_array)
            elasticities = {}
            for name in parameter_names:
                if nominal_output != 0:
                    elasticity = (sensitivities[name] * nominal_point[name]) / nominal_output
                else:
                    elasticity = 0.0
                elasticities[name] = elasticity
            
            # Calculate analysis time
            analysis_time = (datetime.now() - start_time).total_seconds()
            
            # Create result
            result = LocalSensitivityResult(
                success=True,
                method="AutomaticDifferentiation",
                parameter_names=parameter_names,
                nominal_point=nominal_point,
                sensitivity_gradients=sensitivities,
                sensitivity_elasticities=elasticities,
                analysis_time=analysis_time
            )
            
            # Cache result
            self._cache_result("automatic_differentiation", result)
            
            logger.info(f"Automatic differentiation analysis completed: {analysis_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Error in automatic differentiation analysis: {e}")
            return LocalSensitivityResult(
                success=False,
                method="AutomaticDifferentiation",
                parameter_names=parameter_names or [],
                nominal_point=nominal_point,
                sensitivity_gradients={},
                sensitivity_elasticities={},
                analysis_time=0.0,
                error_message=str(e)
            )
    
    def _calculate_derivatives(
        self, 
        model_function: Callable, 
        nominal_array: np.ndarray, 
        parameter_names: List[str]
    ) -> Dict[str, float]:
        """Calculate derivatives using finite differences."""
        derivatives = {}
        
        for i, name in enumerate(parameter_names):
            # Create wrapper function for single parameter
            def wrapper(x):
                param_array = nominal_array.copy()
                param_array[i] = x
                return model_function(param_array)
            
            # Calculate derivative
            try:
                derivative_value = derivative(
                    wrapper,
                    nominal_array[i],
                    dx=self.config.step_size,
                    n=self.config.num_diff_order,
                    order=self.config.num_diff_accuracy
                )
                derivatives[name] = derivative_value
            except Exception as e:
                logger.warning(f"Error calculating derivative for {name}: {e}")
                derivatives[name] = 0.0
        
        return derivatives
    
    def _calculate_elasticities(
        self, 
        derivatives: Dict[str, float], 
        nominal_point: Dict[str, float], 
        parameter_names: List[str]
    ) -> Dict[str, float]:
        """Calculate elasticities (relative sensitivities)."""
        elasticities = {}
        
        # Calculate nominal output for elasticity calculation
        nominal_array = np.array([nominal_point[name] for name in parameter_names])
        nominal_output = self._calculate_nominal_output(nominal_array)
        
        for name in parameter_names:
            if nominal_output != 0:
                elasticity = (derivatives[name] * nominal_point[name]) / nominal_output
            else:
                elasticity = 0.0
            elasticities[name] = elasticity
        
        return elasticities
    
    def _calculate_nominal_output(self, nominal_array: np.ndarray) -> float:
        """Calculate nominal output (placeholder - should be implemented by user)."""
        # This is a placeholder - in practice, the user should provide
        # a way to calculate the nominal output
        return 1.0
    
    def _cache_result(self, method: str, result: LocalSensitivityResult):
        """Cache analysis result."""
        cache_key = f"{method}_{hash(str(result.parameter_names))}"
        self.analysis_cache[cache_key] = result
    
    def get_cached_result(self, method: str, parameter_names: List[str]) -> Optional[LocalSensitivityResult]:
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
                'perturbation_size': self.config.perturbation_size,
                'finite_difference_method': self.config.finite_difference_method,
                'step_size': self.config.step_size,
                'num_diff_order': self.config.num_diff_order,
                'num_diff_accuracy': self.config.num_diff_accuracy
            }
        }


class DerivativeAnalyzer(LocalSensitivityAnalyzer):
    """Specialized derivative-based sensitivity analyzer."""
    
    def __init__(self, config: LocalSensitivityConfig = None):
        super().__init__(config)
        self.method_name = "Derivatives"
    
    def analyze(self, model_function: Callable, nominal_point: Dict[str, float], 
                parameter_names: List[str] = None) -> LocalSensitivityResult:
        """Perform derivative analysis."""
        return self.analyze_derivatives(model_function, nominal_point, parameter_names)
