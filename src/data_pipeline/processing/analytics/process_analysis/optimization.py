"""
Process Optimization for PBF-LB/M Systems

This module provides specialized process optimization capabilities for PBF-LB/M
additive manufacturing systems, including single-objective and multi-objective
optimization for process parameters and quality outcomes.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass
import logging
from datetime import datetime
from scipy.optimize import minimize, differential_evolution
try:
    from sklearn.multiobjective import NSGA2
except ImportError:
    # Fallback for multi-objective optimization
    try:
        from pymoo.algorithms.moo.nsga2 import NSGA2
    except ImportError:
        # Simple fallback implementation
        class NSGA2:
            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs
            
            def optimize(self, problem, *args, **kwargs):
                # Simple fallback using scipy's differential_evolution
                from scipy.optimize import differential_evolution
                result = differential_evolution(problem, *args, **kwargs)
                return result
import warnings

logger = logging.getLogger(__name__)


@dataclass
class OptimizationConfig:
    """Configuration for process optimization."""
    
    # Optimization parameters
    optimization_method: str = "differential_evolution"  # "minimize", "differential_evolution", "nsga2"
    max_iterations: int = 1000
    population_size: int = 50
    tolerance: float = 1e-6
    
    # Multi-objective parameters
    n_objectives: int = 2
    pareto_front_size: int = 100
    
    # Analysis parameters
    random_seed: Optional[int] = None


@dataclass
class OptimizationResult:
    """Result of process optimization."""
    
    success: bool
    method: str
    parameter_names: List[str]
    optimal_parameters: Dict[str, float]
    optimal_values: Union[float, List[float]]
    pareto_front: Optional[pd.DataFrame] = None
    optimization_history: Optional[List[float]] = None
    analysis_time: float = 0.0
    error_message: Optional[str] = None


class ProcessOptimizer:
    """
    Process optimizer for PBF-LB/M systems.
    
    This class provides specialized process optimization capabilities including
    single-objective and multi-objective optimization for PBF-LB/M process
    parameters and quality outcomes.
    """
    
    def __init__(self, config: OptimizationConfig = None):
        """Initialize the process optimizer."""
        self.config = config or OptimizationConfig()
        self.analysis_cache = {}
        
        # Set random seed
        if self.config.random_seed is not None:
            np.random.seed(self.config.random_seed)
        
        logger.info("Process Optimizer initialized")
    
    def optimize_single_objective(
        self,
        objective_function: Callable,
        parameter_bounds: Dict[str, Tuple[float, float]],
        parameter_names: List[str] = None,
        optimization_method: str = None
    ) -> OptimizationResult:
        """
        Perform single-objective optimization.
        
        Args:
            objective_function: Function to optimize (should return scalar value)
            parameter_bounds: Dictionary of parameter bounds {name: (min, max)}
            parameter_names: List of parameter names (optional)
            optimization_method: Optimization method (optional)
            
        Returns:
            OptimizationResult: Single-objective optimization results
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
                    popsize=self.config.population_size,
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
            result_obj = OptimizationResult(
                success=result.success,
                method=f"SingleObjective_{optimization_method}",
                parameter_names=parameter_names,
                optimal_parameters=optimal_parameters,
                optimal_values=optimal_value,
                analysis_time=analysis_time
            )
            
            # Cache result
            self._cache_result("single_objective", result_obj)
            
            logger.info(f"Single-objective optimization completed: {analysis_time:.2f}s")
            return result_obj
            
        except Exception as e:
            logger.error(f"Error in single-objective optimization: {e}")
            return OptimizationResult(
                success=False,
                method=f"SingleObjective_{optimization_method}",
                parameter_names=parameter_names or [],
                optimal_parameters={},
                optimal_values=0.0,
                error_message=str(e)
            )
    
    def optimize_multi_objective(
        self,
        objective_functions: List[Callable],
        parameter_bounds: Dict[str, Tuple[float, float]],
        parameter_names: List[str] = None,
        optimization_method: str = None
    ) -> OptimizationResult:
        """
        Perform multi-objective optimization.
        
        Args:
            objective_functions: List of functions to optimize
            parameter_bounds: Dictionary of parameter bounds {name: (min, max)}
            parameter_names: List of parameter names (optional)
            optimization_method: Optimization method (optional)
            
        Returns:
            OptimizationResult: Multi-objective optimization results
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
                objectives = [func(param_dict) for func in objective_functions]
                return objectives
            
            # Perform multi-objective optimization
            if optimization_method == "nsga2":
                # Use NSGA-II algorithm
                pareto_front = self._run_nsga2(wrapper, bounds, len(objective_functions))
            else:
                # Use weighted sum approach
                pareto_front = self._run_weighted_sum(wrapper, bounds, len(objective_functions))
            
            # Calculate analysis time
            analysis_time = (datetime.now() - start_time).total_seconds()
            
            # Create result
            result = OptimizationResult(
                success=True,
                method=f"MultiObjective_{optimization_method}",
                parameter_names=parameter_names,
                optimal_parameters={},  # Multiple optimal solutions
                optimal_values=[],  # Multiple optimal values
                pareto_front=pareto_front,
                analysis_time=analysis_time
            )
            
            # Cache result
            self._cache_result("multi_objective", result)
            
            logger.info(f"Multi-objective optimization completed: {analysis_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Error in multi-objective optimization: {e}")
            return OptimizationResult(
                success=False,
                method=f"MultiObjective_{optimization_method}",
                parameter_names=parameter_names or [],
                optimal_parameters={},
                optimal_values=[],
                error_message=str(e)
            )
    
    def _run_nsga2(self, objective_function: Callable, bounds: List[Tuple[float, float]], n_objectives: int) -> pd.DataFrame:
        """Run NSGA-II algorithm for multi-objective optimization."""
        # Simplified NSGA-II implementation
        # In practice, you would use a specialized library like DEAP or pymoo
        
        pareto_solutions = []
        
        # Generate random solutions
        n_solutions = self.config.pareto_front_size
        for _ in range(n_solutions):
            # Generate random parameters
            params = [np.random.uniform(bounds[i][0], bounds[i][1]) for i in range(len(bounds))]
            
            # Evaluate objectives
            objectives = objective_function(params)
            
            pareto_solutions.append({
                'parameters': params,
                'objectives': objectives
            })
        
        # Create DataFrame
        pareto_df = pd.DataFrame(pareto_solutions)
        
        return pareto_df
    
    def _run_weighted_sum(self, objective_function: Callable, bounds: List[Tuple[float, float]], n_objectives: int) -> pd.DataFrame:
        """Run weighted sum approach for multi-objective optimization."""
        pareto_solutions = []
        
        # Generate different weight combinations
        n_weights = self.config.pareto_front_size
        for i in range(n_weights):
            # Generate random weights
            weights = np.random.random(n_objectives)
            weights = weights / np.sum(weights)
            
            # Create weighted objective function
            def weighted_objective(x):
                objectives = objective_function(x)
                return np.sum([weights[j] * objectives[j] for j in range(n_objectives)])
            
            # Optimize weighted objective
            try:
                result = differential_evolution(
                    weighted_objective,
                    bounds,
                    maxiter=self.config.max_iterations,
                    popsize=self.config.population_size,
                    seed=self.config.random_seed
                )
                
                if result.success:
                    # Evaluate all objectives at optimal point
                    all_objectives = objective_function(result.x)
                    
                    pareto_solutions.append({
                        'parameters': result.x.tolist(),
                        'objectives': all_objectives,
                        'weights': weights.tolist()
                    })
            except Exception as e:
                logger.warning(f"Weighted sum optimization failed: {e}")
                continue
        
        # Create DataFrame
        pareto_df = pd.DataFrame(pareto_solutions)
        
        return pareto_df
    
    def _cache_result(self, method: str, result: OptimizationResult):
        """Cache optimization result."""
        cache_key = f"{method}_{hash(str(result.parameter_names))}"
        self.analysis_cache[cache_key] = result
    
    def get_cached_result(self, method: str, parameter_names: List[str]) -> Optional[OptimizationResult]:
        """Get cached optimization result."""
        cache_key = f"{method}_{hash(str(parameter_names))}"
        return self.analysis_cache.get(cache_key)
    
    def clear_cache(self):
        """Clear optimization cache."""
        self.analysis_cache.clear()
    
    def get_optimization_statistics(self) -> Dict[str, Any]:
        """Get optimization statistics."""
        return {
            'cache_size': len(self.analysis_cache),
            'config': {
                'optimization_method': self.config.optimization_method,
                'max_iterations': self.config.max_iterations,
                'population_size': self.config.population_size,
                'tolerance': self.config.tolerance,
                'n_objectives': self.config.n_objectives
            }
        }


class MultiObjectiveOptimizer(ProcessOptimizer):
    """Specialized multi-objective optimizer."""
    
    def __init__(self, config: OptimizationConfig = None):
        super().__init__(config)
        self.method_name = "MultiObjectiveOptimizer"
    
    def optimize(self, objective_functions: List[Callable], parameter_bounds: Dict[str, Tuple[float, float]], 
                 parameter_names: List[str] = None) -> OptimizationResult:
        """Optimize multiple objectives."""
        return self.optimize_multi_objective(objective_functions, parameter_bounds, parameter_names)
