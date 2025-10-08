"""
Global Sensitivity Analysis for PBF-LB/M Process Parameters

This module provides global sensitivity analysis methods including Sobol indices
and Morris screening for understanding the global influence of process parameters
on PBF-LB/M outcomes.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass
import logging
from datetime import datetime
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import warnings

# Import SALib for sensitivity analysis
try:
    from SALib.sample import saltelli, morris
    from SALib.analyze import sobol, morris as morris_analyze
    SALIB_AVAILABLE = True
except ImportError:
    SALIB_AVAILABLE = False
    warnings.warn("SALib not available. Install with: pip install SALib")

logger = logging.getLogger(__name__)


@dataclass
class SensitivityConfig:
    """Configuration for sensitivity analysis."""
    
    # Analysis parameters
    sample_size: int = 1000
    confidence_level: float = 0.95
    random_seed: Optional[int] = None
    
    # Sobol analysis parameters
    sobol_order: int = 2  # First and second order indices
    sobol_n_bootstrap: int = 100
    
    # Morris analysis parameters
    morris_levels: int = 10
    morris_num_trajectories: int = 10
    
    # Performance parameters
    parallel_processing: bool = True
    max_workers: int = 4
    memory_limit_gb: float = 8.0


@dataclass
class SensitivityResult:
    """Result of sensitivity analysis."""
    
    success: bool
    method: str
    parameter_names: List[str]
    sensitivity_indices: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]
    analysis_time: float
    sample_size: int
    error_message: Optional[str] = None


class GlobalSensitivityAnalyzer:
    """
    Global sensitivity analyzer for PBF-LB/M process parameters.
    
    This class provides comprehensive global sensitivity analysis capabilities
    including Sobol indices and Morris screening for understanding parameter
    influences across the entire parameter space.
    """
    
    def __init__(self, config: SensitivityConfig = None):
        """Initialize the global sensitivity analyzer."""
        self.config = config or SensitivityConfig()
        self.analysis_cache = {}
        
        if not SALIB_AVAILABLE:
            logger.warning("SALib not available. Some methods may not work properly.")
        
        logger.info("Global Sensitivity Analyzer initialized")
    
    def analyze_sobol(
        self,
        model_function: Callable,
        parameter_bounds: Dict[str, Tuple[float, float]],
        parameter_names: List[str] = None
    ) -> SensitivityResult:
        """
        Perform Sobol sensitivity analysis.
        
        Args:
            model_function: Function that takes parameter array and returns output
            parameter_bounds: Dictionary of parameter bounds {name: (min, max)}
            parameter_names: List of parameter names (optional)
            
        Returns:
            SensitivityResult: Sobol analysis results
        """
        try:
            start_time = datetime.now()
            
            if not SALIB_AVAILABLE:
                return self._fallback_sobol_analysis(model_function, parameter_bounds, parameter_names)
            
            # Prepare parameter names and bounds
            if parameter_names is None:
                parameter_names = list(parameter_bounds.keys())
            
            # Create SALib problem definition
            problem = {
                'num_vars': len(parameter_names),
                'names': parameter_names,
                'bounds': [parameter_bounds[name] for name in parameter_names]
            }
            
            # Generate samples using Saltelli sequence
            param_values = saltelli.sample(
                problem, 
                self.config.sample_size,
                calc_second_order=True
            )
            
            # Evaluate model for all samples
            model_outputs = self._evaluate_model_parallel(model_function, param_values)
            
            # Calculate Sobol indices
            sobol_indices = sobol.analyze(
                problem, 
                model_outputs,
                calc_second_order=True,
                conf_level=self.config.confidence_level,
                num_resamples=self.config.sobol_n_bootstrap
            )
            
            # Extract results
            sensitivity_indices = {}
            confidence_intervals = {}
            
            # First order indices
            for i, name in enumerate(parameter_names):
                sensitivity_indices[f'S1_{name}'] = sobol_indices['S1'][i]
                if 'S1_conf' in sobol_indices:
                    confidence_intervals[f'S1_{name}'] = (
                        sobol_indices['S1_conf'][i][0],
                        sobol_indices['S1_conf'][i][1]
                    )
            
            # Total order indices
            for i, name in enumerate(parameter_names):
                sensitivity_indices[f'ST_{name}'] = sobol_indices['ST'][i]
                if 'ST_conf' in sobol_indices:
                    confidence_intervals[f'ST_{name}'] = (
                        sobol_indices['ST_conf'][i][0],
                        sobol_indices['ST_conf'][i][1]
                    )
            
            # Second order indices (if available)
            if 'S2' in sobol_indices:
                for i in range(len(parameter_names)):
                    for j in range(i+1, len(parameter_names)):
                        name_i, name_j = parameter_names[i], parameter_names[j]
                        sensitivity_indices[f'S2_{name_i}_{name_j}'] = sobol_indices['S2'][i, j]
            
            # Calculate analysis time
            analysis_time = (datetime.now() - start_time).total_seconds()
            
            # Create result
            result = SensitivityResult(
                success=True,
                method="Sobol",
                parameter_names=parameter_names,
                sensitivity_indices=sensitivity_indices,
                confidence_intervals=confidence_intervals,
                analysis_time=analysis_time,
                sample_size=len(param_values)
            )
            
            # Cache result
            self._cache_result("sobol", result)
            
            logger.info(f"Sobol analysis completed: {analysis_time:.2f}s, {len(param_values)} samples")
            return result
            
        except Exception as e:
            logger.error(f"Error in Sobol analysis: {e}")
            return SensitivityResult(
                success=False,
                method="Sobol",
                parameter_names=parameter_names or [],
                sensitivity_indices={},
                confidence_intervals={},
                analysis_time=0.0,
                sample_size=0,
                error_message=str(e)
            )
    
    def analyze_morris(
        self,
        model_function: Callable,
        parameter_bounds: Dict[str, Tuple[float, float]],
        parameter_names: List[str] = None
    ) -> SensitivityResult:
        """
        Perform Morris screening analysis.
        
        Args:
            model_function: Function that takes parameter array and returns output
            parameter_bounds: Dictionary of parameter bounds {name: (min, max)}
            parameter_names: List of parameter names (optional)
            
        Returns:
            SensitivityResult: Morris analysis results
        """
        try:
            start_time = datetime.now()
            
            if not SALIB_AVAILABLE:
                return self._fallback_morris_analysis(model_function, parameter_bounds, parameter_names)
            
            # Prepare parameter names and bounds
            if parameter_names is None:
                parameter_names = list(parameter_bounds.keys())
            
            # Create SALib problem definition
            problem = {
                'num_vars': len(parameter_names),
                'names': parameter_names,
                'bounds': [parameter_bounds[name] for name in parameter_names]
            }
            
            # Generate Morris samples
            param_values = morris.sample(
                problem,
                self.config.morris_num_trajectories,
                num_levels=self.config.morris_levels
            )
            
            # Evaluate model for all samples
            model_outputs = self._evaluate_model_parallel(model_function, param_values)
            
            # Calculate Morris indices
            morris_indices = morris_analyze.analyze(
                problem,
                param_values,
                model_outputs,
                conf_level=self.config.confidence_level,
                num_resamples=self.config.sobol_n_bootstrap
            )
            
            # Extract results
            sensitivity_indices = {}
            confidence_intervals = {}
            
            for i, name in enumerate(parameter_names):
                # Elementary effects mean (mu)
                sensitivity_indices[f'mu_{name}'] = morris_indices['mu'][i]
                if 'mu_conf' in morris_indices:
                    confidence_intervals[f'mu_{name}'] = (
                        morris_indices['mu_conf'][i][0],
                        morris_indices['mu_conf'][i][1]
                    )
                
                # Elementary effects standard deviation (sigma)
                sensitivity_indices[f'sigma_{name}'] = morris_indices['sigma'][i]
                if 'sigma_conf' in morris_indices:
                    confidence_intervals[f'sigma_{name}'] = (
                        morris_indices['sigma_conf'][i][0],
                        morris_indices['sigma_conf'][i][1]
                    )
                
                # Elementary effects mean of absolute values (mu_star)
                sensitivity_indices[f'mu_star_{name}'] = morris_indices['mu_star'][i]
                if 'mu_star_conf' in morris_indices:
                    confidence_intervals[f'mu_star_{name}'] = (
                        morris_indices['mu_star_conf'][i][0],
                        morris_indices['mu_star_conf'][i][1]
                    )
            
            # Calculate analysis time
            analysis_time = (datetime.now() - start_time).total_seconds()
            
            # Create result
            result = SensitivityResult(
                success=True,
                method="Morris",
                parameter_names=parameter_names,
                sensitivity_indices=sensitivity_indices,
                confidence_intervals=confidence_intervals,
                analysis_time=analysis_time,
                sample_size=len(param_values)
            )
            
            # Cache result
            self._cache_result("morris", result)
            
            logger.info(f"Morris analysis completed: {analysis_time:.2f}s, {len(param_values)} samples")
            return result
            
        except Exception as e:
            logger.error(f"Error in Morris analysis: {e}")
            return SensitivityResult(
                success=False,
                method="Morris",
                parameter_names=parameter_names or [],
                sensitivity_indices={},
                confidence_intervals={},
                analysis_time=0.0,
                sample_size=0,
                error_message=str(e)
            )
    
    def analyze_variance_based(
        self,
        model_function: Callable,
        parameter_bounds: Dict[str, Tuple[float, float]],
        parameter_names: List[str] = None
    ) -> SensitivityResult:
        """
        Perform variance-based sensitivity analysis using random sampling.
        
        Args:
            model_function: Function that takes parameter array and returns output
            parameter_bounds: Dictionary of parameter bounds {name: (min, max)}
            parameter_names: List of parameter names (optional)
            
        Returns:
            SensitivityResult: Variance-based analysis results
        """
        try:
            start_time = datetime.now()
            
            if parameter_names is None:
                parameter_names = list(parameter_bounds.keys())
            
            n_params = len(parameter_names)
            n_samples = self.config.sample_size
            
            # Generate random samples
            np.random.seed(self.config.random_seed)
            samples = np.random.uniform(0, 1, (n_samples, n_params))
            
            # Scale samples to parameter bounds
            param_samples = np.zeros_like(samples)
            for i, name in enumerate(parameter_names):
                min_val, max_val = parameter_bounds[name]
                param_samples[:, i] = samples[:, i] * (max_val - min_val) + min_val
            
            # Evaluate model
            model_outputs = self._evaluate_model_parallel(model_function, param_samples)
            
            # Calculate variance-based sensitivity indices
            total_variance = np.var(model_outputs)
            sensitivity_indices = {}
            
            # Calculate first-order indices using correlation
            for i, name in enumerate(parameter_names):
                correlation = np.corrcoef(param_samples[:, i], model_outputs)[0, 1]
                sensitivity_indices[f'correlation_{name}'] = abs(correlation)
            
            # Calculate partial variance indices
            for i, name in enumerate(parameter_names):
                # Fix parameter i and vary others
                fixed_samples = param_samples.copy()
                fixed_samples[:, i] = np.mean(param_samples[:, i])
                fixed_outputs = self._evaluate_model_parallel(model_function, fixed_samples)
                
                # Calculate partial variance
                partial_variance = np.var(model_outputs) - np.var(fixed_outputs)
                sensitivity_indices[f'partial_var_{name}'] = partial_variance / total_variance
            
            # Calculate analysis time
            analysis_time = (datetime.now() - start_time).total_seconds()
            
            # Create result
            result = SensitivityResult(
                success=True,
                method="VarianceBased",
                parameter_names=parameter_names,
                sensitivity_indices=sensitivity_indices,
                confidence_intervals={},
                analysis_time=analysis_time,
                sample_size=n_samples
            )
            
            logger.info(f"Variance-based analysis completed: {analysis_time:.2f}s, {n_samples} samples")
            return result
            
        except Exception as e:
            logger.error(f"Error in variance-based analysis: {e}")
            return SensitivityResult(
                success=False,
                method="VarianceBased",
                parameter_names=parameter_names or [],
                sensitivity_indices={},
                confidence_intervals={},
                analysis_time=0.0,
                sample_size=0,
                error_message=str(e)
            )
    
    def _evaluate_model_parallel(self, model_function: Callable, param_values: np.ndarray) -> np.ndarray:
        """Evaluate model function for multiple parameter sets."""
        try:
            if self.config.parallel_processing and len(param_values) > 100:
                # Use parallel processing for large sample sizes
                from concurrent.futures import ProcessPoolExecutor
                
                with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
                    results = list(executor.map(model_function, param_values))
                return np.array(results)
            else:
                # Sequential evaluation
                return np.array([model_function(params) for params in param_values])
                
        except Exception as e:
            logger.warning(f"Parallel evaluation failed, falling back to sequential: {e}")
            return np.array([model_function(params) for params in param_values])
    
    def _fallback_sobol_analysis(
        self,
        model_function: Callable,
        parameter_bounds: Dict[str, Tuple[float, float]],
        parameter_names: List[str] = None
    ) -> SensitivityResult:
        """Fallback Sobol analysis when SALib is not available."""
        logger.warning("SALib not available, using simplified Sobol analysis")
        
        if parameter_names is None:
            parameter_names = list(parameter_bounds.keys())
        
        # Simple Monte Carlo approach
        n_params = len(parameter_names)
        n_samples = self.config.sample_size
        
        # Generate samples
        np.random.seed(self.config.random_seed)
        samples = np.random.uniform(0, 1, (n_samples, n_params))
        
        # Scale to parameter bounds
        param_samples = np.zeros_like(samples)
        for i, name in enumerate(parameter_names):
            min_val, max_val = parameter_bounds[name]
            param_samples[:, i] = samples[:, i] * (max_val - min_val) + min_val
        
        # Evaluate model
        model_outputs = self._evaluate_model_parallel(model_function, param_samples)
        
        # Calculate simplified sensitivity indices
        sensitivity_indices = {}
        total_variance = np.var(model_outputs)
        
        for i, name in enumerate(parameter_names):
            # Calculate correlation-based sensitivity
            correlation = np.corrcoef(param_samples[:, i], model_outputs)[0, 1]
            sensitivity_indices[f'S1_{name}'] = abs(correlation)
            sensitivity_indices[f'ST_{name}'] = abs(correlation)
        
        return SensitivityResult(
            success=True,
            method="Sobol_Fallback",
            parameter_names=parameter_names,
            sensitivity_indices=sensitivity_indices,
            confidence_intervals={},
            analysis_time=0.0,
            sample_size=n_samples
        )
    
    def _fallback_morris_analysis(
        self,
        model_function: Callable,
        parameter_bounds: Dict[str, Tuple[float, float]],
        parameter_names: List[str] = None
    ) -> SensitivityResult:
        """Fallback Morris analysis when SALib is not available."""
        logger.warning("SALib not available, using simplified Morris analysis")
        
        if parameter_names is None:
            parameter_names = list(parameter_bounds.keys())
        
        # Simple Morris-like analysis
        n_params = len(parameter_names)
        sensitivity_indices = {}
        
        # Calculate elementary effects for each parameter
        for i, name in enumerate(parameter_names):
            # Generate two samples differing only in parameter i
            base_sample = np.random.uniform(0, 1, n_params)
            delta_sample = base_sample.copy()
            delta_sample[i] += 0.1  # Small perturbation
            
            # Scale to parameter bounds
            base_params = np.zeros(n_params)
            delta_params = np.zeros(n_params)
            for j, param_name in enumerate(parameter_names):
                min_val, max_val = parameter_bounds[param_name]
                base_params[j] = base_sample[j] * (max_val - min_val) + min_val
                delta_params[j] = delta_sample[j] * (max_val - min_val) + min_val
            
            # Evaluate model
            base_output = model_function(base_params)
            delta_output = model_function(delta_params)
            
            # Calculate elementary effect
            param_range = parameter_bounds[name][1] - parameter_bounds[name][0]
            elementary_effect = (delta_output - base_output) / (0.1 * param_range)
            
            sensitivity_indices[f'mu_star_{name}'] = abs(elementary_effect)
        
        return SensitivityResult(
            success=True,
            method="Morris_Fallback",
            parameter_names=parameter_names,
            sensitivity_indices=sensitivity_indices,
            confidence_intervals={},
            analysis_time=0.0,
            sample_size=2 * n_params
        )
    
    def _cache_result(self, method: str, result: SensitivityResult):
        """Cache analysis result."""
        cache_key = f"{method}_{hash(str(result.parameter_names))}"
        self.analysis_cache[cache_key] = result
    
    def get_cached_result(self, method: str, parameter_names: List[str]) -> Optional[SensitivityResult]:
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
            'salib_available': SALIB_AVAILABLE,
            'config': {
                'sample_size': self.config.sample_size,
                'confidence_level': self.config.confidence_level,
                'parallel_processing': self.config.parallel_processing,
                'max_workers': self.config.max_workers
            }
        }


class SobolAnalyzer(GlobalSensitivityAnalyzer):
    """Specialized Sobol sensitivity analyzer."""
    
    def __init__(self, config: SensitivityConfig = None):
        super().__init__(config)
        self.method_name = "Sobol"
    
    def analyze(self, model_function: Callable, parameter_bounds: Dict[str, Tuple[float, float]], 
                parameter_names: List[str] = None) -> SensitivityResult:
        """Perform Sobol analysis."""
        return self.analyze_sobol(model_function, parameter_bounds, parameter_names)


class MorrisAnalyzer(GlobalSensitivityAnalyzer):
    """Specialized Morris sensitivity analyzer."""
    
    def __init__(self, config: SensitivityConfig = None):
        super().__init__(config)
        self.method_name = "Morris"
    
    def analyze(self, model_function: Callable, parameter_bounds: Dict[str, Tuple[float, float]], 
                parameter_names: List[str] = None) -> SensitivityResult:
        """Perform Morris analysis."""
        return self.analyze_morris(model_function, parameter_bounds, parameter_names)
