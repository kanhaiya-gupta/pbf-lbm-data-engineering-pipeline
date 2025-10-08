"""
Uncertainty Quantification for PBF-LB/M Process Analysis

This module provides comprehensive uncertainty quantification capabilities
including Monte Carlo analysis, Bayesian analysis, and uncertainty propagation
for PBF-LB/M process parameters and outcomes.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass
import logging
from datetime import datetime
from scipy import stats
from scipy.optimize import minimize
import warnings

# Try to import PyMC4 (the modern Bayesian library)
try:
    import pymc as pm
    import pytensor.tensor as pt
    PYMC_AVAILABLE = True
    PYMC_VERSION = "4"
except ImportError as e:
    PYMC_AVAILABLE = False
    PYMC_VERSION = None
    warnings.warn("PyMC4 not available. Install with: pip install pymc")

logger = logging.getLogger(__name__)


@dataclass
class UncertaintyConfig:
    """Configuration for uncertainty quantification."""
    
    # Monte Carlo parameters
    monte_carlo_samples: int = 10000
    monte_carlo_burn_in: int = 1000
    monte_carlo_thinning: int = 1
    
    # Bayesian parameters
    bayesian_samples: int = 5000
    bayesian_tune: int = 1000
    bayesian_chains: int = 2
    
    # Uncertainty propagation parameters
    propagation_method: str = "monte_carlo"  # "monte_carlo", "taylor", "polynomial_chaos"
    taylor_order: int = 2
    
    # Analysis parameters
    confidence_level: float = 0.95
    significance_level: float = 0.05
    random_seed: Optional[int] = None


@dataclass
class UncertaintyResult:
    """Result of uncertainty quantification analysis."""
    
    success: bool
    method: str
    parameter_names: List[str]
    parameter_distributions: Dict[str, Dict[str, Any]]
    output_statistics: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]
    sensitivity_analysis: Dict[str, float]
    analysis_time: float
    sample_size: int
    error_message: Optional[str] = None


class UncertaintyQuantifier:
    """
    Uncertainty quantifier for PBF-LB/M process analysis.
    
    This class provides comprehensive uncertainty quantification capabilities
    including Monte Carlo analysis, Bayesian analysis, and uncertainty propagation
    for understanding parameter and model uncertainties.
    """
    
    def __init__(self, config: UncertaintyConfig = None):
        """Initialize the uncertainty quantifier."""
        self.config = config or UncertaintyConfig()
        self.analysis_cache = {}
        
        if not PYMC_AVAILABLE:
            logger.warning("PyMC not available. Bayesian analysis will not be available.")
        else:
            logger.info(f"PyMC{PYMC_VERSION} available for Bayesian analysis")
        
        logger.info("Uncertainty Quantifier initialized")
    
    def analyze_monte_carlo(
        self,
        model_function: Callable,
        parameter_distributions: Dict[str, Dict[str, Any]],
        parameter_names: List[str] = None,
        n_samples: int = None
    ) -> UncertaintyResult:
        """
        Perform Monte Carlo uncertainty analysis.
        
        Args:
            model_function: Function that takes parameter array and returns output
            parameter_distributions: Dictionary of parameter distributions {name: {type: str, params: dict}}
            parameter_names: List of parameter names (optional)
            n_samples: Number of Monte Carlo samples (optional)
            
        Returns:
            UncertaintyResult: Monte Carlo analysis results
        """
        try:
            start_time = datetime.now()
            
            if parameter_names is None:
                parameter_names = list(parameter_distributions.keys())
            
            if n_samples is None:
                n_samples = self.config.monte_carlo_samples
            
            # Set random seed
            if self.config.random_seed is not None:
                np.random.seed(self.config.random_seed)
            
            # Generate parameter samples
            parameter_samples = self._generate_parameter_samples(
                parameter_distributions, parameter_names, n_samples
            )
            
            # Evaluate model for all samples
            model_outputs = self._evaluate_model_parallel(model_function, parameter_samples)
            
            # Calculate output statistics
            output_stats = self._calculate_output_statistics(model_outputs)
            
            # Calculate confidence intervals
            confidence_intervals = self._calculate_confidence_intervals(
                model_outputs, self.config.confidence_level
            )
            
            # Calculate sensitivity analysis
            sensitivity_analysis = self._calculate_sensitivity_analysis(
                parameter_samples, model_outputs, parameter_names
            )
            
            # Calculate analysis time
            analysis_time = (datetime.now() - start_time).total_seconds()
            
            # Create result
            result = UncertaintyResult(
                success=True,
                method="MonteCarlo",
                parameter_names=parameter_names,
                parameter_distributions=parameter_distributions,
                output_statistics=output_stats,
                confidence_intervals=confidence_intervals,
                sensitivity_analysis=sensitivity_analysis,
                analysis_time=analysis_time,
                sample_size=n_samples
            )
            
            # Cache result
            self._cache_result("monte_carlo", result)
            
            logger.info(f"Monte Carlo analysis completed: {analysis_time:.2f}s, {n_samples} samples")
            return result
            
        except Exception as e:
            logger.error(f"Error in Monte Carlo analysis: {e}")
            return UncertaintyResult(
                success=False,
                method="MonteCarlo",
                parameter_names=parameter_names or [],
                parameter_distributions=parameter_distributions,
                output_statistics={},
                confidence_intervals={},
                sensitivity_analysis={},
                analysis_time=0.0,
                sample_size=0,
                error_message=str(e)
            )
    
    def analyze_bayesian(
        self,
        model_function: Callable,
        parameter_distributions: Dict[str, Dict[str, Any]],
        parameter_names: List[str] = None,
        observed_data: Optional[np.ndarray] = None
    ) -> UncertaintyResult:
        """
        Perform Bayesian uncertainty analysis.
        
        Args:
            model_function: Function that takes parameter array and returns output
            parameter_distributions: Dictionary of parameter distributions {name: {type: str, params: dict}}
            parameter_names: List of parameter names (optional)
            observed_data: Observed data for Bayesian inference (optional)
            
        Returns:
            UncertaintyResult: Bayesian analysis results
        """
        try:
            start_time = datetime.now()
            
            if not PYMC_AVAILABLE:
                raise ImportError("PyMC4 not available for Bayesian analysis")
            
            if parameter_names is None:
                parameter_names = list(parameter_distributions.keys())
            
            # Set random seed
            if self.config.random_seed is not None:
                np.random.seed(self.config.random_seed)
            
            # Create Bayesian model
            with pm.Model() as model:
                # Define priors
                priors = {}
                for name in parameter_names:
                    dist_info = parameter_distributions[name]
                    dist_type = dist_info['type']
                    dist_params = dist_info['params']
                    
                    if dist_type == 'normal':
                        priors[name] = pm.Normal(name, mu=dist_params['mu'], sigma=dist_params['sigma'])
                    elif dist_type == 'uniform':
                        priors[name] = pm.Uniform(name, lower=dist_params['lower'], upper=dist_params['upper'])
                    elif dist_type == 'gamma':
                        priors[name] = pm.Gamma(name, alpha=dist_params['alpha'], beta=dist_params['beta'])
                    else:
                        raise ValueError(f"Unsupported distribution type: {dist_type}")
                
                # Define likelihood
                if observed_data is not None:
                    # Use observed data for likelihood
                    likelihood = pm.Normal('likelihood', mu=model_function, sigma=0.1, observed=observed_data)
                else:
                    # Use model function as deterministic
                    model_output = pm.Deterministic('model_output', model_function)
                
                # Sample from posterior
                trace = pm.sample(
                    draws=self.config.bayesian_samples,
                    tune=self.config.bayesian_tune,
                    chains=self.config.bayesian_chains,
                    random_seed=self.config.random_seed
                )
            
            # Extract samples
            parameter_samples = {}
            for name in parameter_names:
                parameter_samples[name] = trace[name].flatten()
            
            # Convert to array format
            param_array = np.column_stack([parameter_samples[name] for name in parameter_names])
            
            # Evaluate model for all samples
            model_outputs = self._evaluate_model_parallel(model_function, param_array)
            
            # Calculate output statistics
            output_stats = self._calculate_output_statistics(model_outputs)
            
            # Calculate confidence intervals
            confidence_intervals = self._calculate_confidence_intervals(
                model_outputs, self.config.confidence_level
            )
            
            # Calculate sensitivity analysis
            sensitivity_analysis = self._calculate_sensitivity_analysis(
                param_array, model_outputs, parameter_names
            )
            
            # Calculate analysis time
            analysis_time = (datetime.now() - start_time).total_seconds()
            
            # Create result
            result = UncertaintyResult(
                success=True,
                method="Bayesian",
                parameter_names=parameter_names,
                parameter_distributions=parameter_distributions,
                output_statistics=output_stats,
                confidence_intervals=confidence_intervals,
                sensitivity_analysis=sensitivity_analysis,
                analysis_time=analysis_time,
                sample_size=len(param_array)
            )
            
            # Cache result
            self._cache_result("bayesian", result)
            
            logger.info(f"Bayesian analysis completed: {analysis_time:.2f}s, {len(param_array)} samples")
            return result
            
        except Exception as e:
            logger.error(f"Error in Bayesian analysis: {e}")
            return UncertaintyResult(
                success=False,
                method="Bayesian",
                parameter_names=parameter_names or [],
                parameter_distributions=parameter_distributions,
                output_statistics={},
                confidence_intervals={},
                sensitivity_analysis={},
                analysis_time=0.0,
                sample_size=0,
                error_message=str(e)
            )
    
    def analyze_uncertainty_propagation(
        self,
        model_function: Callable,
        parameter_distributions: Dict[str, Dict[str, Any]],
        parameter_names: List[str] = None,
        method: str = None
    ) -> UncertaintyResult:
        """
        Perform uncertainty propagation analysis.
        
        Args:
            model_function: Function that takes parameter array and returns output
            parameter_distributions: Dictionary of parameter distributions {name: {type: str, params: dict}}
            parameter_names: List of parameter names (optional)
            method: Propagation method ("monte_carlo", "taylor", "polynomial_chaos")
            
        Returns:
            UncertaintyResult: Uncertainty propagation analysis results
        """
        try:
            start_time = datetime.now()
            
            if parameter_names is None:
                parameter_names = list(parameter_distributions.keys())
            
            if method is None:
                method = self.config.propagation_method
            
            if method == "monte_carlo":
                return self.analyze_monte_carlo(model_function, parameter_distributions, parameter_names)
            elif method == "taylor":
                return self._analyze_taylor_propagation(model_function, parameter_distributions, parameter_names)
            elif method == "polynomial_chaos":
                return self._analyze_polynomial_chaos_propagation(model_function, parameter_distributions, parameter_names)
            else:
                raise ValueError(f"Unsupported propagation method: {method}")
                
        except Exception as e:
            logger.error(f"Error in uncertainty propagation analysis: {e}")
            return UncertaintyResult(
                success=False,
                method=f"UncertaintyPropagation_{method}",
                parameter_names=parameter_names or [],
                parameter_distributions=parameter_distributions,
                output_statistics={},
                confidence_intervals={},
                sensitivity_analysis={},
                analysis_time=0.0,
                sample_size=0,
                error_message=str(e)
            )
    
    def _generate_parameter_samples(
        self, 
        parameter_distributions: Dict[str, Dict[str, Any]], 
        parameter_names: List[str], 
        n_samples: int
    ) -> np.ndarray:
        """Generate parameter samples from distributions."""
        samples = np.zeros((n_samples, len(parameter_names)))
        
        for i, name in enumerate(parameter_names):
            dist_info = parameter_distributions[name]
            dist_type = dist_info['type']
            dist_params = dist_info['params']
            
            if dist_type == 'normal':
                samples[:, i] = np.random.normal(dist_params['mu'], dist_params['sigma'], n_samples)
            elif dist_type == 'uniform':
                samples[:, i] = np.random.uniform(dist_params['lower'], dist_params['upper'], n_samples)
            elif dist_type == 'gamma':
                samples[:, i] = np.random.gamma(dist_params['alpha'], 1/dist_params['beta'], n_samples)
            elif dist_type == 'beta':
                samples[:, i] = np.random.beta(dist_params['alpha'], dist_params['beta'], n_samples)
            elif dist_type == 'lognormal':
                samples[:, i] = np.random.lognormal(dist_params['mu'], dist_params['sigma'], n_samples)
            else:
                raise ValueError(f"Unsupported distribution type: {dist_type}")
        
        return samples
    
    def _evaluate_model_parallel(self, model_function: Callable, param_values: np.ndarray) -> np.ndarray:
        """Evaluate model function for multiple parameter sets."""
        try:
            # Sequential evaluation for now
            return np.array([model_function(params) for params in param_values])
        except Exception as e:
            logger.warning(f"Model evaluation failed: {e}")
            return np.zeros(len(param_values))
    
    def _calculate_output_statistics(self, model_outputs: np.ndarray) -> Dict[str, float]:
        """Calculate output statistics."""
        return {
            'mean': np.mean(model_outputs),
            'std': np.std(model_outputs),
            'var': np.var(model_outputs),
            'min': np.min(model_outputs),
            'max': np.max(model_outputs),
            'median': np.median(model_outputs),
            'skewness': stats.skew(model_outputs),
            'kurtosis': stats.kurtosis(model_outputs),
            'percentile_5': np.percentile(model_outputs, 5),
            'percentile_95': np.percentile(model_outputs, 95)
        }
    
    def _calculate_confidence_intervals(
        self, 
        model_outputs: np.ndarray, 
        confidence_level: float
    ) -> Dict[str, Tuple[float, float]]:
        """Calculate confidence intervals."""
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        return {
            'confidence_interval': (
                np.percentile(model_outputs, lower_percentile),
                np.percentile(model_outputs, upper_percentile)
            ),
            'prediction_interval': (
                np.percentile(model_outputs, lower_percentile),
                np.percentile(model_outputs, upper_percentile)
            )
        }
    
    def _calculate_sensitivity_analysis(
        self, 
        parameter_samples: np.ndarray, 
        model_outputs: np.ndarray, 
        parameter_names: List[str]
    ) -> Dict[str, float]:
        """Calculate sensitivity analysis from samples."""
        sensitivities = {}
        
        # Calculate correlation-based sensitivities
        for i, name in enumerate(parameter_names):
            correlation = np.corrcoef(parameter_samples[:, i], model_outputs)[0, 1]
            sensitivities[f'correlation_{name}'] = abs(correlation)
        
        # Calculate partial variance sensitivities
        total_variance = np.var(model_outputs)
        for i, name in enumerate(parameter_names):
            # Calculate partial variance
            partial_variance = np.var(model_outputs) - np.var(model_outputs - parameter_samples[:, i])
            sensitivities[f'partial_variance_{name}'] = partial_variance / total_variance
        
        return sensitivities
    
    def _analyze_taylor_propagation(
        self, 
        model_function: Callable, 
        parameter_distributions: Dict[str, Dict[str, Any]], 
        parameter_names: List[str]
    ) -> UncertaintyResult:
        """Analyze uncertainty propagation using Taylor series expansion."""
        try:
            start_time = datetime.now()
            
            # Calculate nominal point (mean of distributions)
            nominal_point = {}
            for name in parameter_names:
                dist_info = parameter_distributions[name]
                dist_type = dist_info['type']
                dist_params = dist_info['params']
                
                if dist_type == 'normal':
                    nominal_point[name] = dist_params['mu']
                elif dist_type == 'uniform':
                    nominal_point[name] = (dist_params['lower'] + dist_params['upper']) / 2
                elif dist_type == 'gamma':
                    nominal_point[name] = dist_params['alpha'] / dist_params['beta']
                else:
                    nominal_point[name] = 0.0
            
            # Calculate nominal output
            nominal_array = np.array([nominal_point[name] for name in parameter_names])
            nominal_output = model_function(nominal_array)
            
            # Calculate first-order Taylor expansion
            taylor_variance = 0.0
            for i, name in enumerate(parameter_names):
                # Calculate partial derivative
                def wrapper(x):
                    param_array = nominal_array.copy()
                    param_array[i] = x
                    return model_function(param_array)
                
                try:
                    partial_derivative = derivative(wrapper, nominal_array[i], dx=1e-6)
                except:
                    partial_derivative = 0.0
                
                # Calculate parameter variance
                dist_info = parameter_distributions[name]
                dist_type = dist_info['type']
                dist_params = dist_info['params']
                
                if dist_type == 'normal':
                    param_variance = dist_params['sigma'] ** 2
                elif dist_type == 'uniform':
                    param_variance = (dist_params['upper'] - dist_params['lower']) ** 2 / 12
                elif dist_type == 'gamma':
                    param_variance = dist_params['alpha'] / (dist_params['beta'] ** 2)
                else:
                    param_variance = 0.0
                
                # Add to total variance
                taylor_variance += (partial_derivative ** 2) * param_variance
            
            # Calculate output statistics
            output_stats = {
                'mean': nominal_output,
                'std': np.sqrt(taylor_variance),
                'var': taylor_variance
            }
            
            # Calculate confidence intervals
            confidence_intervals = {
                'confidence_interval': (
                    nominal_output - 1.96 * np.sqrt(taylor_variance),
                    nominal_output + 1.96 * np.sqrt(taylor_variance)
                )
            }
            
            # Calculate analysis time
            analysis_time = (datetime.now() - start_time).total_seconds()
            
            # Create result
            result = UncertaintyResult(
                success=True,
                method="TaylorPropagation",
                parameter_names=parameter_names,
                parameter_distributions=parameter_distributions,
                output_statistics=output_stats,
                confidence_intervals=confidence_intervals,
                sensitivity_analysis={},
                analysis_time=analysis_time,
                sample_size=1
            )
            
            logger.info(f"Taylor propagation analysis completed: {analysis_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Error in Taylor propagation analysis: {e}")
            raise
    
    def _analyze_polynomial_chaos_propagation(
        self, 
        model_function: Callable, 
        parameter_distributions: Dict[str, Dict[str, Any]], 
        parameter_names: List[str]
    ) -> UncertaintyResult:
        """Analyze uncertainty propagation using polynomial chaos expansion."""
        # This is a simplified implementation
        # In practice, you would use specialized libraries like Chaospy or UQLab
        
        logger.warning("Polynomial chaos propagation not fully implemented, falling back to Monte Carlo")
        return self.analyze_monte_carlo(model_function, parameter_distributions, parameter_names)
    
    def _cache_result(self, method: str, result: UncertaintyResult):
        """Cache analysis result."""
        cache_key = f"{method}_{hash(str(result.parameter_names))}"
        self.analysis_cache[cache_key] = result
    
    def get_cached_result(self, method: str, parameter_names: List[str]) -> Optional[UncertaintyResult]:
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
            'pymc_available': PYMC_AVAILABLE,
            'pymc_version': PYMC_VERSION,
            'config': {
                'monte_carlo_samples': self.config.monte_carlo_samples,
                'bayesian_samples': self.config.bayesian_samples,
                'propagation_method': self.config.propagation_method,
                'confidence_level': self.config.confidence_level
            }
        }


class MonteCarloAnalyzer(UncertaintyQuantifier):
    """Specialized Monte Carlo uncertainty analyzer."""
    
    def __init__(self, config: UncertaintyConfig = None):
        super().__init__(config)
        self.method_name = "MonteCarlo"
    
    def analyze(self, model_function: Callable, parameter_distributions: Dict[str, Dict[str, Any]], 
                parameter_names: List[str] = None) -> UncertaintyResult:
        """Perform Monte Carlo analysis."""
        return self.analyze_monte_carlo(model_function, parameter_distributions, parameter_names)


class BayesianAnalyzer(UncertaintyQuantifier):
    """Specialized Bayesian uncertainty analyzer."""
    
    def __init__(self, config: UncertaintyConfig = None):
        super().__init__(config)
        self.method_name = "Bayesian"
    
    def analyze(self, model_function: Callable, parameter_distributions: Dict[str, Dict[str, Any]], 
                parameter_names: List[str] = None, observed_data: Optional[np.ndarray] = None) -> UncertaintyResult:
        """Perform Bayesian analysis."""
        return self.analyze_bayesian(model_function, parameter_distributions, parameter_names, observed_data)
