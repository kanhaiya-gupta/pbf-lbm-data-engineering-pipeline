"""
Design of Experiments for PBF-LB/M Process Analysis

This module provides comprehensive design of experiments (DOE) capabilities
for PBF-LB/M process parameter analysis, including factorial designs,
response surface methodology, and optimal experimental designs.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import logging
from datetime import datetime
from itertools import product
from scipy import stats
from scipy.optimize import minimize
import warnings

logger = logging.getLogger(__name__)


@dataclass
class DOEConfig:
    """Configuration for design of experiments."""
    
    # Design parameters
    design_type: str = "factorial"  # "factorial", "response_surface", "optimal"
    randomization: bool = True
    blocking: bool = False
    replication: int = 1
    
    # Factorial design parameters
    factorial_levels: int = 2  # 2^k or 3^k factorial
    center_points: int = 0  # Center points for factorial designs
    
    # Response surface parameters
    rs_design_type: str = "ccd"  # "ccd", "bbd", "d_optimal"
    rs_alpha: float = 1.414  # Alpha for central composite design
    
    # Optimal design parameters
    optimal_criterion: str = "d_optimal"  # "d_optimal", "a_optimal", "g_optimal"
    optimal_iterations: int = 1000
    
    # Analysis parameters
    confidence_level: float = 0.95
    significance_level: float = 0.05


@dataclass
class ExperimentalDesign:
    """Experimental design result."""
    
    design_type: str
    design_matrix: pd.DataFrame
    parameter_names: List[str]
    parameter_bounds: Dict[str, Tuple[float, float]]
    design_points: int
    design_quality: Dict[str, float]
    randomization_seed: Optional[int]
    creation_time: datetime


class ExperimentalDesigner:
    """
    Experimental designer for PBF-LB/M process analysis.
    
    This class provides comprehensive DOE capabilities including factorial designs,
    response surface methodology, and optimal experimental designs for systematic
    process parameter analysis.
    """
    
    def __init__(self, config: DOEConfig = None):
        """Initialize the experimental designer."""
        self.config = config or DOEConfig()
        self.design_cache = {}
        
        logger.info("Experimental Designer initialized")
    
    def create_factorial_design(
        self,
        parameter_bounds: Dict[str, Tuple[float, float]],
        parameter_names: List[str] = None,
        levels: int = None
    ) -> ExperimentalDesign:
        """
        Create factorial experimental design.
        
        Args:
            parameter_bounds: Dictionary of parameter bounds {name: (min, max)}
            parameter_names: List of parameter names (optional)
            levels: Number of levels (2 or 3)
            
        Returns:
            ExperimentalDesign: Factorial design
        """
        try:
            if parameter_names is None:
                parameter_names = list(parameter_bounds.keys())
            
            if levels is None:
                levels = self.config.factorial_levels
            
            n_factors = len(parameter_names)
            
            # Generate factorial design matrix
            if levels == 2:
                design_matrix = self._create_2k_factorial(n_factors)
            elif levels == 3:
                design_matrix = self._create_3k_factorial(n_factors)
            else:
                raise ValueError(f"Unsupported number of levels: {levels}")
            
            # Add center points if specified
            if self.config.center_points > 0:
                center_points = self._add_center_points(n_factors, self.config.center_points)
                design_matrix = pd.concat([design_matrix, center_points], ignore_index=True)
            
            # Scale design matrix to parameter bounds
            scaled_matrix = self._scale_design_matrix(design_matrix, parameter_bounds, parameter_names)
            
            # Randomize if requested
            if self.config.randomization:
                np.random.seed(self.config.randomization_seed if hasattr(self.config, 'randomization_seed') else None)
                scaled_matrix = scaled_matrix.sample(frac=1).reset_index(drop=True)
            
            # Calculate design quality metrics
            design_quality = self._calculate_design_quality(scaled_matrix, parameter_names)
            
            # Create design result
            design = ExperimentalDesign(
                design_type=f"{levels}^k_factorial",
                design_matrix=scaled_matrix,
                parameter_names=parameter_names,
                parameter_bounds=parameter_bounds,
                design_points=len(scaled_matrix),
                design_quality=design_quality,
                randomization_seed=getattr(self.config, 'randomization_seed', None),
                creation_time=datetime.now()
            )
            
            # Cache design
            self._cache_design("factorial", design)
            
            logger.info(f"Factorial design created: {levels}^{n_factors} with {len(scaled_matrix)} points")
            return design
            
        except Exception as e:
            logger.error(f"Error creating factorial design: {e}")
            raise
    
    def create_response_surface_design(
        self,
        parameter_bounds: Dict[str, Tuple[float, float]],
        parameter_names: List[str] = None,
        design_type: str = None
    ) -> ExperimentalDesign:
        """
        Create response surface experimental design.
        
        Args:
            parameter_bounds: Dictionary of parameter bounds {name: (min, max)}
            parameter_names: List of parameter names (optional)
            design_type: Type of response surface design ("ccd", "bbd", "d_optimal")
            
        Returns:
            ExperimentalDesign: Response surface design
        """
        try:
            if parameter_names is None:
                parameter_names = list(parameter_bounds.keys())
            
            if design_type is None:
                design_type = self.config.rs_design_type
            
            n_factors = len(parameter_names)
            
            # Generate response surface design matrix
            if design_type == "ccd":
                design_matrix = self._create_central_composite_design(n_factors)
            elif design_type == "bbd":
                design_matrix = self._create_box_behnken_design(n_factors)
            elif design_type == "d_optimal":
                design_matrix = self._create_d_optimal_design(n_factors)
            else:
                raise ValueError(f"Unsupported response surface design type: {design_type}")
            
            # Scale design matrix to parameter bounds
            scaled_matrix = self._scale_design_matrix(design_matrix, parameter_bounds, parameter_names)
            
            # Randomize if requested
            if self.config.randomization:
                np.random.seed(getattr(self.config, 'randomization_seed', None))
                scaled_matrix = scaled_matrix.sample(frac=1).reset_index(drop=True)
            
            # Calculate design quality metrics
            design_quality = self._calculate_design_quality(scaled_matrix, parameter_names)
            
            # Create design result
            design = ExperimentalDesign(
                design_type=f"response_surface_{design_type}",
                design_matrix=scaled_matrix,
                parameter_names=parameter_names,
                parameter_bounds=parameter_bounds,
                design_points=len(scaled_matrix),
                design_quality=design_quality,
                randomization_seed=getattr(self.config, 'randomization_seed', None),
                creation_time=datetime.now()
            )
            
            # Cache design
            self._cache_design("response_surface", design)
            
            logger.info(f"Response surface design created: {design_type} with {len(scaled_matrix)} points")
            return design
            
        except Exception as e:
            logger.error(f"Error creating response surface design: {e}")
            raise
    
    def create_optimal_design(
        self,
        parameter_bounds: Dict[str, Tuple[float, float]],
        parameter_names: List[str] = None,
        n_points: int = None,
        criterion: str = None
    ) -> ExperimentalDesign:
        """
        Create optimal experimental design.
        
        Args:
            parameter_bounds: Dictionary of parameter bounds {name: (min, max)}
            parameter_names: List of parameter names (optional)
            n_points: Number of design points
            criterion: Optimality criterion ("d_optimal", "a_optimal", "g_optimal")
            
        Returns:
            ExperimentalDesign: Optimal design
        """
        try:
            if parameter_names is None:
                parameter_names = list(parameter_bounds.keys())
            
            if n_points is None:
                n_points = len(parameter_names) * 3  # Default: 3x number of factors
            
            if criterion is None:
                criterion = self.config.optimal_criterion
            
            n_factors = len(parameter_names)
            
            # Generate initial random design
            np.random.seed(getattr(self.config, 'randomization_seed', None))
            initial_design = np.random.uniform(-1, 1, (n_points, n_factors))
            
            # Optimize design
            if criterion == "d_optimal":
                optimized_design = self._optimize_d_optimal(initial_design, n_factors)
            elif criterion == "a_optimal":
                optimized_design = self._optimize_a_optimal(initial_design, n_factors)
            elif criterion == "g_optimal":
                optimized_design = self._optimize_g_optimal(initial_design, n_factors)
            else:
                raise ValueError(f"Unsupported optimality criterion: {criterion}")
            
            # Create design matrix
            design_matrix = pd.DataFrame(optimized_design, columns=parameter_names)
            
            # Scale design matrix to parameter bounds
            scaled_matrix = self._scale_design_matrix(design_matrix, parameter_bounds, parameter_names)
            
            # Calculate design quality metrics
            design_quality = self._calculate_design_quality(scaled_matrix, parameter_names)
            design_quality[f'{criterion}_value'] = self._calculate_optimality_criterion(scaled_matrix, criterion)
            
            # Create design result
            design = ExperimentalDesign(
                design_type=f"optimal_{criterion}",
                design_matrix=scaled_matrix,
                parameter_names=parameter_names,
                parameter_bounds=parameter_bounds,
                design_points=len(scaled_matrix),
                design_quality=design_quality,
                randomization_seed=getattr(self.config, 'randomization_seed', None),
                creation_time=datetime.now()
            )
            
            # Cache design
            self._cache_design("optimal", design)
            
            logger.info(f"Optimal design created: {criterion} with {len(scaled_matrix)} points")
            return design
            
        except Exception as e:
            logger.error(f"Error creating optimal design: {e}")
            raise
    
    def _create_2k_factorial(self, n_factors: int) -> pd.DataFrame:
        """Create 2^k factorial design matrix."""
        # Generate all combinations of -1 and 1
        levels = [-1, 1]
        combinations = list(product(levels, repeat=n_factors))
        
        design_matrix = pd.DataFrame(combinations, columns=[f'x{i+1}' for i in range(n_factors)])
        return design_matrix
    
    def _create_3k_factorial(self, n_factors: int) -> pd.DataFrame:
        """Create 3^k factorial design matrix."""
        # Generate all combinations of -1, 0, and 1
        levels = [-1, 0, 1]
        combinations = list(product(levels, repeat=n_factors))
        
        design_matrix = pd.DataFrame(combinations, columns=[f'x{i+1}' for i in range(n_factors)])
        return design_matrix
    
    def _add_center_points(self, n_factors: int, n_center_points: int) -> pd.DataFrame:
        """Add center points to factorial design."""
        center_points = pd.DataFrame(
            np.zeros((n_center_points, n_factors)),
            columns=[f'x{i+1}' for i in range(n_factors)]
        )
        return center_points
    
    def _create_central_composite_design(self, n_factors: int) -> pd.DataFrame:
        """Create central composite design."""
        # Factorial portion (2^k)
        factorial_portion = self._create_2k_factorial(n_factors)
        
        # Axial points
        axial_points = []
        alpha = self.config.rs_alpha
        
        for i in range(n_factors):
            # Positive axial point
            axial_pos = np.zeros(n_factors)
            axial_pos[i] = alpha
            axial_points.append(axial_pos)
            
            # Negative axial point
            axial_neg = np.zeros(n_factors)
            axial_neg[i] = -alpha
            axial_points.append(axial_neg)
        
        axial_portion = pd.DataFrame(axial_points, columns=[f'x{i+1}' for i in range(n_factors)])
        
        # Center points
        center_portion = self._add_center_points(n_factors, 1)
        
        # Combine all portions
        design_matrix = pd.concat([factorial_portion, axial_portion, center_portion], ignore_index=True)
        return design_matrix
    
    def _create_box_behnken_design(self, n_factors: int) -> pd.DataFrame:
        """Create Box-Behnken design."""
        if n_factors < 3:
            raise ValueError("Box-Behnken design requires at least 3 factors")
        
        # Box-Behnken designs are more complex and depend on the number of factors
        # For simplicity, we'll create a basic version
        design_points = []
        
        # Create combinations for Box-Behnken
        for i in range(n_factors):
            for j in range(i+1, n_factors):
                # Create points where two factors are at ±1 and others at 0
                point = np.zeros(n_factors)
                point[i] = 1
                point[j] = 1
                design_points.append(point.copy())
                
                point[i] = 1
                point[j] = -1
                design_points.append(point.copy())
                
                point[i] = -1
                point[j] = 1
                design_points.append(point.copy())
                
                point[i] = -1
                point[j] = -1
                design_points.append(point.copy())
        
        # Add center points
        center_points = [np.zeros(n_factors) for _ in range(3)]
        design_points.extend(center_points)
        
        design_matrix = pd.DataFrame(design_points, columns=[f'x{i+1}' for i in range(n_factors)])
        return design_matrix
    
    def _create_d_optimal_design(self, n_factors: int) -> pd.DataFrame:
        """Create D-optimal design."""
        # For simplicity, create a Latin hypercube design
        n_points = n_factors * 3
        
        design_matrix = np.zeros((n_points, n_factors))
        for i in range(n_factors):
            design_matrix[:, i] = np.random.uniform(-1, 1, n_points)
        
        return pd.DataFrame(design_matrix, columns=[f'x{i+1}' for i in range(n_factors)])
    
    def _scale_design_matrix(
        self, 
        design_matrix: pd.DataFrame, 
        parameter_bounds: Dict[str, Tuple[float, float]], 
        parameter_names: List[str]
    ) -> pd.DataFrame:
        """Scale design matrix from [-1, 1] to parameter bounds."""
        scaled_matrix = design_matrix.copy()
        
        for i, name in enumerate(parameter_names):
            if name in parameter_bounds:
                min_val, max_val = parameter_bounds[name]
                # Scale from [-1, 1] to [min_val, max_val]
                scaled_matrix.iloc[:, i] = (scaled_matrix.iloc[:, i] + 1) / 2 * (max_val - min_val) + min_val
        
        return scaled_matrix
    
    def _calculate_design_quality(self, design_matrix: pd.DataFrame, parameter_names: List[str]) -> Dict[str, float]:
        """Calculate design quality metrics."""
        quality_metrics = {}
        
        # Calculate correlation matrix
        corr_matrix = design_matrix.corr()
        
        # Maximum correlation (should be low for good designs)
        max_corr = corr_matrix.abs().max().max()
        quality_metrics['max_correlation'] = max_corr
        
        # Condition number of correlation matrix
        try:
            condition_number = np.linalg.cond(corr_matrix.values)
            quality_metrics['condition_number'] = condition_number
        except:
            quality_metrics['condition_number'] = np.inf
        
        # Design efficiency (simplified)
        n_points = len(design_matrix)
        n_factors = len(parameter_names)
        efficiency = n_factors / n_points
        quality_metrics['design_efficiency'] = efficiency
        
        return quality_metrics
    
    def _optimize_d_optimal(self, initial_design: np.ndarray, n_factors: int) -> np.ndarray:
        """Optimize design for D-optimality."""
        def d_optimal_criterion(design_flat):
            design = design_flat.reshape(-1, n_factors)
            try:
                # Create design matrix with interactions
                X = self._create_model_matrix(design)
                det_XtX = np.linalg.det(X.T @ X)
                return -det_XtX  # Minimize negative determinant
            except:
                return 1e10  # Large penalty for singular matrices
        
        # Flatten design for optimization
        design_flat = initial_design.flatten()
        
        # Optimize
        result = minimize(d_optimal_criterion, design_flat, method='L-BFGS-B', 
                         bounds=[(-1, 1)] * len(design_flat))
        
        return result.x.reshape(-1, n_factors)
    
    def _optimize_a_optimal(self, initial_design: np.ndarray, n_factors: int) -> np.ndarray:
        """Optimize design for A-optimality."""
        def a_optimal_criterion(design_flat):
            design = design_flat.reshape(-1, n_factors)
            try:
                X = self._create_model_matrix(design)
                trace_inv_XtX = np.trace(np.linalg.inv(X.T @ X))
                return trace_inv_XtX
            except:
                return 1e10
        
        design_flat = initial_design.flatten()
        result = minimize(a_optimal_criterion, design_flat, method='L-BFGS-B',
                         bounds=[(-1, 1)] * len(design_flat))
        
        return result.x.reshape(-1, n_factors)
    
    def _optimize_g_optimal(self, initial_design: np.ndarray, n_factors: int) -> np.ndarray:
        """Optimize design for G-optimality."""
        def g_optimal_criterion(design_flat):
            design = design_flat.reshape(-1, n_factors)
            try:
                X = self._create_model_matrix(design)
                inv_XtX = np.linalg.inv(X.T @ X)
                max_variance = np.max(np.diag(X @ inv_XtX @ X.T))
                return max_variance
            except:
                return 1e10
        
        design_flat = initial_design.flatten()
        result = minimize(g_optimal_criterion, design_flat, method='L-BFGS-B',
                         bounds=[(-1, 1)] * len(design_flat))
        
        return result.x.reshape(-1, n_factors)
    
    def _create_model_matrix(self, design: np.ndarray) -> np.ndarray:
        """Create model matrix with main effects and interactions."""
        n_points, n_factors = design.shape
        
        # Start with intercept
        X = np.ones((n_points, 1))
        
        # Add main effects
        X = np.column_stack([X, design])
        
        # Add two-factor interactions
        for i in range(n_factors):
            for j in range(i+1, n_factors):
                interaction = design[:, i] * design[:, j]
                X = np.column_stack([X, interaction])
        
        return X
    
    def _calculate_optimality_criterion(self, design_matrix: pd.DataFrame, criterion: str) -> float:
        """Calculate optimality criterion value."""
        design = design_matrix.values
        X = self._create_model_matrix(design)
        
        try:
            if criterion == "d_optimal":
                return np.linalg.det(X.T @ X)
            elif criterion == "a_optimal":
                return np.trace(np.linalg.inv(X.T @ X))
            elif criterion == "g_optimal":
                inv_XtX = np.linalg.inv(X.T @ X)
                return np.max(np.diag(X @ inv_XtX @ X.T))
            else:
                return 0.0
        except:
            return 0.0
    
    def _cache_design(self, design_type: str, design: ExperimentalDesign):
        """Cache experimental design."""
        cache_key = f"{design_type}_{hash(str(design.parameter_names))}"
        self.design_cache[cache_key] = design
    
    def get_cached_design(self, design_type: str, parameter_names: List[str]) -> Optional[ExperimentalDesign]:
        """Get cached experimental design."""
        cache_key = f"{design_type}_{hash(str(parameter_names))}"
        return self.design_cache.get(cache_key)
    
    def clear_cache(self):
        """Clear design cache."""
        self.design_cache.clear()
    
    def get_design_statistics(self) -> Dict[str, Any]:
        """Get design statistics."""
        return {
            'cache_size': len(self.design_cache),
            'config': {
                'design_type': self.config.design_type,
                'factorial_levels': self.config.factorial_levels,
                'center_points': self.config.center_points,
                'rs_design_type': self.config.rs_design_type,
                'optimal_criterion': self.config.optimal_criterion
            }
        }


class FactorialDesign(ExperimentalDesigner):
    """Specialized factorial design creator."""
    
    def __init__(self, config: DOEConfig = None):
        super().__init__(config)
        self.design_type = "factorial"
    
    def create_design(self, parameter_bounds: Dict[str, Tuple[float, float]], 
                     parameter_names: List[str] = None, levels: int = 2) -> ExperimentalDesign:
        """Create factorial design."""
        return self.create_factorial_design(parameter_bounds, parameter_names, levels)


class ResponseSurfaceDesign(ExperimentalDesigner):
    """Specialized response surface design creator."""
    
    def __init__(self, config: DOEConfig = None):
        super().__init__(config)
        self.design_type = "response_surface"
    
    def create_design(self, parameter_bounds: Dict[str, Tuple[float, float]], 
                     parameter_names: List[str] = None, design_type: str = "ccd") -> ExperimentalDesign:
        """Create response surface design."""
        return self.create_response_surface_design(parameter_bounds, parameter_names, design_type)
