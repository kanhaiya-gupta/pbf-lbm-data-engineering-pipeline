"""
Experiment Design for PBF-LB/M Virtual Environment

This module provides experiment design capabilities including virtual experiment
design, parameter sweep design, and Design of Experiments (DoE) for PBF-LB/M
virtual testing and simulation environments.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import logging
from datetime import datetime, timedelta
from enum import Enum
import uuid
import json
import asyncio
from itertools import product, combinations
from scipy.stats import uniform, norm
import warnings

logger = logging.getLogger(__name__)


class ExperimentType(Enum):
    """Experiment type enumeration."""
    PARAMETER_SWEEP = "parameter_sweep"
    FACTORIAL = "factorial"
    RESPONSE_SURFACE = "response_surface"
    OPTIMIZATION = "optimization"
    SENSITIVITY = "sensitivity"
    VALIDATION = "validation"


class DesignType(Enum):
    """Design type enumeration."""
    FULL_FACTORIAL = "full_factorial"
    FRACTIONAL_FACTORIAL = "fractional_factorial"
    CENTRAL_COMPOSITE = "central_composite"
    BOX_BEHNKEN = "box_behnken"
    LATIN_HYPERCUBE = "latin_hypercube"
    SOBOL_SEQUENCE = "sobol_sequence"


@dataclass
class ExperimentConfig:
    """Experiment configuration."""
    
    experiment_id: str
    name: str
    experiment_type: ExperimentType
    design_type: DesignType
    
    # Parameters
    parameters: Dict[str, Dict[str, Any]]
    responses: List[str]
    created_at: datetime
    updated_at: datetime
    
    # Design parameters
    sample_size: int = 100
    random_seed: int = 42
    
    # Constraints
    constraints: Dict[str, Any] = None


@dataclass
class ExperimentResult:
    """Experiment result."""
    
    experiment_id: str
    run_id: str
    timestamp: datetime
    
    # Input parameters
    input_parameters: Dict[str, Any]
    
    # Output responses
    output_responses: Dict[str, Any]
    
    # Metadata
    execution_time: float
    status: str
    error_message: Optional[str] = None


class VirtualExperimentDesigner:
    """
    Virtual experiment designer for PBF-LB/M virtual environment.
    
    This class provides comprehensive experiment design capabilities including
    virtual experiment design, parameter optimization, and experimental planning
    for PBF-LB/M virtual testing and simulation environments.
    """
    
    def __init__(self):
        """Initialize the virtual experiment designer."""
        self.experiment_configs = {}
        self.experiment_results = {}
        self.design_cache = {}
        
        logger.info("Virtual Experiment Designer initialized")
    
    async def create_experiment(
        self,
        name: str,
        experiment_type: ExperimentType,
        design_type: DesignType,
        parameters: Dict[str, Dict[str, Any]],
        responses: List[str],
        sample_size: int = 100
    ) -> str:
        """
        Create virtual experiment.
        
        Args:
            name: Experiment name
            experiment_type: Type of experiment
            design_type: Design type
            parameters: Parameter definitions
            responses: Response variables
            sample_size: Sample size
            
        Returns:
            str: Experiment ID
        """
        try:
            experiment_id = str(uuid.uuid4())
            
            config = ExperimentConfig(
                experiment_id=experiment_id,
                name=name,
                experiment_type=experiment_type,
                design_type=design_type,
                parameters=parameters,
                responses=responses,
                sample_size=sample_size,
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            
            self.experiment_configs[experiment_id] = config
            
            # Generate experimental design
            design_matrix = await self._generate_design_matrix(config)
            self.design_cache[experiment_id] = design_matrix
            
            logger.info(f"Virtual experiment created: {experiment_id}")
            return experiment_id
            
        except Exception as e:
            logger.error(f"Error creating virtual experiment: {e}")
            return ""
    
    async def run_experiment(
        self,
        experiment_id: str,
        simulation_function: callable = None
    ) -> List[ExperimentResult]:
        """
        Run virtual experiment.
        
        Args:
            experiment_id: Experiment ID
            simulation_function: Simulation function to execute
            
        Returns:
            List[ExperimentResult]: Experiment results
        """
        try:
            if experiment_id not in self.experiment_configs:
                raise ValueError(f"Experiment not found: {experiment_id}")
            
            config = self.experiment_configs[experiment_id]
            design_matrix = self.design_cache[experiment_id]
            
            results = []
            
            # Run experiments
            for i, parameter_set in enumerate(design_matrix):
                run_id = str(uuid.uuid4())
                start_time = datetime.now()
                
                try:
                    # Execute simulation
                    if simulation_function:
                        output_responses = await simulation_function(parameter_set)
                    else:
                        # Default simulation (for testing)
                        output_responses = await self._default_simulation(parameter_set, config.responses)
                    
                    execution_time = (datetime.now() - start_time).total_seconds()
                    
                    result = ExperimentResult(
                        experiment_id=experiment_id,
                        run_id=run_id,
                        timestamp=start_time,
                        input_parameters=parameter_set,
                        output_responses=output_responses,
                        execution_time=execution_time,
                        status="completed"
                    )
                    
                    results.append(result)
                    
                except Exception as e:
                    execution_time = (datetime.now() - start_time).total_seconds()
                    
                    result = ExperimentResult(
                        experiment_id=experiment_id,
                        run_id=run_id,
                        timestamp=start_time,
                        input_parameters=parameter_set,
                        output_responses={},
                        execution_time=execution_time,
                        status="failed",
                        error_message=str(e)
                    )
                    
                    results.append(result)
            
            # Store results
            self.experiment_results[experiment_id] = results
            
            logger.info(f"Virtual experiment completed: {experiment_id}, {len(results)} runs")
            return results
            
        except Exception as e:
            logger.error(f"Error running virtual experiment: {e}")
            return []
    
    async def analyze_experiment_results(
        self,
        experiment_id: str
    ) -> Dict[str, Any]:
        """
        Analyze experiment results.
        
        Args:
            experiment_id: Experiment ID
            
        Returns:
            Dict: Analysis results
        """
        try:
            if experiment_id not in self.experiment_results:
                raise ValueError(f"Experiment results not found: {experiment_id}")
            
            results = self.experiment_results[experiment_id]
            config = self.experiment_configs[experiment_id]
            
            # Convert results to DataFrame
            df = self._results_to_dataframe(results)
            
            # Perform analysis based on experiment type
            if config.experiment_type == ExperimentType.PARAMETER_SWEEP:
                analysis = await self._analyze_parameter_sweep(df, config)
            elif config.experiment_type == ExperimentType.FACTORIAL:
                analysis = await self._analyze_factorial(df, config)
            elif config.experiment_type == ExperimentType.RESPONSE_SURFACE:
                analysis = await self._analyze_response_surface(df, config)
            elif config.experiment_type == ExperimentType.OPTIMIZATION:
                analysis = await self._analyze_optimization(df, config)
            elif config.experiment_type == ExperimentType.SENSITIVITY:
                analysis = await self._analyze_sensitivity(df, config)
            else:
                analysis = await self._analyze_general(df, config)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing experiment results: {e}")
            return {}
    
    async def _generate_design_matrix(self, config: ExperimentConfig) -> List[Dict[str, Any]]:
        """Generate experimental design matrix."""
        try:
            if config.design_type == DesignType.FULL_FACTORIAL:
                return self._generate_full_factorial(config)
            elif config.design_type == DesignType.FRACTIONAL_FACTORIAL:
                return self._generate_fractional_factorial(config)
            elif config.design_type == DesignType.CENTRAL_COMPOSITE:
                return self._generate_central_composite(config)
            elif config.design_type == DesignType.BOX_BEHNKEN:
                return self._generate_box_behnken(config)
            elif config.design_type == DesignType.LATIN_HYPERCUBE:
                return self._generate_latin_hypercube(config)
            elif config.design_type == DesignType.SOBOL_SEQUENCE:
                return self._generate_sobol_sequence(config)
            else:
                return self._generate_random_design(config)
                
        except Exception as e:
            logger.error(f"Error generating design matrix: {e}")
            return []
    
    def _generate_full_factorial(self, config: ExperimentConfig) -> List[Dict[str, Any]]:
        """Generate full factorial design."""
        try:
            # Extract parameter levels
            parameter_levels = {}
            for param_name, param_config in config.parameters.items():
                if param_config['type'] == 'categorical':
                    parameter_levels[param_name] = param_config['levels']
                else:  # continuous
                    # Generate levels for continuous parameters
                    min_val = param_config['min']
                    max_val = param_config['max']
                    n_levels = param_config.get('levels', 3)
                    parameter_levels[param_name] = np.linspace(min_val, max_val, n_levels)
            
            # Generate all combinations
            design_matrix = []
            for combination in product(*parameter_levels.values()):
                parameter_set = dict(zip(parameter_levels.keys(), combination))
                design_matrix.append(parameter_set)
            
            return design_matrix
            
        except Exception as e:
            logger.error(f"Error generating full factorial design: {e}")
            return []
    
    def _generate_fractional_factorial(self, config: ExperimentConfig) -> List[Dict[str, Any]]:
        """Generate fractional factorial design."""
        try:
            # Simplified fractional factorial design
            # In real implementation, this would use proper fractional factorial methods
            
            # Generate full factorial first
            full_design = self._generate_full_factorial(config)
            
            # Sample a fraction
            fraction_size = min(len(full_design), config.sample_size)
            np.random.seed(config.random_seed)
            indices = np.random.choice(len(full_design), fraction_size, replace=False)
            
            return [full_design[i] for i in indices]
            
        except Exception as e:
            logger.error(f"Error generating fractional factorial design: {e}")
            return []
    
    def _generate_central_composite(self, config: ExperimentConfig) -> List[Dict[str, Any]]:
        """Generate central composite design."""
        try:
            # Simplified central composite design
            # In real implementation, this would use proper CCD methods
            
            design_matrix = []
            parameter_names = list(config.parameters.keys())
            n_params = len(parameter_names)
            
            # Generate factorial points
            for i in range(2**n_params):
                parameter_set = {}
                for j, param_name in enumerate(parameter_names):
                    param_config = config.parameters[param_name]
                    if param_config['type'] == 'continuous':
                        min_val = param_config['min']
                        max_val = param_config['max']
                        # Alternate between min and max
                        value = max_val if (i >> j) & 1 else min_val
                        parameter_set[param_name] = value
                    else:
                        # Categorical parameter
                        levels = param_config['levels']
                        parameter_set[param_name] = levels[i % len(levels)]
                
                design_matrix.append(parameter_set)
            
            # Add center points
            center_point = {}
            for param_name, param_config in config.parameters.items():
                if param_config['type'] == 'continuous':
                    min_val = param_config['min']
                    max_val = param_config['max']
                    center_point[param_name] = (min_val + max_val) / 2
                else:
                    # Use first level for categorical
                    center_point[param_name] = param_config['levels'][0]
            
            # Add multiple center points
            for _ in range(max(3, config.sample_size - len(design_matrix))):
                design_matrix.append(center_point.copy())
            
            return design_matrix[:config.sample_size]
            
        except Exception as e:
            logger.error(f"Error generating central composite design: {e}")
            return []
    
    def _generate_box_behnken(self, config: ExperimentConfig) -> List[Dict[str, Any]]:
        """Generate Box-Behnken design."""
        try:
            # Simplified Box-Behnken design
            # In real implementation, this would use proper BBD methods
            
            design_matrix = []
            parameter_names = list(config.parameters.keys())
            
            # Generate Box-Behnken points
            for i in range(config.sample_size):
                parameter_set = {}
                for param_name, param_config in config.parameters.items():
                    if param_config['type'] == 'continuous':
                        min_val = param_config['min']
                        max_val = param_config['max']
                        # Generate values around center
                        center = (min_val + max_val) / 2
                        range_val = (max_val - min_val) / 2
                        value = center + np.random.normal(0, range_val * 0.3)
                        parameter_set[param_name] = max(min_val, min(max_val, value))
                    else:
                        # Categorical parameter
                        levels = param_config['levels']
                        parameter_set[param_name] = np.random.choice(levels)
                
                design_matrix.append(parameter_set)
            
            return design_matrix
            
        except Exception as e:
            logger.error(f"Error generating Box-Behnken design: {e}")
            return []
    
    def _generate_latin_hypercube(self, config: ExperimentConfig) -> List[Dict[str, Any]]:
        """Generate Latin Hypercube design."""
        try:
            design_matrix = []
            parameter_names = list(config.parameters.keys())
            n_params = len(parameter_names)
            
            # Generate Latin Hypercube samples
            np.random.seed(config.random_seed)
            lhs_samples = np.random.uniform(0, 1, (config.sample_size, n_params))
            
            for i in range(config.sample_size):
                parameter_set = {}
                for j, param_name in enumerate(parameter_names):
                    param_config = config.parameters[param_name]
                    if param_config['type'] == 'continuous':
                        min_val = param_config['min']
                        max_val = param_config['max']
                        # Scale to parameter range
                        value = min_val + lhs_samples[i, j] * (max_val - min_val)
                        parameter_set[param_name] = value
                    else:
                        # Categorical parameter
                        levels = param_config['levels']
                        level_index = int(lhs_samples[i, j] * len(levels))
                        level_index = min(level_index, len(levels) - 1)
                        parameter_set[param_name] = levels[level_index]
                
                design_matrix.append(parameter_set)
            
            return design_matrix
            
        except Exception as e:
            logger.error(f"Error generating Latin Hypercube design: {e}")
            return []
    
    def _generate_sobol_sequence(self, config: ExperimentConfig) -> List[Dict[str, Any]]:
        """Generate Sobol sequence design."""
        try:
            # Simplified Sobol sequence
            # In real implementation, this would use proper Sobol sequence generation
            
            design_matrix = []
            parameter_names = list(config.parameters.keys())
            n_params = len(parameter_names)
            
            # Generate Sobol-like sequence
            np.random.seed(config.random_seed)
            sobol_samples = np.random.uniform(0, 1, (config.sample_size, n_params))
            
            for i in range(config.sample_size):
                parameter_set = {}
                for j, param_name in enumerate(parameter_names):
                    param_config = config.parameters[param_name]
                    if param_config['type'] == 'continuous':
                        min_val = param_config['min']
                        max_val = param_config['max']
                        # Scale to parameter range
                        value = min_val + sobol_samples[i, j] * (max_val - min_val)
                        parameter_set[param_name] = value
                    else:
                        # Categorical parameter
                        levels = param_config['levels']
                        level_index = int(sobol_samples[i, j] * len(levels))
                        level_index = min(level_index, len(levels) - 1)
                        parameter_set[param_name] = levels[level_index]
                
                design_matrix.append(parameter_set)
            
            return design_matrix
            
        except Exception as e:
            logger.error(f"Error generating Sobol sequence design: {e}")
            return []
    
    def _generate_random_design(self, config: ExperimentConfig) -> List[Dict[str, Any]]:
        """Generate random design."""
        try:
            design_matrix = []
            np.random.seed(config.random_seed)
            
            for i in range(config.sample_size):
                parameter_set = {}
                for param_name, param_config in config.parameters.items():
                    if param_config['type'] == 'continuous':
                        min_val = param_config['min']
                        max_val = param_config['max']
                        value = np.random.uniform(min_val, max_val)
                        parameter_set[param_name] = value
                    else:
                        # Categorical parameter
                        levels = param_config['levels']
                        parameter_set[param_name] = np.random.choice(levels)
                
                design_matrix.append(parameter_set)
            
            return design_matrix
            
        except Exception as e:
            logger.error(f"Error generating random design: {e}")
            return []
    
    async def _default_simulation(self, parameters: Dict[str, Any], responses: List[str]) -> Dict[str, Any]:
        """Default simulation function for testing."""
        try:
            # Simulate some responses based on parameters
            output_responses = {}
            
            for response in responses:
                if response == 'quality_score':
                    # Simulate quality score based on parameters
                    quality = 0.8 + np.random.normal(0, 0.1)
                    output_responses[response] = max(0.0, min(1.0, quality))
                elif response == 'defect_probability':
                    # Simulate defect probability
                    defect_prob = 0.05 + np.random.normal(0, 0.02)
                    output_responses[response] = max(0.0, min(1.0, defect_prob))
                elif response == 'dimensional_accuracy':
                    # Simulate dimensional accuracy
                    accuracy = 0.95 + np.random.normal(0, 0.02)
                    output_responses[response] = max(0.0, min(1.0, accuracy))
                elif response == 'surface_roughness':
                    # Simulate surface roughness
                    roughness = 0.1 + np.random.normal(0, 0.02)
                    output_responses[response] = max(0.0, roughness)
                else:
                    # Default response
                    output_responses[response] = np.random.uniform(0, 1)
            
            return output_responses
            
        except Exception as e:
            logger.error(f"Error in default simulation: {e}")
            return {}
    
    def _results_to_dataframe(self, results: List[ExperimentResult]) -> pd.DataFrame:
        """Convert experiment results to DataFrame."""
        try:
            data = []
            for result in results:
                row = result.input_parameters.copy()
                row.update(result.output_responses)
                row['run_id'] = result.run_id
                row['timestamp'] = result.timestamp
                row['execution_time'] = result.execution_time
                row['status'] = result.status
                data.append(row)
            
            return pd.DataFrame(data)
            
        except Exception as e:
            logger.error(f"Error converting results to DataFrame: {e}")
            return pd.DataFrame()
    
    async def _analyze_parameter_sweep(self, df: pd.DataFrame, config: ExperimentConfig) -> Dict[str, Any]:
        """Analyze parameter sweep results."""
        try:
            analysis = {
                'experiment_type': 'parameter_sweep',
                'parameter_effects': {},
                'response_statistics': {},
                'correlations': {}
            }
            
            # Analyze parameter effects
            for param_name in config.parameters.keys():
                if param_name in df.columns:
                    param_effects = {}
                    for response in config.responses:
                        if response in df.columns:
                            correlation = df[param_name].corr(df[response])
                            param_effects[response] = float(correlation) if not pd.isna(correlation) else 0.0
                    
                    analysis['parameter_effects'][param_name] = param_effects
            
            # Analyze response statistics
            for response in config.responses:
                if response in df.columns:
                    analysis['response_statistics'][response] = {
                        'mean': float(df[response].mean()),
                        'std': float(df[response].std()),
                        'min': float(df[response].min()),
                        'max': float(df[response].max()),
                        'median': float(df[response].median())
                    }
            
            # Calculate correlations
            response_cols = [col for col in config.responses if col in df.columns]
            if len(response_cols) > 1:
                correlation_matrix = df[response_cols].corr()
                analysis['correlations'] = correlation_matrix.to_dict()
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing parameter sweep: {e}")
            return {}
    
    async def _analyze_factorial(self, df: pd.DataFrame, config: ExperimentConfig) -> Dict[str, Any]:
        """Analyze factorial experiment results."""
        try:
            analysis = {
                'experiment_type': 'factorial',
                'main_effects': {},
                'interaction_effects': {},
                'anova_results': {}
            }
            
            # Analyze main effects
            for param_name in config.parameters.keys():
                if param_name in df.columns:
                    main_effects = {}
                    for response in config.responses:
                        if response in df.columns:
                            # Calculate main effect
                            param_levels = df[param_name].unique()
                            if len(param_levels) > 1:
                                effect = df.groupby(param_name)[response].mean().max() - df.groupby(param_name)[response].mean().min()
                                main_effects[response] = float(effect)
                            else:
                                main_effects[response] = 0.0
                    
                    analysis['main_effects'][param_name] = main_effects
            
            # Analyze interaction effects (simplified)
            param_names = list(config.parameters.keys())
            if len(param_names) >= 2:
                for i, param1 in enumerate(param_names):
                    for param2 in param_names[i+1:]:
                        if param1 in df.columns and param2 in df.columns:
                            interaction_effects = {}
                            for response in config.responses:
                                if response in df.columns:
                                    # Simplified interaction effect calculation
                                    interaction_effect = df.groupby([param1, param2])[response].mean().std()
                                    interaction_effects[response] = float(interaction_effect) if not pd.isna(interaction_effect) else 0.0
                            
                            analysis['interaction_effects'][f"{param1}_x_{param2}"] = interaction_effects
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing factorial experiment: {e}")
            return {}
    
    async def _analyze_response_surface(self, df: pd.DataFrame, config: ExperimentConfig) -> Dict[str, Any]:
        """Analyze response surface experiment results."""
        try:
            analysis = {
                'experiment_type': 'response_surface',
                'response_surface_models': {},
                'optimal_conditions': {},
                'contour_analysis': {}
            }
            
            # Analyze response surface models
            for response in config.responses:
                if response in df.columns:
                    # Simplified response surface analysis
                    param_names = list(config.parameters.keys())
                    if len(param_names) >= 2:
                        # Calculate response surface statistics
                        response_stats = {
                            'mean': float(df[response].mean()),
                            'std': float(df[response].std()),
                            'min': float(df[response].min()),
                            'max': float(df[response].max())
                        }
                        
                        analysis['response_surface_models'][response] = response_stats
                        
                        # Find optimal conditions
                        optimal_idx = df[response].idxmax()
                        optimal_conditions = {}
                        for param_name in param_names:
                            if param_name in df.columns:
                                optimal_conditions[param_name] = float(df.loc[optimal_idx, param_name])
                        
                        analysis['optimal_conditions'][response] = optimal_conditions
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing response surface: {e}")
            return {}
    
    async def _analyze_optimization(self, df: pd.DataFrame, config: ExperimentConfig) -> Dict[str, Any]:
        """Analyze optimization experiment results."""
        try:
            analysis = {
                'experiment_type': 'optimization',
                'optimal_solutions': {},
                'convergence_analysis': {},
                'multi_objective_analysis': {}
            }
            
            # Find optimal solutions
            for response in config.responses:
                if response in df.columns:
                    # Find best solution
                    best_idx = df[response].idxmax()
                    optimal_solution = {}
                    
                    for param_name in config.parameters.keys():
                        if param_name in df.columns:
                            optimal_solution[param_name] = float(df.loc[best_idx, param_name])
                    
                    optimal_solution['objective_value'] = float(df.loc[best_idx, response])
                    analysis['optimal_solutions'][response] = optimal_solution
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing optimization: {e}")
            return {}
    
    async def _analyze_sensitivity(self, df: pd.DataFrame, config: ExperimentConfig) -> Dict[str, Any]:
        """Analyze sensitivity experiment results."""
        try:
            analysis = {
                'experiment_type': 'sensitivity',
                'sensitivity_indices': {},
                'parameter_importance': {},
                'uncertainty_analysis': {}
            }
            
            # Calculate sensitivity indices
            for param_name in config.parameters.keys():
                if param_name in df.columns:
                    sensitivity_indices = {}
                    for response in config.responses:
                        if response in df.columns:
                            # Simplified sensitivity index calculation
                            param_std = df[param_name].std()
                            response_std = df[response].std()
                            
                            if param_std > 0 and response_std > 0:
                                correlation = df[param_name].corr(df[response])
                                sensitivity_index = abs(correlation) * (param_std / response_std)
                                sensitivity_indices[response] = float(sensitivity_index) if not pd.isna(sensitivity_index) else 0.0
                            else:
                                sensitivity_indices[response] = 0.0
                    
                    analysis['sensitivity_indices'][param_name] = sensitivity_indices
            
            # Calculate parameter importance
            for response in config.responses:
                if response in df.columns:
                    importance_scores = {}
                    for param_name in config.parameters.keys():
                        if param_name in df.columns and param_name in analysis['sensitivity_indices']:
                            importance_scores[param_name] = analysis['sensitivity_indices'][param_name].get(response, 0.0)
                    
                    # Sort by importance
                    sorted_importance = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
                    analysis['parameter_importance'][response] = dict(sorted_importance)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing sensitivity: {e}")
            return {}
    
    async def _analyze_general(self, df: pd.DataFrame, config: ExperimentConfig) -> Dict[str, Any]:
        """General experiment analysis."""
        try:
            analysis = {
                'experiment_type': 'general',
                'summary_statistics': {},
                'parameter_distributions': {},
                'response_distributions': {}
            }
            
            # Summary statistics
            analysis['summary_statistics'] = {
                'total_runs': len(df),
                'successful_runs': len(df[df['status'] == 'completed']),
                'failed_runs': len(df[df['status'] == 'failed']),
                'average_execution_time': float(df['execution_time'].mean()) if 'execution_time' in df.columns else 0.0
            }
            
            # Parameter distributions
            for param_name in config.parameters.keys():
                if param_name in df.columns:
                    analysis['parameter_distributions'][param_name] = {
                        'mean': float(df[param_name].mean()),
                        'std': float(df[param_name].std()),
                        'min': float(df[param_name].min()),
                        'max': float(df[param_name].max())
                    }
            
            # Response distributions
            for response in config.responses:
                if response in df.columns:
                    analysis['response_distributions'][response] = {
                        'mean': float(df[response].mean()),
                        'std': float(df[response].std()),
                        'min': float(df[response].min()),
                        'max': float(df[response].max())
                    }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in general analysis: {e}")
            return {}


class ParameterSweepDesigner:
    """
    Parameter sweep designer for PBF-LB/M virtual environment.
    
    This class provides specialized parameter sweep design capabilities including
    one-factor-at-a-time (OFAT) and multi-factor parameter sweeps.
    """
    
    def __init__(self):
        """Initialize the parameter sweep designer."""
        self.sweep_configs = {}
        self.sweep_results = {}
        
        logger.info("Parameter Sweep Designer initialized")
    
    async def create_parameter_sweep(
        self,
        name: str,
        parameters: Dict[str, Dict[str, Any]],
        responses: List[str],
        sweep_type: str = "one_factor_at_a_time"
    ) -> str:
        """
        Create parameter sweep experiment.
        
        Args:
            name: Experiment name
            parameters: Parameter definitions
            responses: Response variables
            sweep_type: Type of parameter sweep
            
        Returns:
            str: Parameter sweep ID
        """
        try:
            sweep_id = str(uuid.uuid4())
            
            config = {
                'sweep_id': sweep_id,
                'name': name,
                'parameters': parameters,
                'responses': responses,
                'sweep_type': sweep_type,
                'created_at': datetime.now()
            }
            
            self.sweep_configs[sweep_id] = config
            
            # Generate sweep design
            sweep_design = await self._generate_sweep_design(config)
            config['sweep_design'] = sweep_design
            
            logger.info(f"Parameter sweep created: {sweep_id}")
            return sweep_id
            
        except Exception as e:
            logger.error(f"Error creating parameter sweep: {e}")
            return ""
    
    async def _generate_sweep_design(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate parameter sweep design."""
        try:
            if config['sweep_type'] == "one_factor_at_a_time":
                return self._generate_ofat_design(config)
            elif config['sweep_type'] == "multi_factor":
                return self._generate_multi_factor_design(config)
            else:
                return self._generate_random_sweep_design(config)
                
        except Exception as e:
            logger.error(f"Error generating sweep design: {e}")
            return []
    
    def _generate_ofat_design(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate one-factor-at-a-time design."""
        try:
            sweep_design = []
            parameter_names = list(config['parameters'].keys())
            
            # Set baseline values
            baseline = {}
            for param_name, param_config in config['parameters'].items():
                if param_config['type'] == 'continuous':
                    baseline[param_name] = (param_config['min'] + param_config['max']) / 2
                else:
                    baseline[param_name] = param_config['levels'][0]
            
            # Sweep each parameter
            for param_name, param_config in config['parameters'].items():
                if param_config['type'] == 'continuous':
                    # Generate sweep values
                    min_val = param_config['min']
                    max_val = param_config['max']
                    n_points = param_config.get('sweep_points', 10)
                    sweep_values = np.linspace(min_val, max_val, n_points)
                    
                    for value in sweep_values:
                        parameter_set = baseline.copy()
                        parameter_set[param_name] = value
                        sweep_design.append(parameter_set)
                else:
                    # Categorical parameter
                    for level in param_config['levels']:
                        parameter_set = baseline.copy()
                        parameter_set[param_name] = level
                        sweep_design.append(parameter_set)
            
            return sweep_design
            
        except Exception as e:
            logger.error(f"Error generating OFAT design: {e}")
            return []
    
    def _generate_multi_factor_design(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate multi-factor sweep design."""
        try:
            sweep_design = []
            parameter_names = list(config['parameters'].keys())
            
            # Generate combinations of parameter values
            parameter_combinations = []
            for param_name, param_config in config['parameters'].items():
                if param_config['type'] == 'continuous':
                    min_val = param_config['min']
                    max_val = param_config['max']
                    n_points = param_config.get('sweep_points', 5)
                    values = np.linspace(min_val, max_val, n_points)
                    parameter_combinations.append(values)
                else:
                    parameter_combinations.append(param_config['levels'])
            
            # Generate all combinations
            for combination in product(*parameter_combinations):
                parameter_set = dict(zip(parameter_names, combination))
                sweep_design.append(parameter_set)
            
            return sweep_design
            
        except Exception as e:
            logger.error(f"Error generating multi-factor design: {e}")
            return []
    
    def _generate_random_sweep_design(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate random sweep design."""
        try:
            sweep_design = []
            n_samples = config.get('n_samples', 100)
            
            for _ in range(n_samples):
                parameter_set = {}
                for param_name, param_config in config['parameters'].items():
                    if param_config['type'] == 'continuous':
                        min_val = param_config['min']
                        max_val = param_config['max']
                        value = np.random.uniform(min_val, max_val)
                        parameter_set[param_name] = value
                    else:
                        levels = param_config['levels']
                        parameter_set[param_name] = np.random.choice(levels)
                
                sweep_design.append(parameter_set)
            
            return sweep_design
            
        except Exception as e:
            logger.error(f"Error generating random sweep design: {e}")
            return []


class DoEDesigner:
    """
    Design of Experiments (DoE) designer for PBF-LB/M virtual environment.
    
    This class provides specialized DoE design capabilities including factorial
    designs, response surface designs, and optimal designs.
    """
    
    def __init__(self):
        """Initialize the DoE designer."""
        self.doe_configs = {}
        self.doe_results = {}
        
        logger.info("DoE Designer initialized")
    
    async def create_doe_experiment(
        self,
        name: str,
        design_type: DesignType,
        parameters: Dict[str, Dict[str, Any]],
        responses: List[str],
        sample_size: int = 100
    ) -> str:
        """
        Create DoE experiment.
        
        Args:
            name: Experiment name
            design_type: Type of DoE design
            parameters: Parameter definitions
            responses: Response variables
            sample_size: Sample size
            
        Returns:
            str: DoE experiment ID
        """
        try:
            doe_id = str(uuid.uuid4())
            
            config = {
                'doe_id': doe_id,
                'name': name,
                'design_type': design_type,
                'parameters': parameters,
                'responses': responses,
                'sample_size': sample_size,
                'created_at': datetime.now()
            }
            
            self.doe_configs[doe_id] = config
            
            # Generate DoE design
            doe_design = await self._generate_doe_design(config)
            config['doe_design'] = doe_design
            
            logger.info(f"DoE experiment created: {doe_id}")
            return doe_id
            
        except Exception as e:
            logger.error(f"Error creating DoE experiment: {e}")
            return ""
    
    async def _generate_doe_design(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate DoE design."""
        try:
            design_type = config['design_type']
            
            if design_type == DesignType.FULL_FACTORIAL:
                return self._generate_full_factorial_doe(config)
            elif design_type == DesignType.FRACTIONAL_FACTORIAL:
                return self._generate_fractional_factorial_doe(config)
            elif design_type == DesignType.CENTRAL_COMPOSITE:
                return self._generate_central_composite_doe(config)
            elif design_type == DesignType.BOX_BEHNKEN:
                return self._generate_box_behnken_doe(config)
            elif design_type == DesignType.LATIN_HYPERCUBE:
                return self._generate_latin_hypercube_doe(config)
            else:
                return self._generate_random_doe_design(config)
                
        except Exception as e:
            logger.error(f"Error generating DoE design: {e}")
            return []
    
    def _generate_full_factorial_doe(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate full factorial DoE design."""
        try:
            # Use the same method as in VirtualExperimentDesigner
            parameter_levels = {}
            for param_name, param_config in config['parameters'].items():
                if param_config['type'] == 'categorical':
                    parameter_levels[param_name] = param_config['levels']
                else:
                    min_val = param_config['min']
                    max_val = param_config['max']
                    n_levels = param_config.get('levels', 3)
                    parameter_levels[param_name] = np.linspace(min_val, max_val, n_levels)
            
            doe_design = []
            for combination in product(*parameter_levels.values()):
                parameter_set = dict(zip(parameter_levels.keys(), combination))
                doe_design.append(parameter_set)
            
            return doe_design
            
        except Exception as e:
            logger.error(f"Error generating full factorial DoE: {e}")
            return []
    
    def _generate_fractional_factorial_doe(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate fractional factorial DoE design."""
        try:
            # Generate full factorial first
            full_design = self._generate_full_factorial_doe(config)
            
            # Sample a fraction
            fraction_size = min(len(full_design), config['sample_size'])
            np.random.seed(42)
            indices = np.random.choice(len(full_design), fraction_size, replace=False)
            
            return [full_design[i] for i in indices]
            
        except Exception as e:
            logger.error(f"Error generating fractional factorial DoE: {e}")
            return []
    
    def _generate_central_composite_doe(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate central composite DoE design."""
        try:
            # Use the same method as in VirtualExperimentDesigner
            doe_design = []
            parameter_names = list(config['parameters'].keys())
            n_params = len(parameter_names)
            
            # Generate factorial points
            for i in range(2**n_params):
                parameter_set = {}
                for j, param_name in enumerate(parameter_names):
                    param_config = config['parameters'][param_name]
                    if param_config['type'] == 'continuous':
                        min_val = param_config['min']
                        max_val = param_config['max']
                        value = max_val if (i >> j) & 1 else min_val
                        parameter_set[param_name] = value
                    else:
                        levels = param_config['levels']
                        parameter_set[param_name] = levels[i % len(levels)]
                
                doe_design.append(parameter_set)
            
            # Add center points
            center_point = {}
            for param_name, param_config in config['parameters'].items():
                if param_config['type'] == 'continuous':
                    min_val = param_config['min']
                    max_val = param_config['max']
                    center_point[param_name] = (min_val + max_val) / 2
                else:
                    center_point[param_name] = param_config['levels'][0]
            
            # Add multiple center points
            for _ in range(max(3, config['sample_size'] - len(doe_design))):
                doe_design.append(center_point.copy())
            
            return doe_design[:config['sample_size']]
            
        except Exception as e:
            logger.error(f"Error generating central composite DoE: {e}")
            return []
    
    def _generate_box_behnken_doe(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate Box-Behnken DoE design."""
        try:
            # Use the same method as in VirtualExperimentDesigner
            doe_design = []
            parameter_names = list(config['parameters'].keys())
            
            for i in range(config['sample_size']):
                parameter_set = {}
                for param_name, param_config in config['parameters'].items():
                    if param_config['type'] == 'continuous':
                        min_val = param_config['min']
                        max_val = param_config['max']
                        center = (min_val + max_val) / 2
                        range_val = (max_val - min_val) / 2
                        value = center + np.random.normal(0, range_val * 0.3)
                        parameter_set[param_name] = max(min_val, min(max_val, value))
                    else:
                        levels = param_config['levels']
                        parameter_set[param_name] = np.random.choice(levels)
                
                doe_design.append(parameter_set)
            
            return doe_design
            
        except Exception as e:
            logger.error(f"Error generating Box-Behnken DoE: {e}")
            return []
    
    def _generate_latin_hypercube_doe(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate Latin Hypercube DoE design."""
        try:
            # Use the same method as in VirtualExperimentDesigner
            doe_design = []
            parameter_names = list(config['parameters'].keys())
            n_params = len(parameter_names)
            
            np.random.seed(42)
            lhs_samples = np.random.uniform(0, 1, (config['sample_size'], n_params))
            
            for i in range(config['sample_size']):
                parameter_set = {}
                for j, param_name in enumerate(parameter_names):
                    param_config = config['parameters'][param_name]
                    if param_config['type'] == 'continuous':
                        min_val = param_config['min']
                        max_val = param_config['max']
                        value = min_val + lhs_samples[i, j] * (max_val - min_val)
                        parameter_set[param_name] = value
                    else:
                        levels = param_config['levels']
                        level_index = int(lhs_samples[i, j] * len(levels))
                        level_index = min(level_index, len(levels) - 1)
                        parameter_set[param_name] = levels[level_index]
                
                doe_design.append(parameter_set)
            
            return doe_design
            
        except Exception as e:
            logger.error(f"Error generating Latin Hypercube DoE: {e}")
            return []
    
    def _generate_random_doe_design(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate random DoE design."""
        try:
            # Use the same method as in VirtualExperimentDesigner
            doe_design = []
            np.random.seed(42)
            
            for i in range(config['sample_size']):
                parameter_set = {}
                for param_name, param_config in config['parameters'].items():
                    if param_config['type'] == 'continuous':
                        min_val = param_config['min']
                        max_val = param_config['max']
                        value = np.random.uniform(min_val, max_val)
                        parameter_set[param_name] = value
                    else:
                        levels = param_config['levels']
                        parameter_set[param_name] = np.random.choice(levels)
                
                doe_design.append(parameter_set)
            
            return doe_design
            
        except Exception as e:
            logger.error(f"Error generating random DoE design: {e}")
            return []
