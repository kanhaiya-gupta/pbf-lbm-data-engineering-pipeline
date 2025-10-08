"""
Mechanical Simulation for PBF-LB/M Virtual Environment

This module provides mechanical simulation capabilities including stress analysis,
deformation analysis, and mechanical behavior prediction for PBF-LB/M virtual
testing and simulation environments.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import logging
from datetime import datetime
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import warnings

logger = logging.getLogger(__name__)


@dataclass
class MechanicalConfig:
    """Configuration for mechanical simulation."""
    
    # Simulation parameters
    time_step: float = 0.001  # Time step in seconds
    max_time: float = 10.0    # Maximum simulation time
    spatial_resolution: float = 0.1  # Spatial resolution in mm
    
    # Material properties
    young_modulus: float = 200e9  # Pa (steel)
    poisson_ratio: float = 0.3
    density: float = 7850.0  # kg/m³ (steel)
    yield_strength: float = 250e6  # Pa
    ultimate_strength: float = 400e6  # Pa
    
    # Loading conditions
    applied_force: float = 1000.0  # N
    applied_pressure: float = 1e6  # Pa
    temperature: float = 25.0  # °C
    
    # Numerical parameters
    convergence_tolerance: float = 1e-6
    max_iterations: int = 1000
    damping_factor: float = 0.1


@dataclass
class MechanicalResult:
    """Result of mechanical simulation."""
    
    success: bool
    displacement_field: np.ndarray
    stress_field: np.ndarray
    strain_field: np.ndarray
    time_steps: np.ndarray
    max_displacement: float
    max_stress: float
    max_strain: float
    simulation_time: float
    convergence_info: Dict[str, Any]
    error_message: Optional[str] = None


class MechanicalSimulator:
    """
    Mechanical simulator for PBF-LB/M processes.
    
    This class provides comprehensive mechanical simulation capabilities including
    stress analysis, deformation analysis, and mechanical behavior prediction for
    PBF-LB/M virtual testing and simulation environments.
    """
    
    def __init__(self, config: MechanicalConfig = None):
        """Initialize the mechanical simulator."""
        self.config = config or MechanicalConfig()
        self.stress_solver = StressSolver(self.config)
        self.deformation_analyzer = DeformationAnalyzer()
        
        logger.info("Mechanical Simulator initialized")
    
    def simulate_stress_analysis(
        self,
        geometry: np.ndarray,
        boundary_conditions: Dict[str, Any] = None,
        loading_conditions: Dict[str, Any] = None
    ) -> MechanicalResult:
        """
        Simulate stress analysis.
        
        Args:
            geometry: 3D geometry array (1 = material, 0 = void)
            boundary_conditions: Boundary condition specifications
            loading_conditions: Loading condition specifications
            
        Returns:
            MechanicalResult: Mechanical simulation results
        """
        try:
            start_time = datetime.now()
            
            if boundary_conditions is None:
                boundary_conditions = {
                    'fixed_surfaces': ['bottom'],  # Fixed bottom surface
                    'free_surfaces': ['top', 'sides']
                }
            
            if loading_conditions is None:
                loading_conditions = {
                    'applied_force': self.config.applied_force,
                    'applied_pressure': self.config.applied_pressure,
                    'direction': [0, 0, -1]  # Downward force
                }
            
            # Set up time stepping
            time_steps = np.arange(0, self.config.max_time, self.config.time_step)
            
            # Run mechanical simulation
            displacement_field, stress_field, strain_field, convergence_info = self.stress_solver.solve_mechanical_equations(
                geometry, boundary_conditions, loading_conditions, time_steps
            )
            
            # Analyze results
            max_displacement = np.max(np.linalg.norm(displacement_field, axis=0))
            max_stress = np.max(np.linalg.norm(stress_field, axis=0))
            max_strain = np.max(np.linalg.norm(strain_field, axis=0))
            
            # Calculate simulation time
            simulation_time = (datetime.now() - start_time).total_seconds()
            
            # Create result
            result = MechanicalResult(
                success=True,
                displacement_field=displacement_field,
                stress_field=stress_field,
                strain_field=strain_field,
                time_steps=time_steps,
                max_displacement=max_displacement,
                max_stress=max_stress,
                max_strain=max_strain,
                simulation_time=simulation_time,
                convergence_info=convergence_info
            )
            
            logger.info(f"Mechanical simulation completed: {simulation_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Error in mechanical simulation: {e}")
            return MechanicalResult(
                success=False,
                displacement_field=np.array([]),
                stress_field=np.array([]),
                strain_field=np.array([]),
                time_steps=np.array([]),
                max_displacement=0.0,
                max_stress=0.0,
                max_strain=0.0,
                simulation_time=0.0,
                convergence_info={},
                error_message=str(e)
            )
    
    def simulate_thermal_stress(
        self,
        geometry: np.ndarray,
        temperature_field: np.ndarray,
        reference_temperature: float = 25.0
    ) -> MechanicalResult:
        """
        Simulate thermal stress analysis.
        
        Args:
            geometry: 3D geometry array
            temperature_field: Temperature field
            reference_temperature: Reference temperature for thermal expansion
            
        Returns:
            MechanicalResult: Thermal stress simulation results
        """
        try:
            # Calculate thermal expansion coefficient
            thermal_expansion = 12e-6  # 1/K (steel)
            
            # Calculate thermal strain
            thermal_strain = thermal_expansion * (temperature_field - reference_temperature)
            
            # Set up thermal loading conditions
            loading_conditions = {
                'thermal_strain': thermal_strain,
                'thermal_expansion_coefficient': thermal_expansion,
                'reference_temperature': reference_temperature
            }
            
            # Set up boundary conditions (fixed at reference points)
            boundary_conditions = {
                'fixed_surfaces': ['bottom'],
                'free_surfaces': ['top', 'sides'],
                'thermal_constraint': True
            }
            
            # Run thermal stress simulation
            result = self.simulate_stress_analysis(geometry, boundary_conditions, loading_conditions)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in thermal stress simulation: {e}")
            return MechanicalResult(
                success=False,
                displacement_field=np.array([]),
                stress_field=np.array([]),
                strain_field=np.array([]),
                time_steps=np.array([]),
                max_displacement=0.0,
                max_stress=0.0,
                max_strain=0.0,
                simulation_time=0.0,
                convergence_info={},
                error_message=str(e)
            )
    
    def simulate_fatigue_analysis(
        self,
        geometry: np.ndarray,
        stress_history: np.ndarray,
        cycles: int = 1000
    ) -> Dict[str, Any]:
        """
        Simulate fatigue analysis.
        
        Args:
            geometry: 3D geometry array
            stress_history: Stress history over time
            cycles: Number of fatigue cycles
            
        Returns:
            Dict: Fatigue analysis results
        """
        try:
            # Calculate stress range
            stress_range = np.max(stress_history) - np.min(stress_history)
            
            # Calculate mean stress
            mean_stress = np.mean(stress_history)
            
            # Calculate stress amplitude
            stress_amplitude = stress_range / 2
            
            # Calculate fatigue life using S-N curve (simplified)
            fatigue_life = self._calculate_fatigue_life(stress_amplitude, mean_stress)
            
            # Calculate damage accumulation
            damage = cycles / fatigue_life if fatigue_life > 0 else 0
            
            # Determine failure
            failure = damage >= 1.0
            
            fatigue_results = {
                'stress_range': float(stress_range),
                'mean_stress': float(mean_stress),
                'stress_amplitude': float(stress_amplitude),
                'fatigue_life': float(fatigue_life),
                'damage': float(damage),
                'failure': failure,
                'cycles_to_failure': int(fatigue_life) if fatigue_life > 0 else 0
            }
            
            logger.info(f"Fatigue analysis completed: {damage:.3f} damage")
            return fatigue_results
            
        except Exception as e:
            logger.error(f"Error in fatigue analysis: {e}")
            return {}
    
    def _calculate_fatigue_life(self, stress_amplitude: float, mean_stress: float) -> float:
        """Calculate fatigue life using S-N curve."""
        try:
            # Simplified S-N curve parameters (steel)
            fatigue_strength = 200e6  # Pa
            fatigue_exponent = -0.1
            
            # Calculate fatigue life
            if stress_amplitude > 0:
                fatigue_life = (fatigue_strength / stress_amplitude) ** (1 / fatigue_exponent)
                return max(1, fatigue_life)  # Minimum 1 cycle
            
            return float('inf')
            
        except Exception as e:
            logger.error(f"Error calculating fatigue life: {e}")
            return 0.0


class StressSolver:
    """
    Stress solver for mechanical equations.
    
    This class provides numerical methods for solving mechanical equations
    including stress, strain, and displacement calculations.
    """
    
    def __init__(self, config: MechanicalConfig):
        """Initialize the stress solver."""
        self.config = config
        
        logger.info("Stress Solver initialized")
    
    def solve_mechanical_equations(
        self,
        geometry: np.ndarray,
        boundary_conditions: Dict[str, Any],
        loading_conditions: Dict[str, Any],
        time_steps: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Solve mechanical equations numerically.
        
        Args:
            geometry: 3D geometry array
            boundary_conditions: Boundary condition specifications
            loading_conditions: Loading condition specifications
            time_steps: Time steps for simulation
            
        Returns:
            Tuple: (displacement_field, stress_field, strain_field, convergence_info)
        """
        try:
            # Initialize fields
            displacement_field = np.zeros((3, *geometry.shape))
            stress_field = np.zeros((6, *geometry.shape))  # 6 stress components
            strain_field = np.zeros((6, *geometry.shape))  # 6 strain components
            
            convergence_info = {
                'iterations': 0,
                'converged': True,
                'max_residual': 0.0
            }
            
            # Time stepping
            for t_idx, t in enumerate(time_steps):
                # Solve equilibrium equations
                displacement_field = self._solve_equilibrium_equations(
                    displacement_field, geometry, boundary_conditions, loading_conditions
                )
                
                # Calculate strain from displacement
                strain_field = self._calculate_strain(displacement_field, geometry)
                
                # Calculate stress from strain
                stress_field = self._calculate_stress(strain_field, geometry)
                
                # Check convergence
                if t_idx % 100 == 0:  # Check every 100 time steps
                    residual = self._calculate_residual(displacement_field, stress_field)
                    convergence_info['max_residual'] = max(convergence_info['max_residual'], residual)
                    
                    if residual < self.config.convergence_tolerance:
                        convergence_info['converged'] = True
                        break
                
                convergence_info['iterations'] += 1
            
            return displacement_field, stress_field, strain_field, convergence_info
            
        except Exception as e:
            logger.error(f"Error solving mechanical equations: {e}")
            return np.zeros((3, *geometry.shape)), np.zeros((6, *geometry.shape)), np.zeros((6, *geometry.shape)), {'converged': False, 'error': str(e)}
    
    def _solve_equilibrium_equations(
        self,
        displacement_field: np.ndarray,
        geometry: np.ndarray,
        boundary_conditions: Dict[str, Any],
        loading_conditions: Dict[str, Any]
    ) -> np.ndarray:
        """Solve equilibrium equations."""
        try:
            new_displacement = displacement_field.copy()
            
            # Apply finite element method (simplified)
            for i in range(1, geometry.shape[0] - 1):
                for j in range(1, geometry.shape[1] - 1):
                    for k in range(1, geometry.shape[2] - 1):
                        if geometry[i, j, k] == 1:  # Material region
                            # Calculate stiffness matrix (simplified)
                            stiffness = self._calculate_stiffness_matrix(i, j, k, geometry)
                            
                            # Calculate force vector
                            force_vector = self._calculate_force_vector(i, j, k, loading_conditions)
                            
                            # Solve for displacement
                            displacement = np.linalg.solve(stiffness, force_vector)
                            
                            # Update displacement field
                            new_displacement[:, i, j, k] = displacement
            
            # Apply boundary conditions
            new_displacement = self._apply_boundary_conditions(
                new_displacement, boundary_conditions, geometry
            )
            
            return new_displacement
            
        except Exception as e:
            logger.error(f"Error solving equilibrium equations: {e}")
            return displacement_field
    
    def _calculate_stiffness_matrix(
        self,
        i: int,
        j: int,
        k: int,
        geometry: np.ndarray
    ) -> np.ndarray:
        """Calculate stiffness matrix for element."""
        try:
            # Simplified stiffness matrix (3x3 for 3D)
            E = self.config.young_modulus
            nu = self.config.poisson_ratio
            
            # Calculate element volume
            element_volume = self.config.spatial_resolution ** 3
            
            # Simplified stiffness matrix
            stiffness = np.array([
                [E * (1 - nu) / ((1 + nu) * (1 - 2 * nu)), 0, 0],
                [0, E * (1 - nu) / ((1 + nu) * (1 - 2 * nu)), 0],
                [0, 0, E * (1 - nu) / ((1 + nu) * (1 - 2 * nu))]
            ]) * element_volume
            
            return stiffness
            
        except Exception as e:
            logger.error(f"Error calculating stiffness matrix: {e}")
            return np.eye(3) * self.config.young_modulus
    
    def _calculate_force_vector(
        self,
        i: int,
        j: int,
        k: int,
        loading_conditions: Dict[str, Any]
    ) -> np.ndarray:
        """Calculate force vector for element."""
        try:
            force_vector = np.zeros(3)
            
            # Apply external forces
            if 'applied_force' in loading_conditions:
                force_magnitude = loading_conditions['applied_force']
                direction = loading_conditions.get('direction', [0, 0, -1])
                force_vector = np.array(direction) * force_magnitude
            
            # Apply pressure loads
            if 'applied_pressure' in loading_conditions:
                pressure = loading_conditions['applied_pressure']
                area = self.config.spatial_resolution ** 2
                force_vector[2] += pressure * area  # z-direction
            
            # Apply thermal loads
            if 'thermal_strain' in loading_conditions:
                thermal_strain = loading_conditions['thermal_strain']
                thermal_expansion = loading_conditions.get('thermal_expansion_coefficient', 12e-6)
                E = self.config.young_modulus
                
                thermal_stress = E * thermal_expansion * thermal_strain
                force_vector += thermal_stress * self.config.spatial_resolution ** 2
            
            return force_vector
            
        except Exception as e:
            logger.error(f"Error calculating force vector: {e}")
            return np.zeros(3)
    
    def _calculate_strain(
        self,
        displacement_field: np.ndarray,
        geometry: np.ndarray
    ) -> np.ndarray:
        """Calculate strain from displacement."""
        try:
            strain_field = np.zeros((6, *geometry.shape))
            
            # Calculate strain components using finite differences
            for i in range(1, geometry.shape[0] - 1):
                for j in range(1, geometry.shape[1] - 1):
                    for k in range(1, geometry.shape[2] - 1):
                        if geometry[i, j, k] == 1:  # Material region
                            dx = self.config.spatial_resolution
                            
                            # Calculate strain components
                            # Normal strains
                            strain_field[0, i, j, k] = (displacement_field[0, i+1, j, k] - displacement_field[0, i-1, j, k]) / (2 * dx)  # εxx
                            strain_field[1, i, j, k] = (displacement_field[1, i, j+1, k] - displacement_field[1, i, j-1, k]) / (2 * dx)  # εyy
                            strain_field[2, i, j, k] = (displacement_field[2, i, j, k+1] - displacement_field[2, i, j, k-1]) / (2 * dx)  # εzz
                            
                            # Shear strains
                            strain_field[3, i, j, k] = 0.5 * (
                                (displacement_field[0, i, j+1, k] - displacement_field[0, i, j-1, k]) / (2 * dx) +
                                (displacement_field[1, i+1, j, k] - displacement_field[1, i-1, j, k]) / (2 * dx)
                            )  # γxy
                            
                            strain_field[4, i, j, k] = 0.5 * (
                                (displacement_field[0, i, j, k+1] - displacement_field[0, i, j, k-1]) / (2 * dx) +
                                (displacement_field[2, i+1, j, k] - displacement_field[2, i-1, j, k]) / (2 * dx)
                            )  # γxz
                            
                            strain_field[5, i, j, k] = 0.5 * (
                                (displacement_field[1, i, j, k+1] - displacement_field[1, i, j, k-1]) / (2 * dx) +
                                (displacement_field[2, i, j+1, k] - displacement_field[2, i, j-1, k]) / (2 * dx)
                            )  # γyz
            
            return strain_field
            
        except Exception as e:
            logger.error(f"Error calculating strain: {e}")
            return np.zeros((6, *geometry.shape))
    
    def _calculate_stress(
        self,
        strain_field: np.ndarray,
        geometry: np.ndarray
    ) -> np.ndarray:
        """Calculate stress from strain using Hooke's law."""
        try:
            stress_field = np.zeros((6, *geometry.shape))
            
            E = self.config.young_modulus
            nu = self.config.poisson_ratio
            
            # Calculate stress components using Hooke's law
            for i in range(geometry.shape[0]):
                for j in range(geometry.shape[1]):
                    for k in range(geometry.shape[2]):
                        if geometry[i, j, k] == 1:  # Material region
                            # Normal stresses
                            stress_field[0, i, j, k] = E / ((1 + nu) * (1 - 2 * nu)) * (
                                (1 - nu) * strain_field[0, i, j, k] +
                                nu * (strain_field[1, i, j, k] + strain_field[2, i, j, k])
                            )  # σxx
                            
                            stress_field[1, i, j, k] = E / ((1 + nu) * (1 - 2 * nu)) * (
                                (1 - nu) * strain_field[1, i, j, k] +
                                nu * (strain_field[0, i, j, k] + strain_field[2, i, j, k])
                            )  # σyy
                            
                            stress_field[2, i, j, k] = E / ((1 + nu) * (1 - 2 * nu)) * (
                                (1 - nu) * strain_field[2, i, j, k] +
                                nu * (strain_field[0, i, j, k] + strain_field[1, i, j, k])
                            )  # σzz
                            
                            # Shear stresses
                            stress_field[3, i, j, k] = E / (2 * (1 + nu)) * strain_field[3, i, j, k]  # τxy
                            stress_field[4, i, j, k] = E / (2 * (1 + nu)) * strain_field[4, i, j, k]  # τxz
                            stress_field[5, i, j, k] = E / (2 * (1 + nu)) * strain_field[5, i, j, k]  # τyz
            
            return stress_field
            
        except Exception as e:
            logger.error(f"Error calculating stress: {e}")
            return np.zeros((6, *geometry.shape))
    
    def _apply_boundary_conditions(
        self,
        displacement_field: np.ndarray,
        boundary_conditions: Dict[str, Any],
        geometry: np.ndarray
    ) -> np.ndarray:
        """Apply boundary conditions."""
        try:
            new_displacement = displacement_field.copy()
            
            # Apply fixed boundary conditions
            if 'fixed_surfaces' in boundary_conditions:
                for surface in boundary_conditions['fixed_surfaces']:
                    if surface == 'bottom':
                        # Fix bottom surface
                        new_displacement[:, :, :, 0] = 0.0
                    elif surface == 'top':
                        # Fix top surface
                        new_displacement[:, :, :, -1] = 0.0
                    elif surface == 'sides':
                        # Fix side surfaces
                        new_displacement[:, 0, :, :] = 0.0
                        new_displacement[:, -1, :, :] = 0.0
                        new_displacement[:, :, 0, :] = 0.0
                        new_displacement[:, :, -1, :] = 0.0
            
            return new_displacement
            
        except Exception as e:
            logger.error(f"Error applying boundary conditions: {e}")
            return displacement_field
    
    def _calculate_residual(
        self,
        displacement_field: np.ndarray,
        stress_field: np.ndarray
    ) -> float:
        """Calculate residual for convergence checking."""
        try:
            # Calculate displacement residual
            displacement_residual = np.max(np.abs(displacement_field))
            
            # Calculate stress residual
            stress_residual = np.max(np.abs(stress_field))
            
            return max(displacement_residual, stress_residual)
            
        except Exception as e:
            logger.error(f"Error calculating residual: {e}")
            return 0.0


class DeformationAnalyzer:
    """
    Deformation analyzer for PBF-LB/M processes.
    
    This class provides deformation analysis capabilities including
    displacement analysis, strain analysis, and deformation visualization.
    """
    
    def __init__(self):
        """Initialize the deformation analyzer."""
        logger.info("Deformation Analyzer initialized")
    
    def analyze_deformation(
        self,
        displacement_field: np.ndarray,
        stress_field: np.ndarray,
        strain_field: np.ndarray
    ) -> Dict[str, Any]:
        """
        Analyze deformation characteristics.
        
        Args:
            displacement_field: Displacement field
            stress_field: Stress field
            strain_field: Strain field
            
        Returns:
            Dict: Deformation analysis results
        """
        try:
            analysis_results = {
                'displacement_analysis': {},
                'stress_analysis': {},
                'strain_analysis': {},
                'safety_factors': {}
            }
            
            # Analyze displacement
            displacement_magnitude = np.linalg.norm(displacement_field, axis=0)
            analysis_results['displacement_analysis'] = {
                'max_displacement': float(np.max(displacement_magnitude)),
                'mean_displacement': float(np.mean(displacement_magnitude)),
                'displacement_std': float(np.std(displacement_magnitude)),
                'displacement_range': float(np.max(displacement_magnitude) - np.min(displacement_magnitude))
            }
            
            # Analyze stress
            stress_magnitude = np.linalg.norm(stress_field, axis=0)
            analysis_results['stress_analysis'] = {
                'max_stress': float(np.max(stress_magnitude)),
                'mean_stress': float(np.mean(stress_magnitude)),
                'stress_std': float(np.std(stress_magnitude)),
                'stress_concentration': float(np.max(stress_magnitude) / np.mean(stress_magnitude))
            }
            
            # Analyze strain
            strain_magnitude = np.linalg.norm(strain_field, axis=0)
            analysis_results['strain_analysis'] = {
                'max_strain': float(np.max(strain_magnitude)),
                'mean_strain': float(np.mean(strain_magnitude)),
                'strain_std': float(np.std(strain_magnitude)),
                'strain_range': float(np.max(strain_magnitude) - np.min(strain_magnitude))
            }
            
            # Calculate safety factors
            analysis_results['safety_factors'] = self._calculate_safety_factors(stress_field)
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Error analyzing deformation: {e}")
            return {}
    
    def _calculate_safety_factors(self, stress_field: np.ndarray) -> Dict[str, float]:
        """Calculate safety factors."""
        try:
            # Calculate von Mises stress
            von_mises_stress = self._calculate_von_mises_stress(stress_field)
            
            # Calculate safety factors
            yield_safety_factor = self.config.yield_strength / np.max(von_mises_stress) if np.max(von_mises_stress) > 0 else float('inf')
            ultimate_safety_factor = self.config.ultimate_strength / np.max(von_mises_stress) if np.max(von_mises_stress) > 0 else float('inf')
            
            return {
                'yield_safety_factor': float(yield_safety_factor),
                'ultimate_safety_factor': float(ultimate_safety_factor),
                'max_von_mises_stress': float(np.max(von_mises_stress)),
                'mean_von_mises_stress': float(np.mean(von_mises_stress))
            }
            
        except Exception as e:
            logger.error(f"Error calculating safety factors: {e}")
            return {}
    
    def _calculate_von_mises_stress(self, stress_field: np.ndarray) -> np.ndarray:
        """Calculate von Mises stress."""
        try:
            # Extract stress components
            sigma_xx = stress_field[0]
            sigma_yy = stress_field[1]
            sigma_zz = stress_field[2]
            tau_xy = stress_field[3]
            tau_xz = stress_field[4]
            tau_yz = stress_field[5]
            
            # Calculate von Mises stress
            von_mises = np.sqrt(
                0.5 * (
                    (sigma_xx - sigma_yy)**2 +
                    (sigma_yy - sigma_zz)**2 +
                    (sigma_zz - sigma_xx)**2 +
                    6 * (tau_xy**2 + tau_xz**2 + tau_yz**2)
                )
            )
            
            return von_mises
            
        except Exception as e:
            logger.error(f"Error calculating von Mises stress: {e}")
            return np.array([])
