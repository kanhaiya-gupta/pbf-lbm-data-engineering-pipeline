"""
Thermal Simulation for PBF-LB/M Virtual Environment

This module provides thermal simulation capabilities including heat transfer,
thermal analysis, and temperature prediction for PBF-LB/M virtual testing
and simulation environments.
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
class ThermalConfig:
    """Configuration for thermal simulation."""
    
    # Simulation parameters
    time_step: float = 0.001  # Time step in seconds
    max_time: float = 10.0    # Maximum simulation time
    spatial_resolution: float = 0.1  # Spatial resolution in mm
    
    # Material properties
    thermal_conductivity: float = 50.0  # W/(m·K)
    density: float = 8000.0  # kg/m³
    specific_heat: float = 500.0  # J/(kg·K)
    
    # Boundary conditions
    ambient_temperature: float = 25.0  # °C
    laser_power: float = 200.0  # W
    laser_radius: float = 0.1  # mm
    
    # Numerical parameters
    convergence_tolerance: float = 1e-6
    max_iterations: int = 1000


@dataclass
class ThermalResult:
    """Result of thermal simulation."""
    
    success: bool
    temperature_field: np.ndarray
    time_steps: np.ndarray
    max_temperature: float
    min_temperature: float
    simulation_time: float
    convergence_info: Dict[str, Any]
    error_message: Optional[str] = None


class ThermalSimulator:
    """
    Thermal simulator for PBF-LB/M processes.
    
    This class provides comprehensive thermal simulation capabilities including
    heat transfer analysis, temperature prediction, and thermal field computation
    for PBF-LB/M virtual testing and simulation environments.
    """
    
    def __init__(self, config: ThermalConfig = None):
        """Initialize the thermal simulator."""
        self.config = config or ThermalConfig()
        self.solver = ThermalSolver(self.config)
        self.analyzer = ThermalAnalyzer()
        
        logger.info("Thermal Simulator initialized")
    
    def simulate_heat_transfer(
        self,
        geometry: np.ndarray,
        initial_temperature: float = None,
        heat_sources: List[Dict[str, Any]] = None
    ) -> ThermalResult:
        """
        Simulate heat transfer in PBF-LB/M process.
        
        Args:
            geometry: 3D geometry array (1 = material, 0 = air)
            initial_temperature: Initial temperature field
            heat_sources: List of heat source configurations
            
        Returns:
            ThermalResult: Thermal simulation results
        """
        try:
            start_time = datetime.now()
            
            if initial_temperature is None:
                initial_temperature = self.config.ambient_temperature
            
            if heat_sources is None:
                heat_sources = [{
                    'position': (0, 0, 0),
                    'power': self.config.laser_power,
                    'radius': self.config.laser_radius,
                    'duration': self.config.max_time
                }]
            
            # Initialize temperature field
            temperature_field = np.full_like(geometry, initial_temperature, dtype=float)
            
            # Set up time stepping
            time_steps = np.arange(0, self.config.max_time, self.config.time_step)
            
            # Run thermal simulation
            temperature_field, convergence_info = self.solver.solve_heat_equation(
                geometry, temperature_field, heat_sources, time_steps
            )
            
            # Analyze results
            max_temp = np.max(temperature_field)
            min_temp = np.min(temperature_field)
            
            # Calculate simulation time
            simulation_time = (datetime.now() - start_time).total_seconds()
            
            # Create result
            result = ThermalResult(
                success=True,
                temperature_field=temperature_field,
                time_steps=time_steps,
                max_temperature=max_temp,
                min_temperature=min_temp,
                simulation_time=simulation_time,
                convergence_info=convergence_info
            )
            
            logger.info(f"Thermal simulation completed: {simulation_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Error in thermal simulation: {e}")
            return ThermalResult(
                success=False,
                temperature_field=np.array([]),
                time_steps=np.array([]),
                max_temperature=0.0,
                min_temperature=0.0,
                simulation_time=0.0,
                convergence_info={},
                error_message=str(e)
            )
    
    def simulate_laser_heating(
        self,
        geometry: np.ndarray,
        laser_path: List[Tuple[float, float, float]],
        laser_parameters: Dict[str, float] = None
    ) -> ThermalResult:
        """
        Simulate laser heating along a path.
        
        Args:
            geometry: 3D geometry array
            laser_path: List of (x, y, z) coordinates for laser path
            laser_parameters: Laser parameters (power, speed, etc.)
            
        Returns:
            ThermalResult: Thermal simulation results
        """
        try:
            if laser_parameters is None:
                laser_parameters = {
                    'power': self.config.laser_power,
                    'speed': 1000.0,  # mm/s
                    'radius': self.config.laser_radius
                }
            
            # Create heat sources along laser path
            heat_sources = []
            for i, (x, y, z) in enumerate(laser_path):
                heat_sources.append({
                    'position': (x, y, z),
                    'power': laser_parameters['power'],
                    'radius': laser_parameters['radius'],
                    'duration': self.config.time_step,
                    'start_time': i * self.config.time_step
                })
            
            # Run thermal simulation
            result = self.simulate_heat_transfer(geometry, heat_sources=heat_sources)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in laser heating simulation: {e}")
            return ThermalResult(
                success=False,
                temperature_field=np.array([]),
                time_steps=np.array([]),
                max_temperature=0.0,
                min_temperature=0.0,
                simulation_time=0.0,
                convergence_info={},
                error_message=str(e)
            )
    
    def analyze_thermal_history(
        self,
        temperature_field: np.ndarray,
        time_steps: np.ndarray,
        analysis_points: List[Tuple[int, int, int]] = None
    ) -> Dict[str, Any]:
        """
        Analyze thermal history at specific points.
        
        Args:
            temperature_field: Temperature field over time
            time_steps: Time steps
            analysis_points: List of (x, y, z) points to analyze
            
        Returns:
            Dict: Thermal analysis results
        """
        try:
            if analysis_points is None:
                # Default analysis points
                analysis_points = [
                    (0, 0, 0),  # Center
                    (10, 0, 0),  # Edge
                    (0, 10, 0),  # Corner
                ]
            
            # Analyze thermal history
            analysis_results = self.analyzer.analyze_thermal_history(
                temperature_field, time_steps, analysis_points
            )
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Error in thermal history analysis: {e}")
            return {}


class ThermalSolver:
    """
    Thermal solver for heat equation.
    
    This class provides numerical methods for solving the heat equation
    in PBF-LB/M thermal simulations.
    """
    
    def __init__(self, config: ThermalConfig):
        """Initialize the thermal solver."""
        self.config = config
        
        logger.info("Thermal Solver initialized")
    
    def solve_heat_equation(
        self,
        geometry: np.ndarray,
        initial_temperature: np.ndarray,
        heat_sources: List[Dict[str, Any]],
        time_steps: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Solve the heat equation numerically.
        
        Args:
            geometry: 3D geometry array
            initial_temperature: Initial temperature field
            heat_sources: List of heat source configurations
            time_steps: Time steps for simulation
            
        Returns:
            Tuple: (temperature_field, convergence_info)
        """
        try:
            # Initialize temperature field
            temperature_field = initial_temperature.copy()
            
            # Set up finite difference scheme
            dx = self.config.spatial_resolution
            dt = self.config.time_step
            
            # Calculate thermal diffusivity
            alpha = self.config.thermal_conductivity / (
                self.config.density * self.config.specific_heat
            )
            
            # Stability condition
            stability_factor = alpha * dt / (dx**2)
            if stability_factor > 0.5:
                logger.warning(f"Stability factor {stability_factor:.3f} > 0.5, simulation may be unstable")
            
            convergence_info = {
                'iterations': 0,
                'converged': True,
                'max_residual': 0.0
            }
            
            # Time stepping
            for t_idx, t in enumerate(time_steps):
                # Apply heat sources
                temperature_field = self._apply_heat_sources(
                    temperature_field, heat_sources, t
                )
                
                # Solve heat equation for one time step
                temperature_field = self._solve_time_step(
                    temperature_field, geometry, alpha, dx, dt
                )
                
                # Check convergence
                if t_idx % 100 == 0:  # Check every 100 time steps
                    residual = self._calculate_residual(temperature_field, geometry)
                    convergence_info['max_residual'] = max(convergence_info['max_residual'], residual)
                    
                    if residual < self.config.convergence_tolerance:
                        convergence_info['converged'] = True
                        break
                
                convergence_info['iterations'] += 1
            
            return temperature_field, convergence_info
            
        except Exception as e:
            logger.error(f"Error solving heat equation: {e}")
            return initial_temperature, {'converged': False, 'error': str(e)}
    
    def _apply_heat_sources(
        self,
        temperature_field: np.ndarray,
        heat_sources: List[Dict[str, Any]],
        current_time: float
    ) -> np.ndarray:
        """Apply heat sources to temperature field."""
        try:
            temp_field = temperature_field.copy()
            
            for heat_source in heat_sources:
                # Check if heat source is active at current time
                start_time = heat_source.get('start_time', 0.0)
                duration = heat_source.get('duration', self.config.max_time)
                
                if start_time <= current_time <= start_time + duration:
                    # Apply heat source
                    position = heat_source['position']
                    power = heat_source['power']
                    radius = heat_source['radius']
                    
                    # Convert position to grid coordinates
                    x, y, z = position
                    grid_x = int(x / self.config.spatial_resolution)
                    grid_y = int(y / self.config.spatial_resolution)
                    grid_z = int(z / self.config.spatial_resolution)
                    
                    # Apply Gaussian heat source
                    temp_field = self._apply_gaussian_heat_source(
                        temp_field, grid_x, grid_y, grid_z, power, radius
                    )
            
            return temp_field
            
        except Exception as e:
            logger.error(f"Error applying heat sources: {e}")
            return temperature_field
    
    def _apply_gaussian_heat_source(
        self,
        temperature_field: np.ndarray,
        center_x: int,
        center_y: int,
        center_z: int,
        power: float,
        radius: float
    ) -> np.ndarray:
        """Apply Gaussian heat source to temperature field."""
        try:
            temp_field = temperature_field.copy()
            
            # Calculate heat source region
            grid_radius = int(radius / self.config.spatial_resolution)
            
            # Apply heat source in 3D region
            for dx in range(-grid_radius, grid_radius + 1):
                for dy in range(-grid_radius, grid_radius + 1):
                    for dz in range(-grid_radius, grid_radius + 1):
                        x = center_x + dx
                        y = center_y + dy
                        z = center_z + dz
                        
                        # Check bounds
                        if (0 <= x < temp_field.shape[0] and
                            0 <= y < temp_field.shape[1] and
                            0 <= z < temp_field.shape[2]):
                            
                            # Calculate distance from center
                            distance = np.sqrt(dx**2 + dy**2 + dz**2) * self.config.spatial_resolution
                            
                            # Apply Gaussian heat source
                            if distance <= radius:
                                heat_intensity = power * np.exp(-(distance**2) / (2 * (radius/3)**2))
                                temp_field[x, y, z] += heat_intensity * self.config.time_step / (
                                    self.config.density * self.config.specific_heat
                                )
            
            return temp_field
            
        except Exception as e:
            logger.error(f"Error applying Gaussian heat source: {e}")
            return temperature_field
    
    def _solve_time_step(
        self,
        temperature_field: np.ndarray,
        geometry: np.ndarray,
        alpha: float,
        dx: float,
        dt: float
    ) -> np.ndarray:
        """Solve heat equation for one time step using finite differences."""
        try:
            temp_field = temperature_field.copy()
            
            # Apply finite difference scheme
            for i in range(1, temp_field.shape[0] - 1):
                for j in range(1, temp_field.shape[1] - 1):
                    for k in range(1, temp_field.shape[2] - 1):
                        # Only solve for material regions
                        if geometry[i, j, k] == 1:
                            # Calculate Laplacian
                            laplacian = (
                                temp_field[i+1, j, k] - 2*temp_field[i, j, k] + temp_field[i-1, j, k]
                            ) / (dx**2) + (
                                temp_field[i, j+1, k] - 2*temp_field[i, j, k] + temp_field[i, j-1, k]
                            ) / (dx**2) + (
                                temp_field[i, j, k+1] - 2*temp_field[i, j, k] + temp_field[i, j, k-1]
                            ) / (dx**2)
                            
                            # Update temperature
                            temp_field[i, j, k] += alpha * laplacian * dt
            
            return temp_field
            
        except Exception as e:
            logger.error(f"Error solving time step: {e}")
            return temperature_field
    
    def _calculate_residual(
        self,
        temperature_field: np.ndarray,
        geometry: np.ndarray
    ) -> float:
        """Calculate residual for convergence checking."""
        try:
            residual = 0.0
            count = 0
            
            for i in range(1, temperature_field.shape[0] - 1):
                for j in range(1, temperature_field.shape[1] - 1):
                    for k in range(1, temperature_field.shape[2] - 1):
                        if geometry[i, j, k] == 1:
                            # Calculate local residual
                            local_residual = abs(temperature_field[i, j, k])
                            residual = max(residual, local_residual)
                            count += 1
            
            return residual / count if count > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating residual: {e}")
            return 0.0


class ThermalAnalyzer:
    """
    Thermal analyzer for PBF-LB/M processes.
    
    This class provides thermal analysis capabilities including thermal history
    analysis, temperature gradient analysis, and thermal field characterization.
    """
    
    def __init__(self):
        """Initialize the thermal analyzer."""
        logger.info("Thermal Analyzer initialized")
    
    def analyze_thermal_history(
        self,
        temperature_field: np.ndarray,
        time_steps: np.ndarray,
        analysis_points: List[Tuple[int, int, int]]
    ) -> Dict[str, Any]:
        """
        Analyze thermal history at specific points.
        
        Args:
            temperature_field: Temperature field over time
            time_steps: Time steps
            analysis_points: List of (x, y, z) points to analyze
            
        Returns:
            Dict: Thermal analysis results
        """
        try:
            analysis_results = {
                'thermal_history': {},
                'peak_temperatures': {},
                'cooling_rates': {},
                'thermal_cycles': {}
            }
            
            for point in analysis_points:
                x, y, z = point
                point_key = f"point_{x}_{y}_{z}"
                
                # Extract temperature history
                temp_history = temperature_field[x, y, z, :] if len(temperature_field.shape) == 4 else [temperature_field[x, y, z]]
                
                # Calculate peak temperature
                peak_temp = np.max(temp_history)
                analysis_results['peak_temperatures'][point_key] = peak_temp
                
                # Calculate cooling rate
                if len(temp_history) > 1:
                    cooling_rate = np.gradient(temp_history, time_steps)
                    analysis_results['cooling_rates'][point_key] = np.min(cooling_rate)
                
                # Store thermal history
                analysis_results['thermal_history'][point_key] = {
                    'time': time_steps.tolist(),
                    'temperature': temp_history.tolist()
                }
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Error in thermal history analysis: {e}")
            return {}
    
    def calculate_temperature_gradients(
        self,
        temperature_field: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Calculate temperature gradients.
        
        Args:
            temperature_field: Temperature field
            
        Returns:
            Dict: Temperature gradients
        """
        try:
            gradients = {}
            
            # Calculate gradients using finite differences
            if len(temperature_field.shape) == 3:
                # 3D temperature field
                grad_x = np.gradient(temperature_field, axis=0)
                grad_y = np.gradient(temperature_field, axis=1)
                grad_z = np.gradient(temperature_field, axis=2)
                
                gradients['gradient_x'] = grad_x
                gradients['gradient_y'] = grad_y
                gradients['gradient_z'] = grad_z
                gradients['gradient_magnitude'] = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
            
            return gradients
            
        except Exception as e:
            logger.error(f"Error calculating temperature gradients: {e}")
            return {}
    
    def analyze_thermal_cycles(
        self,
        temperature_history: np.ndarray,
        time_steps: np.ndarray
    ) -> Dict[str, Any]:
        """
        Analyze thermal cycles.
        
        Args:
            temperature_history: Temperature history
            time_steps: Time steps
            
        Returns:
            Dict: Thermal cycle analysis
        """
        try:
            analysis = {
                'cycle_count': 0,
                'cycle_amplitudes': [],
                'cycle_periods': [],
                'mean_temperature': np.mean(temperature_history),
                'temperature_range': np.max(temperature_history) - np.min(temperature_history)
            }
            
            # Find temperature peaks and valleys
            from scipy.signal import find_peaks
            
            peaks, _ = find_peaks(temperature_history)
            valleys, _ = find_peaks(-temperature_history)
            
            # Calculate cycle amplitudes and periods
            if len(peaks) > 0 and len(valleys) > 0:
                analysis['cycle_count'] = min(len(peaks), len(valleys))
                
                for i in range(min(len(peaks), len(valleys))):
                    amplitude = temperature_history[peaks[i]] - temperature_history[valleys[i]]
                    analysis['cycle_amplitudes'].append(amplitude)
                    
                    if i > 0:
                        period = time_steps[peaks[i]] - time_steps[peaks[i-1]]
                        analysis['cycle_periods'].append(period)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing thermal cycles: {e}")
            return {}
