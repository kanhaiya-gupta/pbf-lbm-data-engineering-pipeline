"""
Simulation Engines Module for PBF-LB/M Virtual Environment

This module provides comprehensive simulation engine capabilities including
thermal simulation, fluid dynamics, mechanical simulation, material physics,
and multi-physics coupling for PBF-LB/M virtual testing and simulation
environments.
"""

from .thermal_simulation import ThermalSimulator, ThermalSolver, ThermalAnalyzer
from .fluid_dynamics import FluidDynamicsSimulator, CFDSolver, FlowAnalyzer
from .mechanical_simulation import MechanicalSimulator, StressSolver, DeformationAnalyzer
from .material_physics import MaterialPhysicsSimulator, PhaseChangeSolver, MicrostructureAnalyzer
from .multi_physics import MultiPhysicsSimulator, PhysicsCoupler, CoupledSolver

__all__ = [
    'ThermalSimulator',
    'ThermalSolver',
    'ThermalAnalyzer',
    'FluidDynamicsSimulator',
    'CFDSolver',
    'FlowAnalyzer',
    'MechanicalSimulator',
    'StressSolver',
    'DeformationAnalyzer',
    'MaterialPhysicsSimulator',
    'PhaseChangeSolver',
    'MicrostructureAnalyzer',
    'MultiPhysicsSimulator',
    'PhysicsCoupler',
    'CoupledSolver',
]
