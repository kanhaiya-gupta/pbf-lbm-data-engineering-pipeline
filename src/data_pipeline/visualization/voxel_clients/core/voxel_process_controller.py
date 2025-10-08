"""
Voxel-Level Process Control for PBF-LB/M Systems

This module provides advanced voxel-level process control capabilities that enable
spatially-resolved optimization of PBF-LB/M process parameters. It integrates with
the multi-modal data fusion system to provide real-time process control and
optimization at the voxel level.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import logging
from datetime import datetime
from enum import Enum
import json

from src.core.domain.value_objects.voxel_coordinates import VoxelCoordinates
from src.core.domain.value_objects.process_parameters import ProcessParameters
from src.core.domain.value_objects.quality_metrics import QualityMetrics
from src.core.domain.value_objects.defect_classification import DefectClassification

from .cad_voxelizer import VoxelGrid
from .multi_modal_fusion import FusedVoxelData

logger = logging.getLogger(__name__)


class ControlMode(Enum):
    """Process control modes."""
    REACTIVE = "reactive"          # React to detected issues
    PREDICTIVE = "predictive"      # Predict and prevent issues
    OPTIMIZING = "optimizing"      # Continuously optimize parameters
    ADAPTIVE = "adaptive"          # Adapt to changing conditions


class OptimizationObjective(Enum):
    """Optimization objectives."""
    QUALITY_MAXIMIZATION = "quality_maximization"
    DEFECT_MINIMIZATION = "defect_minimization"
    SPEED_MAXIMIZATION = "speed_maximization"
    ENERGY_MINIMIZATION = "energy_minimization"
    MULTI_OBJECTIVE = "multi_objective"


@dataclass
class VoxelControlAction:
    """Control action for a specific voxel."""
    
    voxel_index: Tuple[int, int, int]
    action_type: str  # "laser_power_adjust", "scan_speed_adjust", "pause", "skip"
    parameter_changes: Dict[str, float]
    confidence: float
    expected_improvement: float
    risk_level: str  # "low", "medium", "high"
    timestamp: datetime
    reasoning: str


@dataclass
class ProcessControlConfig:
    """Configuration for voxel-level process control."""
    
    control_mode: ControlMode = ControlMode.PREDICTIVE
    optimization_objective: OptimizationObjective = OptimizationObjective.QUALITY_MAXIMIZATION
    
    # Control thresholds
    quality_threshold: float = 80.0
    defect_threshold: float = 0.05
    temperature_threshold: float = 2000.0  # Celsius
    
    # Parameter adjustment limits
    max_laser_power_change: float = 0.1  # 10% max change
    max_scan_speed_change: float = 0.2   # 20% max change
    min_laser_power: float = 50.0        # Watts
    max_laser_power: float = 500.0       # Watts
    min_scan_speed: float = 100.0        # mm/s
    max_scan_speed: float = 2000.0       # mm/s
    
    # Control parameters
    control_frequency: float = 1.0       # Hz
    prediction_horizon: int = 10         # voxels ahead
    safety_margin: float = 0.1           # 10% safety margin
    
    # Machine learning parameters
    ml_model_path: Optional[str] = None
    feature_importance_threshold: float = 0.1
    prediction_confidence_threshold: float = 0.8


class VoxelProcessController:
    """
    Voxel-level process controller for PBF-LB/M systems.
    
    This class provides comprehensive process control capabilities including:
    - Real-time voxel-level parameter optimization
    - Predictive process control
    - Defect prevention and mitigation
    - Quality optimization
    - Multi-objective optimization
    """
    
    def __init__(self, config: ProcessControlConfig = None):
        """Initialize the voxel process controller."""
        self.config = config or ProcessControlConfig()
        self.control_history = []
        self.performance_metrics = {}
        self.ml_models = {}
        
        logger.info(f"Voxel Process Controller initialized in {self.config.control_mode.value} mode")
    
    def control_voxel_process(
        self,
        voxel_grid: VoxelGrid,
        fused_data: Dict[Tuple[int, int, int], FusedVoxelData],
        current_layer: int,
        build_progress: float
    ) -> List[VoxelControlAction]:
        """
        Control process parameters at the voxel level.
        
        Args:
            voxel_grid: Voxel grid representation
            fused_data: Fused voxel data with ISPM, CT, and quality information
            current_layer: Current build layer
            build_progress: Build progress (0.0 to 1.0)
            
        Returns:
            List of control actions for voxels
        """
        try:
            logger.info(f"Starting voxel-level process control for layer {current_layer}")
            
            control_actions = []
            
            # Get voxels for current layer
            layer_voxels = self._get_layer_voxels(voxel_grid, current_layer)
            
            # Control each voxel in the layer
            for voxel_idx in layer_voxels:
                if voxel_idx in fused_data:
                    action = self._control_single_voxel(
                        voxel_idx, fused_data[voxel_idx], voxel_grid
                    )
                    if action:
                        control_actions.append(action)
            
            # Apply global optimizations
            global_actions = self._apply_global_optimizations(
                control_actions, voxel_grid, fused_data, current_layer
            )
            control_actions.extend(global_actions)
            
            # Validate control actions
            validated_actions = self._validate_control_actions(control_actions)
            
            # Store control history
            self._store_control_history(validated_actions, current_layer)
            
            logger.info(f"Voxel-level control completed: {len(validated_actions)} actions generated")
            return validated_actions
            
        except Exception as e:
            logger.error(f"Error in voxel-level process control: {e}")
            raise
    
    def _get_layer_voxels(
        self, 
        voxel_grid: VoxelGrid, 
        layer_number: int
    ) -> List[Tuple[int, int, int]]:
        """Get voxel indices for a specific layer."""
        layer_voxels = []
        
        # Find voxels in the specified layer
        for z in range(voxel_grid.dimensions[2]):
            layer_voxels_z = np.where(
                (voxel_grid.voxels[:, :, z] > 0) & 
                (voxel_grid.process_map['layer_number'][:, :, z] == layer_number)
            )
            
            for i in range(len(layer_voxels_z[0])):
                voxel_idx = (layer_voxels_z[0][i], layer_voxels_z[1][i], z)
                layer_voxels.append(voxel_idx)
        
        return layer_voxels
    
    def _control_single_voxel(
        self,
        voxel_idx: Tuple[int, int, int],
        voxel_data: FusedVoxelData,
        voxel_grid: VoxelGrid
    ) -> Optional[VoxelControlAction]:
        """Control process parameters for a single voxel."""
        
        # Analyze voxel state
        voxel_analysis = self._analyze_voxel_state(voxel_data)
        
        # Determine control action based on mode
        if self.config.control_mode == ControlMode.REACTIVE:
            action = self._reactive_control(voxel_idx, voxel_data, voxel_analysis)
        elif self.config.control_mode == ControlMode.PREDICTIVE:
            action = self._predictive_control(voxel_idx, voxel_data, voxel_analysis)
        elif self.config.control_mode == ControlMode.OPTIMIZING:
            action = self._optimizing_control(voxel_idx, voxel_data, voxel_analysis)
        elif self.config.control_mode == ControlMode.ADAPTIVE:
            action = self._adaptive_control(voxel_idx, voxel_data, voxel_analysis)
        else:
            action = None
        
        return action
    
    def _analyze_voxel_state(self, voxel_data: FusedVoxelData) -> Dict:
        """Analyze the current state of a voxel."""
        analysis = {
            'quality_score': voxel_data.overall_quality_score,
            'defect_probability': voxel_data.ct_defect_probability or 0.0,
            'temperature': voxel_data.ispm_temperature or 0.0,
            'porosity': voxel_data.ct_porosity or 0.0,
            'fusion_confidence': voxel_data.fusion_confidence,
            'data_completeness': voxel_data.data_completeness,
            'risk_level': 'low'
        }
        
        # Determine risk level
        if (analysis['quality_score'] < self.config.quality_threshold or
            analysis['defect_probability'] > self.config.defect_threshold or
            analysis['temperature'] > self.config.temperature_threshold):
            analysis['risk_level'] = 'high'
        elif (analysis['quality_score'] < self.config.quality_threshold * 1.1 or
              analysis['defect_probability'] > self.config.defect_threshold * 0.8):
            analysis['risk_level'] = 'medium'
        
        return analysis
    
    def _reactive_control(
        self,
        voxel_idx: Tuple[int, int, int],
        voxel_data: FusedVoxelData,
        analysis: Dict
    ) -> Optional[VoxelControlAction]:
        """Reactive control - respond to detected issues."""
        
        if analysis['risk_level'] == 'high':
            # Immediate corrective action needed
            parameter_changes = self._calculate_corrective_parameters(voxel_data, analysis)
            
            if parameter_changes:
                return VoxelControlAction(
                    voxel_index=voxel_idx,
                    action_type="parameter_adjustment",
                    parameter_changes=parameter_changes,
                    confidence=0.9,
                    expected_improvement=0.2,
                    risk_level="medium",
                    timestamp=datetime.now(),
                    reasoning=f"Reactive correction for {analysis['risk_level']} risk level"
                )
        
        return None
    
    def _predictive_control(
        self,
        voxel_idx: Tuple[int, int, int],
        voxel_data: FusedVoxelData,
        analysis: Dict
    ) -> Optional[VoxelControlAction]:
        """Predictive control - prevent issues before they occur."""
        
        # Predict future state
        future_state = self._predict_voxel_future_state(voxel_data, analysis)
        
        if future_state['predicted_risk'] == 'high':
            # Preventive action needed
            parameter_changes = self._calculate_preventive_parameters(voxel_data, future_state)
            
            if parameter_changes:
                return VoxelControlAction(
                    voxel_index=voxel_idx,
                    action_type="preventive_adjustment",
                    parameter_changes=parameter_changes,
                    confidence=future_state['prediction_confidence'],
                    expected_improvement=future_state['expected_improvement'],
                    risk_level="low",
                    timestamp=datetime.now(),
                    reasoning=f"Predictive prevention: {future_state['predicted_issue']}"
                )
        
        return None
    
    def _optimizing_control(
        self,
        voxel_idx: Tuple[int, int, int],
        voxel_data: FusedVoxelData,
        analysis: Dict
    ) -> Optional[VoxelControlAction]:
        """Optimizing control - continuously optimize parameters."""
        
        # Find optimal parameters for this voxel
        optimal_params = self._optimize_voxel_parameters(voxel_data, analysis)
        
        if optimal_params:
            # Calculate parameter changes
            current_params = {
                'laser_power': voxel_data.laser_power,
                'scan_speed': voxel_data.scan_speed
            }
            
            parameter_changes = {}
            for param, optimal_value in optimal_params.items():
                if param in current_params:
                    change = (optimal_value - current_params[param]) / current_params[param]
                    if abs(change) > 0.01:  # Only change if significant
                        parameter_changes[param] = change
            
            if parameter_changes:
                return VoxelControlAction(
                    voxel_index=voxel_idx,
                    action_type="optimization",
                    parameter_changes=parameter_changes,
                    confidence=0.8,
                    expected_improvement=0.15,
                    risk_level="low",
                    timestamp=datetime.now(),
                    reasoning=f"Parameter optimization for {self.config.optimization_objective.value}"
                )
        
        return None
    
    def _adaptive_control(
        self,
        voxel_idx: Tuple[int, int, int],
        voxel_data: FusedVoxelData,
        analysis: Dict
    ) -> Optional[VoxelControlAction]:
        """Adaptive control - adapt to changing conditions."""
        
        # Learn from historical data
        historical_performance = self._get_historical_performance(voxel_idx)
        
        # Adapt parameters based on learning
        adapted_params = self._adapt_parameters(voxel_data, analysis, historical_performance)
        
        if adapted_params:
            return VoxelControlAction(
                voxel_index=voxel_idx,
                action_type="adaptive_adjustment",
                parameter_changes=adapted_params,
                confidence=0.7,
                expected_improvement=0.1,
                risk_level="low",
                timestamp=datetime.now(),
                reasoning="Adaptive parameter adjustment based on historical performance"
            )
        
        return None
    
    def _calculate_corrective_parameters(
        self, 
        voxel_data: FusedVoxelData, 
        analysis: Dict
    ) -> Dict[str, float]:
        """Calculate corrective parameter changes."""
        parameter_changes = {}
        
        # Adjust laser power based on quality issues
        if analysis['quality_score'] < self.config.quality_threshold:
            # Increase laser power to improve quality
            power_increase = min(0.1, (self.config.quality_threshold - analysis['quality_score']) / 100)
            parameter_changes['laser_power'] = power_increase
        
        # Adjust scan speed based on defects
        if analysis['defect_probability'] > self.config.defect_threshold:
            # Reduce scan speed to reduce defects
            speed_reduction = min(0.2, analysis['defect_probability'] * 2)
            parameter_changes['scan_speed'] = -speed_reduction
        
        # Adjust for temperature issues
        if analysis['temperature'] > self.config.temperature_threshold:
            # Reduce laser power to lower temperature
            power_reduction = min(0.15, (analysis['temperature'] - self.config.temperature_threshold) / 1000)
            parameter_changes['laser_power'] = -power_reduction
        
        return parameter_changes
    
    def _calculate_preventive_parameters(
        self, 
        voxel_data: FusedVoxelData, 
        future_state: Dict
    ) -> Dict[str, float]:
        """Calculate preventive parameter changes."""
        parameter_changes = {}
        
        # Prevent predicted quality issues
        if future_state.get('predicted_quality_issue'):
            parameter_changes['laser_power'] = 0.05  # Slight increase
        
        # Prevent predicted defects
        if future_state.get('predicted_defect_issue'):
            parameter_changes['scan_speed'] = -0.1  # Slight reduction
        
        return parameter_changes
    
    def _optimize_voxel_parameters(
        self, 
        voxel_data: FusedVoxelData, 
        analysis: Dict
    ) -> Dict[str, float]:
        """Optimize parameters for a voxel based on objective."""
        
        if self.config.optimization_objective == OptimizationObjective.QUALITY_MAXIMIZATION:
            return self._optimize_for_quality(voxel_data, analysis)
        elif self.config.optimization_objective == OptimizationObjective.DEFECT_MINIMIZATION:
            return self._optimize_for_defect_reduction(voxel_data, analysis)
        elif self.config.optimization_objective == OptimizationObjective.SPEED_MAXIMIZATION:
            return self._optimize_for_speed(voxel_data, analysis)
        elif self.config.optimization_objective == OptimizationObjective.ENERGY_MINIMIZATION:
            return self._optimize_for_energy_efficiency(voxel_data, analysis)
        elif self.config.optimization_objective == OptimizationObjective.MULTI_OBJECTIVE:
            return self._optimize_multi_objective(voxel_data, analysis)
        
        return {}
    
    def _optimize_for_quality(self, voxel_data: FusedVoxelData, analysis: Dict) -> Dict[str, float]:
        """Optimize parameters for maximum quality."""
        optimal_params = {}
        
        # Increase laser power for better fusion
        if analysis['quality_score'] < 95:
            optimal_params['laser_power'] = voxel_data.laser_power * 1.05
        
        # Optimize scan speed for quality
        if analysis['defect_probability'] > 0.01:
            optimal_params['scan_speed'] = voxel_data.scan_speed * 0.95
        
        return optimal_params
    
    def _optimize_for_defect_reduction(self, voxel_data: FusedVoxelData, analysis: Dict) -> Dict[str, float]:
        """Optimize parameters for defect minimization."""
        optimal_params = {}
        
        # Reduce scan speed to minimize defects
        if analysis['defect_probability'] > 0.02:
            optimal_params['scan_speed'] = voxel_data.scan_speed * 0.9
        
        # Adjust laser power for optimal fusion
        if analysis['porosity'] > 0.03:
            optimal_params['laser_power'] = voxel_data.laser_power * 1.1
        
        return optimal_params
    
    def _optimize_for_speed(self, voxel_data: FusedVoxelData, analysis: Dict) -> Dict[str, float]:
        """Optimize parameters for maximum speed."""
        optimal_params = {}
        
        # Increase scan speed while maintaining quality
        if analysis['quality_score'] > 85:
            optimal_params['scan_speed'] = voxel_data.scan_speed * 1.1
        
        return optimal_params
    
    def _optimize_for_energy_efficiency(self, voxel_data: FusedVoxelData, analysis: Dict) -> Dict[str, float]:
        """Optimize parameters for energy efficiency."""
        optimal_params = {}
        
        # Reduce laser power while maintaining quality
        if analysis['quality_score'] > 90:
            optimal_params['laser_power'] = voxel_data.laser_power * 0.95
        
        return optimal_params
    
    def _optimize_multi_objective(self, voxel_data: FusedVoxelData, analysis: Dict) -> Dict[str, float]:
        """Multi-objective optimization."""
        # Combine multiple objectives with weights
        quality_params = self._optimize_for_quality(voxel_data, analysis)
        defect_params = self._optimize_for_defect_reduction(voxel_data, analysis)
        energy_params = self._optimize_for_energy_efficiency(voxel_data, analysis)
        
        # Weighted combination
        optimal_params = {}
        for param in ['laser_power', 'scan_speed']:
            values = []
            if param in quality_params:
                values.append(quality_params[param] * 0.4)
            if param in defect_params:
                values.append(defect_params[param] * 0.4)
            if param in energy_params:
                values.append(energy_params[param] * 0.2)
            
            if values:
                optimal_params[param] = sum(values)
        
        return optimal_params
    
    def _predict_voxel_future_state(
        self, 
        voxel_data: FusedVoxelData, 
        analysis: Dict
    ) -> Dict:
        """Predict future state of a voxel."""
        # Simplified prediction model - can be enhanced with ML
        predicted_risk = 'low'
        predicted_issue = None
        prediction_confidence = 0.7
        expected_improvement = 0.1
        
        # Predict based on current trends
        if analysis['quality_score'] < 85 and analysis['defect_probability'] > 0.03:
            predicted_risk = 'high'
            predicted_issue = 'quality_degradation'
            prediction_confidence = 0.8
            expected_improvement = 0.2
        
        return {
            'predicted_risk': predicted_risk,
            'predicted_issue': predicted_issue,
            'prediction_confidence': prediction_confidence,
            'expected_improvement': expected_improvement
        }
    
    def _get_historical_performance(self, voxel_idx: Tuple[int, int, int]) -> Dict:
        """Get historical performance data for a voxel."""
        # Placeholder implementation
        return {
            'average_quality': 85.0,
            'defect_rate': 0.02,
            'success_rate': 0.95
        }
    
    def _adapt_parameters(
        self, 
        voxel_data: FusedVoxelData, 
        analysis: Dict, 
        historical_performance: Dict
    ) -> Dict[str, float]:
        """Adapt parameters based on historical performance."""
        parameter_changes = {}
        
        # Adapt based on historical quality
        if historical_performance['average_quality'] < 90:
            parameter_changes['laser_power'] = 0.05
        
        # Adapt based on historical defect rate
        if historical_performance['defect_rate'] > 0.03:
            parameter_changes['scan_speed'] = -0.1
        
        return parameter_changes
    
    def _apply_global_optimizations(
        self,
        control_actions: List[VoxelControlAction],
        voxel_grid: VoxelGrid,
        fused_data: Dict[Tuple[int, int, int], FusedVoxelData],
        current_layer: int
    ) -> List[VoxelControlAction]:
        """Apply global optimizations across the layer."""
        global_actions = []
        
        # Analyze layer-wide patterns
        layer_analysis = self._analyze_layer_patterns(voxel_grid, fused_data, current_layer)
        
        # Apply global corrections if needed
        if layer_analysis['layer_quality'] < 80:
            # Global quality improvement
            for voxel_idx in self._get_layer_voxels(voxel_grid, current_layer):
                if voxel_idx in fused_data:
                    global_action = VoxelControlAction(
                        voxel_index=voxel_idx,
                        action_type="global_quality_improvement",
                        parameter_changes={'laser_power': 0.02},
                        confidence=0.6,
                        expected_improvement=0.05,
                        risk_level="low",
                        timestamp=datetime.now(),
                        reasoning="Global layer quality improvement"
                    )
                    global_actions.append(global_action)
        
        return global_actions
    
    def _analyze_layer_patterns(
        self,
        voxel_grid: VoxelGrid,
        fused_data: Dict[Tuple[int, int, int], FusedVoxelData],
        layer_number: int
    ) -> Dict:
        """Analyze patterns across the entire layer."""
        layer_voxels = self._get_layer_voxels(voxel_grid, layer_number)
        
        quality_scores = []
        defect_probabilities = []
        
        for voxel_idx in layer_voxels:
            if voxel_idx in fused_data:
                voxel_data = fused_data[voxel_idx]
                quality_scores.append(voxel_data.overall_quality_score or 100.0)
                defect_probabilities.append(voxel_data.ct_defect_probability or 0.0)
        
        return {
            'layer_quality': np.mean(quality_scores) if quality_scores else 100.0,
            'layer_defect_rate': np.mean(defect_probabilities) if defect_probabilities else 0.0,
            'quality_variance': np.var(quality_scores) if quality_scores else 0.0
        }
    
    def _validate_control_actions(
        self, 
        control_actions: List[VoxelControlAction]
    ) -> List[VoxelControlAction]:
        """Validate control actions for safety and feasibility."""
        validated_actions = []
        
        for action in control_actions:
            # Check parameter limits
            if self._validate_parameter_limits(action):
                validated_actions.append(action)
            else:
                logger.warning(f"Control action rejected due to parameter limits: {action.voxel_index}")
        
        return validated_actions
    
    def _validate_parameter_limits(self, action: VoxelControlAction) -> bool:
        """Validate that parameter changes are within limits."""
        for param, change in action.parameter_changes.items():
            if param == 'laser_power':
                if abs(change) > self.config.max_laser_power_change:
                    return False
            elif param == 'scan_speed':
                if abs(change) > self.config.max_scan_speed_change:
                    return False
        
        return True
    
    def _store_control_history(
        self, 
        control_actions: List[VoxelControlAction], 
        layer_number: int
    ):
        """Store control actions in history."""
        history_entry = {
            'timestamp': datetime.now(),
            'layer_number': layer_number,
            'actions': control_actions,
            'action_count': len(control_actions)
        }
        
        self.control_history.append(history_entry)
        
        # Keep only recent history (last 100 layers)
        if len(self.control_history) > 100:
            self.control_history = self.control_history[-100:]
    
    def get_control_performance_metrics(self) -> Dict:
        """Get performance metrics for the control system."""
        if not self.control_history:
            return {}
        
        total_actions = sum(entry['action_count'] for entry in self.control_history)
        avg_actions_per_layer = total_actions / len(self.control_history)
        
        return {
            'total_control_actions': total_actions,
            'average_actions_per_layer': avg_actions_per_layer,
            'control_history_length': len(self.control_history),
            'control_mode': self.config.control_mode.value,
            'optimization_objective': self.config.optimization_objective.value
        }
    
    def export_control_config(self, output_path: str):
        """Export control configuration to file."""
        try:
            config_dict = {
                'control_mode': self.config.control_mode.value,
                'optimization_objective': self.config.optimization_objective.value,
                'quality_threshold': self.config.quality_threshold,
                'defect_threshold': self.config.defect_threshold,
                'temperature_threshold': self.config.temperature_threshold,
                'max_laser_power_change': self.config.max_laser_power_change,
                'max_scan_speed_change': self.config.max_scan_speed_change,
                'control_frequency': self.config.control_frequency,
                'prediction_horizon': self.config.prediction_horizon,
                'safety_margin': self.config.safety_margin
            }
            
            with open(output_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
            
            logger.info(f"Control configuration exported to: {output_path}")
            
        except Exception as e:
            logger.error(f"Error exporting control configuration: {e}")
            raise
