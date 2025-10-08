"""
Voxel Interaction Controller for PBF-LB/M Visualization

This module provides the main voxel interaction controller that enables users
to interact with voxel data through various input methods including mouse,
keyboard, and touch gestures.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable, Any
from dataclasses import dataclass
import logging
from datetime import datetime
from enum import Enum

from src.core.domain.value_objects.voxel_coordinates import VoxelCoordinates
from ..core.cad_voxelizer import VoxelGrid
from ..core.multi_modal_fusion import FusedVoxelData

logger = logging.getLogger(__name__)


class InteractionMode(Enum):
    """Interaction modes for voxel visualization."""
    SELECT = "select"          # Select voxels
    INSPECT = "inspect"        # Inspect voxel properties
    ANNOTATE = "annotate"      # Add annotations
    MEASURE = "measure"        # Measure distances/volumes
    NAVIGATE = "navigate"      # Navigate 3D view
    ANALYZE = "analyze"        # Analyze regions


class SelectionType(Enum):
    """Types of voxel selection."""
    SINGLE = "single"          # Single voxel
    RECTANGLE = "rectangle"    # Rectangular region
    CIRCLE = "circle"          # Circular region
    POLYGON = "polygon"        # Polygonal region
    VOLUME = "volume"          # 3D volume
    BY_VALUE = "by_value"      # Select by value range


@dataclass
class InteractionConfig:
    """Configuration for voxel interaction."""
    
    # Selection parameters
    selection_tolerance: float = 0.1  # mm
    multi_select_enabled: bool = True
    selection_highlight_color: str = "#FF0000"  # Red
    selection_alpha: float = 0.7
    
    # Navigation parameters
    zoom_sensitivity: float = 1.2
    pan_sensitivity: float = 1.0
    rotation_sensitivity: float = 1.0
    smooth_camera: bool = True
    
    # Interaction parameters
    double_click_timeout: float = 0.3  # seconds
    long_press_timeout: float = 1.0    # seconds
    gesture_threshold: float = 10.0    # pixels
    
    # Performance parameters
    max_selected_voxels: int = 10000
    update_frequency: float = 60.0     # Hz
    enable_caching: bool = True


@dataclass
class VoxelSelection:
    """Voxel selection information."""
    
    selection_id: str
    selection_type: SelectionType
    voxel_indices: List[Tuple[int, int, int]]
    world_coordinates: List[Tuple[float, float, float]]
    selection_time: datetime
    metadata: Dict[str, Any]


@dataclass
class InteractionEvent:
    """Interaction event information."""
    
    event_type: str  # "click", "drag", "zoom", "rotate", "select"
    position: Tuple[float, float, float]  # World coordinates
    voxel_index: Optional[Tuple[int, int, int]]
    timestamp: datetime
    metadata: Dict[str, Any]


class VoxelController:
    """
    Main voxel interaction controller for PBF-LB/M visualization.
    
    This class provides comprehensive interaction capabilities including:
    - Voxel selection and highlighting
    - 3D navigation and camera control
    - Property inspection and analysis
    - Annotation and measurement tools
    - Gesture recognition and handling
    - Event management and callbacks
    """
    
    def __init__(self, config: InteractionConfig = None):
        """Initialize the voxel controller."""
        self.config = config or InteractionConfig()
        self.current_mode = InteractionMode.SELECT
        self.selections = []
        self.interaction_history = []
        self.event_callbacks = {}
        self.camera_state = {
            'position': (0, 0, 10),
            'target': (0, 0, 0),
            'up': (0, 1, 0),
            'fov': 45.0,
            'zoom': 1.0
        }
        
        logger.info("Voxel Controller initialized")
    
    def set_interaction_mode(self, mode: InteractionMode):
        """Set the current interaction mode."""
        self.current_mode = mode
        logger.info(f"Interaction mode set to: {mode.value}")
    
    def handle_mouse_click(
        self, 
        position: Tuple[float, float], 
        button: str = "left",
        voxel_grid: Optional[VoxelGrid] = None,
        fused_data: Optional[Dict[Tuple[int, int, int], FusedVoxelData]] = None
    ) -> Optional[InteractionEvent]:
        """
        Handle mouse click events.
        
        Args:
            position: Screen coordinates (x, y)
            button: Mouse button ("left", "right", "middle")
            voxel_grid: Voxel grid for coordinate conversion
            fused_data: Fused voxel data for inspection
            
        Returns:
            InteractionEvent: Generated interaction event
        """
        try:
            # Convert screen coordinates to world coordinates
            world_pos = self._screen_to_world(position)
            
            # Find voxel at position
            voxel_idx = None
            if voxel_grid:
                voxel_idx = self._world_to_voxel(world_pos, voxel_grid)
            
            # Create interaction event
            event = InteractionEvent(
                event_type="click",
                position=world_pos,
                voxel_index=voxel_idx,
                timestamp=datetime.now(),
                metadata={
                    'button': button,
                    'screen_position': position,
                    'interaction_mode': self.current_mode.value
                }
            )
            
            # Handle based on interaction mode
            if self.current_mode == InteractionMode.SELECT:
                self._handle_selection_click(event, voxel_grid, fused_data)
            elif self.current_mode == InteractionMode.INSPECT:
                self._handle_inspection_click(event, voxel_grid, fused_data)
            elif self.current_mode == InteractionMode.ANNOTATE:
                self._handle_annotation_click(event, voxel_grid)
            elif self.current_mode == InteractionMode.MEASURE:
                self._handle_measurement_click(event, voxel_grid)
            
            # Store event
            self.interaction_history.append(event)
            
            # Trigger callbacks
            self._trigger_callbacks('click', event)
            
            return event
            
        except Exception as e:
            logger.error(f"Error handling mouse click: {e}")
            return None
    
    def handle_mouse_drag(
        self, 
        start_position: Tuple[float, float],
        end_position: Tuple[float, float],
        button: str = "left",
        voxel_grid: Optional[VoxelGrid] = None
    ) -> Optional[InteractionEvent]:
        """Handle mouse drag events."""
        try:
            # Convert screen coordinates to world coordinates
            start_world = self._screen_to_world(start_position)
            end_world = self._screen_to_world(end_position)
            
            # Create interaction event
            event = InteractionEvent(
                event_type="drag",
                position=end_world,
                voxel_index=None,
                timestamp=datetime.now(),
                metadata={
                    'button': button,
                    'start_screen': start_position,
                    'end_screen': end_position,
                    'start_world': start_world,
                    'end_world': end_world,
                    'interaction_mode': self.current_mode.value
                }
            )
            
            # Handle based on interaction mode
            if self.current_mode == InteractionMode.SELECT:
                self._handle_selection_drag(event, voxel_grid)
            elif self.current_mode == InteractionMode.NAVIGATE:
                self._handle_navigation_drag(event)
            elif self.current_mode == InteractionMode.MEASURE:
                self._handle_measurement_drag(event, voxel_grid)
            
            # Store event
            self.interaction_history.append(event)
            
            # Trigger callbacks
            self._trigger_callbacks('drag', event)
            
            return event
            
        except Exception as e:
            logger.error(f"Error handling mouse drag: {e}")
            return None
    
    def handle_mouse_wheel(
        self, 
        position: Tuple[float, float],
        delta: float,
        voxel_grid: Optional[VoxelGrid] = None
    ) -> Optional[InteractionEvent]:
        """Handle mouse wheel events for zooming."""
        try:
            # Convert screen coordinates to world coordinates
            world_pos = self._screen_to_world(position)
            
            # Update camera zoom
            zoom_factor = 1.0 + delta * 0.1 * self.config.zoom_sensitivity
            self.camera_state['zoom'] *= zoom_factor
            self.camera_state['zoom'] = max(0.1, min(10.0, self.camera_state['zoom']))
            
            # Create interaction event
            event = InteractionEvent(
                event_type="zoom",
                position=world_pos,
                voxel_index=None,
                timestamp=datetime.now(),
                metadata={
                    'delta': delta,
                    'zoom_factor': zoom_factor,
                    'new_zoom': self.camera_state['zoom'],
                    'screen_position': position
                }
            )
            
            # Store event
            self.interaction_history.append(event)
            
            # Trigger callbacks
            self._trigger_callbacks('zoom', event)
            
            return event
            
        except Exception as e:
            logger.error(f"Error handling mouse wheel: {e}")
            return None
    
    def handle_keyboard(
        self, 
        key: str, 
        modifiers: List[str] = None,
        voxel_grid: Optional[VoxelGrid] = None
    ) -> Optional[InteractionEvent]:
        """Handle keyboard events."""
        try:
            if modifiers is None:
                modifiers = []
            
            # Create interaction event
            event = InteractionEvent(
                event_type="keyboard",
                position=(0, 0, 0),
                voxel_index=None,
                timestamp=datetime.now(),
                metadata={
                    'key': key,
                    'modifiers': modifiers,
                    'interaction_mode': self.current_mode.value
                }
            )
            
            # Handle keyboard shortcuts
            if key == 'Escape':
                self._clear_selections()
            elif key == 'Delete' and 'ctrl' in modifiers:
                self._delete_selected_voxels()
            elif key == 's' and 'ctrl' in modifiers:
                self._save_selections()
            elif key == 'l' and 'ctrl' in modifiers:
                self._load_selections()
            elif key == '1':
                self.set_interaction_mode(InteractionMode.SELECT)
            elif key == '2':
                self.set_interaction_mode(InteractionMode.INSPECT)
            elif key == '3':
                self.set_interaction_mode(InteractionMode.ANNOTATE)
            elif key == '4':
                self.set_interaction_mode(InteractionMode.MEASURE)
            elif key == '5':
                self.set_interaction_mode(InteractionMode.NAVIGATE)
            elif key == '6':
                self.set_interaction_mode(InteractionMode.ANALYZE)
            
            # Store event
            self.interaction_history.append(event)
            
            # Trigger callbacks
            self._trigger_callbacks('keyboard', event)
            
            return event
            
        except Exception as e:
            logger.error(f"Error handling keyboard: {e}")
            return None
    
    def select_voxels_by_value_range(
        self, 
        value_range: Tuple[float, float],
        value_type: str = "quality",
        voxel_grid: Optional[VoxelGrid] = None,
        fused_data: Optional[Dict[Tuple[int, int, int], FusedVoxelData]] = None
    ) -> VoxelSelection:
        """Select voxels by value range."""
        try:
            selected_voxels = []
            selected_coords = []
            
            if value_type == "quality" and fused_data:
                for voxel_idx, voxel_data in fused_data.items():
                    if voxel_data.overall_quality_score is not None:
                        if value_range[0] <= voxel_data.overall_quality_score <= value_range[1]:
                            selected_voxels.append(voxel_idx)
                            if voxel_grid:
                                world_coord = self._voxel_to_world(voxel_idx, voxel_grid)
                                selected_coords.append(world_coord)
            
            elif value_type == "porosity" and fused_data:
                for voxel_idx, voxel_data in fused_data.items():
                    if voxel_data.ct_porosity is not None:
                        if value_range[0] <= voxel_data.ct_porosity <= value_range[1]:
                            selected_voxels.append(voxel_idx)
                            if voxel_grid:
                                world_coord = self._voxel_to_world(voxel_idx, voxel_grid)
                                selected_coords.append(world_coord)
            
            # Create selection
            selection = VoxelSelection(
                selection_id=f"value_range_{len(self.selections)}",
                selection_type=SelectionType.BY_VALUE,
                voxel_indices=selected_voxels,
                world_coordinates=selected_coords,
                selection_time=datetime.now(),
                metadata={
                    'value_type': value_type,
                    'value_range': value_range,
                    'voxel_count': len(selected_voxels)
                }
            )
            
            self.selections.append(selection)
            
            logger.info(f"Selected {len(selected_voxels)} voxels by {value_type} range {value_range}")
            return selection
            
        except Exception as e:
            logger.error(f"Error selecting voxels by value range: {e}")
            return None
    
    def get_selected_voxel_info(
        self, 
        voxel_idx: Tuple[int, int, int],
        fused_data: Optional[Dict[Tuple[int, int, int], FusedVoxelData]] = None
    ) -> Dict[str, Any]:
        """Get detailed information about a selected voxel."""
        if not fused_data or voxel_idx not in fused_data:
            return {}
        
        voxel_data = fused_data[voxel_idx]
        
        return {
            'voxel_index': voxel_idx,
            'world_coordinates': (voxel_data.voxel_coordinates.x, 
                                voxel_data.voxel_coordinates.y, 
                                voxel_data.voxel_coordinates.z),
            'is_solid': voxel_data.is_solid,
            'material_type': voxel_data.material_type,
            'laser_power': voxel_data.laser_power,
            'scan_speed': voxel_data.scan_speed,
            'layer_number': voxel_data.layer_number,
            'build_time': voxel_data.build_time,
            'ispm_temperature': voxel_data.ispm_temperature,
            'ispm_melt_pool_size': voxel_data.ispm_melt_pool_size,
            'ispm_acoustic_emissions': voxel_data.ispm_acoustic_emissions,
            'ispm_plume_intensity': voxel_data.ispm_plume_intensity,
            'ct_density': voxel_data.ct_density,
            'ct_intensity': voxel_data.ct_intensity,
            'ct_defect_probability': voxel_data.ct_defect_probability,
            'ct_porosity': voxel_data.ct_porosity,
            'overall_quality_score': voxel_data.overall_quality_score,
            'dimensional_accuracy': voxel_data.dimensional_accuracy,
            'surface_roughness': voxel_data.surface_roughness,
            'defect_count': voxel_data.defect_count,
            'defect_types': voxel_data.defect_types,
            'fusion_confidence': voxel_data.fusion_confidence,
            'data_completeness': voxel_data.data_completeness
        }
    
    def register_callback(self, event_type: str, callback: Callable):
        """Register a callback for interaction events."""
        if event_type not in self.event_callbacks:
            self.event_callbacks[event_type] = []
        self.event_callbacks[event_type].append(callback)
    
    def unregister_callback(self, event_type: str, callback: Callable):
        """Unregister a callback for interaction events."""
        if event_type in self.event_callbacks:
            if callback in self.event_callbacks[event_type]:
                self.event_callbacks[event_type].remove(callback)
    
    # Private methods
    def _screen_to_world(self, screen_pos: Tuple[float, float]) -> Tuple[float, float, float]:
        """Convert screen coordinates to world coordinates."""
        # Simplified conversion - would need proper camera matrix in real implementation
        x = (screen_pos[0] - 400) * 0.01  # Convert to world units
        y = (screen_pos[1] - 300) * 0.01
        z = 0.0  # Project to z=0 plane
        return (x, y, z)
    
    def _world_to_voxel(self, world_pos: Tuple[float, float, float], voxel_grid: VoxelGrid) -> Optional[Tuple[int, int, int]]:
        """Convert world coordinates to voxel index."""
        # Convert to grid coordinates
        grid_x = int((world_pos[0] - voxel_grid.origin[0]) / voxel_grid.voxel_size)
        grid_y = int((world_pos[1] - voxel_grid.origin[1]) / voxel_grid.voxel_size)
        grid_z = int((world_pos[2] - voxel_grid.origin[2]) / voxel_grid.voxel_size)
        
        # Check bounds
        if (0 <= grid_x < voxel_grid.dimensions[0] and
            0 <= grid_y < voxel_grid.dimensions[1] and
            0 <= grid_z < voxel_grid.dimensions[2]):
            return (grid_x, grid_y, grid_z)
        
        return None
    
    def _voxel_to_world(self, voxel_idx: Tuple[int, int, int], voxel_grid: VoxelGrid) -> Tuple[float, float, float]:
        """Convert voxel index to world coordinates."""
        x = voxel_grid.origin[0] + voxel_idx[0] * voxel_grid.voxel_size
        y = voxel_grid.origin[1] + voxel_idx[1] * voxel_grid.voxel_size
        z = voxel_grid.origin[2] + voxel_idx[2] * voxel_grid.voxel_size
        return (x, y, z)
    
    def _handle_selection_click(self, event: InteractionEvent, voxel_grid: Optional[VoxelGrid], fused_data: Optional[Dict]):
        """Handle selection click events."""
        if event.voxel_index and voxel_grid:
            # Create single voxel selection
            selection = VoxelSelection(
                selection_id=f"single_{len(self.selections)}",
                selection_type=SelectionType.SINGLE,
                voxel_indices=[event.voxel_index],
                world_coordinates=[event.position],
                selection_time=event.timestamp,
                metadata={'click_position': event.position}
            )
            
            self.selections.append(selection)
    
    def _handle_inspection_click(self, event: InteractionEvent, voxel_grid: Optional[VoxelGrid], fused_data: Optional[Dict]):
        """Handle inspection click events."""
        if event.voxel_index and fused_data:
            voxel_info = self.get_selected_voxel_info(event.voxel_index, fused_data)
            # Trigger inspection callback
            self._trigger_callbacks('inspect', voxel_info)
    
    def _handle_annotation_click(self, event: InteractionEvent, voxel_grid: Optional[VoxelGrid]):
        """Handle annotation click events."""
        # Create annotation at click position
        annotation = {
            'position': event.position,
            'voxel_index': event.voxel_index,
            'timestamp': event.timestamp,
            'text': f"Annotation at {event.position}"
        }
        # Trigger annotation callback
        self._trigger_callbacks('annotate', annotation)
    
    def _handle_measurement_click(self, event: InteractionEvent, voxel_grid: Optional[VoxelGrid]):
        """Handle measurement click events."""
        # Add measurement point
        measurement_point = {
            'position': event.position,
            'voxel_index': event.voxel_index,
            'timestamp': event.timestamp
        }
        # Trigger measurement callback
        self._trigger_callbacks('measure', measurement_point)
    
    def _handle_selection_drag(self, event: InteractionEvent, voxel_grid: Optional[VoxelGrid]):
        """Handle selection drag events."""
        # Create rectangular selection
        start_pos = event.metadata['start_world']
        end_pos = event.metadata['end_world']
        
        # Find voxels in rectangular region
        selected_voxels = []
        selected_coords = []
        
        if voxel_grid:
            # Simple rectangular selection
            min_x, max_x = min(start_pos[0], end_pos[0]), max(start_pos[0], end_pos[0])
            min_y, max_y = min(start_pos[1], end_pos[1]), max(start_pos[1], end_pos[1])
            
            for x in np.arange(min_x, max_x, voxel_grid.voxel_size):
                for y in np.arange(min_y, max_y, voxel_grid.voxel_size):
                    voxel_idx = self._world_to_voxel((x, y, 0), voxel_grid)
                    if voxel_idx and voxel_grid.voxels[voxel_idx] > 0:
                        selected_voxels.append(voxel_idx)
                        selected_coords.append((x, y, 0))
        
        if selected_voxels:
            selection = VoxelSelection(
                selection_id=f"rectangle_{len(self.selections)}",
                selection_type=SelectionType.RECTANGLE,
                voxel_indices=selected_voxels,
                world_coordinates=selected_coords,
                selection_time=event.timestamp,
                metadata={'drag_region': (start_pos, end_pos)}
            )
            
            self.selections.append(selection)
    
    def _handle_navigation_drag(self, event: InteractionEvent):
        """Handle navigation drag events."""
        # Update camera position
        start_pos = event.metadata['start_world']
        end_pos = event.metadata['end_world']
        
        # Calculate pan delta
        pan_delta = (
            end_pos[0] - start_pos[0],
            end_pos[1] - start_pos[1],
            end_pos[2] - start_pos[2]
        )
        
        # Update camera target
        self.camera_state['target'] = (
            self.camera_state['target'][0] - pan_delta[0] * self.config.pan_sensitivity,
            self.camera_state['target'][1] - pan_delta[1] * self.config.pan_sensitivity,
            self.camera_state['target'][2] - pan_delta[2] * self.config.pan_sensitivity
        )
    
    def _handle_measurement_drag(self, event: InteractionEvent, voxel_grid: Optional[VoxelGrid]):
        """Handle measurement drag events."""
        start_pos = event.metadata['start_world']
        end_pos = event.metadata['end_world']
        
        # Calculate distance
        distance = np.sqrt(
            (end_pos[0] - start_pos[0])**2 +
            (end_pos[1] - start_pos[1])**2 +
            (end_pos[2] - start_pos[2])**2
        )
        
        measurement = {
            'start_position': start_pos,
            'end_position': end_pos,
            'distance': distance,
            'timestamp': event.timestamp
        }
        
        # Trigger measurement callback
        self._trigger_callbacks('measure', measurement)
    
    def _trigger_callbacks(self, event_type: str, data: Any):
        """Trigger registered callbacks for an event type."""
        if event_type in self.event_callbacks:
            for callback in self.event_callbacks[event_type]:
                try:
                    callback(data)
                except Exception as e:
                    logger.error(f"Error in callback for {event_type}: {e}")
    
    def _clear_selections(self):
        """Clear all selections."""
        self.selections.clear()
        logger.info("All selections cleared")
    
    def _delete_selected_voxels(self):
        """Delete selected voxels (placeholder implementation)."""
        logger.info(f"Deleting {len(self.selections)} selections")
        # In a real implementation, this would modify the voxel data
    
    def _save_selections(self):
        """Save selections to file (placeholder implementation)."""
        logger.info(f"Saving {len(self.selections)} selections")
        # In a real implementation, this would save to file
    
    def _load_selections(self):
        """Load selections from file (placeholder implementation)."""
        logger.info("Loading selections from file")
        # In a real implementation, this would load from file
    
    def get_interaction_statistics(self) -> Dict[str, Any]:
        """Get interaction statistics."""
        return {
            'current_mode': self.current_mode.value,
            'selection_count': len(self.selections),
            'interaction_history_length': len(self.interaction_history),
            'camera_state': self.camera_state.copy(),
            'registered_callbacks': {event_type: len(callbacks) for event_type, callbacks in self.event_callbacks.items()}
        }
