"""
Timestamp Extractor for PBF-LB/M Build Files using libSLM's built-in timing.

This module leverages libSLM's native timing capabilities to extract
timestamp-coordinate data efficiently without manual calculations.
"""

from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path
import logging
import numpy as np
from datetime import datetime, timedelta

from ....external import LIBSLM_AVAILABLE

logger = logging.getLogger(__name__)


class TimestampExtractor:
    """
    Extractor for timestamp-coordinate data using libSLM's built-in timing.
    
    This extractor leverages libSLM's native timing capabilities to extract
    timestamp-coordinate data efficiently without manual calculations.
    """
    
    def __init__(self):
        """Initialize the timestamp extractor."""
        self.libslm_available = LIBSLM_AVAILABLE
        
        if not self.libslm_available:
            logger.warning("libSLM not available - timestamp extraction will be limited")
        else:
            logger.info("Timestamp extractor initialized with libSLM support")
    
    def extract_timestamp_data(self, build_data: Any, build_start_time: datetime) -> Dict[str, Any]:
        """
        Extract timestamp-coordinate data using libSLM's built-in timing.
        
        Args:
            build_data: Parsed build file data from libSLM
            build_start_time: Absolute start time for the build
            
        Returns:
            Dictionary containing timestamp-coordinate data
        """
        try:
            logger.info("Extracting timestamp data using libSLM's built-in timing")
            
            timestamp_data = {
                'build_metadata': {
                    'build_start_time': build_start_time,
                    'total_build_time': self._get_total_build_time(build_data),
                    'build_end_time': build_start_time + timedelta(seconds=self._get_total_build_time(build_data))
                },
                'layer_timestamps': self._extract_layer_timestamps(build_data, build_start_time),
                'geometry_timestamps': self._extract_geometry_timestamps(build_data, build_start_time),
                'laser_timeline': self._extract_laser_timeline(build_data, build_start_time),
                'position_timeline': self._extract_position_timeline(build_data, build_start_time)
            }
            
            logger.info("Successfully extracted timestamp data using libSLM")
            return timestamp_data
            
        except Exception as e:
            logger.error(f"Error extracting timestamp data: {e}")
            raise
    
    def _get_total_build_time(self, build_data: Any) -> float:
        """Get total build time using libSLM's built-in method."""
        try:
            if hasattr(build_data, 'getBuildTime'):
                return build_data.getBuildTime()
            else:
                logger.warning("getBuildTime method not available, calculating manually")
                return self._calculate_total_build_time(build_data)
        except Exception as e:
            logger.warning(f"Error getting total build time: {e}")
            return 0.0
    
    def _extract_layer_timestamps(self, build_data: Any, build_start_time: datetime) -> List[Dict[str, Any]]:
        """Extract layer timestamps using libSLM's built-in methods."""
        layer_timestamps = []
        
        try:
            if hasattr(build_data, 'getTimeByLayerId'):
                # Use libSLM's built-in timing
                for layer_idx in range(len(build_data.layers)):
                    layer_time = build_data.getTimeByLayerId(layer_idx)
                    layer_timestamps.append({
                        'layer_index': layer_idx,
                        'relative_time': layer_time,
                        'absolute_timestamp': build_start_time + timedelta(seconds=layer_time),
                        'layer_duration': self._get_layer_duration(build_data, layer_idx)
                    })
            else:
                # Fallback to manual calculation
                layer_timestamps = self._calculate_layer_timestamps(build_data, build_start_time)
                
        except Exception as e:
            logger.warning(f"Error extracting layer timestamps: {e}")
            layer_timestamps = self._calculate_layer_timestamps(build_data, build_start_time)
        
        return layer_timestamps
    
    def _extract_geometry_timestamps(self, build_data: Any, build_start_time: datetime) -> List[Dict[str, Any]]:
        """Extract geometry timestamps using libSLM's built-in methods."""
        geometry_timestamps = []
        
        try:
            if hasattr(build_data, 'getTimeByLayerGeomId'):
                # Use libSLM's built-in timing
                for layer_idx, layer in enumerate(build_data.layers):
                    for geom_idx in range(len(layer.geometries)):
                        geom_time = build_data.getTimeByLayerGeomId(layer_idx, geom_idx)
                        geometry_timestamps.append({
                            'layer_index': layer_idx,
                            'geometry_index': geom_idx,
                            'relative_time': geom_time,
                            'absolute_timestamp': build_start_time + timedelta(seconds=geom_time),
                            'geometry_duration': self._get_geometry_duration(build_data, layer_idx, geom_idx)
                        })
            else:
                # Fallback to manual calculation
                geometry_timestamps = self._calculate_geometry_timestamps(build_data, build_start_time)
                
        except Exception as e:
            logger.warning(f"Error extracting geometry timestamps: {e}")
            geometry_timestamps = self._calculate_geometry_timestamps(build_data, build_start_time)
        
        return geometry_timestamps
    
    def _extract_laser_timeline(self, build_data: Any, build_start_time: datetime) -> List[Dict[str, Any]]:
        """Extract laser timeline using libSLM's built-in methods."""
        laser_timeline = []
        
        try:
            if hasattr(build_data, 'getLaserParameters'):
                # Use libSLM's built-in laser parameter extraction
                total_time = self._get_total_build_time(build_data)
                time_step = 0.1  # 100ms intervals
                
                current_time = 0.0
                while current_time < total_time:
                    power = 0.0
                    exp_time = 0
                    pnt_dist = 0
                    is_laser_on = False
                    
                    build_data.getLaserParameters(current_time, power, exp_time, pnt_dist, is_laser_on)
                    
                    laser_timeline.append({
                        'relative_time': current_time,
                        'absolute_timestamp': build_start_time + timedelta(seconds=current_time),
                        'power': power,
                        'exposure_time': exp_time,
                        'point_distance': pnt_dist,
                        'is_laser_on': is_laser_on
                    })
                    
                    current_time += time_step
            else:
                # Fallback to manual calculation
                laser_timeline = self._calculate_laser_timeline(build_data, build_start_time)
                
        except Exception as e:
            logger.warning(f"Error extracting laser timeline: {e}")
            laser_timeline = self._calculate_laser_timeline(build_data, build_start_time)
        
        return laser_timeline
    
    def _extract_position_timeline(self, build_data: Any, build_start_time: datetime) -> List[Dict[str, Any]]:
        """Extract position timeline using libSLM's built-in methods."""
        position_timeline = []
        
        try:
            if hasattr(build_data, 'getLaserPositionByTime'):
                # Use libSLM's built-in position extraction
                total_time = self._get_total_build_time(build_data)
                time_step = 0.1  # 100ms intervals
                
                current_time = 0.0
                while current_time < total_time:
                    position = build_data.getLaserPositionByTime(current_time)
                    
                    position_timeline.append({
                        'relative_time': current_time,
                        'absolute_timestamp': build_start_time + timedelta(seconds=current_time),
                        'x': position[0] if len(position) > 0 else 0.0,
                        'y': position[1] if len(position) > 1 else 0.0,
                        'z': self._get_z_position_by_time(build_data, current_time)
                    })
                    
                    current_time += time_step
            else:
                # Fallback to manual calculation
                position_timeline = self._calculate_position_timeline(build_data, build_start_time)
                
        except Exception as e:
            logger.warning(f"Error extracting position timeline: {e}")
            position_timeline = self._calculate_position_timeline(build_data, build_start_time)
        
        return position_timeline
    
    def _get_layer_duration(self, build_data: Any, layer_idx: int) -> float:
        """Get layer duration using libSLM's built-in methods."""
        try:
            if hasattr(build_data, 'getTimeByLayerId'):
                if layer_idx == 0:
                    return build_data.getTimeByLayerId(0)
                else:
                    return build_data.getTimeByLayerId(layer_idx) - build_data.getTimeByLayerId(layer_idx - 1)
            else:
                return 0.0
        except Exception as e:
            logger.warning(f"Error getting layer duration: {e}")
            return 0.0
    
    def _get_geometry_duration(self, build_data: Any, layer_idx: int, geom_idx: int) -> float:
        """Get geometry duration using libSLM's built-in methods."""
        try:
            if hasattr(build_data, 'calcGeomTime'):
                layer = build_data.layers[layer_idx]
                geometry = layer.geometries[geom_idx]
                return build_data.calcGeomTime(geometry)
            else:
                return 0.0
        except Exception as e:
            logger.warning(f"Error getting geometry duration: {e}")
            return 0.0
    
    def _get_z_position_by_time(self, build_data: Any, time: float) -> float:
        """Get Z position by time using libSLM's built-in methods."""
        try:
            if hasattr(build_data, 'getLayerIdByTime'):
                layer_id = build_data.getLayerIdByTime(time)
                # Assume layer thickness of 0.05mm (50μm)
                return layer_id * 0.05
            else:
                return 0.0
        except Exception as e:
            logger.warning(f"Error getting Z position: {e}")
            return 0.0
    
    def _calculate_total_build_time(self, build_data: Any) -> float:
        """Fallback method to calculate total build time manually."""
        total_time = 0.0
        
        try:
            for layer in build_data.layers:
                layer_time = 0.0
                for geometry in layer.geometries:
                    # Calculate geometry time manually
                    if hasattr(geometry, 'coords') and geometry.coords is not None:
                        coords = geometry.coords
                        # Simple calculation based on coordinates and build style
                        # This is a fallback - libSLM's calcGeomTime is more accurate
                        layer_time += len(coords) * 0.1  # Assume 100ms per point
                total_time += layer_time
        except Exception as e:
            logger.warning(f"Error calculating total build time: {e}")
        
        return total_time
    
    def _calculate_layer_timestamps(self, build_data: Any, build_start_time: datetime) -> List[Dict[str, Any]]:
        """Fallback method to calculate layer timestamps manually."""
        layer_timestamps = []
        current_time = 0.0
        
        try:
            for layer_idx, layer in enumerate(build_data.layers):
                layer_start_time = current_time
                layer_duration = 0.0
                
                for geometry in layer.geometries:
                    if hasattr(geometry, 'coords') and geometry.coords is not None:
                        layer_duration += len(geometry.coords) * 0.1  # Assume 100ms per point
                
                layer_timestamps.append({
                    'layer_index': layer_idx,
                    'relative_time': current_time,
                    'absolute_timestamp': build_start_time + timedelta(seconds=current_time),
                    'layer_duration': layer_duration
                })
                
                current_time += layer_duration
        except Exception as e:
            logger.warning(f"Error calculating layer timestamps: {e}")
        
        return layer_timestamps
    
    def _calculate_geometry_timestamps(self, build_data: Any, build_start_time: datetime) -> List[Dict[str, Any]]:
        """Fallback method to calculate geometry timestamps manually."""
        geometry_timestamps = []
        current_time = 0.0
        
        try:
            for layer_idx, layer in enumerate(build_data.layers):
                for geom_idx, geometry in enumerate(layer.geometries):
                    geom_duration = 0.0
                    if hasattr(geometry, 'coords') and geometry.coords is not None:
                        geom_duration = len(geometry.coords) * 0.1  # Assume 100ms per point
                    
                    geometry_timestamps.append({
                        'layer_index': layer_idx,
                        'geometry_index': geom_idx,
                        'relative_time': current_time,
                        'absolute_timestamp': build_start_time + timedelta(seconds=current_time),
                        'geometry_duration': geom_duration
                    })
                    
                    current_time += geom_duration
        except Exception as e:
            logger.warning(f"Error calculating geometry timestamps: {e}")
        
        return geometry_timestamps
    
    def _calculate_laser_timeline(self, build_data: Any, build_start_time: datetime) -> List[Dict[str, Any]]:
        """Fallback method to calculate laser timeline manually."""
        laser_timeline = []
        
        try:
            # This is a simplified fallback - libSLM's getLaserParameters is more accurate
            for layer_idx, layer in enumerate(build_data.layers):
                for geom_idx, geometry in enumerate(layer.geometries):
                    if hasattr(geometry, 'coords') and geometry.coords is not None:
                        coords = geometry.coords
                        for i in range(0, len(coords), 2):
                            if i + 1 < len(coords):
                                laser_timeline.append({
                                    'relative_time': current_time,
                                    'absolute_timestamp': build_start_time + timedelta(seconds=current_time),
                                    'power': 200.0,  # Default power
                                    'exposure_time': 100,  # Default exposure time
                                    'point_distance': 0.05,  # Default point distance
                                    'is_laser_on': True
                                })
                                current_time += 0.1  # Assume 100ms per point
        except Exception as e:
            logger.warning(f"Error calculating laser timeline: {e}")
        
        return laser_timeline
    
    def _calculate_position_timeline(self, build_data: Any, build_start_time: datetime) -> List[Dict[str, Any]]:
        """Fallback method to calculate position timeline manually."""
        position_timeline = []
        current_time = 0.0
        
        try:
            for layer_idx, layer in enumerate(build_data.layers):
                for geom_idx, geometry in enumerate(layer.geometries):
                    if hasattr(geometry, 'coords') and geometry.coords is not None:
                        coords = geometry.coords
                        for i in range(0, len(coords), 2):
                            if i + 1 < len(coords):
                                position_timeline.append({
                                    'relative_time': current_time,
                                    'absolute_timestamp': build_start_time + timedelta(seconds=current_time),
                                    'x': coords[i],
                                    'y': coords[i + 1],
                                    'z': layer_idx * 0.05  # Assume 50μm layer thickness
                                })
                                current_time += 0.1  # Assume 100ms per point
        except Exception as e:
            logger.warning(f"Error calculating position timeline: {e}")
        
        return position_timeline
    
    def get_extractor_info(self) -> Dict[str, Any]:
        """Get extractor information."""
        return {
            'name': 'Timestamp Extractor',
            'libslm_available': self.libslm_available,
            'description': 'Extracts timestamp-coordinate data using libSLM\'s built-in timing capabilities'
        }
