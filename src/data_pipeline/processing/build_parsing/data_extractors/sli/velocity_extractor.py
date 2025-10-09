"""
Velocity Extractor for EOS/SLI Build Files.

This module provides velocity parameter extraction capabilities for EOS/SLI build files.
Note: EOS/SLI files contain only geometry data, not process parameters like velocity, power, etc.
"""

from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path
import logging
import numpy as np

from .....external import LIBSLM_AVAILABLE

logger = logging.getLogger(__name__)


class VelocityExtractor:
    """
    Extractor for laser velocity parameters from EOS/SLI build files.
    
    Note: EOS/SLI files contain only geometry data (coordinates, layer info) but do not
    contain process parameters like velocity, power, exposure time, etc. These parameters
    are typically defined in separate EOS job files or machine settings.
    """
    
    def __init__(self):
        """Initialize the velocity extractor."""
        self.libslm_available = LIBSLM_AVAILABLE
        
        if not self.libslm_available:
            logger.warning("libSLM not available - velocity analysis will be limited")
        else:
            logger.info("Velocity extractor initialized with libSLM support")
    
    def extract_velocity_data(self, build_data: Any) -> Dict[str, Any]:
        """
        Extract velocity data from EOS/SLI build file data.
        
        Args:
            build_data: Parsed build file data
            
        Returns:
            Dictionary containing velocity analysis results (mostly empty for EOS files)
        """
        try:
            logger.info("Extracting velocity data from EOS/SLI build file")
            
            velocity_data = {
                'global_velocity': self._extract_global_velocity(build_data),
                'layer_velocity': self._extract_layer_velocity(build_data),
                'hatch_velocity': self._extract_hatch_velocity(build_data),
                'contour_velocity': self._extract_contour_velocity(build_data),
                'velocity_statistics': self._calculate_velocity_statistics(build_data),
                'velocity_distribution': self._analyze_velocity_distribution(build_data),
                'spatial_velocity_map': self._create_spatial_velocity_map(build_data),
                'format_note': 'EOS/SLI files contain only geometry data, no process parameters'
            }
            
            logger.info("Successfully extracted velocity data from EOS/SLI file")
            return velocity_data
            
        except Exception as e:
            logger.error(f"Error extracting velocity data: {e}")
            raise
    
    def _extract_global_velocity(self, build_data: Any) -> Dict[str, Any]:
        """Extract global velocity settings (not available in EOS format)."""
        return {
            'note': 'EOS/SLI files do not contain global velocity settings',
            'available_parameters': ['scale_factor', 'z_unit', 'layer_count']
        }
    
    def _extract_layer_velocity(self, build_data: Any) -> List[Dict[str, Any]]:
        """Extract layer-specific velocity settings (not available in EOS format)."""
        layer_velocity = []
        
        try:
            reader = build_data.get('reader_object')
            if reader and hasattr(reader, 'layers'):
                for i, layer in enumerate(reader.layers):
                    layer_info = {
                        'layer_index': i,
                        'z_height': getattr(layer, 'z', None),
                        'is_loaded': layer.isLoaded() if hasattr(layer, 'isLoaded') else None,
                        'velocity_settings': {
                            'note': 'EOS/SLI files do not contain layer velocity settings'
                        }
                    }
                    layer_velocity.append(layer_info)
        except Exception as e:
            logger.warning(f"Error extracting layer velocity: {e}")
        
        return layer_velocity
    
    def _extract_hatch_velocity(self, build_data: Any) -> List[Dict[str, Any]]:
        """Extract hatch geometry data (coordinates only, no velocity parameters)."""
        hatch_velocity = []
        
        try:
            reader = build_data.get('reader_object')
            if not reader:
                logger.warning("No reader object available for hatch geometry extraction")
                return hatch_velocity
            
            if hasattr(reader, 'layers'):
                layers = reader.layers
                for layer_idx, layer in enumerate(layers):
                    try:
                        hatch_geometries = layer.getHatchGeometry()
                        logger.debug(f"Layer {layer_idx}: Found {len(hatch_geometries)} hatch geometries")
                        
                        for hatch_idx, hatch_geom in enumerate(hatch_geometries):
                            coords = hatch_geom.coords if hasattr(hatch_geom, 'coords') else None
                            
                            hatch_info = {
                                'layer_index': layer_idx,
                                'hatch_index': hatch_idx,
                                'build_style_id': getattr(hatch_geom, 'bid', None),
                                'model_id': getattr(hatch_geom, 'mid', None),
                                'coordinates': coords.tolist() if coords is not None else None,
                                'num_segments': coords.shape[0] // 2 if coords is not None else 0,
                                'note': 'EOS/SLI files contain only geometry coordinates, no velocity parameters'
                            }
                            hatch_velocity.append(hatch_info)
                    except Exception as layer_error:
                        logger.warning(f"Error accessing layer {layer_idx}: {layer_error}")
                        continue
        
        except Exception as e:
            logger.warning(f"Error extracting hatch geometry: {e}")
        
        return hatch_velocity
    
    def _extract_contour_velocity(self, build_data: Any) -> List[Dict[str, Any]]:
        """Extract contour geometry data (coordinates only, no velocity parameters)."""
        contour_velocity = []
        
        try:
            reader = build_data.get('reader_object')
            if not reader:
                return contour_velocity
            
            if hasattr(reader, 'layers'):
                layers = reader.layers
                for layer_idx, layer in enumerate(layers):
                    try:
                        contour_geometries = layer.getContourGeometry()
                        logger.debug(f"Layer {layer_idx}: Found {len(contour_geometries)} contour geometries")
                        
                        for contour_idx, contour_geom in enumerate(contour_geometries):
                            coords = contour_geom.coords if hasattr(contour_geom, 'coords') else None
                            
                            contour_info = {
                                'layer_index': layer_idx,
                                'contour_index': contour_idx,
                                'build_style_id': getattr(contour_geom, 'bid', None),
                                'model_id': getattr(contour_geom, 'mid', None),
                                'coordinates': coords.tolist() if coords is not None else None,
                                'num_points': coords.shape[0] if coords is not None else 0,
                                'note': 'EOS/SLI files contain only geometry coordinates, no velocity parameters'
                            }
                            contour_velocity.append(contour_info)
                    except Exception as layer_error:
                        logger.warning(f"Error accessing layer {layer_idx}: {layer_error}")
                        continue
        
        except Exception as e:
            logger.warning(f"Error extracting contour geometry: {e}")
        
        return contour_velocity
    
    def _calculate_velocity_statistics(self, build_data: Any) -> Dict[str, Any]:
        """Calculate velocity statistics (not applicable for EOS format)."""
        return {
            'note': 'EOS/SLI files do not contain velocity data for statistical analysis',
            'count': 0,
            'mean': None,
            'std': None,
            'min': None,
            'max': None,
            'median': None,
            'q25': None,
            'q75': None
        }
    
    def _analyze_velocity_distribution(self, build_data: Any) -> Dict[str, Any]:
        """Analyze velocity distribution patterns (not applicable for EOS format)."""
        return {
            'note': 'EOS/SLI files do not contain velocity data for distribution analysis',
            'layer_variation': None,
            'geometry_distribution': {}
        }
    
    def _create_spatial_velocity_map(self, build_data: Any) -> Dict[str, Any]:
        """Create spatial mapping for geometry data (no velocity values)."""
        spatial_map = {
            'voxel_resolution': None,
            'geometry_voxels': [],
            'coordinate_system': 'build_coordinates',
            'bounds': None,
            'note': 'EOS/SLI files contain only geometry coordinates, no velocity data'
        }
        
        try:
            reader = build_data.get('reader_object')
            if not reader:
                return spatial_map
            
            # Extract build volume bounds from reader
            if hasattr(reader, 'getZUnit'):
                z_unit = reader.getZUnit()
                spatial_map['z_unit'] = z_unit
            
            if hasattr(reader, 'getLayerThickness'):
                layer_thickness = reader.getLayerThickness()
                spatial_map['layer_thickness'] = layer_thickness
            
            # Collect all geometry points for spatial mapping
            geometry_points = []
            
            # From hatches
            hatch_geometries = self._extract_hatch_velocity(build_data)
            for hatch in hatch_geometries:
                if hatch['coordinates']:
                    coords = hatch['coordinates']
                    # Process each coordinate pair in the hatch
                    if coords and len(coords) > 0:
                        for i in range(0, len(coords), 2):
                            if i + 1 < len(coords):
                                # Handle coordinate data properly
                                x_val = coords[i] if isinstance(coords[i], (int, float)) else coords[i][0] if isinstance(coords[i], list) and len(coords[i]) > 0 else 0.0
                                y_val = coords[i + 1] if isinstance(coords[i + 1], (int, float)) else coords[i + 1][0] if isinstance(coords[i + 1], list) and len(coords[i + 1]) > 0 else 0.0
                                
                                geometry_points.append({
                                    'x': float(x_val),
                                    'y': float(y_val),
                                    'z': hatch['layer_index'] * 0.05,  # Assuming 50μm layer thickness
                                    'geometry_type': 'hatch',
                                    'build_style_id': hatch['build_style_id']
                                })
            
            # From contours
            contour_geometries = self._extract_contour_velocity(build_data)
            for contour in contour_geometries:
                if contour['coordinates']:
                    coords = contour['coordinates']
                    # Process each coordinate pair in the contour
                    if coords and len(coords) > 0:
                        for i in range(0, len(coords), 2):
                            if i + 1 < len(coords):
                                # Handle coordinate data properly
                                x_val = coords[i] if isinstance(coords[i], (int, float)) else coords[i][0] if isinstance(coords[i], list) and len(coords[i]) > 0 else 0.0
                                y_val = coords[i + 1] if isinstance(coords[i + 1], (int, float)) else coords[i + 1][0] if isinstance(coords[i + 1], list) and len(coords[i + 1]) > 0 else 0.0
                                
                                geometry_points.append({
                                    'x': float(x_val),
                                    'y': float(y_val),
                                    'z': contour['layer_index'] * 0.05,
                                    'geometry_type': 'contour',
                                    'build_style_id': contour['build_style_id']
                                })
            
            spatial_map['geometry_voxels'] = geometry_points
            spatial_map['total_points'] = len(geometry_points)
            
            # Calculate suggested voxel resolution based on geometry density
            if geometry_points:
                x_coords = [float(p['x']) for p in geometry_points if p['x'] is not None]
                y_coords = [float(p['y']) for p in geometry_points if p['y'] is not None]
                z_coords = [float(p['z']) for p in geometry_points if p['z'] is not None]
                
                if x_coords and y_coords and z_coords:
                    x_range = max(x_coords) - min(x_coords)
                    y_range = max(y_coords) - min(y_coords)
                    z_range = max(z_coords) - min(z_coords)
                    
                    # Suggest resolution based on data density
                    suggested_resolution = min(x_range, y_range, z_range) / 100
                    spatial_map['voxel_resolution'] = max(0.01, suggested_resolution)
                    
                    spatial_map['bounds'] = {
                        'x_min': min(x_coords),
                        'x_max': max(x_coords),
                        'y_min': min(y_coords),
                        'y_max': max(y_coords),
                        'z_min': min(z_coords),
                        'z_max': max(z_coords)
                    }
        
        except Exception as e:
            logger.warning(f"Error creating spatial geometry map: {e}")
            spatial_map['error'] = str(e)
        
        return spatial_map
    
    def get_extractor_info(self) -> Dict[str, Any]:
        """Get extractor information."""
        return {
            'name': 'EOS/SLI Velocity Extractor',
            'libslm_available': self.libslm_available,
            'description': 'Extracts geometry data from EOS/SLI build files (no process parameters available)',
            'note': 'EOS/SLI files contain only geometry coordinates, not velocity/power/process parameters'
        }