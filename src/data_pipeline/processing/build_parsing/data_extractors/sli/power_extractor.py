"""
Power Extractor for EOS/SLI Build Files.

This module provides power parameter extraction capabilities for EOS/SLI build files.
Note: EOS/SLI files contain only geometry data, not process parameters like power, velocity, etc.
"""

from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path
import logging
import numpy as np

from .....external import LIBSLM_AVAILABLE

logger = logging.getLogger(__name__)


class PowerExtractor:
    """
    Extractor for laser power parameters from EOS/SLI build files.
    
    Note: EOS/SLI files contain only geometry data (coordinates, layer info) but do not
    contain process parameters like power, velocity, exposure time, etc. These parameters
    are typically defined in separate EOS job files or machine settings.
    """
    
    def __init__(self):
        """Initialize the power extractor."""
        self.libslm_available = LIBSLM_AVAILABLE
        
        if not self.libslm_available:
            logger.warning("libSLM not available - power analysis will be limited")
        else:
            logger.info("Power extractor initialized with libSLM support")
    
    def extract_power_data(self, build_data: Any) -> Dict[str, Any]:
        """
        Extract power data from EOS/SLI build file data.
        
        Args:
            build_data: Parsed build file data
            
        Returns:
            Dictionary containing power analysis results (mostly empty for EOS files)
        """
        try:
            logger.info("Extracting power data from EOS/SLI build file")
            
            power_data = {
                'global_power': self._extract_global_power(build_data),
                'layer_power': self._extract_layer_power(build_data),
                'geometry_power': self._extract_geometry_power(build_data),
                'hatch_power': self._extract_hatch_power(build_data),
                'contour_power': self._extract_contour_power(build_data),
                'point_power': self._extract_point_power(build_data),
                'power_statistics': self._calculate_power_statistics(build_data),
                'power_distribution': self._analyze_power_distribution(build_data),
                'spatial_power_map': self._create_spatial_power_map(build_data),
                'format_note': 'EOS/SLI files contain only geometry data, no process parameters'
            }
            
            logger.info("Successfully extracted power data from EOS/SLI file")
            return power_data
            
        except Exception as e:
            logger.error(f"Error extracting power data: {e}")
            raise
    
    def _extract_global_power(self, build_data: Any) -> Dict[str, Any]:
        """Extract global power settings (not available in EOS format)."""
        return {
            'note': 'EOS/SLI files do not contain global power settings',
            'available_parameters': ['scale_factor', 'z_unit', 'layer_count']
        }
    
    def _extract_layer_power(self, build_data: Any) -> List[Dict[str, Any]]:
        """Extract layer-specific power settings (not available in EOS format)."""
        layer_power = []
        
        try:
            reader = build_data.get('reader_object')
            if reader and hasattr(reader, 'layers'):
                for i, layer in enumerate(reader.layers):
                    layer_info = {
                        'layer_index': i,
                        'z_height': getattr(layer, 'z', None),
                        'is_loaded': layer.isLoaded() if hasattr(layer, 'isLoaded') else None,
                        'power_settings': {
                            'note': 'EOS/SLI files do not contain layer power settings'
                        }
                    }
                    layer_power.append(layer_info)
        except Exception as e:
            logger.warning(f"Error extracting layer power: {e}")
        
        return layer_power
    
    def _extract_geometry_power(self, build_data: Any) -> List[Dict[str, Any]]:
        """Extract geometry-specific power settings (not available in EOS format)."""
        return [{
            'note': 'EOS/SLI files do not contain geometry-specific power settings',
            'available_data': 'Only geometry coordinates and build style IDs (bid)'
        }]
    
    def _calculate_power_statistics(self, build_data: Any) -> Dict[str, Any]:
        """Calculate power statistics (not applicable for EOS format)."""
        return {
            'note': 'EOS/SLI files do not contain power data for statistical analysis',
            'count': 0,
            'mean': None,
            'std': None,
            'min': None,
            'max': None,
            'median': None,
            'q25': None,
            'q75': None
        }
    
    def _analyze_power_distribution(self, build_data: Any) -> Dict[str, Any]:
        """Analyze power distribution patterns (not applicable for EOS format)."""
        return {
            'note': 'EOS/SLI files do not contain power data for distribution analysis',
            'layer_variation': None,
            'geometry_distribution': {}
        }
    
    def _extract_hatch_power(self, build_data: Any) -> List[Dict[str, Any]]:
        """Extract hatch geometry data (coordinates only, no power parameters)."""
        hatch_power = []
        
        try:
            reader = build_data.get('reader_object')
            if not reader:
                logger.warning("No reader object available for hatch geometry extraction")
                return hatch_power
            
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
                                'note': 'EOS/SLI files contain only geometry coordinates, no process parameters'
                            }
                            hatch_power.append(hatch_info)
                    except Exception as layer_error:
                        logger.warning(f"Error accessing layer {layer_idx}: {layer_error}")
                        continue
        
        except Exception as e:
            logger.warning(f"Error extracting hatch geometry: {e}")
        
        return hatch_power
    
    def _extract_contour_power(self, build_data: Any) -> List[Dict[str, Any]]:
        """Extract contour geometry data (coordinates only, no power parameters)."""
        contour_power = []
        
        try:
            reader = build_data.get('reader_object')
            if not reader:
                return contour_power
            
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
                                'note': 'EOS/SLI files contain only geometry coordinates, no process parameters'
                            }
                            contour_power.append(contour_info)
                    except Exception as layer_error:
                        logger.warning(f"Error accessing layer {layer_idx}: {layer_error}")
                        continue
        
        except Exception as e:
            logger.warning(f"Error extracting contour geometry: {e}")
        
        return contour_power
    
    def _extract_point_power(self, build_data: Any) -> List[Dict[str, Any]]:
        """Extract point geometry data (coordinates only, no power parameters)."""
        point_power = []
        
        try:
            reader = build_data.get('reader_object')
            if not reader:
                return point_power
            
            if hasattr(reader, 'layers'):
                layers = reader.layers
                for layer_idx, layer in enumerate(layers):
                    try:
                        point_geometries = layer.getPointsGeometry()
                        logger.debug(f"Layer {layer_idx}: Found {len(point_geometries)} point geometries")
                        
                        for point_idx, point_geom in enumerate(point_geometries):
                            coords = point_geom.coords if hasattr(point_geom, 'coords') else None
                            
                            point_info = {
                                'layer_index': layer_idx,
                                'point_index': point_idx,
                                'build_style_id': getattr(point_geom, 'bid', None),
                                'model_id': getattr(point_geom, 'mid', None),
                                'coordinates': coords.tolist() if coords is not None else None,
                                'num_points': coords.shape[0] if coords is not None else 0,
                                'note': 'EOS/SLI files contain only geometry coordinates, no process parameters'
                            }
                            point_power.append(point_info)
                    except Exception as layer_error:
                        logger.warning(f"Error accessing layer {layer_idx}: {layer_error}")
                        continue
        
        except Exception as e:
            logger.warning(f"Error extracting point geometry: {e}")
        
        return point_power
    
    def _create_spatial_power_map(self, build_data: Any) -> Dict[str, Any]:
        """Create spatial mapping for geometry data (no power values)."""
        spatial_map = {
            'voxel_resolution': None,
            'geometry_voxels': [],  # Changed from power_voxels to geometry_voxels
            'coordinate_system': 'build_coordinates',
            'bounds': None,
            'note': 'EOS/SLI files contain only geometry coordinates, no power data'
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
            hatch_geometries = self._extract_hatch_power(build_data)
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
            contour_geometries = self._extract_contour_power(build_data)
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
            
            # From points
            point_geometries = self._extract_point_power(build_data)
            for point in point_geometries:
                if point['coordinates']:
                    coords = point['coordinates']
                    # Process each coordinate pair in the point geometry
                    if coords and len(coords) > 0:
                        for i in range(0, len(coords), 2):
                            if i + 1 < len(coords):
                                # Handle coordinate data properly
                                x_val = coords[i] if isinstance(coords[i], (int, float)) else coords[i][0] if isinstance(coords[i], list) and len(coords[i]) > 0 else 0.0
                                y_val = coords[i + 1] if isinstance(coords[i + 1], (int, float)) else coords[i + 1][0] if isinstance(coords[i + 1], list) and len(coords[i + 1]) > 0 else 0.0
                                
                                geometry_points.append({
                                    'x': float(x_val),
                                    'y': float(y_val),
                                    'z': point['layer_index'] * 0.05,
                                    'geometry_type': 'point',
                                    'build_style_id': point['build_style_id']
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
            'name': 'EOS/SLI Power Extractor',
            'libslm_available': self.libslm_available,
            'description': 'Extracts geometry data from EOS/SLI build files (no process parameters available)',
            'note': 'EOS/SLI files contain only geometry coordinates, not power/velocity/process parameters'
        }