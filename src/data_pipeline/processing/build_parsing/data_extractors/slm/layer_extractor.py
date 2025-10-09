"""
Layer Extractor for PBF-LB/M Build Files.

This module provides layer information extraction capabilities for PBF-LB/M build files,
leveraging libSLM for accessing layer-specific data and build progression.
"""

from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path
import logging
import numpy as np

from .....external import LIBSLM_AVAILABLE

logger = logging.getLogger(__name__)


class LayerExtractor:
    """
    Extractor for layer information from PBF-LB/M build files.
    
    This extractor analyzes layer data, thickness, and build progression
    using libSLM data structures.
    """
    
    def __init__(self):
        """Initialize the layer extractor."""
        self.libslm_available = LIBSLM_AVAILABLE
        
        if not self.libslm_available:
            logger.warning("libSLM not available - layer analysis will be limited")
        else:
            logger.info("Layer extractor initialized with libSLM support")
    
    def extract_layer_data(self, build_data: Any) -> Dict[str, Any]:
        """
        Extract layer data from build file data.
        
        Args:
            build_data: Parsed build file data from libSLM
            
        Returns:
            Dictionary containing layer analysis results
        """
        try:
            logger.info("Extracting layer data from build file")
            
            layer_data = {
                'layer_info': self._extract_layer_info(build_data),
                'layer_statistics': self._calculate_layer_statistics(build_data),
                'build_progression': self._analyze_build_progression(build_data),
                'layer_geometry': self._analyze_layer_geometry(build_data),
                'spatial_layer_map': self._create_spatial_layer_map(build_data)
            }
            
            logger.info("Successfully extracted layer data")
            return layer_data
            
        except Exception as e:
            logger.error(f"Error extracting layer data: {e}")
            raise
    
    def _extract_layer_info(self, build_data: Any) -> List[Dict[str, Any]]:
        """Extract detailed layer information using libSLM data structures."""
        layer_info = []
        
        try:
            if hasattr(build_data, 'layers'):
                for layer_idx, layer in enumerate(build_data.layers):
                    layer_data = {
                        'layer_index': layer_idx,
                        'z_height': getattr(layer, 'z_height', None),
                        'thickness': getattr(layer, 'thickness', None),
                        'hatch_count': 0,
                        'contour_count': 0,
                        'point_count': 0,
                        'total_length': 0.0,
                        'area': 0.0,
                        'volume': 0.0,
                        'parameters': {}
                    }
                    
                    # Count geometries using libSLM methods
                    hatch_geometries = layer.getHatchGeometry() if hasattr(layer, 'getHatchGeometry') else []
                    layer_data['hatch_count'] = len(hatch_geometries)
                    
                    # Calculate total hatch length from coordinates
                    for hatch_geom in hatch_geometries:
                        coords = hatch_geom.coords if hasattr(hatch_geom, 'coords') else None
                        if coords is not None:
                            length = self._calculate_path_length_from_coords(coords)
                            layer_data['total_length'] += length
                    
                    contour_geometries = layer.getContourGeometry() if hasattr(layer, 'getContourGeometry') else []
                    layer_data['contour_count'] = len(contour_geometries)
                    # Calculate layer area from contours
                    layer_data['area'] = self._calculate_layer_area_from_geometries(contour_geometries)
                    
                    point_geometries = layer.getPointGeometry() if hasattr(layer, 'getPointGeometry') else []
                    layer_data['point_count'] = len(point_geometries)
                    
                    # Calculate volume (area * thickness)
                    if layer_data['area'] > 0 and layer_data['thickness']:
                        layer_data['volume'] = layer_data['area'] * layer_data['thickness']
                    
                    # Extract layer parameters
                    if hasattr(layer, 'parameters'):
                        layer_params = layer.parameters
                        layer_data['parameters'] = {
                            'hatch_power': getattr(layer_params, 'hatch_power', None),
                            'contour_power': getattr(layer_params, 'contour_power', None),
                            'hatch_velocity': getattr(layer_params, 'hatch_velocity', None),
                            'contour_velocity': getattr(layer_params, 'contour_velocity', None),
                            'hatch_distance': getattr(layer_params, 'hatch_distance', None),
                            'contour_offset': getattr(layer_params, 'contour_offset', None)
                        }
                    
                    layer_info.append(layer_data)
        
        except Exception as e:
            logger.warning(f"Error extracting layer info: {e}")
        
        return layer_info
    
    def _calculate_layer_statistics(self, build_data: Any) -> Dict[str, Any]:
        """Calculate layer statistics across the build."""
        statistics = {}
        
        try:
            layer_info = self._extract_layer_info(build_data)
            
            if layer_info:
                # Extract arrays for statistics
                z_heights = [l['z_height'] for l in layer_info if l['z_height'] is not None]
                thicknesses = [l['thickness'] for l in layer_info if l['thickness'] is not None]
                hatch_counts = [l['hatch_count'] for l in layer_info]
                contour_counts = [l['contour_count'] for l in layer_info]
                point_counts = [l['point_count'] for l in layer_info]
                total_lengths = [l['total_length'] for l in layer_info]
                areas = [l['area'] for l in layer_info if l['area'] > 0]
                volumes = [l['volume'] for l in layer_info if l['volume'] > 0]
                
                statistics = {
                    'total_layers': len(layer_info),
                    'build_height': max(z_heights) - min(z_heights) if z_heights else 0,
                    'z_height_stats': self._calculate_stats(z_heights),
                    'thickness_stats': self._calculate_stats(thicknesses),
                    'hatch_count_stats': self._calculate_stats(hatch_counts),
                    'contour_count_stats': self._calculate_stats(contour_counts),
                    'point_count_stats': self._calculate_stats(point_counts),
                    'total_length_stats': self._calculate_stats(total_lengths),
                    'area_stats': self._calculate_stats(areas),
                    'volume_stats': self._calculate_stats(volumes)
                }
            else:
                statistics = {
                    'total_layers': 0,
                    'build_height': 0,
                    'error': 'No layer data available'
                }
        
        except Exception as e:
            logger.warning(f"Error calculating layer statistics: {e}")
            statistics = {'error': str(e)}
        
        return statistics
    
    def _analyze_build_progression(self, build_data: Any) -> Dict[str, Any]:
        """Analyze build progression patterns."""
        progression = {}
        
        try:
            layer_info = self._extract_layer_info(build_data)
            
            if layer_info:
                # Analyze thickness progression
                thicknesses = [l['thickness'] for l in layer_info if l['thickness'] is not None]
                if thicknesses:
                    thickness_array = np.array(thicknesses)
                    progression['thickness_progression'] = {
                        'trend': self._detect_trend(thickness_array),
                        'consistency': float(1.0 - np.std(thickness_array) / np.mean(thickness_array)) if np.mean(thickness_array) > 0 else 0,
                        'variation_coefficient': float(np.std(thickness_array) / np.mean(thickness_array)) if np.mean(thickness_array) > 0 else 0
                    }
                
                # Analyze geometry progression
                hatch_counts = [l['hatch_count'] for l in layer_info]
                contour_counts = [l['contour_count'] for l in layer_info]
                areas = [l['area'] for l in layer_info if l['area'] > 0]
                
                if hatch_counts:
                    hatch_array = np.array(hatch_counts)
                    progression['hatch_progression'] = {
                        'trend': self._detect_trend(hatch_array),
                        'total_hatches': int(np.sum(hatch_array)),
                        'average_hatches_per_layer': float(np.mean(hatch_array))
                    }
                
                if contour_counts:
                    contour_array = np.array(contour_counts)
                    progression['contour_progression'] = {
                        'trend': self._detect_trend(contour_array),
                        'total_contours': int(np.sum(contour_array)),
                        'average_contours_per_layer': float(np.mean(contour_array))
                    }
                
                if areas:
                    area_array = np.array(areas)
                    progression['area_progression'] = {
                        'trend': self._detect_trend(area_array),
                        'total_area': float(np.sum(area_array)),
                        'average_area_per_layer': float(np.mean(area_array))
                    }
        
        except Exception as e:
            logger.warning(f"Error analyzing build progression: {e}")
            progression = {'error': str(e)}
        
        return progression
    
    def _analyze_layer_geometry(self, build_data: Any) -> Dict[str, Any]:
        """Analyze layer geometry patterns."""
        geometry = {}
        
        try:
            layer_info = self._extract_layer_info(build_data)
            
            if layer_info:
                # Analyze layer complexity
                complexity_scores = []
                for layer in layer_info:
                    # Simple complexity score based on geometry counts
                    complexity = (layer['hatch_count'] * 1.0 + 
                                layer['contour_count'] * 2.0 + 
                                layer['point_count'] * 0.5)
                    complexity_scores.append(complexity)
                
                if complexity_scores:
                    complexity_array = np.array(complexity_scores)
                    geometry['complexity_analysis'] = {
                        'mean_complexity': float(np.mean(complexity_array)),
                        'std_complexity': float(np.std(complexity_array)),
                        'max_complexity': float(np.max(complexity_array)),
                        'min_complexity': float(np.min(complexity_array)),
                        'complexity_trend': self._detect_trend(complexity_array)
                    }
                
                # Analyze layer density
                densities = []
                for layer in layer_info:
                    if layer['area'] > 0 and layer['thickness']:
                        # Density = total_length / area
                        density = layer['total_length'] / layer['area'] if layer['area'] > 0 else 0
                        densities.append(density)
                
                if densities:
                    density_array = np.array(densities)
                    geometry['density_analysis'] = {
                        'mean_density': float(np.mean(density_array)),
                        'std_density': float(np.std(density_array)),
                        'density_trend': self._detect_trend(density_array)
                    }
        
        except Exception as e:
            logger.warning(f"Error analyzing layer geometry: {e}")
            geometry = {'error': str(e)}
        
        return geometry
    
    def _create_spatial_layer_map(self, build_data: Any) -> Dict[str, Any]:
        """Create 3D spatial layer mapping for voxelization."""
        spatial_map = {
            'voxel_resolution': None,
            'layer_voxels': [],
            'coordinate_system': 'build_coordinates',
            'bounds': None
        }
        
        try:
            # Extract build volume bounds
            if hasattr(build_data, 'build_volume'):
                spatial_map['bounds'] = {
                    'x_min': getattr(build_data.build_volume, 'x_min', 0),
                    'x_max': getattr(build_data.build_volume, 'x_max', 100),
                    'y_min': getattr(build_data.build_volume, 'y_min', 0),
                    'y_max': getattr(build_data.build_volume, 'y_max', 100),
                    'z_min': getattr(build_data.build_volume, 'z_min', 0),
                    'z_max': getattr(build_data.build_volume, 'z_max', 100)
                }
            
            # Create layer voxels
            layer_info = self._extract_layer_info(build_data)
            layer_voxels = []
            
            for layer in layer_info:
                if layer['z_height'] is not None:
                    layer_voxel = {
                        'layer_index': layer['layer_index'],
                        'z_height': layer['z_height'],
                        'thickness': layer['thickness'],
                        'hatch_count': layer['hatch_count'],
                        'contour_count': layer['contour_count'],
                        'point_count': layer['point_count'],
                        'total_length': layer['total_length'],
                        'area': layer['area'],
                        'volume': layer['volume'],
                        'complexity': (layer['hatch_count'] * 1.0 + 
                                     layer['contour_count'] * 2.0 + 
                                     layer['point_count'] * 0.5)
                    }
                    layer_voxels.append(layer_voxel)
            
            spatial_map['layer_voxels'] = layer_voxels
            spatial_map['total_layers'] = len(layer_voxels)
            
            # Calculate suggested voxel resolution
            if layer_voxels and spatial_map['bounds']:
                bounds = spatial_map['bounds']
                z_range = bounds['z_max'] - bounds['z_min']
                
                # Use average layer thickness for z-resolution
                thicknesses = [l['thickness'] for l in layer_voxels if l['thickness']]
                if thicknesses:
                    avg_thickness = np.mean(thicknesses)
                    spatial_map['voxel_resolution'] = {
                        'x': 0.1,  # 100μm in x-y
                        'y': 0.1,  # 100μm in x-y
                        'z': max(0.01, avg_thickness / 2)  # Half layer thickness in z
                    }
        
        except Exception as e:
            logger.warning(f"Error creating spatial layer map: {e}")
            spatial_map['error'] = str(e)
        
        return spatial_map
    
    def _calculate_layer_area(self, contours: List[Any]) -> float:
        """Calculate total area of a layer from contours."""
        try:
            total_area = 0.0
            
            for contour in contours:
                points = getattr(contour, 'points', None)
                if points and len(points) >= 3:
                    # Use shoelace formula
                    x_coords = [p[0] for p in points if len(p) > 0]
                    y_coords = [p[1] for p in points if len(p) > 1]
                    
                    if len(x_coords) == len(y_coords) and len(x_coords) >= 3:
                        area = 0.5 * abs(sum(x_coords[i] * y_coords[(i+1) % len(x_coords)] - 
                                           x_coords[(i+1) % len(x_coords)] * y_coords[i] 
                                           for i in range(len(x_coords))))
                        total_area += area
            
            return total_area
        
        except Exception:
            return 0.0
    
    def _calculate_stats(self, values: List[float]) -> Dict[str, Any]:
        """Calculate basic statistics for a list of values."""
        if not values:
            return {'count': 0, 'mean': None, 'std': None, 'min': None, 'max': None}
        
        try:
            value_array = np.array(values)
            return {
                'count': len(values),
                'mean': float(np.mean(value_array)),
                'std': float(np.std(value_array)),
                'min': float(np.min(value_array)),
                'max': float(np.max(value_array)),
                'median': float(np.median(value_array))
            }
        
        except Exception:
            return {'count': 0, 'mean': None, 'std': None, 'min': None, 'max': None, 'median': None}
    
    def _detect_trend(self, values: np.ndarray) -> str:
        """Detect trend in values."""
        try:
            if len(values) < 2:
                return 'insufficient_data'
            
            x = np.arange(len(values))
            slope = np.polyfit(x, values, 1)[0]
            
            if abs(slope) < 0.01:
                return 'stable'
            elif slope > 0:
                return 'increasing'
            else:
                return 'decreasing'
        
        except Exception:
            return 'unknown'
    
    def get_extractor_info(self) -> Dict[str, Any]:
        """Get extractor information."""
        return {
            'name': 'Layer Extractor',
            'libslm_available': self.libslm_available,
            'description': 'Extracts and analyzes layer information from PBF-LB/M build files using libSLM'
        }
