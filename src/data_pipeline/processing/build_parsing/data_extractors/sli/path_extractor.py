"""
Path Extractor for EOS/SLI Build Files.

This module provides path parameter extraction capabilities for EOS/SLI build files.
Note: EOS/SLI files contain only geometry data, not process parameters like power, velocity, etc.
"""

from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path
import logging
import numpy as np

from .....external import LIBSLM_AVAILABLE

logger = logging.getLogger(__name__)


class PathExtractor:
    """
    Extractor for scan path parameters from EOS/SLI build files.
    
    Note: EOS/SLI files contain only geometry data (coordinates, layer info) but do not
    contain process parameters like power, velocity, exposure time, etc. These parameters
    are typically defined in separate EOS job files or machine settings.
    """
    
    def __init__(self):
        """Initialize the path extractor."""
        self.libslm_available = LIBSLM_AVAILABLE
        
        if not self.libslm_available:
            logger.warning("libSLM not available - path analysis will be limited")
        else:
            logger.info("Path extractor initialized with libSLM support")
    
    def extract_path_data(self, build_data: Any) -> Dict[str, Any]:
        """
        Extract path data from EOS/SLI build file data.
        
        Args:
            build_data: Parsed build file data
            
        Returns:
            Dictionary containing path analysis results (geometry data only for EOS files)
        """
        try:
            logger.info("Extracting path data from EOS/SLI build file")
            
            path_data = {
                'hatch_paths': self._extract_hatch_paths(build_data),
                'contour_paths': self._extract_contour_paths(build_data),
                'point_paths': self._extract_point_paths(build_data),
                'path_statistics': self._calculate_path_statistics(build_data),
                'path_geometry': self._analyze_path_geometry(build_data),
                'spatial_path_map': self._create_spatial_path_map(build_data),
                'format_note': 'EOS/SLI files contain only geometry data, no process parameters'
            }
            
            logger.info("Successfully extracted path data from EOS/SLI file")
            return path_data
            
        except Exception as e:
            logger.error(f"Error extracting path data: {e}")
            raise
    
    def _extract_hatch_paths(self, build_data: Any) -> List[Dict[str, Any]]:
        """Extract hatch path geometry data."""
        hatch_paths = []
        
        try:
            reader = build_data.get('reader_object')
            if not reader:
                logger.warning("No reader object available for hatch path extraction")
                return hatch_paths
            
            if hasattr(reader, 'layers'):
                layers = reader.layers
                for layer_idx, layer in enumerate(layers):
                    try:
                        hatch_geometries = layer.getHatchGeometry()
                        logger.debug(f"Layer {layer_idx}: Found {len(hatch_geometries)} hatch geometries")
                        
                        for hatch_idx, hatch_geom in enumerate(hatch_geometries):
                            coords = hatch_geom.coords if hasattr(hatch_geom, 'coords') else None
                            
                            # Calculate path length from coordinates
                            path_length = self._calculate_path_length(coords) if coords is not None else 0
                            
                            hatch_info = {
                                'layer_index': layer_idx,
                                'hatch_index': hatch_idx,
                                'build_style_id': getattr(hatch_geom, 'bid', None),
                                'model_id': getattr(hatch_geom, 'mid', None),
                                'coordinates': coords.tolist() if coords is not None else None,
                                'num_segments': coords.shape[0] // 2 if coords is not None else 0,
                                'path_length': path_length,
                                'start_point': coords[0].tolist() if coords is not None and len(coords) > 0 else None,
                                'end_point': coords[-1].tolist() if coords is not None and len(coords) > 0 else None,
                                'note': 'EOS/SLI files contain only geometry coordinates, no process parameters'
                            }
                            hatch_paths.append(hatch_info)
                    except Exception as layer_error:
                        logger.warning(f"Error accessing layer {layer_idx}: {layer_error}")
                        continue
        
        except Exception as e:
            logger.warning(f"Error extracting hatch paths: {e}")
        
        return hatch_paths
    
    def _extract_contour_paths(self, build_data: Any) -> List[Dict[str, Any]]:
        """Extract contour path geometry data."""
        contour_paths = []
        
        try:
            reader = build_data.get('reader_object')
            if not reader:
                return contour_paths
            
            if hasattr(reader, 'layers'):
                layers = reader.layers
                for layer_idx, layer in enumerate(layers):
                    try:
                        contour_geometries = layer.getContourGeometry()
                        logger.debug(f"Layer {layer_idx}: Found {len(contour_geometries)} contour geometries")
                        
                        for contour_idx, contour_geom in enumerate(contour_geometries):
                            coords = contour_geom.coords if hasattr(contour_geom, 'coords') else None
                            
                            # Calculate path length from coordinates
                            path_length = self._calculate_path_length(coords) if coords is not None else 0
                            
                            contour_info = {
                                'layer_index': layer_idx,
                                'contour_index': contour_idx,
                                'build_style_id': getattr(contour_geom, 'bid', None),
                                'model_id': getattr(contour_geom, 'mid', None),
                                'coordinates': coords.tolist() if coords is not None else None,
                                'num_points': coords.shape[0] if coords is not None else 0,
                                'path_length': path_length,
                                'start_point': coords[0].tolist() if coords is not None and len(coords) > 0 else None,
                                'end_point': coords[-1].tolist() if coords is not None and len(coords) > 0 else None,
                                'note': 'EOS/SLI files contain only geometry coordinates, no process parameters'
                            }
                            contour_paths.append(contour_info)
                    except Exception as layer_error:
                        logger.warning(f"Error accessing layer {layer_idx}: {layer_error}")
                        continue
        
        except Exception as e:
            logger.warning(f"Error extracting contour paths: {e}")
        
        return contour_paths
    
    def _extract_point_paths(self, build_data: Any) -> List[Dict[str, Any]]:
        """Extract point path geometry data."""
        point_paths = []
        
        try:
            reader = build_data.get('reader_object')
            if not reader:
                return point_paths
            
            if hasattr(reader, 'layers'):
                layers = reader.layers
                for layer_idx, layer in enumerate(layers):
                    try:
                        point_geometries = layer.getPointsGeometry()
                        logger.debug(f"Layer {layer_idx}: Found {len(point_geometries)} point geometries")
                        
                        for point_idx, point_geom in enumerate(point_geometries):
                            coords = point_geom.coords if hasattr(point_geom, 'coords') else None
                            
                            # Calculate path length from coordinates
                            path_length = self._calculate_path_length(coords) if coords is not None else 0
                            
                            point_info = {
                                'layer_index': layer_idx,
                                'point_index': point_idx,
                                'build_style_id': getattr(point_geom, 'bid', None),
                                'model_id': getattr(point_geom, 'mid', None),
                                'coordinates': coords.tolist() if coords is not None else None,
                                'num_points': coords.shape[0] if coords is not None else 0,
                                'path_length': path_length,
                                'start_point': coords[0].tolist() if coords is not None and len(coords) > 0 else None,
                                'end_point': coords[-1].tolist() if coords is not None and len(coords) > 0 else None,
                                'note': 'EOS/SLI files contain only geometry coordinates, no process parameters'
                            }
                            point_paths.append(point_info)
                    except Exception as layer_error:
                        logger.warning(f"Error accessing layer {layer_idx}: {layer_error}")
                        continue
        
        except Exception as e:
            logger.warning(f"Error extracting point paths: {e}")
        
        return point_paths
    
    def _calculate_path_length(self, coords: np.ndarray) -> float:
        """Calculate total path length from coordinate array."""
        if coords is None or len(coords) < 2:
            return 0.0
        
        try:
            total_length = 0.0
            for i in range(len(coords) - 1):
                # Calculate distance between consecutive points
                dx = coords[i + 1, 0] - coords[i, 0]
                dy = coords[i + 1, 1] - coords[i, 1]
                segment_length = np.sqrt(dx * dx + dy * dy)
                total_length += segment_length
            
            return float(total_length)
        except Exception as e:
            logger.warning(f"Error calculating path length: {e}")
            return 0.0
    
    def _calculate_path_statistics(self, build_data: Any) -> Dict[str, Any]:
        """Calculate path statistics from geometry data."""
        statistics = {
            'total_hatch_paths': 0,
            'total_contour_paths': 0,
            'total_point_paths': 0,
            'total_path_length': 0.0,
            'average_hatch_length': 0.0,
            'average_contour_length': 0.0,
            'average_point_length': 0.0,
            'longest_path': 0.0,
            'shortest_path': 0.0
        }
        
        try:
            # Get all path data
            hatch_paths = self._extract_hatch_paths(build_data)
            contour_paths = self._extract_contour_paths(build_data)
            point_paths = self._extract_point_paths(build_data)
            
            statistics['total_hatch_paths'] = len(hatch_paths)
            statistics['total_contour_paths'] = len(contour_paths)
            statistics['total_point_paths'] = len(point_paths)
            
            # Calculate length statistics
            all_lengths = []
            
            for path in hatch_paths:
                if path['path_length'] > 0:
                    all_lengths.append(path['path_length'])
            
            for path in contour_paths:
                if path['path_length'] > 0:
                    all_lengths.append(path['path_length'])
            
            for path in point_paths:
                if path['path_length'] > 0:
                    all_lengths.append(path['path_length'])
            
            if all_lengths:
                statistics['total_path_length'] = sum(all_lengths)
                statistics['longest_path'] = max(all_lengths)
                statistics['shortest_path'] = min(all_lengths)
                
                # Calculate averages by type
                hatch_lengths = [p['path_length'] for p in hatch_paths if p['path_length'] > 0]
                contour_lengths = [p['path_length'] for p in contour_paths if p['path_length'] > 0]
                point_lengths = [p['path_length'] for p in point_paths if p['path_length'] > 0]
                
                if hatch_lengths:
                    statistics['average_hatch_length'] = sum(hatch_lengths) / len(hatch_lengths)
                if contour_lengths:
                    statistics['average_contour_length'] = sum(contour_lengths) / len(contour_lengths)
                if point_lengths:
                    statistics['average_point_length'] = sum(point_lengths) / len(point_lengths)
        
        except Exception as e:
            logger.warning(f"Error calculating path statistics: {e}")
            statistics['error'] = str(e)
        
        return statistics
    
    def _analyze_path_geometry(self, build_data: Any) -> Dict[str, Any]:
        """Analyze path geometry patterns."""
        geometry_analysis = {
            'layer_distribution': {},
            'geometry_type_distribution': {},
            'path_density': {},
            'note': 'EOS/SLI files contain only geometry coordinates, no process parameters'
        }
        
        try:
            reader = build_data.get('reader_object')
            if not reader:
                return geometry_analysis
            
            if hasattr(reader, 'layers'):
                layers = reader.layers
                
                # Analyze layer distribution
                for layer_idx, layer in enumerate(layers):
                    try:
                        hatch_count = len(layer.getHatchGeometry())
                        contour_count = len(layer.getContourGeometry())
                        point_count = len(layer.getPointsGeometry())
                        
                        geometry_analysis['layer_distribution'][layer_idx] = {
                            'hatch_count': hatch_count,
                            'contour_count': contour_count,
                            'point_count': point_count,
                            'total_geometry': hatch_count + contour_count + point_count
                        }
                    except Exception as layer_error:
                        logger.warning(f"Error analyzing layer {layer_idx}: {layer_error}")
                        continue
                
                # Calculate overall distribution
                total_hatches = sum(layer_data['hatch_count'] for layer_data in geometry_analysis['layer_distribution'].values())
                total_contours = sum(layer_data['contour_count'] for layer_data in geometry_analysis['layer_distribution'].values())
                total_points = sum(layer_data['point_count'] for layer_data in geometry_analysis['layer_distribution'].values())
                
                geometry_analysis['geometry_type_distribution'] = {
                    'hatch': total_hatches,
                    'contour': total_contours,
                    'point': total_points,
                    'total': total_hatches + total_contours + total_points
                }
        
        except Exception as e:
            logger.warning(f"Error analyzing path geometry: {e}")
            geometry_analysis['error'] = str(e)
        
        return geometry_analysis
    
    def _create_spatial_path_map(self, build_data: Any) -> Dict[str, Any]:
        """Create spatial mapping for path data."""
        spatial_map = {
            'voxel_resolution': None,
            'path_voxels': [],
            'coordinate_system': 'build_coordinates',
            'bounds': None,
            'note': 'EOS/SLI files contain only geometry coordinates, no process parameters'
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
            
            # Collect all path points for spatial mapping
            path_points = []
            
            # From hatches
            hatch_paths = self._extract_hatch_paths(build_data)
            for hatch in hatch_paths:
                if hatch['coordinates']:
                    coords = hatch['coordinates']
                    if coords and len(coords) > 0:
                        for i in range(0, len(coords), 2):
                            if i + 1 < len(coords):
                                # Handle coordinate data properly
                                x_val = coords[i] if isinstance(coords[i], (int, float)) else coords[i][0] if isinstance(coords[i], list) and len(coords[i]) > 0 else 0.0
                                y_val = coords[i + 1] if isinstance(coords[i + 1], (int, float)) else coords[i + 1][0] if isinstance(coords[i + 1], list) and len(coords[i + 1]) > 0 else 0.0
                                
                                path_points.append({
                                    'x': float(x_val),
                                    'y': float(y_val),
                                    'z': hatch['layer_index'] * 0.05,  # Assuming 50μm layer thickness
                                    'geometry_type': 'hatch',
                                    'build_style_id': hatch['build_style_id'],
                                    'path_length': hatch['path_length']
                                })
            
            # From contours
            contour_paths = self._extract_contour_paths(build_data)
            for contour in contour_paths:
                if contour['coordinates']:
                    coords = contour['coordinates']
                    if coords and len(coords) > 0:
                        for i in range(0, len(coords), 2):
                            if i + 1 < len(coords):
                                # Handle coordinate data properly
                                x_val = coords[i] if isinstance(coords[i], (int, float)) else coords[i][0] if isinstance(coords[i], list) and len(coords[i]) > 0 else 0.0
                                y_val = coords[i + 1] if isinstance(coords[i + 1], (int, float)) else coords[i + 1][0] if isinstance(coords[i + 1], list) and len(coords[i + 1]) > 0 else 0.0
                                
                                path_points.append({
                                    'x': float(x_val),
                                    'y': float(y_val),
                                    'z': contour['layer_index'] * 0.05,
                                    'geometry_type': 'contour',
                                    'build_style_id': contour['build_style_id'],
                                    'path_length': contour['path_length']
                                })
            
            # From points
            point_paths = self._extract_point_paths(build_data)
            for point in point_paths:
                if point['coordinates']:
                    coords = point['coordinates']
                    if coords and len(coords) > 0:
                        for i in range(0, len(coords), 2):
                            if i + 1 < len(coords):
                                # Handle coordinate data properly
                                x_val = coords[i] if isinstance(coords[i], (int, float)) else coords[i][0] if isinstance(coords[i], list) and len(coords[i]) > 0 else 0.0
                                y_val = coords[i + 1] if isinstance(coords[i + 1], (int, float)) else coords[i + 1][0] if isinstance(coords[i + 1], list) and len(coords[i + 1]) > 0 else 0.0
                                
                                path_points.append({
                                    'x': float(x_val),
                                    'y': float(y_val),
                                    'z': point['layer_index'] * 0.05,
                                    'geometry_type': 'point',
                                    'build_style_id': point['build_style_id'],
                                    'path_length': point['path_length']
                                })
            
            spatial_map['path_voxels'] = path_points
            spatial_map['total_points'] = len(path_points)
            
            # Calculate suggested voxel resolution based on path density
            if path_points:
                x_coords = [float(p['x']) for p in path_points if p['x'] is not None]
                y_coords = [float(p['y']) for p in path_points if p['y'] is not None]
                z_coords = [float(p['z']) for p in path_points if p['z'] is not None]
                
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
            logger.warning(f"Error creating spatial path map: {e}")
            spatial_map['error'] = str(e)
        
        return spatial_map
    
    def get_extractor_info(self) -> Dict[str, Any]:
        """Get extractor information."""
        return {
            'name': 'EOS/SLI Path Extractor',
            'libslm_available': self.libslm_available,
            'description': 'Extracts geometry path data from EOS/SLI build files (no process parameters available)',
            'note': 'EOS/SLI files contain only geometry coordinates, not power/velocity/process parameters'
        }