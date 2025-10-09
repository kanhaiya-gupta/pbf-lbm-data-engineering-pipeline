"""
Path Extractor for PBF-LB/M Build Files.

This module provides scan path extraction capabilities for PBF-LB/M build files,
leveraging libSLM for accessing individual scan paths and their geometries.
"""

from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path
import logging
import numpy as np

from .....external import LIBSLM_AVAILABLE

logger = logging.getLogger(__name__)


class PathExtractor:
    """
    Extractor for scan path data from PBF-LB/M build files.
    
    This extractor analyzes scan paths, their geometries, and provides
    spatial analysis using libSLM data structures.
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
        Extract scan path data from build file data.
        
        Args:
            build_data: Parsed build file data from libSLM
            
        Returns:
            Dictionary containing path analysis results
        """
        try:
            logger.info("Extracting path data from build file")
            
            path_data = {
                'hatch_paths': self._extract_hatch_paths(build_data),
                'contour_paths': self._extract_contour_paths(build_data),
                'point_paths': self._extract_point_paths(build_data),
                'path_statistics': self._calculate_path_statistics(build_data),
                'path_geometry': self._analyze_path_geometry(build_data),
                'spatial_path_map': self._create_spatial_path_map(build_data)
            }
            
            logger.info("Successfully extracted path data")
            return path_data
            
        except Exception as e:
            logger.error(f"Error extracting path data: {e}")
            raise
    
    def _extract_hatch_paths(self, build_data: Any) -> List[Dict[str, Any]]:
        """Extract hatch path data using libSLM data structures."""
        hatch_paths = []
        
        try:
            # Get the reader object from build_data
            reader = build_data.get('reader_object')
            if not reader:
                logger.warning("No reader object available for hatch path extraction")
                return hatch_paths
            
            # Get build styles for parameter lookup
            build_styles = {}
            if hasattr(reader, 'models'):
                models = reader.models
                for model in models:
                    if hasattr(model, 'buildStyles'):
                        # Access build styles vector
                        build_styles_vector = model.buildStyles
                        for i in range(len(build_styles_vector)):
                            build_style = build_styles_vector[i]
                            # Use the actual bid from BuildStyle
                            build_styles[build_style.bid] = build_style
            
            # Use direct layer access (this works even with "Skipping Layer Section")
            if hasattr(reader, 'layers'):
                layers = reader.layers
                for layer_idx, layer in enumerate(layers):
                    # Get hatch geometry from layer - this forces loading
                    try:
                        hatch_geometries = layer.getHatchGeometry()
                        logger.debug(f"Layer {layer_idx}: Found {len(hatch_geometries)} hatch geometries")
                    except Exception as layer_error:
                        logger.warning(f"Error accessing layer {layer_idx}: {layer_error}")
                        continue
                    
                    for hatch_idx, hatch_geom in enumerate(hatch_geometries):
                        # Get build style parameters using bid
                        build_style = build_styles.get(hatch_geom.bid)
                        
                        # Extract coordinates from libSLM coords matrix
                        coords = hatch_geom.coords if hasattr(hatch_geom, 'coords') else None
                        
                        # Calculate path properties from coordinates
                        path_length = self._calculate_path_length(coords) if coords is not None else None
                        direction = self._calculate_path_direction(coords) if coords is not None else None
                        angle = self._calculate_path_angle(coords) if coords is not None else None
                        
                        hatch_info = {
                            'layer_index': layer_idx,
                            'hatch_index': hatch_idx,
                            'build_style_id': hatch_geom.bid,
                            'model_id': hatch_geom.mid,
                            'coordinates': coords.tolist() if coords is not None else None,
                            'num_segments': coords.shape[0] // 2 if coords is not None else 0,
                            'path_length': path_length,
                            'direction': direction,
                            'angle': angle,
                            # Process parameters from BuildStyle
                            'power': build_style.laserPower if build_style else None,
                            'velocity': build_style.laserSpeed if build_style else None,
                            'exposure_time': build_style.pointExposureTime if build_style else None,
                            'point_distance': build_style.pointDistance if build_style else None,
                            'laser_focus': build_style.laserFocus if build_style else None,
                            'point_delay': build_style.pointDelay if build_style else None,
                            'jump_speed': build_style.jumpSpeed if build_style else None,
                            'jump_delay': build_style.jumpDelay if build_style else None
                        }
                        hatch_paths.append(hatch_info)
        
        except Exception as e:
            logger.warning(f"Error extracting hatch paths: {e}")
        
        return hatch_paths
    
    def _extract_contour_paths(self, build_data: Any) -> List[Dict[str, Any]]:
        """Extract contour path data using libSLM data structures."""
        contour_paths = []
        
        try:
            # Get the reader object from build_data
            reader = build_data.get('reader_object')
            if not reader:
                logger.warning("No reader object available for contour path extraction")
                return contour_paths
            
            # Get build styles for parameter lookup
            build_styles = {}
            if hasattr(reader, 'models'):
                models = reader.models
                for model in models:
                    if hasattr(model, 'buildStyles'):
                        # Access build styles vector
                        build_styles_vector = model.buildStyles
                        for i in range(len(build_styles_vector)):
                            build_style = build_styles_vector[i]
                            # Use the actual bid from BuildStyle
                            build_styles[build_style.bid] = build_style
            
            # Use libSLM's native layer and geometry access
            if hasattr(reader, 'getLayers'):
                layers = reader.getLayers()
                for layer_idx, layer in enumerate(layers):
                    # Get contour geometry from layer
                    contour_geometries = layer.getContourGeometry() if hasattr(layer, 'getContourGeometry') else []
                    
                    for contour_idx, contour_geom in enumerate(contour_geometries):
                        # Get build style parameters using bid
                        build_style = build_styles.get(contour_geom.bid)
                        
                        # Extract coordinates from libSLM coords matrix
                        coords = contour_geom.coords if hasattr(contour_geom, 'coords') else None
                        
                        # Calculate contour properties from coordinates
                        is_closed = self._is_contour_closed(coords) if coords is not None else False
                        area = self._calculate_contour_area(coords) if coords is not None else None
                        perimeter = self._calculate_contour_perimeter(coords) if coords is not None else None
                        
                        contour_info = {
                            'layer_index': layer_idx,
                            'contour_index': contour_idx,
                            'build_style_id': contour_geom.bid,
                            'model_id': contour_geom.mid,
                            'coordinates': coords.tolist() if coords is not None else None,
                            'point_count': coords.shape[0] if coords is not None else 0,
                            'is_closed': is_closed,
                            'area': area,
                            'perimeter': perimeter,
                            # Process parameters from BuildStyle
                            'power': build_style.laserPower if build_style else None,
                            'velocity': build_style.laserSpeed if build_style else None,
                            'exposure_time': build_style.pointExposureTime if build_style else None,
                            'point_distance': build_style.pointDistance if build_style else None,
                            'laser_focus': build_style.laserFocus if build_style else None,
                            'point_delay': build_style.pointDelay if build_style else None,
                            'jump_speed': build_style.jumpSpeed if build_style else None,
                            'jump_delay': build_style.jumpDelay if build_style else None
                        }
                        contour_paths.append(contour_info)
        
        except Exception as e:
            logger.warning(f"Error extracting contour paths: {e}")
        
        return contour_paths
    
    def _extract_point_paths(self, build_data: Any) -> List[Dict[str, Any]]:
        """Extract point path data using libSLM data structures."""
        point_paths = []
        
        try:
            # Get the reader object from build_data
            reader = build_data.get('reader_object')
            if not reader:
                logger.warning("No reader object available for point path extraction")
                return point_paths
            
            # Get build styles for parameter lookup
            build_styles = {}
            if hasattr(reader, 'models'):
                models = reader.models
                for model in models:
                    if hasattr(model, 'buildStyles'):
                        # Access build styles vector
                        build_styles_vector = model.buildStyles
                        for i in range(len(build_styles_vector)):
                            build_style = build_styles_vector[i]
                            # Use the actual bid from BuildStyle
                            build_styles[build_style.bid] = build_style
            
            # Use libSLM's native layer and geometry access
            if hasattr(reader, 'getLayers'):
                layers = reader.getLayers()
                for layer_idx, layer in enumerate(layers):
                    # Get point geometry from layer
                    point_geometries = layer.getPointGeometry() if hasattr(layer, 'getPointGeometry') else []
                    
                    for point_idx, point_geom in enumerate(point_geometries):
                        # Get build style parameters using bid
                        build_style = build_styles.get(point_geom.bid)
                        
                        # Extract coordinates from libSLM coords matrix
                        coords = point_geom.coords if hasattr(point_geom, 'coords') else None
                        
                        point_info = {
                            'layer_index': layer_idx,
                            'point_index': point_idx,
                            'build_style_id': point_geom.bid,
                            'model_id': point_geom.mid,
                            'coordinates': coords.tolist() if coords is not None else None,
                            'num_points': coords.shape[0] if coords is not None else 0,
                            # Process parameters from BuildStyle
                            'power': build_style.laserPower if build_style else None,
                            'velocity': build_style.laserSpeed if build_style else None,
                            'exposure_time': build_style.pointExposureTime if build_style else None,
                            'point_distance': build_style.pointDistance if build_style else None,
                            'laser_focus': build_style.laserFocus if build_style else None,
                            'point_delay': build_style.pointDelay if build_style else None,
                            'jump_speed': build_style.jumpSpeed if build_style else None,
                            'jump_delay': build_style.jumpDelay if build_style else None
                        }
                        point_paths.append(point_info)
        
        except Exception as e:
            logger.warning(f"Error extracting point paths: {e}")
        
        return point_paths
    
    def _calculate_path_statistics(self, build_data: Any) -> Dict[str, Any]:
        """Calculate path statistics across the build."""
        statistics = {}
        
        try:
            hatch_paths = self._extract_hatch_paths(build_data)
            contour_paths = self._extract_contour_paths(build_data)
            point_paths = self._extract_point_paths(build_data)
            
            # Hatch statistics
            hatch_lengths = [h['length'] for h in hatch_paths if h['length'] is not None]
            hatch_angles = [h['angle'] for h in hatch_paths if h['angle'] is not None]
            
            # Contour statistics
            contour_lengths = [c['length'] for c in contour_paths if c['length'] is not None]
            contour_areas = [c['area'] for c in contour_paths if c['area'] is not None]
            
            statistics = {
                'total_hatches': len(hatch_paths),
                'total_contours': len(contour_paths),
                'total_points': len(point_paths),
                'hatch_length_stats': self._calculate_stats(hatch_lengths),
                'hatch_angle_stats': self._calculate_stats(hatch_angles),
                'contour_length_stats': self._calculate_stats(contour_lengths),
                'contour_area_stats': self._calculate_stats(contour_areas)
            }
        
        except Exception as e:
            logger.warning(f"Error calculating path statistics: {e}")
            statistics = {'error': str(e)}
        
        return statistics
    
    def _analyze_path_geometry(self, build_data: Any) -> Dict[str, Any]:
        """Analyze path geometry patterns."""
        geometry = {}
        
        try:
            hatch_paths = self._extract_hatch_paths(build_data)
            
            # Analyze hatch patterns
            if hatch_paths:
                angles = [h['angle'] for h in hatch_paths if h['angle'] is not None]
                if angles:
                    angle_array = np.array(angles)
                    geometry['hatch_patterns'] = {
                        'dominant_angles': self._find_dominant_angles(angle_array),
                        'angle_distribution': self._analyze_angle_distribution(angle_array),
                        'scan_strategy': self._identify_scan_strategy(angle_array)
                    }
        
        except Exception as e:
            logger.warning(f"Error analyzing path geometry: {e}")
            geometry = {'error': str(e)}
        
        return geometry
    
    def _create_spatial_path_map(self, build_data: Any) -> Dict[str, Any]:
        """Create 3D spatial path mapping for voxelization."""
        spatial_map = {
            'voxel_resolution': None,
            'path_voxels': [],
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
            
            # Collect all path points for voxelization
            path_points = []
            
            # From hatches
            hatch_paths = self._extract_hatch_paths(build_data)
            for hatch in hatch_paths:
                if hatch['start_point'] and hatch['end_point']:
                    # Interpolate points along hatch line
                    interpolated_points = self._interpolate_hatch_points(
                        hatch['start_point'], hatch['end_point'], hatch['layer_index']
                    )
                    path_points.extend(interpolated_points)
            
            # From contours
            contour_paths = self._extract_contour_paths(build_data)
            for contour in contour_paths:
                if contour['points']:
                    for point in contour['points']:
                        path_points.append({
                            'x': point[0] if len(point) > 0 else 0,
                            'y': point[1] if len(point) > 1 else 0,
                            'z': contour['layer_index'] * 0.05,
                            'path_type': 'contour'
                        })
            
            # From points
            point_paths = self._extract_point_paths(build_data)
            for point in point_paths:
                if point['position']:
                    path_points.append({
                        'x': point['position'][0] if len(point['position']) > 0 else 0,
                        'y': point['position'][1] if len(point['position']) > 1 else 0,
                        'z': point['layer_index'] * 0.05,
                        'path_type': 'point'
                    })
            
            spatial_map['path_voxels'] = path_points
            spatial_map['total_points'] = len(path_points)
            
            # Calculate suggested voxel resolution
            if path_points and spatial_map['bounds']:
                bounds = spatial_map['bounds']
                x_range = bounds['x_max'] - bounds['x_min']
                y_range = bounds['y_max'] - bounds['y_min']
                z_range = bounds['z_max'] - bounds['z_min']
                
                suggested_resolution = min(x_range, y_range, z_range) / 100
                spatial_map['voxel_resolution'] = max(0.01, suggested_resolution)
        
        except Exception as e:
            logger.warning(f"Error creating spatial path map: {e}")
            spatial_map['error'] = str(e)
        
        return spatial_map
    
    def _calculate_direction(self, start_point: Optional[List[float]], end_point: Optional[List[float]]) -> Optional[List[float]]:
        """Calculate direction vector from start to end point."""
        if not start_point or not end_point or len(start_point) < 2 or len(end_point) < 2:
            return None
        
        try:
            dx = end_point[0] - start_point[0]
            dy = end_point[1] - start_point[1]
            length = np.sqrt(dx*dx + dy*dy)
            
            if length > 0:
                return [dx/length, dy/length]
            return None
        
        except Exception:
            return None
    
    def _calculate_angle(self, start_point: Optional[List[float]], end_point: Optional[List[float]]) -> Optional[float]:
        """Calculate angle of hatch line in degrees."""
        direction = self._calculate_direction(start_point, end_point)
        if not direction:
            return None
        
        try:
            angle_rad = np.arctan2(direction[1], direction[0])
            angle_deg = np.degrees(angle_rad)
            return float(angle_deg)
        
        except Exception:
            return None
    
    def _is_contour_closed(self, points: Optional[List[List[float]]]) -> bool:
        """Check if contour is closed (first and last points are the same)."""
        if not points or len(points) < 3:
            return False
        
        try:
            first_point = points[0]
            last_point = points[-1]
            
            if len(first_point) >= 2 and len(last_point) >= 2:
                dx = abs(first_point[0] - last_point[0])
                dy = abs(first_point[1] - last_point[1])
                return dx < 0.001 and dy < 0.001  # Tolerance for floating point
        
        except Exception:
            pass
        
        return False
    
    def _calculate_contour_area(self, points: Optional[List[List[float]]]) -> Optional[float]:
        """Calculate area of contour using shoelace formula."""
        if not points or len(points) < 3:
            return None
        
        try:
            x_coords = [p[0] for p in points if len(p) > 0]
            y_coords = [p[1] for p in points if len(p) > 1]
            
            if len(x_coords) != len(y_coords) or len(x_coords) < 3:
                return None
            
            # Shoelace formula
            area = 0.5 * abs(sum(x_coords[i] * y_coords[(i+1) % len(x_coords)] - 
                               x_coords[(i+1) % len(x_coords)] * y_coords[i] 
                               for i in range(len(x_coords))))
            return float(area)
        
        except Exception:
            return None
    
    def _calculate_contour_perimeter(self, points: Optional[List[List[float]]]) -> Optional[float]:
        """Calculate perimeter of contour."""
        if not points or len(points) < 2:
            return None
        
        try:
            perimeter = 0.0
            for i in range(len(points)):
                current = points[i]
                next_point = points[(i + 1) % len(points)]
                
                if len(current) >= 2 and len(next_point) >= 2:
                    dx = next_point[0] - current[0]
                    dy = next_point[1] - current[1]
                    perimeter += np.sqrt(dx*dx + dy*dy)
            
            return float(perimeter)
        
        except Exception:
            return None
    
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
                'max': float(np.max(value_array))
            }
        
        except Exception:
            return {'count': 0, 'mean': None, 'std': None, 'min': None, 'max': None}
    
    def _find_dominant_angles(self, angles: np.ndarray) -> List[float]:
        """Find dominant hatch angles."""
        try:
            # Bin angles and find peaks
            hist, bin_edges = np.histogram(angles, bins=36, range=(-180, 180))
            peak_indices = np.where(hist > np.mean(hist) + np.std(hist))[0]
            
            dominant_angles = []
            for idx in peak_indices:
                angle = (bin_edges[idx] + bin_edges[idx+1]) / 2
                dominant_angles.append(float(angle))
            
            return dominant_angles
        
        except Exception:
            return []
    
    def _analyze_angle_distribution(self, angles: np.ndarray) -> Dict[str, Any]:
        """Analyze distribution of hatch angles."""
        try:
            return {
                'uniformity': float(1.0 - np.std(angles) / 180.0),  # 1.0 = perfectly uniform
                'dominant_direction': float(np.mean(angles)),
                'angle_variance': float(np.var(angles))
            }
        
        except Exception:
            return {'uniformity': 0, 'dominant_direction': 0, 'angle_variance': 0}
    
    def _identify_scan_strategy(self, angles: np.ndarray) -> str:
        """Identify scan strategy based on angle patterns."""
        try:
            if len(angles) < 2:
                return 'unknown'
            
            # Check for alternating patterns (common in PBF-LB/M)
            unique_angles = np.unique(np.round(angles, 1))
            
            if len(unique_angles) == 1:
                return 'unidirectional'
            elif len(unique_angles) == 2 and abs(unique_angles[0] - unique_angles[1]) > 80:
                return 'bidirectional'
            elif len(unique_angles) > 2:
                return 'multi_directional'
            else:
                return 'mixed'
        
        except Exception:
            return 'unknown'
    
    def _interpolate_hatch_points(self, start_point: List[float], end_point: List[float], layer_index: int) -> List[Dict[str, Any]]:
        """Interpolate points along a hatch line for voxelization."""
        try:
            if len(start_point) < 2 or len(end_point) < 2:
                return []
            
            # Simple linear interpolation
            num_points = max(2, int(np.sqrt((end_point[0] - start_point[0])**2 + (end_point[1] - start_point[1])**2) / 0.1))
            
            points = []
            for i in range(num_points):
                t = i / (num_points - 1) if num_points > 1 else 0
                x = start_point[0] + t * (end_point[0] - start_point[0])
                y = start_point[1] + t * (end_point[1] - start_point[1])
                z = layer_index * 0.05  # Assuming 50μm layer thickness
                
                points.append({
                    'x': float(x),
                    'y': float(y),
                    'z': float(z),
                    'path_type': 'hatch'
                })
            
            return points
        
        except Exception:
            return []
    
    def get_extractor_info(self) -> Dict[str, Any]:
        """Get extractor information."""
        return {
            'name': 'Path Extractor',
            'libslm_available': self.libslm_available,
            'description': 'Extracts and analyzes scan path data from PBF-LB/M build files using libSLM'
        }
