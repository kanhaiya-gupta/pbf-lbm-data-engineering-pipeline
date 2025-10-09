"""
Velocity Extractor for PBF-LB/M Build Files.

This module provides velocity parameter extraction capabilities for PBF-LB/M build files,
leveraging libSLM for accessing individual scan path velocities.
"""

from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path
import logging
import numpy as np

from .....external import LIBSLM_AVAILABLE

logger = logging.getLogger(__name__)


class VelocityExtractor:
    """
    Extractor for laser velocity parameters from PBF-LB/M build files.
    
    This extractor analyzes velocity settings across different scan strategies
    and provides statistical analysis of velocity variations using libSLM data.
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
        Extract velocity data from build file data.
        
        Args:
            build_data: Parsed build file data from libSLM
            
        Returns:
            Dictionary containing velocity analysis results
        """
        try:
            logger.info("Extracting velocity data from build file")
            
            velocity_data = {
                'global_velocity': self._extract_global_velocity(build_data),
                'layer_velocity': self._extract_layer_velocity(build_data),
                'hatch_velocity': self._extract_hatch_velocity(build_data),
                'contour_velocity': self._extract_contour_velocity(build_data),
                'velocity_statistics': self._calculate_velocity_statistics(build_data),
                'velocity_distribution': self._analyze_velocity_distribution(build_data),
                'spatial_velocity_map': self._create_spatial_velocity_map(build_data)
            }
            
            logger.info("Successfully extracted velocity data")
            return velocity_data
            
        except Exception as e:
            logger.error(f"Error extracting velocity data: {e}")
            raise
    
    def _extract_global_velocity(self, build_data: Any) -> Dict[str, Any]:
        """Extract global velocity settings."""
        global_velocity = {}
        
        try:
            if hasattr(build_data, 'parameters'):
                params = build_data.parameters
                global_velocity.update({
                    'default_velocity': getattr(params, 'default_velocity', None),
                    'max_velocity': getattr(params, 'max_velocity', None),
                    'min_velocity': getattr(params, 'min_velocity', None),
                    'velocity_control_mode': getattr(params, 'velocity_control_mode', None)
                })
        
        except Exception as e:
            logger.warning(f"Error extracting global velocity: {e}")
        
        return global_velocity
    
    def _extract_layer_velocity(self, build_data: Any) -> List[Dict[str, Any]]:
        """Extract layer-specific velocity settings."""
        layer_velocity = []
        
        try:
            if hasattr(build_data, 'layers'):
                for i, layer in enumerate(build_data.layers):
                    layer_info = {
                        'layer_index': i,
                        'z_height': getattr(layer, 'z_height', None),
                        'velocity_settings': {}
                    }
                    
                    if hasattr(layer, 'parameters'):
                        layer_params = layer.parameters
                        layer_info['velocity_settings'] = {
                            'hatch_velocity': getattr(layer_params, 'hatch_velocity', None),
                            'contour_velocity': getattr(layer_params, 'contour_velocity', None),
                            'support_velocity': getattr(layer_params, 'support_velocity', None)
                        }
                    
                    layer_velocity.append(layer_info)
        
        except Exception as e:
            logger.warning(f"Error extracting layer velocity: {e}")
        
        return layer_velocity
    
    def _extract_hatch_velocity(self, build_data: Any) -> List[Dict[str, Any]]:
        """Extract hatch-specific velocity settings using libSLM data structures."""
        hatch_velocity = []
        
        try:
            # Get the reader object from build_data
            reader = build_data.get('reader_object')
            if not reader:
                logger.warning("No reader object available for hatch velocity extraction")
                return hatch_velocity
            
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
                        
                        hatch_info = {
                            'layer_index': layer_idx,
                            'hatch_index': hatch_idx,
                            'build_style_id': hatch_geom.bid,
                            'model_id': hatch_geom.mid,
                            'coordinates': coords.tolist() if coords is not None else None,
                            'num_segments': coords.shape[0] // 2 if coords is not None else 0,
                            # Process parameters from BuildStyle
                            'velocity': build_style.laserSpeed if build_style else None,
                            'power': build_style.laserPower if build_style else None,
                            'exposure_time': build_style.pointExposureTime if build_style else None,
                            'point_distance': build_style.pointDistance if build_style else None,
                            'laser_focus': build_style.laserFocus if build_style else None,
                            'point_delay': build_style.pointDelay if build_style else None,
                            'jump_speed': build_style.jumpSpeed if build_style else None,
                            'jump_delay': build_style.jumpDelay if build_style else None
                        }
                        hatch_velocity.append(hatch_info)
        
        except Exception as e:
            logger.warning(f"Error extracting hatch velocity: {e}")
        
        return hatch_velocity
    
    def _extract_contour_velocity(self, build_data: Any) -> List[Dict[str, Any]]:
        """Extract contour-specific velocity settings using libSLM data structures."""
        contour_velocity = []
        
        try:
            # Get build styles for parameter lookup
            build_styles = {}
            if hasattr(build_data, 'models'):
                for model in build_data.models:
                    if hasattr(model, 'buildStyles'):
                        for build_style in model.buildStyles:
                            build_styles[build_style.id] = build_style
            
            # Use libSLM's native layer and geometry access
            if hasattr(build_data, 'layers'):
                for layer_idx, layer in enumerate(build_data.layers):
                    # Get contour geometry from layer
                    contour_geometries = layer.getContourGeometry() if hasattr(layer, 'getContourGeometry') else []
                    
                    for contour_idx, contour_geom in enumerate(contour_geometries):
                        # Get build style parameters using bid
                        build_style = build_styles.get(contour_geom.bid)
                        
                        # Extract coordinates from libSLM coords matrix
                        coords = contour_geom.coords if hasattr(contour_geom, 'coords') else None
                        
                        contour_info = {
                            'layer_index': layer_idx,
                            'contour_index': contour_idx,
                            'build_style_id': contour_geom.bid,
                            'model_id': contour_geom.mid,
                            'coordinates': coords.tolist() if coords is not None else None,
                            'num_points': coords.shape[0] if coords is not None else 0,
                            # Process parameters from BuildStyle
                            'velocity': build_style.laserSpeed if build_style else None,
                            'power': build_style.laserPower if build_style else None,
                            'exposure_time': build_style.pointExposureTime if build_style else None,
                            'point_distance': build_style.pointDistance if build_style else None,
                            'laser_focus': build_style.laserFocus if build_style else None,
                            'point_delay': build_style.pointDelay if build_style else None,
                            'jump_speed': build_style.jumpSpeed if build_style else None,
                            'jump_delay': build_style.jumpDelay if build_style else None
                        }
                        contour_velocity.append(contour_info)
        
        except Exception as e:
            logger.warning(f"Error extracting contour velocity: {e}")
        
        return contour_velocity
    
    def _calculate_velocity_statistics(self, build_data: Any) -> Dict[str, Any]:
        """Calculate velocity statistics across the build."""
        statistics = {}
        
        try:
            velocity_values = []
            
            # Collect all velocity values from hatches
            hatch_velocity = self._extract_hatch_velocity(build_data)
            for hatch in hatch_velocity:
                if hatch['velocity'] is not None:
                    velocity_values.append(hatch['velocity'])
            
            # Collect all velocity values from contours
            contour_velocity = self._extract_contour_velocity(build_data)
            for contour in contour_velocity:
                if contour['velocity'] is not None:
                    velocity_values.append(contour['velocity'])
            
            # Calculate statistics
            if velocity_values:
                velocity_array = np.array(velocity_values)
                statistics = {
                    'count': len(velocity_values),
                    'mean': float(np.mean(velocity_array)),
                    'std': float(np.std(velocity_array)),
                    'min': float(np.min(velocity_array)),
                    'max': float(np.max(velocity_array)),
                    'median': float(np.median(velocity_array)),
                    'q25': float(np.percentile(velocity_array, 25)),
                    'q75': float(np.percentile(velocity_array, 75))
                }
            else:
                statistics = {
                    'count': 0,
                    'mean': None,
                    'std': None,
                    'min': None,
                    'max': None,
                    'median': None,
                    'q25': None,
                    'q75': None
                }
        
        except Exception as e:
            logger.warning(f"Error calculating velocity statistics: {e}")
            statistics = {'error': str(e)}
        
        return statistics
    
    def _analyze_velocity_distribution(self, build_data: Any) -> Dict[str, Any]:
        """Analyze velocity distribution patterns."""
        distribution = {}
        
        try:
            # Analyze velocity variation across layers
            layer_velocities = []
            if hasattr(build_data, 'layers'):
                for layer in build_data.layers:
                    if hasattr(layer, 'parameters'):
                        layer_params = layer.parameters
                        if hasattr(layer_params, 'hatch_velocity') and layer_params.hatch_velocity is not None:
                            layer_velocities.append(layer_params.hatch_velocity)
            
            if layer_velocities:
                layer_velocity_array = np.array(layer_velocities)
                distribution['layer_variation'] = {
                    'coefficient_of_variation': float(np.std(layer_velocity_array) / np.mean(layer_velocity_array)) if np.mean(layer_velocity_array) > 0 else 0,
                    'trend': self._detect_velocity_trend(layer_velocity_array),
                    'outliers': self._detect_velocity_outliers(layer_velocity_array)
                }
        
        except Exception as e:
            logger.warning(f"Error analyzing velocity distribution: {e}")
            distribution = {'error': str(e)}
        
        return distribution
    
    def _create_spatial_velocity_map(self, build_data: Any) -> Dict[str, Any]:
        """Create 3D spatial velocity mapping for voxelization."""
        spatial_map = {
            'voxel_resolution': None,
            'velocity_voxels': [],
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
            
            # Collect all velocity points for voxelization
            velocity_points = []
            
            # From hatches
            hatch_velocity = self._extract_hatch_velocity(build_data)
            for hatch in hatch_velocity:
                if hatch['start_point'] and hatch['velocity']:
                    velocity_points.append({
                        'x': hatch['start_point'][0] if len(hatch['start_point']) > 0 else 0,
                        'y': hatch['start_point'][1] if len(hatch['start_point']) > 1 else 0,
                        'z': hatch['layer_index'] * 0.05,  # Assuming 50μm layer thickness
                        'velocity': hatch['velocity']
                    })
                if hatch['end_point'] and hatch['velocity']:
                    velocity_points.append({
                        'x': hatch['end_point'][0] if len(hatch['end_point']) > 0 else 0,
                        'y': hatch['end_point'][1] if len(hatch['end_point']) > 1 else 0,
                        'z': hatch['layer_index'] * 0.05,
                        'velocity': hatch['velocity']
                    })
            
            # From contours
            contour_velocity = self._extract_contour_velocity(build_data)
            for contour in contour_velocity:
                if contour['points'] and contour['velocity']:
                    for point in contour['points']:
                        velocity_points.append({
                            'x': point[0] if len(point) > 0 else 0,
                            'y': point[1] if len(point) > 1 else 0,
                            'z': contour['layer_index'] * 0.05,
                            'velocity': contour['velocity']
                        })
            
            spatial_map['velocity_voxels'] = velocity_points
            spatial_map['total_points'] = len(velocity_points)
            
            # Calculate suggested voxel resolution
            if velocity_points and spatial_map['bounds']:
                bounds = spatial_map['bounds']
                x_range = bounds['x_max'] - bounds['x_min']
                y_range = bounds['y_max'] - bounds['y_min']
                z_range = bounds['z_max'] - bounds['z_min']
                
                suggested_resolution = min(x_range, y_range, z_range) / 100
                spatial_map['voxel_resolution'] = max(0.01, suggested_resolution)
        
        except Exception as e:
            logger.warning(f"Error creating spatial velocity map: {e}")
            spatial_map['error'] = str(e)
        
        return spatial_map
    
    def _detect_velocity_trend(self, velocity_values: np.ndarray) -> str:
        """Detect trend in velocity values across layers."""
        try:
            if len(velocity_values) < 2:
                return 'insufficient_data'
            
            x = np.arange(len(velocity_values))
            slope = np.polyfit(x, velocity_values, 1)[0]
            
            if abs(slope) < 0.01:
                return 'stable'
            elif slope > 0:
                return 'increasing'
            else:
                return 'decreasing'
        
        except Exception:
            return 'unknown'
    
    def _detect_velocity_outliers(self, velocity_values: np.ndarray) -> List[int]:
        """Detect velocity outliers using IQR method."""
        try:
            if len(velocity_values) < 4:
                return []
            
            q1 = np.percentile(velocity_values, 25)
            q3 = np.percentile(velocity_values, 75)
            iqr = q3 - q1
            
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outliers = []
            for i, value in enumerate(velocity_values):
                if value < lower_bound or value > upper_bound:
                    outliers.append(i)
            
            return outliers
        
        except Exception:
            return []
    
    def get_extractor_info(self) -> Dict[str, Any]:
        """Get extractor information."""
        return {
            'name': 'Velocity Extractor',
            'libslm_available': self.libslm_available,
            'description': 'Extracts and analyzes laser velocity parameters from PBF-LB/M build files using libSLM'
        }
