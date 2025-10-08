"""
Power Extractor for PBF-LB/M Build Files.

This module provides power parameter extraction capabilities for PBF-LB/M build files,
leveraging PySLM for advanced analysis when available.
"""

from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path
import logging
import numpy as np

from ....external import LIBSLM_AVAILABLE

logger = logging.getLogger(__name__)


class PowerExtractor:
    """
    Extractor for laser power parameters from PBF-LB/M build files.
    
    This extractor analyzes power settings across different scan strategies
    and provides statistical analysis of power variations.
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
        Extract power data from build file data.
        
        Args:
            build_data: Parsed build file data
            
        Returns:
            Dictionary containing power analysis results
        """
        try:
            logger.info("Extracting power data from build file")
            
            power_data = {
                'global_power': self._extract_global_power(build_data),
                'layer_power': self._extract_layer_power(build_data),
                'geometry_power': self._extract_geometry_power(build_data),
                'hatch_power': self._extract_hatch_power(build_data),  # NEW: Per-hatch power
                'contour_power': self._extract_contour_power(build_data),  # NEW: Per-contour power
                'point_power': self._extract_point_power(build_data),  # NEW: Per-point power
                'power_statistics': self._calculate_power_statistics(build_data),
                'power_distribution': self._analyze_power_distribution(build_data),
                'spatial_power_map': self._create_spatial_power_map(build_data)  # NEW: 3D spatial mapping
            }
            
            logger.info("Successfully extracted power data")
            return power_data
            
        except Exception as e:
            logger.error(f"Error extracting power data: {e}")
            raise
    
    def _extract_global_power(self, build_data: Any) -> Dict[str, Any]:
        """Extract global power settings."""
        global_power = {}
        
        try:
            # Extract from build parameters
            if hasattr(build_data, 'parameters'):
                params = build_data.parameters
                global_power.update({
                    'default_power': getattr(params, 'default_power', None),
                    'max_power': getattr(params, 'max_power', None),
                    'min_power': getattr(params, 'min_power', None),
                    'power_control_mode': getattr(params, 'power_control_mode', None)
                })
            
            # Extract from machine settings
            if hasattr(build_data, 'machine_settings'):
                machine = build_data.machine_settings
                global_power.update({
                    'laser_power_capacity': getattr(machine, 'laser_power_capacity', None),
                    'power_calibration': getattr(machine, 'power_calibration', None)
                })
        
        except Exception as e:
            logger.warning(f"Error extracting global power: {e}")
        
        return global_power
    
    def _extract_layer_power(self, build_data: Any) -> List[Dict[str, Any]]:
        """Extract layer-specific power settings."""
        layer_power = []
        
        try:
            if hasattr(build_data, 'layers'):
                for i, layer in enumerate(build_data.layers):
                    layer_info = {
                        'layer_index': i,
                        'z_height': getattr(layer, 'z_height', None),
                        'power_settings': {}
                    }
                    
                    # Extract layer power parameters
                    if hasattr(layer, 'parameters'):
                        layer_params = layer.parameters
                        layer_info['power_settings'] = {
                            'hatch_power': getattr(layer_params, 'hatch_power', None),
                            'contour_power': getattr(layer_params, 'contour_power', None),
                            'support_power': getattr(layer_params, 'support_power', None),
                            'power_ramp_start': getattr(layer_params, 'power_ramp_start', None),
                            'power_ramp_end': getattr(layer_params, 'power_ramp_end', None)
                        }
                    
                    layer_power.append(layer_info)
        
        except Exception as e:
            logger.warning(f"Error extracting layer power: {e}")
        
        return layer_power
    
    def _extract_geometry_power(self, build_data: Any) -> List[Dict[str, Any]]:
        """Extract geometry-specific power settings."""
        geometry_power = []
        
        try:
            if hasattr(build_data, 'geometries'):
                for i, geometry in enumerate(build_data.geometries):
                    geom_info = {
                        'geometry_index': i,
                        'geometry_type': getattr(geometry, 'type', 'unknown'),
                        'power_settings': {}
                    }
                    
                    # Extract geometry power parameters
                    if hasattr(geometry, 'parameters'):
                        geom_params = geometry.parameters
                        geom_info['power_settings'] = {
                            'power': getattr(geom_params, 'power', None),
                            'power_density': getattr(geom_params, 'power_density', None),
                            'energy_density': getattr(geom_params, 'energy_density', None),
                            'power_modulation': getattr(geom_params, 'power_modulation', None)
                        }
                    
                    geometry_power.append(geom_info)
        
        except Exception as e:
            logger.warning(f"Error extracting geometry power: {e}")
        
        return geometry_power
    
    def _calculate_power_statistics(self, build_data: Any) -> Dict[str, Any]:
        """Calculate power statistics across the build."""
        statistics = {}
        
        try:
            power_values = []
            
            # Collect all power values
            if hasattr(build_data, 'layers'):
                for layer in build_data.layers:
                    if hasattr(layer, 'parameters'):
                        layer_params = layer.parameters
                        if hasattr(layer_params, 'hatch_power') and layer_params.hatch_power is not None:
                            power_values.append(layer_params.hatch_power)
                        if hasattr(layer_params, 'contour_power') and layer_params.contour_power is not None:
                            power_values.append(layer_params.contour_power)
            
            if hasattr(build_data, 'geometries'):
                for geometry in build_data.geometries:
                    if hasattr(geometry, 'parameters'):
                        geom_params = geometry.parameters
                        if hasattr(geom_params, 'power') and geom_params.power is not None:
                            power_values.append(geom_params.power)
            
            # Calculate statistics
            if power_values:
                power_array = np.array(power_values)
                statistics = {
                    'count': len(power_values),
                    'mean': float(np.mean(power_array)),
                    'std': float(np.std(power_array)),
                    'min': float(np.min(power_array)),
                    'max': float(np.max(power_array)),
                    'median': float(np.median(power_array)),
                    'q25': float(np.percentile(power_array, 25)),
                    'q75': float(np.percentile(power_array, 75))
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
            logger.warning(f"Error calculating power statistics: {e}")
            statistics = {'error': str(e)}
        
        return statistics
    
    def _analyze_power_distribution(self, build_data: Any) -> Dict[str, Any]:
        """Analyze power distribution patterns."""
        distribution = {}
        
        try:
            # Analyze power variation across layers
            layer_powers = []
            if hasattr(build_data, 'layers'):
                for layer in build_data.layers:
                    if hasattr(layer, 'parameters'):
                        layer_params = layer.parameters
                        if hasattr(layer_params, 'hatch_power') and layer_params.hatch_power is not None:
                            layer_powers.append(layer_params.hatch_power)
            
            if layer_powers:
                layer_power_array = np.array(layer_powers)
                distribution['layer_variation'] = {
                    'coefficient_of_variation': float(np.std(layer_power_array) / np.mean(layer_power_array)) if np.mean(layer_power_array) > 0 else 0,
                    'trend': self._detect_power_trend(layer_power_array),
                    'outliers': self._detect_power_outliers(layer_power_array)
                }
            
            # Analyze power distribution by geometry type
            geometry_powers = {}
            if hasattr(build_data, 'geometries'):
                for geometry in build_data.geometries:
                    geom_type = getattr(geometry, 'type', 'unknown')
                    if hasattr(geometry, 'parameters'):
                        geom_params = geometry.parameters
                        if hasattr(geom_params, 'power') and geom_params.power is not None:
                            if geom_type not in geometry_powers:
                                geometry_powers[geom_type] = []
                            geometry_powers[geom_type].append(geom_params.power)
            
            distribution['geometry_distribution'] = {}
            for geom_type, powers in geometry_powers.items():
                if powers:
                    power_array = np.array(powers)
                    distribution['geometry_distribution'][geom_type] = {
                        'count': len(powers),
                        'mean': float(np.mean(power_array)),
                        'std': float(np.std(power_array))
                    }
        
        except Exception as e:
            logger.warning(f"Error analyzing power distribution: {e}")
            distribution = {'error': str(e)}
        
        return distribution
    
    def _detect_power_trend(self, power_values: np.ndarray) -> str:
        """Detect trend in power values across layers."""
        try:
            if len(power_values) < 2:
                return 'insufficient_data'
            
            # Simple linear trend detection
            x = np.arange(len(power_values))
            slope = np.polyfit(x, power_values, 1)[0]
            
            if abs(slope) < 0.01:  # Threshold for "no trend"
                return 'stable'
            elif slope > 0:
                return 'increasing'
            else:
                return 'decreasing'
        
        except Exception:
            return 'unknown'
    
    def _detect_power_outliers(self, power_values: np.ndarray) -> List[int]:
        """Detect power outliers using IQR method."""
        try:
            if len(power_values) < 4:
                return []
            
            q1 = np.percentile(power_values, 25)
            q3 = np.percentile(power_values, 75)
            iqr = q3 - q1
            
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outliers = []
            for i, value in enumerate(power_values):
                if value < lower_bound or value > upper_bound:
                    outliers.append(i)
            
            return outliers
        
        except Exception:
            return []
    
    def _extract_hatch_power(self, build_data: Any) -> List[Dict[str, Any]]:
        """Extract hatch-specific power settings using libSLM data structures."""
        hatch_power = []
        
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
                    # Get hatch geometry from layer
                    hatch_geometries = layer.getHatchGeometry() if hasattr(layer, 'getHatchGeometry') else []
                    
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
                            'power': build_style.laserPower if build_style else None,
                            'velocity': build_style.laserSpeed if build_style else None,
                            'exposure_time': build_style.pointExposureTime if build_style else None,
                            'point_distance': build_style.pointDistance if build_style else None,
                            'laser_focus': build_style.laserFocus if build_style else None,
                            'point_delay': build_style.pointDelay if build_style else None,
                            'jump_speed': build_style.jumpSpeed if build_style else None,
                            'jump_delay': build_style.jumpDelay if build_style else None
                        }
                        hatch_power.append(hatch_info)
        
        except Exception as e:
            logger.warning(f"Error extracting hatch power: {e}")
        
        return hatch_power
    
    def _extract_contour_power(self, build_data: Any) -> List[Dict[str, Any]]:
        """Extract contour-specific power settings using libSLM data structures."""
        contour_power = []
        
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
                            'power': build_style.laserPower if build_style else None,
                            'velocity': build_style.laserSpeed if build_style else None,
                            'exposure_time': build_style.pointExposureTime if build_style else None,
                            'point_distance': build_style.pointDistance if build_style else None,
                            'laser_focus': build_style.laserFocus if build_style else None,
                            'point_delay': build_style.pointDelay if build_style else None,
                            'jump_speed': build_style.jumpSpeed if build_style else None,
                            'jump_delay': build_style.jumpDelay if build_style else None
                        }
                        contour_power.append(contour_info)
        
        except Exception as e:
            logger.warning(f"Error extracting contour power: {e}")
        
        return contour_power
    
    def _extract_point_power(self, build_data: Any) -> List[Dict[str, Any]]:
        """Extract point-specific power settings using libSLM data structures."""
        point_power = []
        
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
                        point_power.append(point_info)
        
        except Exception as e:
            logger.warning(f"Error extracting point power: {e}")
        
        return point_power
    
    def _create_spatial_power_map(self, build_data: Any) -> Dict[str, Any]:
        """Create 3D spatial power mapping for voxelization."""
        spatial_map = {
            'voxel_resolution': None,
            'power_voxels': [],
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
            
            # Collect all power points for voxelization
            power_points = []
            
            # From hatches
            hatch_power = self._extract_hatch_power(build_data)
            for hatch in hatch_power:
                if hatch['coordinates'] and hatch['power']:
                    coords = hatch['coordinates']
                    # Process each coordinate pair in the hatch
                    for i in range(0, len(coords), 2):
                        if i + 1 < len(coords):
                            power_points.append({
                                'x': coords[i],
                                'y': coords[i + 1],
                                'z': hatch['layer_index'] * 0.05,  # Assuming 50μm layer thickness
                                'power': hatch['power'],
                                'velocity': hatch['velocity'],
                                'geometry_type': 'hatch'
                            })
            
            # From contours
            contour_power = self._extract_contour_power(build_data)
            for contour in contour_power:
                if contour['coordinates'] and contour['power']:
                    coords = contour['coordinates']
                    # Process each coordinate pair in the contour
                    for i in range(0, len(coords), 2):
                        if i + 1 < len(coords):
                            power_points.append({
                                'x': coords[i],
                                'y': coords[i + 1],
                                'z': contour['layer_index'] * 0.05,
                                'power': contour['power'],
                                'velocity': contour['velocity'],
                                'geometry_type': 'contour'
                            })
            
            # From points
            point_power = self._extract_point_power(build_data)
            for point in point_power:
                if point['coordinates'] and point['power']:
                    coords = point['coordinates']
                    # Process each coordinate pair in the point geometry
                    for i in range(0, len(coords), 2):
                        if i + 1 < len(coords):
                            power_points.append({
                                'x': coords[i],
                                'y': coords[i + 1],
                                'z': point['layer_index'] * 0.05,
                                'power': point['power'],
                                'velocity': point['velocity'],
                                'geometry_type': 'point'
                            })
            
            spatial_map['power_voxels'] = power_points
            spatial_map['total_points'] = len(power_points)
            
            # Calculate suggested voxel resolution
            if power_points and spatial_map['bounds']:
                bounds = spatial_map['bounds']
                x_range = bounds['x_max'] - bounds['x_min']
                y_range = bounds['y_max'] - bounds['y_min']
                z_range = bounds['z_max'] - bounds['z_min']
                
                # Suggest resolution based on data density
                suggested_resolution = min(x_range, y_range, z_range) / 100  # 100 voxels per dimension
                spatial_map['voxel_resolution'] = max(0.01, suggested_resolution)  # Minimum 10μm
        
        except Exception as e:
            logger.warning(f"Error creating spatial power map: {e}")
            spatial_map['error'] = str(e)
        
        return spatial_map
    
    def get_extractor_info(self) -> Dict[str, Any]:
        """Get extractor information."""
        return {
            'name': 'Power Extractor',
            'libslm_available': self.libslm_available,
            'description': 'Extracts and analyzes laser power parameters from PBF-LB/M build files using libSLM'
        }
