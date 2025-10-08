"""
Energy Extractor for PBF-LB/M Build Files.

This module provides energy parameter extraction capabilities for PBF-LB/M build files,
leveraging libSLM for accessing energy density and exposure parameters.
"""

from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path
import logging
import numpy as np

from ....external import LIBSLM_AVAILABLE

logger = logging.getLogger(__name__)


class EnergyExtractor:
    """
    Extractor for energy parameters from PBF-LB/M build files.
    
    This extractor analyzes energy density, exposure time, and energy distribution
    using libSLM data structures.
    """
    
    def __init__(self):
        """Initialize the energy extractor."""
        self.libslm_available = LIBSLM_AVAILABLE
        
        if not self.libslm_available:
            logger.warning("libSLM not available - energy analysis will be limited")
        else:
            logger.info("Energy extractor initialized with libSLM support")
    
    def extract_energy_data(self, build_data: Any) -> Dict[str, Any]:
        """
        Extract energy data from build file data.
        
        Args:
            build_data: Parsed build file data from libSLM
            
        Returns:
            Dictionary containing energy analysis results
        """
        try:
            logger.info("Extracting energy data from build file")
            
            energy_data = {
                'energy_density': self._extract_energy_density(build_data),
                'exposure_parameters': self._extract_exposure_parameters(build_data),
                'energy_statistics': self._calculate_energy_statistics(build_data),
                'energy_distribution': self._analyze_energy_distribution(build_data),
                'spatial_energy_map': self._create_spatial_energy_map(build_data)
            }
            
            logger.info("Successfully extracted energy data")
            return energy_data
            
        except Exception as e:
            logger.error(f"Error extracting energy data: {e}")
            raise
    
    def _extract_energy_density(self, build_data: Any) -> List[Dict[str, Any]]:
        """Extract energy density data using libSLM data structures."""
        energy_density = []
        
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
                        
                        # Get process parameters from BuildStyle
                        power = build_style.laserPower if build_style else None
                        velocity = build_style.laserSpeed if build_style else None
                        exposure_time = build_style.pointExposureTime if build_style else None
                        
                        # Calculate energy density
                        energy_density_value = None
                        if power is not None and velocity is not None and velocity > 0:
                            energy_density_value = power / velocity
                        elif power is not None and exposure_time is not None:
                            energy_density_value = power * exposure_time
                        
                        hatch_info = {
                            'layer_index': layer_idx,
                            'hatch_index': hatch_idx,
                            'build_style_id': hatch_geom.bid,
                            'model_id': hatch_geom.mid,
                            'coordinates': coords.tolist() if coords is not None else None,
                            'num_segments': coords.shape[0] // 2 if coords is not None else 0,
                            'power': power,
                            'velocity': velocity,
                            'exposure_time': exposure_time,
                            'energy_density': energy_density_value,
                            'point_distance': build_style.pointDistance if build_style else None,
                            'laser_focus': build_style.laserFocus if build_style else None,
                            'point_delay': build_style.pointDelay if build_style else None,
                            'jump_speed': build_style.jumpSpeed if build_style else None,
                            'jump_delay': build_style.jumpDelay if build_style else None
                        }
                        energy_density.append(hatch_info)
        
        except Exception as e:
            logger.warning(f"Error extracting energy density: {e}")
        
        return energy_density
    
    def _extract_exposure_parameters(self, build_data: Any) -> List[Dict[str, Any]]:
        """Extract exposure parameters using libSLM data structures."""
        exposure_params = []
        
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
                    # Hatch exposure parameters
                    hatch_geometries = layer.getHatchGeometry() if hasattr(layer, 'getHatchGeometry') else []
                    for hatch_idx, hatch_geom in enumerate(hatch_geometries):
                        build_style = build_styles.get(hatch_geom.bid)
                        exposure_info = {
                            'layer_index': layer_idx,
                            'hatch_index': hatch_idx,
                            'build_style_id': hatch_geom.bid,
                            'model_id': hatch_geom.mid,
                            'exposure_time': build_style.pointExposureTime if build_style else None,
                            'point_distance': build_style.pointDistance if build_style else None,
                            'point_delay': build_style.pointDelay if build_style else None,
                            'jump_speed': build_style.jumpSpeed if build_style else None,
                            'jump_delay': build_style.jumpDelay if build_style else None,
                            'laser_focus': build_style.laserFocus if build_style else None
                        }
                        exposure_params.append(exposure_info)
                    
                    # Contour exposure parameters
                    contour_geometries = layer.getContourGeometry() if hasattr(layer, 'getContourGeometry') else []
                    for contour_idx, contour_geom in enumerate(contour_geometries):
                        build_style = build_styles.get(contour_geom.bid)
                        exposure_info = {
                            'layer_index': layer_idx,
                            'contour_index': contour_idx,
                            'build_style_id': contour_geom.bid,
                            'model_id': contour_geom.mid,
                            'exposure_time': build_style.pointExposureTime if build_style else None,
                            'point_distance': build_style.pointDistance if build_style else None,
                            'point_delay': build_style.pointDelay if build_style else None,
                            'jump_speed': build_style.jumpSpeed if build_style else None,
                            'jump_delay': build_style.jumpDelay if build_style else None,
                            'laser_focus': build_style.laserFocus if build_style else None
                        }
                        exposure_params.append(exposure_info)
                    
                    # Point exposure parameters
                    point_geometries = layer.getPointGeometry() if hasattr(layer, 'getPointGeometry') else []
                    for point_idx, point_geom in enumerate(point_geometries):
                        build_style = build_styles.get(point_geom.bid)
                        exposure_info = {
                            'layer_index': layer_idx,
                            'point_index': point_idx,
                            'build_style_id': point_geom.bid,
                            'model_id': point_geom.mid,
                            'exposure_time': build_style.pointExposureTime if build_style else None,
                            'point_distance': build_style.pointDistance if build_style else None,
                            'point_delay': build_style.pointDelay if build_style else None,
                            'jump_speed': build_style.jumpSpeed if build_style else None,
                            'jump_delay': build_style.jumpDelay if build_style else None,
                            'laser_focus': build_style.laserFocus if build_style else None
                        }
                        exposure_params.append(exposure_info)
        
        except Exception as e:
            logger.warning(f"Error extracting exposure parameters: {e}")
        
        return exposure_params
    
    def _calculate_energy_statistics(self, build_data: Any) -> Dict[str, Any]:
        """Calculate energy statistics across the build."""
        statistics = {}
        
        try:
            energy_density = self._extract_energy_density(build_data)
            exposure_params = self._extract_exposure_parameters(build_data)
            
            # Energy density statistics
            energy_values = [e['energy_density'] for e in energy_density if e['energy_density'] is not None]
            power_values = [e['power'] for e in energy_density if e['power'] is not None]
            velocity_values = [e['velocity'] for e in energy_density if e['velocity'] is not None]
            exposure_times = [e['exposure_time'] for e in exposure_params if e['exposure_time'] is not None]
            
            statistics = {
                'energy_density_stats': self._calculate_stats(energy_values),
                'power_stats': self._calculate_stats(power_values),
                'velocity_stats': self._calculate_stats(velocity_values),
                'exposure_time_stats': self._calculate_stats(exposure_times),
                'total_energy_points': len(energy_density),
                'total_exposure_points': len(exposure_params)
            }
        
        except Exception as e:
            logger.warning(f"Error calculating energy statistics: {e}")
            statistics = {'error': str(e)}
        
        return statistics
    
    def _analyze_energy_distribution(self, build_data: Any) -> Dict[str, Any]:
        """Analyze energy distribution patterns."""
        distribution = {}
        
        try:
            energy_density = self._extract_energy_density(build_data)
            
            # Analyze energy variation across layers
            layer_energies = {}
            for energy_point in energy_density:
                layer_idx = energy_point['layer_index']
                energy_value = energy_point['energy_density']
                
                if energy_value is not None:
                    if layer_idx not in layer_energies:
                        layer_energies[layer_idx] = []
                    layer_energies[layer_idx].append(energy_value)
            
            if layer_energies:
                layer_energy_stats = {}
                for layer_idx, energies in layer_energies.items():
                    if energies:
                        energy_array = np.array(energies)
                        layer_energy_stats[layer_idx] = {
                            'mean': float(np.mean(energy_array)),
                            'std': float(np.std(energy_array)),
                            'min': float(np.min(energy_array)),
                            'max': float(np.max(energy_array)),
                            'count': len(energies)
                        }
                
                distribution['layer_energy_variation'] = layer_energy_stats
                
                # Calculate overall energy trend
                layer_means = [stats['mean'] for stats in layer_energy_stats.values()]
                if len(layer_means) > 1:
                    layer_means_array = np.array(layer_means)
                    distribution['energy_trend'] = self._detect_energy_trend(layer_means_array)
                    distribution['energy_consistency'] = float(1.0 - np.std(layer_means_array) / np.mean(layer_means_array)) if np.mean(layer_means_array) > 0 else 0
        
        except Exception as e:
            logger.warning(f"Error analyzing energy distribution: {e}")
            distribution = {'error': str(e)}
        
        return distribution
    
    def _create_spatial_energy_map(self, build_data: Any) -> Dict[str, Any]:
        """Create 3D spatial energy mapping for voxelization."""
        spatial_map = {
            'voxel_resolution': None,
            'energy_voxels': [],
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
            
            # Collect all energy points for voxelization
            energy_points = []
            
            # From energy density data
            energy_density = self._extract_energy_density(build_data)
            for energy_point in energy_density:
                if (energy_point['start_point'] and energy_point['end_point'] and 
                    energy_point['energy_density'] is not None):
                    
                    # Add start point
                    energy_points.append({
                        'x': energy_point['start_point'][0] if len(energy_point['start_point']) > 0 else 0,
                        'y': energy_point['start_point'][1] if len(energy_point['start_point']) > 1 else 0,
                        'z': energy_point['layer_index'] * 0.05,  # Assuming 50μm layer thickness
                        'energy_density': energy_point['energy_density'],
                        'power': energy_point['power'],
                        'velocity': energy_point['velocity']
                    })
                    
                    # Add end point
                    energy_points.append({
                        'x': energy_point['end_point'][0] if len(energy_point['end_point']) > 0 else 0,
                        'y': energy_point['end_point'][1] if len(energy_point['end_point']) > 1 else 0,
                        'z': energy_point['layer_index'] * 0.05,
                        'energy_density': energy_point['energy_density'],
                        'power': energy_point['power'],
                        'velocity': energy_point['velocity']
                    })
            
            spatial_map['energy_voxels'] = energy_points
            spatial_map['total_points'] = len(energy_points)
            
            # Calculate suggested voxel resolution
            if energy_points and spatial_map['bounds']:
                bounds = spatial_map['bounds']
                x_range = bounds['x_max'] - bounds['x_min']
                y_range = bounds['y_max'] - bounds['y_min']
                z_range = bounds['z_max'] - bounds['z_min']
                
                suggested_resolution = min(x_range, y_range, z_range) / 100
                spatial_map['voxel_resolution'] = max(0.01, suggested_resolution)
        
        except Exception as e:
            logger.warning(f"Error creating spatial energy map: {e}")
            spatial_map['error'] = str(e)
        
        return spatial_map
    
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
    
    def _detect_energy_trend(self, energy_values: np.ndarray) -> str:
        """Detect trend in energy values across layers."""
        try:
            if len(energy_values) < 2:
                return 'insufficient_data'
            
            x = np.arange(len(energy_values))
            slope = np.polyfit(x, energy_values, 1)[0]
            
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
            'name': 'Energy Extractor',
            'libslm_available': self.libslm_available,
            'description': 'Extracts and analyzes energy parameters from PBF-LB/M build files using libSLM'
        }
