"""
Laser Focus Extractor for PBF-LB/M Build Files.

This module provides laser focus parameter extraction capabilities for PBF-LB/M build files,
leveraging libSLM for accessing individual scan path focus settings and beam quality analysis.
"""

from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path
import logging
import numpy as np

from .....external import LIBSLM_AVAILABLE

logger = logging.getLogger(__name__)


class LaserFocusExtractor:
    """
    Extractor for laser focus parameters from PBF-LB/M build files.
    
    This extractor analyzes focus settings across different scan strategies
    and provides beam quality analysis using libSLM data.
    """
    
    def __init__(self):
        """Initialize the laser focus extractor."""
        self.libslm_available = LIBSLM_AVAILABLE
        
        if not self.libslm_available:
            logger.warning("libSLM not available - laser focus analysis will be limited")
        else:
            logger.info("Laser focus extractor initialized with libSLM support")
    
    def extract_focus_data(self, build_data: Any) -> Dict[str, Any]:
        """
        Extract laser focus data from build file data.
        
        Args:
            build_data: Parsed build file data from libSLM
            
        Returns:
            Dictionary containing laser focus analysis results
        """
        try:
            logger.info("Extracting laser focus data from build file")
            
            focus_data = {
                'global_focus': self._extract_global_focus(build_data),
                'layer_focus': self._extract_layer_focus(build_data),
                'geometry_focus': self._extract_geometry_focus(build_data),
                'hatch_focus': self._extract_hatch_focus(build_data),
                'contour_focus': self._extract_contour_focus(build_data),
                'point_focus': self._extract_point_focus(build_data),
                'focus_statistics': self._calculate_focus_statistics(build_data),
                'focus_distribution': self._analyze_focus_distribution(build_data),
                'beam_quality_analysis': self._analyze_beam_quality(build_data)
            }
            
            logger.info("Successfully extracted laser focus data")
            return focus_data
            
        except Exception as e:
            logger.error(f"Error extracting laser focus data: {e}")
            raise
    
    def _extract_global_focus(self, build_data: Any) -> Dict[str, Any]:
        """Extract global focus settings."""
        global_focus = {}
        
        try:
            if hasattr(build_data, 'models'):
                for model in build_data.models:
                    if hasattr(model, 'buildStyles'):
                        focus_values = []
                        for build_style in model.buildStyles:
                            if hasattr(build_style, 'laserFocus'):
                                focus_values.append(build_style.laserFocus)
                        
                        if focus_values:
                            global_focus = {
                                'min_focus': min(focus_values),
                                'max_focus': max(focus_values),
                                'mean_focus': np.mean(focus_values),
                                'std_focus': np.std(focus_values),
                                'unique_focus_values': list(set(focus_values)),
                                'focus_count': len(focus_values)
                            }
        except Exception as e:
            logger.warning(f"Error extracting global focus: {e}")
        
        return global_focus
    
    def _extract_layer_focus(self, build_data: Any) -> List[Dict[str, Any]]:
        """Extract layer-specific focus settings."""
        layer_focus = []
        
        try:
            if hasattr(build_data, 'layers'):
                for layer_idx, layer in enumerate(build_data.layers):
                    layer_focus_data = {
                        'layer_index': layer_idx,
                        'focus_values': [],
                        'focus_statistics': {}
                    }
                    
                    # Get all geometries in this layer
                    all_geometries = []
                    if hasattr(layer, 'getHatchGeometry'):
                        all_geometries.extend(layer.getHatchGeometry())
                    if hasattr(layer, 'getContourGeometry'):
                        all_geometries.extend(layer.getContourGeometry())
                    if hasattr(layer, 'getPointGeometry'):
                        all_geometries.extend(layer.getPointGeometry())
                    
                    # Extract focus values from geometries
                    for geom in all_geometries:
                        if hasattr(geom, 'bid'):
                            build_style = self._get_build_style_by_id(build_data, geom.bid)
                            if build_style and hasattr(build_style, 'laserFocus'):
                                layer_focus_data['focus_values'].append(build_style.laserFocus)
                    
                    # Calculate layer statistics
                    if layer_focus_data['focus_values']:
                        focus_values = layer_focus_data['focus_values']
                        layer_focus_data['focus_statistics'] = {
                            'min_focus': min(focus_values),
                            'max_focus': max(focus_values),
                            'mean_focus': np.mean(focus_values),
                            'std_focus': np.std(focus_values),
                            'unique_focus_count': len(set(focus_values))
                        }
                    
                    layer_focus.append(layer_focus_data)
        except Exception as e:
            logger.warning(f"Error extracting layer focus: {e}")
        
        return layer_focus
    
    def _extract_geometry_focus(self, build_data: Any) -> List[Dict[str, Any]]:
        """Extract geometry-specific focus settings."""
        geometry_focus = []
        
        try:
            if hasattr(build_data, 'layers'):
                for layer_idx, layer in enumerate(build_data.layers):
                    # Hatch geometries
                    if hasattr(layer, 'getHatchGeometry'):
                        for hatch_idx, hatch_geom in enumerate(layer.getHatchGeometry()):
                            build_style = self._get_build_style_by_id(build_data, hatch_geom.bid)
                            if build_style and hasattr(build_style, 'laserFocus'):
                                geometry_focus.append({
                                    'layer_index': layer_idx,
                                    'geometry_index': hatch_idx,
                                    'geometry_type': 'hatch',
                                    'build_style_id': hatch_geom.bid,
                                    'laser_focus': build_style.laserFocus,
                                    'laser_power': build_style.laserPower if hasattr(build_style, 'laserPower') else None,
                                    'laser_speed': build_style.laserSpeed if hasattr(build_style, 'laserSpeed') else None
                                })
                    
                    # Contour geometries
                    if hasattr(layer, 'getContourGeometry'):
                        for contour_idx, contour_geom in enumerate(layer.getContourGeometry()):
                            build_style = self._get_build_style_by_id(build_data, contour_geom.bid)
                            if build_style and hasattr(build_style, 'laserFocus'):
                                geometry_focus.append({
                                    'layer_index': layer_idx,
                                    'geometry_index': contour_idx,
                                    'geometry_type': 'contour',
                                    'build_style_id': contour_geom.bid,
                                    'laser_focus': build_style.laserFocus,
                                    'laser_power': build_style.laserPower if hasattr(build_style, 'laserPower') else None,
                                    'laser_speed': build_style.laserSpeed if hasattr(build_style, 'laserSpeed') else None
                                })
                    
                    # Point geometries
                    if hasattr(layer, 'getPointGeometry'):
                        for point_idx, point_geom in enumerate(layer.getPointGeometry()):
                            build_style = self._get_build_style_by_id(build_data, point_geom.bid)
                            if build_style and hasattr(build_style, 'laserFocus'):
                                geometry_focus.append({
                                    'layer_index': layer_idx,
                                    'geometry_index': point_idx,
                                    'geometry_type': 'point',
                                    'build_style_id': point_geom.bid,
                                    'laser_focus': build_style.laserFocus,
                                    'laser_power': build_style.laserPower if hasattr(build_style, 'laserPower') else None,
                                    'laser_speed': build_style.laserSpeed if hasattr(build_style, 'laserSpeed') else None
                                })
        except Exception as e:
            logger.warning(f"Error extracting geometry focus: {e}")
        
        return geometry_focus
    
    def _extract_hatch_focus(self, build_data: Any) -> List[Dict[str, Any]]:
        """Extract hatch-specific focus settings using libSLM data structures."""
        hatch_focus = []
        
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
                            # Focus parameters from BuildStyle
                            'laser_focus': build_style.laserFocus if build_style else None,
                            'laser_power': build_style.laserPower if build_style else None,
                            'laser_speed': build_style.laserSpeed if build_style else None,
                            'exposure_time': build_style.pointExposureTime if build_style else None,
                            'point_distance': build_style.pointDistance if build_style else None,
                            'point_delay': build_style.pointDelay if build_style else None,
                            'jump_speed': build_style.jumpSpeed if build_style else None,
                            'jump_delay': build_style.jumpDelay if build_style else None
                        }
                        hatch_focus.append(hatch_info)
        except Exception as e:
            logger.warning(f"Error extracting hatch focus: {e}")
        
        return hatch_focus
    
    def _extract_contour_focus(self, build_data: Any) -> List[Dict[str, Any]]:
        """Extract contour-specific focus settings using libSLM data structures."""
        contour_focus = []
        
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
                            # Focus parameters from BuildStyle
                            'laser_focus': build_style.laserFocus if build_style else None,
                            'laser_power': build_style.laserPower if build_style else None,
                            'laser_speed': build_style.laserSpeed if build_style else None,
                            'exposure_time': build_style.pointExposureTime if build_style else None,
                            'point_distance': build_style.pointDistance if build_style else None,
                            'point_delay': build_style.pointDelay if build_style else None,
                            'jump_speed': build_style.jumpSpeed if build_style else None,
                            'jump_delay': build_style.jumpDelay if build_style else None
                        }
                        contour_focus.append(contour_info)
        except Exception as e:
            logger.warning(f"Error extracting contour focus: {e}")
        
        return contour_focus
    
    def _extract_point_focus(self, build_data: Any) -> List[Dict[str, Any]]:
        """Extract point-specific focus settings using libSLM data structures."""
        point_focus = []
        
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
                            # Focus parameters from BuildStyle
                            'laser_focus': build_style.laserFocus if build_style else None,
                            'laser_power': build_style.laserPower if build_style else None,
                            'laser_speed': build_style.laserSpeed if build_style else None,
                            'exposure_time': build_style.pointExposureTime if build_style else None,
                            'point_distance': build_style.pointDistance if build_style else None,
                            'point_delay': build_style.pointDelay if build_style else None,
                            'jump_speed': build_style.jumpSpeed if build_style else None,
                            'jump_delay': build_style.jumpDelay if build_style else None
                        }
                        point_focus.append(point_info)
        except Exception as e:
            logger.warning(f"Error extracting point focus: {e}")
        
        return point_focus
    
    def _calculate_focus_statistics(self, build_data: Any) -> Dict[str, Any]:
        """Calculate focus statistics across the build."""
        statistics = {}
        
        try:
            all_focus_values = []
            
            if hasattr(build_data, 'models'):
                for model in build_data.models:
                    if hasattr(model, 'buildStyles'):
                        for build_style in model.buildStyles:
                            if hasattr(build_style, 'laserFocus'):
                                all_focus_values.append(build_style.laserFocus)
            
            if all_focus_values:
                statistics = {
                    'total_focus_measurements': len(all_focus_values),
                    'min_focus': min(all_focus_values),
                    'max_focus': max(all_focus_values),
                    'mean_focus': np.mean(all_focus_values),
                    'median_focus': np.median(all_focus_values),
                    'std_focus': np.std(all_focus_values),
                    'focus_range': max(all_focus_values) - min(all_focus_values),
                    'unique_focus_values': len(set(all_focus_values)),
                    'focus_distribution': self._calculate_focus_distribution_stats(all_focus_values)
                }
        except Exception as e:
            logger.warning(f"Error calculating focus statistics: {e}")
        
        return statistics
    
    def _analyze_focus_distribution(self, build_data: Any) -> Dict[str, Any]:
        """Analyze focus distribution across the build."""
        distribution = {}
        
        try:
            focus_by_geometry_type = {
                'hatch': [],
                'contour': [],
                'point': []
            }
            
            if hasattr(build_data, 'layers'):
                for layer_idx, layer in enumerate(build_data.layers):
                    # Hatch focus
                    if hasattr(layer, 'getHatchGeometry'):
                        for hatch_geom in layer.getHatchGeometry():
                            build_style = self._get_build_style_by_id(build_data, hatch_geom.bid)
                            if build_style and hasattr(build_style, 'laserFocus'):
                                focus_by_geometry_type['hatch'].append(build_style.laserFocus)
                    
                    # Contour focus
                    if hasattr(layer, 'getContourGeometry'):
                        for contour_geom in layer.getContourGeometry():
                            build_style = self._get_build_style_by_id(build_data, contour_geom.bid)
                            if build_style and hasattr(build_style, 'laserFocus'):
                                focus_by_geometry_type['contour'].append(build_style.laserFocus)
                    
                    # Point focus
                    if hasattr(layer, 'getPointGeometry'):
                        for point_geom in layer.getPointGeometry():
                            build_style = self._get_build_style_by_id(build_data, point_geom.bid)
                            if build_style and hasattr(build_style, 'laserFocus'):
                                focus_by_geometry_type['point'].append(build_style.laserFocus)
            
            # Calculate distribution statistics for each geometry type
            for geom_type, focus_values in focus_by_geometry_type.items():
                if focus_values:
                    distribution[geom_type] = {
                        'count': len(focus_values),
                        'min_focus': min(focus_values),
                        'max_focus': max(focus_values),
                        'mean_focus': np.mean(focus_values),
                        'std_focus': np.std(focus_values),
                        'unique_values': len(set(focus_values))
                    }
        except Exception as e:
            logger.warning(f"Error analyzing focus distribution: {e}")
        
        return distribution
    
    def _analyze_beam_quality(self, build_data: Any) -> Dict[str, Any]:
        """Analyze beam quality based on focus parameters."""
        beam_quality = {}
        
        try:
            focus_values = []
            power_values = []
            speed_values = []
            
            if hasattr(build_data, 'models'):
                for model in build_data.models:
                    if hasattr(model, 'buildStyles'):
                        for build_style in model.buildStyles:
                            if hasattr(build_style, 'laserFocus'):
                                focus_values.append(build_style.laserFocus)
                            if hasattr(build_style, 'laserPower'):
                                power_values.append(build_style.laserPower)
                            if hasattr(build_style, 'laserSpeed'):
                                speed_values.append(build_style.laserSpeed)
            
            if focus_values and power_values and speed_values:
                # Calculate beam quality metrics
                focus_consistency = 1.0 - (np.std(focus_values) / np.mean(focus_values)) if np.mean(focus_values) > 0 else 0.0
                power_focus_correlation = np.corrcoef(power_values, focus_values)[0, 1] if len(power_values) == len(focus_values) else 0.0
                
                beam_quality = {
                    'focus_consistency': focus_consistency,
                    'focus_stability': 'high' if focus_consistency > 0.9 else 'medium' if focus_consistency > 0.7 else 'low',
                    'power_focus_correlation': power_focus_correlation,
                    'beam_quality_score': (focus_consistency + abs(power_focus_correlation)) / 2,
                    'focus_variation_coefficient': np.std(focus_values) / np.mean(focus_values) if np.mean(focus_values) > 0 else 0.0,
                    'recommendations': self._generate_focus_recommendations(focus_values, power_values, speed_values)
                }
        except Exception as e:
            logger.warning(f"Error analyzing beam quality: {e}")
        
        return beam_quality
    
    def _calculate_focus_distribution_stats(self, focus_values: List[float]) -> Dict[str, Any]:
        """Calculate detailed focus distribution statistics."""
        if not focus_values:
            return {}
        
        try:
            # Create focus bins for distribution analysis
            min_focus = min(focus_values)
            max_focus = max(focus_values)
            bin_size = (max_focus - min_focus) / 10 if max_focus > min_focus else 1.0
            
            bins = np.arange(min_focus, max_focus + bin_size, bin_size)
            hist, _ = np.histogram(focus_values, bins=bins)
            
            return {
                'histogram': hist.tolist(),
                'bin_edges': bins.tolist(),
                'most_common_focus': bins[np.argmax(hist)],
                'focus_entropy': -np.sum((hist / np.sum(hist)) * np.log2(hist / np.sum(hist) + 1e-10))
            }
        except Exception as e:
            logger.warning(f"Error calculating focus distribution stats: {e}")
            return {}
    
    def _generate_focus_recommendations(self, focus_values: List[float], power_values: List[float], speed_values: List[float]) -> List[str]:
        """Generate recommendations based on focus analysis."""
        recommendations = []
        
        try:
            if not focus_values:
                return ["No focus data available for analysis"]
            
            focus_std = np.std(focus_values)
            focus_mean = np.mean(focus_values)
            
            if focus_std / focus_mean > 0.2:
                recommendations.append("High focus variation detected - consider standardizing focus settings")
            
            if len(set(focus_values)) == 1:
                recommendations.append("Single focus value used throughout build - consider optimizing for different geometries")
            
            if focus_std / focus_mean < 0.05:
                recommendations.append("Very consistent focus settings - good for process stability")
            
            # Check for focus-power correlation
            if len(focus_values) == len(power_values):
                correlation = np.corrcoef(focus_values, power_values)[0, 1]
                if abs(correlation) > 0.7:
                    recommendations.append(f"Strong focus-power correlation ({correlation:.2f}) - consider decoupling for optimization")
            
        except Exception as e:
            logger.warning(f"Error generating focus recommendations: {e}")
            recommendations.append("Error generating recommendations")
        
        return recommendations
    
    def _get_build_style_by_id(self, build_data: Any, build_style_id: int) -> Optional[Any]:
        """Get build style by ID."""
        try:
            if hasattr(build_data, 'models'):
                for model in build_data.models:
                    if hasattr(model, 'buildStyles'):
                        for build_style in model.buildStyles:
                            if hasattr(build_style, 'id') and build_style.id == build_style_id:
                                return build_style
        except Exception as e:
            logger.warning(f"Error getting build style by ID: {e}")
        
        return None
    
    def get_extractor_info(self) -> Dict[str, Any]:
        """Get extractor information."""
        return {
            'name': 'Laser Focus Extractor',
            'libslm_available': self.libslm_available,
            'description': 'Extracts and analyzes laser focus parameters from PBF-LB/M build files using libSLM'
        }
