"""
Jump Parameters Extractor for PBF-LB/M Build Files.

This module provides jump parameter extraction capabilities for PBF-LB/M build files,
leveraging libSLM for accessing jump speed and delay parameters for non-printing movements.
"""

from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path
import logging
import numpy as np

from .....external import LIBSLM_AVAILABLE

logger = logging.getLogger(__name__)


class JumpParametersExtractor:
    """
    Extractor for jump parameters from PBF-LB/M build files.
    
    This extractor analyzes jump speed and delay settings across different scan strategies
    and provides build time optimization analysis using libSLM data.
    """
    
    def __init__(self):
        """Initialize the jump parameters extractor."""
        self.libslm_available = LIBSLM_AVAILABLE
        
        if not self.libslm_available:
            logger.warning("libSLM not available - jump parameters analysis will be limited")
        else:
            logger.info("Jump parameters extractor initialized with libSLM support")
    
    def extract_jump_data(self, build_data: Any) -> Dict[str, Any]:
        """
        Extract jump parameters data from build file data.
        
        Args:
            build_data: Parsed build file data from libSLM
            
        Returns:
            Dictionary containing jump parameters analysis results
        """
        try:
            logger.info("Extracting jump parameters data from build file")
            
            jump_data = {
                'global_jump_parameters': self._extract_global_jump_parameters(build_data),
                'layer_jump_parameters': self._extract_layer_jump_parameters(build_data),
                'geometry_jump_parameters': self._extract_geometry_jump_parameters(build_data),
                'hatch_jump_parameters': self._extract_hatch_jump_parameters(build_data),
                'contour_jump_parameters': self._extract_contour_jump_parameters(build_data),
                'point_jump_parameters': self._extract_point_jump_parameters(build_data),
                'jump_statistics': self._calculate_jump_statistics(build_data),
                'jump_distribution': self._analyze_jump_distribution(build_data),
                'build_time_optimization': self._analyze_build_time_optimization(build_data)
            }
            
            logger.info("Successfully extracted jump parameters data")
            return jump_data
            
        except Exception as e:
            logger.error(f"Error extracting jump parameters data: {e}")
            raise
    
    def _extract_global_jump_parameters(self, build_data: Any) -> Dict[str, Any]:
        """Extract global jump parameters."""
        global_jump = {}
        
        try:
            if hasattr(build_data, 'models'):
                for model in build_data.models:
                    if hasattr(model, 'buildStyles'):
                        jump_speeds = []
                        jump_delays = []
                        
                        for build_style in model.buildStyles:
                            if hasattr(build_style, 'jumpSpeed'):
                                jump_speeds.append(build_style.jumpSpeed)
                            if hasattr(build_style, 'jumpDelay'):
                                jump_delays.append(build_style.jumpDelay)
                        
                        if jump_speeds and jump_delays:
                            global_jump = {
                                'jump_speed': {
                                    'min': min(jump_speeds),
                                    'max': max(jump_speeds),
                                    'mean': np.mean(jump_speeds),
                                    'std': np.std(jump_speeds),
                                    'unique_values': list(set(jump_speeds)),
                                    'count': len(jump_speeds)
                                },
                                'jump_delay': {
                                    'min': min(jump_delays),
                                    'max': max(jump_delays),
                                    'mean': np.mean(jump_delays),
                                    'std': np.std(jump_delays),
                                    'unique_values': list(set(jump_delays)),
                                    'count': len(jump_delays)
                                }
                            }
        except Exception as e:
            logger.warning(f"Error extracting global jump parameters: {e}")
        
        return global_jump
    
    def _extract_layer_jump_parameters(self, build_data: Any) -> List[Dict[str, Any]]:
        """Extract layer-specific jump parameters."""
        layer_jump = []
        
        try:
            if hasattr(build_data, 'layers'):
                for layer_idx, layer in enumerate(build_data.layers):
                    layer_jump_data = {
                        'layer_index': layer_idx,
                        'jump_speeds': [],
                        'jump_delays': [],
                        'jump_statistics': {}
                    }
                    
                    # Get all geometries in this layer
                    all_geometries = []
                    if hasattr(layer, 'getHatchGeometry'):
                        all_geometries.extend(layer.getHatchGeometry())
                    if hasattr(layer, 'getContourGeometry'):
                        all_geometries.extend(layer.getContourGeometry())
                    if hasattr(layer, 'getPointGeometry'):
                        all_geometries.extend(layer.getPointGeometry())
                    
                    # Extract jump parameters from geometries
                    for geom in all_geometries:
                        if hasattr(geom, 'bid'):
                            build_style = self._get_build_style_by_id(build_data, geom.bid)
                            if build_style:
                                if hasattr(build_style, 'jumpSpeed'):
                                    layer_jump_data['jump_speeds'].append(build_style.jumpSpeed)
                                if hasattr(build_style, 'jumpDelay'):
                                    layer_jump_data['jump_delays'].append(build_style.jumpDelay)
                    
                    # Calculate layer statistics
                    if layer_jump_data['jump_speeds'] and layer_jump_data['jump_delays']:
                        jump_speeds = layer_jump_data['jump_speeds']
                        jump_delays = layer_jump_data['jump_delays']
                        
                        layer_jump_data['jump_statistics'] = {
                            'jump_speed': {
                                'min': min(jump_speeds),
                                'max': max(jump_speeds),
                                'mean': np.mean(jump_speeds),
                                'std': np.std(jump_speeds),
                                'unique_count': len(set(jump_speeds))
                            },
                            'jump_delay': {
                                'min': min(jump_delays),
                                'max': max(jump_delays),
                                'mean': np.mean(jump_delays),
                                'std': np.std(jump_delays),
                                'unique_count': len(set(jump_delays))
                            }
                        }
                    
                    layer_jump.append(layer_jump_data)
        except Exception as e:
            logger.warning(f"Error extracting layer jump parameters: {e}")
        
        return layer_jump
    
    def _extract_geometry_jump_parameters(self, build_data: Any) -> List[Dict[str, Any]]:
        """Extract geometry-specific jump parameters."""
        geometry_jump = []
        
        try:
            if hasattr(build_data, 'layers'):
                for layer_idx, layer in enumerate(build_data.layers):
                    # Hatch geometries
                    if hasattr(layer, 'getHatchGeometry'):
                        for hatch_idx, hatch_geom in enumerate(layer.getHatchGeometry()):
                            build_style = self._get_build_style_by_id(build_data, hatch_geom.bid)
                            if build_style:
                                geometry_jump.append({
                                    'layer_index': layer_idx,
                                    'geometry_index': hatch_idx,
                                    'geometry_type': 'hatch',
                                    'build_style_id': hatch_geom.bid,
                                    'jump_speed': build_style.jumpSpeed if hasattr(build_style, 'jumpSpeed') else None,
                                    'jump_delay': build_style.jumpDelay if hasattr(build_style, 'jumpDelay') else None,
                                    'laser_power': build_style.laserPower if hasattr(build_style, 'laserPower') else None,
                                    'laser_speed': build_style.laserSpeed if hasattr(build_style, 'laserSpeed') else None
                                })
                    
                    # Contour geometries
                    if hasattr(layer, 'getContourGeometry'):
                        for contour_idx, contour_geom in enumerate(layer.getContourGeometry()):
                            build_style = self._get_build_style_by_id(build_data, contour_geom.bid)
                            if build_style:
                                geometry_jump.append({
                                    'layer_index': layer_idx,
                                    'geometry_index': contour_idx,
                                    'geometry_type': 'contour',
                                    'build_style_id': contour_geom.bid,
                                    'jump_speed': build_style.jumpSpeed if hasattr(build_style, 'jumpSpeed') else None,
                                    'jump_delay': build_style.jumpDelay if hasattr(build_style, 'jumpDelay') else None,
                                    'laser_power': build_style.laserPower if hasattr(build_style, 'laserPower') else None,
                                    'laser_speed': build_style.laserSpeed if hasattr(build_style, 'laserSpeed') else None
                                })
                    
                    # Point geometries
                    if hasattr(layer, 'getPointGeometry'):
                        for point_idx, point_geom in enumerate(layer.getPointGeometry()):
                            build_style = self._get_build_style_by_id(build_data, point_geom.bid)
                            if build_style:
                                geometry_jump.append({
                                    'layer_index': layer_idx,
                                    'geometry_index': point_idx,
                                    'geometry_type': 'point',
                                    'build_style_id': point_geom.bid,
                                    'jump_speed': build_style.jumpSpeed if hasattr(build_style, 'jumpSpeed') else None,
                                    'jump_delay': build_style.jumpDelay if hasattr(build_style, 'jumpDelay') else None,
                                    'laser_power': build_style.laserPower if hasattr(build_style, 'laserPower') else None,
                                    'laser_speed': build_style.laserSpeed if hasattr(build_style, 'laserSpeed') else None
                                })
        except Exception as e:
            logger.warning(f"Error extracting geometry jump parameters: {e}")
        
        return geometry_jump
    
    def _extract_hatch_jump_parameters(self, build_data: Any) -> List[Dict[str, Any]]:
        """Extract hatch-specific jump parameters using libSLM data structures."""
        hatch_jump = []
        
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
                            # Jump parameters from BuildStyle
                            'jump_speed': build_style.jumpSpeed if build_style else None,
                            'jump_delay': build_style.jumpDelay if build_style else None,
                            'laser_power': build_style.laserPower if build_style else None,
                            'laser_speed': build_style.laserSpeed if build_style else None,
                            'exposure_time': build_style.pointExposureTime if build_style else None,
                            'point_distance': build_style.pointDistance if build_style else None,
                            'point_delay': build_style.pointDelay if build_style else None,
                            'laser_focus': build_style.laserFocus if build_style else None
                        }
                        hatch_jump.append(hatch_info)
        except Exception as e:
            logger.warning(f"Error extracting hatch jump parameters: {e}")
        
        return hatch_jump
    
    def _extract_contour_jump_parameters(self, build_data: Any) -> List[Dict[str, Any]]:
        """Extract contour-specific jump parameters using libSLM data structures."""
        contour_jump = []
        
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
                            # Jump parameters from BuildStyle
                            'jump_speed': build_style.jumpSpeed if build_style else None,
                            'jump_delay': build_style.jumpDelay if build_style else None,
                            'laser_power': build_style.laserPower if build_style else None,
                            'laser_speed': build_style.laserSpeed if build_style else None,
                            'exposure_time': build_style.pointExposureTime if build_style else None,
                            'point_distance': build_style.pointDistance if build_style else None,
                            'point_delay': build_style.pointDelay if build_style else None,
                            'laser_focus': build_style.laserFocus if build_style else None
                        }
                        contour_jump.append(contour_info)
        except Exception as e:
            logger.warning(f"Error extracting contour jump parameters: {e}")
        
        return contour_jump
    
    def _extract_point_jump_parameters(self, build_data: Any) -> List[Dict[str, Any]]:
        """Extract point-specific jump parameters using libSLM data structures."""
        point_jump = []
        
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
                            # Jump parameters from BuildStyle
                            'jump_speed': build_style.jumpSpeed if build_style else None,
                            'jump_delay': build_style.jumpDelay if build_style else None,
                            'laser_power': build_style.laserPower if build_style else None,
                            'laser_speed': build_style.laserSpeed if build_style else None,
                            'exposure_time': build_style.pointExposureTime if build_style else None,
                            'point_distance': build_style.pointDistance if build_style else None,
                            'point_delay': build_style.pointDelay if build_style else None,
                            'laser_focus': build_style.laserFocus if build_style else None
                        }
                        point_jump.append(point_info)
        except Exception as e:
            logger.warning(f"Error extracting point jump parameters: {e}")
        
        return point_jump
    
    def _calculate_jump_statistics(self, build_data: Any) -> Dict[str, Any]:
        """Calculate jump statistics across the build."""
        statistics = {}
        
        try:
            all_jump_speeds = []
            all_jump_delays = []
            
            if hasattr(build_data, 'models'):
                for model in build_data.models:
                    if hasattr(model, 'buildStyles'):
                        for build_style in model.buildStyles:
                            if hasattr(build_style, 'jumpSpeed'):
                                all_jump_speeds.append(build_style.jumpSpeed)
                            if hasattr(build_style, 'jumpDelay'):
                                all_jump_delays.append(build_style.jumpDelay)
            
            if all_jump_speeds and all_jump_delays:
                statistics = {
                    'jump_speed': {
                        'total_measurements': len(all_jump_speeds),
                        'min': min(all_jump_speeds),
                        'max': max(all_jump_speeds),
                        'mean': np.mean(all_jump_speeds),
                        'median': np.median(all_jump_speeds),
                        'std': np.std(all_jump_speeds),
                        'range': max(all_jump_speeds) - min(all_jump_speeds),
                        'unique_values': len(set(all_jump_speeds))
                    },
                    'jump_delay': {
                        'total_measurements': len(all_jump_delays),
                        'min': min(all_jump_delays),
                        'max': max(all_jump_delays),
                        'mean': np.mean(all_jump_delays),
                        'median': np.median(all_jump_delays),
                        'std': np.std(all_jump_delays),
                        'range': max(all_jump_delays) - min(all_jump_delays),
                        'unique_values': len(set(all_jump_delays))
                    }
                }
        except Exception as e:
            logger.warning(f"Error calculating jump statistics: {e}")
        
        return statistics
    
    def _analyze_jump_distribution(self, build_data: Any) -> Dict[str, Any]:
        """Analyze jump parameter distribution across the build."""
        distribution = {}
        
        try:
            jump_by_geometry_type = {
                'hatch': {'jump_speeds': [], 'jump_delays': []},
                'contour': {'jump_speeds': [], 'jump_delays': []},
                'point': {'jump_speeds': [], 'jump_delays': []}
            }
            
            if hasattr(build_data, 'layers'):
                for layer_idx, layer in enumerate(build_data.layers):
                    # Hatch jump parameters
                    if hasattr(layer, 'getHatchGeometry'):
                        for hatch_geom in layer.getHatchGeometry():
                            build_style = self._get_build_style_by_id(build_data, hatch_geom.bid)
                            if build_style:
                                if hasattr(build_style, 'jumpSpeed'):
                                    jump_by_geometry_type['hatch']['jump_speeds'].append(build_style.jumpSpeed)
                                if hasattr(build_style, 'jumpDelay'):
                                    jump_by_geometry_type['hatch']['jump_delays'].append(build_style.jumpDelay)
                    
                    # Contour jump parameters
                    if hasattr(layer, 'getContourGeometry'):
                        for contour_geom in layer.getContourGeometry():
                            build_style = self._get_build_style_by_id(build_data, contour_geom.bid)
                            if build_style:
                                if hasattr(build_style, 'jumpSpeed'):
                                    jump_by_geometry_type['contour']['jump_speeds'].append(build_style.jumpSpeed)
                                if hasattr(build_style, 'jumpDelay'):
                                    jump_by_geometry_type['contour']['jump_delays'].append(build_style.jumpDelay)
                    
                    # Point jump parameters
                    if hasattr(layer, 'getPointGeometry'):
                        for point_geom in layer.getPointGeometry():
                            build_style = self._get_build_style_by_id(build_data, point_geom.bid)
                            if build_style:
                                if hasattr(build_style, 'jumpSpeed'):
                                    jump_by_geometry_type['point']['jump_speeds'].append(build_style.jumpSpeed)
                                if hasattr(build_style, 'jumpDelay'):
                                    jump_by_geometry_type['point']['jump_delays'].append(build_style.jumpDelay)
            
            # Calculate distribution statistics for each geometry type
            for geom_type, jump_data in jump_by_geometry_type.items():
                jump_speeds = jump_data['jump_speeds']
                jump_delays = jump_data['jump_delays']
                
                if jump_speeds and jump_delays:
                    distribution[geom_type] = {
                        'jump_speed': {
                            'count': len(jump_speeds),
                            'min': min(jump_speeds),
                            'max': max(jump_speeds),
                            'mean': np.mean(jump_speeds),
                            'std': np.std(jump_speeds),
                            'unique_values': len(set(jump_speeds))
                        },
                        'jump_delay': {
                            'count': len(jump_delays),
                            'min': min(jump_delays),
                            'max': max(jump_delays),
                            'mean': np.mean(jump_delays),
                            'std': np.std(jump_delays),
                            'unique_values': len(set(jump_delays))
                        }
                    }
        except Exception as e:
            logger.warning(f"Error analyzing jump distribution: {e}")
        
        return distribution
    
    def _analyze_build_time_optimization(self, build_data: Any) -> Dict[str, Any]:
        """Analyze build time optimization based on jump parameters."""
        optimization = {}
        
        try:
            jump_speeds = []
            jump_delays = []
            laser_speeds = []
            
            if hasattr(build_data, 'models'):
                for model in build_data.models:
                    if hasattr(model, 'buildStyles'):
                        for build_style in model.buildStyles:
                            if hasattr(build_style, 'jumpSpeed'):
                                jump_speeds.append(build_style.jumpSpeed)
                            if hasattr(build_style, 'jumpDelay'):
                                jump_delays.append(build_style.jumpDelay)
                            if hasattr(build_style, 'laserSpeed'):
                                laser_speeds.append(build_style.laserSpeed)
            
            if jump_speeds and jump_delays and laser_speeds:
                # Calculate optimization metrics
                jump_speed_consistency = 1.0 - (np.std(jump_speeds) / np.mean(jump_speeds)) if np.mean(jump_speeds) > 0 else 0.0
                jump_delay_consistency = 1.0 - (np.std(jump_delays) / np.mean(jump_delays)) if np.mean(jump_delays) > 0 else 0.0
                
                # Calculate potential time savings
                max_jump_speed = max(jump_speeds)
                min_jump_delay = min(jump_delays)
                avg_jump_speed = np.mean(jump_speeds)
                avg_jump_delay = np.mean(jump_delays)
                
                potential_speed_improvement = (max_jump_speed - avg_jump_speed) / avg_jump_speed if avg_jump_speed > 0 else 0.0
                potential_delay_reduction = (avg_jump_delay - min_jump_delay) / avg_jump_delay if avg_jump_delay > 0 else 0.0
                
                optimization = {
                    'jump_speed_consistency': jump_speed_consistency,
                    'jump_delay_consistency': jump_delay_consistency,
                    'potential_speed_improvement': potential_speed_improvement,
                    'potential_delay_reduction': potential_delay_reduction,
                    'optimization_score': (jump_speed_consistency + jump_delay_consistency) / 2,
                    'recommendations': self._generate_jump_optimization_recommendations(
                        jump_speeds, jump_delays, laser_speeds
                    )
                }
        except Exception as e:
            logger.warning(f"Error analyzing build time optimization: {e}")
        
        return optimization
    
    def _generate_jump_optimization_recommendations(self, jump_speeds: List[float], jump_delays: List[float], laser_speeds: List[float]) -> List[str]:
        """Generate optimization recommendations based on jump parameter analysis."""
        recommendations = []
        
        try:
            if not jump_speeds or not jump_delays:
                return ["No jump data available for analysis"]
            
            jump_speed_std = np.std(jump_speeds)
            jump_speed_mean = np.mean(jump_speeds)
            jump_delay_std = np.std(jump_delays)
            jump_delay_mean = np.mean(jump_delays)
            
            # Jump speed recommendations
            if jump_speed_std / jump_speed_mean > 0.3:
                recommendations.append("High jump speed variation detected - consider standardizing for better build time consistency")
            
            if len(set(jump_speeds)) == 1:
                recommendations.append("Single jump speed used throughout build - consider optimizing for different geometries")
            
            # Jump delay recommendations
            if jump_delay_std / jump_delay_mean > 0.3:
                recommendations.append("High jump delay variation detected - consider standardizing for better build time consistency")
            
            if len(set(jump_delays)) == 1:
                recommendations.append("Single jump delay used throughout build - consider optimizing for different geometries")
            
            # Speed optimization recommendations
            if jump_speed_mean < np.mean(laser_speeds) * 0.5:
                recommendations.append("Jump speeds significantly lower than laser speeds - consider increasing for faster builds")
            
            # Delay optimization recommendations
            if jump_delay_mean > 100:  # Assuming milliseconds
                recommendations.append("High jump delays detected - consider reducing for faster builds")
            
            # Consistency recommendations
            if jump_speed_std / jump_speed_mean < 0.1 and jump_delay_std / jump_delay_mean < 0.1:
                recommendations.append("Very consistent jump parameters - good for build time predictability")
            
        except Exception as e:
            logger.warning(f"Error generating jump optimization recommendations: {e}")
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
            'name': 'Jump Parameters Extractor',
            'libslm_available': self.libslm_available,
            'description': 'Extracts and analyzes jump parameters from PBF-LB/M build files using libSLM'
        }
