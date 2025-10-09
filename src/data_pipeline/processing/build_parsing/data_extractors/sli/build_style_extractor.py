"""
Build Style Extractor for PBF-LB/M Build Files.

This module provides build style metadata extraction capabilities for PBF-LB/M build files,
leveraging libSLM for accessing build style information and process optimization analysis.
"""

from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path
import logging
import numpy as np

from .....external import LIBSLM_AVAILABLE

logger = logging.getLogger(__name__)


class BuildStyleExtractor:
    """
    Extractor for build style metadata from PBF-LB/M build files.
    
    This extractor analyzes build style information across different scan strategies
    and provides process optimization analysis using libSLM data.
    """
    
    def __init__(self):
        """Initialize the build style extractor."""
        self.libslm_available = LIBSLM_AVAILABLE
        
        if not self.libslm_available:
            logger.warning("libSLM not available - build style analysis will be limited")
        else:
            logger.info("Build style extractor initialized with libSLM support")
    
    def extract_build_style_data(self, build_data: Any) -> Dict[str, Any]:
        """
        Extract build style data from build file data.
        
        Args:
            build_data: Parsed build file data from libSLM
            
        Returns:
            Dictionary containing build style analysis results
        """
        try:
            logger.info("Extracting build style data from build file")
            
            build_style_data = {
                'build_styles': self._extract_build_styles(build_data),
                'style_usage': self._extract_style_usage(build_data),
                'style_statistics': self._calculate_style_statistics(build_data),
                'style_comparison': self._compare_build_styles(build_data),
                'process_optimization': self._analyze_process_optimization(build_data),
                'style_recommendations': self._generate_style_recommendations(build_data)
            }
            
            logger.info("Successfully extracted build style data")
            return build_style_data
            
        except Exception as e:
            logger.error(f"Error extracting build style data: {e}")
            raise
    
    def _extract_build_styles(self, build_data: Any) -> List[Dict[str, Any]]:
        """Extract all build styles from the build file."""
        build_styles = []
        
        try:
            if hasattr(build_data, 'models'):
                for model_idx, model in enumerate(build_data.models):
                    if hasattr(model, 'buildStyles'):
                        for style_idx, build_style in enumerate(model.buildStyles):
                            style_info = {
                                'model_index': model_idx,
                                'style_index': style_idx,
                                'style_id': build_style.id if hasattr(build_style, 'id') else None,
                                'laser_id': build_style.laserId if hasattr(build_style, 'laserId') else None,
                                'laser_mode': build_style.laserMode if hasattr(build_style, 'laserMode') else None,
                                'name': build_style.name if hasattr(build_style, 'name') else None,
                                'description': build_style.description if hasattr(build_style, 'description') else None,
                                # Process parameters
                                'laser_power': build_style.laserPower if hasattr(build_style, 'laserPower') else None,
                                'laser_focus': build_style.laserFocus if hasattr(build_style, 'laserFocus') else None,
                                'laser_speed': build_style.laserSpeed if hasattr(build_style, 'laserSpeed') else None,
                                'point_distance': build_style.pointDistance if hasattr(build_style, 'pointDistance') else None,
                                'point_delay': build_style.pointDelay if hasattr(build_style, 'pointDelay') else None,
                                'point_exposure_time': build_style.pointExposureTime if hasattr(build_style, 'pointExposureTime') else None,
                                'jump_speed': build_style.jumpSpeed if hasattr(build_style, 'jumpSpeed') else None,
                                'jump_delay': build_style.jumpDelay if hasattr(build_style, 'jumpDelay') else None,
                                # Calculated parameters
                                'energy_density': self._calculate_energy_density(build_style),
                                'scan_speed': self._calculate_scan_speed(build_style),
                                'duty_cycle': self._calculate_duty_cycle(build_style)
                            }
                            build_styles.append(style_info)
        except Exception as e:
            logger.warning(f"Error extracting build styles: {e}")
        
        return build_styles
    
    def _extract_style_usage(self, build_data: Any) -> Dict[str, Any]:
        """Extract build style usage statistics."""
        style_usage = {}
        
        try:
            style_usage_count = {}
            style_usage_by_layer = {}
            style_usage_by_geometry = {}
            
            if hasattr(build_data, 'layers'):
                for layer_idx, layer in enumerate(build_data.layers):
                    layer_style_usage = {}
                    
                    # Count hatch style usage
                    if hasattr(layer, 'getHatchGeometry'):
                        for hatch_geom in layer.getHatchGeometry():
                            style_id = hatch_geom.bid
                            style_usage_count[style_id] = style_usage_count.get(style_id, 0) + 1
                            layer_style_usage[style_id] = layer_style_usage.get(style_id, 0) + 1
                            style_usage_by_geometry[style_id] = style_usage_by_geometry.get(style_id, {'hatch': 0, 'contour': 0, 'point': 0})
                            style_usage_by_geometry[style_id]['hatch'] += 1
                    
                    # Count contour style usage
                    if hasattr(layer, 'getContourGeometry'):
                        for contour_geom in layer.getContourGeometry():
                            style_id = contour_geom.bid
                            style_usage_count[style_id] = style_usage_count.get(style_id, 0) + 1
                            layer_style_usage[style_id] = layer_style_usage.get(style_id, 0) + 1
                            style_usage_by_geometry[style_id] = style_usage_by_geometry.get(style_id, {'hatch': 0, 'contour': 0, 'point': 0})
                            style_usage_by_geometry[style_id]['contour'] += 1
                    
                    # Count point style usage
                    if hasattr(layer, 'getPointGeometry'):
                        for point_geom in layer.getPointGeometry():
                            style_id = point_geom.bid
                            style_usage_count[style_id] = style_usage_count.get(style_id, 0) + 1
                            layer_style_usage[style_id] = layer_style_usage.get(style_id, 0) + 1
                            style_usage_by_geometry[style_id] = style_usage_by_geometry.get(style_id, {'hatch': 0, 'contour': 0, 'point': 0})
                            style_usage_by_geometry[style_id]['point'] += 1
                    
                    style_usage_by_layer[layer_idx] = layer_style_usage
            
            style_usage = {
                'total_usage_count': style_usage_count,
                'usage_by_layer': style_usage_by_layer,
                'usage_by_geometry_type': style_usage_by_geometry,
                'most_used_style': max(style_usage_count.items(), key=lambda x: x[1]) if style_usage_count else None,
                'least_used_style': min(style_usage_count.items(), key=lambda x: x[1]) if style_usage_count else None,
                'total_geometries': sum(style_usage_count.values()) if style_usage_count else 0
            }
        except Exception as e:
            logger.warning(f"Error extracting style usage: {e}")
        
        return style_usage
    
    def _calculate_style_statistics(self, build_data: Any) -> Dict[str, Any]:
        """Calculate build style statistics."""
        statistics = {}
        
        try:
            all_powers = []
            all_speeds = []
            all_focuses = []
            all_exposure_times = []
            all_jump_speeds = []
            all_jump_delays = []
            
            if hasattr(build_data, 'models'):
                for model in build_data.models:
                    if hasattr(model, 'buildStyles'):
                        for build_style in model.buildStyles:
                            if hasattr(build_style, 'laserPower'):
                                all_powers.append(build_style.laserPower)
                            if hasattr(build_style, 'laserSpeed'):
                                all_speeds.append(build_style.laserSpeed)
                            if hasattr(build_style, 'laserFocus'):
                                all_focuses.append(build_style.laserFocus)
                            if hasattr(build_style, 'pointExposureTime'):
                                all_exposure_times.append(build_style.pointExposureTime)
                            if hasattr(build_style, 'jumpSpeed'):
                                all_jump_speeds.append(build_style.jumpSpeed)
                            if hasattr(build_style, 'jumpDelay'):
                                all_jump_delays.append(build_style.jumpDelay)
            
            statistics = {
                'total_build_styles': len(all_powers) if all_powers else 0,
                'laser_power': self._calculate_parameter_statistics(all_powers),
                'laser_speed': self._calculate_parameter_statistics(all_speeds),
                'laser_focus': self._calculate_parameter_statistics(all_focuses),
                'exposure_time': self._calculate_parameter_statistics(all_exposure_times),
                'jump_speed': self._calculate_parameter_statistics(all_jump_speeds),
                'jump_delay': self._calculate_parameter_statistics(all_jump_delays)
            }
        except Exception as e:
            logger.warning(f"Error calculating style statistics: {e}")
        
        return statistics
    
    def _compare_build_styles(self, build_data: Any) -> Dict[str, Any]:
        """Compare different build styles."""
        comparison = {}
        
        try:
            build_styles = self._extract_build_styles(build_data)
            
            if len(build_styles) > 1:
                # Compare power ranges
                powers = [style['laser_power'] for style in build_styles if style['laser_power'] is not None]
                speeds = [style['laser_speed'] for style in build_styles if style['laser_speed'] is not None]
                focuses = [style['laser_focus'] for style in build_styles if style['laser_focus'] is not None]
                
                comparison = {
                    'power_comparison': {
                        'min_power': min(powers) if powers else None,
                        'max_power': max(powers) if powers else None,
                        'power_range': max(powers) - min(powers) if powers else None,
                        'power_variation': np.std(powers) / np.mean(powers) if powers and np.mean(powers) > 0 else None
                    },
                    'speed_comparison': {
                        'min_speed': min(speeds) if speeds else None,
                        'max_speed': max(speeds) if speeds else None,
                        'speed_range': max(speeds) - min(speeds) if speeds else None,
                        'speed_variation': np.std(speeds) / np.mean(speeds) if speeds and np.mean(speeds) > 0 else None
                    },
                    'focus_comparison': {
                        'min_focus': min(focuses) if focuses else None,
                        'max_focus': max(focuses) if focuses else None,
                        'focus_range': max(focuses) - min(focuses) if focuses else None,
                        'focus_variation': np.std(focuses) / np.mean(focuses) if focuses and np.mean(focuses) > 0 else None
                    },
                    'style_diversity': {
                        'unique_power_values': len(set(powers)) if powers else 0,
                        'unique_speed_values': len(set(speeds)) if speeds else 0,
                        'unique_focus_values': len(set(focuses)) if focuses else 0,
                        'diversity_score': self._calculate_diversity_score(build_styles)
                    }
                }
        except Exception as e:
            logger.warning(f"Error comparing build styles: {e}")
        
        return comparison
    
    def _analyze_process_optimization(self, build_data: Any) -> Dict[str, Any]:
        """Analyze process optimization opportunities."""
        optimization = {}
        
        try:
            build_styles = self._extract_build_styles(build_data)
            style_usage = self._extract_style_usage(build_data)
            
            if build_styles and style_usage:
                # Analyze parameter consistency
                powers = [style['laser_power'] for style in build_styles if style['laser_power'] is not None]
                speeds = [style['laser_speed'] for style in build_styles if style['laser_speed'] is not None]
                focuses = [style['laser_focus'] for style in build_styles if style['laser_focus'] is not None]
                
                optimization = {
                    'parameter_consistency': {
                        'power_consistency': 1.0 - (np.std(powers) / np.mean(powers)) if powers and np.mean(powers) > 0 else 0.0,
                        'speed_consistency': 1.0 - (np.std(speeds) / np.mean(speeds)) if speeds and np.mean(speeds) > 0 else 0.0,
                        'focus_consistency': 1.0 - (np.std(focuses) / np.mean(focuses)) if focuses and np.mean(focuses) > 0 else 0.0
                    },
                    'style_efficiency': {
                        'most_efficient_style': self._find_most_efficient_style(build_styles),
                        'least_efficient_style': self._find_least_efficient_style(build_styles),
                        'efficiency_variation': self._calculate_efficiency_variation(build_styles)
                    },
                    'optimization_opportunities': self._identify_optimization_opportunities(build_styles, style_usage)
                }
        except Exception as e:
            logger.warning(f"Error analyzing process optimization: {e}")
        
        return optimization
    
    def _generate_style_recommendations(self, build_data: Any) -> List[str]:
        """Generate build style recommendations."""
        recommendations = []
        
        try:
            build_styles = self._extract_build_styles(build_data)
            style_usage = self._extract_style_usage(build_data)
            
            if not build_styles:
                return ["No build styles found for analysis"]
            
            # Check for style diversity
            if len(build_styles) == 1:
                recommendations.append("Single build style used - consider creating specialized styles for different geometries")
            elif len(build_styles) > 10:
                recommendations.append("Many build styles used - consider consolidating similar styles for better process control")
            
            # Check for parameter consistency
            powers = [style['laser_power'] for style in build_styles if style['laser_power'] is not None]
            if powers and len(set(powers)) == 1:
                recommendations.append("Single power value used across all styles - consider optimizing for different geometries")
            
            speeds = [style['laser_speed'] for style in build_styles if style['laser_speed'] is not None]
            if speeds and len(set(speeds)) == 1:
                recommendations.append("Single speed value used across all styles - consider optimizing for different geometries")
            
            # Check for usage patterns
            if style_usage and 'most_used_style' in style_usage and 'least_used_style' in style_usage:
                most_used = style_usage['most_used_style']
                least_used = style_usage['least_used_style']
                
                if most_used and least_used:
                    usage_ratio = most_used[1] / least_used[1] if least_used[1] > 0 else float('inf')
                    if usage_ratio > 10:
                        recommendations.append(f"High usage imbalance detected - style {most_used[0]} used {usage_ratio:.1f}x more than style {least_used[0]}")
            
            # Check for naming consistency
            named_styles = [style for style in build_styles if style['name']]
            if len(named_styles) < len(build_styles) * 0.5:
                recommendations.append("Many build styles lack descriptive names - consider adding names for better process management")
            
        except Exception as e:
            logger.warning(f"Error generating style recommendations: {e}")
            recommendations.append("Error generating recommendations")
        
        return recommendations
    
    def _calculate_energy_density(self, build_style: Any) -> Optional[float]:
        """Calculate energy density for a build style."""
        try:
            if hasattr(build_style, 'laserPower') and hasattr(build_style, 'laserSpeed'):
                power = build_style.laserPower
                speed = build_style.laserSpeed
                if power is not None and speed is not None and speed > 0:
                    return power / speed
        except Exception as e:
            logger.warning(f"Error calculating energy density: {e}")
        return None
    
    def _calculate_scan_speed(self, build_style: Any) -> Optional[float]:
        """Calculate scan speed for a build style."""
        try:
            if hasattr(build_style, 'laserSpeed'):
                return build_style.laserSpeed
        except Exception as e:
            logger.warning(f"Error calculating scan speed: {e}")
        return None
    
    def _calculate_duty_cycle(self, build_style: Any) -> Optional[float]:
        """Calculate duty cycle for a build style."""
        try:
            if hasattr(build_style, 'pointExposureTime') and hasattr(build_style, 'pointDelay'):
                exposure_time = build_style.pointExposureTime
                delay_time = build_style.pointDelay
                if exposure_time is not None and delay_time is not None:
                    total_time = exposure_time + delay_time
                    if total_time > 0:
                        return exposure_time / total_time
        except Exception as e:
            logger.warning(f"Error calculating duty cycle: {e}")
        return None
    
    def _calculate_parameter_statistics(self, values: List[float]) -> Dict[str, Any]:
        """Calculate statistics for a parameter."""
        if not values:
            return {}
        
        try:
            return {
                'count': len(values),
                'min': min(values),
                'max': max(values),
                'mean': np.mean(values),
                'median': np.median(values),
                'std': np.std(values),
                'range': max(values) - min(values),
                'unique_values': len(set(values))
            }
        except Exception as e:
            logger.warning(f"Error calculating parameter statistics: {e}")
            return {}
    
    def _calculate_diversity_score(self, build_styles: List[Dict[str, Any]]) -> float:
        """Calculate diversity score for build styles."""
        try:
            if len(build_styles) <= 1:
                return 0.0
            
            # Calculate diversity based on parameter variation
            powers = [style['laser_power'] for style in build_styles if style['laser_power'] is not None]
            speeds = [style['laser_speed'] for style in build_styles if style['laser_speed'] is not None]
            focuses = [style['laser_focus'] for style in build_styles if style['laser_focus'] is not None]
            
            diversity_score = 0.0
            
            if powers:
                diversity_score += len(set(powers)) / len(powers)
            if speeds:
                diversity_score += len(set(speeds)) / len(speeds)
            if focuses:
                diversity_score += len(set(focuses)) / len(focuses)
            
            return diversity_score / 3.0  # Normalize by number of parameters
        except Exception as e:
            logger.warning(f"Error calculating diversity score: {e}")
            return 0.0
    
    def _find_most_efficient_style(self, build_styles: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Find the most efficient build style."""
        try:
            if not build_styles:
                return None
            
            # Calculate efficiency based on energy density and speed
            best_style = None
            best_efficiency = float('inf')
            
            for style in build_styles:
                energy_density = style.get('energy_density')
                scan_speed = style.get('scan_speed')
                
                if energy_density is not None and scan_speed is not None:
                    # Lower energy density and higher speed = more efficient
                    efficiency = energy_density / scan_speed if scan_speed > 0 else float('inf')
                    if efficiency < best_efficiency:
                        best_efficiency = efficiency
                        best_style = style
            
            return best_style
        except Exception as e:
            logger.warning(f"Error finding most efficient style: {e}")
            return None
    
    def _find_least_efficient_style(self, build_styles: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Find the least efficient build style."""
        try:
            if not build_styles:
                return None
            
            # Calculate efficiency based on energy density and speed
            worst_style = None
            worst_efficiency = 0.0
            
            for style in build_styles:
                energy_density = style.get('energy_density')
                scan_speed = style.get('scan_speed')
                
                if energy_density is not None and scan_speed is not None:
                    # Lower energy density and higher speed = more efficient
                    efficiency = energy_density / scan_speed if scan_speed > 0 else float('inf')
                    if efficiency > worst_efficiency:
                        worst_efficiency = efficiency
                        worst_style = style
            
            return worst_style
        except Exception as e:
            logger.warning(f"Error finding least efficient style: {e}")
            return None
    
    def _calculate_efficiency_variation(self, build_styles: List[Dict[str, Any]]) -> float:
        """Calculate efficiency variation across build styles."""
        try:
            if len(build_styles) <= 1:
                return 0.0
            
            efficiencies = []
            for style in build_styles:
                energy_density = style.get('energy_density')
                scan_speed = style.get('scan_speed')
                
                if energy_density is not None and scan_speed is not None:
                    efficiency = energy_density / scan_speed if scan_speed > 0 else float('inf')
                    efficiencies.append(efficiency)
            
            if efficiencies:
                return np.std(efficiencies) / np.mean(efficiencies) if np.mean(efficiencies) > 0 else 0.0
            
            return 0.0
        except Exception as e:
            logger.warning(f"Error calculating efficiency variation: {e}")
            return 0.0
    
    def _identify_optimization_opportunities(self, build_styles: List[Dict[str, Any]], style_usage: Dict[str, Any]) -> List[str]:
        """Identify optimization opportunities."""
        opportunities = []
        
        try:
            if not build_styles or not style_usage:
                return ["No data available for optimization analysis"]
            
            # Check for unused styles
            total_usage = style_usage.get('total_usage_count', {})
            for style in build_styles:
                style_id = style.get('style_id')
                if style_id and style_id not in total_usage:
                    opportunities.append(f"Build style {style_id} is defined but never used - consider removing")
            
            # Check for overused styles
            if total_usage:
                most_used = max(total_usage.items(), key=lambda x: x[1])
                total_geometries = sum(total_usage.values())
                usage_percentage = (most_used[1] / total_geometries) * 100
                
                if usage_percentage > 80:
                    opportunities.append(f"Build style {most_used[0]} used for {usage_percentage:.1f}% of geometries - consider creating specialized styles")
            
            # Check for parameter optimization
            powers = [style['laser_power'] for style in build_styles if style['laser_power'] is not None]
            if powers and len(set(powers)) > 1:
                power_range = max(powers) - min(powers)
                if power_range > np.mean(powers) * 0.5:
                    opportunities.append("Large power variation detected - consider optimizing power settings for different geometries")
            
        except Exception as e:
            logger.warning(f"Error identifying optimization opportunities: {e}")
            opportunities.append("Error identifying optimization opportunities")
        
        return opportunities
    
    def get_extractor_info(self) -> Dict[str, Any]:
        """Get extractor information."""
        return {
            'name': 'Build Style Extractor',
            'libslm_available': self.libslm_available,
            'description': 'Extracts and analyzes build style metadata from PBF-LB/M build files using libSLM'
        }
