"""
Geometry Type Extractor for PBF-LB/M Build Files.

This module provides geometry type extraction capabilities for PBF-LB/M build files,
leveraging libSLM for accessing geometry type information and scan strategy analysis.
"""

from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path
import logging
import numpy as np

from ....external import LIBSLM_AVAILABLE

logger = logging.getLogger(__name__)


class GeometryTypeExtractor:
    """
    Extractor for geometry type information from PBF-LB/M build files.
    
    This extractor analyzes geometry type distribution across different scan strategies
    and provides scan strategy analysis using libSLM data.
    """
    
    def __init__(self):
        """Initialize the geometry type extractor."""
        self.libslm_available = LIBSLM_AVAILABLE
        
        if not self.libslm_available:
            logger.warning("libSLM not available - geometry type analysis will be limited")
        else:
            logger.info("Geometry type extractor initialized with libSLM support")
    
    def extract_geometry_type_data(self, build_data: Any) -> Dict[str, Any]:
        """
        Extract geometry type data from build file data.
        
        Args:
            build_data: Parsed build file data from libSLM
            
        Returns:
            Dictionary containing geometry type analysis results
        """
        try:
            logger.info("Extracting geometry type data from build file")
            
            geometry_data = {
                'geometry_distribution': self._extract_geometry_distribution(build_data),
                'layer_geometry_analysis': self._extract_layer_geometry_analysis(build_data),
                'geometry_statistics': self._calculate_geometry_statistics(build_data),
                'scan_strategy_analysis': self._analyze_scan_strategy(build_data),
                'geometry_efficiency': self._analyze_geometry_efficiency(build_data),
                'strategy_recommendations': self._generate_strategy_recommendations(build_data)
            }
            
            logger.info("Successfully extracted geometry type data")
            return geometry_data
            
        except Exception as e:
            logger.error(f"Error extracting geometry type data: {e}")
            raise
    
    def _extract_geometry_distribution(self, build_data: Any) -> Dict[str, Any]:
        """Extract geometry type distribution across the build."""
        distribution = {}
        
        try:
            geometry_counts = {'hatch': 0, 'contour': 0, 'point': 0}
            geometry_by_layer = {}
            geometry_by_style = {}
            
            if hasattr(build_data, 'layers'):
                for layer_idx, layer in enumerate(build_data.layers):
                    layer_geometry = {'hatch': 0, 'contour': 0, 'point': 0}
                    
                    # Count hatch geometries
                    if hasattr(layer, 'getHatchGeometry'):
                        hatch_geometries = layer.getHatchGeometry()
                        layer_geometry['hatch'] = len(hatch_geometries)
                        geometry_counts['hatch'] += len(hatch_geometries)
                        
                        # Count by build style
                        for hatch_geom in hatch_geometries:
                            style_id = hatch_geom.bid
                            if style_id not in geometry_by_style:
                                geometry_by_style[style_id] = {'hatch': 0, 'contour': 0, 'point': 0}
                            geometry_by_style[style_id]['hatch'] += 1
                    
                    # Count contour geometries
                    if hasattr(layer, 'getContourGeometry'):
                        contour_geometries = layer.getContourGeometry()
                        layer_geometry['contour'] = len(contour_geometries)
                        geometry_counts['contour'] += len(contour_geometries)
                        
                        # Count by build style
                        for contour_geom in contour_geometries:
                            style_id = contour_geom.bid
                            if style_id not in geometry_by_style:
                                geometry_by_style[style_id] = {'hatch': 0, 'contour': 0, 'point': 0}
                            geometry_by_style[style_id]['contour'] += 1
                    
                    # Count point geometries
                    if hasattr(layer, 'getPointGeometry'):
                        point_geometries = layer.getPointGeometry()
                        layer_geometry['point'] = len(point_geometries)
                        geometry_counts['point'] += len(point_geometries)
                        
                        # Count by build style
                        for point_geom in point_geometries:
                            style_id = point_geom.bid
                            if style_id not in geometry_by_style:
                                geometry_by_style[style_id] = {'hatch': 0, 'contour': 0, 'point': 0}
                            geometry_by_style[style_id]['point'] += 1
                    
                    geometry_by_layer[layer_idx] = layer_geometry
            
            total_geometries = sum(geometry_counts.values())
            
            distribution = {
                'total_geometries': total_geometries,
                'geometry_counts': geometry_counts,
                'geometry_percentages': {
                    geom_type: (count / total_geometries * 100) if total_geometries > 0 else 0
                    for geom_type, count in geometry_counts.items()
                },
                'geometry_by_layer': geometry_by_layer,
                'geometry_by_style': geometry_by_style,
                'dominant_geometry_type': max(geometry_counts.items(), key=lambda x: x[1])[0] if total_geometries > 0 else None
            }
        except Exception as e:
            logger.warning(f"Error extracting geometry distribution: {e}")
        
        return distribution
    
    def _extract_layer_geometry_analysis(self, build_data: Any) -> List[Dict[str, Any]]:
        """Extract layer-specific geometry analysis."""
        layer_analysis = []
        
        try:
            if hasattr(build_data, 'layers'):
                for layer_idx, layer in enumerate(build_data.layers):
                    layer_info = {
                        'layer_index': layer_idx,
                        'geometry_counts': {'hatch': 0, 'contour': 0, 'point': 0},
                        'geometry_details': [],
                        'layer_statistics': {}
                    }
                    
                    # Analyze hatch geometries
                    if hasattr(layer, 'getHatchGeometry'):
                        hatch_geometries = layer.getHatchGeometry()
                        layer_info['geometry_counts']['hatch'] = len(hatch_geometries)
                        
                        for hatch_idx, hatch_geom in enumerate(hatch_geometries):
                            coords = hatch_geom.coords if hasattr(hatch_geom, 'coords') else None
                            layer_info['geometry_details'].append({
                                'geometry_type': 'hatch',
                                'geometry_index': hatch_idx,
                                'build_style_id': hatch_geom.bid,
                                'model_id': hatch_geom.mid,
                                'num_segments': coords.shape[0] // 2 if coords is not None else 0,
                                'coordinates': coords.tolist() if coords is not None else None
                            })
                    
                    # Analyze contour geometries
                    if hasattr(layer, 'getContourGeometry'):
                        contour_geometries = layer.getContourGeometry()
                        layer_info['geometry_counts']['contour'] = len(contour_geometries)
                        
                        for contour_idx, contour_geom in enumerate(contour_geometries):
                            coords = contour_geom.coords if hasattr(contour_geom, 'coords') else None
                            layer_info['geometry_details'].append({
                                'geometry_type': 'contour',
                                'geometry_index': contour_idx,
                                'build_style_id': contour_geom.bid,
                                'model_id': contour_geom.mid,
                                'num_points': coords.shape[0] if coords is not None else 0,
                                'coordinates': coords.tolist() if coords is not None else None
                            })
                    
                    # Analyze point geometries
                    if hasattr(layer, 'getPointGeometry'):
                        point_geometries = layer.getPointGeometry()
                        layer_info['geometry_counts']['point'] = len(point_geometries)
                        
                        for point_idx, point_geom in enumerate(point_geometries):
                            coords = point_geom.coords if hasattr(point_geom, 'coords') else None
                            layer_info['geometry_details'].append({
                                'geometry_type': 'point',
                                'geometry_index': point_idx,
                                'build_style_id': point_geom.bid,
                                'model_id': point_geom.mid,
                                'num_points': coords.shape[0] if coords is not None else 0,
                                'coordinates': coords.tolist() if coords is not None else None
                            })
                    
                    # Calculate layer statistics
                    total_geometries = sum(layer_info['geometry_counts'].values())
                    if total_geometries > 0:
                        layer_info['layer_statistics'] = {
                            'total_geometries': total_geometries,
                            'geometry_diversity': len([count for count in layer_info['geometry_counts'].values() if count > 0]),
                            'dominant_geometry': max(layer_info['geometry_counts'].items(), key=lambda x: x[1])[0],
                            'geometry_balance': self._calculate_geometry_balance(layer_info['geometry_counts'])
                        }
                    
                    layer_analysis.append(layer_info)
        except Exception as e:
            logger.warning(f"Error extracting layer geometry analysis: {e}")
        
        return layer_analysis
    
    def _calculate_geometry_statistics(self, build_data: Any) -> Dict[str, Any]:
        """Calculate geometry statistics across the build."""
        statistics = {}
        
        try:
            geometry_distribution = self._extract_geometry_distribution(build_data)
            layer_analysis = self._extract_layer_geometry_analysis(build_data)
            
            if geometry_distribution and layer_analysis:
                # Calculate overall statistics
                total_geometries = geometry_distribution['total_geometries']
                geometry_counts = geometry_distribution['geometry_counts']
                
                # Calculate layer statistics
                layer_geometry_counts = [layer['geometry_counts'] for layer in layer_analysis]
                layer_totals = [sum(layer['geometry_counts'].values()) for layer in layer_analysis]
                
                statistics = {
                    'total_geometries': total_geometries,
                    'total_layers': len(layer_analysis),
                    'average_geometries_per_layer': np.mean(layer_totals) if layer_totals else 0,
                    'geometry_distribution': geometry_counts,
                    'geometry_percentages': geometry_distribution['geometry_percentages'],
                    'layer_statistics': {
                        'min_geometries_per_layer': min(layer_totals) if layer_totals else 0,
                        'max_geometries_per_layer': max(layer_totals) if layer_totals else 0,
                        'std_geometries_per_layer': np.std(layer_totals) if layer_totals else 0,
                        'layers_with_hatch': len([layer for layer in layer_analysis if layer['geometry_counts']['hatch'] > 0]),
                        'layers_with_contour': len([layer for layer in layer_analysis if layer['geometry_counts']['contour'] > 0]),
                        'layers_with_point': len([layer for layer in layer_analysis if layer['geometry_counts']['point'] > 0])
                    },
                    'geometry_efficiency_metrics': self._calculate_geometry_efficiency_metrics(layer_analysis)
                }
        except Exception as e:
            logger.warning(f"Error calculating geometry statistics: {e}")
        
        return statistics
    
    def _analyze_scan_strategy(self, build_data: Any) -> Dict[str, Any]:
        """Analyze scan strategy based on geometry types."""
        scan_strategy = {}
        
        try:
            geometry_distribution = self._extract_geometry_distribution(build_data)
            layer_analysis = self._extract_layer_geometry_analysis(build_data)
            
            if geometry_distribution and layer_analysis:
                # Analyze strategy patterns
                strategy_patterns = self._identify_strategy_patterns(layer_analysis)
                geometry_ratios = self._calculate_geometry_ratios(geometry_distribution)
                
                scan_strategy = {
                    'strategy_type': self._classify_scan_strategy(geometry_distribution),
                    'strategy_patterns': strategy_patterns,
                    'geometry_ratios': geometry_ratios,
                    'strategy_consistency': self._calculate_strategy_consistency(layer_analysis),
                    'strategy_efficiency': self._calculate_strategy_efficiency(geometry_distribution, layer_analysis)
                }
        except Exception as e:
            logger.warning(f"Error analyzing scan strategy: {e}")
        
        return scan_strategy
    
    def _analyze_geometry_efficiency(self, build_data: Any) -> Dict[str, Any]:
        """Analyze geometry efficiency."""
        efficiency = {}
        
        try:
            geometry_distribution = self._extract_geometry_distribution(build_data)
            layer_analysis = self._extract_layer_geometry_analysis(build_data)
            
            if geometry_distribution and layer_analysis:
                # Calculate efficiency metrics
                efficiency = {
                    'geometry_utilization': self._calculate_geometry_utilization(geometry_distribution),
                    'layer_efficiency': self._calculate_layer_efficiency(layer_analysis),
                    'strategy_optimization': self._calculate_strategy_optimization(geometry_distribution),
                    'efficiency_recommendations': self._generate_efficiency_recommendations(geometry_distribution, layer_analysis)
                }
        except Exception as e:
            logger.warning(f"Error analyzing geometry efficiency: {e}")
        
        return efficiency
    
    def _generate_strategy_recommendations(self, build_data: Any) -> List[str]:
        """Generate scan strategy recommendations."""
        recommendations = []
        
        try:
            geometry_distribution = self._extract_geometry_distribution(build_data)
            layer_analysis = self._extract_layer_geometry_analysis(build_data)
            
            if not geometry_distribution or not layer_analysis:
                return ["No geometry data available for analysis"]
            
            geometry_counts = geometry_distribution['geometry_counts']
            total_geometries = geometry_distribution['total_geometries']
            
            if total_geometries == 0:
                return ["No geometries found in build file"]
            
            # Check for geometry balance
            hatch_percentage = geometry_counts['hatch'] / total_geometries * 100
            contour_percentage = geometry_counts['contour'] / total_geometries * 100
            point_percentage = geometry_counts['point'] / total_geometries * 100
            
            if hatch_percentage > 90:
                recommendations.append("Build heavily relies on hatch patterns - consider adding contour geometries for better surface quality")
            elif contour_percentage > 50:
                recommendations.append("High contour usage detected - consider optimizing hatch patterns for better build speed")
            elif point_percentage > 30:
                recommendations.append("High point geometry usage - consider using hatch patterns for better build efficiency")
            
            # Check for layer consistency
            layer_geometry_counts = [layer['geometry_counts'] for layer in layer_analysis]
            if len(layer_geometry_counts) > 1:
                hatch_consistency = self._calculate_consistency([layer['hatch'] for layer in layer_geometry_counts])
                if hatch_consistency < 0.7:
                    recommendations.append("Inconsistent hatch pattern usage across layers - consider standardizing scan strategy")
            
            # Check for geometry diversity
            layers_with_multiple_types = len([layer for layer in layer_analysis if sum(1 for count in layer['geometry_counts'].values() if count > 0) > 1])
            if layers_with_multiple_types < len(layer_analysis) * 0.3:
                recommendations.append("Low geometry diversity across layers - consider using multiple geometry types for better quality")
            
        except Exception as e:
            logger.warning(f"Error generating strategy recommendations: {e}")
            recommendations.append("Error generating recommendations")
        
        return recommendations
    
    def _calculate_geometry_balance(self, geometry_counts: Dict[str, int]) -> float:
        """Calculate geometry balance score."""
        try:
            total = sum(geometry_counts.values())
            if total == 0:
                return 0.0
            
            # Calculate entropy for balance
            proportions = [count / total for count in geometry_counts.values() if count > 0]
            if len(proportions) <= 1:
                return 0.0
            
            entropy = -sum(p * np.log2(p) for p in proportions if p > 0)
            max_entropy = np.log2(len(proportions))
            
            return entropy / max_entropy if max_entropy > 0 else 0.0
        except Exception as e:
            logger.warning(f"Error calculating geometry balance: {e}")
            return 0.0
    
    def _calculate_geometry_efficiency_metrics(self, layer_analysis: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate geometry efficiency metrics."""
        try:
            if not layer_analysis:
                return {}
            
            # Calculate efficiency metrics
            total_geometries = sum(sum(layer['geometry_counts'].values()) for layer in layer_analysis)
            total_layers = len(layer_analysis)
            
            # Calculate geometry density per layer
            geometry_densities = [sum(layer['geometry_counts'].values()) for layer in layer_analysis]
            
            # Calculate geometry diversity per layer
            geometry_diversities = [layer['layer_statistics'].get('geometry_diversity', 0) for layer in layer_analysis]
            
            return {
                'average_geometry_density': np.mean(geometry_densities) if geometry_densities else 0,
                'geometry_density_variation': np.std(geometry_densities) if geometry_densities else 0,
                'average_geometry_diversity': np.mean(geometry_diversities) if geometry_diversities else 0,
                'geometry_diversity_variation': np.std(geometry_diversities) if geometry_diversities else 0,
                'efficiency_score': self._calculate_overall_efficiency_score(geometry_densities, geometry_diversities)
            }
        except Exception as e:
            logger.warning(f"Error calculating geometry efficiency metrics: {e}")
            return {}
    
    def _identify_strategy_patterns(self, layer_analysis: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Identify scan strategy patterns."""
        try:
            if not layer_analysis:
                return {}
            
            patterns = {
                'consistent_strategy': True,
                'hatch_dominant': True,
                'contour_dominant': True,
                'point_dominant': True,
                'mixed_strategy': False
            }
            
            # Analyze patterns across layers
            for layer in layer_analysis:
                geometry_counts = layer['geometry_counts']
                total = sum(geometry_counts.values())
                
                if total > 0:
                    hatch_ratio = geometry_counts['hatch'] / total
                    contour_ratio = geometry_counts['contour'] / total
                    point_ratio = geometry_counts['point'] / total
                    
                    # Check for dominant patterns
                    if hatch_ratio < 0.8:
                        patterns['hatch_dominant'] = False
                    if contour_ratio < 0.8:
                        patterns['contour_dominant'] = False
                    if point_ratio < 0.8:
                        patterns['point_dominant'] = False
                    
                    # Check for mixed strategy
                    if sum(1 for ratio in [hatch_ratio, contour_ratio, point_ratio] if ratio > 0.1) > 1:
                        patterns['mixed_strategy'] = True
            
            return patterns
        except Exception as e:
            logger.warning(f"Error identifying strategy patterns: {e}")
            return {}
    
    def _calculate_geometry_ratios(self, geometry_distribution: Dict[str, Any]) -> Dict[str, float]:
        """Calculate geometry ratios."""
        try:
            geometry_counts = geometry_distribution.get('geometry_counts', {})
            total = sum(geometry_counts.values())
            
            if total == 0:
                return {'hatch': 0, 'contour': 0, 'point': 0}
            
            return {
                'hatch': geometry_counts['hatch'] / total,
                'contour': geometry_counts['contour'] / total,
                'point': geometry_counts['point'] / total
            }
        except Exception as e:
            logger.warning(f"Error calculating geometry ratios: {e}")
            return {'hatch': 0, 'contour': 0, 'point': 0}
    
    def _classify_scan_strategy(self, geometry_distribution: Dict[str, Any]) -> str:
        """Classify the scan strategy type."""
        try:
            geometry_percentages = geometry_distribution.get('geometry_percentages', {})
            
            hatch_pct = geometry_percentages.get('hatch', 0)
            contour_pct = geometry_percentages.get('contour', 0)
            point_pct = geometry_percentages.get('point', 0)
            
            if hatch_pct > 80:
                return 'hatch_dominant'
            elif contour_pct > 50:
                return 'contour_dominant'
            elif point_pct > 30:
                return 'point_dominant'
            elif max(hatch_pct, contour_pct, point_pct) < 60:
                return 'mixed_strategy'
            else:
                return 'balanced_strategy'
        except Exception as e:
            logger.warning(f"Error classifying scan strategy: {e}")
            return 'unknown_strategy'
    
    def _calculate_strategy_consistency(self, layer_analysis: List[Dict[str, Any]]) -> float:
        """Calculate strategy consistency across layers."""
        try:
            if len(layer_analysis) <= 1:
                return 1.0
            
            # Calculate consistency based on geometry ratios
            layer_ratios = []
            for layer in layer_analysis:
                geometry_counts = layer['geometry_counts']
                total = sum(geometry_counts.values())
                if total > 0:
                    layer_ratios.append({
                        'hatch': geometry_counts['hatch'] / total,
                        'contour': geometry_counts['contour'] / total,
                        'point': geometry_counts['point'] / total
                    })
            
            if len(layer_ratios) <= 1:
                return 1.0
            
            # Calculate standard deviation of ratios
            hatch_ratios = [ratio['hatch'] for ratio in layer_ratios]
            contour_ratios = [ratio['contour'] for ratio in layer_ratios]
            point_ratios = [ratio['point'] for ratio in layer_ratios]
            
            consistency = 1.0 - (np.std(hatch_ratios) + np.std(contour_ratios) + np.std(point_ratios)) / 3.0
            return max(0.0, min(1.0, consistency))
        except Exception as e:
            logger.warning(f"Error calculating strategy consistency: {e}")
            return 0.0
    
    def _calculate_strategy_efficiency(self, geometry_distribution: Dict[str, Any], layer_analysis: List[Dict[str, Any]]) -> float:
        """Calculate strategy efficiency score."""
        try:
            if not geometry_distribution or not layer_analysis:
                return 0.0
            
            # Calculate efficiency based on geometry distribution and layer utilization
            geometry_percentages = geometry_distribution.get('geometry_percentages', {})
            
            # Hatch is generally more efficient for bulk material
            hatch_efficiency = geometry_percentages.get('hatch', 0) / 100.0
            
            # Contour is good for surface quality
            contour_efficiency = geometry_percentages.get('contour', 0) / 100.0 * 0.8
            
            # Point is less efficient for bulk material
            point_efficiency = geometry_percentages.get('point', 0) / 100.0 * 0.6
            
            # Calculate layer utilization efficiency
            layer_totals = [sum(layer['geometry_counts'].values()) for layer in layer_analysis]
            layer_utilization = 1.0 - (np.std(layer_totals) / np.mean(layer_totals)) if layer_totals and np.mean(layer_totals) > 0 else 0.0
            
            # Combine efficiency metrics
            efficiency = (hatch_efficiency + contour_efficiency + point_efficiency) * 0.7 + layer_utilization * 0.3
            return max(0.0, min(1.0, efficiency))
        except Exception as e:
            logger.warning(f"Error calculating strategy efficiency: {e}")
            return 0.0
    
    def _calculate_geometry_utilization(self, geometry_distribution: Dict[str, Any]) -> float:
        """Calculate geometry utilization efficiency."""
        try:
            geometry_percentages = geometry_distribution.get('geometry_percentages', {})
            
            # Optimal utilization: 70% hatch, 20% contour, 10% point
            optimal_hatch = 70.0
            optimal_contour = 20.0
            optimal_point = 10.0
            
            actual_hatch = geometry_percentages.get('hatch', 0)
            actual_contour = geometry_percentages.get('contour', 0)
            actual_point = geometry_percentages.get('point', 0)
            
            # Calculate deviation from optimal
            hatch_deviation = abs(actual_hatch - optimal_hatch) / optimal_hatch
            contour_deviation = abs(actual_contour - optimal_contour) / optimal_contour
            point_deviation = abs(actual_point - optimal_point) / optimal_point
            
            # Calculate utilization score (lower deviation = higher score)
            utilization = 1.0 - (hatch_deviation + contour_deviation + point_deviation) / 3.0
            return max(0.0, min(1.0, utilization))
        except Exception as e:
            logger.warning(f"Error calculating geometry utilization: {e}")
            return 0.0
    
    def _calculate_layer_efficiency(self, layer_analysis: List[Dict[str, Any]]) -> float:
        """Calculate layer efficiency."""
        try:
            if not layer_analysis:
                return 0.0
            
            # Calculate efficiency based on layer utilization and geometry diversity
            layer_totals = [sum(layer['geometry_counts'].values()) for layer in layer_analysis]
            layer_diversities = [layer['layer_statistics'].get('geometry_diversity', 0) for layer in layer_analysis]
            
            # Layer utilization efficiency
            utilization_efficiency = 1.0 - (np.std(layer_totals) / np.mean(layer_totals)) if layer_totals and np.mean(layer_totals) > 0 else 0.0
            
            # Geometry diversity efficiency
            diversity_efficiency = np.mean(layer_diversities) / 3.0  # Normalize by max diversity (3 types)
            
            # Combine efficiency metrics
            efficiency = utilization_efficiency * 0.6 + diversity_efficiency * 0.4
            return max(0.0, min(1.0, efficiency))
        except Exception as e:
            logger.warning(f"Error calculating layer efficiency: {e}")
            return 0.0
    
    def _calculate_strategy_optimization(self, geometry_distribution: Dict[str, Any]) -> float:
        """Calculate strategy optimization score."""
        try:
            if not geometry_distribution:
                return 0.0
            
            # Calculate optimization based on geometry balance and efficiency
            geometry_balance = self._calculate_geometry_balance(geometry_distribution.get('geometry_counts', {}))
            geometry_utilization = self._calculate_geometry_utilization(geometry_distribution)
            
            # Combine optimization metrics
            optimization = geometry_balance * 0.4 + geometry_utilization * 0.6
            return max(0.0, min(1.0, optimization))
        except Exception as e:
            logger.warning(f"Error calculating strategy optimization: {e}")
            return 0.0
    
    def _generate_efficiency_recommendations(self, geometry_distribution: Dict[str, Any], layer_analysis: List[Dict[str, Any]]) -> List[str]:
        """Generate efficiency recommendations."""
        recommendations = []
        
        try:
            if not geometry_distribution or not layer_analysis:
                return ["No data available for efficiency analysis"]
            
            geometry_percentages = geometry_distribution.get('geometry_percentages', {})
            
            # Check for geometry balance
            hatch_pct = geometry_percentages.get('hatch', 0)
            contour_pct = geometry_percentages.get('contour', 0)
            point_pct = geometry_percentages.get('point', 0)
            
            if hatch_pct < 50:
                recommendations.append("Low hatch usage - consider increasing hatch patterns for better build efficiency")
            elif hatch_pct > 90:
                recommendations.append("Very high hatch usage - consider adding contour patterns for better surface quality")
            
            if contour_pct < 10:
                recommendations.append("Low contour usage - consider adding contour patterns for better surface finish")
            elif contour_pct > 40:
                recommendations.append("High contour usage - consider optimizing for build speed")
            
            if point_pct > 20:
                recommendations.append("High point usage - consider using hatch patterns for better efficiency")
            
            # Check for layer efficiency
            layer_totals = [sum(layer['geometry_counts'].values()) for layer in layer_analysis]
            if layer_totals:
                layer_variation = np.std(layer_totals) / np.mean(layer_totals) if np.mean(layer_totals) > 0 else 0
                if layer_variation > 0.5:
                    recommendations.append("High layer geometry variation - consider standardizing scan strategy across layers")
            
        except Exception as e:
            logger.warning(f"Error generating efficiency recommendations: {e}")
            recommendations.append("Error generating efficiency recommendations")
        
        return recommendations
    
    def _calculate_consistency(self, values: List[float]) -> float:
        """Calculate consistency score for a list of values."""
        try:
            if len(values) <= 1:
                return 1.0
            
            mean_val = np.mean(values)
            if mean_val == 0:
                return 1.0 if all(v == 0 for v in values) else 0.0
            
            std_val = np.std(values)
            consistency = 1.0 - (std_val / mean_val)
            return max(0.0, min(1.0, consistency))
        except Exception as e:
            logger.warning(f"Error calculating consistency: {e}")
            return 0.0
    
    def _calculate_overall_efficiency_score(self, geometry_densities: List[int], geometry_diversities: List[int]) -> float:
        """Calculate overall efficiency score."""
        try:
            if not geometry_densities or not geometry_diversities:
                return 0.0
            
            # Normalize density (higher is better, up to a point)
            density_score = min(1.0, np.mean(geometry_densities) / 100.0)  # Assume 100 is optimal
            
            # Normalize diversity (higher is better, up to 3)
            diversity_score = np.mean(geometry_diversities) / 3.0
            
            # Combine scores
            efficiency = density_score * 0.6 + diversity_score * 0.4
            return max(0.0, min(1.0, efficiency))
        except Exception as e:
            logger.warning(f"Error calculating overall efficiency score: {e}")
            return 0.0
    
    def get_extractor_info(self) -> Dict[str, Any]:
        """Get extractor information."""
        return {
            'name': 'Geometry Type Extractor',
            'libslm_available': self.libslm_available,
            'description': 'Extracts and analyzes geometry type distribution from PBF-LB/M build files using libSLM'
        }
