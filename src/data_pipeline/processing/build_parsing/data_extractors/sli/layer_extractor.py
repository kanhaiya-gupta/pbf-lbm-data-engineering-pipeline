"""
Layer Extractor for EOS/SLI Build Files.

This module provides layer parameter extraction capabilities for EOS/SLI build files.
Note: EOS/SLI files contain only geometry data, not process parameters like power, velocity, etc.
"""

from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path
import logging
import numpy as np

from .....external import LIBSLM_AVAILABLE

logger = logging.getLogger(__name__)


class LayerExtractor:
    """
    Extractor for layer parameters from EOS/SLI build files.
    
    Note: EOS/SLI files contain only geometry data (coordinates, layer info) but do not
    contain process parameters like power, velocity, exposure time, etc. These parameters
    are typically defined in separate EOS job files or machine settings.
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
        Extract layer data from EOS/SLI build file data.
        
        Args:
            build_data: Parsed build file data
            
        Returns:
            Dictionary containing layer analysis results (geometry data only for EOS files)
        """
        try:
            logger.info("Extracting layer data from EOS/SLI build file")
            
            layer_data = {
                'layer_count': self._extract_layer_count(build_data),
                'layer_info': self._extract_layer_info(build_data),
                'layer_geometry': self._extract_layer_geometry(build_data),
                'layer_statistics': self._calculate_layer_statistics(build_data),
                'layer_distribution': self._analyze_layer_distribution(build_data),
                'spatial_layer_map': self._create_spatial_layer_map(build_data),
                'format_note': 'EOS/SLI files contain only geometry data, no process parameters'
            }
            
            logger.info("Successfully extracted layer data from EOS/SLI file")
            return layer_data
            
        except Exception as e:
            logger.error(f"Error extracting layer data: {e}")
            raise
    
    def _extract_layer_count(self, build_data: Any) -> int:
        """Extract total number of layers."""
        try:
            reader = build_data.get('reader_object')
            if reader and hasattr(reader, 'layers'):
                return len(reader.layers)
            return 0
        except Exception as e:
            logger.warning(f"Error extracting layer count: {e}")
            return 0
    
    def _extract_layer_info(self, build_data: Any) -> List[Dict[str, Any]]:
        """Extract detailed layer information."""
        layer_info = []
        
        try:
            reader = build_data.get('reader_object')
            if not reader:
                logger.warning("No reader object available for layer extraction")
                return layer_info
            
            if hasattr(reader, 'layers'):
                layers = reader.layers
                for layer_idx, layer in enumerate(layers):
                    try:
                        # Get layer geometry counts
                        hatch_count = len(layer.getHatchGeometry())
                        contour_count = len(layer.getContourGeometry())
                        point_count = len(layer.getPointsGeometry())
                        
                        layer_data = {
                            'layer_index': layer_idx,
                            'z_height': getattr(layer, 'z', None),
                            'is_loaded': layer.isLoaded() if hasattr(layer, 'isLoaded') else None,
                            'file_position': getattr(layer, 'layerFilePosition', None),
                            'geometry_counts': {
                                'hatch': hatch_count,
                                'contour': contour_count,
                                'point': point_count,
                                'total': hatch_count + contour_count + point_count
                            },
                            'note': 'EOS/SLI files contain only geometry data, no process parameters'
                        }
                        layer_info.append(layer_data)
                    except Exception as layer_error:
                        logger.warning(f"Error accessing layer {layer_idx}: {layer_error}")
                        continue
        
        except Exception as e:
            logger.warning(f"Error extracting layer info: {e}")
        
        return layer_info
    
    def _extract_layer_geometry(self, build_data: Any) -> Dict[str, Any]:
        """Extract layer geometry summary."""
        geometry_summary = {
            'total_layers': 0,
            'layers_with_geometry': 0,
            'total_hatch_geometries': 0,
            'total_contour_geometries': 0,
            'total_point_geometries': 0,
            'geometry_by_layer': {}
        }
        
        try:
            reader = build_data.get('reader_object')
            if not reader:
                return geometry_summary
            
            if hasattr(reader, 'layers'):
                layers = reader.layers
                geometry_summary['total_layers'] = len(layers)
                
                for layer_idx, layer in enumerate(layers):
                    try:
                        hatch_count = len(layer.getHatchGeometry())
                        contour_count = len(layer.getContourGeometry())
                        point_count = len(layer.getPointsGeometry())
                        
                        total_geometry = hatch_count + contour_count + point_count
                        if total_geometry > 0:
                            geometry_summary['layers_with_geometry'] += 1
                        
                        geometry_summary['total_hatch_geometries'] += hatch_count
                        geometry_summary['total_contour_geometries'] += contour_count
                        geometry_summary['total_point_geometries'] += point_count
                        
                        geometry_summary['geometry_by_layer'][layer_idx] = {
                            'hatch': hatch_count,
                            'contour': contour_count,
                            'point': point_count,
                            'total': total_geometry
                        }
                    except Exception as layer_error:
                        logger.warning(f"Error analyzing layer {layer_idx}: {layer_error}")
                        continue
        
        except Exception as e:
            logger.warning(f"Error extracting layer geometry: {e}")
            geometry_summary['error'] = str(e)
        
        return geometry_summary
    
    def _calculate_layer_statistics(self, build_data: Any) -> Dict[str, Any]:
        """Calculate layer statistics."""
        statistics = {
            'total_layers': 0,
            'layers_with_geometry': 0,
            'average_geometry_per_layer': 0.0,
            'max_geometry_per_layer': 0,
            'min_geometry_per_layer': 0,
            'geometry_distribution': {},
            'z_height_range': {'min': None, 'max': None}
        }
        
        try:
            reader = build_data.get('reader_object')
            if not reader:
                return statistics
            
            if hasattr(reader, 'layers'):
                layers = reader.layers
                statistics['total_layers'] = len(layers)
                
                geometry_counts = []
                z_heights = []
                
                for layer in layers:
                    try:
                        # Count geometry
                        hatch_count = len(layer.getHatchGeometry())
                        contour_count = len(layer.getContourGeometry())
                        point_count = len(layer.getPointsGeometry())
                        total_geometry = hatch_count + contour_count + point_count
                        
                        if total_geometry > 0:
                            statistics['layers_with_geometry'] += 1
                            geometry_counts.append(total_geometry)
                        
                        # Collect z heights
                        z_height = getattr(layer, 'z', None)
                        if z_height is not None:
                            z_heights.append(z_height)
                    except Exception as layer_error:
                        logger.warning(f"Error analyzing layer: {layer_error}")
                        continue
                
                # Calculate statistics
                if geometry_counts:
                    statistics['average_geometry_per_layer'] = sum(geometry_counts) / len(geometry_counts)
                    statistics['max_geometry_per_layer'] = max(geometry_counts)
                    statistics['min_geometry_per_layer'] = min(geometry_counts)
                
                if z_heights:
                    statistics['z_height_range']['min'] = min(z_heights)
                    statistics['z_height_range']['max'] = max(z_heights)
                
                # Geometry distribution
                if geometry_counts:
                    statistics['geometry_distribution'] = {
                        'layers_with_hatch': sum(1 for layer in layers if len(layer.getHatchGeometry()) > 0),
                        'layers_with_contour': sum(1 for layer in layers if len(layer.getContourGeometry()) > 0),
                        'layers_with_point': sum(1 for layer in layers if len(layer.getPointsGeometry()) > 0)
                    }
        
        except Exception as e:
            logger.warning(f"Error calculating layer statistics: {e}")
            statistics['error'] = str(e)
        
        return statistics
    
    def _analyze_layer_distribution(self, build_data: Any) -> Dict[str, Any]:
        """Analyze layer distribution patterns."""
        distribution = {
            'layer_density': {},
            'geometry_concentration': {},
            'z_height_distribution': {},
            'note': 'EOS/SLI files contain only geometry data, no process parameters'
        }
        
        try:
            reader = build_data.get('reader_object')
            if not reader:
                return distribution
            
            if hasattr(reader, 'layers'):
                layers = reader.layers
                
                # Analyze layer density (geometry per layer)
                geometry_per_layer = []
                for layer in layers:
                    try:
                        hatch_count = len(layer.getHatchGeometry())
                        contour_count = len(layer.getContourGeometry())
                        point_count = len(layer.getPointsGeometry())
                        total_geometry = hatch_count + contour_count + point_count
                        geometry_per_layer.append(total_geometry)
                    except Exception:
                        geometry_per_layer.append(0)
                
                if geometry_per_layer:
                    distribution['layer_density'] = {
                        'mean': np.mean(geometry_per_layer),
                        'std': np.std(geometry_per_layer),
                        'coefficient_of_variation': np.std(geometry_per_layer) / np.mean(geometry_per_layer) if np.mean(geometry_per_layer) > 0 else 0
                    }
                
                # Analyze z height distribution
                z_heights = []
                for layer in layers:
                    z_height = getattr(layer, 'z', None)
                    if z_height is not None:
                        z_heights.append(z_height)
                
                if z_heights:
                    distribution['z_height_distribution'] = {
                        'range': max(z_heights) - min(z_heights),
                        'mean': np.mean(z_heights),
                        'std': np.std(z_heights)
                    }
        
        except Exception as e:
            logger.warning(f"Error analyzing layer distribution: {e}")
            distribution['error'] = str(e)
        
        return distribution
    
    def _create_spatial_layer_map(self, build_data: Any) -> Dict[str, Any]:
        """Create spatial mapping for layer data."""
        spatial_map = {
            'voxel_resolution': None,
            'layer_voxels': [],
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
            
            # Collect layer information for spatial mapping
            layer_voxels = []
            
            if hasattr(reader, 'layers'):
                layers = reader.layers
                for layer_idx, layer in enumerate(layers):
                    try:
                        # Get layer geometry counts
                        hatch_count = len(layer.getHatchGeometry())
                        contour_count = len(layer.getContourGeometry())
                        point_count = len(layer.getPointsGeometry())
                        
                        layer_voxel = {
                            'layer_index': layer_idx,
                            'z_height': getattr(layer, 'z', None),
                            'geometry_counts': {
                                'hatch': hatch_count,
                                'contour': contour_count,
                                'point': point_count,
                                'total': hatch_count + contour_count + point_count
                            },
                            'is_loaded': layer.isLoaded() if hasattr(layer, 'isLoaded') else None
                        }
                        layer_voxels.append(layer_voxel)
                    except Exception as layer_error:
                        logger.warning(f"Error processing layer {layer_idx}: {layer_error}")
                        continue
            
            spatial_map['layer_voxels'] = layer_voxels
            spatial_map['total_layers'] = len(layer_voxels)
            
            # Calculate bounds from z heights
            z_heights = [voxel['z_height'] for voxel in layer_voxels if voxel['z_height'] is not None]
            if z_heights:
                spatial_map['bounds'] = {
                    'z_min': min(z_heights),
                    'z_max': max(z_heights),
                    'z_range': max(z_heights) - min(z_heights)
                }
        
        except Exception as e:
            logger.warning(f"Error creating spatial layer map: {e}")
            spatial_map['error'] = str(e)
        
        return spatial_map
    
    def get_extractor_info(self) -> Dict[str, Any]:
        """Get extractor information."""
        return {
            'name': 'EOS/SLI Layer Extractor',
            'libslm_available': self.libslm_available,
            'description': 'Extracts layer data from EOS/SLI build files (geometry data only)',
            'note': 'EOS/SLI files contain only geometry coordinates, not power/velocity/process parameters'
        }