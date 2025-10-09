"""
Energy Extractor for EOS/SLI Build Files.

This module provides energy parameter extraction capabilities for EOS/SLI build files.
Note: EOS/SLI files contain only geometry data, not process parameters like power, velocity, etc.
"""

from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path
import logging
import numpy as np

from .....external import LIBSLM_AVAILABLE

logger = logging.getLogger(__name__)


class EnergyExtractor:
    """
    Extractor for energy parameters from EOS/SLI build files.
    
    Note: EOS/SLI files contain only geometry data (coordinates, layer info) but do not
    contain process parameters like power, velocity, exposure time, etc. These parameters
    are typically defined in separate EOS job files or machine settings.
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
        Extract energy data from EOS/SLI build file data.
        
        Args:
            build_data: Parsed build file data
            
        Returns:
            Dictionary containing energy analysis results (mostly empty for EOS files)
        """
        try:
            logger.info("Extracting energy data from EOS/SLI build file")
            
            energy_data = {
                'global_energy': self._extract_global_energy(build_data),
                'layer_energy': self._extract_layer_energy(build_data),
                'geometry_energy': self._extract_geometry_energy(build_data),
                'energy_statistics': self._calculate_energy_statistics(build_data),
                'energy_distribution': self._analyze_energy_distribution(build_data),
                'spatial_energy_map': self._create_spatial_energy_map(build_data),
                'format_note': 'EOS/SLI files contain only geometry data, no process parameters'
            }
            
            logger.info("Successfully extracted energy data from EOS/SLI file")
            return energy_data
            
        except Exception as e:
            logger.error(f"Error extracting energy data: {e}")
            raise
    
    def _extract_global_energy(self, build_data: Any) -> Dict[str, Any]:
        """Extract global energy settings (not available in EOS format)."""
        return {
            'note': 'EOS/SLI files do not contain global energy settings',
            'available_parameters': ['scale_factor', 'z_unit', 'layer_count']
        }
    
    def _extract_layer_energy(self, build_data: Any) -> List[Dict[str, Any]]:
        """Extract layer-specific energy settings (not available in EOS format)."""
        layer_energy = []
        
        try:
            reader = build_data.get('reader_object')
            if reader and hasattr(reader, 'layers'):
                for i, layer in enumerate(reader.layers):
                    layer_info = {
                        'layer_index': i,
                        'z_height': getattr(layer, 'z', None),
                        'is_loaded': layer.isLoaded() if hasattr(layer, 'isLoaded') else None,
                        'energy_settings': {
                            'note': 'EOS/SLI files do not contain layer energy settings'
                        }
                    }
                    layer_energy.append(layer_info)
        except Exception as e:
            logger.warning(f"Error extracting layer energy: {e}")
        
        return layer_energy
    
    def _extract_geometry_energy(self, build_data: Any) -> List[Dict[str, Any]]:
        """Extract geometry-specific energy settings (not available in EOS format)."""
        return [{
            'note': 'EOS/SLI files do not contain geometry-specific energy settings',
            'available_data': 'Only geometry coordinates and build style IDs (bid)'
        }]
    
    def _calculate_energy_statistics(self, build_data: Any) -> Dict[str, Any]:
        """Calculate energy statistics (not applicable for EOS format)."""
        return {
            'note': 'EOS/SLI files do not contain energy data for statistical analysis',
            'count': 0,
            'mean': None,
            'std': None,
            'min': None,
            'max': None,
            'median': None,
            'q25': None,
            'q75': None
        }
    
    def _analyze_energy_distribution(self, build_data: Any) -> Dict[str, Any]:
        """Analyze energy distribution patterns (not applicable for EOS format)."""
        return {
            'note': 'EOS/SLI files do not contain energy data for distribution analysis',
            'layer_variation': None,
            'geometry_distribution': {}
        }
    
    def _create_spatial_energy_map(self, build_data: Any) -> Dict[str, Any]:
        """Create spatial mapping for geometry data (no energy values)."""
        spatial_map = {
            'voxel_resolution': None,
            'geometry_voxels': [],
            'coordinate_system': 'build_coordinates',
            'bounds': None,
            'note': 'EOS/SLI files contain only geometry coordinates, no energy data'
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
            
            # Collect all geometry points for spatial mapping
            geometry_points = []
            
            # From hatches
            if hasattr(reader, 'layers'):
                layers = reader.layers
                for layer_idx, layer in enumerate(layers):
                    try:
                        hatch_geometries = layer.getHatchGeometry()
                        for hatch_idx, hatch_geom in enumerate(hatch_geometries):
                            coords = hatch_geom.coords if hasattr(hatch_geom, 'coords') else None
                            if coords is not None:
                                for i in range(0, len(coords), 2):
                                    if i + 1 < len(coords):
                                        geometry_points.append({
                                            'x': coords[i],
                                            'y': coords[i + 1],
                                            'z': layer_idx * 0.05,  # Assuming 50μm layer thickness
                                            'geometry_type': 'hatch',
                                            'build_style_id': getattr(hatch_geom, 'bid', None)
                                        })
                    except Exception as layer_error:
                        logger.warning(f"Error accessing layer {layer_idx}: {layer_error}")
                        continue
            
            spatial_map['geometry_voxels'] = geometry_points
            spatial_map['total_points'] = len(geometry_points)
            
            # Calculate suggested voxel resolution based on geometry density
            if geometry_points:
                x_coords = [p['x'] for p in geometry_points]
                y_coords = [p['y'] for p in geometry_points]
                z_coords = [p['z'] for p in geometry_points]
                
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
            logger.warning(f"Error creating spatial energy map: {e}")
            spatial_map['error'] = str(e)
        
        return spatial_map
    
    def get_extractor_info(self) -> Dict[str, Any]:
        """Get extractor information."""
        return {
            'name': 'EOS/SLI Energy Extractor',
            'libslm_available': self.libslm_available,
            'description': 'Extracts geometry data from EOS/SLI build files (no process parameters available)',
            'note': 'EOS/SLI files contain only geometry coordinates, not energy/power/process parameters'
        }