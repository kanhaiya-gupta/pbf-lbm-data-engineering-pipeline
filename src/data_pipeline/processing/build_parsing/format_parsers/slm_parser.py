"""
SLM Parser for PBF-LB/M Build Files.

This module provides parsing capabilities for SLM build files (.slm),
leveraging libSLM's SLM reader when available.
"""

from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import logging

from ..base_parser import BaseBuildParser
from ....external import LIBSLM_AVAILABLE, slmsol

logger = logging.getLogger(__name__)


class SLMParser(BaseBuildParser):
    """
    Parser for SLM build files (.slm).
    
    This parser leverages libSLM's SLM reader to extract build data
    from SLM machine build files.
    """
    
    def __init__(self):
        """Initialize the SLM parser."""
        super().__init__()
        self.supported_formats = ['.slm']
        self.parser_name = "SLM Parser"
        
        # Check libSLM availability
        if LIBSLM_AVAILABLE:
            try:
                self.slm_reader = slmsol.Reader()
                self.libslm_available = True
                logger.info("SLM parser initialized with libSLM support")
            except Exception as e:
                logger.warning(f"libSLM slmsol module not available: {e}")
                self.slm_reader = None
                self.libslm_available = False
        else:
            self.slm_reader = None
            self.libslm_available = False
            logger.warning("SLM parser initialized without libSLM support")
    
    def can_parse(self, file_path: Union[str, Path]) -> bool:
        """Check if this parser can handle the given file."""
        file_path = Path(file_path)
        return file_path.suffix.lower() in self.supported_formats
    
    def parse_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Parse an SLM build file.
        
        Args:
            file_path: Path to the SLM build file
            
        Returns:
            Dictionary containing parsed build data
        """
        file_path = Path(file_path)
        
        if not self.can_parse(file_path):
            raise ValueError(f"SLM parser cannot handle file: {file_path}")
        
        if not self.libslm_available:
            raise RuntimeError("libSLM SLM module not available")
        
        try:
            logger.info(f"Parsing SLM file: {file_path}")
            
            # Use libSLM SLM reader
            self.slm_reader.setFilePath(str(file_path))
            build_data = self.slm_reader.parse()
            
            # Extract build information
            result = {
                'file_path': str(file_path),
                'file_format': file_path.suffix.lower(),
                'parser': self.parser_name,
                'build_data': build_data,
                'metadata': self._extract_metadata(build_data),
                'layers': self._extract_layers(build_data),
                'parameters': self._extract_parameters(build_data)
            }
            
            logger.info(f"Successfully parsed SLM file: {file_path}")
            return result
            
        except Exception as e:
            logger.error(f"Error parsing SLM file {file_path}: {e}")
            raise
    
    def _extract_metadata(self, build_data: Any) -> Dict[str, Any]:
        """Extract metadata from SLM build data."""
        metadata = {}
        
        try:
            # Extract basic metadata from reader
            if hasattr(self.slm_reader, 'getLayerThickness'):
                metadata['layer_thickness'] = self.slm_reader.getLayerThickness()
            
            if hasattr(self.slm_reader, 'getFileSize'):
                metadata['file_size'] = self.slm_reader.getFileSize()
            
            if hasattr(self.slm_reader, 'getZUnit'):
                metadata['z_unit'] = self.slm_reader.getZUnit()
            
            # Add layer count
            if hasattr(self.slm_reader, 'layers'):
                metadata['layer_count'] = len(self.slm_reader.layers)
            
            # Extract basic metadata from build_data
            if hasattr(build_data, 'metadata'):
                metadata.update(build_data.metadata)
            
            # Extract build dimensions
            if hasattr(build_data, 'dimensions'):
                metadata['dimensions'] = build_data.dimensions
            
            # Extract build volume
            if hasattr(build_data, 'build_volume'):
                metadata['build_volume'] = build_data.build_volume
            
            # Extract machine information
            if hasattr(build_data, 'machine_info'):
                metadata['machine_info'] = build_data.machine_info
            
        except Exception as e:
            logger.warning(f"Error extracting metadata: {e}")
        
        return metadata
    
    def _extract_layers(self, build_data: Any) -> List[Dict[str, Any]]:
        """Extract layer information from SLM build data."""
        layers = []
        
        try:
            # Extract from reader object (where the actual data is)
            if hasattr(self.slm_reader, 'layers') and self.slm_reader.layers:
                for i, layer in enumerate(self.slm_reader.layers):
                    layer_info = {
                        'layer_index': i,
                        'z_height': getattr(layer, 'z', None),  # libSLM uses 'z' not 'z_height'
                        'thickness': getattr(layer, 'thickness', None),
                        'hatch_count': 0,  # Will be calculated from hatch geometry
                        'contour_count': 0,  # Will be calculated from contour geometry
                        'point_count': 0,  # Will be calculated from point geometry
                        'layer_id': getattr(layer, 'layerId', i),
                        'is_loaded': layer.isLoaded() if hasattr(layer, 'isLoaded') and callable(layer.isLoaded) else False
                    }
                    layers.append(layer_info)
            
        except Exception as e:
            logger.warning(f"Error extracting layers: {e}")
        
        return layers
    
    def _extract_parameters(self, build_data: Any) -> Dict[str, Any]:
        """Extract process parameters from SLM build data."""
        parameters = {}
        
        try:
            # Extract global parameters from reader
            if hasattr(self.slm_reader, 'getLayerThickness'):
                parameters['layer_thickness'] = self.slm_reader.getLayerThickness()
            
            if hasattr(self.slm_reader, 'getFileSize'):
                parameters['file_size'] = self.slm_reader.getFileSize()
            
            if hasattr(self.slm_reader, 'getZUnit'):
                parameters['z_unit'] = self.slm_reader.getZUnit()
            
            # Extract layer-specific parameters from reader
            if hasattr(self.slm_reader, 'layers') and self.slm_reader.layers:
                layer_params = {}
                for i, layer in enumerate(self.slm_reader.layers):
                    layer_info = {
                        'layer_id': getattr(layer, 'layerId', i),
                        'z_position': getattr(layer, 'z', None),
                        'is_loaded': layer.isLoaded() if hasattr(layer, 'isLoaded') and callable(layer.isLoaded) else False
                    }
                    layer_params[f'layer_{i}'] = layer_info
                if layer_params:
                    parameters['layer_parameters'] = layer_params
            
        except Exception as e:
            logger.warning(f"Error extracting parameters: {e}")
        
        return parameters
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported file formats."""
        return self.supported_formats.copy()
    
    def is_format_supported(self, file_extension: str) -> bool:
        """Check if a file format is supported by this parser."""
        return file_extension.lower() in self.supported_formats
    
    def get_parser_info(self) -> Dict[str, Any]:
        """Get parser information."""
        return {
            'name': self.parser_name,
            'supported_formats': self.supported_formats,
            'libslm_available': self.libslm_available,
            'description': 'Parser for SLM build files (.slm) using libSLM'
        }
