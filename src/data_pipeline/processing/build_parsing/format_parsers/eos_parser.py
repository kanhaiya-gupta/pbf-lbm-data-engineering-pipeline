"""
EOS Parser for PBF-LB/M Build Files.

This module provides parsing capabilities for EOS build files (.sli, .cli),
leveraging libSLM's EOS reader when available.
"""

from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import logging

from ..base_parser import BaseBuildParser
from ....external import LIBSLM_AVAILABLE

logger = logging.getLogger(__name__)


class EOSParser(BaseBuildParser):
    """
    Parser for EOS build files (.sli, .cli).
    
    This parser leverages libSLM's EOS reader to extract build data
    from EOS machine build files.
    """
    
    def __init__(self):
        """Initialize the EOS parser."""
        super().__init__()
        self.supported_formats = ['.sli', '.cli']
        self.parser_name = "EOS Parser"
        
        # Check libSLM availability
        if LIBSLM_AVAILABLE:
            try:
                from libSLM import eos
                self.eos_reader = eos.Reader()
                self.libslm_available = True
                logger.info("EOS parser initialized with libSLM support")
            except ImportError as e:
                logger.warning(f"libSLM EOS module not available: {e}")
                self.eos_reader = None
                self.libslm_available = False
        else:
            self.eos_reader = None
            self.libslm_available = False
            logger.warning("EOS parser initialized without libSLM support")
    
    def can_parse(self, file_path: Union[str, Path]) -> bool:
        """Check if this parser can handle the given file."""
        file_path = Path(file_path)
        return file_path.suffix.lower() in self.supported_formats
    
    def parse_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Parse an EOS build file.
        
        Args:
            file_path: Path to the EOS build file
            
        Returns:
            Dictionary containing parsed build data
        """
        file_path = Path(file_path)
        
        if not self.can_parse(file_path):
            raise ValueError(f"EOS parser cannot handle file: {file_path}")
        
        if not self.libslm_available:
            raise RuntimeError("libSLM EOS module not available")
        
        try:
            logger.info(f"Parsing EOS file: {file_path}")
            
            # Use libSLM EOS reader
            build_data = self.eos_reader.read(str(file_path))
            
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
            
            logger.info(f"Successfully parsed EOS file: {file_path}")
            return result
            
        except Exception as e:
            logger.error(f"Error parsing EOS file {file_path}: {e}")
            raise
    
    def _extract_metadata(self, build_data: Any) -> Dict[str, Any]:
        """Extract metadata from EOS build data."""
        metadata = {}
        
        try:
            # Extract basic metadata
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
        """Extract layer information from EOS build data."""
        layers = []
        
        try:
            if hasattr(build_data, 'layers'):
                for i, layer in enumerate(build_data.layers):
                    layer_info = {
                        'layer_index': i,
                        'z_height': getattr(layer, 'z_height', None),
                        'thickness': getattr(layer, 'thickness', None),
                        'hatch_count': getattr(layer, 'hatch_count', 0),
                        'contour_count': getattr(layer, 'contour_count', 0),
                        'point_count': getattr(layer, 'point_count', 0)
                    }
                    layers.append(layer_info)
            
        except Exception as e:
            logger.warning(f"Error extracting layers: {e}")
        
        return layers
    
    def _extract_parameters(self, build_data: Any) -> Dict[str, Any]:
        """Extract process parameters from EOS build data."""
        parameters = {}
        
        try:
            # Extract global parameters
            if hasattr(build_data, 'parameters'):
                parameters.update(build_data.parameters)
            
            # Extract layer-specific parameters
            if hasattr(build_data, 'layers'):
                layer_params = {}
                for i, layer in enumerate(build_data.layers):
                    if hasattr(layer, 'parameters'):
                        layer_params[f'layer_{i}'] = layer.parameters
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
            'description': 'Parser for EOS build files (.sli, .cli) using libSLM'
        }
