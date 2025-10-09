"""
EOS Parser for PBF-LB/M Build Files.

This module provides parsing capabilities for EOS build files (.sli, .cli),
leveraging libSLM's EOS reader when available.
"""

from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import logging

from ..base_parser import BaseBuildParser
from ....external import LIBSLM_AVAILABLE, eos

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
                self.eos_reader = eos.Reader()
                self.libslm_available = True
                logger.info("EOS parser initialized with libSLM support")
            except Exception as e:
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
            
            # Use libSLM EOS reader with correct methods
            self.eos_reader.setFilePath(str(file_path))
            parse_result = self.eos_reader.parse()
            
            # For EOS files, parse result 1 often means success with warnings
            if parse_result < 0:
                logger.warning(f"EOS parse failed with result: {parse_result}")
            elif parse_result > 0:
                logger.info(f"EOS parse completed with result: {parse_result} (may include warnings)")
            
            # Extract build information using the reader object
            result = {
                'file_path': str(file_path),
                'file_format': file_path.suffix.lower(),
                'parser': self.parser_name,
                'build_data': self.eos_reader,  # Store the reader object
                'metadata': self._extract_metadata(self.eos_reader),
                'layers': self._extract_layers(self.eos_reader),
                'parameters': self._extract_parameters(self.eos_reader)
            }
            
            logger.info(f"Successfully parsed EOS file: {file_path}")
            return result
            
        except Exception as e:
            logger.error(f"Error parsing EOS file {file_path}: {e}")
            raise
    
    def _extract_metadata(self, reader: Any) -> Dict[str, Any]:
        """Extract metadata from EOS reader object."""
        metadata = {}
        
        try:
            # Extract basic metadata from EOS reader
            if hasattr(reader, 'getModels'):
                models = reader.getModels()
                if models and len(models) > 0:
                    model = models[0]
                    metadata['model_count'] = len(models)
                    if hasattr(model, 'id'):
                        metadata['model_id'] = model.id
            
            # Extract layer information
            if hasattr(reader, 'getLayers'):
                layers = reader.getLayers()
                metadata['layer_count'] = len(layers) if layers else 0
            elif hasattr(reader, 'layers'):
                metadata['layer_count'] = len(reader.layers) if reader.layers else 0
            
            # Extract file information
            if hasattr(reader, 'getFilePath'):
                metadata['file_path'] = reader.getFilePath()
            
            # Add EOS-specific metadata
            metadata['file_type'] = 'EOS CLI/SLI'
            metadata['parser'] = 'EOS Parser'
            
        except Exception as e:
            logger.warning(f"Error extracting EOS metadata: {e}")
        
        return metadata
    
    def _extract_layers(self, reader: Any) -> List[Dict[str, Any]]:
        """Extract layer information from EOS reader object."""
        layers = []
        
        try:
            # Get layers from EOS reader
            if hasattr(reader, 'getLayers'):
                reader_layers = reader.getLayers()
            elif hasattr(reader, 'layers'):
                reader_layers = reader.layers
            else:
                reader_layers = []
            
            if reader_layers:
                for i, layer in enumerate(reader_layers):
                    # Try to get layer thickness from the reader or use default
                    layer_thickness = 0.05  # Default EOS layer thickness
                    
                    layer_info = {
                        'layer_index': i,
                        'z_height': i * layer_thickness,  # Calculate Z height
                        'thickness': layer_thickness,
                        'hatch_count': 0,  # Will be populated by data extractors
                        'contour_count': 0,  # Will be populated by data extractors
                        'point_count': 0,  # Will be populated by data extractors
                        'is_loaded': getattr(layer, 'isLoaded', lambda: False)()
                    }
                    layers.append(layer_info)
            
        except Exception as e:
            logger.warning(f"Error extracting EOS layers: {e}")
        
        return layers
    
    def _extract_parameters(self, reader: Any) -> Dict[str, Any]:
        """Extract basic parameters from EOS reader object."""
        parameters = {}
        
        try:
            # Extract basic file parameters from EOS reader
            if hasattr(reader, 'getLayerThickness'):
                layer_thickness = reader.getLayerThickness()
                parameters['layer_thickness'] = layer_thickness
            else:
                parameters['layer_thickness'] = 0.05  # Default EOS layer thickness
            
            if hasattr(reader, 'getZUnit'):
                z_unit = reader.getZUnit()
                parameters['z_unit'] = z_unit
            
            if hasattr(reader, 'scaleFactor'):
                scale_factor = reader.scaleFactor
                parameters['scale_factor'] = scale_factor
            
            # Add EOS-specific metadata
            parameters['file_type'] = 'EOS CLI/SLI'
            parameters['parser'] = 'EOS Parser'
            
        except Exception as e:
            logger.warning(f"Error extracting EOS parameters: {e}")
        
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
