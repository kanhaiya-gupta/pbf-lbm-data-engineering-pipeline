"""
CLI Parser for PBF-LB/M Build Files.

This module provides parsing capabilities for CLI build files (.cli),
leveraging libSLM's CLI reader when available.
"""

from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import logging

from ..base_parser import BaseBuildParser
from ....external import LIBSLM_AVAILABLE, cli

logger = logging.getLogger(__name__)


class CLIParser(BaseBuildParser):
    """
    Parser for CLI build files (.cli).
    
    This parser leverages libSLM's CLI reader to extract build data
    from CLI machine build files.
    """
    
    def __init__(self):
        """Initialize the CLI parser."""
        super().__init__()
        self.supported_formats = ['.cli']
        self.parser_name = "CLI Parser"
        
        # Check libSLM availability
        if LIBSLM_AVAILABLE:
            try:
                self.cli_reader = cli.Reader()
                self.libslm_available = True
                logger.info("CLI parser initialized with libSLM support")
            except Exception as e:
                logger.warning(f"libSLM CLI module not available: {e}")
                self.cli_reader = None
                self.libslm_available = False
        else:
            self.cli_reader = None
            self.libslm_available = False
            logger.warning("CLI parser initialized without libSLM support")
    
    def can_parse(self, file_path: Union[str, Path]) -> bool:
        """Check if this parser can handle the given file."""
        file_path = Path(file_path)
        return file_path.suffix.lower() in self.supported_formats
    
    def parse_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Parse a CLI build file.
        
        Args:
            file_path: Path to the CLI build file
            
        Returns:
            Dictionary containing parsed build data
        """
        file_path = Path(file_path)
        
        if not self.can_parse(file_path):
            raise ValueError(f"CLI parser cannot handle file: {file_path}")
        
        if not self.libslm_available:
            raise RuntimeError("libSLM CLI module not available")
        
        try:
            logger.info(f"Parsing CLI file: {file_path}")
            
            # Use libSLM CLI reader with correct methods
            self.cli_reader.setFilePath(str(file_path))
            parse_result = self.cli_reader.parse()
            
            # For CLI files, parse result 1 means success
            if parse_result < 0:
                logger.warning(f"CLI parse failed with result: {parse_result}")
            elif parse_result > 0:
                logger.info(f"CLI parse completed with result: {parse_result}")
            
            # Extract build information using the reader object
            result = {
                'file_path': str(file_path),
                'file_format': file_path.suffix.lower(),
                'parser': self.parser_name,
                'build_data': self.cli_reader,  # Store the reader object
                'metadata': self._extract_metadata(self.cli_reader),
                'layers': self._extract_layers(self.cli_reader),
                'parameters': self._extract_parameters(self.cli_reader)
            }
            
            logger.info(f"Successfully parsed CLI file: {file_path}")
            return result
            
        except Exception as e:
            logger.error(f"Error parsing CLI file {file_path}: {e}")
            raise
    
    def _extract_metadata(self, reader) -> Dict[str, Any]:
        """Extract metadata from the CLI file."""
        try:
            metadata = {
                'file_format': 'cli',
                'layer_thickness': reader.getLayerThickness(),
                'z_unit': reader.getZUnit() if hasattr(reader, 'getZUnit') else None,
                'is_binary': reader.isBinaryFormat() if hasattr(reader, 'isBinaryFormat') else None,
            }
            return metadata
        except Exception as e:
            logger.warning(f"Error extracting metadata: {e}")
            return {}
    
    def _extract_layers(self, reader) -> List[Dict[str, Any]]:
        """Extract layer information from the CLI file."""
        try:
            layers = reader.getLayers()
            layer_data = []
            
            for layer in layers:
                layer_info = {
                    'z': layer.z / 1000.0 if hasattr(layer, 'z') and layer.z else 0.0,  # Convert from microns to mm
                    'layer_id': layer.layerId if hasattr(layer, 'layerId') else None,
                    'geometry_count': len(layer.geometry) if hasattr(layer, 'geometry') else 0,
                }
                layer_data.append(layer_info)
            
            return layer_data
        except Exception as e:
            logger.warning(f"Error extracting layers: {e}")
            return []
    
    def _extract_parameters(self, reader) -> Dict[str, Any]:
        """Extract build parameters from the CLI file."""
        try:
            params = {}
            
            # Try to extract parameters if available
            if hasattr(reader, 'getParameters'):
                params = reader.getParameters()
            
            return params
        except Exception as e:
            logger.warning(f"Error extracting parameters: {e}")
            return {}

