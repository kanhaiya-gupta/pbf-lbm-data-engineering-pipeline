"""
ILT Parser for PBF-LB/M Build Files.

This module provides parsing capabilities for ILT build files (.ilt),
which are ZIP archives containing CLI files and parameter files.
"""

from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import logging
import zipfile
import tempfile
import shutil
import configparser
import re

from ..base_parser import BaseBuildParser
from ....external import LIBSLM_AVAILABLE, cli

logger = logging.getLogger(__name__)


class ILTParser(BaseBuildParser):
    """
    Parser for ILT build files (.ilt).
    
    ILT files are ZIP archives containing:
    - Multiple CLI files (geometry data)
    - modelsection_param.txt (process parameters)
    
    This parser extracts both geometry and process parameters.
    """
    
    def __init__(self):
        """Initialize the ILT parser."""
        super().__init__()
        self.supported_formats = ['.ilt']
        self.parser_name = "ILT Parser"
        
        # Check libSLM availability
        if LIBSLM_AVAILABLE:
            try:
                self.cli_reader = cli.Reader()
                self.libslm_available = True
                logger.info("ILT parser initialized with libSLM CLI support")
            except Exception as e:
                logger.warning(f"libSLM CLI module not available: {e}")
                self.cli_reader = None
                self.libslm_available = False
        else:
            self.cli_reader = None
            self.libslm_available = False
            logger.warning("ILT parser initialized without libSLM support")
    
    def can_parse(self, file_path: Union[str, Path]) -> bool:
        """Check if this parser can handle the given file."""
        file_path = Path(file_path)
        return file_path.suffix.lower() in self.supported_formats
    
    def parse_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Parse an ILT build file.
        
        Args:
            file_path: Path to the ILT build file
            
        Returns:
            Dictionary containing parsed build data with both geometry and parameters
        """
        file_path = Path(file_path)
        
        if not self.can_parse(file_path):
            raise ValueError(f"ILT parser cannot handle file: {file_path}")
        
        if not self.libslm_available:
            raise RuntimeError("libSLM CLI module not available")
        
        try:
            logger.info(f"Parsing ILT file: {file_path}")
            
            # Extract ILT contents to temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Extract ZIP contents
                self._extract_ilt_contents(file_path, temp_path)
                
                # Find CLI files and parameter file
                cli_files = list(temp_path.glob("*.cli"))
                param_file = temp_path / "modelsection_param.txt"
                
                if not cli_files:
                    raise ValueError("No CLI files found in ILT archive")
                
                if not param_file.exists():
                    logger.warning("No modelsection_param.txt found in ILT archive")
                    param_file = None
                
                # Parse parameter file
                parameters = self._parse_parameter_file(param_file) if param_file else {}
                
                # Parse CLI files and associate with parameters
                cli_data = self._parse_cli_files(cli_files, parameters)
                
                # Combine all data
                result = {
                    'file_path': str(file_path),
                    'file_format': file_path.suffix.lower(),
                    'parser': self.parser_name,
                    'build_data': cli_data,
                    'metadata': self._extract_metadata(cli_data),
                    'layers': self._extract_layers(cli_data),
                    'parameters': parameters,
                    'cli_files': [str(f.name) for f in cli_files]
                }
                
                logger.info(f"Successfully parsed ILT file: {file_path}")
                return result
                
        except Exception as e:
            logger.error(f"Error parsing ILT file {file_path}: {e}")
            raise
    
    def _extract_ilt_contents(self, ilt_path: Path, temp_path: Path) -> None:
        """Extract ILT ZIP contents to temporary directory."""
        try:
            with zipfile.ZipFile(ilt_path, 'r') as zip_ref:
                zip_ref.extractall(temp_path)
                logger.info(f"Extracted {len(zip_ref.namelist())} files from ILT archive")
        except Exception as e:
            raise ValueError(f"Failed to extract ILT archive: {e}")
    
    def _parse_parameter_file(self, param_file: Path) -> Dict[str, Any]:
        """Parse the modelsection_param.txt file."""
        try:
            parameters = {}
            
            with open(param_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse INI-style sections
            config = configparser.ConfigParser()
            config.read_string(content)
            
            for section_name in config.sections():
                # Extract CLI filename from section name
                cli_filename = section_name.strip('[]')
                
                # Extract parameters for this CLI file
                section_params = {}
                for key, value in config[section_name].items():
                    # Convert values to appropriate types
                    try:
                        if '.' in value:
                            section_params[key] = float(value)
                        else:
                            section_params[key] = int(value)
                    except ValueError:
                        section_params[key] = value
                
                parameters[cli_filename] = section_params
            
            logger.info(f"Parsed parameters for {len(parameters)} CLI files")
            return parameters
            
        except Exception as e:
            logger.warning(f"Error parsing parameter file: {e}")
            return {}
    
    def _parse_cli_files(self, cli_files: List[Path], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Parse all CLI files and associate with parameters."""
        cli_data = {}
        
        for cli_file in cli_files:
            try:
                logger.info(f"Parsing CLI file: {cli_file.name}")
                
                # Use libSLM CLI reader
                self.cli_reader.setFilePath(str(cli_file))
                parse_result = self.cli_reader.parse()
                
                if parse_result < 0:
                    logger.warning(f"CLI parse failed for {cli_file.name}: {parse_result}")
                    continue
                
                # Get parameters for this CLI file
                cli_params = parameters.get(cli_file.name, {})
                
                # Store CLI data with associated parameters
                cli_data[cli_file.name] = {
                    'reader': self.cli_reader,
                    'parameters': cli_params,
                    'metadata': self._extract_metadata(self.cli_reader),
                    'layers': self._extract_layers(self.cli_reader)
                }
                
            except Exception as e:
                logger.error(f"Error parsing CLI file {cli_file.name}: {e}")
                continue
        
        return cli_data
    
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
