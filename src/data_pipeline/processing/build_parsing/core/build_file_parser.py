"""
Main Build File Parser - Orchestrator

This module provides the main orchestrator for build file parsing that coordinates
between different format parsers and data extractors, leveraging libSLM and PySLM
for world-class PBF-LB/M build file processing.
"""

from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import logging
from datetime import datetime

from ..base_parser import BaseBuildParser
from ....external import LIBSLM_AVAILABLE, PYSLM_AVAILABLE
from .format_detector import FormatDetector
from .metadata_extractor import MetadataExtractor

# Import format parsers
from ..format_parsers.eos_parser import EOSParser
from ..format_parsers.mtt_parser import MTTParser
from ..format_parsers.realizer_parser import RealizerParser
from ..format_parsers.slm_parser import SLMParser
from ..format_parsers.generic_parser import GenericParser

# Import data extractors
from ..data_extractors.power_extractor import PowerExtractor
from ..data_extractors.velocity_extractor import VelocityExtractor
from ..data_extractors.path_extractor import PathExtractor
from ..data_extractors.energy_extractor import EnergyExtractor
from ..data_extractors.layer_extractor import LayerExtractor

logger = logging.getLogger(__name__)


class BuildFileParser(BaseBuildParser):
    """
    Main orchestrator for PBF-LB/M build file parsing.
    
    This class coordinates between different format parsers and data extractors,
    leveraging libSLM for file parsing and PySLM for advanced analysis. It provides
    a unified interface for parsing various PBF-LB/M machine build file formats.
    """
    
    def __init__(self):
        """Initialize the main build file parser."""
        super().__init__()
        
        # Initialize components
        self.format_detector = FormatDetector()
        self.metadata_extractor = MetadataExtractor()
        
        # Initialize format parsers
        self.format_parsers = {
            '.sli': EOSParser(),
            '.cli': EOSParser(),
            '.mtt': MTTParser(),
            '.rea': RealizerParser(),
            '.slm': SLMParser(),
        }
        
        # Initialize data extractors
        self.data_extractors = {
            'power': PowerExtractor(),
            'velocity': VelocityExtractor(),
            'path': PathExtractor(),
            'energy': EnergyExtractor(),
            'layer': LayerExtractor(),
        }
        
        # Fallback parser for unsupported formats
        self.generic_parser = GenericParser()
        
        logger.info("BuildFileParser initialized with all components")
    
    def parse_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Parse a build file using the appropriate format parser.
        
        Args:
            file_path: Path to the build file to parse
            
        Returns:
            Dictionary containing parsed build data
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file format is not supported
            RuntimeError: If required dependencies are not available
        """
        file_path = Path(file_path)
        
        # Validate file
        if not self.validate_file(file_path):
            raise FileNotFoundError(f"Invalid or inaccessible file: {file_path}")
        
        # Detect format
        file_extension = file_path.suffix.lower()
        if not self.is_format_supported(file_extension):
            logger.warning(f"Format {file_extension} not directly supported, using generic parser")
            return self.generic_parser.parse_file(file_path)
        
        # Get appropriate parser
        parser = self.format_parsers.get(file_extension, self.generic_parser)
        
        logger.info(f"Parsing {file_path} using {parser.__class__.__name__}")
        
        try:
            # Parse the file
            build_data = parser.parse_file(file_path)
            
            # Extract additional metadata
            metadata = self.metadata_extractor.extract_metadata(file_path, build_data)
            build_data['metadata'] = metadata
            
            # Perform data extraction if PySLM is available
            if self.PYSLM_AVAILABLE:
                build_data = self._extract_additional_data(build_data)
            
            logger.info(f"Successfully parsed {file_path}")
            return build_data
            
        except Exception as e:
            logger.error(f"Error parsing {file_path}: {e}")
            raise
    
    def _extract_additional_data(self, build_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract additional data using PySLM-based extractors.
        
        Args:
            build_data: Basic parsed build data
            
        Returns:
            Enhanced build data with additional analysis
        """
        try:
            # Extract power analysis
            if 'power' in self.data_extractors:
                power_data = self.data_extractors['power'].extract_power_data(build_data)
                build_data['power_analysis'] = power_data
            
            # Extract velocity analysis
            if 'velocity' in self.data_extractors:
                velocity_data = self.data_extractors['velocity'].extract_velocity_data(build_data)
                build_data['velocity_analysis'] = velocity_data
            
            # Extract path analysis
            if 'path' in self.data_extractors:
                path_data = self.data_extractors['path'].extract_path_data(build_data)
                build_data['path_analysis'] = path_data
            
            # Extract energy analysis
            if 'energy' in self.data_extractors:
                energy_data = self.data_extractors['energy'].extract_energy_data(build_data)
                build_data['energy_analysis'] = energy_data
            
            # Extract layer analysis
            if 'layer' in self.data_extractors:
                layer_data = self.data_extractors['layer'].extract_layer_data(build_data)
                build_data['layer_analysis'] = layer_data
            
            logger.info("Additional data extraction completed using PySLM")
            return build_data
            
        except Exception as e:
            logger.warning(f"Error in additional data extraction: {e}")
            return build_data
    
    def get_supported_formats(self) -> List[str]:
        """
        Get list of all supported file formats.
        
        Returns:
            List of supported file extensions
        """
        return list(self.format_parsers.keys())
    
    def is_format_supported(self, file_extension: str) -> bool:
        """
        Check if a file format is supported.
        
        Args:
            file_extension: File extension to check
            
        Returns:
            True if the format is supported, False otherwise
        """
        return file_extension.lower() in self.format_parsers
    
    def get_parser_for_format(self, file_extension: str) -> Optional[BaseBuildParser]:
        """
        Get the parser instance for a specific format.
        
        Args:
            file_extension: File extension
            
        Returns:
            Parser instance or None if not supported
        """
        return self.format_parsers.get(file_extension.lower())
    
    def get_available_extractors(self) -> List[str]:
        """
        Get list of available data extractors.
        
        Returns:
            List of available extractor names
        """
        return list(self.data_extractors.keys())
    
    def get_extractor(self, extractor_name: str) -> Optional[Any]:
        """
        Get a specific data extractor.
        
        Args:
            extractor_name: Name of the extractor
            
        Returns:
            Extractor instance or None if not available
        """
        return self.data_extractors.get(extractor_name)
    
    def analyze_build_file(self, file_path: Union[str, Path], analysis_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Perform comprehensive analysis of a build file.
        
        Args:
            file_path: Path to the build file
            analysis_types: List of analysis types to perform (None for all)
            
        Returns:
            Dictionary containing analysis results
        """
        # Parse the file first
        build_data = self.parse_file(file_path)
        
        # Determine which analyses to perform
        if analysis_types is None:
            analysis_types = self.get_available_extractors()
        
        analysis_results = {
            'file_path': str(file_path),
            'analysis_timestamp': datetime.now().isoformat(),
            'analysis_types': analysis_types,
            'build_data': build_data
        }
        
        # Perform requested analyses
        for analysis_type in analysis_types:
            if analysis_type in self.data_extractors:
                try:
                    extractor = self.data_extractors[analysis_type]
                    if hasattr(extractor, 'analyze'):
                        result = extractor.analyze(build_data)
                        analysis_results[f'{analysis_type}_analysis'] = result
                except Exception as e:
                    logger.warning(f"Error in {analysis_type} analysis: {e}")
                    analysis_results[f'{analysis_type}_analysis'] = {'error': str(e)}
        
        return analysis_results
    
    def get_parser_info(self) -> Dict[str, Any]:
        """
        Get information about available parsers and extractors.
        
        Returns:
            Dictionary containing parser and extractor information
        """
        return {
            'supported_formats': self.get_supported_formats(),
            'available_extractors': self.get_available_extractors(),
            'dependencies': self.check_dependencies(),
            'format_parsers': {
                fmt: parser.__class__.__name__ 
                for fmt, parser in self.format_parsers.items()
            },
            'data_extractors': {
                name: extractor.__class__.__name__ 
                for name, extractor in self.data_extractors.items()
            }
        }
