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

# Import format-specific data extractors
from ..data_extractors.slm.power_extractor import PowerExtractor as SLMPowerExtractor
from ..data_extractors.slm.velocity_extractor import VelocityExtractor as SLMVelocityExtractor
from ..data_extractors.slm.path_extractor import PathExtractor as SLMPathExtractor
from ..data_extractors.slm.energy_extractor import EnergyExtractor as SLMEnergyExtractor
from ..data_extractors.slm.layer_extractor import LayerExtractor as SLMLayerExtractor

from ..data_extractors.sli.power_extractor import PowerExtractor as SLIPowerExtractor
from ..data_extractors.sli.velocity_extractor import VelocityExtractor as SLIVelocityExtractor
from ..data_extractors.sli.path_extractor import PathExtractor as SLIPathExtractor
from ..data_extractors.sli.energy_extractor import EnergyExtractor as SLIEnergyExtractor
from ..data_extractors.sli.layer_extractor import LayerExtractor as SLILayerExtractor

from ..data_extractors.mtt.power_extractor import PowerExtractor as MTTPowerExtractor
from ..data_extractors.mtt.velocity_extractor import VelocityExtractor as MTTVelocityExtractor
from ..data_extractors.mtt.path_extractor import PathExtractor as MTTPathExtractor
from ..data_extractors.mtt.energy_extractor import EnergyExtractor as MTTEnergyExtractor
from ..data_extractors.mtt.layer_extractor import LayerExtractor as MTTLayerExtractor

# CLI extractors not needed - CLI is not supported by libSLM

from ..data_extractors.realizer.power_extractor import PowerExtractor as RealizerPowerExtractor
from ..data_extractors.realizer.velocity_extractor import VelocityExtractor as RealizerVelocityExtractor
from ..data_extractors.realizer.path_extractor import PathExtractor as RealizerPathExtractor
from ..data_extractors.realizer.energy_extractor import EnergyExtractor as RealizerEnergyExtractor
from ..data_extractors.realizer.layer_extractor import LayerExtractor as RealizerLayerExtractor

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
            '.mtt': MTTParser(),
            '.rea': RealizerParser(),
            '.slm': SLMParser(),
        }
        
        # CLI is not supported by libSLM - will be handled as unsupported format
        
        # Initialize format-specific data extractors
        self.data_extractors = {
            'slm': {
                'power': SLMPowerExtractor(),
                'velocity': SLMVelocityExtractor(),
                'path': SLMPathExtractor(),
                'energy': SLMEnergyExtractor(),
                'layer': SLMLayerExtractor(),
            },
            'sli': {
                'power': SLIPowerExtractor(),
                'velocity': SLIVelocityExtractor(),
                'path': SLIPathExtractor(),
                'energy': SLIEnergyExtractor(),
                'layer': SLILayerExtractor(),
            },
            # CLI is not supported by libSLM - no extractors needed
            'mtt': {
                'power': MTTPowerExtractor(),
                'velocity': MTTVelocityExtractor(),
                'path': MTTPathExtractor(),
                'energy': MTTEnergyExtractor(),
                'layer': MTTLayerExtractor(),
            },
            'rea': {
                'power': RealizerPowerExtractor(),
                'velocity': RealizerVelocityExtractor(),
                'path': RealizerPathExtractor(),
                'energy': RealizerEnergyExtractor(),
                'layer': RealizerLayerExtractor(),
            }
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
        
        # Handle CLI files specifically (not supported by libSLM)
        if file_extension == '.cli':
            error_msg = (
                f"CLI (.cli) files are not supported by libSLM library. "
                f"CLI is an older format primarily for SLA systems and lacks PBF-LB/M process parameters. "
                f"Supported formats: {', '.join(self.get_supported_formats())}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        if not self.is_format_supported(file_extension):
            logger.warning(f"Format {file_extension} not directly supported, using generic parser")
            return self.generic_parser.parse_file(file_path)
        
        # Get appropriate parser
        parser = self.format_parsers.get(file_extension, self.generic_parser)
        
        logger.info(f"Parsing {file_path} using {parser.__class__.__name__}")
        
        try:
            # Parse the file
            build_data = parser.parse_file(file_path)
            
            # Store the reader object for data extractors (if available)
            # This works for any format parser that has a reader object
            reader_attr_names = ['reader', 'slm_reader', 'mtt_reader', 'eos_reader', 'realizer_reader']
            for attr_name in reader_attr_names:
                if hasattr(parser, attr_name):
                    reader_obj = getattr(parser, attr_name)
                    if reader_obj:
                        build_data['reader_object'] = reader_obj
                        break
            
            # Extract additional metadata and merge with parser metadata
            additional_metadata = self.metadata_extractor.extract_metadata(file_path, build_data)
            
            # Merge metadata instead of overwriting
            if 'metadata' in build_data:
                # Merge parser metadata with additional metadata
                parser_metadata = build_data['metadata']
                merged_metadata = {**parser_metadata, **additional_metadata}
                build_data['metadata'] = merged_metadata
            else:
                build_data['metadata'] = additional_metadata
            
            # Perform data extraction if PySLM is available
            if self.PYSLM_AVAILABLE:
                build_data = self._extract_additional_data(build_data, file_extension)
            
            logger.info(f"Successfully parsed {file_path}")
            return build_data
            
        except Exception as e:
            logger.error(f"Error parsing {file_path}: {e}")
            raise
    
    def _extract_additional_data(self, build_data: Dict[str, Any], file_extension: str) -> Dict[str, Any]:
        """
        Extract additional data using format-specific extractors.
        
        Args:
            build_data: Basic parsed build data
            file_extension: File extension to determine format-specific extractors
            
        Returns:
            Enhanced build data with additional analysis
        """
        try:
            # Determine format for extractor selection
            format_key = file_extension.lstrip('.').lower()
            if format_key in ['sli', 'cli']:
                format_key = 'sli'  # Both use SLI extractors
            elif format_key == 'rea':
                format_key = 'rea'  # Realizer format
            elif format_key not in self.data_extractors:
                format_key = 'slm'  # Default to SLM for unsupported formats
            
            # Get format-specific extractors
            extractors = self.data_extractors.get(format_key, {})
            
            # Extract power analysis
            if 'power' in extractors:
                power_data = extractors['power'].extract_power_data(build_data)
                build_data['power_analysis'] = power_data
            
            # Extract velocity analysis
            if 'velocity' in extractors:
                velocity_data = extractors['velocity'].extract_velocity_data(build_data)
                build_data['velocity_analysis'] = velocity_data
            
            # Extract path analysis
            if 'path' in extractors:
                path_data = extractors['path'].extract_path_data(build_data)
                build_data['path_analysis'] = path_data
            
            # Extract energy analysis
            if 'energy' in extractors:
                energy_data = extractors['energy'].extract_energy_data(build_data)
                build_data['energy_analysis'] = energy_data
            
            # Extract layer analysis
            if 'layer' in extractors:
                layer_data = extractors['layer'].extract_layer_data(build_data)
                build_data['layer_analysis'] = layer_data
            
            logger.info(f"Additional data extraction completed using {format_key.upper()} extractors")
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
        
        # Determine format for extractor selection
        file_extension = Path(file_path).suffix.lower()
        format_key = file_extension.lstrip('.').lower()
        if format_key in ['sli', 'cli']:
            format_key = 'sli'  # Both use SLI extractors
        elif format_key == 'rea':
            format_key = 'rea'  # Realizer format
        elif format_key not in self.data_extractors:
            format_key = 'slm'  # Default to SLM for unsupported formats
        
        # Get format-specific extractors
        extractors = self.data_extractors.get(format_key, {})
        
        # Perform requested analyses
        for analysis_type in analysis_types:
            if analysis_type in extractors:
                try:
                    extractor = extractors[analysis_type]
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
