"""
Abstract Base Parser for PBF-LB/M Build Files

This module provides the abstract base class that all build file parsers must implement.
It defines the interface for parsing PBF-LB/M machine build files using libSLM and PySLM.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import logging

# Import external dependencies
from ...external import LIBSLM_AVAILABLE, PYSLM_AVAILABLE

logger = logging.getLogger(__name__)


class BaseBuildParser(ABC):
    """
    Abstract base class for all PBF-LB/M build file parsers.
    
    This class defines the interface that all format-specific parsers must implement.
    It ensures consistent behavior across different file formats while leveraging
    the libSLM and PySLM libraries for actual parsing and analysis.
    """
    
    def __init__(self):
        """Initialize the base parser with dependency checks."""
        self.LIBSLM_AVAILABLE = LIBSLM_AVAILABLE
        self.PYSLM_AVAILABLE = PYSLM_AVAILABLE
        
        if not self.LIBSLM_AVAILABLE:
            logger.warning("libSLM not available - build file parsing will be limited")
        
        if not self.PYSLM_AVAILABLE:
            logger.warning("PySLM not available - advanced analysis will be limited")
        
        logger.info(f"BaseBuildParser initialized - libSLM: {self.LIBSLM_AVAILABLE}, PySLM: {self.PYSLM_AVAILABLE}")
    
    @abstractmethod
    def parse_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Parse a build file and return structured data.
        
        Args:
            file_path: Path to the build file to parse
            
        Returns:
            Dictionary containing parsed build data with the following structure:
            {
                'metadata': BuildFileMetadata,
                'layers': List[LayerData],
                'scan_paths': List[ScanPath],
                'build_parameters': Dict[str, Any],
                'parsing_method': str
            }
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file format is not supported
            RuntimeError: If required dependencies are not available
        """
        pass
    
    @abstractmethod
    def get_supported_formats(self) -> List[str]:
        """
        Get list of supported file formats for this parser.
        
        Returns:
            List of supported file extensions (e.g., ['.mtt', '.sli'])
        """
        pass
    
    @abstractmethod
    def is_format_supported(self, file_extension: str) -> bool:
        """
        Check if a file format is supported by this parser.
        
        Args:
            file_extension: File extension to check (e.g., '.mtt')
            
        Returns:
            True if the format is supported, False otherwise
        """
        pass
    
    def validate_file(self, file_path: Union[str, Path]) -> bool:
        """
        Validate that a file exists and is readable.
        
        Args:
            file_path: Path to the file to validate
            
        Returns:
            True if the file is valid, False otherwise
        """
        try:
            path = Path(file_path)
            return path.exists() and path.is_file() and path.stat().st_size > 0
        except Exception as e:
            logger.warning(f"File validation failed for {file_path}: {e}")
            return False
    
    def get_file_metadata(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Get basic file metadata without parsing the full file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary containing file metadata
        """
        try:
            path = Path(file_path)
            stat = path.stat()
            
            return {
                'file_path': str(path),
                'file_name': path.name,
                'file_extension': path.suffix,
                'file_size': stat.st_size,
                'creation_time': stat.st_ctime,
                'modification_time': stat.st_mtime,
                'is_readable': path.is_file() and path.stat().st_size > 0
            }
        except Exception as e:
            logger.error(f"Error getting file metadata for {file_path}: {e}")
            return {}
    
    def check_dependencies(self) -> Dict[str, bool]:
        """
        Check the availability of required dependencies.
        
        Returns:
            Dictionary indicating which dependencies are available
        """
        return {
            'libSLM': self.LIBSLM_AVAILABLE,
            'PySLM': self.PYSLM_AVAILABLE
        }
    
    def __str__(self) -> str:
        """String representation of the parser."""
        return f"{self.__class__.__name__}(libSLM: {self.LIBSLM_AVAILABLE}, PySLM: {self.PYSLM_AVAILABLE})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the parser."""
        return f"{self.__class__.__name__}(supported_formats: {self.get_supported_formats()})"
