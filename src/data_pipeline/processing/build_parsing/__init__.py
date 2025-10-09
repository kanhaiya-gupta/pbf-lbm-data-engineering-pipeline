"""
Build Parsing Module for PBF-LB/M Data Pipeline

This module provides comprehensive build file parsing capabilities that leverage
libSLM and PySLM libraries for world-class PBF-LB/M machine build file processing.

Key Features:
- Leverages libSLM for file format parsing (.mtt, .sli, .cli, .rea, .slm)
- Uses PySLM for advanced analysis and visualization
- Provides clean integration with the data pipeline
- Focuses on orchestration rather than reinvention

Architecture:
- base_parser.py: Abstract base class for all parsers
- core/: Main orchestrator and core functionality
- format_parsers/: Format-specific wrappers around libSLM
- data_extractors/: Extract specific data types using PySLM
- utils/: Common utilities and validation
"""

from .base_parser import BaseBuildParser
from .core.build_file_parser import BuildFileParser
from .core.format_detector import FormatDetector
from .core.metadata_extractor import MetadataExtractor

# Legacy compatibility classes
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

@dataclass
class BuildFileMetadata:
    """Legacy compatibility class for build file metadata."""
    file_path: str
    file_format: str
    file_size: int
    created_at: str
    modified_at: str
    layer_count: int
    build_volume: Dict[str, float]
    build_styles: List[Dict[str, Any]]
    models: List[Dict[str, Any]]

@dataclass 
class ScanPath:
    """Legacy compatibility class for scan path data."""
    path_id: str
    layer_index: int
    geometry_type: str  # 'hatch', 'contour', 'point'
    coordinates: List[List[float]]
    power: float
    velocity: float
    build_style_id: str

@dataclass
class LayerData:
    """Legacy compatibility class for layer data."""
    layer_index: int
    layer_height: float
    scan_paths: List[ScanPath]
    total_energy: float
    build_time: float

# Format parsers
from .format_parsers.eos_parser import EOSParser
from .format_parsers.mtt_parser import MTTParser
from .format_parsers.realizer_parser import RealizerParser
from .format_parsers.slm_parser import SLMParser
from .format_parsers.generic_parser import GenericParser

# Data extractors - Format-specific extractors are available through BuildFileParser
# Individual extractors can be imported from their format-specific directories:
# from .data_extractors.slm.power_extractor import PowerExtractor as SLMPowerExtractor
# from .data_extractors.sli.power_extractor import PowerExtractor as SLIPowerExtractor
# etc.

# Utilities
from .utils.file_utils import FileUtils
from .utils.validation_utils import ValidationUtils

__all__ = [
    # Core components
    "BaseBuildParser",
    "BuildFileParser", 
    "FormatDetector",
    "MetadataExtractor",
    
    # Legacy compatibility classes
    "BuildFileMetadata",
    "ScanPath",
    "LayerData",
    
    # Format parsers
    "EOSParser",
    "MTTParser", 
    "RealizerParser",
    "SLMParser",
    "GenericParser",
    
    # Data extractors - Use BuildFileParser for format-specific extractors
    
    # Utilities
    "FileUtils",
    "ValidationUtils",
]

# Version information
__version__ = "1.0.0"
__author__ = "PBF-LB/M Research Team"
__description__ = "Build file parsing leveraging libSLM and PySLM"
