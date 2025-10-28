"""
Generic Parser for PBF-LB/M Build Files.

This module provides a generic parser for build files that don't have
specific format parsers, using basic file analysis and structure detection.
"""

from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import logging
import struct
import json
from src.data_pipeline.processing.knowledge_graph.utils.json_parser import safe_json_loads_with_fallback

from ..base_parser import BaseBuildParser

logger = logging.getLogger(__name__)


class GenericParser(BaseBuildParser):
    """
    Generic parser for build files without specific format support.
    
    This parser provides basic file analysis and structure detection
    for build files that don't have dedicated format parsers.
    """
    
    def __init__(self):
        """Initialize the generic parser."""
        super().__init__()
        self.supported_formats = ['.txt', '.json', '.xml', '.csv', '.dat']
        self.parser_name = "Generic Parser"
        logger.info("Generic parser initialized")
    
    def can_parse(self, file_path: Union[str, Path]) -> bool:
        """Check if this parser can handle the given file."""
        file_path = Path(file_path)
        return file_path.suffix.lower() in self.supported_formats
    
    def parse_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Parse a build file using generic methods.
        
        Args:
            file_path: Path to the build file
            
        Returns:
            Dictionary containing parsed build data
        """
        file_path = Path(file_path)
        
        if not self.can_parse(file_path):
            raise ValueError(f"Generic parser cannot handle file: {file_path}")
        
        try:
            logger.info(f"Parsing file with generic parser: {file_path}")
            
            # Read file content
            with open(file_path, 'rb') as f:
                raw_data = f.read()
            
            # Try to decode as text
            try:
                text_content = raw_data.decode('utf-8')
                is_text = True
            except UnicodeDecodeError:
                text_content = None
                is_text = False
            
            # Extract basic information
            result = {
                'file_path': str(file_path),
                'file_format': file_path.suffix.lower(),
                'parser': self.parser_name,
                'file_size': len(raw_data),
                'is_text': is_text,
                'metadata': self._extract_basic_metadata(file_path, raw_data),
                'content': self._parse_content(text_content, file_path.suffix.lower()) if is_text else None,
                'binary_analysis': self._analyze_binary_structure(raw_data) if not is_text else None
            }
            
            logger.info(f"Successfully parsed file with generic parser: {file_path}")
            return result
            
        except Exception as e:
            logger.error(f"Error parsing file with generic parser {file_path}: {e}")
            raise
    
    def _extract_basic_metadata(self, file_path: Path, raw_data: bytes) -> Dict[str, Any]:
        """Extract basic metadata from the file."""
        metadata = {
            'file_name': file_path.name,
            'file_size': len(raw_data),
            'file_extension': file_path.suffix.lower(),
            'creation_time': None,
            'modification_time': None
        }
        
        try:
            stat = file_path.stat()
            metadata['creation_time'] = stat.st_ctime
            metadata['modification_time'] = stat.st_mtime
        except Exception as e:
            logger.warning(f"Error getting file stats: {e}")
        
        return metadata
    
    def _parse_content(self, text_content: str, file_extension: str) -> Dict[str, Any]:
        """Parse text content based on file extension."""
        content_info = {
            'lines': text_content.count('\n') + 1,
            'characters': len(text_content),
            'words': len(text_content.split()),
            'parsed_data': None
        }
        
        try:
            if file_extension == '.json':
                content_info['parsed_data'] = safe_json_loads_with_fallback(text_content, 'text_content', 5000, {})
            elif file_extension == '.xml':
                # Basic XML structure detection
                content_info['parsed_data'] = self._parse_xml_basic(text_content)
            elif file_extension == '.csv':
                # Basic CSV parsing
                content_info['parsed_data'] = self._parse_csv_basic(text_content)
            else:
                # Generic text analysis
                content_info['parsed_data'] = self._analyze_text_structure(text_content)
        
        except Exception as e:
            logger.warning(f"Error parsing content: {e}")
            content_info['parsed_data'] = {'error': str(e)}
        
        return content_info
    
    def _parse_xml_basic(self, text_content: str) -> Dict[str, Any]:
        """Basic XML structure analysis."""
        lines = text_content.split('\n')
        xml_info = {
            'root_elements': [],
            'total_elements': 0,
            'has_declaration': text_content.strip().startswith('<?xml'),
            'structure': 'unknown'
        }
        
        # Count elements (basic)
        xml_info['total_elements'] = text_content.count('<') - text_content.count('</')
        
        # Find root elements
        for line in lines:
            line = line.strip()
            if line.startswith('<') and not line.startswith('<?') and not line.startswith('<!--'):
                if not line.startswith('</'):
                    element_name = line.split()[0].replace('<', '').replace('>', '')
                    if element_name not in xml_info['root_elements']:
                        xml_info['root_elements'].append(element_name)
        
        return xml_info
    
    def _parse_csv_basic(self, text_content: str) -> Dict[str, Any]:
        """Basic CSV structure analysis."""
        lines = text_content.split('\n')
        csv_info = {
            'rows': len([line for line in lines if line.strip()]),
            'columns': 0,
            'has_header': False,
            'delimiter': ','
        }
        
        if lines:
            # Detect delimiter
            first_line = lines[0]
            if ';' in first_line and ',' not in first_line:
                csv_info['delimiter'] = ';'
            elif '\t' in first_line and ',' not in first_line:
                csv_info['delimiter'] = '\t'
            
            # Count columns
            csv_info['columns'] = len(first_line.split(csv_info['delimiter']))
            
            # Check if first line looks like a header
            if csv_info['rows'] > 1:
                second_line = lines[1] if len(lines) > 1 else ""
                if not second_line.replace('.', '').replace('-', '').replace('+', '').replace('e', '').replace('E', '').isdigit():
                    csv_info['has_header'] = True
        
        return csv_info
    
    def _analyze_text_structure(self, text_content: str) -> Dict[str, Any]:
        """Analyze general text structure."""
        lines = text_content.split('\n')
        structure_info = {
            'indentation_pattern': 'unknown',
            'has_comments': False,
            'has_numbers': False,
            'has_coordinates': False,
            'structure_type': 'unknown'
        }
        
        # Check for common patterns
        if any(line.strip().startswith('#') for line in lines):
            structure_info['has_comments'] = True
        
        if any(char.isdigit() for char in text_content):
            structure_info['has_numbers'] = True
        
        # Check for coordinate-like patterns
        if any(',' in line and any(char.isdigit() for char in line) for line in lines):
            structure_info['has_coordinates'] = True
        
        # Detect indentation pattern
        indentations = [len(line) - len(line.lstrip()) for line in lines if line.strip()]
        if indentations:
            if all(indent % 4 == 0 for indent in indentations):
                structure_info['indentation_pattern'] = 'spaces_4'
            elif all(indent % 2 == 0 for indent in indentations):
                structure_info['indentation_pattern'] = 'spaces_2'
            elif all(indent == 0 for indent in indentations):
                structure_info['indentation_pattern'] = 'none'
        
        return structure_info
    
    def _analyze_binary_structure(self, raw_data: bytes) -> Dict[str, Any]:
        """Analyze binary file structure."""
        binary_info = {
            'file_type': 'unknown',
            'has_header': False,
            'header_bytes': raw_data[:16].hex() if len(raw_data) >= 16 else raw_data.hex(),
            'size': len(raw_data),
            'entropy': self._calculate_entropy(raw_data)
        }
        
        # Check for common file signatures
        if raw_data.startswith(b'PK'):
            binary_info['file_type'] = 'zip_archive'
        elif raw_data.startswith(b'\x89PNG'):
            binary_info['file_type'] = 'png_image'
        elif raw_data.startswith(b'\xff\xd8\xff'):
            binary_info['file_type'] = 'jpeg_image'
        elif raw_data.startswith(b'%PDF'):
            binary_info['file_type'] = 'pdf_document'
        
        # Check if it looks like structured binary data
        if len(raw_data) > 0:
            # Look for repeated patterns that might indicate structured data
            chunk_size = min(32, len(raw_data) // 4)
            if chunk_size > 0:
                chunks = [raw_data[i:i+chunk_size] for i in range(0, len(raw_data), chunk_size)]
                unique_chunks = len(set(chunks))
                if unique_chunks < len(chunks) * 0.5:
                    binary_info['has_header'] = True
        
        return binary_info
    
    def _calculate_entropy(self, data: bytes) -> float:
        """Calculate Shannon entropy of binary data."""
        if not data:
            return 0.0
        
        # Count byte frequencies
        byte_counts = [0] * 256
        for byte in data:
            byte_counts[byte] += 1
        
        # Calculate entropy
        entropy = 0.0
        data_len = len(data)
        for count in byte_counts:
            if count > 0:
                probability = count / data_len
                entropy -= probability * (probability.bit_length() - 1)
        
        return entropy
    
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
            'description': 'Generic parser for build files without specific format support'
        }
