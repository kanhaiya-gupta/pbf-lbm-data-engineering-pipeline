"""
Metadata Extraction for PBF-LB/M Build Files

This module provides metadata extraction capabilities for PBF-LB/M build files,
extracting file metadata, build parameters, and other relevant information.
"""

from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import logging
from datetime import datetime
import hashlib

logger = logging.getLogger(__name__)


class MetadataExtractor:
    """
    Extracts metadata from PBF-LB/M build files.
    
    This class provides comprehensive metadata extraction capabilities for
    build files, including file metadata, build parameters, and system information.
    """
    
    def __init__(self):
        """Initialize the metadata extractor."""
        logger.info("MetadataExtractor initialized")
    
    def extract_metadata(self, file_path: Union[str, Path], build_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Extract comprehensive metadata from a build file.
        
        Args:
            file_path: Path to the build file
            build_data: Optional parsed build data
            
        Returns:
            Dictionary containing extracted metadata
        """
        file_path = Path(file_path)
        
        metadata = {
            'file_metadata': self._extract_file_metadata(file_path),
            'build_metadata': self._extract_build_metadata(build_data) if build_data else {},
            'system_metadata': self._extract_system_metadata(),
            'extraction_timestamp': datetime.now().isoformat()
        }
        
        return metadata
    
    def _extract_file_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Extract file-level metadata."""
        try:
            stat = file_path.stat()
            
            # Calculate file hash for integrity checking
            file_hash = self._calculate_file_hash(file_path)
            
            return {
                'file_path': str(file_path),
                'file_name': file_path.name,
                'file_extension': file_path.suffix,
                'file_size': stat.st_size,
                'file_size_mb': round(stat.st_size / (1024 * 1024), 2),
                'creation_time': datetime.fromtimestamp(stat.st_ctime).isoformat(),
                'modification_time': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                'access_time': datetime.fromtimestamp(stat.st_atime).isoformat(),
                'file_hash_md5': file_hash['md5'],
                'file_hash_sha256': file_hash['sha256'],
                'is_readable': file_path.is_file() and stat.st_size > 0,
                'parent_directory': str(file_path.parent)
            }
            
        except Exception as e:
            logger.error(f"Error extracting file metadata: {e}")
            return {
                'file_path': str(file_path),
                'error': str(e)
            }
    
    def _extract_build_metadata(self, build_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract build-specific metadata from parsed data."""
        try:
            metadata = {
                'parsing_method': build_data.get('parsing_method', 'unknown'),
                'build_parameters': build_data.get('build_parameters', {}),
                'layer_count': len(build_data.get('layers', [])),
                'scan_path_count': len(build_data.get('scan_paths', [])),
                'has_metadata': 'metadata' in build_data
            }
            
            # Extract layer information
            layers = build_data.get('layers', [])
            if layers:
                layer_heights = [layer.get('layer_height', 0) for layer in layers if isinstance(layer, dict)]
                if layer_heights:
                    metadata['layer_height_stats'] = {
                        'min': min(layer_heights),
                        'max': max(layer_heights),
                        'mean': sum(layer_heights) / len(layer_heights),
                        'count': len(layer_heights)
                    }
            
            # Extract scan path information
            scan_paths = build_data.get('scan_paths', [])
            if scan_paths:
                path_types = {}
                for path in scan_paths:
                    if isinstance(path, dict):
                        path_type = path.get('path_type', 'unknown')
                        path_types[path_type] = path_types.get(path_type, 0) + 1
                metadata['scan_path_types'] = path_types
            
            # Extract build volume information
            if 'build_volume' in build_data:
                metadata['build_volume'] = build_data['build_volume']
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error extracting build metadata: {e}")
            return {'error': str(e)}
    
    def _extract_system_metadata(self) -> Dict[str, Any]:
        """Extract system-level metadata."""
        try:
            import platform
            import sys
            
            return {
                'python_version': sys.version,
                'platform': platform.platform(),
                'architecture': platform.architecture(),
                'processor': platform.processor(),
                'system': platform.system(),
                'machine': platform.machine(),
                'node': platform.node(),
                'python_implementation': platform.python_implementation(),
                'python_compiler': platform.python_compiler()
            }
            
        except Exception as e:
            logger.error(f"Error extracting system metadata: {e}")
            return {'error': str(e)}
    
    def _calculate_file_hash(self, file_path: Path) -> Dict[str, str]:
        """Calculate MD5 and SHA256 hashes of the file."""
        try:
            md5_hash = hashlib.md5()
            sha256_hash = hashlib.sha256()
            
            with open(file_path, 'rb') as f:
                # Read file in chunks to handle large files
                for chunk in iter(lambda: f.read(4096), b""):
                    md5_hash.update(chunk)
                    sha256_hash.update(chunk)
            
            return {
                'md5': md5_hash.hexdigest(),
                'sha256': sha256_hash.hexdigest()
            }
            
        except Exception as e:
            logger.error(f"Error calculating file hash: {e}")
            return {
                'md5': 'error',
                'sha256': 'error'
            }
    
    def extract_build_parameters(self, build_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract build parameters from parsed build data.
        
        Args:
            build_data: Parsed build data
            
        Returns:
            Dictionary containing build parameters
        """
        try:
            parameters = build_data.get('build_parameters', {})
            
            # Extract common build parameters
            extracted_params = {
                'laser_power': self._extract_parameter_range(parameters, 'laser_power'),
                'scan_speed': self._extract_parameter_range(parameters, 'scan_speed'),
                'layer_thickness': self._extract_parameter_range(parameters, 'layer_thickness'),
                'hatch_spacing': self._extract_parameter_range(parameters, 'hatch_spacing'),
                'exposure_time': self._extract_parameter_range(parameters, 'exposure_time'),
                'point_distance': self._extract_parameter_range(parameters, 'point_distance'),
                'build_style': parameters.get('build_style'),
                'laser_mode': parameters.get('laser_mode'),
                'scan_mode': parameters.get('scan_mode')
            }
            
            # Remove None values
            extracted_params = {k: v for k, v in extracted_params.items() if v is not None}
            
            return extracted_params
            
        except Exception as e:
            logger.error(f"Error extracting build parameters: {e}")
            return {'error': str(e)}
    
    def _extract_parameter_range(self, parameters: Dict[str, Any], param_name: str) -> Optional[Dict[str, float]]:
        """Extract parameter range information."""
        try:
            param_value = parameters.get(param_name)
            
            if param_value is None:
                return None
            
            if isinstance(param_value, (int, float)):
                return {
                    'value': float(param_value),
                    'min': float(param_value),
                    'max': float(param_value),
                    'type': 'single_value'
                }
            
            if isinstance(param_value, list):
                numeric_values = [float(v) for v in param_value if isinstance(v, (int, float))]
                if numeric_values:
                    return {
                        'min': min(numeric_values),
                        'max': max(numeric_values),
                        'mean': sum(numeric_values) / len(numeric_values),
                        'count': len(numeric_values),
                        'type': 'range'
                    }
            
            if isinstance(param_value, dict):
                return {
                    'min': param_value.get('min'),
                    'max': param_value.get('max'),
                    'mean': param_value.get('mean'),
                    'std': param_value.get('std'),
                    'type': 'statistics'
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error extracting parameter range for {param_name}: {e}")
            return None
    
    def get_metadata_summary(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get a summary of extracted metadata.
        
        Args:
            metadata: Extracted metadata
            
        Returns:
            Summary of metadata
        """
        try:
            file_meta = metadata.get('file_metadata', {})
            build_meta = metadata.get('build_metadata', {})
            
            summary = {
                'file_info': {
                    'name': file_meta.get('file_name'),
                    'size_mb': file_meta.get('file_size_mb'),
                    'format': file_meta.get('file_extension')
                },
                'build_info': {
                    'layer_count': build_meta.get('layer_count', 0),
                    'scan_path_count': build_meta.get('scan_path_count', 0),
                    'parsing_method': build_meta.get('parsing_method')
                },
                'extraction_info': {
                    'timestamp': metadata.get('extraction_timestamp'),
                    'has_build_data': bool(build_meta)
                }
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error creating metadata summary: {e}")
            return {'error': str(e)}
