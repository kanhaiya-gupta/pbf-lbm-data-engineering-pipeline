"""
Validation Utilities for PBF-LB/M Build Files.

This module provides validation functions for build file data,
ensuring data integrity and format compliance.
"""

from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path
import logging
import struct

logger = logging.getLogger(__name__)


class ValidationUtils:
    """
    Utility class for validating PBF-LB/M build file data.
    
    This class provides validation functions for build file integrity,
    format compliance, and data consistency checks.
    """
    
    @staticmethod
    def validate_build_data(build_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate build file data structure and content.
        
        Args:
            build_data: Build file data to validate
            
        Returns:
            Dictionary containing validation results
        """
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'checks_performed': []
        }
        
        try:
            # Check required fields
            required_fields = ['file_path', 'file_format', 'parser']
            for field in required_fields:
                if field not in build_data:
                    validation_results['errors'].append(f"Missing required field: {field}")
                    validation_results['is_valid'] = False
                else:
                    validation_results['checks_performed'].append(f"Required field '{field}' present")
            
            # Validate file format
            if 'file_format' in build_data:
                format_validation = ValidationUtils.validate_file_format(build_data['file_format'])
                validation_results['checks_performed'].append("File format validation")
                if not format_validation['is_valid']:
                    validation_results['errors'].extend(format_validation['errors'])
                    validation_results['is_valid'] = False
            
            # Validate metadata
            if 'metadata' in build_data:
                metadata_validation = ValidationUtils.validate_metadata(build_data['metadata'])
                validation_results['checks_performed'].append("Metadata validation")
                if not metadata_validation['is_valid']:
                    validation_results['warnings'].extend(metadata_validation['warnings'])
            
            # Validate layers
            if 'layers' in build_data:
                layers_validation = ValidationUtils.validate_layers(build_data['layers'])
                validation_results['checks_performed'].append("Layers validation")
                if not layers_validation['is_valid']:
                    validation_results['errors'].extend(layers_validation['errors'])
                    validation_results['is_valid'] = False
                validation_results['warnings'].extend(layers_validation['warnings'])
            
            # Validate parameters
            if 'parameters' in build_data:
                params_validation = ValidationUtils.validate_parameters(build_data['parameters'])
                validation_results['checks_performed'].append("Parameters validation")
                if not params_validation['is_valid']:
                    validation_results['warnings'].extend(params_validation['warnings'])
        
        except Exception as e:
            validation_results['errors'].append(f"Validation error: {str(e)}")
            validation_results['is_valid'] = False
        
        return validation_results
    
    @staticmethod
    def validate_file_format(file_format: str) -> Dict[str, Any]:
        """
        Validate file format string.
        
        Args:
            file_format: File format string to validate
            
        Returns:
            Validation results
        """
        result = {'is_valid': True, 'errors': []}
        
        try:
            if not file_format:
                result['errors'].append("File format is empty")
                result['is_valid'] = False
                return result
            
            # Check if format starts with dot
            if not file_format.startswith('.'):
                result['errors'].append("File format should start with '.'")
                result['is_valid'] = False
            
            # Check if format is supported
            supported_formats = ['.mtt', '.sli', '.cli', '.slm', '.rea', '.txt', '.json', '.xml', '.csv', '.dat']
            if file_format.lower() not in supported_formats:
                result['errors'].append(f"Unsupported file format: {file_format}")
                result['is_valid'] = False
        
        except Exception as e:
            result['errors'].append(f"Format validation error: {str(e)}")
            result['is_valid'] = False
        
        return result
    
    @staticmethod
    def validate_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate metadata structure and content.
        
        Args:
            metadata: Metadata dictionary to validate
            
        Returns:
            Validation results
        """
        result = {'is_valid': True, 'warnings': []}
        
        try:
            if not isinstance(metadata, dict):
                result['warnings'].append("Metadata should be a dictionary")
                result['is_valid'] = False
                return result
            
            # Check for common metadata fields
            expected_fields = ['file_name', 'file_size', 'creation_time', 'modification_time']
            for field in expected_fields:
                if field not in metadata:
                    result['warnings'].append(f"Missing metadata field: {field}")
            
            # Validate file size
            if 'file_size' in metadata:
                file_size = metadata['file_size']
                if not isinstance(file_size, (int, float)) or file_size <= 0:
                    result['warnings'].append("Invalid file size in metadata")
            
            # Validate timestamps
            timestamp_fields = ['creation_time', 'modification_time']
            for field in timestamp_fields:
                if field in metadata:
                    timestamp = metadata[field]
                    if not isinstance(timestamp, (int, float)) or timestamp <= 0:
                        result['warnings'].append(f"Invalid timestamp in {field}")
        
        except Exception as e:
            result['warnings'].append(f"Metadata validation error: {str(e)}")
            result['is_valid'] = False
        
        return result
    
    @staticmethod
    def validate_layers(layers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate layers data structure and content.
        
        Args:
            layers: List of layer dictionaries to validate
            
        Returns:
            Validation results
        """
        result = {'is_valid': True, 'errors': [], 'warnings': []}
        
        try:
            if not isinstance(layers, list):
                result['errors'].append("Layers should be a list")
                result['is_valid'] = False
                return result
            
            if not layers:
                result['warnings'].append("No layers found in build data")
                return result
            
            # Validate each layer
            for i, layer in enumerate(layers):
                layer_validation = ValidationUtils.validate_single_layer(layer, i)
                if not layer_validation['is_valid']:
                    result['errors'].extend([f"Layer {i}: {error}" for error in layer_validation['errors']])
                    result['is_valid'] = False
                result['warnings'].extend([f"Layer {i}: {warning}" for warning in layer_validation['warnings']])
            
            # Check layer consistency
            layer_indices = [layer.get('layer_index', -1) for layer in layers]
            if layer_indices != list(range(len(layers))):
                result['warnings'].append("Layer indices are not sequential")
        
        except Exception as e:
            result['errors'].append(f"Layers validation error: {str(e)}")
            result['is_valid'] = False
        
        return result
    
    @staticmethod
    def validate_single_layer(layer: Dict[str, Any], expected_index: int) -> Dict[str, Any]:
        """
        Validate a single layer.
        
        Args:
            layer: Layer dictionary to validate
            expected_index: Expected layer index
            
        Returns:
            Validation results
        """
        result = {'is_valid': True, 'errors': [], 'warnings': []}
        
        try:
            if not isinstance(layer, dict):
                result['errors'].append("Layer should be a dictionary")
                result['is_valid'] = False
                return result
            
            # Check layer index
            if 'layer_index' in layer:
                if layer['layer_index'] != expected_index:
                    result['warnings'].append(f"Layer index mismatch: expected {expected_index}, got {layer['layer_index']}")
            
            # Validate z_height
            if 'z_height' in layer:
                z_height = layer['z_height']
                if not isinstance(z_height, (int, float)) or z_height < 0:
                    result['warnings'].append("Invalid z_height value")
            
            # Validate thickness
            if 'thickness' in layer:
                thickness = layer['thickness']
                if not isinstance(thickness, (int, float)) or thickness <= 0:
                    result['warnings'].append("Invalid thickness value")
            
            # Validate geometry counts
            geometry_fields = ['hatch_count', 'contour_count', 'point_count']
            for field in geometry_fields:
                if field in layer:
                    count = layer[field]
                    if not isinstance(count, int) or count < 0:
                        result['warnings'].append(f"Invalid {field} value")
        
        except Exception as e:
            result['errors'].append(f"Layer validation error: {str(e)}")
            result['is_valid'] = False
        
        return result
    
    @staticmethod
    def validate_parameters(parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate parameters data structure and content.
        
        Args:
            parameters: Parameters dictionary to validate
            
        Returns:
            Validation results
        """
        result = {'is_valid': True, 'warnings': []}
        
        try:
            if not isinstance(parameters, dict):
                result['warnings'].append("Parameters should be a dictionary")
                result['is_valid'] = False
                return result
            
            # Validate power parameters
            power_fields = ['default_power', 'max_power', 'min_power']
            for field in power_fields:
                if field in parameters:
                    power = parameters[field]
                    if not isinstance(power, (int, float)) or power < 0:
                        result['warnings'].append(f"Invalid {field} value")
            
            # Validate velocity parameters
            velocity_fields = ['default_velocity', 'max_velocity', 'min_velocity']
            for field in velocity_fields:
                if field in parameters:
                    velocity = parameters[field]
                    if not isinstance(velocity, (int, float)) or velocity <= 0:
                        result['warnings'].append(f"Invalid {field} value")
            
            # Validate layer parameters
            if 'layer_parameters' in parameters:
                layer_params = parameters['layer_parameters']
                if isinstance(layer_params, dict):
                    for layer_key, layer_param in layer_params.items():
                        if not isinstance(layer_param, dict):
                            result['warnings'].append(f"Layer parameters for {layer_key} should be a dictionary")
        
        except Exception as e:
            result['warnings'].append(f"Parameters validation error: {str(e)}")
            result['is_valid'] = False
        
        return result
    
    @staticmethod
    def validate_coordinates(coordinates: List[float], expected_dimensions: int = 3) -> bool:
        """
        Validate coordinate data.
        
        Args:
            coordinates: List of coordinate values
            expected_dimensions: Expected number of dimensions
            
        Returns:
            True if coordinates are valid, False otherwise
        """
        try:
            if not isinstance(coordinates, list):
                return False
            
            if len(coordinates) < expected_dimensions:
                return False
            
            # Check if all values are numeric
            for coord in coordinates[:expected_dimensions]:
                if not isinstance(coord, (int, float)):
                    return False
            
            return True
        
        except Exception:
            return False
    
    @staticmethod
    def validate_file_header(file_path: Union[str, Path], expected_signatures: List[bytes]) -> bool:
        """
        Validate file header against expected signatures.
        
        Args:
            file_path: Path to the file
            expected_signatures: List of expected header signatures
            
        Returns:
            True if header matches expected signature, False otherwise
        """
        try:
            path = Path(file_path)
            if not path.exists():
                return False
            
            with open(path, 'rb') as f:
                header = f.read(max(len(sig) for sig in expected_signatures))
            
            for signature in expected_signatures:
                if header.startswith(signature):
                    return True
            
            return False
        
        except Exception:
            return False
    
    @staticmethod
    def validate_build_volume(bounds: Dict[str, float]) -> bool:
        """
        Validate build volume bounds.
        
        Args:
            bounds: Dictionary containing build volume bounds
            
        Returns:
            True if bounds are valid, False otherwise
        """
        try:
            required_fields = ['x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max']
            
            for field in required_fields:
                if field not in bounds:
                    return False
                
                value = bounds[field]
                if not isinstance(value, (int, float)):
                    return False
            
            # Check that min < max for each dimension
            if bounds['x_min'] >= bounds['x_max']:
                return False
            if bounds['y_min'] >= bounds['y_max']:
                return False
            if bounds['z_min'] >= bounds['z_max']:
                return False
            
            return True
        
        except Exception:
            return False
    
    @staticmethod
    def get_validation_summary(validation_results: Dict[str, Any]) -> str:
        """
        Get a summary of validation results.
        
        Args:
            validation_results: Validation results dictionary
            
        Returns:
            Summary string
        """
        try:
            is_valid = validation_results.get('is_valid', False)
            error_count = len(validation_results.get('errors', []))
            warning_count = len(validation_results.get('warnings', []))
            checks_count = len(validation_results.get('checks_performed', []))
            
            status = "VALID" if is_valid else "INVALID"
            summary = f"Validation Status: {status}\n"
            summary += f"Checks Performed: {checks_count}\n"
            summary += f"Errors: {error_count}\n"
            summary += f"Warnings: {warning_count}"
            
            return summary
        
        except Exception as e:
            return f"Error generating validation summary: {str(e)}"
