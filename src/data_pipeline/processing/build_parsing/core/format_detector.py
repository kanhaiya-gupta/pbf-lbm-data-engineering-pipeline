"""
Format Detection for PBF-LB/M Build Files

This module provides format detection capabilities for PBF-LB/M build files,
leveraging libSLM's format detection capabilities when available.
"""

from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import logging
import struct

from ....external import LIBSLM_AVAILABLE

logger = logging.getLogger(__name__)


class FormatDetector:
    """
    Detects PBF-LB/M build file formats using multiple methods.
    
    This class provides format detection capabilities that leverage libSLM
    when available, with fallback methods for basic format detection.
    """
    
    def __init__(self):
        """Initialize the format detector."""
        self.LIBSLM_AVAILABLE = LIBSLM_AVAILABLE
        
        # Known format signatures
        self.format_signatures = {
            b'MTT': '.mtt',
            b'EOS': '.sli',
            b'SLM': '.slm',
            b'REA': '.rea',
            b'CLI': '.cli',
        }
        
        # File extension mappings
        self.extension_mappings = {
            '.mtt': 'Machine Tool Technology format',
            '.sli': 'EOS SLI format',
            '.cli': 'EOS CLI format',
            '.rea': 'Realizer format',
            '.slm': 'SLM Solutions format',
            '.f&s': 'F&S format',
        }
        
        logger.info(f"FormatDetector initialized - libSLM: {self.LIBSLM_AVAILABLE}")
    
    def detect_format(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Detect the format of a build file using multiple methods.
        
        Args:
            file_path: Path to the file to analyze
            
        Returns:
            Dictionary containing format detection results
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        detection_results = {
            'file_path': str(file_path),
            'file_extension': file_path.suffix.lower(),
            'detection_methods': [],
            'detected_format': None,
            'confidence': 0.0,
            'format_info': {}
        }
        
        # Method 1: File extension
        extension_result = self._detect_by_extension(file_path)
        detection_results['detection_methods'].append(extension_result)
        
        # Method 2: File signature (magic bytes)
        signature_result = self._detect_by_signature(file_path)
        detection_results['detection_methods'].append(signature_result)
        
        # Method 3: libSLM detection (if available)
        if self.LIBSLM_AVAILABLE:
            libslm_result = self._detect_by_libslm(file_path)
            detection_results['detection_methods'].append(libslm_result)
        
        # Determine final format based on results
        final_format = self._determine_final_format(detection_results['detection_methods'])
        detection_results['detected_format'] = final_format
        detection_results['confidence'] = self._calculate_confidence(detection_results['detection_methods'])
        
        # Get format information
        if final_format:
            detection_results['format_info'] = self._get_format_info(final_format)
        
        return detection_results
    
    def _detect_by_extension(self, file_path: Path) -> Dict[str, Any]:
        """Detect format by file extension."""
        extension = file_path.suffix.lower()
        
        result = {
            'method': 'file_extension',
            'detected_format': extension if extension in self.extension_mappings else None,
            'confidence': 0.8 if extension in self.extension_mappings else 0.0,
            'details': f"Extension: {extension}"
        }
        
        return result
    
    def _detect_by_signature(self, file_path: Path) -> Dict[str, Any]:
        """Detect format by file signature (magic bytes)."""
        try:
            with open(file_path, 'rb') as f:
                # Read first few bytes
                header = f.read(16)
                
                # Check for known signatures
                detected_format = None
                confidence = 0.0
                
                for signature, format_ext in self.format_signatures.items():
                    if header.startswith(signature):
                        detected_format = format_ext
                        confidence = 0.9
                        break
                
                # Check for common patterns
                if not detected_format:
                    # Check for binary vs text
                    if b'\x00' in header:
                        # Likely binary format
                        confidence = 0.3
                    else:
                        # Likely text format
                        confidence = 0.2
                
                result = {
                    'method': 'file_signature',
                    'detected_format': detected_format,
                    'confidence': confidence,
                    'details': f"Header: {header[:8].hex() if len(header) >= 8 else header.hex()}"
                }
                
                return result
                
        except Exception as e:
            return {
                'method': 'file_signature',
                'detected_format': None,
                'confidence': 0.0,
                'details': f"Error reading file: {e}"
            }
    
    def _detect_by_libslm(self, file_path: Path) -> Dict[str, Any]:
        """Detect format using libSLM capabilities."""
        try:
            if not self.LIBSLM_AVAILABLE:
                return {
                    'method': 'libslm',
                    'detected_format': None,
                    'confidence': 0.0,
                    'details': 'libSLM not available'
                }
            
            # Try to import libSLM modules
            import sys
            from pathlib import Path
            
            # Add libSLM to Python path
            external_dir = Path(__file__).parent.parent.parent / "external"
            libslm_path = external_dir / "libSLM" / "python" / "libSLM"
            if str(libslm_path) not in sys.path:
                sys.path.insert(0, str(libslm_path))
            
            import slm
            import translators
            
            # Try to detect format by attempting to read with different parsers
            detected_format = None
            confidence = 0.0
            
            # Try EOS parser for .sli/.cli files
            try:
                reader = translators.eos.Reader()
                # Just check if we can open the file, don't parse fully
                with open(file_path, 'rb') as f:
                    header = f.read(100)
                    if b'EOS' in header or b'SLI' in header:
                        detected_format = '.sli'
                        confidence = 0.95
            except:
                pass
            
            # Try MTT parser for .mtt files
            if not detected_format:
                try:
                    reader = translators.mtt.Reader()
                    with open(file_path, 'rb') as f:
                        header = f.read(100)
                        if b'MTT' in header:
                            detected_format = '.mtt'
                            confidence = 0.95
                except:
                    pass
            
            result = {
                'method': 'libslm',
                'detected_format': detected_format,
                'confidence': confidence,
                'details': f"libSLM detection successful" if detected_format else "libSLM detection inconclusive"
            }
            
            return result
            
        except Exception as e:
            return {
                'method': 'libslm',
                'detected_format': None,
                'confidence': 0.0,
                'details': f"libSLM detection error: {e}"
            }
    
    def _determine_final_format(self, detection_methods: List[Dict[str, Any]]) -> Optional[str]:
        """Determine the final format based on detection results."""
        if not detection_methods:
            return None
        
        # Find the method with highest confidence
        best_method = max(detection_methods, key=lambda x: x.get('confidence', 0.0))
        
        if best_method['confidence'] > 0.5:
            return best_method['detected_format']
        
        # If no high-confidence detection, use extension as fallback
        extension_method = next((m for m in detection_methods if m['method'] == 'file_extension'), None)
        if extension_method and extension_method['detected_format']:
            return extension_method['detected_format']
        
        return None
    
    def _calculate_confidence(self, detection_methods: List[Dict[str, Any]]) -> float:
        """Calculate overall confidence in format detection."""
        if not detection_methods:
            return 0.0
        
        # Weight different methods
        method_weights = {
            'libslm': 0.5,
            'file_signature': 0.3,
            'file_extension': 0.2
        }
        
        weighted_confidence = 0.0
        total_weight = 0.0
        
        for method in detection_methods:
            method_name = method['method']
            confidence = method.get('confidence', 0.0)
            weight = method_weights.get(method_name, 0.1)
            
            weighted_confidence += confidence * weight
            total_weight += weight
        
        return weighted_confidence / total_weight if total_weight > 0 else 0.0
    
    def _get_format_info(self, format_extension: str) -> Dict[str, Any]:
        """Get information about a detected format."""
        return {
            'extension': format_extension,
            'description': self.extension_mappings.get(format_extension, 'Unknown format'),
            'supported': format_extension in self.extension_mappings
        }
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported formats."""
        return list(self.extension_mappings.keys())
    
    def is_format_supported(self, format_extension: str) -> bool:
        """Check if a format is supported."""
        return format_extension.lower() in self.extension_mappings
