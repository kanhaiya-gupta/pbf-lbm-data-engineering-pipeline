"""
Domain-specific exceptions for PBF-LB/M operations.
"""

from typing import Any, Dict, Optional


class DomainException(Exception):
    """Base exception for domain-related errors."""
    
    def __init__(self, message: str, error_code: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}


class ValidationException(DomainException):
    """Exception raised when validation fails."""
    pass


class RepositoryException(DomainException):
    """Exception raised when repository operations fail."""
    pass


class MonitoringException(DomainException):
    """Exception raised when monitoring operations fail."""
    pass


class DataQualityException(DomainException):
    """Exception raised when data quality operations fail."""
    pass


class ISPMException(DomainException):
    """Exception raised when ISPM operations fail."""
    pass


class CTScannerException(DomainException):
    """Exception raised when CT scanner operations fail."""
    pass


class PowderBedException(DomainException):
    """Exception raised when powder bed operations fail."""
    pass
