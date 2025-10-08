"""
Custom metrics for PBF-LB/M operations.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class CustomMetrics(ABC):
    """Interface for custom business metrics."""
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the custom metrics."""
        pass
    
    @abstractmethod
    async def record_metric(self, metric_name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a custom metric."""
        pass
    
    @abstractmethod
    async def increment_counter(self, counter_name: str, value: float = 1.0, tags: Optional[Dict[str, str]] = None) -> None:
        """Increment a custom counter."""
        pass
    
    @abstractmethod
    async def record_histogram(self, histogram_name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a custom histogram."""
        pass
    
    @abstractmethod
    async def get_metrics_summary(self) -> Dict[str, Any]:
        """Get custom metrics summary."""
        pass
    
    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the custom metrics."""
        pass
