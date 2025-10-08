"""
Prometheus client for metrics collection.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class PrometheusClient(ABC):
    """Interface for Prometheus metrics collection."""
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the Prometheus client."""
        pass
    
    @abstractmethod
    async def record_metric(self, metric_name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a metric value."""
        pass
    
    @abstractmethod
    async def increment_counter(self, counter_name: str, value: float = 1.0, tags: Optional[Dict[str, str]] = None) -> None:
        """Increment a counter metric."""
        pass
    
    @abstractmethod
    async def record_histogram(self, histogram_name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a histogram value."""
        pass
    
    @abstractmethod
    async def get_metrics_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        pass
    
    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the Prometheus client."""
        pass
