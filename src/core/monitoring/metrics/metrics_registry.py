"""
Metrics registry for centralized metrics management.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class MetricsRegistry(ABC):
    """Interface for metrics registry."""
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the metrics registry."""
        pass
    
    @abstractmethod
    async def register_metric(self, metric_name: str, metric_type: str, description: str) -> None:
        """Register a new metric."""
        pass
    
    @abstractmethod
    async def get_metric(self, metric_name: str) -> Optional[Dict[str, Any]]:
        """Get metric information."""
        pass
    
    @abstractmethod
    async def list_metrics(self) -> List[str]:
        """List all registered metrics."""
        pass
    
    @abstractmethod
    async def get_metrics_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        pass
    
    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the metrics registry."""
        pass
