"""
DataDog client for APM integration.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class DataDogClient(ABC):
    """Interface for DataDog APM integration."""
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the DataDog client."""
        pass
    
    @abstractmethod
    async def record_metric(self, metric_name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a metric."""
        pass
    
    @abstractmethod
    async def increment_counter(self, counter_name: str, value: float = 1.0, tags: Optional[Dict[str, str]] = None) -> None:
        """Increment a counter."""
        pass
    
    @abstractmethod
    async def record_histogram(self, histogram_name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a histogram."""
        pass
    
    @abstractmethod
    async def create_event(self, event_data: Dict[str, Any]) -> str:
        """Create an event."""
        pass
    
    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the DataDog client."""
        pass
