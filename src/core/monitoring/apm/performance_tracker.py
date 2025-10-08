"""
Performance tracker for monitoring system performance.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class PerformanceTracker(ABC):
    """Interface for performance tracking."""
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the performance tracker."""
        pass
    
    @abstractmethod
    async def start_tracking(self, operation_name: str) -> str:
        """Start performance tracking."""
        pass
    
    @abstractmethod
    async def stop_tracking(self, tracking_id: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Stop performance tracking."""
        pass
    
    @abstractmethod
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        pass
    
    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the performance tracker."""
        pass
