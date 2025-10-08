"""
Health checker for system health monitoring.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class HealthChecker(ABC):
    """Interface for health checking."""
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the health checker."""
        pass
    
    @abstractmethod
    async def check_health(self, component: Optional[str] = None) -> Dict[str, Any]:
        """Check system health."""
        pass
    
    @abstractmethod
    async def register_health_check(self, component: str, health_check_func: callable) -> None:
        """Register a health check function."""
        pass
    
    @abstractmethod
    async def get_health_status(self) -> Dict[str, Any]:
        """Get overall health status."""
        pass
    
    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the health checker."""
        pass
