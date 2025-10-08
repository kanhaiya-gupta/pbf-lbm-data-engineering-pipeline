"""
Alert manager for monitoring alerts.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class AlertManager(ABC):
    """Interface for alert management."""
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the alert manager."""
        pass
    
    @abstractmethod
    async def create_alert(self, alert_data: Dict[str, Any]) -> str:
        """Create an alert."""
        pass
    
    @abstractmethod
    async def resolve_alert(self, alert_id: str, resolution_data: Optional[Dict[str, Any]] = None) -> bool:
        """Resolve an alert."""
        pass
    
    @abstractmethod
    async def get_alert(self, alert_id: str) -> Optional[Dict[str, Any]]:
        """Get alert information."""
        pass
    
    @abstractmethod
    async def list_alerts(self, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """List alerts."""
        pass
    
    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the alert manager."""
        pass
