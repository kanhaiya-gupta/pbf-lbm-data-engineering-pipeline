"""
Grafana client for dashboard integration.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class GrafanaClient(ABC):
    """Interface for Grafana integration."""
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the Grafana client."""
        pass
    
    @abstractmethod
    async def create_dashboard(self, dashboard_config: Dict[str, Any]) -> str:
        """Create a dashboard."""
        pass
    
    @abstractmethod
    async def update_dashboard(self, dashboard_id: str, dashboard_config: Dict[str, Any]) -> bool:
        """Update a dashboard."""
        pass
    
    @abstractmethod
    async def get_dashboard(self, dashboard_id: str) -> Optional[Dict[str, Any]]:
        """Get dashboard information."""
        pass
    
    @abstractmethod
    async def list_dashboards(self) -> List[Dict[str, Any]]:
        """List all dashboards."""
        pass
    
    @abstractmethod
    async def delete_dashboard(self, dashboard_id: str) -> bool:
        """Delete a dashboard."""
        pass
    
    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the Grafana client."""
        pass
