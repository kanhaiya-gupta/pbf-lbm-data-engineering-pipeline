"""
Dashboard manager for centralized dashboard management.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class DashboardManager(ABC):
    """Interface for dashboard management."""
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the dashboard manager."""
        pass
    
    @abstractmethod
    async def get_dashboard_data(self, dashboard_name: str, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get dashboard data."""
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
    async def list_dashboards(self) -> List[Dict[str, Any]]:
        """List all dashboards."""
        pass
    
    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the dashboard manager."""
        pass
