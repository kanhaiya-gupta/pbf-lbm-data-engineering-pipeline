"""
Jaeger client for distributed tracing.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class JaegerClient(ABC):
    """Interface for Jaeger distributed tracing."""
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the Jaeger client."""
        pass
    
    @abstractmethod
    async def start_trace(self, operation_name: str, tags: Optional[Dict[str, str]] = None) -> str:
        """Start a new trace."""
        pass
    
    @abstractmethod
    async def finish_trace(self, trace_id: str, tags: Optional[Dict[str, str]] = None) -> None:
        """Finish a trace."""
        pass
    
    @abstractmethod
    async def add_trace_tag(self, trace_id: str, key: str, value: str) -> None:
        """Add a tag to a trace."""
        pass
    
    @abstractmethod
    async def get_trace_info(self, trace_id: str) -> Optional[Dict[str, Any]]:
        """Get trace information."""
        pass
    
    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the Jaeger client."""
        pass
