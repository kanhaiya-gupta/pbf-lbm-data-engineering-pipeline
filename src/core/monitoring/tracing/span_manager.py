"""
Span manager for distributed tracing.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class SpanManager(ABC):
    """Interface for span management."""
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the span manager."""
        pass
    
    @abstractmethod
    async def start_span(self, operation_name: str, tags: Optional[Dict[str, str]] = None) -> str:
        """Start a new span."""
        pass
    
    @abstractmethod
    async def finish_span(self, span_id: str, tags: Optional[Dict[str, str]] = None) -> None:
        """Finish a span."""
        pass
    
    @abstractmethod
    async def add_tag(self, span_id: str, key: str, value: str) -> None:
        """Add a tag to a span."""
        pass
    
    @abstractmethod
    async def get_span_info(self, span_id: str) -> Optional[Dict[str, Any]]:
        """Get span information."""
        pass
    
    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the span manager."""
        pass
