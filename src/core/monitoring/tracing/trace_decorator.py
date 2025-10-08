"""
Trace decorator for automatic function tracing.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional


class TraceDecorator(ABC):
    """Interface for trace decorator."""
    
    @abstractmethod
    def trace(self, operation_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None) -> Callable:
        """Create a trace decorator."""
        pass
    
    @abstractmethod
    async def trace_async(self, operation_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None) -> Callable:
        """Create an async trace decorator."""
        pass
