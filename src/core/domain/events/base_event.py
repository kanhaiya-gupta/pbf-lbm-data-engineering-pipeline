"""
Base domain event for PBF-LB/M operations.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import json
import uuid
from dataclasses import dataclass, asdict
from enum import Enum

from ..enums import DataModelType


class EventType(Enum):
    """Event type enumeration."""
    DOMAIN = "domain"
    INTEGRATION = "integration"
    TECHNICAL = "technical"
    BUSINESS = "business"


class EventSeverity(Enum):
    """Event severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass(frozen=True)
class BaseEvent(ABC):
    """
    Base class for all domain events in the PBF-LB/M system.
    
    Domain events represent important business occurrences that
    other parts of the system need to know about.
    """
    
    # Event identification
    event_id: str
    event_type: EventType
    event_name: str
    severity: EventSeverity
    
    # Event metadata
    occurred_at: datetime
    source: str  # Component that generated the event
    version: str = "1.0"
    
    # Event data
    data: Dict[str, Any] = None
    metadata: Dict[str, Any] = None
    
    # Correlation and causation
    correlation_id: Optional[str] = None
    causation_id: Optional[str] = None
    parent_event_id: Optional[str] = None
    
    # Processing information
    processed: bool = False
    processed_at: Optional[datetime] = None
    processing_errors: List[str] = None
    
    def __post_init__(self):
        """Initialize default values and validate."""
        if self.data is None:
            object.__setattr__(self, 'data', {})
        if self.metadata is None:
            object.__setattr__(self, 'metadata', {})
        if self.processing_errors is None:
            object.__setattr__(self, 'processing_errors', [])
        self.validate()
    
    @abstractmethod
    def validate(self) -> None:
        """
        Validate the event.
        
        Raises:
            ValueError: If validation fails
        """
        pass
    
    @classmethod
    def generate_event_id(cls) -> str:
        """Generate a unique event ID."""
        return str(uuid.uuid4())
    
    @classmethod
    def generate_correlation_id(cls) -> str:
        """Generate a unique correlation ID."""
        return str(uuid.uuid4())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert event to JSON string."""
        return json.dumps(self.to_dict(), default=self._json_serializer)
    
    def to_sql_dict(self) -> Dict[str, Any]:
        """Convert event to SQL-compatible dictionary."""
        sql_dict = {}
        for key, value in self.to_dict().items():
            sql_dict[key] = self._convert_for_sql(value)
        return sql_dict
    
    def to_document_dict(self) -> Dict[str, Any]:
        """Convert event to document (MongoDB) dictionary."""
        doc_dict = {}
        for key, value in self.to_dict().items():
            doc_dict[key] = self._convert_for_document(value)
        return doc_dict
    
    def to_key_value_dict(self) -> Dict[str, Any]:
        """Convert event to key-value (Redis) dictionary."""
        kv_dict = {}
        for key, value in self.to_dict().items():
            kv_dict[key] = self._convert_for_key_value(value)
        return kv_dict
    
    def to_columnar_dict(self) -> Dict[str, Any]:
        """Convert event to columnar (Cassandra) dictionary."""
        col_dict = {}
        for key, value in self.to_dict().items():
            col_dict[key] = self._convert_for_columnar(value)
        return col_dict
    
    def to_graph_dict(self) -> Dict[str, Any]:
        """Convert event to graph (Neo4j) dictionary."""
        graph_dict = {}
        for key, value in self.to_dict().items():
            graph_dict[key] = self._convert_for_graph(value)
        return graph_dict
    
    def to_search_dict(self) -> Dict[str, Any]:
        """Convert event to search (Elasticsearch) dictionary."""
        search_dict = {}
        for key, value in self.to_dict().items():
            search_dict[key] = self._convert_for_search(value)
        return search_dict
    
    def to_model_dict(self, model_type: DataModelType) -> Dict[str, Any]:
        """Convert event to specific model type dictionary."""
        conversion_map = {
            DataModelType.SQL: self.to_sql_dict,
            DataModelType.RELATIONAL: self.to_sql_dict,
            DataModelType.DOCUMENT: self.to_document_dict,
            DataModelType.MONGODB: self.to_document_dict,
            DataModelType.KEY_VALUE: self.to_key_value_dict,
            DataModelType.REDIS: self.to_key_value_dict,
            DataModelType.COLUMNAR: self.to_columnar_dict,
            DataModelType.CASSANDRA: self.to_columnar_dict,
            DataModelType.GRAPH: self.to_graph_dict,
            DataModelType.NEO4J: self.to_graph_dict,
            DataModelType.SEARCH: self.to_search_dict,
            DataModelType.ELASTICSEARCH: self.to_search_dict,
        }
        
        converter = conversion_map.get(model_type, self.to_dict)
        return converter()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BaseEvent':
        """Create event from dictionary."""
        # Convert datetime strings back to datetime objects
        if 'occurred_at' in data and isinstance(data['occurred_at'], str):
            data['occurred_at'] = datetime.fromisoformat(data['occurred_at'])
        if 'processed_at' in data and isinstance(data['processed_at'], str):
            data['processed_at'] = datetime.fromisoformat(data['processed_at'])
        
        return cls(**data)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'BaseEvent':
        """Create event from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    def _json_serializer(self, obj: Any) -> Any:
        """Custom JSON serializer for datetime and other objects."""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, Enum):
            return obj.value
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        else:
            return str(obj)
    
    def _convert_for_sql(self, value: Any) -> Any:
        """Convert value for SQL storage."""
        if isinstance(value, datetime):
            return value
        elif isinstance(value, Enum):
            return value.value
        elif isinstance(value, (list, dict)):
            return json.dumps(value)
        else:
            return value
    
    def _convert_for_document(self, value: Any) -> Any:
        """Convert value for document storage (MongoDB)."""
        if isinstance(value, datetime):
            return value
        elif isinstance(value, Enum):
            return value.value
        elif isinstance(value, BaseEvent):
            return value.to_document_dict()
        else:
            return value
    
    def _convert_for_key_value(self, value: Any) -> Any:
        """Convert value for key-value storage (Redis)."""
        if isinstance(value, datetime):
            return value.isoformat()
        elif isinstance(value, Enum):
            return value.value
        elif isinstance(value, (list, dict, BaseEvent)):
            return json.dumps(value, default=self._json_serializer)
        else:
            return str(value)
    
    def _convert_for_columnar(self, value: Any) -> Any:
        """Convert value for columnar storage (Cassandra)."""
        if isinstance(value, datetime):
            return value
        elif isinstance(value, Enum):
            return value.value
        elif isinstance(value, (list, dict)):
            return json.dumps(value)
        else:
            return value
    
    def _convert_for_graph(self, value: Any) -> Any:
        """Convert value for graph storage (Neo4j)."""
        if isinstance(value, datetime):
            return value.isoformat()
        elif isinstance(value, Enum):
            return value.value
        elif isinstance(value, BaseEvent):
            return value.to_graph_dict()
        else:
            return value
    
    def _convert_for_search(self, value: Any) -> Any:
        """Convert value for search storage (Elasticsearch)."""
        if isinstance(value, datetime):
            return value.isoformat()
        elif isinstance(value, Enum):
            return value.value
        elif isinstance(value, BaseEvent):
            return value.to_search_dict()
        else:
            return value
    
    def is_critical(self) -> bool:
        """Check if event is critical."""
        return self.severity == EventSeverity.CRITICAL
    
    def is_error(self) -> bool:
        """Check if event is an error."""
        return self.severity in [EventSeverity.ERROR, EventSeverity.CRITICAL]
    
    def is_warning(self) -> bool:
        """Check if event is a warning."""
        return self.severity == EventSeverity.WARNING
    
    def is_info(self) -> bool:
        """Check if event is informational."""
        return self.severity == EventSeverity.INFO
    
    def requires_immediate_attention(self) -> bool:
        """Check if event requires immediate attention."""
        return self.is_critical() or self.is_error()
    
    def get_age_seconds(self) -> float:
        """Get age of event in seconds."""
        return (datetime.utcnow() - self.occurred_at).total_seconds()
    
    def get_age_minutes(self) -> float:
        """Get age of event in minutes."""
        return self.get_age_seconds() / 60
    
    def get_age_hours(self) -> float:
        """Get age of event in hours."""
        return self.get_age_minutes() / 60
    
    def is_stale(self, max_age_hours: float = 24) -> bool:
        """Check if event is stale (older than max_age_hours)."""
        return self.get_age_hours() > max_age_hours
    
    def mark_as_processed(self, processed_at: Optional[datetime] = None) -> 'BaseEvent':
        """Create a new event instance marked as processed."""
        if processed_at is None:
            processed_at = datetime.utcnow()
        
        # Create new instance with processed flag
        event_dict = self.to_dict()
        event_dict['processed'] = True
        event_dict['processed_at'] = processed_at
        
        return self.__class__(**event_dict)
    
    def add_processing_error(self, error: str) -> 'BaseEvent':
        """Create a new event instance with processing error added."""
        event_dict = self.to_dict()
        event_dict['processing_errors'] = self.processing_errors + [error]
        
        return self.__class__(**event_dict)
    
    def get_event_summary(self) -> Dict[str, Any]:
        """Get comprehensive event summary."""
        return {
            "event_id": self.event_id,
            "event_name": self.event_name,
            "event_type": self.event_type.value,
            "severity": self.severity.value,
            "occurred_at": self.occurred_at.isoformat(),
            "source": self.source,
            "correlation_id": self.correlation_id,
            "causation_id": self.causation_id,
            "processed": self.processed,
            "processed_at": self.processed_at.isoformat() if self.processed_at else None,
            "age_seconds": self.get_age_seconds(),
            "age_minutes": self.get_age_minutes(),
            "age_hours": self.get_age_hours(),
            "is_critical": self.is_critical(),
            "is_error": self.is_error(),
            "is_warning": self.is_warning(),
            "is_info": self.is_info(),
            "requires_attention": self.requires_immediate_attention(),
            "is_stale": self.is_stale(),
            "processing_errors": self.processing_errors,
            "data_keys": list(self.data.keys()) if self.data else [],
            "metadata_keys": list(self.metadata.keys()) if self.metadata else []
        }
    
    def __str__(self) -> str:
        """String representation of event."""
        return f"{self.__class__.__name__}({self.event_name}, {self.severity.value}, {self.occurred_at})"
    
    def __repr__(self) -> str:
        """Detailed string representation of event."""
        return f"{self.__class__.__name__}(id={self.event_id}, name={self.event_name}, severity={self.severity.value}, occurred_at={self.occurred_at})"
