"""
Base domain entity for PBF-LB/M operations.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import json
import uuid
from dataclasses import dataclass, asdict, field
from enum import Enum

from ..enums import DataModelType


@dataclass
class BaseEntity(ABC):
    """
    Base class for all domain entities in the PBF-LB/M system.
    
    Domain entities represent the core business objects with identity
    and lifecycle in the PBF-LB/M manufacturing domain.
    """
    
    # Entity identification
    id: str = field()
    created_at: datetime = field()
    updated_at: datetime = field()
    
    # Entity metadata with defaults
    version: int = field(default=1)
    created_by: Optional[str] = field(default=None)
    updated_by: Optional[str] = field(default=None)
    
    # Entity state
    is_active: bool = field(default=True)
    is_deleted: bool = field(default=False)
    
    # Entity data
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize default values and validate."""
        # Data and metadata are now handled by default_factory
        self.validate()
    
    @abstractmethod
    def validate(self) -> None:
        """
        Validate the entity.
        
        Raises:
            ValueError: If validation fails
        """
        pass
    
    @classmethod
    def generate_id(cls) -> str:
        """Generate a unique entity ID."""
        return str(uuid.uuid4())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert entity to dictionary."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert entity to JSON string."""
        return json.dumps(self.to_dict(), default=self._json_serializer)
    
    def to_sql_dict(self) -> Dict[str, Any]:
        """Convert entity to SQL-compatible dictionary."""
        sql_dict = {}
        for key, value in self.to_dict().items():
            sql_dict[key] = self._convert_for_sql(value)
        return sql_dict
    
    def to_document_dict(self) -> Dict[str, Any]:
        """Convert entity to document (MongoDB) dictionary."""
        doc_dict = {}
        for key, value in self.to_dict().items():
            doc_dict[key] = self._convert_for_document(value)
        return doc_dict
    
    def to_key_value_dict(self) -> Dict[str, Any]:
        """Convert entity to key-value (Redis) dictionary."""
        kv_dict = {}
        for key, value in self.to_dict().items():
            kv_dict[key] = self._convert_for_key_value(value)
        return kv_dict
    
    def to_columnar_dict(self) -> Dict[str, Any]:
        """Convert entity to columnar (Cassandra) dictionary."""
        col_dict = {}
        for key, value in self.to_dict().items():
            col_dict[key] = self._convert_for_columnar(value)
        return col_dict
    
    def to_graph_dict(self) -> Dict[str, Any]:
        """Convert entity to graph (Neo4j) dictionary."""
        graph_dict = {}
        for key, value in self.to_dict().items():
            graph_dict[key] = self._convert_for_graph(value)
        return graph_dict
    
    def to_search_dict(self) -> Dict[str, Any]:
        """Convert entity to search (Elasticsearch) dictionary."""
        search_dict = {}
        for key, value in self.to_dict().items():
            search_dict[key] = self._convert_for_search(value)
        return search_dict
    
    def to_model_dict(self, model_type: DataModelType) -> Dict[str, Any]:
        """Convert entity to specific model type dictionary."""
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
    def from_dict(cls, data: Dict[str, Any]) -> 'BaseEntity':
        """Create entity from dictionary."""
        # Convert datetime strings back to datetime objects
        if 'created_at' in data and isinstance(data['created_at'], str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        if 'updated_at' in data and isinstance(data['updated_at'], str):
            data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        
        return cls(**data)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'BaseEntity':
        """Create entity from JSON string."""
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
        elif isinstance(value, BaseEntity):
            return value.to_document_dict()
        else:
            return value
    
    def _convert_for_key_value(self, value: Any) -> Any:
        """Convert value for key-value storage (Redis)."""
        if isinstance(value, datetime):
            return value.isoformat()
        elif isinstance(value, Enum):
            return value.value
        elif isinstance(value, (list, dict, BaseEntity)):
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
        elif isinstance(value, BaseEntity):
            return value.to_graph_dict()
        else:
            return value
    
    def _convert_for_search(self, value: Any) -> Any:
        """Convert value for search storage (Elasticsearch)."""
        if isinstance(value, datetime):
            return value.isoformat()
        elif isinstance(value, Enum):
            return value.value
        elif isinstance(value, BaseEntity):
            return value.to_search_dict()
        else:
            return value
    
    def update(self, **kwargs) -> 'BaseEntity':
        """Create a new entity instance with updated values."""
        current_dict = self.to_dict()
        current_dict.update(kwargs)
        current_dict['updated_at'] = datetime.utcnow()
        current_dict['version'] = self.version + 1
        
        return self.__class__(**current_dict)
    
    def mark_as_deleted(self, deleted_by: Optional[str] = None) -> 'BaseEntity':
        """Create a new entity instance marked as deleted."""
        return self.update(
            is_deleted=True,
            is_active=False,
            updated_by=deleted_by
        )
    
    def mark_as_active(self, updated_by: Optional[str] = None) -> 'BaseEntity':
        """Create a new entity instance marked as active."""
        return self.update(
            is_active=True,
            is_deleted=False,
            updated_by=updated_by
        )
    
    def get_field_names(self) -> List[str]:
        """Get list of field names in this entity."""
        return list(self.to_dict().keys())
    
    def get_field_value(self, field_name: str) -> Any:
        """Get value of a specific field."""
        return getattr(self, field_name, None)
    
    def has_field(self, field_name: str) -> bool:
        """Check if entity has a specific field."""
        return hasattr(self, field_name)
    
    def is_empty(self) -> bool:
        """Check if entity is empty (all fields are None or empty)."""
        for value in self.to_dict().values():
            if value is not None and value != "" and value != [] and value != {}:
                return False
        return True
    
    def get_age_seconds(self) -> float:
        """Get age of entity in seconds."""
        return (datetime.utcnow() - self.created_at).total_seconds()
    
    def get_age_minutes(self) -> float:
        """Get age of entity in minutes."""
        return self.get_age_seconds() / 60
    
    def get_age_hours(self) -> float:
        """Get age of entity in hours."""
        return self.get_age_minutes() / 60
    
    def get_age_days(self) -> float:
        """Get age of entity in days."""
        return self.get_age_hours() / 24
    
    def is_stale(self, max_age_hours: float = 24) -> bool:
        """Check if entity is stale (older than max_age_hours)."""
        return self.get_age_hours() > max_age_hours
    
    def get_entity_summary(self) -> Dict[str, Any]:
        """Get comprehensive entity summary."""
        return {
            "id": self.id,
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "created_by": self.created_by,
            "updated_by": self.updated_by,
            "is_active": self.is_active,
            "is_deleted": self.is_deleted,
            "age_seconds": self.get_age_seconds(),
            "age_minutes": self.get_age_minutes(),
            "age_hours": self.get_age_hours(),
            "age_days": self.get_age_days(),
            "is_stale": self.is_stale(),
            "data_keys": list(self.data.keys()) if self.data else [],
            "metadata_keys": list(self.metadata.keys()) if self.metadata else []
        }
    
    def __str__(self) -> str:
        """String representation of entity."""
        return f"{self.__class__.__name__}(id={self.id}, version={self.version})"
    
    def __repr__(self) -> str:
        """Detailed string representation of entity."""
        return f"{self.__class__.__name__}(id={self.id}, version={self.version}, created_at={self.created_at}, updated_at={self.updated_at})"
