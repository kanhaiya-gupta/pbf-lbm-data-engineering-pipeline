"""
Base value object for PBF-LB/M domain.

Value objects are immutable objects that represent concepts
important to the domain but have no conceptual identity.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import json
from dataclasses import dataclass, asdict
from enum import Enum

from ..enums import DataModelType


@dataclass(frozen=True)
class BaseValueObject(ABC):
    """
    Base class for all value objects in the PBF-LB/M domain.
    
    Value objects are immutable and represent concepts without identity.
    They support serialization to multiple data models (SQL, NoSQL).
    """
    
    created_at: datetime
    updated_at: datetime
    
    def __post_init__(self):
        """Validate the value object after initialization."""
        self.validate()
    
    @abstractmethod
    def validate(self) -> None:
        """
        Validate the value object.
        
        Raises:
            ValueError: If validation fails
        """
        pass
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert value object to dictionary."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert value object to JSON string."""
        return json.dumps(self.to_dict(), default=self._json_serializer)
    
    def to_sql_dict(self) -> Dict[str, Any]:
        """Convert value object to SQL-compatible dictionary."""
        sql_dict = {}
        for key, value in self.to_dict().items():
            sql_dict[key] = self._convert_for_sql(value)
        return sql_dict
    
    def to_document_dict(self) -> Dict[str, Any]:
        """Convert value object to document (MongoDB) dictionary."""
        doc_dict = {}
        for key, value in self.to_dict().items():
            doc_dict[key] = self._convert_for_document(value)
        return doc_dict
    
    def to_key_value_dict(self) -> Dict[str, Any]:
        """Convert value object to key-value (Redis) dictionary."""
        kv_dict = {}
        for key, value in self.to_dict().items():
            kv_dict[key] = self._convert_for_key_value(value)
        return kv_dict
    
    def to_columnar_dict(self) -> Dict[str, Any]:
        """Convert value object to columnar (Cassandra) dictionary."""
        col_dict = {}
        for key, value in self.to_dict().items():
            col_dict[key] = self._convert_for_columnar(value)
        return col_dict
    
    def to_graph_dict(self) -> Dict[str, Any]:
        """Convert value object to graph (Neo4j) dictionary."""
        graph_dict = {}
        for key, value in self.to_dict().items():
            graph_dict[key] = self._convert_for_graph(value)
        return graph_dict
    
    def to_search_dict(self) -> Dict[str, Any]:
        """Convert value object to search (Elasticsearch) dictionary."""
        search_dict = {}
        for key, value in self.to_dict().items():
            search_dict[key] = self._convert_for_search(value)
        return search_dict
    
    def to_model_dict(self, model_type: DataModelType) -> Dict[str, Any]:
        """Convert value object to specific model type dictionary."""
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
    def from_dict(cls, data: Dict[str, Any]) -> 'BaseValueObject':
        """Create value object from dictionary."""
        # Convert datetime strings back to datetime objects
        if 'created_at' in data and isinstance(data['created_at'], str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        if 'updated_at' in data and isinstance(data['updated_at'], str):
            data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        
        return cls(**data)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'BaseValueObject':
        """Create value object from JSON string."""
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
        elif isinstance(value, BaseValueObject):
            return value.to_document_dict()
        else:
            return value
    
    def _convert_for_key_value(self, value: Any) -> Any:
        """Convert value for key-value storage (Redis)."""
        if isinstance(value, datetime):
            return value.isoformat()
        elif isinstance(value, Enum):
            return value.value
        elif isinstance(value, (list, dict, BaseValueObject)):
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
        elif isinstance(value, BaseValueObject):
            return value.to_graph_dict()
        else:
            return value
    
    def _convert_for_search(self, value: Any) -> Any:
        """Convert value for search storage (Elasticsearch)."""
        if isinstance(value, datetime):
            return value.isoformat()
        elif isinstance(value, Enum):
            return value.value
        elif isinstance(value, BaseValueObject):
            return value.to_search_dict()
        else:
            return value
    
    def get_field_names(self) -> List[str]:
        """Get list of field names in this value object."""
        return list(self.to_dict().keys())
    
    def get_field_value(self, field_name: str) -> Any:
        """Get value of a specific field."""
        return getattr(self, field_name, None)
    
    def has_field(self, field_name: str) -> bool:
        """Check if value object has a specific field."""
        return hasattr(self, field_name)
    
    def is_empty(self) -> bool:
        """Check if value object is empty (all fields are None or empty)."""
        for value in self.to_dict().values():
            if value is not None and value != "" and value != [] and value != {}:
                return False
        return True
    
    def __str__(self) -> str:
        """String representation of value object."""
        return f"{self.__class__.__name__}({self.to_dict()})"
    
    def __repr__(self) -> str:
        """Detailed string representation of value object."""
        return f"{self.__class__.__name__}({self.to_dict()})"
