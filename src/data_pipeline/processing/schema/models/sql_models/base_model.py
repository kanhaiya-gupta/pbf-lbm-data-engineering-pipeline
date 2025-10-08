"""
Base Model

This module provides the base Pydantic model class for all PBF-LB/M data models.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union, List
from datetime import datetime
from pydantic import BaseModel, Field, validator, root_validator
import uuid
import json

class BaseDataModel(BaseModel, ABC):
    """
    Base Pydantic model for all PBF-LB/M data models.
    
    This class provides common functionality and validation for all data models
    in the PBF-LB/M data pipeline.
    """
    
    # Common fields for all models
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Record creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Record last update timestamp")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")
    
    class Config:
        """Pydantic configuration."""
        # Use enum values instead of enum names
        use_enum_values = True
        # Validate assignment
        validate_assignment = True
        # Allow population by field name
        allow_population_by_field_name = True
        # Use string representations for datetime
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
        # Generate example schema
        schema_extra = {
            "example": {}
        }
    
    @validator('created_at', 'updated_at', pre=True)
    def parse_datetime(cls, v):
        """Parse datetime strings to datetime objects."""
        if isinstance(v, str):
            try:
                return datetime.fromisoformat(v.replace('Z', '+00:00'))
            except ValueError:
                raise ValueError(f"Invalid datetime format: {v}")
        return v
    
    @validator('metadata', pre=True)
    def parse_metadata(cls, v):
        """Parse metadata to dictionary."""
        if isinstance(v, str):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                raise ValueError(f"Invalid JSON format for metadata: {v}")
        return v or {}
    
    @root_validator
    def validate_timestamps(cls, values):
        """Validate that updated_at is not before created_at."""
        created_at = values.get('created_at')
        updated_at = values.get('updated_at')
        
        if created_at and updated_at and updated_at < created_at:
            raise ValueError("updated_at cannot be before created_at")
        
        return values
    
    def update_timestamp(self):
        """Update the updated_at timestamp."""
        self.updated_at = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary."""
        return self.dict()
    
    def to_json(self) -> str:
        """Convert model to JSON string."""
        return self.json()
    
    def to_json_bytes(self) -> bytes:
        """Convert model to JSON bytes."""
        return self.json().encode('utf-8')
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BaseDataModel':
        """Create model instance from dictionary."""
        return cls(**data)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'BaseDataModel':
        """Create model instance from JSON string."""
        return cls.parse_raw(json_str)
    
    @classmethod
    def from_json_bytes(cls, json_bytes: bytes) -> 'BaseDataModel':
        """Create model instance from JSON bytes."""
        return cls.parse_raw(json_bytes.decode('utf-8'))
    
    def validate_data_quality(self) -> Dict[str, Any]:
        """
        Validate data quality for the model.
        
        Returns:
            Dict containing validation results and quality metrics.
        """
        quality_metrics = {
            'completeness': self._calculate_completeness(),
            'consistency': self._calculate_consistency(),
            'accuracy': self._calculate_accuracy(),
            'timeliness': self._calculate_timeliness(),
            'validity': self._calculate_validity()
        }
        
        overall_score = sum(quality_metrics.values()) / len(quality_metrics)
        
        return {
            'overall_score': overall_score,
            'quality_metrics': quality_metrics,
            'validation_timestamp': datetime.utcnow().isoformat(),
            'model_type': self.__class__.__name__
        }
    
    def _calculate_completeness(self) -> float:
        """Calculate data completeness score (0-1)."""
        fields = self.__fields__
        total_fields = len(fields)
        non_null_fields = sum(1 for field_name in fields 
                            if getattr(self, field_name) is not None)
        return non_null_fields / total_fields if total_fields > 0 else 0.0
    
    def _calculate_consistency(self) -> float:
        """Calculate data consistency score (0-1)."""
        # Base implementation - can be overridden by subclasses
        return 1.0
    
    def _calculate_accuracy(self) -> float:
        """Calculate data accuracy score (0-1)."""
        # Base implementation - can be overridden by subclasses
        return 1.0
    
    def _calculate_timeliness(self) -> float:
        """Calculate data timeliness score (0-1)."""
        if hasattr(self, 'timestamp') and self.timestamp:
            time_diff = datetime.utcnow() - self.timestamp
            # Score decreases with age (1 hour = 0.9, 1 day = 0.5, 1 week = 0.1)
            hours_old = time_diff.total_seconds() / 3600
            return max(0.0, 1.0 - (hours_old / 168))  # 168 hours = 1 week
        return 1.0
    
    def _calculate_validity(self) -> float:
        """Calculate data validity score (0-1)."""
        # Base implementation - can be overridden by subclasses
        return 1.0
    
    def get_field_info(self) -> Dict[str, Any]:
        """Get information about model fields."""
        return {
            field_name: {
                'type': field_info.type_,
                'required': field_info.is_required(),
                'default': field_info.default,
                'description': field_info.field_info.description
            }
            for field_name, field_info in self.__fields__.items()
        }
    
    def get_required_fields(self) -> List[str]:
        """Get list of required fields."""
        return [field_name for field_name, field_info in self.__fields__.items() 
                if field_info.is_required()]
    
    def get_optional_fields(self) -> List[str]:
        """Get list of optional fields."""
        return [field_name for field_name, field_info in self.__fields__.items() 
                if not field_info.is_required()]
    
    def has_field(self, field_name: str) -> bool:
        """Check if model has a specific field."""
        return field_name in self.__fields__
    
    def get_field_value(self, field_name: str) -> Any:
        """Get value of a specific field."""
        if not self.has_field(field_name):
            raise ValueError(f"Field '{field_name}' not found in model")
        return getattr(self, field_name)
    
    def set_field_value(self, field_name: str, value: Any) -> None:
        """Set value of a specific field."""
        if not self.has_field(field_name):
            raise ValueError(f"Field '{field_name}' not found in model")
        setattr(self, field_name, value)
        self.update_timestamp()
    
    def copy_with_updates(self, **updates) -> 'BaseDataModel':
        """Create a copy of the model with updated fields."""
        data = self.dict()
        data.update(updates)
        return self.__class__(**data)
    
    def merge_with(self, other: 'BaseDataModel') -> 'BaseDataModel':
        """Merge this model with another model of the same type."""
        if not isinstance(other, self.__class__):
            raise ValueError(f"Cannot merge {self.__class__.__name__} with {other.__class__.__name__}")
        
        data = self.dict()
        other_data = other.dict()
        
        # Merge data, with other taking precedence
        merged_data = {**data, **other_data}
        
        return self.__class__(**merged_data)
    
    @abstractmethod
    def get_primary_key(self) -> str:
        """Get the primary key field name."""
        pass
    
    @abstractmethod
    def get_primary_key_value(self) -> Any:
        """Get the primary key value."""
        pass
    
    def generate_id(self) -> str:
        """Generate a unique ID for the model."""
        return str(uuid.uuid4())
    
    def is_valid_for_storage(self) -> bool:
        """Check if the model is valid for storage."""
        try:
            # Validate the model
            self.validate(self.dict())
            return True
        except Exception:
            return False
    
    def get_storage_representation(self) -> Dict[str, Any]:
        """Get representation suitable for storage."""
        data = self.dict()
        # Convert datetime objects to ISO strings
        for key, value in data.items():
            if isinstance(value, datetime):
                data[key] = value.isoformat()
        return data
    
    def __str__(self) -> str:
        """String representation of the model."""
        return f"{self.__class__.__name__}({self.get_primary_key()}={self.get_primary_key_value()})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the model."""
        return f"{self.__class__.__name__}({self.dict()})"