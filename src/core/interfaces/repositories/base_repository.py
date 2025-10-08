"""
Base repository interface for PBF-LB/M operations.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TypeVar, Generic
from datetime import datetime

from ...domain.entities.base_entity import BaseEntity
from ...domain.enums import DataModelType

T = TypeVar('T', bound=BaseEntity)


class BaseRepository(ABC, Generic[T]):
    """
    Base repository interface for PBF-LB/M domain entities.
    
    This interface defines the contract for data access operations
    across different storage models and implementations.
    """
    
    @abstractmethod
    async def create(self, entity: T) -> T:
        """
        Create a new entity.
        
        Args:
            entity: The entity to create
            
        Returns:
            The created entity with generated ID and timestamps
            
        Raises:
            RepositoryException: If creation fails
        """
        pass
    
    @abstractmethod
    async def get_by_id(self, entity_id: str) -> Optional[T]:
        """
        Get an entity by its ID.
        
        Args:
            entity_id: The ID of the entity to retrieve
            
        Returns:
            The entity if found, None otherwise
            
        Raises:
            RepositoryException: If retrieval fails
        """
        pass
    
    @abstractmethod
    async def update(self, entity: T) -> T:
        """
        Update an existing entity.
        
        Args:
            entity: The entity to update
            
        Returns:
            The updated entity
            
        Raises:
            RepositoryException: If update fails
        """
        pass
    
    @abstractmethod
    async def delete(self, entity_id: str) -> bool:
        """
        Delete an entity by its ID.
        
        Args:
            entity_id: The ID of the entity to delete
            
        Returns:
            True if deleted, False if not found
            
        Raises:
            RepositoryException: If deletion fails
        """
        pass
    
    @abstractmethod
    async def list_all(self, limit: Optional[int] = None, offset: Optional[int] = None) -> List[T]:
        """
        List all entities with optional pagination.
        
        Args:
            limit: Maximum number of entities to return
            offset: Number of entities to skip
            
        Returns:
            List of entities
            
        Raises:
            RepositoryException: If listing fails
        """
        pass
    
    @abstractmethod
    async def count(self) -> int:
        """
        Get the total count of entities.
        
        Returns:
            Total count of entities
            
        Raises:
            RepositoryException: If count fails
        """
        pass
    
    @abstractmethod
    async def exists(self, entity_id: str) -> bool:
        """
        Check if an entity exists by its ID.
        
        Args:
            entity_id: The ID of the entity to check
            
        Returns:
            True if entity exists, False otherwise
            
        Raises:
            RepositoryException: If check fails
        """
        pass
    
    @abstractmethod
    async def search(self, filters: Dict[str, Any], limit: Optional[int] = None, offset: Optional[int] = None) -> List[T]:
        """
        Search entities by filters.
        
        Args:
            filters: Dictionary of field-value pairs to filter by
            limit: Maximum number of entities to return
            offset: Number of entities to skip
            
        Returns:
            List of matching entities
            
        Raises:
            RepositoryException: If search fails
        """
        pass
    
    @abstractmethod
    async def bulk_create(self, entities: List[T]) -> List[T]:
        """
        Create multiple entities in a single operation.
        
        Args:
            entities: List of entities to create
            
        Returns:
            List of created entities
            
        Raises:
            RepositoryException: If bulk creation fails
        """
        pass
    
    @abstractmethod
    async def bulk_update(self, entities: List[T]) -> List[T]:
        """
        Update multiple entities in a single operation.
        
        Args:
            entities: List of entities to update
            
        Returns:
            List of updated entities
            
        Raises:
            RepositoryException: If bulk update fails
        """
        pass
    
    @abstractmethod
    async def bulk_delete(self, entity_ids: List[str]) -> int:
        """
        Delete multiple entities by their IDs.
        
        Args:
            entity_ids: List of entity IDs to delete
            
        Returns:
            Number of entities deleted
            
        Raises:
            RepositoryException: If bulk deletion fails
        """
        pass
    
    @abstractmethod
    async def get_by_field(self, field_name: str, field_value: Any) -> List[T]:
        """
        Get entities by a specific field value.
        
        Args:
            field_name: Name of the field to search by
            field_value: Value to search for
            
        Returns:
            List of matching entities
            
        Raises:
            RepositoryException: If retrieval fails
        """
        pass
    
    @abstractmethod
    async def get_by_fields(self, field_filters: Dict[str, Any]) -> List[T]:
        """
        Get entities by multiple field values.
        
        Args:
            field_filters: Dictionary of field-value pairs to filter by
            
        Returns:
            List of matching entities
            
        Raises:
            RepositoryException: If retrieval fails
        """
        pass
    
    @abstractmethod
    async def get_created_after(self, timestamp: datetime) -> List[T]:
        """
        Get entities created after a specific timestamp.
        
        Args:
            timestamp: The timestamp to filter by
            
        Returns:
            List of entities created after the timestamp
            
        Raises:
            RepositoryException: If retrieval fails
        """
        pass
    
    @abstractmethod
    async def get_created_before(self, timestamp: datetime) -> List[T]:
        """
        Get entities created before a specific timestamp.
        
        Args:
            timestamp: The timestamp to filter by
            
        Returns:
            List of entities created before the timestamp
            
        Raises:
            RepositoryException: If retrieval fails
        """
        pass
    
    @abstractmethod
    async def get_updated_after(self, timestamp: datetime) -> List[T]:
        """
        Get entities updated after a specific timestamp.
        
        Args:
            timestamp: The timestamp to filter by
            
        Returns:
            List of entities updated after the timestamp
            
        Raises:
            RepositoryException: If retrieval fails
        """
        pass
    
    @abstractmethod
    async def get_updated_before(self, timestamp: datetime) -> List[T]:
        """
        Get entities updated before a specific timestamp.
        
        Args:
            timestamp: The timestamp to filter by
            
        Returns:
            List of entities updated before the timestamp
            
        Raises:
            RepositoryException: If retrieval fails
        """
        pass
    
    @abstractmethod
    async def get_active_entities(self) -> List[T]:
        """
        Get all active entities.
        
        Returns:
            List of active entities
            
        Raises:
            RepositoryException: If retrieval fails
        """
        pass
    
    @abstractmethod
    async def get_deleted_entities(self) -> List[T]:
        """
        Get all deleted entities.
        
        Returns:
            List of deleted entities
            
        Raises:
            RepositoryException: If retrieval fails
        """
        pass
    
    @abstractmethod
    async def soft_delete(self, entity_id: str) -> bool:
        """
        Soft delete an entity (mark as deleted without removing from storage).
        
        Args:
            entity_id: The ID of the entity to soft delete
            
        Returns:
            True if soft deleted, False if not found
            
        Raises:
            RepositoryException: If soft deletion fails
        """
        pass
    
    @abstractmethod
    async def restore(self, entity_id: str) -> bool:
        """
        Restore a soft-deleted entity.
        
        Args:
            entity_id: The ID of the entity to restore
            
        Returns:
            True if restored, False if not found
            
        Raises:
            RepositoryException: If restoration fails
        """
        pass
    
    @abstractmethod
    async def get_entity_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about entities in the repository.
        
        Returns:
            Dictionary containing entity statistics
            
        Raises:
            RepositoryException: If statistics retrieval fails
        """
        pass
    
    @abstractmethod
    async def get_supported_model_types(self) -> List[DataModelType]:
        """
        Get the data model types supported by this repository.
        
        Returns:
            List of supported data model types
        """
        pass
    
    @abstractmethod
    async def get_connection_info(self) -> Dict[str, Any]:
        """
        Get connection information for this repository.
        
        Returns:
            Dictionary containing connection information
        """
        pass
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on the repository.
        
        Returns:
            Dictionary containing health check results
        """
        pass
    
    @abstractmethod
    async def close(self) -> None:
        """
        Close the repository and clean up resources.
        """
        pass
