"""
Multi-model repository interface for PBF-LB/M operations.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TypeVar, Generic
from datetime import datetime

from ...domain.entities.base_entity import BaseEntity
from ...domain.enums import DataModelType

T = TypeVar('T', bound=BaseEntity)


class MultiModelRepository(ABC, Generic[T]):
    """
    Multi-model repository interface for PBF-LB/M domain entities.
    
    This interface defines the contract for data access operations
    across multiple storage models (SQL, NoSQL, etc.) simultaneously.
    """
    
    @abstractmethod
    async def create_in_model(self, entity: T, model_type: DataModelType) -> T:
        """
        Create an entity in a specific data model.
        
        Args:
            entity: The entity to create
            model_type: The data model type to create in
            
        Returns:
            The created entity
            
        Raises:
            RepositoryException: If creation fails
        """
        pass
    
    @abstractmethod
    async def create_in_all_models(self, entity: T, model_types: List[DataModelType]) -> Dict[DataModelType, T]:
        """
        Create an entity in multiple data models.
        
        Args:
            entity: The entity to create
            model_types: List of data model types to create in
            
        Returns:
            Dictionary mapping model types to created entities
            
        Raises:
            RepositoryException: If creation fails in any model
        """
        pass
    
    @abstractmethod
    async def get_from_model(self, entity_id: str, model_type: DataModelType) -> Optional[T]:
        """
        Get an entity from a specific data model.
        
        Args:
            entity_id: The ID of the entity to retrieve
            model_type: The data model type to retrieve from
            
        Returns:
            The entity if found, None otherwise
            
        Raises:
            RepositoryException: If retrieval fails
        """
        pass
    
    @abstractmethod
    async def get_from_all_models(self, entity_id: str, model_types: List[DataModelType]) -> Dict[DataModelType, Optional[T]]:
        """
        Get an entity from multiple data models.
        
        Args:
            entity_id: The ID of the entity to retrieve
            model_types: List of data model types to retrieve from
            
        Returns:
            Dictionary mapping model types to entities (None if not found)
            
        Raises:
            RepositoryException: If retrieval fails in any model
        """
        pass
    
    @abstractmethod
    async def update_in_model(self, entity: T, model_type: DataModelType) -> T:
        """
        Update an entity in a specific data model.
        
        Args:
            entity: The entity to update
            model_type: The data model type to update in
            
        Returns:
            The updated entity
            
        Raises:
            RepositoryException: If update fails
        """
        pass
    
    @abstractmethod
    async def update_in_all_models(self, entity: T, model_types: List[DataModelType]) -> Dict[DataModelType, T]:
        """
        Update an entity in multiple data models.
        
        Args:
            entity: The entity to update
            model_types: List of data model types to update in
            
        Returns:
            Dictionary mapping model types to updated entities
            
        Raises:
            RepositoryException: If update fails in any model
        """
        pass
    
    @abstractmethod
    async def delete_from_model(self, entity_id: str, model_type: DataModelType) -> bool:
        """
        Delete an entity from a specific data model.
        
        Args:
            entity_id: The ID of the entity to delete
            model_type: The data model type to delete from
            
        Returns:
            True if deleted, False if not found
            
        Raises:
            RepositoryException: If deletion fails
        """
        pass
    
    @abstractmethod
    async def delete_from_all_models(self, entity_id: str, model_types: List[DataModelType]) -> Dict[DataModelType, bool]:
        """
        Delete an entity from multiple data models.
        
        Args:
            entity_id: The ID of the entity to delete
            model_types: List of data model types to delete from
            
        Returns:
            Dictionary mapping model types to deletion results
            
        Raises:
            RepositoryException: If deletion fails in any model
        """
        pass
    
    @abstractmethod
    async def search_across_models(self, filters: Dict[str, Any], model_types: List[DataModelType], limit: Optional[int] = None, offset: Optional[int] = None) -> Dict[DataModelType, List[T]]:
        """
        Search entities across multiple data models.
        
        Args:
            filters: Dictionary of field-value pairs to filter by
            model_types: List of data model types to search in
            limit: Maximum number of entities to return per model
            offset: Number of entities to skip per model
            
        Returns:
            Dictionary mapping model types to lists of matching entities
            
        Raises:
            RepositoryException: If search fails in any model
        """
        pass
    
    @abstractmethod
    async def sync_models(self, entity_id: str, source_model: DataModelType, target_models: List[DataModelType]) -> Dict[DataModelType, bool]:
        """
        Synchronize an entity from one model to others.
        
        Args:
            entity_id: The ID of the entity to synchronize
            source_model: The source data model type
            target_models: List of target data model types
            
        Returns:
            Dictionary mapping target model types to sync results
            
        Raises:
            RepositoryException: If synchronization fails
        """
        pass
    
    @abstractmethod
    async def get_model_consistency(self, entity_id: str, model_types: List[DataModelType]) -> Dict[DataModelType, Dict[str, Any]]:
        """
        Check consistency of an entity across multiple models.
        
        Args:
            entity_id: The ID of the entity to check
            model_types: List of data model types to check
            
        Returns:
            Dictionary mapping model types to consistency information
            
        Raises:
            RepositoryException: If consistency check fails
        """
        pass
    
    @abstractmethod
    async def get_optimal_model_for_operation(self, operation_type: str, filters: Dict[str, Any]) -> DataModelType:
        """
        Get the optimal data model for a specific operation.
        
        Args:
            operation_type: Type of operation (read, write, search, etc.)
            filters: Operation-specific filters or parameters
            
        Returns:
            The optimal data model type for the operation
        """
        pass
    
    @abstractmethod
    async def get_model_performance_metrics(self, model_types: List[DataModelType]) -> Dict[DataModelType, Dict[str, Any]]:
        """
        Get performance metrics for multiple data models.
        
        Args:
            model_types: List of data model types to get metrics for
            
        Returns:
            Dictionary mapping model types to performance metrics
        """
        pass
    
    @abstractmethod
    async def get_model_capabilities(self, model_types: List[DataModelType]) -> Dict[DataModelType, Dict[str, Any]]:
        """
        Get capabilities and features for multiple data models.
        
        Args:
            model_types: List of data model types to get capabilities for
            
        Returns:
            Dictionary mapping model types to capabilities
        """
        pass
    
    @abstractmethod
    async def get_supported_operations(self, model_type: DataModelType) -> List[str]:
        """
        Get supported operations for a specific data model.
        
        Args:
            model_type: The data model type to check
            
        Returns:
            List of supported operation types
        """
        pass
    
    @abstractmethod
    async def get_model_health_status(self, model_types: List[DataModelType]) -> Dict[DataModelType, Dict[str, Any]]:
        """
        Get health status for multiple data models.
        
        Args:
            model_types: List of data model types to check
            
        Returns:
            Dictionary mapping model types to health status
        """
        pass
    
    @abstractmethod
    async def get_cross_model_relationships(self, entity_id: str, model_types: List[DataModelType]) -> Dict[str, Any]:
        """
        Get relationships between entities across different models.
        
        Args:
            entity_id: The ID of the entity to get relationships for
            model_types: List of data model types to check
            
        Returns:
            Dictionary containing cross-model relationship information
        """
        pass
    
    @abstractmethod
    async def create_cross_model_relationship(self, source_entity_id: str, target_entity_id: str, relationship_type: str, model_types: List[DataModelType]) -> bool:
        """
        Create a relationship between entities across different models.
        
        Args:
            source_entity_id: ID of the source entity
            target_entity_id: ID of the target entity
            relationship_type: Type of relationship to create
            model_types: List of data model types to create relationship in
            
        Returns:
            True if relationship created successfully
            
        Raises:
            RepositoryException: If relationship creation fails
        """
        pass
    
    @abstractmethod
    async def get_model_statistics(self, model_types: List[DataModelType]) -> Dict[DataModelType, Dict[str, Any]]:
        """
        Get statistics for multiple data models.
        
        Args:
            model_types: List of data model types to get statistics for
            
        Returns:
            Dictionary mapping model types to statistics
        """
        pass
    
    @abstractmethod
    async def optimize_model_performance(self, model_type: DataModelType, optimization_type: str) -> Dict[str, Any]:
        """
        Optimize performance for a specific data model.
        
        Args:
            model_type: The data model type to optimize
            optimization_type: Type of optimization to perform
            
        Returns:
            Dictionary containing optimization results
        """
        pass
