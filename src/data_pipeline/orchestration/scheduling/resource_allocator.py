"""
Resource Allocator

This module manages resource allocation for the PBF-LB/M data pipeline jobs.
"""

from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime, timedelta
import threading

from src.data_pipeline.orchestration.scheduling.job_scheduler import Job, JobStatus, JobPriority

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResourceType(Enum):
    """Resource type enumeration."""
    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"
    DATABASE_CONNECTION = "database_connection"
    SPARK_CLUSTER = "spark_cluster"
    KAFKA_CONSUMER = "kafka_consumer"

@dataclass
class Resource:
    """Resource data class."""
    id: str
    name: str
    resource_type: ResourceType
    total_capacity: float
    available_capacity: float
    unit: str
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

@dataclass
class ResourceRequirement:
    """Resource requirement data class."""
    resource_type: ResourceType
    required_capacity: float
    unit: str
    priority: JobPriority = JobPriority.MEDIUM

@dataclass
class ResourceAllocation:
    """Resource allocation data class."""
    job_id: str
    resource_id: str
    allocated_capacity: float
    allocated_at: datetime
    expires_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.allocated_at is None:
            self.allocated_at = datetime.now()

@dataclass
class ResourcePool:
    """Resource pool data class."""
    id: str
    name: str
    description: str
    resources: List[Resource]
    max_concurrent_allocations: int
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

class ResourceAllocator:
    """
    Manages resource allocation for jobs.
    """
    
    def __init__(self):
        self.resources: Dict[str, Resource] = {}
        self.allocations: Dict[str, List[ResourceAllocation]] = {}  # job_id -> list of allocations
        self.resource_requirements: Dict[str, List[ResourceRequirement]] = {}  # job_id -> list of requirements
        self.allocation_lock = threading.Lock()
        
    def add_resource(self, resource: Resource) -> bool:
        """
        Add a resource to the allocator.
        
        Args:
            resource: The resource to add
            
        Returns:
            bool: True if resource was added successfully, False otherwise
        """
        try:
            if resource.id in self.resources:
                logger.warning(f"Resource {resource.id} already exists. Updating existing resource.")
            
            self.resources[resource.id] = resource
            logger.info(f"Added resource {resource.id}: {resource.name}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding resource {resource.id}: {e}")
            return False
    
    def remove_resource(self, resource_id: str) -> bool:
        """
        Remove a resource from the allocator.
        
        Args:
            resource_id: The ID of the resource to remove
            
        Returns:
            bool: True if resource was removed successfully, False otherwise
        """
        try:
            if resource_id in self.resources:
                # Check if resource is currently allocated
                if self._is_resource_allocated(resource_id):
                    logger.error(f"Cannot remove resource {resource_id}: currently allocated")
                    return False
                
                del self.resources[resource_id]
                logger.info(f"Removed resource {resource_id}")
                return True
            else:
                logger.warning(f"Resource {resource_id} not found")
                return False
                
        except Exception as e:
            logger.error(f"Error removing resource {resource_id}: {e}")
            return False
    
    def set_job_requirements(self, job_id: str, requirements: List[ResourceRequirement]) -> bool:
        """
        Set resource requirements for a job.
        
        Args:
            job_id: The job ID
            requirements: List of resource requirements
            
        Returns:
            bool: True if requirements were set successfully, False otherwise
        """
        try:
            self.resource_requirements[job_id] = requirements
            logger.info(f"Set resource requirements for job {job_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error setting resource requirements for job {job_id}: {e}")
            return False
    
    def allocate_resources(self, job_id: str, job_priority: JobPriority) -> bool:
        """
        Allocate resources for a job.
        
        Args:
            job_id: The job ID
            job_priority: The job priority
            
        Returns:
            bool: True if resources were allocated successfully, False otherwise
        """
        try:
            with self.allocation_lock:
                if job_id not in self.resource_requirements:
                    logger.error(f"No resource requirements set for job {job_id}")
                    return False
                
                requirements = self.resource_requirements[job_id]
                allocations = []
                
                # Try to allocate each required resource
                for requirement in requirements:
                    allocation = self._allocate_single_resource(job_id, requirement, job_priority)
                    if allocation:
                        allocations.append(allocation)
                    else:
                        # If any resource allocation fails, release all allocated resources
                        self._release_job_allocations(job_id)
                        logger.error(f"Failed to allocate resource {requirement.resource_type} for job {job_id}")
                        return False
                
                # Store allocations
                self.allocations[job_id] = allocations
                logger.info(f"Allocated resources for job {job_id}")
                return True
                
        except Exception as e:
            logger.error(f"Error allocating resources for job {job_id}: {e}")
            return False
    
    def release_resources(self, job_id: str) -> bool:
        """
        Release resources allocated to a job.
        
        Args:
            job_id: The job ID
            
        Returns:
            bool: True if resources were released successfully, False otherwise
        """
        try:
            with self.allocation_lock:
                return self._release_job_allocations(job_id)
                
        except Exception as e:
            logger.error(f"Error releasing resources for job {job_id}: {e}")
            return False
    
    def get_available_resources(self, resource_type: ResourceType) -> List[Resource]:
        """
        Get all available resources of a specific type.
        
        Args:
            resource_type: The type of resource to get
            
        Returns:
            List[Resource]: List of available resources
        """
        try:
            available_resources = []
            
            for resource in self.resources.values():
                if (resource.resource_type == resource_type and 
                    resource.available_capacity > 0):
                    available_resources.append(resource)
            
            return available_resources
            
        except Exception as e:
            logger.error(f"Error getting available resources: {e}")
            return []
    
    def get_resource_utilization(self, resource_id: str) -> float:
        """
        Get the utilization percentage of a resource.
        
        Args:
            resource_id: The resource ID
            
        Returns:
            float: Utilization percentage (0.0 to 1.0)
        """
        try:
            if resource_id not in self.resources:
                logger.error(f"Resource {resource_id} not found")
                return 0.0
            
            resource = self.resources[resource_id]
            used_capacity = resource.total_capacity - resource.available_capacity
            utilization = used_capacity / resource.total_capacity
            
            return min(utilization, 1.0)
            
        except Exception as e:
            logger.error(f"Error getting resource utilization: {e}")
            return 0.0
    
    def get_job_allocations(self, job_id: str) -> List[ResourceAllocation]:
        """
        Get resource allocations for a job.
        
        Args:
            job_id: The job ID
            
        Returns:
            List[ResourceAllocation]: List of resource allocations
        """
        return self.allocations.get(job_id, [])
    
    def get_allocator_status(self) -> Dict[str, any]:
        """
        Get the current status of the resource allocator.
        
        Returns:
            Dict[str, any]: Allocator status information
        """
        try:
            total_resources = len(self.resources)
            total_allocations = sum(len(allocations) for allocations in self.allocations.values())
            
            resource_utilization = {}
            for resource_id, resource in self.resources.items():
                utilization = self.get_resource_utilization(resource_id)
                resource_utilization[resource_id] = {
                    "name": resource.name,
                    "type": resource.resource_type.value,
                    "utilization": utilization,
                    "available_capacity": resource.available_capacity,
                    "total_capacity": resource.total_capacity
                }
            
            return {
                "total_resources": total_resources,
                "total_allocations": total_allocations,
                "resource_utilization": resource_utilization
            }
            
        except Exception as e:
            logger.error(f"Error getting allocator status: {e}")
            return {}
    
    def cleanup_expired_allocations(self):
        """Clean up expired resource allocations."""
        try:
            with self.allocation_lock:
                current_time = datetime.now()
                expired_jobs = []
                
                for job_id, allocations in self.allocations.items():
                    expired_allocations = []
                    
                    for allocation in allocations:
                        if allocation.expires_at and allocation.expires_at < current_time:
                            expired_allocations.append(allocation)
                    
                    # Remove expired allocations
                    for allocation in expired_allocations:
                        allocations.remove(allocation)
                        self._release_single_allocation(allocation)
                    
                    # If no allocations remain, mark job for cleanup
                    if not allocations:
                        expired_jobs.append(job_id)
                
                # Remove jobs with no allocations
                for job_id in expired_jobs:
                    del self.allocations[job_id]
                
                if expired_jobs:
                    logger.info(f"Cleaned up expired allocations for {len(expired_jobs)} jobs")
                    
        except Exception as e:
            logger.error(f"Error cleaning up expired allocations: {e}")
    
    def _allocate_single_resource(self, job_id: str, requirement: ResourceRequirement, job_priority: JobPriority) -> Optional[ResourceAllocation]:
        """
        Allocate a single resource for a job.
        
        Args:
            job_id: The job ID
            requirement: The resource requirement
            job_priority: The job priority
            
        Returns:
            ResourceAllocation: The allocation if successful, None otherwise
        """
        try:
            # Find available resources of the required type
            available_resources = self.get_available_resources(requirement.resource_type)
            
            if not available_resources:
                logger.warning(f"No available resources of type {requirement.resource_type}")
                return None
            
            # Sort resources by available capacity (descending) and priority
            available_resources.sort(key=lambda r: r.available_capacity, reverse=True)
            
            # Try to allocate from the best available resource
            for resource in available_resources:
                if resource.available_capacity >= requirement.required_capacity:
                    # Create allocation
                    allocation = ResourceAllocation(
                        job_id=job_id,
                        resource_id=resource.id,
                        allocated_capacity=requirement.required_capacity,
                        allocated_at=datetime.now(),
                        expires_at=datetime.now() + timedelta(hours=24)  # Default 24-hour expiration
                    )
                    
                    # Update resource availability
                    resource.available_capacity -= requirement.required_capacity
                    
                    logger.info(f"Allocated {requirement.required_capacity} {requirement.unit} of {requirement.resource_type.value} for job {job_id}")
                    return allocation
            
            logger.warning(f"Insufficient capacity for resource {requirement.resource_type}")
            return None
            
        except Exception as e:
            logger.error(f"Error allocating single resource: {e}")
            return None
    
    def _release_job_allocations(self, job_id: str) -> bool:
        """
        Release all resource allocations for a job.
        
        Args:
            job_id: The job ID
            
        Returns:
            bool: True if allocations were released successfully, False otherwise
        """
        try:
            if job_id not in self.allocations:
                logger.warning(f"No allocations found for job {job_id}")
                return True
            
            allocations = self.allocations[job_id]
            
            for allocation in allocations:
                self._release_single_allocation(allocation)
            
            del self.allocations[job_id]
            logger.info(f"Released all resources for job {job_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error releasing job allocations: {e}")
            return False
    
    def _release_single_allocation(self, allocation: ResourceAllocation):
        """
        Release a single resource allocation.
        
        Args:
            allocation: The allocation to release
        """
        try:
            if allocation.resource_id in self.resources:
                resource = self.resources[allocation.resource_id]
                resource.available_capacity += allocation.allocated_capacity
                
                logger.info(f"Released {allocation.allocated_capacity} of resource {allocation.resource_id}")
            
        except Exception as e:
            logger.error(f"Error releasing single allocation: {e}")
    
    def _is_resource_allocated(self, resource_id: str) -> bool:
        """
        Check if a resource is currently allocated.
        
        Args:
            resource_id: The resource ID
            
        Returns:
            bool: True if resource is allocated, False otherwise
        """
        try:
            for allocations in self.allocations.values():
                for allocation in allocations:
                    if allocation.resource_id == resource_id:
                        return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking if resource is allocated: {e}")
            return False


# Convenience functions
def create_resource_allocator(**kwargs) -> ResourceAllocator:
    """Create a resource allocator with custom configuration."""
    return ResourceAllocator(**kwargs)