"""
Dependency Manager

This module manages job dependencies and execution order for the PBF-LB/M data pipeline.
"""

from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime

from src.data_pipeline.orchestration.scheduling.job_scheduler import Job, JobStatus

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DependencyType(Enum):
    """Dependency type enumeration."""
    SEQUENTIAL = "sequential"  # Job B must complete before Job A starts
    PARALLEL = "parallel"      # Jobs can run in parallel
    CONDITIONAL = "conditional"  # Job A runs only if condition is met
    RESOURCE = "resource"      # Jobs share resources

@dataclass
class Dependency:
    """Dependency data class."""
    source_job_id: str
    target_job_id: str
    dependency_type: DependencyType
    condition: Optional[str] = None
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

class DependencyManager:
    """
    Manages job dependencies and execution order.
    """
    
    def __init__(self):
        self.dependencies: List[Dependency] = []
        self.dependency_graph: Dict[str, Set[str]] = {}  # job_id -> set of dependent job IDs
        self.reverse_dependency_graph: Dict[str, Set[str]] = {}  # job_id -> set of prerequisite job IDs
        self.execution_order: List[List[str]] = []  # List of job ID lists for each execution level
        
    def add_dependency(self, dependency: Dependency) -> bool:
        """
        Add a dependency between jobs.
        
        Args:
            dependency: The dependency to add
            
        Returns:
            bool: True if dependency was added successfully, False otherwise
        """
        try:
            # Check for circular dependencies
            if self._would_create_circular_dependency(dependency):
                logger.error(f"Adding dependency {dependency.source_job_id} -> {dependency.target_job_id} would create a circular dependency")
                return False
            
            # Add dependency
            self.dependencies.append(dependency)
            
            # Update dependency graphs
            self._update_dependency_graphs(dependency)
            
            # Recalculate execution order
            self._calculate_execution_order()
            
            logger.info(f"Added dependency: {dependency.source_job_id} -> {dependency.target_job_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding dependency: {e}")
            return False
    
    def remove_dependency(self, source_job_id: str, target_job_id: str) -> bool:
        """
        Remove a dependency between jobs.
        
        Args:
            source_job_id: The source job ID
            target_job_id: The target job ID
            
        Returns:
            bool: True if dependency was removed successfully, False otherwise
        """
        try:
            # Find and remove dependency
            dependency_to_remove = None
            for dependency in self.dependencies:
                if (dependency.source_job_id == source_job_id and 
                    dependency.target_job_id == target_job_id):
                    dependency_to_remove = dependency
                    break
            
            if dependency_to_remove:
                self.dependencies.remove(dependency_to_remove)
                
                # Update dependency graphs
                self._rebuild_dependency_graphs()
                
                # Recalculate execution order
                self._calculate_execution_order()
                
                logger.info(f"Removed dependency: {source_job_id} -> {target_job_id}")
                return True
            else:
                logger.warning(f"Dependency {source_job_id} -> {target_job_id} not found")
                return False
                
        except Exception as e:
            logger.error(f"Error removing dependency: {e}")
            return False
    
    def get_dependencies(self, job_id: str) -> List[Dependency]:
        """
        Get all dependencies for a specific job.
        
        Args:
            job_id: The job ID to get dependencies for
            
        Returns:
            List[Dependency]: List of dependencies for the job
        """
        return [dep for dep in self.dependencies if dep.target_job_id == job_id]
    
    def get_prerequisites(self, job_id: str) -> List[str]:
        """
        Get all prerequisite job IDs for a specific job.
        
        Args:
            job_id: The job ID to get prerequisites for
            
        Returns:
            List[str]: List of prerequisite job IDs
        """
        return list(self.reverse_dependency_graph.get(job_id, set()))
    
    def get_dependents(self, job_id: str) -> List[str]:
        """
        Get all dependent job IDs for a specific job.
        
        Args:
            job_id: The job ID to get dependents for
            
        Returns:
            List[str]: List of dependent job IDs
        """
        return list(self.dependency_graph.get(job_id, set()))
    
    def can_execute_job(self, job_id: str, job_statuses: Dict[str, JobStatus]) -> bool:
        """
        Check if a job can be executed based on its dependencies.
        
        Args:
            job_id: The job ID to check
            job_statuses: Dictionary of job ID to status mappings
            
        Returns:
            bool: True if job can be executed, False otherwise
        """
        try:
            prerequisites = self.get_prerequisites(job_id)
            
            for prereq_id in prerequisites:
                prereq_status = job_statuses.get(prereq_id)
                if prereq_status != JobStatus.COMPLETED:
                    logger.info(f"Job {job_id} cannot execute: prerequisite {prereq_id} status is {prereq_status}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking if job {job_id} can execute: {e}")
            return False
    
    def get_execution_order(self) -> List[List[str]]:
        """
        Get the execution order for all jobs.
        
        Returns:
            List[List[str]]: List of job ID lists for each execution level
        """
        return self.execution_order.copy()
    
    def get_next_executable_jobs(self, job_statuses: Dict[str, JobStatus]) -> List[str]:
        """
        Get the next jobs that can be executed.
        
        Args:
            job_statuses: Dictionary of job ID to status mappings
            
        Returns:
            List[str]: List of job IDs that can be executed next
        """
        try:
            executable_jobs = []
            
            for job_id, status in job_statuses.items():
                if status == JobStatus.PENDING and self.can_execute_job(job_id, job_statuses):
                    executable_jobs.append(job_id)
            
            return executable_jobs
            
        except Exception as e:
            logger.error(f"Error getting next executable jobs: {e}")
            return []
    
    def get_dependency_chain(self, job_id: str) -> List[str]:
        """
        Get the complete dependency chain for a job.
        
        Args:
            job_id: The job ID to get dependency chain for
            
        Returns:
            List[str]: List of job IDs in dependency order
        """
        try:
            chain = []
            visited = set()
            
            def _build_chain(current_job_id: str):
                if current_job_id in visited:
                    return
                
                visited.add(current_job_id)
                prerequisites = self.get_prerequisites(current_job_id)
                
                for prereq_id in prerequisites:
                    _build_chain(prereq_id)
                
                chain.append(current_job_id)
            
            _build_chain(job_id)
            return chain
            
        except Exception as e:
            logger.error(f"Error getting dependency chain for job {job_id}: {e}")
            return []
    
    def validate_dependencies(self) -> List[str]:
        """
        Validate all dependencies and return any issues.
        
        Returns:
            List[str]: List of validation error messages
        """
        try:
            issues = []
            
            # Check for circular dependencies
            if self._has_circular_dependencies():
                issues.append("Circular dependencies detected")
            
            # Check for orphaned dependencies
            for dependency in self.dependencies:
                if not self._job_exists(dependency.source_job_id):
                    issues.append(f"Dependency references non-existent source job: {dependency.source_job_id}")
                if not self._job_exists(dependency.target_job_id):
                    issues.append(f"Dependency references non-existent target job: {dependency.target_job_id}")
            
            return issues
            
        except Exception as e:
            logger.error(f"Error validating dependencies: {e}")
            return [f"Error validating dependencies: {e}"]
    
    def _would_create_circular_dependency(self, new_dependency: Dependency) -> bool:
        """
        Check if adding a new dependency would create a circular dependency.
        
        Args:
            new_dependency: The new dependency to check
            
        Returns:
            bool: True if circular dependency would be created, False otherwise
        """
        try:
            # Temporarily add the dependency
            self.dependencies.append(new_dependency)
            self._update_dependency_graphs(new_dependency)
            
            # Check for circular dependencies
            has_circular = self._has_circular_dependencies()
            
            # Remove the temporary dependency
            self.dependencies.remove(new_dependency)
            self._rebuild_dependency_graphs()
            
            return has_circular
            
        except Exception as e:
            logger.error(f"Error checking for circular dependency: {e}")
            return True
    
    def _has_circular_dependencies(self) -> bool:
        """
        Check if there are any circular dependencies.
        
        Returns:
            bool: True if circular dependencies exist, False otherwise
        """
        try:
            visited = set()
            rec_stack = set()
            
            def _has_cycle(job_id: str) -> bool:
                visited.add(job_id)
                rec_stack.add(job_id)
                
                dependents = self.get_dependents(job_id)
                for dependent_id in dependents:
                    if dependent_id not in visited:
                        if _has_cycle(dependent_id):
                            return True
                    elif dependent_id in rec_stack:
                        return True
                
                rec_stack.remove(job_id)
                return False
            
            for job_id in self.dependency_graph.keys():
                if job_id not in visited:
                    if _has_cycle(job_id):
                        return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking for circular dependencies: {e}")
            return True
    
    def _update_dependency_graphs(self, dependency: Dependency):
        """
        Update dependency graphs with a new dependency.
        
        Args:
            dependency: The dependency to add to the graphs
        """
        try:
            # Update forward dependency graph
            if dependency.source_job_id not in self.dependency_graph:
                self.dependency_graph[dependency.source_job_id] = set()
            self.dependency_graph[dependency.source_job_id].add(dependency.target_job_id)
            
            # Update reverse dependency graph
            if dependency.target_job_id not in self.reverse_dependency_graph:
                self.reverse_dependency_graph[dependency.target_job_id] = set()
            self.reverse_dependency_graph[dependency.target_job_id].add(dependency.source_job_id)
            
        except Exception as e:
            logger.error(f"Error updating dependency graphs: {e}")
    
    def _rebuild_dependency_graphs(self):
        """Rebuild dependency graphs from scratch."""
        try:
            self.dependency_graph.clear()
            self.reverse_dependency_graph.clear()
            
            for dependency in self.dependencies:
                self._update_dependency_graphs(dependency)
                
        except Exception as e:
            logger.error(f"Error rebuilding dependency graphs: {e}")
    
    def _calculate_execution_order(self):
        """
        Calculate the execution order for all jobs using topological sorting.
        """
        try:
            self.execution_order.clear()
            
            # Get all job IDs
            all_jobs = set()
            for dependency in self.dependencies:
                all_jobs.add(dependency.source_job_id)
                all_jobs.add(dependency.target_job_id)
            
            # Calculate in-degrees
            in_degree = {job_id: 0 for job_id in all_jobs}
            for dependency in self.dependencies:
                in_degree[dependency.target_job_id] += 1
            
            # Find jobs with no dependencies
            queue = [job_id for job_id in all_jobs if in_degree[job_id] == 0]
            
            while queue:
                # Add current level to execution order
                self.execution_order.append(queue.copy())
                
                # Process current level
                next_queue = []
                for job_id in queue:
                    dependents = self.get_dependents(job_id)
                    for dependent_id in dependents:
                        in_degree[dependent_id] -= 1
                        if in_degree[dependent_id] == 0:
                            next_queue.append(dependent_id)
                
                queue = next_queue
            
            # Check for remaining jobs (circular dependencies)
            remaining_jobs = [job_id for job_id in all_jobs if in_degree[job_id] > 0]
            if remaining_jobs:
                logger.warning(f"Circular dependencies detected for jobs: {remaining_jobs}")
                
        except Exception as e:
            logger.error(f"Error calculating execution order: {e}")
    
    def _job_exists(self, job_id: str) -> bool:
        """
        Check if a job exists (placeholder implementation).
        
        Args:
            job_id: The job ID to check
            
        Returns:
            bool: True if job exists, False otherwise
        """
        # This is a placeholder implementation
        # In a real system, you would check against your job registry
        return True


# Convenience functions
def create_dependency_manager(**kwargs) -> DependencyManager:
    """Create a dependency manager with custom configuration."""
    return DependencyManager(**kwargs)