"""
Job Scheduler

This module provides job scheduling capabilities for the PBF-LB/M data pipeline.
"""

import schedule
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from enum import Enum
import logging

from src.data_pipeline.config.pipeline_config import get_pipeline_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class JobStatus(Enum):
    """Job status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class JobPriority(Enum):
    """Job priority enumeration."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

@dataclass
class Job:
    """Job data class."""
    id: str
    name: str
    description: str
    function: Callable
    schedule: str
    priority: JobPriority
    timeout_minutes: int
    retry_attempts: int
    retry_delay_minutes: int
    dependencies: List[str]
    status: JobStatus = JobStatus.PENDING
    created_at: datetime = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

class JobScheduler:
    """
    Job scheduler for managing and executing scheduled jobs.
    """
    
    def __init__(self):
        self.jobs: Dict[str, Job] = {}
        self.running_jobs: Dict[str, Job] = {}
        self.completed_jobs: List[Job] = []
        self.failed_jobs: List[Job] = []
        self.scheduler_thread: Optional[threading.Thread] = None
        self.is_running = False
        self.config = get_pipeline_config()
        
    def add_job(self, job: Job) -> bool:
        """
        Add a job to the scheduler.
        
        Args:
            job: The job to add
            
        Returns:
            bool: True if job was added successfully, False otherwise
        """
        try:
            if job.id in self.jobs:
                logger.warning(f"Job {job.id} already exists. Updating existing job.")
            
            self.jobs[job.id] = job
            logger.info(f"Added job {job.id}: {job.name}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding job {job.id}: {e}")
            return False
    
    def remove_job(self, job_id: str) -> bool:
        """
        Remove a job from the scheduler.
        
        Args:
            job_id: The ID of the job to remove
            
        Returns:
            bool: True if job was removed successfully, False otherwise
        """
        try:
            if job_id in self.jobs:
                del self.jobs[job_id]
                logger.info(f"Removed job {job_id}")
                return True
            else:
                logger.warning(f"Job {job_id} not found")
                return False
                
        except Exception as e:
            logger.error(f"Error removing job {job_id}: {e}")
            return False
    
    def get_job(self, job_id: str) -> Optional[Job]:
        """
        Get a job by ID.
        
        Args:
            job_id: The ID of the job to get
            
        Returns:
            Job: The job if found, None otherwise
        """
        return self.jobs.get(job_id)
    
    def get_jobs_by_status(self, status: JobStatus) -> List[Job]:
        """
        Get all jobs with a specific status.
        
        Args:
            status: The status to filter by
            
        Returns:
            List[Job]: List of jobs with the specified status
        """
        return [job for job in self.jobs.values() if job.status == status]
    
    def get_jobs_by_priority(self, priority: JobPriority) -> List[Job]:
        """
        Get all jobs with a specific priority.
        
        Args:
            priority: The priority to filter by
            
        Returns:
            List[Job]: List of jobs with the specified priority
        """
        return [job for job in self.jobs.values() if job.priority == priority]
    
    def start_scheduler(self) -> bool:
        """
        Start the job scheduler.
        
        Returns:
            bool: True if scheduler started successfully, False otherwise
        """
        try:
            if self.is_running:
                logger.warning("Scheduler is already running")
                return False
            
            self.is_running = True
            self.scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
            self.scheduler_thread.start()
            
            logger.info("Job scheduler started")
            return True
            
        except Exception as e:
            logger.error(f"Error starting scheduler: {e}")
            return False
    
    def stop_scheduler(self) -> bool:
        """
        Stop the job scheduler.
        
        Returns:
            bool: True if scheduler stopped successfully, False otherwise
        """
        try:
            if not self.is_running:
                logger.warning("Scheduler is not running")
                return False
            
            self.is_running = False
            
            if self.scheduler_thread and self.scheduler_thread.is_alive():
                self.scheduler_thread.join(timeout=10)
            
            logger.info("Job scheduler stopped")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping scheduler: {e}")
            return False
    
    def execute_job(self, job_id: str) -> bool:
        """
        Execute a specific job.
        
        Args:
            job_id: The ID of the job to execute
            
        Returns:
            bool: True if job was executed successfully, False otherwise
        """
        try:
            job = self.get_job(job_id)
            if not job:
                logger.error(f"Job {job_id} not found")
                return False
            
            if job.status == JobStatus.RUNNING:
                logger.warning(f"Job {job_id} is already running")
                return False
            
            # Check dependencies
            if not self._check_dependencies(job):
                logger.error(f"Job {job_id} dependencies not met")
                return False
            
            # Execute job in a separate thread
            job_thread = threading.Thread(
                target=self._execute_job_thread,
                args=(job,),
                daemon=True
            )
            job_thread.start()
            
            return True
            
        except Exception as e:
            logger.error(f"Error executing job {job_id}: {e}")
            return False
    
    def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a running job.
        
        Args:
            job_id: The ID of the job to cancel
            
        Returns:
            bool: True if job was cancelled successfully, False otherwise
        """
        try:
            job = self.get_job(job_id)
            if not job:
                logger.error(f"Job {job_id} not found")
                return False
            
            if job.status != JobStatus.RUNNING:
                logger.warning(f"Job {job_id} is not running")
                return False
            
            job.status = JobStatus.CANCELLED
            job.completed_at = datetime.now()
            
            logger.info(f"Cancelled job {job_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error cancelling job {job_id}: {e}")
            return False
    
    def get_scheduler_status(self) -> Dict[str, Any]:
        """
        Get the current status of the scheduler.
        
        Returns:
            Dict[str, Any]: Scheduler status information
        """
        try:
            total_jobs = len(self.jobs)
            running_jobs = len(self.get_jobs_by_status(JobStatus.RUNNING))
            pending_jobs = len(self.get_jobs_by_status(JobStatus.PENDING))
            completed_jobs = len(self.completed_jobs)
            failed_jobs = len(self.failed_jobs)
            
            return {
                "is_running": self.is_running,
                "total_jobs": total_jobs,
                "running_jobs": running_jobs,
                "pending_jobs": pending_jobs,
                "completed_jobs": completed_jobs,
                "failed_jobs": failed_jobs,
                "uptime": datetime.now() - (self.completed_jobs[0].created_at if self.completed_jobs else datetime.now())
            }
            
        except Exception as e:
            logger.error(f"Error getting scheduler status: {e}")
            return {}
    
    def _run_scheduler(self):
        """Run the scheduler loop."""
        while self.is_running:
            try:
                # Check for jobs that need to be executed
                self._check_scheduled_jobs()
                
                # Sleep for a short interval
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
                time.sleep(60)
    
    def _check_scheduled_jobs(self):
        """Check for jobs that need to be executed based on their schedule."""
        try:
            current_time = datetime.now()
            
            for job in self.jobs.values():
                if job.status == JobStatus.PENDING:
                    # Check if job should be executed based on schedule
                    if self._should_execute_job(job, current_time):
                        self.execute_job(job.id)
                        
        except Exception as e:
            logger.error(f"Error checking scheduled jobs: {e}")
    
    def _should_execute_job(self, job: Job, current_time: datetime) -> bool:
        """
        Check if a job should be executed based on its schedule.
        
        Args:
            job: The job to check
            current_time: The current time
            
        Returns:
            bool: True if job should be executed, False otherwise
        """
        try:
            # Parse schedule string (e.g., "0 0 * * *" for daily at midnight)
            # This is a simplified implementation
            if job.schedule == "0 0 * * *":  # Daily at midnight
                return current_time.hour == 0 and current_time.minute == 0
            elif job.schedule == "0 */15 * * *":  # Every 15 minutes
                return current_time.minute % 15 == 0
            elif job.schedule == "0 */30 * * *":  # Every 30 minutes
                return current_time.minute % 30 == 0
            elif job.schedule == "0 2 * * *":  # Daily at 2 AM
                return current_time.hour == 2 and current_time.minute == 0
            elif job.schedule == "0 3 * * *":  # Daily at 3 AM
                return current_time.hour == 3 and current_time.minute == 0
            else:
                # Default to not executing
                return False
                
        except Exception as e:
            logger.error(f"Error checking job schedule: {e}")
            return False
    
    def _check_dependencies(self, job: Job) -> bool:
        """
        Check if all job dependencies are met.
        
        Args:
            job: The job to check dependencies for
            
        Returns:
            bool: True if all dependencies are met, False otherwise
        """
        try:
            for dependency_id in job.dependencies:
                dependency_job = self.get_job(dependency_id)
                if not dependency_job:
                    logger.error(f"Dependency job {dependency_id} not found")
                    return False
                
                if dependency_job.status != JobStatus.COMPLETED:
                    logger.info(f"Dependency job {dependency_id} not completed")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking dependencies: {e}")
            return False
    
    def _execute_job_thread(self, job: Job):
        """
        Execute a job in a separate thread.
        
        Args:
            job: The job to execute
        """
        try:
            job.status = JobStatus.RUNNING
            job.started_at = datetime.now()
            job.retry_count = 0
            
            logger.info(f"Starting job {job.id}: {job.name}")
            
            # Execute the job function
            result = job.function()
            
            # Mark job as completed
            job.status = JobStatus.COMPLETED
            job.completed_at = datetime.now()
            job.error_message = None
            
            # Move to completed jobs
            self.completed_jobs.append(job)
            
            logger.info(f"Completed job {job.id}: {job.name}")
            
        except Exception as e:
            logger.error(f"Error executing job {job.id}: {e}")
            
            # Handle retry logic
            if job.retry_count < job.retry_attempts:
                job.retry_count += 1
                job.status = JobStatus.PENDING
                job.error_message = str(e)
                
                logger.info(f"Retrying job {job.id} (attempt {job.retry_count}/{job.retry_attempts})")
                
                # Schedule retry
                time.sleep(job.retry_delay_minutes * 60)
                self._execute_job_thread(job)
            else:
                # Mark job as failed
                job.status = JobStatus.FAILED
                job.completed_at = datetime.now()
                job.error_message = str(e)
                
                # Move to failed jobs
                self.failed_jobs.append(job)
                
                logger.error(f"Failed job {job.id}: {job.name} - {e}")
    
    def cleanup_old_jobs(self, days_to_keep: int = 30):
        """
        Clean up old completed and failed jobs.
        
        Args:
            days_to_keep: Number of days to keep old jobs
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            # Clean up completed jobs
            self.completed_jobs = [
                job for job in self.completed_jobs
                if job.completed_at and job.completed_at > cutoff_date
            ]
            
            # Clean up failed jobs
            self.failed_jobs = [
                job for job in self.failed_jobs
                if job.completed_at and job.completed_at > cutoff_date
            ]
            
            logger.info(f"Cleaned up jobs older than {days_to_keep} days")
            
        except Exception as e:
            logger.error(f"Error cleaning up old jobs: {e}")


# Convenience functions
def create_job_scheduler(**kwargs) -> JobScheduler:
    """Create a job scheduler with custom configuration."""
    return JobScheduler(**kwargs)