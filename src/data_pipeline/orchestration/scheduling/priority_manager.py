"""
Priority Manager

This module manages job priorities and execution order for the PBF-LB/M data pipeline.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime, timedelta
import heapq

from src.data_pipeline.orchestration.scheduling.job_scheduler import Job, JobStatus, JobPriority

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PriorityRule(Enum):
    """Priority rule enumeration."""
    FIFO = "fifo"  # First In, First Out
    LIFO = "lifo"  # Last In, First Out
    PRIORITY_ONLY = "priority_only"  # Based on priority only
    WEIGHTED = "weighted"  # Based on priority and other factors
    DEADLINE = "deadline"  # Based on deadline

@dataclass
class PriorityWeight:
    """Priority weight data class."""
    priority: JobPriority
    weight: float
    max_concurrent: int

@dataclass
class JobQueueEntry:
    """Job queue entry data class."""
    job: Job
    priority_score: float
    queued_at: datetime
    deadline: Optional[datetime] = None
    
    def __lt__(self, other):
        # Higher priority score means higher priority
        return self.priority_score > other.priority_score

class PriorityManager:
    """
    Manages job priorities and execution order.
    """
    
    def __init__(self):
        self.priority_weights: Dict[JobPriority, PriorityWeight] = {
            JobPriority.CRITICAL: PriorityWeight(JobPriority.CRITICAL, 10.0, 2),
            JobPriority.HIGH: PriorityWeight(JobPriority.HIGH, 8.0, 5),
            JobPriority.MEDIUM: PriorityWeight(JobPriority.MEDIUM, 5.0, 10),
            JobPriority.LOW: PriorityWeight(JobPriority.LOW, 2.0, 20)
        }
        self.priority_rule = PriorityRule.WEIGHTED
        self.job_queue: List[JobQueueEntry] = []
        self.running_jobs: Dict[JobPriority, List[Job]] = {
            priority: [] for priority in JobPriority
        }
        self.completed_jobs: List[Job] = []
        self.failed_jobs: List[Job] = []
        
    def set_priority_rule(self, rule: PriorityRule) -> bool:
        """
        Set the priority rule for job execution.
        
        Args:
            rule: The priority rule to use
            
        Returns:
            bool: True if rule was set successfully, False otherwise
        """
        try:
            self.priority_rule = rule
            logger.info(f"Set priority rule to {rule.value}")
            return True
            
        except Exception as e:
            logger.error(f"Error setting priority rule: {e}")
            return False
    
    def set_priority_weight(self, priority: JobPriority, weight: float, max_concurrent: int) -> bool:
        """
        Set the priority weight for a job priority level.
        
        Args:
            priority: The job priority
            weight: The weight value
            max_concurrent: Maximum concurrent jobs for this priority
            
        Returns:
            bool: True if weight was set successfully, False otherwise
        """
        try:
            self.priority_weights[priority] = PriorityWeight(priority, weight, max_concurrent)
            logger.info(f"Set priority weight for {priority.value}: weight={weight}, max_concurrent={max_concurrent}")
            return True
            
        except Exception as e:
            logger.error(f"Error setting priority weight: {e}")
            return False
    
    def add_job_to_queue(self, job: Job, deadline: Optional[datetime] = None) -> bool:
        """
        Add a job to the priority queue.
        
        Args:
            job: The job to add
            deadline: Optional deadline for the job
            
        Returns:
            bool: True if job was added successfully, False otherwise
        """
        try:
            # Calculate priority score
            priority_score = self._calculate_priority_score(job, deadline)
            
            # Create queue entry
            queue_entry = JobQueueEntry(
                job=job,
                priority_score=priority_score,
                queued_at=datetime.now(),
                deadline=deadline
            )
            
            # Add to priority queue
            heapq.heappush(self.job_queue, queue_entry)
            
            logger.info(f"Added job {job.id} to priority queue with score {priority_score}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding job to queue: {e}")
            return False
    
    def get_next_job(self) -> Optional[Job]:
        """
        Get the next job to execute based on priority.
        
        Returns:
            Job: The next job to execute, or None if no jobs available
        """
        try:
            while self.job_queue:
                # Get the highest priority job
                queue_entry = heapq.heappop(self.job_queue)
                job = queue_entry.job
                
                # Check if job can be executed (not already running, dependencies met, etc.)
                if self._can_execute_job(job):
                    # Check if we can run more jobs of this priority
                    if self._can_run_priority_job(job.priority):
                        # Mark job as running
                        job.status = JobStatus.RUNNING
                        job.started_at = datetime.now()
                        
                        # Add to running jobs
                        self.running_jobs[job.priority].append(job)
                        
                        logger.info(f"Selected job {job.id} for execution")
                        return job
                    else:
                        # Put job back in queue
                        heapq.heappush(self.job_queue, queue_entry)
                        break
                else:
                    # Job cannot be executed, skip it
                    logger.info(f"Job {job.id} cannot be executed, skipping")
                    continue
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting next job: {e}")
            return None
    
    def complete_job(self, job_id: str, success: bool = True) -> bool:
        """
        Mark a job as completed.
        
        Args:
            job_id: The job ID
            success: Whether the job completed successfully
            
        Returns:
            bool: True if job was marked as completed successfully, False otherwise
        """
        try:
            # Find the job in running jobs
            job = None
            for priority_jobs in self.running_jobs.values():
                for running_job in priority_jobs:
                    if running_job.id == job_id:
                        job = running_job
                        priority_jobs.remove(running_job)
                        break
                if job:
                    break
            
            if not job:
                logger.error(f"Job {job_id} not found in running jobs")
                return False
            
            # Update job status
            job.status = JobStatus.COMPLETED if success else JobStatus.FAILED
            job.completed_at = datetime.now()
            
            # Move to appropriate list
            if success:
                self.completed_jobs.append(job)
            else:
                self.failed_jobs.append(job)
            
            logger.info(f"Marked job {job.id} as {'completed' if success else 'failed'}")
            return True
            
        except Exception as e:
            logger.error(f"Error completing job {job_id}: {e}")
            return False
    
    def get_queue_status(self) -> Dict[str, any]:
        """
        Get the current status of the priority queue.
        
        Returns:
            Dict[str, any]: Queue status information
        """
        try:
            queue_size = len(self.job_queue)
            running_jobs_count = sum(len(jobs) for jobs in self.running_jobs.values())
            completed_jobs_count = len(self.completed_jobs)
            failed_jobs_count = len(self.failed_jobs)
            
            priority_counts = {}
            for priority in JobPriority:
                priority_counts[priority.value] = {
                    "queued": len([entry for entry in self.job_queue if entry.job.priority == priority]),
                    "running": len(self.running_jobs[priority]),
                    "completed": len([job for job in self.completed_jobs if job.priority == priority]),
                    "failed": len([job for job in self.failed_jobs if job.priority == priority])
                }
            
            return {
                "queue_size": queue_size,
                "running_jobs": running_jobs_count,
                "completed_jobs": completed_jobs_count,
                "failed_jobs": failed_jobs_count,
                "priority_counts": priority_counts,
                "priority_rule": self.priority_rule.value
            }
            
        except Exception as e:
            logger.error(f"Error getting queue status: {e}")
            return {}
    
    def get_next_deadline_job(self) -> Optional[Job]:
        """
        Get the next job that is approaching its deadline.
        
        Returns:
            Job: The next job approaching deadline, or None if no jobs approaching deadline
        """
        try:
            current_time = datetime.now()
            deadline_threshold = timedelta(minutes=30)  # 30 minutes before deadline
            
            approaching_deadline = []
            
            for queue_entry in self.job_queue:
                if queue_entry.deadline:
                    time_to_deadline = queue_entry.deadline - current_time
                    if 0 < time_to_deadline <= deadline_threshold:
                        approaching_deadline.append((queue_entry, time_to_deadline))
            
            if approaching_deadline:
                # Sort by time to deadline (ascending)
                approaching_deadline.sort(key=lambda x: x[1])
                return approaching_deadline[0][0].job
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting next deadline job: {e}")
            return None
    
    def rebalance_priorities(self) -> bool:
        """
        Rebalance job priorities based on current conditions.
        
        Returns:
            bool: True if priorities were rebalanced successfully, False otherwise
        """
        try:
            # Recalculate priority scores for all queued jobs
            new_queue = []
            
            for queue_entry in self.job_queue:
                new_priority_score = self._calculate_priority_score(queue_entry.job, queue_entry.deadline)
                new_queue_entry = JobQueueEntry(
                    job=queue_entry.job,
                    priority_score=new_priority_score,
                    queued_at=queue_entry.queued_at,
                    deadline=queue_entry.deadline
                )
                new_queue.append(new_queue_entry)
            
            # Rebuild the priority queue
            self.job_queue = new_queue
            heapq.heapify(self.job_queue)
            
            logger.info("Rebalanced job priorities")
            return True
            
        except Exception as e:
            logger.error(f"Error rebalancing priorities: {e}")
            return False
    
    def _calculate_priority_score(self, job: Job, deadline: Optional[datetime] = None) -> float:
        """
        Calculate the priority score for a job.
        
        Args:
            job: The job to calculate score for
            deadline: Optional deadline for the job
            
        Returns:
            float: The priority score
        """
        try:
            base_score = self.priority_weights[job.priority].weight
            
            if self.priority_rule == PriorityRule.PRIORITY_ONLY:
                return base_score
            
            elif self.priority_rule == PriorityRule.FIFO:
                # Lower score for older jobs (FIFO)
                age_hours = (datetime.now() - job.created_at).total_seconds() / 3600
                return base_score - (age_hours * 0.1)
            
            elif self.priority_rule == PriorityRule.LIFO:
                # Higher score for newer jobs (LIFO)
                age_hours = (datetime.now() - job.created_at).total_seconds() / 3600
                return base_score + (age_hours * 0.1)
            
            elif self.priority_rule == PriorityRule.DEADLINE:
                if deadline:
                    time_to_deadline = (deadline - datetime.now()).total_seconds() / 3600
                    if time_to_deadline > 0:
                        # Higher score for jobs closer to deadline
                        return base_score + (24 / max(time_to_deadline, 0.1))
                    else:
                        # Very high score for overdue jobs
                        return base_score + 100
                else:
                    return base_score
            
            elif self.priority_rule == PriorityRule.WEIGHTED:
                # Weighted score based on multiple factors
                score = base_score
                
                # Factor in job age (older jobs get slight boost)
                age_hours = (datetime.now() - job.created_at).total_seconds() / 3600
                score += min(age_hours * 0.05, 2.0)  # Cap at 2.0
                
                # Factor in deadline if present
                if deadline:
                    time_to_deadline = (deadline - datetime.now()).total_seconds() / 3600
                    if time_to_deadline > 0:
                        score += min(12 / max(time_to_deadline, 0.1), 10.0)  # Cap at 10.0
                    else:
                        score += 20.0  # Overdue jobs get high boost
                
                # Factor in retry count (jobs that have failed get slight boost)
                if job.retry_count > 0:
                    score += min(job.retry_count * 0.5, 3.0)  # Cap at 3.0
                
                return score
            
            else:
                return base_score
                
        except Exception as e:
            logger.error(f"Error calculating priority score: {e}")
            return self.priority_weights[job.priority].weight
    
    def _can_execute_job(self, job: Job) -> bool:
        """
        Check if a job can be executed.
        
        Args:
            job: The job to check
            
        Returns:
            bool: True if job can be executed, False otherwise
        """
        try:
            # Check if job is not already running
            if job.status == JobStatus.RUNNING:
                return False
            
            # Check if job is not completed or failed
            if job.status in [JobStatus.COMPLETED, JobStatus.FAILED]:
                return False
            
            # Check if job is not cancelled
            if job.status == JobStatus.CANCELLED:
                return False
            
            # Additional checks can be added here (dependencies, resources, etc.)
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking if job can execute: {e}")
            return False
    
    def _can_run_priority_job(self, priority: JobPriority) -> bool:
        """
        Check if we can run more jobs of a specific priority.
        
        Args:
            priority: The job priority
            
        Returns:
            bool: True if we can run more jobs of this priority, False otherwise
        """
        try:
            max_concurrent = self.priority_weights[priority].max_concurrent
            current_running = len(self.running_jobs[priority])
            
            return current_running < max_concurrent
            
        except Exception as e:
            logger.error(f"Error checking if priority job can run: {e}")
            return False


# Convenience functions
def create_priority_manager(**kwargs) -> PriorityManager:
    """Create a priority manager with custom configuration."""
    return PriorityManager(**kwargs)