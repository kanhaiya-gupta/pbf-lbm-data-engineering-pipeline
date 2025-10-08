"""
Job Monitor

This module provides job execution monitoring capabilities for the PBF-LB/M data pipeline.
"""

import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import json

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
    TIMEOUT = "timeout"

class JobType(Enum):
    """Job type enumeration."""
    ETL = "etl"
    STREAMING = "streaming"
    QUALITY_CHECK = "quality_check"
    ARCHIVE = "archive"
    TRANSFORMATION = "transformation"
    INGESTION = "ingestion"

@dataclass
class JobExecution:
    """Job execution data class."""
    job_id: str
    job_name: str
    job_type: JobType
    status: JobStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    timeout_minutes: int = 60
    resource_usage: Dict[str, float] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)

@dataclass
class JobMetrics:
    """Job metrics data class."""
    job_id: str
    cpu_usage: float
    memory_usage: float
    disk_io: float
    network_io: float
    records_processed: int
    throughput: float
    error_rate: float
    timestamp: datetime = field(default_factory=datetime.now)

class JobMonitor:
    """
    Monitors job execution and performance.
    """
    
    def __init__(self):
        self.config = get_pipeline_config()
        self.job_executions: Dict[str, JobExecution] = {}
        self.job_metrics: Dict[str, List[JobMetrics]] = {}
        self.monitoring_thread: Optional[threading.Thread] = None
        self.is_monitoring = False
        self.monitoring_interval = 30  # seconds
        self.job_callbacks: List[Callable[[JobExecution], None]] = []
        
    def start_monitoring(self) -> bool:
        """
        Start job monitoring.
        
        Returns:
            bool: True if monitoring started successfully, False otherwise
        """
        try:
            if self.is_monitoring:
                logger.warning("Job monitoring is already running")
                return False
            
            self.is_monitoring = True
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitoring_thread.start()
            
            logger.info("Job monitoring started")
            return True
            
        except Exception as e:
            logger.error(f"Error starting job monitoring: {e}")
            return False
    
    def stop_monitoring(self) -> bool:
        """
        Stop job monitoring.
        
        Returns:
            bool: True if monitoring stopped successfully, False otherwise
        """
        try:
            if not self.is_monitoring:
                logger.warning("Job monitoring is not running")
                return False
            
            self.is_monitoring = False
            
            if self.monitoring_thread and self.monitoring_thread.is_alive():
                self.monitoring_thread.join(timeout=10)
            
            logger.info("Job monitoring stopped")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping job monitoring: {e}")
            return False
    
    def start_job(self, job_id: str, job_name: str, job_type: JobType, timeout_minutes: int = 60) -> bool:
        """
        Start monitoring a job.
        
        Args:
            job_id: The job ID
            job_name: The job name
            job_type: The job type
            timeout_minutes: Job timeout in minutes
            
        Returns:
            bool: True if job monitoring started successfully, False otherwise
        """
        try:
            if job_id in self.job_executions:
                logger.warning(f"Job {job_id} is already being monitored")
                return False
            
            job_execution = JobExecution(
                job_id=job_id,
                job_name=job_name,
                job_type=job_type,
                status=JobStatus.RUNNING,
                start_time=datetime.now(),
                timeout_minutes=timeout_minutes
            )
            
            self.job_executions[job_id] = job_execution
            self.job_metrics[job_id] = []
            
            logger.info(f"Started monitoring job {job_id}: {job_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error starting job monitoring for {job_id}: {e}")
            return False
    
    def complete_job(self, job_id: str, success: bool = True, error_message: Optional[str] = None) -> bool:
        """
        Mark a job as completed.
        
        Args:
            job_id: The job ID
            success: Whether the job completed successfully
            error_message: Error message if job failed
            
        Returns:
            bool: True if job was marked as completed successfully, False otherwise
        """
        try:
            if job_id not in self.job_executions:
                logger.error(f"Job {job_id} not found in monitoring")
                return False
            
            job_execution = self.job_executions[job_id]
            job_execution.end_time = datetime.now()
            job_execution.duration_seconds = (job_execution.end_time - job_execution.start_time).total_seconds()
            
            if success:
                job_execution.status = JobStatus.COMPLETED
            else:
                job_execution.status = JobStatus.FAILED
                job_execution.error_message = error_message
            
            # Call job callbacks
            for callback in self.job_callbacks:
                try:
                    callback(job_execution)
                except Exception as e:
                    logger.error(f"Error in job callback: {e}")
            
            logger.info(f"Completed job {job_id}: {job_execution.status.value}")
            return True
            
        except Exception as e:
            logger.error(f"Error completing job {job_id}: {e}")
            return False
    
    def add_job_metrics(self, job_id: str, metrics: JobMetrics) -> bool:
        """
        Add metrics for a job.
        
        Args:
            job_id: The job ID
            metrics: The job metrics
            
        Returns:
            bool: True if metrics were added successfully, False otherwise
        """
        try:
            if job_id not in self.job_metrics:
                logger.error(f"Job {job_id} not found in monitoring")
                return False
            
            self.job_metrics[job_id].append(metrics)
            
            # Keep only last 1000 metrics per job
            if len(self.job_metrics[job_id]) > 1000:
                self.job_metrics[job_id] = self.job_metrics[job_id][-1000:]
            
            logger.debug(f"Added metrics for job {job_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding metrics for job {job_id}: {e}")
            return False
    
    def get_job_status(self, job_id: str) -> Optional[JobExecution]:
        """
        Get the status of a job.
        
        Args:
            job_id: The job ID
            
        Returns:
            JobExecution: The job execution status, or None if not found
        """
        return self.job_executions.get(job_id)
    
    def get_running_jobs(self) -> List[JobExecution]:
        """
        Get all currently running jobs.
        
        Returns:
            List[JobExecution]: List of running jobs
        """
        return [job for job in self.job_executions.values() if job.status == JobStatus.RUNNING]
    
    def get_job_metrics(self, job_id: str, minutes: int = 60) -> List[JobMetrics]:
        """
        Get job metrics for a specific time period.
        
        Args:
            job_id: The job ID
            minutes: Number of minutes to look back
            
        Returns:
            List[JobMetrics]: List of job metrics within the time period
        """
        try:
            if job_id not in self.job_metrics:
                return []
            
            cutoff_time = datetime.now() - timedelta(minutes=minutes)
            metrics = self.job_metrics[job_id]
            
            return [metric for metric in metrics if metric.timestamp >= cutoff_time]
            
        except Exception as e:
            logger.error(f"Error getting job metrics for {job_id}: {e}")
            return []
    
    def get_job_performance_summary(self, job_id: str) -> Dict[str, Any]:
        """
        Get a performance summary for a job.
        
        Args:
            job_id: The job ID
            
        Returns:
            Dict[str, Any]: Job performance summary
        """
        try:
            if job_id not in self.job_executions:
                return {}
            
            job_execution = self.job_executions[job_id]
            metrics = self.job_metrics.get(job_id, [])
            
            if not metrics:
                return {
                    "job_id": job_id,
                    "job_name": job_execution.job_name,
                    "status": job_execution.status.value,
                    "duration_seconds": job_execution.duration_seconds,
                    "error_message": job_execution.error_message
                }
            
            # Calculate performance metrics
            avg_cpu = sum(m.cpu_usage for m in metrics) / len(metrics)
            avg_memory = sum(m.memory_usage for m in metrics) / len(metrics)
            total_records = sum(m.records_processed for m in metrics)
            avg_throughput = sum(m.throughput for m in metrics) / len(metrics)
            avg_error_rate = sum(m.error_rate for m in metrics) / len(metrics)
            
            return {
                "job_id": job_id,
                "job_name": job_execution.job_name,
                "job_type": job_execution.job_type.value,
                "status": job_execution.status.value,
                "start_time": job_execution.start_time.isoformat(),
                "end_time": job_execution.end_time.isoformat() if job_execution.end_time else None,
                "duration_seconds": job_execution.duration_seconds,
                "retry_count": job_execution.retry_count,
                "error_message": job_execution.error_message,
                "performance": {
                    "avg_cpu_usage": avg_cpu,
                    "avg_memory_usage": avg_memory,
                    "total_records_processed": total_records,
                    "avg_throughput": avg_throughput,
                    "avg_error_rate": avg_error_rate
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting job performance summary for {job_id}: {e}")
            return {}
    
    def get_all_jobs_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all jobs.
        
        Returns:
            Dict[str, Any]: Summary of all jobs
        """
        try:
            total_jobs = len(self.job_executions)
            running_jobs = len(self.get_running_jobs())
            completed_jobs = len([j for j in self.job_executions.values() if j.status == JobStatus.COMPLETED])
            failed_jobs = len([j for j in self.job_executions.values() if j.status == JobStatus.FAILED])
            
            # Calculate average performance metrics
            all_metrics = []
            for metrics_list in self.job_metrics.values():
                all_metrics.extend(metrics_list)
            
            if all_metrics:
                avg_cpu = sum(m.cpu_usage for m in all_metrics) / len(all_metrics)
                avg_memory = sum(m.memory_usage for m in all_metrics) / len(all_metrics)
                total_records = sum(m.records_processed for m in all_metrics)
                avg_throughput = sum(m.throughput for m in all_metrics) / len(all_metrics)
                avg_error_rate = sum(m.error_rate for m in all_metrics) / len(all_metrics)
            else:
                avg_cpu = avg_memory = total_records = avg_throughput = avg_error_rate = 0
            
            return {
                "total_jobs": total_jobs,
                "running_jobs": running_jobs,
                "completed_jobs": completed_jobs,
                "failed_jobs": failed_jobs,
                "success_rate": completed_jobs / total_jobs if total_jobs > 0 else 0,
                "average_performance": {
                    "avg_cpu_usage": avg_cpu,
                    "avg_memory_usage": avg_memory,
                    "total_records_processed": total_records,
                    "avg_throughput": avg_throughput,
                    "avg_error_rate": avg_error_rate
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting all jobs summary: {e}")
            return {}
    
    def add_job_callback(self, callback: Callable[[JobExecution], None]) -> bool:
        """
        Add a job callback function.
        
        Args:
            callback: The callback function to call when jobs complete
            
        Returns:
            bool: True if callback was added successfully, False otherwise
        """
        try:
            self.job_callbacks.append(callback)
            logger.info("Added job callback")
            return True
            
        except Exception as e:
            logger.error(f"Error adding job callback: {e}")
            return False
    
    def cleanup_old_jobs(self, days_to_keep: int = 7):
        """
        Clean up old job executions and metrics.
        
        Args:
            days_to_keep: Number of days to keep old jobs
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            # Clean up old job executions
            old_job_ids = []
            for job_id, job_execution in self.job_executions.items():
                if (job_execution.end_time and job_execution.end_time < cutoff_date):
                    old_job_ids.append(job_id)
            
            for job_id in old_job_ids:
                del self.job_executions[job_id]
                if job_id in self.job_metrics:
                    del self.job_metrics[job_id]
            
            # Clean up old metrics
            for job_id, metrics_list in self.job_metrics.items():
                self.job_metrics[job_id] = [
                    metric for metric in metrics_list
                    if metric.timestamp >= cutoff_date
                ]
            
            if old_job_ids:
                logger.info(f"Cleaned up {len(old_job_ids)} old jobs")
                
        except Exception as e:
            logger.error(f"Error cleaning up old jobs: {e}")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                # Check for timeout jobs
                self._check_timeout_jobs()
                
                # Collect job metrics
                self._collect_job_metrics()
                
                # Sleep for monitoring interval
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in job monitoring loop: {e}")
                time.sleep(self.monitoring_interval)
    
    def _check_timeout_jobs(self):
        """Check for jobs that have timed out."""
        try:
            current_time = datetime.now()
            timeout_jobs = []
            
            for job_id, job_execution in self.job_executions.items():
                if job_execution.status == JobStatus.RUNNING:
                    elapsed_minutes = (current_time - job_execution.start_time).total_seconds() / 60
                    if elapsed_minutes > job_execution.timeout_minutes:
                        timeout_jobs.append(job_id)
            
            # Mark timeout jobs as failed
            for job_id in timeout_jobs:
                self.complete_job(job_id, success=False, error_message="Job timed out")
                logger.warning(f"Job {job_id} timed out after {self.job_executions[job_id].timeout_minutes} minutes")
                
        except Exception as e:
            logger.error(f"Error checking timeout jobs: {e}")
    
    def _collect_job_metrics(self):
        """Collect metrics for running jobs."""
        try:
            # This is a placeholder implementation
            # In a real system, you would collect actual metrics from job execution
            
            for job_id, job_execution in self.job_executions.items():
                if job_execution.status == JobStatus.RUNNING:
                    # Simulate metric collection
                    metrics = JobMetrics(
                        job_id=job_id,
                        cpu_usage=50.0 + (time.time() % 20),  # Simulate CPU usage
                        memory_usage=100.0 + (time.time() % 50),  # Simulate memory usage
                        disk_io=10.0 + (time.time() % 5),  # Simulate disk I/O
                        network_io=5.0 + (time.time() % 3),  # Simulate network I/O
                        records_processed=1000,  # Simulate records processed
                        throughput=100.0 + (time.time() % 20),  # Simulate throughput
                        error_rate=0.01 + (time.time() % 0.02)  # Simulate error rate
                    )
                    
                    self.add_job_metrics(job_id, metrics)
                    
        except Exception as e:
            logger.error(f"Error collecting job metrics: {e}")
