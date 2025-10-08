"""
Distributed Computing for PBF-LB/M Virtual Environment

This module provides distributed computing capabilities including cluster management,
job scheduling, and distributed processing for PBF-LB/M virtual testing and
simulation environments.
"""

import asyncio
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import uuid
import json
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue

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
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class ComputeNode:
    """Compute node configuration."""
    
    node_id: str
    hostname: str
    ip_address: str
    
    # Node specifications
    cpu_cores: int
    memory_gb: int
    storage_gb: int
    created_at: datetime
    updated_at: datetime
    
    # Node specifications with defaults
    gpu_count: int = 0
    
    # Node state
    status: str = "available"
    current_jobs: List[str] = None
    max_jobs: int = 1
    
    # Performance metrics
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    load_average: float = 0.0


@dataclass
class ComputeJob:
    """Compute job definition."""
    
    job_id: str
    name: str
    description: str
    
    # Job configuration
    command: str
    working_directory: str
    environment_variables: Dict[str, str]
    created_at: datetime
    updated_at: datetime
    
    # Resource requirements
    cpu_cores: int = 1
    memory_gb: int = 1
    storage_gb: int = 1
    gpu_count: int = 0
    
    # Job scheduling
    priority: JobPriority = JobPriority.MEDIUM
    max_runtime: float = 3600.0  # seconds
    retry_count: int = 0
    max_retries: int = 3
    
    # Job state
    status: JobStatus = JobStatus.PENDING
    assigned_node: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    # Results
    exit_code: Optional[int] = None
    stdout: str = ""
    stderr: str = ""


class DistributedComputingManager:
    """
    Distributed computing manager for PBF-LB/M virtual environment.
    
    This class provides distributed computing capabilities including cluster
    management, job scheduling, and distributed processing for PBF-LB/M
    virtual testing and simulation environments.
    """
    
    def __init__(self):
        """Initialize the distributed computing manager."""
        self.cluster_manager = ClusterManager()
        self.job_scheduler = JobScheduler()
        self.active_jobs = {}
        self.completed_jobs = {}
        
        logger.info("Distributed Computing Manager initialized")
    
    async def add_compute_node(
        self,
        hostname: str,
        ip_address: str,
        cpu_cores: int,
        memory_gb: int,
        storage_gb: int,
        gpu_count: int = 0
    ) -> str:
        """
        Add a compute node to the cluster.
        
        Args:
            hostname: Node hostname
            ip_address: Node IP address
            cpu_cores: Number of CPU cores
            memory_gb: Memory in GB
            storage_gb: Storage in GB
            gpu_count: Number of GPUs
            
        Returns:
            str: Node ID
        """
        try:
            node_id = await self.cluster_manager.add_node(
                hostname, ip_address, cpu_cores, memory_gb, storage_gb, gpu_count
            )
            
            logger.info(f"Compute node added: {node_id}")
            return node_id
            
        except Exception as e:
            logger.error(f"Error adding compute node: {e}")
            return ""
    
    async def remove_compute_node(self, node_id: str) -> bool:
        """
        Remove a compute node from the cluster.
        
        Args:
            node_id: Node ID
            
        Returns:
            bool: Success status
        """
        try:
            success = await self.cluster_manager.remove_node(node_id)
            
            if success:
                logger.info(f"Compute node removed: {node_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error removing compute node: {e}")
            return False
    
    async def submit_job(
        self,
        name: str,
        command: str,
        working_directory: str = "/tmp",
        environment_variables: Dict[str, str] = None,
        cpu_cores: int = 1,
        memory_gb: int = 1,
        storage_gb: int = 1,
        gpu_count: int = 0,
        priority: JobPriority = JobPriority.MEDIUM,
        max_runtime: float = 3600.0
    ) -> str:
        """
        Submit a job to the cluster.
        
        Args:
            name: Job name
            command: Command to execute
            working_directory: Working directory
            environment_variables: Environment variables
            cpu_cores: CPU cores required
            memory_gb: Memory required in GB
            storage_gb: Storage required in GB
            gpu_count: GPUs required
            priority: Job priority
            max_runtime: Maximum runtime in seconds
            
        Returns:
            str: Job ID
        """
        try:
            job_id = await self.job_scheduler.submit_job(
                name, command, working_directory, environment_variables,
                cpu_cores, memory_gb, storage_gb, gpu_count, priority, max_runtime
            )
            
            # Store job in active jobs
            job = await self.job_scheduler.get_job(job_id)
            self.active_jobs[job_id] = job
            
            logger.info(f"Job submitted: {job_id}")
            return job_id
            
        except Exception as e:
            logger.error(f"Error submitting job: {e}")
            return ""
    
    async def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a running job.
        
        Args:
            job_id: Job ID
            
        Returns:
            bool: Success status
        """
        try:
            success = await self.job_scheduler.cancel_job(job_id)
            
            if success:
                # Move job from active to completed
                if job_id in self.active_jobs:
                    job = self.active_jobs[job_id]
                    job.status = JobStatus.CANCELLED
                    job.end_time = datetime.now()
                    self.completed_jobs[job_id] = job
                    del self.active_jobs[job_id]
                
                logger.info(f"Job cancelled: {job_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error cancelling job: {e}")
            return False
    
    async def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get job status."""
        try:
            job = await self.job_scheduler.get_job(job_id)
            
            if job:
                return {
                    'job_id': job.job_id,
                    'name': job.name,
                    'status': job.status.value,
                    'assigned_node': job.assigned_node,
                    'start_time': job.start_time.isoformat() if job.start_time else None,
                    'end_time': job.end_time.isoformat() if job.end_time else None,
                    'exit_code': job.exit_code,
                    'priority': job.priority.value
                }
            else:
                return {'status': 'not_found'}
                
        except Exception as e:
            logger.error(f"Error getting job status: {e}")
            return {'status': 'error', 'error': str(e)}
    
    async def get_cluster_status(self) -> Dict[str, Any]:
        """Get cluster status."""
        try:
            nodes = await self.cluster_manager.list_nodes()
            
            total_nodes = len(nodes)
            available_nodes = len([n for n in nodes if n.status == 'available'])
            busy_nodes = len([n for n in nodes if n.status == 'busy'])
            
            total_cpu_cores = sum(n.cpu_cores for n in nodes)
            total_memory_gb = sum(n.memory_gb for n in nodes)
            total_storage_gb = sum(n.storage_gb for n in nodes)
            total_gpus = sum(n.gpu_count for n in nodes)
            
            active_jobs_count = len(self.active_jobs)
            completed_jobs_count = len(self.completed_jobs)
            
            return {
                'total_nodes': total_nodes,
                'available_nodes': available_nodes,
                'busy_nodes': busy_nodes,
                'total_cpu_cores': total_cpu_cores,
                'total_memory_gb': total_memory_gb,
                'total_storage_gb': total_storage_gb,
                'total_gpus': total_gpus,
                'active_jobs': active_jobs_count,
                'completed_jobs': completed_jobs_count,
                'nodes': [
                    {
                        'node_id': n.node_id,
                        'hostname': n.hostname,
                        'ip_address': n.ip_address,
                        'status': n.status,
                        'cpu_cores': n.cpu_cores,
                        'memory_gb': n.memory_gb,
                        'current_jobs': len(n.current_jobs) if n.current_jobs else 0,
                        'max_jobs': n.max_jobs
                    }
                    for n in nodes
                ]
            }
            
        except Exception as e:
            logger.error(f"Error getting cluster status: {e}")
            return {}


class ClusterManager:
    """
    Cluster manager for distributed computing.
    
    This class provides cluster management capabilities including node
    management, resource monitoring, and cluster coordination.
    """
    
    def __init__(self):
        """Initialize the cluster manager."""
        self.nodes = {}
        self.node_monitor = None
        self.monitoring_thread = None
        self.monitoring_active = False
        
        logger.info("Cluster Manager initialized")
    
    async def add_node(
        self,
        hostname: str,
        ip_address: str,
        cpu_cores: int,
        memory_gb: int,
        storage_gb: int,
        gpu_count: int = 0
    ) -> str:
        """Add a node to the cluster."""
        try:
            node_id = str(uuid.uuid4())
            
            node = ComputeNode(
                node_id=node_id,
                hostname=hostname,
                ip_address=ip_address,
                cpu_cores=cpu_cores,
                memory_gb=memory_gb,
                storage_gb=storage_gb,
                gpu_count=gpu_count,
                current_jobs=[],
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            
            self.nodes[node_id] = node
            
            # Start monitoring if not already active
            if not self.monitoring_active:
                await self.start_monitoring()
            
            logger.info(f"Node added to cluster: {node_id}")
            return node_id
            
        except Exception as e:
            logger.error(f"Error adding node to cluster: {e}")
            return ""
    
    async def remove_node(self, node_id: str) -> bool:
        """Remove a node from the cluster."""
        try:
            if node_id in self.nodes:
                node = self.nodes[node_id]
                
                # Check if node has running jobs
                if node.current_jobs:
                    logger.warning(f"Node {node_id} has running jobs, cannot remove")
                    return False
                
                del self.nodes[node_id]
                
                # Stop monitoring if no nodes left
                if not self.nodes and self.monitoring_active:
                    await self.stop_monitoring()
                
                logger.info(f"Node removed from cluster: {node_id}")
                return True
            else:
                logger.warning(f"Node not found: {node_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error removing node from cluster: {e}")
            return False
    
    async def get_available_node(
        self,
        cpu_cores: int,
        memory_gb: int,
        storage_gb: int,
        gpu_count: int = 0
    ) -> Optional[ComputeNode]:
        """Get an available node that meets resource requirements."""
        try:
            for node in self.nodes.values():
                if (node.status == 'available' and
                    node.cpu_cores >= cpu_cores and
                    node.memory_gb >= memory_gb and
                    node.storage_gb >= storage_gb and
                    node.gpu_count >= gpu_count and
                    len(node.current_jobs) < node.max_jobs):
                    
                    return node
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting available node: {e}")
            return None
    
    async def assign_job_to_node(self, node_id: str, job_id: str) -> bool:
        """Assign a job to a node."""
        try:
            if node_id in self.nodes:
                node = self.nodes[node_id]
                
                if node.current_jobs is None:
                    node.current_jobs = []
                
                node.current_jobs.append(job_id)
                node.status = 'busy' if len(node.current_jobs) >= node.max_jobs else 'available'
                node.updated_at = datetime.now()
                
                logger.info(f"Job {job_id} assigned to node {node_id}")
                return True
            else:
                logger.warning(f"Node not found: {node_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error assigning job to node: {e}")
            return False
    
    async def release_job_from_node(self, node_id: str, job_id: str) -> bool:
        """Release a job from a node."""
        try:
            if node_id in self.nodes:
                node = self.nodes[node_id]
                
                if node.current_jobs and job_id in node.current_jobs:
                    node.current_jobs.remove(job_id)
                    node.status = 'available' if len(node.current_jobs) < node.max_jobs else 'busy'
                    node.updated_at = datetime.now()
                    
                    logger.info(f"Job {job_id} released from node {node_id}")
                    return True
                else:
                    logger.warning(f"Job {job_id} not found on node {node_id}")
                    return False
            else:
                logger.warning(f"Node not found: {node_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error releasing job from node: {e}")
            return False
    
    async def list_nodes(self) -> List[ComputeNode]:
        """List all nodes in the cluster."""
        try:
            return list(self.nodes.values())
            
        except Exception as e:
            logger.error(f"Error listing nodes: {e}")
            return []
    
    async def get_node_status(self, node_id: str) -> Dict[str, Any]:
        """Get node status."""
        try:
            if node_id in self.nodes:
                node = self.nodes[node_id]
                
                return {
                    'node_id': node.node_id,
                    'hostname': node.hostname,
                    'ip_address': node.ip_address,
                    'status': node.status,
                    'cpu_cores': node.cpu_cores,
                    'memory_gb': node.memory_gb,
                    'storage_gb': node.storage_gb,
                    'gpu_count': node.gpu_count,
                    'current_jobs': len(node.current_jobs) if node.current_jobs else 0,
                    'max_jobs': node.max_jobs,
                    'cpu_usage': node.cpu_usage,
                    'memory_usage': node.memory_usage,
                    'load_average': node.load_average,
                    'created_at': node.created_at.isoformat(),
                    'updated_at': node.updated_at.isoformat()
                }
            else:
                return {'status': 'not_found'}
                
        except Exception as e:
            logger.error(f"Error getting node status: {e}")
            return {'status': 'error', 'error': str(e)}
    
    async def start_monitoring(self):
        """Start cluster monitoring."""
        try:
            if not self.monitoring_active:
                self.monitoring_active = True
                self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
                self.monitoring_thread.start()
                
                logger.info("Cluster monitoring started")
                
        except Exception as e:
            logger.error(f"Error starting cluster monitoring: {e}")
    
    async def stop_monitoring(self):
        """Stop cluster monitoring."""
        try:
            if self.monitoring_active:
                self.monitoring_active = False
                if self.monitoring_thread:
                    self.monitoring_thread.join(timeout=5.0)
                
                logger.info("Cluster monitoring stopped")
                
        except Exception as e:
            logger.error(f"Error stopping cluster monitoring: {e}")
    
    def _monitoring_loop(self):
        """Cluster monitoring loop."""
        try:
            while self.monitoring_active:
                # Update node metrics
                for node in self.nodes.values():
                    # Simulate metric collection
                    node.cpu_usage = min(100.0, node.cpu_usage + 1.0)
                    node.memory_usage = min(100.0, node.memory_usage + 0.5)
                    node.load_average = min(10.0, node.load_average + 0.1)
                    node.updated_at = datetime.now()
                
                # Sleep for monitoring interval
                time.sleep(30.0)  # Monitor every 30 seconds
                
        except Exception as e:
            logger.error(f"Error in cluster monitoring loop: {e}")


class JobScheduler:
    """
    Job scheduler for distributed computing.
    
    This class provides job scheduling capabilities including job queuing,
    resource allocation, and job execution management.
    """
    
    def __init__(self):
        """Initialize the job scheduler."""
        self.jobs = {}
        self.job_queue = queue.PriorityQueue()
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.scheduler_thread = None
        self.scheduler_active = False
        
        logger.info("Job Scheduler initialized")
    
    async def submit_job(
        self,
        name: str,
        command: str,
        working_directory: str = "/tmp",
        environment_variables: Dict[str, str] = None,
        cpu_cores: int = 1,
        memory_gb: int = 1,
        storage_gb: int = 1,
        gpu_count: int = 0,
        priority: JobPriority = JobPriority.MEDIUM,
        max_runtime: float = 3600.0
    ) -> str:
        """Submit a job to the scheduler."""
        try:
            job_id = str(uuid.uuid4())
            
            job = ComputeJob(
                job_id=job_id,
                name=name,
                description=f"Job: {name}",
                command=command,
                working_directory=working_directory,
                environment_variables=environment_variables or {},
                cpu_cores=cpu_cores,
                memory_gb=memory_gb,
                storage_gb=storage_gb,
                gpu_count=gpu_count,
                priority=priority,
                max_runtime=max_runtime,
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            
            self.jobs[job_id] = job
            
            # Add job to queue
            self.job_queue.put((priority.value, job_id))
            
            # Start scheduler if not already active
            if not self.scheduler_active:
                await self.start_scheduler()
            
            logger.info(f"Job submitted to scheduler: {job_id}")
            return job_id
            
        except Exception as e:
            logger.error(f"Error submitting job to scheduler: {e}")
            return ""
    
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a job."""
        try:
            if job_id in self.jobs:
                job = self.jobs[job_id]
                
                if job.status == JobStatus.RUNNING:
                    # Job is running, need to terminate it
                    # This would require process termination logic
                    job.status = JobStatus.CANCELLED
                    job.end_time = datetime.now()
                    job.updated_at = datetime.now()
                    
                    logger.info(f"Job cancelled: {job_id}")
                    return True
                elif job.status == JobStatus.PENDING:
                    # Job is pending, just mark as cancelled
                    job.status = JobStatus.CANCELLED
                    job.end_time = datetime.now()
                    job.updated_at = datetime.now()
                    
                    logger.info(f"Pending job cancelled: {job_id}")
                    return True
                else:
                    logger.warning(f"Job {job_id} is not in a cancellable state")
                    return False
            else:
                logger.warning(f"Job not found: {job_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error cancelling job: {e}")
            return False
    
    async def get_job(self, job_id: str) -> Optional[ComputeJob]:
        """Get job by ID."""
        try:
            return self.jobs.get(job_id)
            
        except Exception as e:
            logger.error(f"Error getting job: {e}")
            return None
    
    async def list_jobs(self, status: JobStatus = None) -> List[ComputeJob]:
        """List jobs."""
        try:
            if status:
                return [job for job in self.jobs.values() if job.status == status]
            else:
                return list(self.jobs.values())
                
        except Exception as e:
            logger.error(f"Error listing jobs: {e}")
            return []
    
    async def start_scheduler(self):
        """Start the job scheduler."""
        try:
            if not self.scheduler_active:
                self.scheduler_active = True
                self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
                self.scheduler_thread.start()
                
                logger.info("Job scheduler started")
                
        except Exception as e:
            logger.error(f"Error starting job scheduler: {e}")
    
    async def stop_scheduler(self):
        """Stop the job scheduler."""
        try:
            if self.scheduler_active:
                self.scheduler_active = False
                if self.scheduler_thread:
                    self.scheduler_thread.join(timeout=5.0)
                
                logger.info("Job scheduler stopped")
                
        except Exception as e:
            logger.error(f"Error stopping job scheduler: {e}")
    
    def _scheduler_loop(self):
        """Job scheduler main loop."""
        try:
            while self.scheduler_active:
                try:
                    # Get next job from queue
                    priority, job_id = self.job_queue.get(timeout=1.0)
                    
                    if job_id in self.jobs:
                        job = self.jobs[job_id]
                        
                        if job.status == JobStatus.PENDING:
                            # Submit job for execution
                            asyncio.create_task(self._execute_job(job))
                    
                except queue.Empty:
                    # No jobs in queue, continue
                    continue
                except Exception as e:
                    logger.error(f"Error in scheduler loop: {e}")
                    continue
                
        except Exception as e:
            logger.error(f"Error in scheduler loop: {e}")
    
    async def _execute_job(self, job: ComputeJob):
        """Execute a job."""
        try:
            # Find available node
            # This would integrate with ClusterManager
            # For now, simulate execution
            
            job.status = JobStatus.RUNNING
            job.start_time = datetime.now()
            job.updated_at = datetime.now()
            
            logger.info(f"Executing job: {job.job_id}")
            
            # Simulate job execution
            await asyncio.sleep(1.0)  # Simulate execution time
            
            # Job completed
            job.status = JobStatus.COMPLETED
            job.end_time = datetime.now()
            job.exit_code = 0
            job.stdout = "Job completed successfully"
            job.updated_at = datetime.now()
            
            logger.info(f"Job completed: {job.job_id}")
            
        except Exception as e:
            # Job failed
            job.status = JobStatus.FAILED
            job.end_time = datetime.now()
            job.exit_code = 1
            job.stderr = str(e)
            job.updated_at = datetime.now()
            
            logger.error(f"Job failed: {job.job_id}, error: {e}")
            
            # Retry if retry count is less than max retries
            if job.retry_count < job.max_retries:
                job.retry_count += 1
                job.status = JobStatus.PENDING
                job.start_time = None
                job.end_time = None
                job.updated_at = datetime.now()
                
                # Re-queue job
                self.job_queue.put((job.priority.value, job.job_id))
                
                logger.info(f"Job retry {job.retry_count}/{job.max_retries}: {job.job_id}")
