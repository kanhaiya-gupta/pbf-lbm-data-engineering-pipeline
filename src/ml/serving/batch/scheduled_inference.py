"""
Scheduled Inference Service

This module implements the scheduled inference service for PBF-LB/M processes.
It provides REST API endpoints for scheduling periodic inference jobs,
automated predictions, and scheduled reporting.
"""

import logging
import time
import asyncio
from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import mlflow
import mlflow.tensorflow
from pathlib import Path
import json
import pickle
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from ...pipelines.inference.batch_inference import BatchInferencePipeline
from ...models.base_model import BaseModel
from ...config.config_manager import ConfigManager

logger = logging.getLogger(__name__)


# Pydantic models for API requests and responses
class ScheduledJobRequest(BaseModel):
    """Request model for scheduled job creation."""
    job_name: str = Field(..., description="Name of the scheduled job")
    job_type: str = Field(..., description="Type of job (inference, analysis, report)")
    schedule_type: str = Field(..., description="Schedule type (cron, interval, once)")
    schedule_config: Dict[str, Any] = Field(..., description="Schedule configuration")
    model_configs: List[Dict[str, Any]] = Field(..., description="List of model configurations")
    data_source: str = Field(..., description="Data source for the job")
    output_config: Dict[str, Any] = Field(..., description="Output configuration")
    notification_config: Optional[Dict[str, Any]] = Field(None, description="Notification configuration")
    enabled: bool = Field(True, description="Whether the job is enabled")


class ScheduledJobResponse(BaseModel):
    """Response model for scheduled job creation."""
    job_id: str = Field(..., description="Scheduled job ID")
    job_name: str = Field(..., description="Job name")
    status: str = Field(..., description="Job status")
    schedule_type: str = Field(..., description="Schedule type")
    next_run: Optional[str] = Field(None, description="Next scheduled run time")
    created_at: str = Field(..., description="Job creation timestamp")
    message: str = Field(..., description="Response message")


class ScheduledJobStatus(BaseModel):
    """Model for scheduled job status."""
    job_id: str = Field(..., description="Job ID")
    job_name: str = Field(..., description="Job name")
    status: str = Field(..., description="Job status")
    schedule_type: str = Field(..., description="Schedule type")
    next_run: Optional[str] = Field(None, description="Next scheduled run time")
    last_run: Optional[str] = Field(None, description="Last run time")
    last_run_status: Optional[str] = Field(None, description="Last run status")
    total_runs: int = Field(..., description="Total number of runs")
    successful_runs: int = Field(..., description="Number of successful runs")
    failed_runs: int = Field(..., description="Number of failed runs")
    enabled: bool = Field(..., description="Whether job is enabled")


class ScheduledJobExecution(BaseModel):
    """Model for scheduled job execution."""
    execution_id: str = Field(..., description="Execution ID")
    job_id: str = Field(..., description="Job ID")
    start_time: str = Field(..., description="Execution start time")
    end_time: Optional[str] = Field(None, description="Execution end time")
    status: str = Field(..., description="Execution status")
    duration: Optional[float] = Field(None, description="Execution duration in seconds")
    records_processed: Optional[int] = Field(None, description="Number of records processed")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    output_path: Optional[str] = Field(None, description="Output file path")


class ScheduledInferenceService:
    """
    Scheduled inference service for PBF-LB/M processes.
    
    This service provides scheduled inference capabilities for:
    - Periodic batch predictions
    - Automated analysis
    - Scheduled reporting
    - Model monitoring
    """
    
    def __init__(self, config_manager: Optional[ConfigManager] = None):
        """
        Initialize the scheduled inference service.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config_manager = config_manager or ConfigManager()
        self.app = FastAPI(
            title="Scheduled Inference Service",
            description="Scheduled inference for PBF-LB/M manufacturing",
            version="1.0.0"
        )
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Initialize batch inference pipeline
        self.batch_pipeline = BatchInferencePipeline(self.config_manager)
        
        # Initialize scheduler
        self.scheduler = AsyncIOScheduler()
        self.scheduler.start()
        
        # Job management
        self.jobs = {}  # Store job information
        self.job_executions = {}  # Store execution history
        self.job_counter = 0
        
        # Service metrics
        self.service_metrics = {
            'total_jobs': 0,
            'active_jobs': 0,
            'total_executions': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'last_execution_time': None
        }
        
        # Setup routes
        self._setup_routes()
        
        logger.info("Initialized ScheduledInferenceService")
    
    def _setup_routes(self):
        """Setup API routes."""
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {
                "status": "healthy",
                "service": "scheduled_inference",
                "timestamp": datetime.now().isoformat(),
                "metrics": self.service_metrics
            }
        
        @self.app.post("/jobs", response_model=ScheduledJobResponse)
        async def create_scheduled_job(request: ScheduledJobRequest):
            """Create a new scheduled job."""
            return await self._create_scheduled_job(request)
        
        @self.app.get("/jobs")
        async def list_scheduled_jobs():
            """List all scheduled jobs."""
            return await self._list_scheduled_jobs()
        
        @self.app.get("/jobs/{job_id}/status", response_model=ScheduledJobStatus)
        async def get_job_status(job_id: str):
            """Get scheduled job status."""
            return await self._get_job_status(job_id)
        
        @self.app.get("/jobs/{job_id}/executions")
        async def get_job_executions(job_id: str, limit: int = 10):
            """Get job execution history."""
            return await self._get_job_executions(job_id, limit)
        
        @self.app.post("/jobs/{job_id}/run")
        async def run_job_now(job_id: str):
            """Run a scheduled job immediately."""
            return await self._run_job_now(job_id)
        
        @self.app.post("/jobs/{job_id}/enable")
        async def enable_job(job_id: str):
            """Enable a scheduled job."""
            return await self._enable_job(job_id)
        
        @self.app.post("/jobs/{job_id}/disable")
        async def disable_job(job_id: str):
            """Disable a scheduled job."""
            return await self._disable_job(job_id)
        
        @self.app.delete("/jobs/{job_id}")
        async def delete_job(job_id: str):
            """Delete a scheduled job."""
            return await self._delete_job(job_id)
        
        @self.app.get("/metrics")
        async def get_metrics():
            """Get service metrics."""
            return self.service_metrics
    
    async def _create_scheduled_job(self, request: ScheduledJobRequest) -> ScheduledJobResponse:
        """
        Create a new scheduled job.
        
        Args:
            request: Scheduled job request
            
        Returns:
            Scheduled job response
        """
        # Generate job ID
        self.job_counter += 1
        job_id = f"scheduled_job_{self.job_counter}_{int(time.time())}"
        
        # Create job entry
        job_info = {
            'job_id': job_id,
            'job_name': request.job_name,
            'job_type': request.job_type,
            'schedule_type': request.schedule_type,
            'schedule_config': request.schedule_config,
            'model_configs': request.model_configs,
            'data_source': request.data_source,
            'output_config': request.output_config,
            'notification_config': request.notification_config,
            'enabled': request.enabled,
            'status': 'created',
            'created_at': datetime.now().isoformat(),
            'total_runs': 0,
            'successful_runs': 0,
            'failed_runs': 0,
            'last_run': None,
            'last_run_status': None
        }
        
        self.jobs[job_id] = job_info
        self.job_executions[job_id] = []
        
        # Schedule the job
        if request.enabled:
            await self._schedule_job(job_id, request)
        
        # Update metrics
        self.service_metrics['total_jobs'] += 1
        if request.enabled:
            self.service_metrics['active_jobs'] += 1
        
        return ScheduledJobResponse(
            job_id=job_id,
            job_name=request.job_name,
            status='created',
            schedule_type=request.schedule_type,
            next_run=self._get_next_run_time(job_id),
            created_at=job_info['created_at'],
            message="Scheduled job created successfully"
        )
    
    async def _schedule_job(self, job_id: str, request: ScheduledJobRequest):
        """Schedule a job with the scheduler."""
        job_info = self.jobs[job_id]
        
        # Create trigger based on schedule type
        if request.schedule_type == 'cron':
            trigger = CronTrigger(**request.schedule_config)
        elif request.schedule_type == 'interval':
            trigger = IntervalTrigger(**request.schedule_config)
        elif request.schedule_type == 'once':
            # Schedule for immediate execution
            trigger = None
        else:
            raise ValueError(f"Unsupported schedule type: {request.schedule_type}")
        
        # Add job to scheduler
        if trigger:
            self.scheduler.add_job(
                func=self._execute_scheduled_job,
                trigger=trigger,
                args=[job_id],
                id=job_id,
                name=request.job_name,
                replace_existing=True
            )
        else:
            # Execute immediately for 'once' schedule
            asyncio.create_task(self._execute_scheduled_job(job_id))
    
    async def _execute_scheduled_job(self, job_id: str):
        """Execute a scheduled job."""
        if job_id not in self.jobs:
            logger.error(f"Job {job_id} not found")
            return
        
        job_info = self.jobs[job_id]
        execution_id = f"exec_{job_id}_{int(time.time())}"
        
        # Create execution record
        execution = {
            'execution_id': execution_id,
            'job_id': job_id,
            'start_time': datetime.now().isoformat(),
            'end_time': None,
            'status': 'running',
            'duration': None,
            'records_processed': None,
            'error_message': None,
            'output_path': None
        }
        
        self.job_executions[job_id].append(execution)
        
        start_time = time.time()
        
        try:
            # Update job status
            job_info['status'] = 'running'
            job_info['last_run'] = execution['start_time']
            
            # Execute the job
            if job_info['job_type'] == 'inference':
                results = await self._execute_inference_job(job_info)
            elif job_info['job_type'] == 'analysis':
                results = await self._execute_analysis_job(job_info)
            elif job_info['job_type'] == 'report':
                results = await self._execute_report_job(job_info)
            else:
                raise ValueError(f"Unsupported job type: {job_info['job_type']}")
            
            # Update execution record
            execution['end_time'] = datetime.now().isoformat()
            execution['status'] = 'completed'
            execution['duration'] = time.time() - start_time
            execution['records_processed'] = results.get('total_records', 0)
            execution['output_path'] = results.get('output_path')
            
            # Update job info
            job_info['status'] = 'completed'
            job_info['last_run_status'] = 'success'
            job_info['total_runs'] += 1
            job_info['successful_runs'] += 1
            
            # Update metrics
            self.service_metrics['total_executions'] += 1
            self.service_metrics['successful_executions'] += 1
            self.service_metrics['last_execution_time'] = execution['end_time']
            
            logger.info(f"Scheduled job {job_id} executed successfully")
            
        except Exception as e:
            logger.error(f"Scheduled job {job_id} failed: {e}")
            
            # Update execution record
            execution['end_time'] = datetime.now().isoformat()
            execution['status'] = 'failed'
            execution['duration'] = time.time() - start_time
            execution['error_message'] = str(e)
            
            # Update job info
            job_info['status'] = 'failed'
            job_info['last_run_status'] = 'failed'
            job_info['total_runs'] += 1
            job_info['failed_runs'] += 1
            
            # Update metrics
            self.service_metrics['total_executions'] += 1
            self.service_metrics['failed_executions'] += 1
            self.service_metrics['last_execution_time'] = execution['end_time']
    
    async def _execute_inference_job(self, job_info: Dict[str, Any]) -> Dict[str, Any]:
        """Execute inference job."""
        results = await self.batch_pipeline.process_batch_data(
            data_source=job_info['data_source'],
            model_configs=job_info['model_configs'],
            batch_size=job_info['output_config'].get('batch_size', 1000),
            output_path=job_info['output_config'].get('output_path')
        )
        return results
    
    async def _execute_analysis_job(self, job_info: Dict[str, Any]) -> Dict[str, Any]:
        """Execute analysis job."""
        # This would implement analysis-specific logic
        # For now, return a mock result
        return {
            'total_records': 1000,
            'output_path': job_info['output_config'].get('output_path'),
            'analysis_type': 'batch_analysis'
        }
    
    async def _execute_report_job(self, job_info: Dict[str, Any]) -> Dict[str, Any]:
        """Execute report job."""
        # This would implement report generation logic
        # For now, return a mock result
        return {
            'total_records': 1000,
            'output_path': job_info['output_config'].get('output_path'),
            'report_type': 'scheduled_report'
        }
    
    def _get_next_run_time(self, job_id: str) -> Optional[str]:
        """Get next scheduled run time for a job."""
        try:
            job = self.scheduler.get_job(job_id)
            if job and job.next_run_time:
                return job.next_run_time.isoformat()
        except Exception as e:
            logger.warning(f"Could not get next run time for job {job_id}: {e}")
        return None
    
    async def _list_scheduled_jobs(self) -> Dict[str, Any]:
        """List all scheduled jobs."""
        job_list = []
        for job_id, job_info in self.jobs.items():
            job_list.append({
                'job_id': job_id,
                'job_name': job_info['job_name'],
                'job_type': job_info['job_type'],
                'schedule_type': job_info['schedule_type'],
                'status': job_info['status'],
                'enabled': job_info['enabled'],
                'next_run': self._get_next_run_time(job_id),
                'last_run': job_info['last_run'],
                'total_runs': job_info['total_runs'],
                'successful_runs': job_info['successful_runs'],
                'failed_runs': job_info['failed_runs']
            })
        
        return {
            'jobs': job_list,
            'total_jobs': len(job_list),
            'active_jobs': sum(1 for job in job_list if job['enabled']),
            'timestamp': datetime.now().isoformat()
        }
    
    async def _get_job_status(self, job_id: str) -> ScheduledJobStatus:
        """Get scheduled job status."""
        if job_id not in self.jobs:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        
        job_info = self.jobs[job_id]
        
        return ScheduledJobStatus(
            job_id=job_id,
            job_name=job_info['job_name'],
            status=job_info['status'],
            schedule_type=job_info['schedule_type'],
            next_run=self._get_next_run_time(job_id),
            last_run=job_info['last_run'],
            last_run_status=job_info['last_run_status'],
            total_runs=job_info['total_runs'],
            successful_runs=job_info['successful_runs'],
            failed_runs=job_info['failed_runs'],
            enabled=job_info['enabled']
        )
    
    async def _get_job_executions(self, job_id: str, limit: int = 10) -> Dict[str, Any]:
        """Get job execution history."""
        if job_id not in self.jobs:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        
        executions = self.job_executions.get(job_id, [])
        limited_executions = executions[-limit:] if limit > 0 else executions
        
        return {
            'job_id': job_id,
            'executions': limited_executions,
            'total_executions': len(executions),
            'timestamp': datetime.now().isoformat()
        }
    
    async def _run_job_now(self, job_id: str) -> Dict[str, Any]:
        """Run a scheduled job immediately."""
        if job_id not in self.jobs:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        
        # Execute job immediately
        asyncio.create_task(self._execute_scheduled_job(job_id))
        
        return {
            'job_id': job_id,
            'status': 'triggered',
            'message': 'Job execution triggered',
            'timestamp': datetime.now().isoformat()
        }
    
    async def _enable_job(self, job_id: str) -> Dict[str, Any]:
        """Enable a scheduled job."""
        if job_id not in self.jobs:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        
        job_info = self.jobs[job_id]
        job_info['enabled'] = True
        
        # Reschedule the job
        await self._schedule_job(job_id, ScheduledJobRequest(**job_info))
        
        self.service_metrics['active_jobs'] += 1
        
        return {
            'job_id': job_id,
            'status': 'enabled',
            'message': 'Job enabled successfully',
            'timestamp': datetime.now().isoformat()
        }
    
    async def _disable_job(self, job_id: str) -> Dict[str, Any]:
        """Disable a scheduled job."""
        if job_id not in self.jobs:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        
        job_info = self.jobs[job_id]
        job_info['enabled'] = False
        
        # Remove job from scheduler
        try:
            self.scheduler.remove_job(job_id)
        except Exception as e:
            logger.warning(f"Could not remove job {job_id} from scheduler: {e}")
        
        self.service_metrics['active_jobs'] -= 1
        
        return {
            'job_id': job_id,
            'status': 'disabled',
            'message': 'Job disabled successfully',
            'timestamp': datetime.now().isoformat()
        }
    
    async def _delete_job(self, job_id: str) -> Dict[str, Any]:
        """Delete a scheduled job."""
        if job_id not in self.jobs:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        
        # Remove job from scheduler
        try:
            self.scheduler.remove_job(job_id)
        except Exception as e:
            logger.warning(f"Could not remove job {job_id} from scheduler: {e}")
        
        # Remove job from memory
        del self.jobs[job_id]
        if job_id in self.job_executions:
            del self.job_executions[job_id]
        
        self.service_metrics['total_jobs'] -= 1
        self.service_metrics['active_jobs'] -= 1
        
        return {
            'job_id': job_id,
            'status': 'deleted',
            'message': 'Job deleted successfully',
            'timestamp': datetime.now().isoformat()
        }
    
    def run(self, host: str = "0.0.0.0", port: int = 8006):
        """Run the service."""
        logger.info(f"Starting Scheduled Inference Service on {host}:{port}")
        uvicorn.run(self.app, host=host, port=port)


# Global service instance
service = ScheduledInferenceService()


def create_app() -> FastAPI:
    """Create FastAPI application."""
    return service.app


if __name__ == "__main__":
    service.run()
