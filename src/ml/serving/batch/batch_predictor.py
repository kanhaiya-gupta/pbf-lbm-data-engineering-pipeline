"""
Batch Prediction Service

This module implements the batch prediction service for PBF-LB/M processes.
It provides REST API endpoints for batch data processing, large-scale predictions,
and result aggregation for historical analysis and reporting.
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

from ...pipelines.inference.batch_inference import BatchInferencePipeline
from ...models.base_model import BaseModel
from ...config.config_manager import ConfigManager

logger = logging.getLogger(__name__)


# Pydantic models for API requests and responses
class BatchPredictionRequest(BaseModel):
    """Request model for batch prediction."""
    data_source: str = Field(..., description="Path to data source or data dictionary")
    model_configs: List[Dict[str, Any]] = Field(..., description="List of model configurations")
    batch_size: int = Field(1000, description="Size of each batch for processing")
    output_format: str = Field("json", description="Output format (json, csv, parquet)")
    output_path: Optional[str] = Field(None, description="Optional output path")
    prediction_type: str = Field("all", description="Type of predictions (all, process_optimization, defect_detection, quality_assessment, maintenance)")


class BatchPredictionResponse(BaseModel):
    """Response model for batch prediction."""
    job_id: str = Field(..., description="Batch prediction job ID")
    status: str = Field(..., description="Job status (pending, running, completed, failed)")
    total_records: int = Field(..., description="Total number of records processed")
    successful_predictions: int = Field(..., description="Number of successful predictions")
    failed_predictions: int = Field(..., description="Number of failed predictions")
    processing_time: float = Field(..., description="Total processing time in seconds")
    output_path: Optional[str] = Field(None, description="Path to output file")
    results_summary: Dict[str, Any] = Field(..., description="Summary of prediction results")
    timestamp: str = Field(..., description="Response timestamp")


class BatchJobStatus(BaseModel):
    """Model for batch job status."""
    job_id: str = Field(..., description="Job ID")
    status: str = Field(..., description="Job status")
    progress: float = Field(..., description="Progress percentage (0-100)")
    current_batch: int = Field(..., description="Current batch being processed")
    total_batches: int = Field(..., description="Total number of batches")
    start_time: str = Field(..., description="Job start time")
    estimated_completion: Optional[str] = Field(None, description="Estimated completion time")
    error_message: Optional[str] = Field(None, description="Error message if failed")


class BatchResultsRequest(BaseModel):
    """Request model for batch results retrieval."""
    job_id: str = Field(..., description="Batch prediction job ID")
    format: str = Field("json", description="Results format (json, csv, parquet)")
    limit: Optional[int] = Field(None, description="Limit number of results returned")


class BatchResultsResponse(BaseModel):
    """Response model for batch results."""
    job_id: str = Field(..., description="Job ID")
    results: Dict[str, Any] = Field(..., description="Prediction results")
    metadata: Dict[str, Any] = Field(..., description="Results metadata")
    total_results: int = Field(..., description="Total number of results")
    timestamp: str = Field(..., description="Response timestamp")


class BatchPredictorService:
    """
    Batch prediction service for PBF-LB/M processes.
    
    This service provides batch prediction capabilities for:
    - Large-scale data processing
    - Historical analysis
    - Batch reporting
    - Model evaluation
    """
    
    def __init__(self, config_manager: Optional[ConfigManager] = None):
        """
        Initialize the batch predictor service.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config_manager = config_manager or ConfigManager()
        self.app = FastAPI(
            title="Batch Prediction Service",
            description="Batch prediction for PBF-LB/M manufacturing",
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
        
        # Job management
        self.jobs = {}  # Store job information
        self.job_counter = 0
        
        # Service metrics
        self.service_metrics = {
            'total_jobs': 0,
            'completed_jobs': 0,
            'failed_jobs': 0,
            'total_records_processed': 0,
            'average_processing_time': 0.0,
            'last_job_time': None
        }
        
        # Setup routes
        self._setup_routes()
        
        logger.info("Initialized BatchPredictorService")
    
    def _setup_routes(self):
        """Setup API routes."""
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {
                "status": "healthy",
                "service": "batch_predictor",
                "timestamp": datetime.now().isoformat(),
                "metrics": self.service_metrics
            }
        
        @self.app.post("/predict", response_model=BatchPredictionResponse)
        async def predict_batch(request: BatchPredictionRequest, background_tasks: BackgroundTasks):
            """Submit batch prediction job."""
            return await self._submit_batch_job(request, background_tasks)
        
        @self.app.get("/job/{job_id}/status", response_model=BatchJobStatus)
        async def get_job_status(job_id: str):
            """Get batch job status."""
            return await self._get_job_status(job_id)
        
        @self.app.get("/job/{job_id}/results", response_model=BatchResultsResponse)
        async def get_job_results(job_id: str, format: str = "json", limit: Optional[int] = None):
            """Get batch job results."""
            return await self._get_job_results(job_id, format, limit)
        
        @self.app.delete("/job/{job_id}")
        async def cancel_job(job_id: str):
            """Cancel batch job."""
            return await self._cancel_job(job_id)
        
        @self.app.get("/jobs")
        async def list_jobs():
            """List all batch jobs."""
            return await self._list_jobs()
        
        @self.app.get("/metrics")
        async def get_metrics():
            """Get service metrics."""
            return self.service_metrics
    
    async def _submit_batch_job(self, request: BatchPredictionRequest, background_tasks: BackgroundTasks) -> BatchPredictionResponse:
        """
        Submit batch prediction job.
        
        Args:
            request: Batch prediction request
            background_tasks: FastAPI background tasks
            
        Returns:
            Batch prediction response
        """
        # Generate job ID
        self.job_counter += 1
        job_id = f"batch_job_{self.job_counter}_{int(time.time())}"
        
        # Create job entry
        job_info = {
            'job_id': job_id,
            'status': 'pending',
            'progress': 0.0,
            'current_batch': 0,
            'total_batches': 0,
            'start_time': datetime.now().isoformat(),
            'request': request.dict(),
            'results': None,
            'error_message': None,
            'processing_time': 0.0
        }
        
        self.jobs[job_id] = job_info
        
        # Submit background task
        background_tasks.add_task(self._process_batch_job, job_id, request)
        
        # Update metrics
        self.service_metrics['total_jobs'] += 1
        
        return BatchPredictionResponse(
            job_id=job_id,
            status='pending',
            total_records=0,
            successful_predictions=0,
            failed_predictions=0,
            processing_time=0.0,
            output_path=request.output_path,
            results_summary={},
            timestamp=datetime.now().isoformat()
        )
    
    async def _process_batch_job(self, job_id: str, request: BatchPredictionRequest):
        """
        Process batch prediction job in background.
        
        Args:
            job_id: Job ID
            request: Batch prediction request
        """
        start_time = time.time()
        
        try:
            # Update job status
            self.jobs[job_id]['status'] = 'running'
            self.jobs[job_id]['start_time'] = datetime.now().isoformat()
            
            # Process batch data
            results = await self.batch_pipeline.process_batch_data(
                data_source=request.data_source,
                model_configs=request.model_configs,
                batch_size=request.batch_size,
                output_path=request.output_path
            )
            
            # Update job with results
            self.jobs[job_id]['status'] = 'completed'
            self.jobs[job_id]['results'] = results
            self.jobs[job_id]['progress'] = 100.0
            self.jobs[job_id]['processing_time'] = time.time() - start_time
            
            # Update metrics
            self.service_metrics['completed_jobs'] += 1
            if 'metadata' in results and 'total_records' in results['metadata']:
                self.service_metrics['total_records_processed'] += results['metadata']['total_records']
            
            logger.info(f"Batch job {job_id} completed successfully")
            
        except Exception as e:
            logger.error(f"Batch job {job_id} failed: {e}")
            
            # Update job with error
            self.jobs[job_id]['status'] = 'failed'
            self.jobs[job_id]['error_message'] = str(e)
            self.jobs[job_id]['processing_time'] = time.time() - start_time
            
            # Update metrics
            self.service_metrics['failed_jobs'] += 1
    
    async def _get_job_status(self, job_id: str) -> BatchJobStatus:
        """
        Get batch job status.
        
        Args:
            job_id: Job ID
            
        Returns:
            Batch job status
        """
        if job_id not in self.jobs:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        
        job_info = self.jobs[job_id]
        
        # Calculate estimated completion if job is running
        estimated_completion = None
        if job_info['status'] == 'running' and job_info['progress'] > 0:
            elapsed_time = time.time() - datetime.fromisoformat(job_info['start_time']).timestamp()
            remaining_progress = 100.0 - job_info['progress']
            if remaining_progress > 0:
                estimated_remaining_time = (elapsed_time / job_info['progress']) * remaining_progress
                estimated_completion = (datetime.now() + timedelta(seconds=estimated_remaining_time)).isoformat()
        
        return BatchJobStatus(
            job_id=job_id,
            status=job_info['status'],
            progress=job_info['progress'],
            current_batch=job_info['current_batch'],
            total_batches=job_info['total_batches'],
            start_time=job_info['start_time'],
            estimated_completion=estimated_completion,
            error_message=job_info.get('error_message')
        )
    
    async def _get_job_results(self, job_id: str, format: str = "json", limit: Optional[int] = None) -> BatchResultsResponse:
        """
        Get batch job results.
        
        Args:
            job_id: Job ID
            format: Results format
            limit: Limit number of results
            
        Returns:
            Batch results response
        """
        if job_id not in self.jobs:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        
        job_info = self.jobs[job_id]
        
        if job_info['status'] != 'completed':
            raise HTTPException(status_code=400, detail=f"Job {job_id} is not completed yet")
        
        results = job_info['results']
        
        # Apply limit if specified
        if limit and 'results' in results:
            for model_name, model_results in results['results'].items():
                if 'model_results' in model_results:
                    for result_type, result_data in model_results['model_results'].items():
                        if 'predictions' in result_data and len(result_data['predictions']) > limit:
                            result_data['predictions'] = result_data['predictions'][:limit]
        
        # Format results based on requested format
        if format == "csv":
            # Convert to CSV format
            formatted_results = self._format_results_as_csv(results)
        elif format == "parquet":
            # Convert to Parquet format
            formatted_results = self._format_results_as_parquet(results)
        else:
            # Default JSON format
            formatted_results = results
        
        return BatchResultsResponse(
            job_id=job_id,
            results=formatted_results,
            metadata=results.get('metadata', {}),
            total_results=results.get('metadata', {}).get('total_records', 0),
            timestamp=datetime.now().isoformat()
        )
    
    async def _cancel_job(self, job_id: str) -> Dict[str, Any]:
        """
        Cancel batch job.
        
        Args:
            job_id: Job ID
            
        Returns:
            Cancellation response
        """
        if job_id not in self.jobs:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        
        job_info = self.jobs[job_id]
        
        if job_info['status'] in ['completed', 'failed']:
            raise HTTPException(status_code=400, detail=f"Job {job_id} is already finished")
        
        # Update job status
        self.jobs[job_id]['status'] = 'cancelled'
        self.jobs[job_id]['error_message'] = 'Job cancelled by user'
        
        return {
            "job_id": job_id,
            "status": "cancelled",
            "message": "Job cancelled successfully",
            "timestamp": datetime.now().isoformat()
        }
    
    async def _list_jobs(self) -> Dict[str, Any]:
        """
        List all batch jobs.
        
        Returns:
            List of jobs
        """
        job_list = []
        for job_id, job_info in self.jobs.items():
            job_list.append({
                "job_id": job_id,
                "status": job_info['status'],
                "progress": job_info['progress'],
                "start_time": job_info['start_time'],
                "processing_time": job_info.get('processing_time', 0.0),
                "error_message": job_info.get('error_message')
            })
        
        return {
            "jobs": job_list,
            "total_jobs": len(job_list),
            "timestamp": datetime.now().isoformat()
        }
    
    def _format_results_as_csv(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Format results as CSV."""
        # This would implement CSV formatting
        # For now, return the original results
        return results
    
    def _format_results_as_parquet(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Format results as Parquet."""
        # This would implement Parquet formatting
        # For now, return the original results
        return results
    
    def _update_service_metrics(self, start_time: float, success: bool, record_count: int = 0):
        """Update service metrics."""
        processing_time = time.time() - start_time
        
        if success:
            self.service_metrics['completed_jobs'] += 1
        else:
            self.service_metrics['failed_jobs'] += 1
        
        self.service_metrics['total_records_processed'] += record_count
        
        # Update average processing time
        total_completed = self.service_metrics['completed_jobs']
        if total_completed > 0:
            current_avg = self.service_metrics['average_processing_time']
            self.service_metrics['average_processing_time'] = (
                (current_avg * (total_completed - 1) + processing_time) / total_completed
            )
        
        self.service_metrics['last_job_time'] = datetime.now().isoformat()
    
    def run(self, host: str = "0.0.0.0", port: int = 8005):
        """Run the service."""
        logger.info(f"Starting Batch Prediction Service on {host}:{port}")
        uvicorn.run(self.app, host=host, port=port)


# Global service instance
service = BatchPredictorService()


def create_app() -> FastAPI:
    """Create FastAPI application."""
    return service.app


if __name__ == "__main__":
    service.run()
