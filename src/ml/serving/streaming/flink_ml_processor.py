"""
Flink ML Processor

This module implements the Flink ML processor for PBF-LB/M processes.
It provides stream processing integration with Apache Flink,
real-time data processing, and distributed ML inference.
"""

import logging
import time
import asyncio
from typing import Dict, Any, Optional, List, Callable
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import pickle
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import mlflow
import mlflow.tensorflow
from pathlib import Path

from ...pipelines.inference.streaming_inference import StreamingInferencePipeline
from ...models.base_model import BaseModel
from ...config.config_manager import ConfigManager

logger = logging.getLogger(__name__)


# Pydantic models for API requests and responses
class FlinkProcessorConfig(BaseModel):
    """Configuration for Flink ML processor."""
    flink_cluster_url: str = Field(..., description="Flink cluster URL")
    job_name: str = Field(..., description="Flink job name")
    parallelism: int = Field(4, description="Job parallelism")
    checkpoint_interval: int = Field(60000, description="Checkpoint interval in ms")
    checkpoint_mode: str = Field("EXACTLY_ONCE", description="Checkpoint mode")
    restart_strategy: str = Field("fixed-delay", description="Restart strategy")
    max_restart_attempts: int = Field(3, description="Max restart attempts")
    restart_delay: int = Field(10000, description="Restart delay in ms")
    input_sources: List[Dict[str, Any]] = Field(..., description="Input data sources")
    output_sinks: List[Dict[str, Any]] = Field(..., description="Output data sinks")


class FlinkProcessingRequest(BaseModel):
    """Request model for Flink processing."""
    processor_id: str = Field(..., description="Processor ID")
    model_configs: List[Dict[str, Any]] = Field(..., description="Model configurations")
    processing_config: Dict[str, Any] = Field(..., description="Processing configuration")
    flink_config: FlinkProcessorConfig = Field(..., description="Flink configuration")
    enabled: bool = Field(True, description="Whether processor is enabled")


class FlinkProcessingResponse(BaseModel):
    """Response model for Flink processing."""
    processor_id: str = Field(..., description="Processor ID")
    status: str = Field(..., description="Processor status")
    flink_job_id: Optional[str] = Field(None, description="Flink job ID")
    job_name: str = Field(..., description="Flink job name")
    models_loaded: List[str] = Field(..., description="List of loaded models")
    created_at: str = Field(..., description="Processor creation timestamp")
    message: str = Field(..., description="Response message")


class FlinkJobStatus(BaseModel):
    """Model for Flink job status."""
    job_id: str = Field(..., description="Flink job ID")
    job_name: str = Field(..., description="Job name")
    status: str = Field(..., description="Job status")
    start_time: str = Field(..., description="Job start time")
    end_time: Optional[str] = Field(None, description="Job end time")
    duration: Optional[float] = Field(None, description="Job duration in seconds")
    parallelism: int = Field(..., description="Job parallelism")
    total_processed_records: int = Field(..., description="Total processed records")
    records_per_second: float = Field(..., description="Records per second")
    checkpoint_count: int = Field(..., description="Number of checkpoints")
    last_checkpoint_time: Optional[str] = Field(None, description="Last checkpoint time")


class FlinkMLProcessor:
    """
    Flink ML processor for PBF-LB/M processes.
    
    This processor provides stream processing capabilities for:
    - Real-time data processing
    - Distributed ML inference
    - Stream processing integration
    - Fault-tolerant processing
    """
    
    def __init__(self, config_manager: Optional[ConfigManager] = None):
        """
        Initialize the Flink ML processor.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config_manager = config_manager or ConfigManager()
        self.app = FastAPI(
            title="Flink ML Processor",
            description="Distributed ML processing for PBF-LB/M manufacturing",
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
        
        # Initialize streaming inference pipeline
        self.streaming_pipeline = StreamingInferencePipeline(self.config_manager)
        
        # Processor management
        self.processors = {}  # Store processor information
        self.flink_jobs = {}  # Store Flink job information
        self.processor_counter = 0
        
        # Service metrics
        self.service_metrics = {
            'total_processors': 0,
            'active_processors': 0,
            'total_jobs': 0,
            'running_jobs': 0,
            'total_records_processed': 0,
            'total_errors': 0,
            'last_processing_time': None
        }
        
        # Setup routes
        self._setup_routes()
        
        logger.info("Initialized FlinkMLProcessor")
    
    def _setup_routes(self):
        """Setup API routes."""
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {
                "status": "healthy",
                "service": "flink_ml_processor",
                "timestamp": datetime.now().isoformat(),
                "metrics": self.service_metrics
            }
        
        @self.app.post("/processors", response_model=FlinkProcessingResponse)
        async def create_processor(request: FlinkProcessingRequest):
            """Create a new Flink processor."""
            return await self._create_processor(request)
        
        @self.app.get("/processors")
        async def list_processors():
            """List all processors."""
            return await self._list_processors()
        
        @self.app.get("/processors/{processor_id}/status")
        async def get_processor_status(processor_id: str):
            """Get processor status."""
            return await self._get_processor_status(processor_id)
        
        @self.app.get("/processors/{processor_id}/job-status", response_model=FlinkJobStatus)
        async def get_job_status(processor_id: str):
            """Get Flink job status."""
            return await self._get_job_status(processor_id)
        
        @self.app.post("/processors/{processor_id}/start")
        async def start_processor(processor_id: str):
            """Start a processor."""
            return await self._start_processor(processor_id)
        
        @self.app.post("/processors/{processor_id}/stop")
        async def stop_processor(processor_id: str):
            """Stop a processor."""
            return await self._stop_processor(processor_id)
        
        @self.app.post("/processors/{processor_id}/restart")
        async def restart_processor(processor_id: str):
            """Restart a processor."""
            return await self._restart_processor(processor_id)
        
        @self.app.delete("/processors/{processor_id}")
        async def delete_processor(processor_id: str):
            """Delete a processor."""
            return await self._delete_processor(processor_id)
        
        @self.app.get("/jobs")
        async def list_jobs():
            """List all Flink jobs."""
            return await self._list_jobs()
        
        @self.app.get("/metrics")
        async def get_metrics():
            """Get service metrics."""
            return self.service_metrics
    
    async def _create_processor(self, request: FlinkProcessingRequest) -> FlinkProcessingResponse:
        """
        Create a new Flink processor.
        
        Args:
            request: Flink processing request
            
        Returns:
            Flink processing response
        """
        # Generate processor ID
        self.processor_counter += 1
        processor_id = f"flink_processor_{self.processor_counter}_{int(time.time())}"
        
        # Create processor entry
        processor_info = {
            'processor_id': processor_id,
            'model_configs': request.model_configs,
            'processing_config': request.processing_config,
            'flink_config': request.flink_config.dict(),
            'enabled': request.enabled,
            'status': 'created',
            'created_at': datetime.now().isoformat(),
            'started_at': None,
            'stopped_at': None,
            'flink_job_id': None,
            'total_records_processed': 0,
            'error_count': 0,
            'last_record_time': None
        }
        
        self.processors[processor_id] = processor_info
        
        # Load models
        models_loaded = []
        for model_config in request.model_configs:
            try:
                model = await self._load_model(model_config)
                models_loaded.append(model_config['model_name'])
            except Exception as e:
                logger.error(f"Failed to load model {model_config['model_name']}: {e}")
        
        # Update metrics
        self.service_metrics['total_processors'] += 1
        if request.enabled:
            self.service_metrics['active_processors'] += 1
        
        return FlinkProcessingResponse(
            processor_id=processor_id,
            status='created',
            flink_job_id=None,
            job_name=request.flink_config.job_name,
            models_loaded=models_loaded,
            created_at=processor_info['created_at'],
            message="Flink processor created successfully"
        )
    
    async def _load_model(self, model_config: Dict[str, Any]) -> Any:
        """Load a model for processing."""
        # This would implement actual model loading logic
        # For now, return a mock model
        return f"model_{model_config['model_name']}"
    
    async def _list_processors(self) -> Dict[str, Any]:
        """List all processors."""
        processor_list = []
        for processor_id, processor_info in self.processors.items():
            processor_list.append({
                'processor_id': processor_id,
                'status': processor_info['status'],
                'job_name': processor_info['flink_config']['job_name'],
                'flink_job_id': processor_info['flink_job_id'],
                'enabled': processor_info['enabled'],
                'total_records_processed': processor_info['total_records_processed'],
                'error_count': processor_info['error_count'],
                'created_at': processor_info['created_at']
            })
        
        return {
            'processors': processor_list,
            'total_processors': len(processor_list),
            'active_processors': sum(1 for p in processor_list if p['enabled']),
            'timestamp': datetime.now().isoformat()
        }
    
    async def _get_processor_status(self, processor_id: str) -> Dict[str, Any]:
        """Get processor status."""
        if processor_id not in self.processors:
            raise HTTPException(status_code=404, detail=f"Processor {processor_id} not found")
        
        processor_info = self.processors[processor_id]
        
        return {
            'processor_id': processor_id,
            'status': processor_info['status'],
            'enabled': processor_info['enabled'],
            'job_name': processor_info['flink_config']['job_name'],
            'flink_job_id': processor_info['flink_job_id'],
            'total_records_processed': processor_info['total_records_processed'],
            'error_count': processor_info['error_count'],
            'last_record_time': processor_info['last_record_time'],
            'created_at': processor_info['created_at'],
            'started_at': processor_info['started_at'],
            'stopped_at': processor_info['stopped_at']
        }
    
    async def _get_job_status(self, processor_id: str) -> FlinkJobStatus:
        """Get Flink job status."""
        if processor_id not in self.processors:
            raise HTTPException(status_code=404, detail=f"Processor {processor_id} not found")
        
        processor_info = self.processors[processor_id]
        
        if processor_info['flink_job_id'] not in self.flink_jobs:
            raise HTTPException(status_code=404, detail=f"Flink job not found")
        
        job_info = self.flink_jobs[processor_info['flink_job_id']]
        
        return FlinkJobStatus(
            job_id=job_info['job_id'],
            job_name=job_info['job_name'],
            status=job_info['status'],
            start_time=job_info['start_time'],
            end_time=job_info['end_time'],
            duration=job_info['duration'],
            parallelism=job_info['parallelism'],
            total_processed_records=job_info['total_processed_records'],
            records_per_second=job_info['records_per_second'],
            checkpoint_count=job_info['checkpoint_count'],
            last_checkpoint_time=job_info['last_checkpoint_time']
        )
    
    async def _start_processor(self, processor_id: str) -> Dict[str, Any]:
        """Start a processor."""
        if processor_id not in self.processors:
            raise HTTPException(status_code=404, detail=f"Processor {processor_id} not found")
        
        processor_info = self.processors[processor_id]
        processor_info['status'] = 'starting'
        processor_info['started_at'] = datetime.now().isoformat()
        processor_info['stopped_at'] = None
        
        try:
            # Submit Flink job
            flink_job_id = await self._submit_flink_job(processor_info)
            processor_info['flink_job_id'] = flink_job_id
            processor_info['status'] = 'running'
            
            # Create job entry
            self.flink_jobs[flink_job_id] = {
                'job_id': flink_job_id,
                'job_name': processor_info['flink_config']['job_name'],
                'status': 'running',
                'start_time': processor_info['started_at'],
                'end_time': None,
                'duration': None,
                'parallelism': processor_info['flink_config']['parallelism'],
                'total_processed_records': 0,
                'records_per_second': 0.0,
                'checkpoint_count': 0,
                'last_checkpoint_time': None
            }
            
            self.service_metrics['active_processors'] += 1
            self.service_metrics['total_jobs'] += 1
            self.service_metrics['running_jobs'] += 1
            
            return {
                'processor_id': processor_id,
                'status': 'started',
                'flink_job_id': flink_job_id,
                'message': 'Flink processor started successfully',
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to start Flink processor {processor_id}: {e}")
            processor_info['status'] = 'error'
            processor_info['error_count'] += 1
            self.service_metrics['total_errors'] += 1
            
            raise HTTPException(status_code=500, detail=str(e))
    
    async def _stop_processor(self, processor_id: str) -> Dict[str, Any]:
        """Stop a processor."""
        if processor_id not in self.processors:
            raise HTTPException(status_code=404, detail=f"Processor {processor_id} not found")
        
        processor_info = self.processors[processor_id]
        processor_info['status'] = 'stopping'
        
        try:
            # Cancel Flink job
            if processor_info['flink_job_id']:
                await self._cancel_flink_job(processor_info['flink_job_id'])
                
                # Update job status
                if processor_info['flink_job_id'] in self.flink_jobs:
                    job_info = self.flink_jobs[processor_info['flink_job_id']]
                    job_info['status'] = 'cancelled'
                    job_info['end_time'] = datetime.now().isoformat()
                    job_info['duration'] = (datetime.now() - datetime.fromisoformat(job_info['start_time'])).total_seconds()
                    
                    self.service_metrics['running_jobs'] -= 1
            
            processor_info['status'] = 'stopped'
            processor_info['stopped_at'] = datetime.now().isoformat()
            
            self.service_metrics['active_processors'] -= 1
            
            return {
                'processor_id': processor_id,
                'status': 'stopped',
                'message': 'Flink processor stopped successfully',
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to stop Flink processor {processor_id}: {e}")
            processor_info['status'] = 'error'
            processor_info['error_count'] += 1
            self.service_metrics['total_errors'] += 1
            
            raise HTTPException(status_code=500, detail=str(e))
    
    async def _restart_processor(self, processor_id: str) -> Dict[str, Any]:
        """Restart a processor."""
        if processor_id not in self.processors:
            raise HTTPException(status_code=404, detail=f"Processor {processor_id} not found")
        
        # Stop the processor
        await self._stop_processor(processor_id)
        
        # Wait a bit
        await asyncio.sleep(2)
        
        # Start the processor
        await self._start_processor(processor_id)
        
        return {
            'processor_id': processor_id,
            'status': 'restarted',
            'message': 'Flink processor restarted successfully',
            'timestamp': datetime.now().isoformat()
        }
    
    async def _delete_processor(self, processor_id: str) -> Dict[str, Any]:
        """Delete a processor."""
        if processor_id not in self.processors:
            raise HTTPException(status_code=404, detail=f"Processor {processor_id} not found")
        
        processor_info = self.processors[processor_id]
        
        # Stop processor if running
        if processor_info['status'] in ['running', 'starting']:
            await self._stop_processor(processor_id)
        
        # Remove processor from memory
        del self.processors[processor_id]
        
        self.service_metrics['total_processors'] -= 1
        self.service_metrics['active_processors'] -= 1
        
        return {
            'processor_id': processor_id,
            'status': 'deleted',
            'message': 'Flink processor deleted successfully',
            'timestamp': datetime.now().isoformat()
        }
    
    async def _submit_flink_job(self, processor_info: Dict[str, Any]) -> str:
        """Submit a Flink job."""
        # This would implement actual Flink job submission
        # For now, return a mock job ID
        job_id = f"flink_job_{int(time.time())}"
        
        # Simulate job submission
        await asyncio.sleep(1)
        
        return job_id
    
    async def _cancel_flink_job(self, job_id: str):
        """Cancel a Flink job."""
        # This would implement actual Flink job cancellation
        # For now, just simulate
        await asyncio.sleep(1)
        
        if job_id in self.flink_jobs:
            job_info = self.flink_jobs[job_id]
            job_info['status'] = 'cancelled'
            job_info['end_time'] = datetime.now().isoformat()
            job_info['duration'] = (datetime.now() - datetime.fromisoformat(job_info['start_time'])).total_seconds()
    
    async def _list_jobs(self) -> Dict[str, Any]:
        """List all Flink jobs."""
        job_list = []
        for job_id, job_info in self.flink_jobs.items():
            job_list.append({
                'job_id': job_id,
                'job_name': job_info['job_name'],
                'status': job_info['status'],
                'start_time': job_info['start_time'],
                'end_time': job_info['end_time'],
                'duration': job_info['duration'],
                'parallelism': job_info['parallelism'],
                'total_processed_records': job_info['total_processed_records'],
                'records_per_second': job_info['records_per_second']
            })
        
        return {
            'jobs': job_list,
            'total_jobs': len(job_list),
            'running_jobs': sum(1 for j in job_list if j['status'] == 'running'),
            'timestamp': datetime.now().isoformat()
        }
    
    def run(self, host: str = "0.0.0.0", port: int = 8010):
        """Run the service."""
        logger.info(f"Starting Flink ML Processor on {host}:{port}")
        uvicorn.run(self.app, host=host, port=port)


# Global service instance
service = FlinkMLProcessor()


def create_app() -> FastAPI:
    """Create FastAPI application."""
    return service.app


if __name__ == "__main__":
    service.run()
