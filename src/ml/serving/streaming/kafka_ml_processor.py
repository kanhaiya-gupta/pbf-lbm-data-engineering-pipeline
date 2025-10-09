"""
Kafka ML Processor

This module implements the Kafka ML processor for PBF-LB/M processes.
It provides real-time data processing, stream processing integration,
and Kafka-based ML inference.
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
from kafka import KafkaProducer, KafkaConsumer
from kafka.errors import KafkaError
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
class KafkaProcessorConfig(BaseModel):
    """Configuration for Kafka ML processor."""
    bootstrap_servers: List[str] = Field(..., description="Kafka bootstrap servers")
    input_topic: str = Field(..., description="Input topic for data")
    output_topic: str = Field(..., description="Output topic for predictions")
    group_id: str = Field(..., description="Consumer group ID")
    auto_offset_reset: str = Field("latest", description="Auto offset reset policy")
    enable_auto_commit: bool = Field(True, description="Enable auto commit")
    max_poll_records: int = Field(500, description="Max poll records")
    session_timeout_ms: int = Field(30000, description="Session timeout in ms")
    heartbeat_interval_ms: int = Field(3000, description="Heartbeat interval in ms")


class StreamProcessingRequest(BaseModel):
    """Request model for stream processing."""
    processor_id: str = Field(..., description="Processor ID")
    model_configs: List[Dict[str, Any]] = Field(..., description="Model configurations")
    processing_config: Dict[str, Any] = Field(..., description="Processing configuration")
    kafka_config: KafkaProcessorConfig = Field(..., description="Kafka configuration")
    enabled: bool = Field(True, description="Whether processor is enabled")


class StreamProcessingResponse(BaseModel):
    """Response model for stream processing."""
    processor_id: str = Field(..., description="Processor ID")
    status: str = Field(..., description="Processor status")
    input_topic: str = Field(..., description="Input topic")
    output_topic: str = Field(..., description="Output topic")
    models_loaded: List[str] = Field(..., description="List of loaded models")
    created_at: str = Field(..., description="Processor creation timestamp")
    message: str = Field(..., description="Response message")


class StreamMetrics(BaseModel):
    """Model for stream processing metrics."""
    processor_id: str = Field(..., description="Processor ID")
    total_messages_processed: int = Field(..., description="Total messages processed")
    messages_per_second: float = Field(..., description="Messages per second")
    average_processing_time: float = Field(..., description="Average processing time in ms")
    error_count: int = Field(..., description="Number of errors")
    last_message_time: Optional[str] = Field(None, description="Last message processing time")
    uptime_seconds: float = Field(..., description="Processor uptime in seconds")


class KafkaMLProcessor:
    """
    Kafka ML processor for PBF-LB/M processes.
    
    This processor provides real-time data processing capabilities for:
    - Real-time inference
    - Stream processing
    - Data transformation
    - Model serving
    """
    
    def __init__(self, config_manager: Optional[ConfigManager] = None):
        """
        Initialize the Kafka ML processor.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config_manager = config_manager or ConfigManager()
        self.app = FastAPI(
            title="Kafka ML Processor",
            description="Real-time ML processing for PBF-LB/M manufacturing",
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
        self.processor_metrics = {}  # Store processor metrics
        self.processor_counter = 0
        
        # Service metrics
        self.service_metrics = {
            'total_processors': 0,
            'active_processors': 0,
            'total_messages_processed': 0,
            'total_errors': 0,
            'last_processing_time': None
        }
        
        # Setup routes
        self._setup_routes()
        
        logger.info("Initialized KafkaMLProcessor")
    
    def _setup_routes(self):
        """Setup API routes."""
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {
                "status": "healthy",
                "service": "kafka_ml_processor",
                "timestamp": datetime.now().isoformat(),
                "metrics": self.service_metrics
            }
        
        @self.app.post("/processors", response_model=StreamProcessingResponse)
        async def create_processor(request: StreamProcessingRequest):
            """Create a new stream processor."""
            return await self._create_processor(request)
        
        @self.app.get("/processors")
        async def list_processors():
            """List all processors."""
            return await self._list_processors()
        
        @self.app.get("/processors/{processor_id}/status")
        async def get_processor_status(processor_id: str):
            """Get processor status."""
            return await self._get_processor_status(processor_id)
        
        @self.app.get("/processors/{processor_id}/metrics", response_model=StreamMetrics)
        async def get_processor_metrics(processor_id: str):
            """Get processor metrics."""
            return await self._get_processor_metrics(processor_id)
        
        @self.app.post("/processors/{processor_id}/start")
        async def start_processor(processor_id: str):
            """Start a processor."""
            return await self._start_processor(processor_id)
        
        @self.app.post("/processors/{processor_id}/stop")
        async def stop_processor(processor_id: str):
            """Stop a processor."""
            return await self._stop_processor(processor_id)
        
        @self.app.delete("/processors/{processor_id}")
        async def delete_processor(processor_id: str):
            """Delete a processor."""
            return await self._delete_processor(processor_id)
        
        @self.app.get("/metrics")
        async def get_metrics():
            """Get service metrics."""
            return self.service_metrics
    
    async def _create_processor(self, request: StreamProcessingRequest) -> StreamProcessingResponse:
        """
        Create a new stream processor.
        
        Args:
            request: Stream processing request
            
        Returns:
            Stream processing response
        """
        # Generate processor ID
        self.processor_counter += 1
        processor_id = f"kafka_processor_{self.processor_counter}_{int(time.time())}"
        
        # Create processor entry
        processor_info = {
            'processor_id': processor_id,
            'model_configs': request.model_configs,
            'processing_config': request.processing_config,
            'kafka_config': request.kafka_config.dict(),
            'enabled': request.enabled,
            'status': 'created',
            'created_at': datetime.now().isoformat(),
            'started_at': None,
            'stopped_at': None,
            'total_messages_processed': 0,
            'error_count': 0,
            'last_message_time': None
        }
        
        self.processors[processor_id] = processor_info
        self.processor_metrics[processor_id] = {
            'total_messages_processed': 0,
            'messages_per_second': 0.0,
            'average_processing_time': 0.0,
            'error_count': 0,
            'last_message_time': None,
            'uptime_seconds': 0.0,
            'start_time': None
        }
        
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
        
        return StreamProcessingResponse(
            processor_id=processor_id,
            status='created',
            input_topic=request.kafka_config.input_topic,
            output_topic=request.kafka_config.output_topic,
            models_loaded=models_loaded,
            created_at=processor_info['created_at'],
            message="Stream processor created successfully"
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
                'input_topic': processor_info['kafka_config']['input_topic'],
                'output_topic': processor_info['kafka_config']['output_topic'],
                'enabled': processor_info['enabled'],
                'total_messages_processed': processor_info['total_messages_processed'],
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
            'input_topic': processor_info['kafka_config']['input_topic'],
            'output_topic': processor_info['kafka_config']['output_topic'],
            'total_messages_processed': processor_info['total_messages_processed'],
            'error_count': processor_info['error_count'],
            'last_message_time': processor_info['last_message_time'],
            'created_at': processor_info['created_at'],
            'started_at': processor_info['started_at'],
            'stopped_at': processor_info['stopped_at']
        }
    
    async def _get_processor_metrics(self, processor_id: str) -> StreamMetrics:
        """Get processor metrics."""
        if processor_id not in self.processors:
            raise HTTPException(status_code=404, detail=f"Processor {processor_id} not found")
        
        processor_info = self.processors[processor_id]
        metrics = self.processor_metrics[processor_id]
        
        # Calculate uptime
        uptime_seconds = 0.0
        if processor_info['started_at']:
            start_time = datetime.fromisoformat(processor_info['started_at'])
            if processor_info['stopped_at']:
                stop_time = datetime.fromisoformat(processor_info['stopped_at'])
                uptime_seconds = (stop_time - start_time).total_seconds()
            else:
                uptime_seconds = (datetime.now() - start_time).total_seconds()
        
        return StreamMetrics(
            processor_id=processor_id,
            total_messages_processed=processor_info['total_messages_processed'],
            messages_per_second=metrics['messages_per_second'],
            average_processing_time=metrics['average_processing_time'],
            error_count=processor_info['error_count'],
            last_message_time=processor_info['last_message_time'],
            uptime_seconds=uptime_seconds
        )
    
    async def _start_processor(self, processor_id: str) -> Dict[str, Any]:
        """Start a processor."""
        if processor_id not in self.processors:
            raise HTTPException(status_code=404, detail=f"Processor {processor_id} not found")
        
        processor_info = self.processors[processor_id]
        processor_info['status'] = 'running'
        processor_info['started_at'] = datetime.now().isoformat()
        processor_info['stopped_at'] = None
        
        # Start the actual processing task
        asyncio.create_task(self._process_stream(processor_id))
        
        self.service_metrics['active_processors'] += 1
        
        return {
            'processor_id': processor_id,
            'status': 'started',
            'message': 'Processor started successfully',
            'timestamp': datetime.now().isoformat()
        }
    
    async def _stop_processor(self, processor_id: str) -> Dict[str, Any]:
        """Stop a processor."""
        if processor_id not in self.processors:
            raise HTTPException(status_code=404, detail=f"Processor {processor_id} not found")
        
        processor_info = self.processors[processor_id]
        processor_info['status'] = 'stopped'
        processor_info['stopped_at'] = datetime.now().isoformat()
        
        self.service_metrics['active_processors'] -= 1
        
        return {
            'processor_id': processor_id,
            'status': 'stopped',
            'message': 'Processor stopped successfully',
            'timestamp': datetime.now().isoformat()
        }
    
    async def _delete_processor(self, processor_id: str) -> Dict[str, Any]:
        """Delete a processor."""
        if processor_id not in self.processors:
            raise HTTPException(status_code=404, detail=f"Processor {processor_id} not found")
        
        processor_info = self.processors[processor_id]
        
        # Stop processor if running
        if processor_info['status'] == 'running':
            await self._stop_processor(processor_id)
        
        # Remove processor from memory
        del self.processors[processor_id]
        if processor_id in self.processor_metrics:
            del self.processor_metrics[processor_id]
        
        self.service_metrics['total_processors'] -= 1
        self.service_metrics['active_processors'] -= 1
        
        return {
            'processor_id': processor_id,
            'status': 'deleted',
            'message': 'Processor deleted successfully',
            'timestamp': datetime.now().isoformat()
        }
    
    async def _process_stream(self, processor_id: str):
        """Process stream data for a processor."""
        if processor_id not in self.processors:
            return
        
        processor_info = self.processors[processor_id]
        kafka_config = processor_info['kafka_config']
        
        try:
            # Create Kafka consumer
            consumer = KafkaConsumer(
                kafka_config['input_topic'],
                bootstrap_servers=kafka_config['bootstrap_servers'],
                group_id=kafka_config['group_id'],
                auto_offset_reset=kafka_config['auto_offset_reset'],
                enable_auto_commit=kafka_config['enable_auto_commit'],
                max_poll_records=kafka_config['max_poll_records'],
                session_timeout_ms=kafka_config['session_timeout_ms'],
                heartbeat_interval_ms=kafka_config['heartbeat_interval_ms'],
                value_deserializer=lambda x: json.loads(x.decode('utf-8'))
            )
            
            # Create Kafka producer
            producer = KafkaProducer(
                bootstrap_servers=kafka_config['bootstrap_servers'],
                value_serializer=lambda x: json.dumps(x).encode('utf-8')
            )
            
            logger.info(f"Started processing stream for processor {processor_id}")
            
            # Process messages
            for message in consumer:
                if processor_info['status'] != 'running':
                    break
                
                try:
                    # Process the message
                    start_time = time.time()
                    result = await self._process_message(message.value, processor_info)
                    processing_time = (time.time() - start_time) * 1000  # Convert to ms
                    
                    # Send result to output topic
                    producer.send(kafka_config['output_topic'], result)
                    producer.flush()
                    
                    # Update metrics
                    processor_info['total_messages_processed'] += 1
                    processor_info['last_message_time'] = datetime.now().isoformat()
                    
                    # Update service metrics
                    self.service_metrics['total_messages_processed'] += 1
                    self.service_metrics['last_processing_time'] = processor_info['last_message_time']
                    
                    # Update processor metrics
                    metrics = self.processor_metrics[processor_id]
                    metrics['total_messages_processed'] = processor_info['total_messages_processed']
                    metrics['average_processing_time'] = (
                        (metrics['average_processing_time'] * (metrics['total_messages_processed'] - 1) + processing_time) /
                        metrics['total_messages_processed']
                    )
                    metrics['last_message_time'] = processor_info['last_message_time']
                    
                    # Calculate messages per second
                    if processor_info['started_at']:
                        start_time = datetime.fromisoformat(processor_info['started_at'])
                        uptime_seconds = (datetime.now() - start_time).total_seconds()
                        if uptime_seconds > 0:
                            metrics['messages_per_second'] = metrics['total_messages_processed'] / uptime_seconds
                    
                except Exception as e:
                    logger.error(f"Error processing message in processor {processor_id}: {e}")
                    processor_info['error_count'] += 1
                    self.service_metrics['total_errors'] += 1
                    self.processor_metrics[processor_id]['error_count'] = processor_info['error_count']
            
            # Close connections
            consumer.close()
            producer.close()
            
        except Exception as e:
            logger.error(f"Error in stream processing for processor {processor_id}: {e}")
            processor_info['status'] = 'error'
            processor_info['error_count'] += 1
            self.service_metrics['total_errors'] += 1
    
    async def _process_message(self, message: Dict[str, Any], processor_info: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single message."""
        # This would implement actual message processing logic
        # For now, return a mock result
        return {
            'message_id': message.get('id', 'unknown'),
            'processed_at': datetime.now().isoformat(),
            'processor_id': processor_info['processor_id'],
            'result': 'processed',
            'data': message
        }
    
    def run(self, host: str = "0.0.0.0", port: int = 8009):
        """Run the service."""
        logger.info(f"Starting Kafka ML Processor on {host}:{port}")
        uvicorn.run(self.app, host=host, port=port)


# Global service instance
service = KafkaMLProcessor()


def create_app() -> FastAPI:
    """Create FastAPI application."""
    return service.app


if __name__ == "__main__":
    service.run()
