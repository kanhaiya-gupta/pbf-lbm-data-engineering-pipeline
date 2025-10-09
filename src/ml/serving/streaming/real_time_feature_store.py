"""
Real-Time Feature Store

This module implements the real-time feature store for PBF-LB/M processes.
It provides real-time feature computation, feature serving,
and feature store management.
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
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import mlflow
import mlflow.tensorflow
from pathlib import Path
from sqlalchemy import create_engine, text
import redis
from collections import defaultdict, deque

from ...features.process_features.laser_parameter_features import LaserParameterFeatures
from ...features.sensor_features.pyrometer_features import PyrometerFeatures
from ...features.temporal_features.time_series_features import TimeSeriesFeatures
from ...config.config_manager import ConfigManager

logger = logging.getLogger(__name__)


# Pydantic models for API requests and responses
class FeatureStoreConfig(BaseModel):
    """Configuration for real-time feature store."""
    redis_host: str = Field("localhost", description="Redis host")
    redis_port: int = Field(6379, description="Redis port")
    redis_db: int = Field(0, description="Redis database")
    redis_password: Optional[str] = Field(None, description="Redis password")
    feature_ttl: int = Field(3600, description="Feature TTL in seconds")
    max_features_per_key: int = Field(1000, description="Max features per key")
    batch_size: int = Field(100, description="Batch size for feature computation")
    update_interval: int = Field(60, description="Update interval in seconds")


class FeatureRequest(BaseModel):
    """Request model for feature retrieval."""
    feature_names: List[str] = Field(..., description="List of feature names")
    entity_ids: List[str] = Field(..., description="List of entity IDs")
    timestamp: Optional[str] = Field(None, description="Timestamp for feature retrieval")
    include_metadata: bool = Field(False, description="Include feature metadata")


class FeatureResponse(BaseModel):
    """Response model for feature retrieval."""
    features: Dict[str, Dict[str, Any]] = Field(..., description="Features by entity ID")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Feature metadata")
    timestamp: str = Field(..., description="Response timestamp")
    total_features: int = Field(..., description="Total number of features")


class FeatureComputationRequest(BaseModel):
    """Request model for feature computation."""
    computation_id: str = Field(..., description="Computation ID")
    feature_definitions: List[Dict[str, Any]] = Field(..., description="Feature definitions")
    data_sources: List[str] = Field(..., description="Data sources")
    computation_config: Dict[str, Any] = Field(..., description="Computation configuration")
    enabled: bool = Field(True, description="Whether computation is enabled")


class FeatureComputationResponse(BaseModel):
    """Response model for feature computation."""
    computation_id: str = Field(..., description="Computation ID")
    status: str = Field(..., description="Computation status")
    features_computed: List[str] = Field(..., description="List of computed features")
    created_at: str = Field(..., description="Computation creation timestamp")
    message: str = Field(..., description="Response message")


class FeatureMetrics(BaseModel):
    """Model for feature store metrics."""
    total_features: int = Field(..., description="Total number of features")
    active_computations: int = Field(..., description="Number of active computations")
    features_per_second: float = Field(..., description="Features computed per second")
    cache_hit_rate: float = Field(..., description="Cache hit rate")
    average_computation_time: float = Field(..., description="Average computation time in ms")
    last_update_time: Optional[str] = Field(None, description="Last update time")


class RealTimeFeatureStore:
    """
    Real-time feature store for PBF-LB/M processes.
    
    This feature store provides real-time capabilities for:
    - Feature computation
    - Feature serving
    - Feature caching
    - Feature monitoring
    """
    
    def __init__(self, config_manager: Optional[ConfigManager] = None):
        """
        Initialize the real-time feature store.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config_manager = config_manager or ConfigManager()
        self.app = FastAPI(
            title="Real-Time Feature Store",
            description="Real-time feature store for PBF-LB/M manufacturing",
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
        
        # Initialize feature extractors
        self.laser_features = LaserParameterFeatures(self.config_manager)
        self.pyrometer_features = PyrometerFeatures(self.config_manager)
        self.time_series_features = TimeSeriesFeatures(self.config_manager)
        
        # Feature store configuration
        self.config = FeatureStoreConfig()
        
        # Initialize Redis connection
        try:
            self.redis_client = redis.Redis(
                host=self.config.redis_host,
                port=self.config.redis_port,
                db=self.config.redis_db,
                password=self.config.redis_password,
                decode_responses=True
            )
            self.redis_client.ping()
            logger.info("Connected to Redis")
        except Exception as e:
            logger.warning(f"Could not connect to Redis: {e}")
            self.redis_client = None
        
        # Feature management
        self.feature_computations = {}  # Store feature computation info
        self.feature_cache = defaultdict(dict)  # In-memory feature cache
        self.computation_counter = 0
        
        # Service metrics
        self.service_metrics = {
            'total_features': 0,
            'active_computations': 0,
            'features_per_second': 0.0,
            'cache_hit_rate': 0.0,
            'average_computation_time': 0.0,
            'last_update_time': None,
            'total_computations': 0,
            'total_errors': 0
        }
        
        # Setup routes
        self._setup_routes()
        
        # Start background tasks
        asyncio.create_task(self._background_feature_update())
        
        logger.info("Initialized RealTimeFeatureStore")
    
    def _setup_routes(self):
        """Setup API routes."""
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {
                "status": "healthy",
                "service": "real_time_feature_store",
                "timestamp": datetime.now().isoformat(),
                "metrics": self.service_metrics
            }
        
        @self.app.post("/features", response_model=FeatureResponse)
        async def get_features(request: FeatureRequest):
            """Get features for entities."""
            return await self._get_features(request)
        
        @self.app.post("/computations", response_model=FeatureComputationResponse)
        async def create_computation(request: FeatureComputationRequest):
            """Create a new feature computation."""
            return await self._create_computation(request)
        
        @self.app.get("/computations")
        async def list_computations():
            """List all feature computations."""
            return await self._list_computations()
        
        @self.app.get("/computations/{computation_id}/status")
        async def get_computation_status(computation_id: str):
            """Get computation status."""
            return await self._get_computation_status(computation_id)
        
        @self.app.post("/computations/{computation_id}/start")
        async def start_computation(computation_id: str):
            """Start a computation."""
            return await self._start_computation(computation_id)
        
        @self.app.post("/computations/{computation_id}/stop")
        async def stop_computation(computation_id: str):
            """Stop a computation."""
            return await self._stop_computation(computation_id)
        
        @self.app.delete("/computations/{computation_id}")
        async def delete_computation(computation_id: str):
            """Delete a computation."""
            return await self._delete_computation(computation_id)
        
        @self.app.get("/features/{feature_name}/stats")
        async def get_feature_stats(feature_name: str):
            """Get feature statistics."""
            return await self._get_feature_stats(feature_name)
        
        @self.app.get("/metrics", response_model=FeatureMetrics)
        async def get_metrics():
            """Get feature store metrics."""
            return await self._get_metrics()
    
    async def _get_features(self, request: FeatureRequest) -> FeatureResponse:
        """
        Get features for entities.
        
        Args:
            request: Feature request
            
        Returns:
            Feature response
        """
        features = {}
        metadata = {}
        
        for entity_id in request.entity_ids:
            entity_features = {}
            
            for feature_name in request.feature_names:
                try:
                    # Try to get feature from cache first
                    feature_value = await self._get_feature_from_cache(entity_id, feature_name)
                    
                    if feature_value is None:
                        # Compute feature if not in cache
                        feature_value = await self._compute_feature(entity_id, feature_name)
                        
                        # Cache the feature
                        await self._cache_feature(entity_id, feature_name, feature_value)
                    
                    entity_features[feature_name] = feature_value
                    
                except Exception as e:
                    logger.error(f"Error getting feature {feature_name} for entity {entity_id}: {e}")
                    entity_features[feature_name] = None
            
            features[entity_id] = entity_features
        
        # Add metadata if requested
        if request.include_metadata:
            metadata = {
                'feature_names': request.feature_names,
                'entity_ids': request.entity_ids,
                'computation_time': datetime.now().isoformat(),
                'cache_hit_rate': await self._calculate_cache_hit_rate()
            }
        
        return FeatureResponse(
            features=features,
            metadata=metadata,
            timestamp=datetime.now().isoformat(),
            total_features=len(request.feature_names) * len(request.entity_ids)
        )
    
    async def _get_feature_from_cache(self, entity_id: str, feature_name: str) -> Optional[Any]:
        """Get feature from cache."""
        # Try Redis first
        if self.redis_client:
            try:
                cache_key = f"feature:{entity_id}:{feature_name}"
                cached_value = self.redis_client.get(cache_key)
                if cached_value:
                    return json.loads(cached_value)
            except Exception as e:
                logger.warning(f"Redis cache error: {e}")
        
        # Try in-memory cache
        if entity_id in self.feature_cache and feature_name in self.feature_cache[entity_id]:
            return self.feature_cache[entity_id][feature_name]
        
        return None
    
    async def _cache_feature(self, entity_id: str, feature_name: str, feature_value: Any):
        """Cache a feature."""
        # Cache in Redis
        if self.redis_client:
            try:
                cache_key = f"feature:{entity_id}:{feature_name}"
                self.redis_client.setex(
                    cache_key,
                    self.config.feature_ttl,
                    json.dumps(feature_value, default=str)
                )
            except Exception as e:
                logger.warning(f"Redis cache error: {e}")
        
        # Cache in memory
        self.feature_cache[entity_id][feature_name] = feature_value
        
        # Limit cache size
        if len(self.feature_cache[entity_id]) > self.config.max_features_per_key:
            # Remove oldest features
            oldest_features = list(self.feature_cache[entity_id].keys())[:len(self.feature_cache[entity_id]) - self.config.max_features_per_key]
            for feature in oldest_features:
                del self.feature_cache[entity_id][feature]
    
    async def _compute_feature(self, entity_id: str, feature_name: str) -> Any:
        """Compute a feature for an entity."""
        # This would implement actual feature computation logic
        # For now, return mock data based on feature name
        
        if 'laser' in feature_name.lower():
            return await self._compute_laser_feature(entity_id, feature_name)
        elif 'pyrometer' in feature_name.lower():
            return await self._compute_pyrometer_feature(entity_id, feature_name)
        elif 'time_series' in feature_name.lower():
            return await self._compute_time_series_feature(entity_id, feature_name)
        else:
            # Default feature computation
            return {
                'value': np.random.normal(0, 1),
                'timestamp': datetime.now().isoformat(),
                'entity_id': entity_id,
                'feature_name': feature_name
            }
    
    async def _compute_laser_feature(self, entity_id: str, feature_name: str) -> Dict[str, Any]:
        """Compute laser-related features."""
        # This would implement actual laser feature computation
        # For now, return mock data
        return {
            'laser_power': np.random.normal(200, 20),
            'scan_speed': np.random.normal(1000, 100),
            'spot_size': np.random.normal(0.1, 0.01),
            'timestamp': datetime.now().isoformat(),
            'entity_id': entity_id,
            'feature_name': feature_name
        }
    
    async def _compute_pyrometer_feature(self, entity_id: str, feature_name: str) -> Dict[str, Any]:
        """Compute pyrometer-related features."""
        # This would implement actual pyrometer feature computation
        # For now, return mock data
        return {
            'temperature': np.random.normal(1000, 50),
            'emissivity': np.random.normal(0.8, 0.1),
            'timestamp': datetime.now().isoformat(),
            'entity_id': entity_id,
            'feature_name': feature_name
        }
    
    async def _compute_time_series_feature(self, entity_id: str, feature_name: str) -> Dict[str, Any]:
        """Compute time series features."""
        # This would implement actual time series feature computation
        # For now, return mock data
        return {
            'trend': np.random.normal(0, 0.1),
            'seasonality': np.random.normal(0, 0.05),
            'volatility': np.random.normal(0.1, 0.02),
            'timestamp': datetime.now().isoformat(),
            'entity_id': entity_id,
            'feature_name': feature_name
        }
    
    async def _create_computation(self, request: FeatureComputationRequest) -> FeatureComputationResponse:
        """
        Create a new feature computation.
        
        Args:
            request: Feature computation request
            
        Returns:
            Feature computation response
        """
        # Generate computation ID
        self.computation_counter += 1
        computation_id = f"computation_{self.computation_counter}_{int(time.time())}"
        
        # Create computation entry
        computation_info = {
            'computation_id': computation_id,
            'feature_definitions': request.feature_definitions,
            'data_sources': request.data_sources,
            'computation_config': request.computation_config,
            'enabled': request.enabled,
            'status': 'created',
            'created_at': datetime.now().isoformat(),
            'started_at': None,
            'stopped_at': None,
            'features_computed': [],
            'total_features_computed': 0,
            'error_count': 0
        }
        
        self.feature_computations[computation_id] = computation_info
        
        # Extract feature names
        features_computed = [fd['name'] for fd in request.feature_definitions]
        
        # Update metrics
        self.service_metrics['total_computations'] += 1
        if request.enabled:
            self.service_metrics['active_computations'] += 1
        
        return FeatureComputationResponse(
            computation_id=computation_id,
            status='created',
            features_computed=features_computed,
            created_at=computation_info['created_at'],
            message="Feature computation created successfully"
        )
    
    async def _list_computations(self) -> Dict[str, Any]:
        """List all feature computations."""
        computation_list = []
        for computation_id, computation_info in self.feature_computations.items():
            computation_list.append({
                'computation_id': computation_id,
                'status': computation_info['status'],
                'enabled': computation_info['enabled'],
                'features_computed': computation_info['features_computed'],
                'total_features_computed': computation_info['total_features_computed'],
                'error_count': computation_info['error_count'],
                'created_at': computation_info['created_at']
            })
        
        return {
            'computations': computation_list,
            'total_computations': len(computation_list),
            'active_computations': sum(1 for c in computation_list if c['enabled']),
            'timestamp': datetime.now().isoformat()
        }
    
    async def _get_computation_status(self, computation_id: str) -> Dict[str, Any]:
        """Get computation status."""
        if computation_id not in self.feature_computations:
            raise HTTPException(status_code=404, detail=f"Computation {computation_id} not found")
        
        computation_info = self.feature_computations[computation_id]
        
        return {
            'computation_id': computation_id,
            'status': computation_info['status'],
            'enabled': computation_info['enabled'],
            'features_computed': computation_info['features_computed'],
            'total_features_computed': computation_info['total_features_computed'],
            'error_count': computation_info['error_count'],
            'created_at': computation_info['created_at'],
            'started_at': computation_info['started_at'],
            'stopped_at': computation_info['stopped_at']
        }
    
    async def _start_computation(self, computation_id: str) -> Dict[str, Any]:
        """Start a computation."""
        if computation_id not in self.feature_computations:
            raise HTTPException(status_code=404, detail=f"Computation {computation_id} not found")
        
        computation_info = self.feature_computations[computation_id]
        computation_info['status'] = 'running'
        computation_info['started_at'] = datetime.now().isoformat()
        computation_info['stopped_at'] = None
        
        # Start the actual computation task
        asyncio.create_task(self._run_computation(computation_id))
        
        self.service_metrics['active_computations'] += 1
        
        return {
            'computation_id': computation_id,
            'status': 'started',
            'message': 'Feature computation started successfully',
            'timestamp': datetime.now().isoformat()
        }
    
    async def _stop_computation(self, computation_id: str) -> Dict[str, Any]:
        """Stop a computation."""
        if computation_id not in self.feature_computations:
            raise HTTPException(status_code=404, detail=f"Computation {computation_id} not found")
        
        computation_info = self.feature_computations[computation_id]
        computation_info['status'] = 'stopped'
        computation_info['stopped_at'] = datetime.now().isoformat()
        
        self.service_metrics['active_computations'] -= 1
        
        return {
            'computation_id': computation_id,
            'status': 'stopped',
            'message': 'Feature computation stopped successfully',
            'timestamp': datetime.now().isoformat()
        }
    
    async def _delete_computation(self, computation_id: str) -> Dict[str, Any]:
        """Delete a computation."""
        if computation_id not in self.feature_computations:
            raise HTTPException(status_code=404, detail=f"Computation {computation_id} not found")
        
        computation_info = self.feature_computations[computation_id]
        
        # Stop computation if running
        if computation_info['status'] == 'running':
            await self._stop_computation(computation_id)
        
        # Remove computation from memory
        del self.feature_computations[computation_id]
        
        self.service_metrics['total_computations'] -= 1
        self.service_metrics['active_computations'] -= 1
        
        return {
            'computation_id': computation_id,
            'status': 'deleted',
            'message': 'Feature computation deleted successfully',
            'timestamp': datetime.now().isoformat()
        }
    
    async def _run_computation(self, computation_id: str):
        """Run a feature computation."""
        if computation_id not in self.feature_computations:
            return
        
        computation_info = self.feature_computations[computation_id]
        
        try:
            while computation_info['status'] == 'running':
                # Compute features
                for feature_def in computation_info['feature_definitions']:
                    try:
                        # This would implement actual feature computation
                        # For now, just simulate
                        await asyncio.sleep(0.1)
                        
                        computation_info['features_computed'].append(feature_def['name'])
                        computation_info['total_features_computed'] += 1
                        
                    except Exception as e:
                        logger.error(f"Error computing feature {feature_def['name']}: {e}")
                        computation_info['error_count'] += 1
                        self.service_metrics['total_errors'] += 1
                
                # Wait before next iteration
                await asyncio.sleep(computation_info['computation_config'].get('interval', 60))
                
        except Exception as e:
            logger.error(f"Error in computation {computation_id}: {e}")
            computation_info['status'] = 'error'
            computation_info['error_count'] += 1
            self.service_metrics['total_errors'] += 1
    
    async def _get_feature_stats(self, feature_name: str) -> Dict[str, Any]:
        """Get feature statistics."""
        # This would implement actual feature statistics
        # For now, return mock data
        return {
            'feature_name': feature_name,
            'total_entities': 1000,
            'last_updated': datetime.now().isoformat(),
            'average_value': np.random.normal(0, 1),
            'std_value': np.random.normal(0, 0.1),
            'min_value': np.random.normal(-2, 0.5),
            'max_value': np.random.normal(2, 0.5)
        }
    
    async def _get_metrics(self) -> FeatureMetrics:
        """Get feature store metrics."""
        return FeatureMetrics(
            total_features=self.service_metrics['total_features'],
            active_computations=self.service_metrics['active_computations'],
            features_per_second=self.service_metrics['features_per_second'],
            cache_hit_rate=self.service_metrics['cache_hit_rate'],
            average_computation_time=self.service_metrics['average_computation_time'],
            last_update_time=self.service_metrics['last_update_time']
        )
    
    async def _calculate_cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        # This would implement actual cache hit rate calculation
        # For now, return a mock value
        return 0.85
    
    async def _background_feature_update(self):
        """Background task for feature updates."""
        while True:
            try:
                # Update features for active computations
                for computation_id, computation_info in self.feature_computations.items():
                    if computation_info['status'] == 'running':
                        # This would implement actual feature updates
                        # For now, just update metrics
                        self.service_metrics['last_update_time'] = datetime.now().isoformat()
                
                # Wait before next update
                await asyncio.sleep(self.config.update_interval)
                
            except Exception as e:
                logger.error(f"Error in background feature update: {e}")
                await asyncio.sleep(60)  # Wait longer on error
    
    def run(self, host: str = "0.0.0.0", port: int = 8011):
        """Run the service."""
        logger.info(f"Starting Real-Time Feature Store on {host}:{port}")
        uvicorn.run(self.app, host=host, port=port)


# Global service instance
service = RealTimeFeatureStore()


def create_app() -> FastAPI:
    """Create FastAPI application."""
    return service.app


if __name__ == "__main__":
    service.run()
