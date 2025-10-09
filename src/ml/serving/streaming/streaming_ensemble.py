"""
Streaming Ensemble

This module implements the streaming ensemble for PBF-LB/M processes.
It provides ensemble methods for streaming data, model combination,
and real-time ensemble predictions.
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
from collections import defaultdict, deque
import threading
from concurrent.futures import ThreadPoolExecutor

from ...pipelines.inference.streaming_inference import StreamingInferencePipeline
from ...models.base_model import BaseModel
from ...config.config_manager import ConfigManager

logger = logging.getLogger(__name__)


# Pydantic models for API requests and responses
class EnsembleConfig(BaseModel):
    """Configuration for streaming ensemble."""
    ensemble_name: str = Field(..., description="Ensemble name")
    ensemble_type: str = Field(..., description="Ensemble type (voting, stacking, bagging)")
    models: List[Dict[str, Any]] = Field(..., description="List of models in ensemble")
    weights: Optional[List[float]] = Field(None, description="Model weights")
    voting_method: str = Field("soft", description="Voting method (soft, hard)")
    meta_model: Optional[Dict[str, Any]] = Field(None, description="Meta-model for stacking")
    update_frequency: int = Field(60, description="Model update frequency in seconds")
    performance_threshold: float = Field(0.7, description="Performance threshold for model inclusion")


class EnsembleRequest(BaseModel):
    """Request model for ensemble prediction."""
    ensemble_id: str = Field(..., description="Ensemble ID")
    input_data: Dict[str, Any] = Field(..., description="Input data for prediction")
    prediction_type: str = Field("real_time", description="Prediction type (real_time, batch)")
    include_confidence: bool = Field(True, description="Include confidence scores")
    include_individual_predictions: bool = Field(False, description="Include individual model predictions")


class EnsembleResponse(BaseModel):
    """Response model for ensemble prediction."""
    ensemble_id: str = Field(..., description="Ensemble ID")
    prediction: Any = Field(..., description="Ensemble prediction")
    confidence: Optional[float] = Field(None, description="Prediction confidence")
    individual_predictions: Optional[Dict[str, Any]] = Field(None, description="Individual model predictions")
    model_weights: List[float] = Field(..., description="Model weights used")
    prediction_time: float = Field(..., description="Prediction time in ms")
    timestamp: str = Field(..., description="Prediction timestamp")


class EnsembleMetrics(BaseModel):
    """Model for ensemble metrics."""
    ensemble_id: str = Field(..., description="Ensemble ID")
    total_predictions: int = Field(..., description="Total predictions made")
    predictions_per_second: float = Field(..., description="Predictions per second")
    average_confidence: float = Field(..., description="Average prediction confidence")
    model_performance: Dict[str, float] = Field(..., description="Individual model performance")
    ensemble_accuracy: float = Field(..., description="Ensemble accuracy")
    last_update_time: Optional[str] = Field(None, description="Last model update time")


class StreamingEnsemble:
    """
    Streaming ensemble for PBF-LB/M processes.
    
    This ensemble provides streaming capabilities for:
    - Real-time ensemble predictions
    - Model combination
    - Performance monitoring
    - Dynamic model weighting
    """
    
    def __init__(self, config_manager: Optional[ConfigManager] = None):
        """
        Initialize the streaming ensemble.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config_manager = config_manager or ConfigManager()
        self.app = FastAPI(
            title="Streaming Ensemble",
            description="Streaming ensemble for PBF-LB/M manufacturing",
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
        
        # Ensemble management
        self.ensembles = {}  # Store ensemble information
        self.ensemble_models = {}  # Store loaded models
        self.ensemble_metrics = {}  # Store ensemble metrics
        self.ensemble_counter = 0
        
        # Service metrics
        self.service_metrics = {
            'total_ensembles': 0,
            'active_ensembles': 0,
            'total_predictions': 0,
            'total_errors': 0,
            'last_prediction_time': None
        }
        
        # Thread pool for parallel model predictions
        self.thread_pool = ThreadPoolExecutor(max_workers=10)
        
        # Setup routes
        self._setup_routes()
        
        logger.info("Initialized StreamingEnsemble")
    
    def _setup_routes(self):
        """Setup API routes."""
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {
                "status": "healthy",
                "service": "streaming_ensemble",
                "timestamp": datetime.now().isoformat(),
                "metrics": self.service_metrics
            }
        
        @self.app.post("/ensembles", response_model=Dict[str, Any])
        async def create_ensemble(config: EnsembleConfig):
            """Create a new ensemble."""
            return await self._create_ensemble(config)
        
        @self.app.get("/ensembles")
        async def list_ensembles():
            """List all ensembles."""
            return await self._list_ensembles()
        
        @self.app.get("/ensembles/{ensemble_id}/status")
        async def get_ensemble_status(ensemble_id: str):
            """Get ensemble status."""
            return await self._get_ensemble_status(ensemble_id)
        
        @self.app.get("/ensembles/{ensemble_id}/metrics", response_model=EnsembleMetrics)
        async def get_ensemble_metrics(ensemble_id: str):
            """Get ensemble metrics."""
            return await self._get_ensemble_metrics(ensemble_id)
        
        @self.app.post("/ensembles/{ensemble_id}/predict", response_model=EnsembleResponse)
        async def predict(request: EnsembleRequest):
            """Make ensemble prediction."""
            return await self._predict(request)
        
        @self.app.post("/ensembles/{ensemble_id}/update")
        async def update_ensemble(ensemble_id: str):
            """Update ensemble models."""
            return await self._update_ensemble(ensemble_id)
        
        @self.app.post("/ensembles/{ensemble_id}/start")
        async def start_ensemble(ensemble_id: str):
            """Start ensemble processing."""
            return await self._start_ensemble(ensemble_id)
        
        @self.app.post("/ensembles/{ensemble_id}/stop")
        async def stop_ensemble(ensemble_id: str):
            """Stop ensemble processing."""
            return await self._stop_ensemble(ensemble_id)
        
        @self.app.delete("/ensembles/{ensemble_id}")
        async def delete_ensemble(ensemble_id: str):
            """Delete an ensemble."""
            return await self._delete_ensemble(ensemble_id)
        
        @self.app.get("/metrics")
        async def get_metrics():
            """Get service metrics."""
            return self.service_metrics
    
    async def _create_ensemble(self, config: EnsembleConfig) -> Dict[str, Any]:
        """
        Create a new ensemble.
        
        Args:
            config: Ensemble configuration
            
        Returns:
            Ensemble creation response
        """
        # Generate ensemble ID
        self.ensemble_counter += 1
        ensemble_id = f"ensemble_{self.ensemble_counter}_{int(time.time())}"
        
        # Create ensemble entry
        ensemble_info = {
            'ensemble_id': ensemble_id,
            'ensemble_name': config.ensemble_name,
            'ensemble_type': config.ensemble_type,
            'models': config.models,
            'weights': config.weights or [1.0 / len(config.models)] * len(config.models),
            'voting_method': config.voting_method,
            'meta_model': config.meta_model,
            'update_frequency': config.update_frequency,
            'performance_threshold': config.performance_threshold,
            'status': 'created',
            'created_at': datetime.now().isoformat(),
            'started_at': None,
            'stopped_at': None,
            'total_predictions': 0,
            'error_count': 0,
            'last_prediction_time': None
        }
        
        self.ensembles[ensemble_id] = ensemble_info
        
        # Load models
        loaded_models = {}
        for i, model_config in enumerate(config.models):
            try:
                model = await self._load_model(model_config)
                loaded_models[model_config['model_name']] = {
                    'model': model,
                    'weight': config.weights[i] if config.weights else 1.0 / len(config.models),
                    'performance': 0.0,
                    'last_update': datetime.now().isoformat()
                }
            except Exception as e:
                logger.error(f"Failed to load model {model_config['model_name']}: {e}")
        
        self.ensemble_models[ensemble_id] = loaded_models
        
        # Initialize metrics
        self.ensemble_metrics[ensemble_id] = {
            'total_predictions': 0,
            'predictions_per_second': 0.0,
            'average_confidence': 0.0,
            'model_performance': {name: 0.0 for name in loaded_models.keys()},
            'ensemble_accuracy': 0.0,
            'last_update_time': None,
            'start_time': None
        }
        
        # Update metrics
        self.service_metrics['total_ensembles'] += 1
        self.service_metrics['active_ensembles'] += 1
        
        return {
            'ensemble_id': ensemble_id,
            'ensemble_name': config.ensemble_name,
            'status': 'created',
            'models_loaded': list(loaded_models.keys()),
            'created_at': ensemble_info['created_at'],
            'message': 'Ensemble created successfully'
        }
    
    async def _load_model(self, model_config: Dict[str, Any]) -> Any:
        """Load a model for the ensemble."""
        # This would implement actual model loading logic
        # For now, return a mock model
        return f"model_{model_config['model_name']}"
    
    async def _list_ensembles(self) -> Dict[str, Any]:
        """List all ensembles."""
        ensemble_list = []
        for ensemble_id, ensemble_info in self.ensembles.items():
            ensemble_list.append({
                'ensemble_id': ensemble_id,
                'ensemble_name': ensemble_info['ensemble_name'],
                'ensemble_type': ensemble_info['ensemble_type'],
                'status': ensemble_info['status'],
                'models_count': len(ensemble_info['models']),
                'total_predictions': ensemble_info['total_predictions'],
                'error_count': ensemble_info['error_count'],
                'created_at': ensemble_info['created_at']
            })
        
        return {
            'ensembles': ensemble_list,
            'total_ensembles': len(ensemble_list),
            'active_ensembles': sum(1 for e in ensemble_list if e['status'] == 'running'),
            'timestamp': datetime.now().isoformat()
        }
    
    async def _get_ensemble_status(self, ensemble_id: str) -> Dict[str, Any]:
        """Get ensemble status."""
        if ensemble_id not in self.ensembles:
            raise HTTPException(status_code=404, detail=f"Ensemble {ensemble_id} not found")
        
        ensemble_info = self.ensembles[ensemble_id]
        
        return {
            'ensemble_id': ensemble_id,
            'ensemble_name': ensemble_info['ensemble_name'],
            'ensemble_type': ensemble_info['ensemble_type'],
            'status': ensemble_info['status'],
            'models_count': len(ensemble_info['models']),
            'total_predictions': ensemble_info['total_predictions'],
            'error_count': ensemble_info['error_count'],
            'last_prediction_time': ensemble_info['last_prediction_time'],
            'created_at': ensemble_info['created_at'],
            'started_at': ensemble_info['started_at'],
            'stopped_at': ensemble_info['stopped_at']
        }
    
    async def _get_ensemble_metrics(self, ensemble_id: str) -> EnsembleMetrics:
        """Get ensemble metrics."""
        if ensemble_id not in self.ensembles:
            raise HTTPException(status_code=404, detail=f"Ensemble {ensemble_id} not found")
        
        ensemble_info = self.ensembles[ensemble_id]
        metrics = self.ensemble_metrics[ensemble_id]
        
        return EnsembleMetrics(
            ensemble_id=ensemble_id,
            total_predictions=ensemble_info['total_predictions'],
            predictions_per_second=metrics['predictions_per_second'],
            average_confidence=metrics['average_confidence'],
            model_performance=metrics['model_performance'],
            ensemble_accuracy=metrics['ensemble_accuracy'],
            last_update_time=metrics['last_update_time']
        )
    
    async def _predict(self, request: EnsembleRequest) -> EnsembleResponse:
        """
        Make ensemble prediction.
        
        Args:
            request: Ensemble prediction request
            
        Returns:
            Ensemble prediction response
        """
        if request.ensemble_id not in self.ensembles:
            raise HTTPException(status_code=404, detail=f"Ensemble {request.ensemble_id} not found")
        
        ensemble_info = self.ensembles[request.ensemble_id]
        ensemble_models = self.ensemble_models[request.ensemble_id]
        
        start_time = time.time()
        
        try:
            # Get individual model predictions
            individual_predictions = {}
            if request.include_individual_predictions:
                individual_predictions = await self._get_individual_predictions(
                    ensemble_models, request.input_data
                )
            
            # Make ensemble prediction
            ensemble_prediction = await self._make_ensemble_prediction(
                ensemble_info, ensemble_models, request.input_data
            )
            
            # Calculate confidence
            confidence = None
            if request.include_confidence:
                confidence = await self._calculate_confidence(
                    ensemble_info, ensemble_models, request.input_data
                )
            
            prediction_time = (time.time() - start_time) * 1000  # Convert to ms
            
            # Update metrics
            ensemble_info['total_predictions'] += 1
            ensemble_info['last_prediction_time'] = datetime.now().isoformat()
            
            self.service_metrics['total_predictions'] += 1
            self.service_metrics['last_prediction_time'] = ensemble_info['last_prediction_time']
            
            # Update ensemble metrics
            metrics = self.ensemble_metrics[request.ensemble_id]
            metrics['total_predictions'] = ensemble_info['total_predictions']
            if confidence is not None:
                metrics['average_confidence'] = (
                    (metrics['average_confidence'] * (metrics['total_predictions'] - 1) + confidence) /
                    metrics['total_predictions']
                )
            
            return EnsembleResponse(
                ensemble_id=request.ensemble_id,
                prediction=ensemble_prediction,
                confidence=confidence,
                individual_predictions=individual_predictions if request.include_individual_predictions else None,
                model_weights=ensemble_info['weights'],
                prediction_time=prediction_time,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"Error making ensemble prediction: {e}")
            ensemble_info['error_count'] += 1
            self.service_metrics['total_errors'] += 1
            raise HTTPException(status_code=500, detail=str(e))
    
    async def _get_individual_predictions(self, ensemble_models: Dict[str, Any], 
                                        input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get individual model predictions."""
        predictions = {}
        
        for model_name, model_info in ensemble_models.items():
            try:
                # This would implement actual model prediction
                # For now, return mock predictions
                prediction = {
                    'value': np.random.normal(0, 1),
                    'confidence': np.random.uniform(0.5, 1.0),
                    'model_name': model_name,
                    'timestamp': datetime.now().isoformat()
                }
                predictions[model_name] = prediction
            except Exception as e:
                logger.error(f"Error getting prediction from model {model_name}: {e}")
                predictions[model_name] = None
        
        return predictions
    
    async def _make_ensemble_prediction(self, ensemble_info: Dict[str, Any], 
                                      ensemble_models: Dict[str, Any], 
                                      input_data: Dict[str, Any]) -> Any:
        """Make ensemble prediction."""
        ensemble_type = ensemble_info['ensemble_type']
        
        if ensemble_type == 'voting':
            return await self._voting_prediction(ensemble_info, ensemble_models, input_data)
        elif ensemble_type == 'stacking':
            return await self._stacking_prediction(ensemble_info, ensemble_models, input_data)
        elif ensemble_type == 'bagging':
            return await self._bagging_prediction(ensemble_info, ensemble_models, input_data)
        else:
            raise ValueError(f"Unsupported ensemble type: {ensemble_type}")
    
    async def _voting_prediction(self, ensemble_info: Dict[str, Any], 
                               ensemble_models: Dict[str, Any], 
                               input_data: Dict[str, Any]) -> Any:
        """Make voting ensemble prediction."""
        voting_method = ensemble_info['voting_method']
        weights = ensemble_info['weights']
        
        # Get individual predictions
        predictions = []
        confidences = []
        
        for i, (model_name, model_info) in enumerate(ensemble_models.items()):
            try:
                # This would implement actual model prediction
                # For now, return mock predictions
                prediction = np.random.normal(0, 1)
                confidence = np.random.uniform(0.5, 1.0)
                
                predictions.append(prediction)
                confidences.append(confidence)
            except Exception as e:
                logger.error(f"Error getting prediction from model {model_name}: {e}")
                predictions.append(0.0)
                confidences.append(0.0)
        
        if voting_method == 'soft':
            # Weighted average
            weighted_prediction = np.average(predictions, weights=weights)
            return weighted_prediction
        else:
            # Hard voting (majority)
            return np.median(predictions)
    
    async def _stacking_prediction(self, ensemble_info: Dict[str, Any], 
                                 ensemble_models: Dict[str, Any], 
                                 input_data: Dict[str, Any]) -> Any:
        """Make stacking ensemble prediction."""
        # Get base model predictions
        base_predictions = []
        for model_name, model_info in ensemble_models.items():
            try:
                # This would implement actual model prediction
                # For now, return mock predictions
                prediction = np.random.normal(0, 1)
                base_predictions.append(prediction)
            except Exception as e:
                logger.error(f"Error getting prediction from model {model_name}: {e}")
                base_predictions.append(0.0)
        
        # Use meta-model to combine predictions
        # This would implement actual meta-model prediction
        # For now, return weighted average
        weights = ensemble_info['weights']
        meta_prediction = np.average(base_predictions, weights=weights)
        
        return meta_prediction
    
    async def _bagging_prediction(self, ensemble_info: Dict[str, Any], 
                                ensemble_models: Dict[str, Any], 
                                input_data: Dict[str, Any]) -> Any:
        """Make bagging ensemble prediction."""
        # Get individual predictions
        predictions = []
        for model_name, model_info in ensemble_models.items():
            try:
                # This would implement actual model prediction
                # For now, return mock predictions
                prediction = np.random.normal(0, 1)
                predictions.append(prediction)
            except Exception as e:
                logger.error(f"Error getting prediction from model {model_name}: {e}")
                predictions.append(0.0)
        
        # Average predictions
        bagging_prediction = np.mean(predictions)
        
        return bagging_prediction
    
    async def _calculate_confidence(self, ensemble_info: Dict[str, Any], 
                                  ensemble_models: Dict[str, Any], 
                                  input_data: Dict[str, Any]) -> float:
        """Calculate prediction confidence."""
        # This would implement actual confidence calculation
        # For now, return mock confidence
        return np.random.uniform(0.7, 0.95)
    
    async def _update_ensemble(self, ensemble_id: str) -> Dict[str, Any]:
        """Update ensemble models."""
        if ensemble_id not in self.ensembles:
            raise HTTPException(status_code=404, detail=f"Ensemble {ensemble_id} not found")
        
        ensemble_info = self.ensembles[ensemble_id]
        ensemble_models = self.ensemble_models[ensemble_id]
        
        try:
            # Update models
            for model_name, model_info in ensemble_models.items():
                # This would implement actual model updating
                # For now, just update the timestamp
                model_info['last_update'] = datetime.now().isoformat()
            
            # Update ensemble metrics
            metrics = self.ensemble_metrics[ensemble_id]
            metrics['last_update_time'] = datetime.now().isoformat()
            
            return {
                'ensemble_id': ensemble_id,
                'status': 'updated',
                'message': 'Ensemble models updated successfully',
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error updating ensemble {ensemble_id}: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def _start_ensemble(self, ensemble_id: str) -> Dict[str, Any]:
        """Start ensemble processing."""
        if ensemble_id not in self.ensembles:
            raise HTTPException(status_code=404, detail=f"Ensemble {ensemble_id} not found")
        
        ensemble_info = self.ensembles[ensemble_id]
        ensemble_info['status'] = 'running'
        ensemble_info['started_at'] = datetime.now().isoformat()
        ensemble_info['stopped_at'] = None
        
        # Start background update task
        asyncio.create_task(self._background_ensemble_update(ensemble_id))
        
        return {
            'ensemble_id': ensemble_id,
            'status': 'started',
            'message': 'Ensemble processing started successfully',
            'timestamp': datetime.now().isoformat()
        }
    
    async def _stop_ensemble(self, ensemble_id: str) -> Dict[str, Any]:
        """Stop ensemble processing."""
        if ensemble_id not in self.ensembles:
            raise HTTPException(status_code=404, detail=f"Ensemble {ensemble_id} not found")
        
        ensemble_info = self.ensembles[ensemble_id]
        ensemble_info['status'] = 'stopped'
        ensemble_info['stopped_at'] = datetime.now().isoformat()
        
        return {
            'ensemble_id': ensemble_id,
            'status': 'stopped',
            'message': 'Ensemble processing stopped successfully',
            'timestamp': datetime.now().isoformat()
        }
    
    async def _delete_ensemble(self, ensemble_id: str) -> Dict[str, Any]:
        """Delete an ensemble."""
        if ensemble_id not in self.ensembles:
            raise HTTPException(status_code=404, detail=f"Ensemble {ensemble_id} not found")
        
        ensemble_info = self.ensembles[ensemble_id]
        
        # Stop ensemble if running
        if ensemble_info['status'] == 'running':
            await self._stop_ensemble(ensemble_id)
        
        # Remove ensemble from memory
        del self.ensembles[ensemble_id]
        if ensemble_id in self.ensemble_models:
            del self.ensemble_models[ensemble_id]
        if ensemble_id in self.ensemble_metrics:
            del self.ensemble_metrics[ensemble_id]
        
        self.service_metrics['total_ensembles'] -= 1
        self.service_metrics['active_ensembles'] -= 1
        
        return {
            'ensemble_id': ensemble_id,
            'status': 'deleted',
            'message': 'Ensemble deleted successfully',
            'timestamp': datetime.now().isoformat()
        }
    
    async def _background_ensemble_update(self, ensemble_id: str):
        """Background task for ensemble updates."""
        if ensemble_id not in self.ensembles:
            return
        
        ensemble_info = self.ensembles[ensemble_id]
        
        try:
            while ensemble_info['status'] == 'running':
                # Update ensemble
                await self._update_ensemble(ensemble_id)
                
                # Wait before next update
                await asyncio.sleep(ensemble_info['update_frequency'])
                
        except Exception as e:
            logger.error(f"Error in background ensemble update for {ensemble_id}: {e}")
            ensemble_info['status'] = 'error'
            ensemble_info['error_count'] += 1
            self.service_metrics['total_errors'] += 1
    
    def run(self, host: str = "0.0.0.0", port: int = 8012):
        """Run the service."""
        logger.info(f"Starting Streaming Ensemble on {host}:{port}")
        uvicorn.run(self.app, host=host, port=port)


# Global service instance
service = StreamingEnsemble()


def create_app() -> FastAPI:
    """Create FastAPI application."""
    return service.app


if __name__ == "__main__":
    service.run()
