"""
Process Optimization Real-time Service

This module implements the real-time service for process optimization in PBF-LB/M processes.
It provides REST API endpoints for real-time process parameter optimization, laser parameter prediction,
build strategy optimization, and material tuning recommendations.
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

from ...pipelines.inference.real_time_inference import RealTimeInferencePipeline
from ...models.process_optimization import LaserParameterPredictor, BuildStrategyOptimizer, MaterialTuningModels, MultiObjectiveOptimizer
from ...config.config_manager import ConfigManager

logger = logging.getLogger(__name__)


# Pydantic models for API requests and responses
class ProcessOptimizationRequest(BaseModel):
    """Request model for process optimization."""
    laser_power: float = Field(..., description="Current laser power (W)")
    scan_speed: float = Field(..., description="Current scan speed (mm/s)")
    layer_height: float = Field(..., description="Current layer height (mm)")
    hatch_spacing: float = Field(..., description="Current hatch spacing (mm)")
    material_type: str = Field(..., description="Material type")
    build_orientation: str = Field(..., description="Build orientation")
    chamber_temperature: float = Field(..., description="Chamber temperature (°C)")
    target_quality: float = Field(..., description="Target quality score (0-1)")
    target_speed: float = Field(..., description="Target build speed (mm³/s)")
    constraints: Optional[Dict[str, Any]] = Field(None, description="Process constraints")


class ProcessOptimizationResponse(BaseModel):
    """Response model for process optimization."""
    optimized_parameters: Dict[str, float] = Field(..., description="Optimized process parameters")
    predicted_quality: float = Field(..., description="Predicted quality score")
    predicted_speed: float = Field(..., description="Predicted build speed")
    confidence: float = Field(..., description="Prediction confidence")
    recommendations: List[str] = Field(..., description="Optimization recommendations")
    processing_time: float = Field(..., description="Processing time in seconds")
    timestamp: str = Field(..., description="Response timestamp")


class LaserParameterRequest(BaseModel):
    """Request model for laser parameter prediction."""
    material_type: str = Field(..., description="Material type")
    layer_height: float = Field(..., description="Layer height (mm)")
    target_quality: float = Field(..., description="Target quality score (0-1)")
    build_orientation: str = Field(..., description="Build orientation")
    chamber_temperature: float = Field(..., description="Chamber temperature (°C)")


class LaserParameterResponse(BaseModel):
    """Response model for laser parameter prediction."""
    recommended_power: float = Field(..., description="Recommended laser power (W)")
    recommended_speed: float = Field(..., description="Recommended scan speed (mm/s)")
    recommended_hatch_spacing: float = Field(..., description="Recommended hatch spacing (mm)")
    confidence: float = Field(..., description="Prediction confidence")
    processing_time: float = Field(..., description="Processing time in seconds")
    timestamp: str = Field(..., description="Response timestamp")


class BuildStrategyRequest(BaseModel):
    """Request model for build strategy optimization."""
    part_geometry: Dict[str, Any] = Field(..., description="Part geometry information")
    material_type: str = Field(..., description="Material type")
    target_quality: float = Field(..., description="Target quality score (0-1)")
    time_constraint: Optional[float] = Field(None, description="Time constraint (hours)")
    cost_constraint: Optional[float] = Field(None, description="Cost constraint ($)")


class BuildStrategyResponse(BaseModel):
    """Response model for build strategy optimization."""
    optimized_strategy: Dict[str, Any] = Field(..., description="Optimized build strategy")
    predicted_quality: float = Field(..., description="Predicted quality score")
    predicted_time: float = Field(..., description="Predicted build time (hours)")
    predicted_cost: float = Field(..., description="Predicted build cost ($)")
    confidence: float = Field(..., description="Prediction confidence")
    processing_time: float = Field(..., description="Processing time in seconds")
    timestamp: str = Field(..., description="Response timestamp")


class MaterialTuningRequest(BaseModel):
    """Request model for material tuning."""
    material_type: str = Field(..., description="Material type")
    current_parameters: Dict[str, float] = Field(..., description="Current process parameters")
    target_properties: Dict[str, float] = Field(..., description="Target material properties")
    constraints: Optional[Dict[str, Any]] = Field(None, description="Tuning constraints")


class MaterialTuningResponse(BaseModel):
    """Response model for material tuning."""
    tuned_parameters: Dict[str, float] = Field(..., description="Tuned process parameters")
    predicted_properties: Dict[str, float] = Field(..., description="Predicted material properties")
    improvement_score: float = Field(..., description="Improvement score")
    confidence: float = Field(..., description="Prediction confidence")
    processing_time: float = Field(..., description="Processing time in seconds")
    timestamp: str = Field(..., description="Response timestamp")


class ProcessOptimizationService:
    """
    Real-time process optimization service for PBF-LB/M processes.
    
    This service provides real-time optimization for:
    - Process parameter optimization
    - Laser parameter prediction
    - Build strategy optimization
    - Material tuning recommendations
    """
    
    def __init__(self, config_manager: Optional[ConfigManager] = None):
        """
        Initialize the process optimization service.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config_manager = config_manager or ConfigManager()
        self.app = FastAPI(
            title="Process Optimization Service",
            description="Real-time process optimization for PBF-LB/M manufacturing",
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
        
        # Initialize inference pipeline
        self.inference_pipeline = RealTimeInferencePipeline(self.config_manager)
        
        # Initialize models
        self.models = {
            'laser_parameter_predictor': LaserParameterPredictor(self.config_manager),
            'build_strategy_optimizer': BuildStrategyOptimizer(self.config_manager),
            'material_tuning_models': MaterialTuningModels(self.config_manager),
            'multi_objective_optimizer': MultiObjectiveOptimizer(self.config_manager)
        }
        
        # Service metrics
        self.service_metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'average_response_time': 0.0,
            'last_request_time': None
        }
        
        # Setup routes
        self._setup_routes()
        
        logger.info("Initialized ProcessOptimizationService")
    
    def _setup_routes(self):
        """Setup API routes."""
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {
                "status": "healthy",
                "service": "process_optimization",
                "timestamp": datetime.now().isoformat(),
                "metrics": self.service_metrics
            }
        
        @self.app.post("/optimize", response_model=ProcessOptimizationResponse)
        async def optimize_process(request: ProcessOptimizationRequest):
            """Optimize process parameters in real-time."""
            return await self._optimize_process(request)
        
        @self.app.post("/predict-laser-parameters", response_model=LaserParameterResponse)
        async def predict_laser_parameters(request: LaserParameterRequest):
            """Predict optimal laser parameters."""
            return await self._predict_laser_parameters(request)
        
        @self.app.post("/optimize-build-strategy", response_model=BuildStrategyResponse)
        async def optimize_build_strategy(request: BuildStrategyRequest):
            """Optimize build strategy."""
            return await self._optimize_build_strategy(request)
        
        @self.app.post("/tune-material", response_model=MaterialTuningResponse)
        async def tune_material(request: MaterialTuningRequest):
            """Tune material parameters."""
            return await self._tune_material(request)
        
        @self.app.get("/metrics")
        async def get_metrics():
            """Get service metrics."""
            return self.service_metrics
        
        @self.app.get("/models")
        async def get_available_models():
            """Get available models."""
            return {
                "available_models": list(self.models.keys()),
                "model_status": {name: "loaded" for name in self.models.keys()}
            }
    
    async def _optimize_process(self, request: ProcessOptimizationRequest) -> ProcessOptimizationResponse:
        """
        Optimize process parameters in real-time.
        
        Args:
            request: Process optimization request
            
        Returns:
            Process optimization response
        """
        start_time = time.time()
        
        try:
            # Prepare input data
            input_data = {
                'timestamp': datetime.now().isoformat(),
                'sensor_data': {
                    'laser_power': request.laser_power,
                    'scan_speed': request.scan_speed,
                    'layer_height': request.layer_height,
                    'hatch_spacing': request.hatch_spacing,
                    'chamber_temperature': request.chamber_temperature
                },
                'process_data': {
                    'material_type': request.material_type,
                    'build_orientation': request.build_orientation,
                    'target_quality': request.target_quality,
                    'target_speed': request.target_speed,
                    'constraints': request.constraints or {}
                }
            }
            
            # Get model configurations
            model_configs = [
                {
                    'model_name': 'multi_objective_optimizer',
                    'version': 'latest'
                }
            ]
            
            # Generate predictions
            predictions = await self.inference_pipeline.process_real_time_data(input_data, model_configs)
            
            if predictions['status'] == 'error':
                raise HTTPException(status_code=500, detail=predictions.get('error', 'Prediction failed'))
            
            # Extract optimization results
            optimization_results = predictions['predictions'].get('multi_objective_optimizer', {})
            
            # Process optimization results
            optimized_parameters = {
                'laser_power': optimization_results.get('prediction', [request.laser_power])[0] if optimization_results.get('prediction') else request.laser_power,
                'scan_speed': optimization_results.get('prediction', [request.scan_speed])[1] if len(optimization_results.get('prediction', [])) > 1 else request.scan_speed,
                'layer_height': optimization_results.get('prediction', [request.layer_height])[2] if len(optimization_results.get('prediction', [])) > 2 else request.layer_height,
                'hatch_spacing': optimization_results.get('prediction', [request.hatch_spacing])[3] if len(optimization_results.get('prediction', [])) > 3 else request.hatch_spacing
            }
            
            # Generate recommendations
            recommendations = self._generate_optimization_recommendations(request, optimized_parameters)
            
            # Update metrics
            self._update_service_metrics(start_time, True)
            
            return ProcessOptimizationResponse(
                optimized_parameters=optimized_parameters,
                predicted_quality=optimization_results.get('prediction', [0.8])[0] if optimization_results.get('prediction') else 0.8,
                predicted_speed=optimization_results.get('prediction', [1.0])[1] if len(optimization_results.get('prediction', [])) > 1 else 1.0,
                confidence=optimization_results.get('confidence', [0.8])[0] if optimization_results.get('confidence') else 0.8,
                recommendations=recommendations,
                processing_time=time.time() - start_time,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"Process optimization failed: {e}")
            self._update_service_metrics(start_time, False)
            raise HTTPException(status_code=500, detail=str(e))
    
    async def _predict_laser_parameters(self, request: LaserParameterRequest) -> LaserParameterResponse:
        """
        Predict optimal laser parameters.
        
        Args:
            request: Laser parameter request
            
        Returns:
            Laser parameter response
        """
        start_time = time.time()
        
        try:
            # Prepare input data
            input_data = {
                'timestamp': datetime.now().isoformat(),
                'sensor_data': {
                    'chamber_temperature': request.chamber_temperature
                },
                'process_data': {
                    'material_type': request.material_type,
                    'layer_height': request.layer_height,
                    'target_quality': request.target_quality,
                    'build_orientation': request.build_orientation
                }
            }
            
            # Get model configurations
            model_configs = [
                {
                    'model_name': 'laser_parameter_predictor',
                    'version': 'latest'
                }
            ]
            
            # Generate predictions
            predictions = await self.inference_pipeline.process_real_time_data(input_data, model_configs)
            
            if predictions['status'] == 'error':
                raise HTTPException(status_code=500, detail=predictions.get('error', 'Prediction failed'))
            
            # Extract laser parameter results
            laser_results = predictions['predictions'].get('laser_parameter_predictor', {})
            
            # Process laser parameter results
            recommended_power = laser_results.get('prediction', [100.0])[0] if laser_results.get('prediction') else 100.0
            recommended_speed = laser_results.get('prediction', [1000.0])[1] if len(laser_results.get('prediction', [])) > 1 else 1000.0
            recommended_hatch_spacing = laser_results.get('prediction', [0.1])[2] if len(laser_results.get('prediction', [])) > 2 else 0.1
            
            # Update metrics
            self._update_service_metrics(start_time, True)
            
            return LaserParameterResponse(
                recommended_power=recommended_power,
                recommended_speed=recommended_speed,
                recommended_hatch_spacing=recommended_hatch_spacing,
                confidence=laser_results.get('confidence', [0.8])[0] if laser_results.get('confidence') else 0.8,
                processing_time=time.time() - start_time,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"Laser parameter prediction failed: {e}")
            self._update_service_metrics(start_time, False)
            raise HTTPException(status_code=500, detail=str(e))
    
    async def _optimize_build_strategy(self, request: BuildStrategyRequest) -> BuildStrategyResponse:
        """
        Optimize build strategy.
        
        Args:
            request: Build strategy request
            
        Returns:
            Build strategy response
        """
        start_time = time.time()
        
        try:
            # Prepare input data
            input_data = {
                'timestamp': datetime.now().isoformat(),
                'sensor_data': {},
                'process_data': {
                    'part_geometry': request.part_geometry,
                    'material_type': request.material_type,
                    'target_quality': request.target_quality,
                    'time_constraint': request.time_constraint,
                    'cost_constraint': request.cost_constraint
                }
            }
            
            # Get model configurations
            model_configs = [
                {
                    'model_name': 'build_strategy_optimizer',
                    'version': 'latest'
                }
            ]
            
            # Generate predictions
            predictions = await self.inference_pipeline.process_real_time_data(input_data, model_configs)
            
            if predictions['status'] == 'error':
                raise HTTPException(status_code=500, detail=predictions.get('error', 'Prediction failed'))
            
            # Extract build strategy results
            strategy_results = predictions['predictions'].get('build_strategy_optimizer', {})
            
            # Process build strategy results
            optimized_strategy = {
                'build_orientation': strategy_results.get('prediction', ['Z'])[0] if strategy_results.get('prediction') else 'Z',
                'support_strategy': strategy_results.get('prediction', ['minimal'])[1] if len(strategy_results.get('prediction', [])) > 1 else 'minimal',
                'layer_thickness': strategy_results.get('prediction', [0.1])[2] if len(strategy_results.get('prediction', [])) > 2 else 0.1,
                'scan_pattern': strategy_results.get('prediction', ['zigzag'])[3] if len(strategy_results.get('prediction', [])) > 3 else 'zigzag'
            }
            
            # Update metrics
            self._update_service_metrics(start_time, True)
            
            return BuildStrategyResponse(
                optimized_strategy=optimized_strategy,
                predicted_quality=strategy_results.get('prediction', [0.8])[0] if strategy_results.get('prediction') else 0.8,
                predicted_time=strategy_results.get('prediction', [10.0])[1] if len(strategy_results.get('prediction', [])) > 1 else 10.0,
                predicted_cost=strategy_results.get('prediction', [100.0])[2] if len(strategy_results.get('prediction', [])) > 2 else 100.0,
                confidence=strategy_results.get('confidence', [0.8])[0] if strategy_results.get('confidence') else 0.8,
                processing_time=time.time() - start_time,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"Build strategy optimization failed: {e}")
            self._update_service_metrics(start_time, False)
            raise HTTPException(status_code=500, detail=str(e))
    
    async def _tune_material(self, request: MaterialTuningRequest) -> MaterialTuningResponse:
        """
        Tune material parameters.
        
        Args:
            request: Material tuning request
            
        Returns:
            Material tuning response
        """
        start_time = time.time()
        
        try:
            # Prepare input data
            input_data = {
                'timestamp': datetime.now().isoformat(),
                'sensor_data': {},
                'process_data': {
                    'material_type': request.material_type,
                    'current_parameters': request.current_parameters,
                    'target_properties': request.target_properties,
                    'constraints': request.constraints or {}
                }
            }
            
            # Get model configurations
            model_configs = [
                {
                    'model_name': 'material_tuning_models',
                    'version': 'latest'
                }
            ]
            
            # Generate predictions
            predictions = await self.inference_pipeline.process_real_time_data(input_data, model_configs)
            
            if predictions['status'] == 'error':
                raise HTTPException(status_code=500, detail=predictions.get('error', 'Prediction failed'))
            
            # Extract material tuning results
            tuning_results = predictions['predictions'].get('material_tuning_models', {})
            
            # Process material tuning results
            tuned_parameters = {
                'laser_power': tuning_results.get('prediction', [100.0])[0] if tuning_results.get('prediction') else 100.0,
                'scan_speed': tuning_results.get('prediction', [1000.0])[1] if len(tuning_results.get('prediction', [])) > 1 else 1000.0,
                'layer_height': tuning_results.get('prediction', [0.1])[2] if len(tuning_results.get('prediction', [])) > 2 else 0.1,
                'hatch_spacing': tuning_results.get('prediction', [0.1])[3] if len(tuning_results.get('prediction', [])) > 3 else 0.1
            }
            
            predicted_properties = {
                'tensile_strength': tuning_results.get('prediction', [500.0])[0] if tuning_results.get('prediction') else 500.0,
                'hardness': tuning_results.get('prediction', [200.0])[1] if len(tuning_results.get('prediction', [])) > 1 else 200.0,
                'density': tuning_results.get('prediction', [7.8])[2] if len(tuning_results.get('prediction', [])) > 2 else 7.8
            }
            
            # Update metrics
            self._update_service_metrics(start_time, True)
            
            return MaterialTuningResponse(
                tuned_parameters=tuned_parameters,
                predicted_properties=predicted_properties,
                improvement_score=tuning_results.get('prediction', [0.1])[0] if tuning_results.get('prediction') else 0.1,
                confidence=tuning_results.get('confidence', [0.8])[0] if tuning_results.get('confidence') else 0.8,
                processing_time=time.time() - start_time,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"Material tuning failed: {e}")
            self._update_service_metrics(start_time, False)
            raise HTTPException(status_code=500, detail=str(e))
    
    def _generate_optimization_recommendations(self, request: ProcessOptimizationRequest, optimized_parameters: Dict[str, float]) -> List[str]:
        """
        Generate optimization recommendations.
        
        Args:
            request: Original request
            optimized_parameters: Optimized parameters
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Compare current vs optimized parameters
        if abs(optimized_parameters['laser_power'] - request.laser_power) > 10:
            recommendations.append(f"Consider adjusting laser power from {request.laser_power}W to {optimized_parameters['laser_power']:.1f}W")
        
        if abs(optimized_parameters['scan_speed'] - request.scan_speed) > 100:
            recommendations.append(f"Consider adjusting scan speed from {request.scan_speed}mm/s to {optimized_parameters['scan_speed']:.1f}mm/s")
        
        if abs(optimized_parameters['layer_height'] - request.layer_height) > 0.01:
            recommendations.append(f"Consider adjusting layer height from {request.layer_height}mm to {optimized_parameters['layer_height']:.3f}mm")
        
        if abs(optimized_parameters['hatch_spacing'] - request.hatch_spacing) > 0.01:
            recommendations.append(f"Consider adjusting hatch spacing from {request.hatch_spacing}mm to {optimized_parameters['hatch_spacing']:.3f}mm")
        
        # Add general recommendations
        if request.chamber_temperature < 80:
            recommendations.append("Consider increasing chamber temperature for better material flow")
        
        if request.target_quality > 0.9:
            recommendations.append("High quality target detected - consider slower build speed for better results")
        
        return recommendations
    
    def _update_service_metrics(self, start_time: float, success: bool):
        """
        Update service metrics.
        
        Args:
            start_time: Request start time
            success: Whether request was successful
        """
        response_time = time.time() - start_time
        
        self.service_metrics['total_requests'] += 1
        if success:
            self.service_metrics['successful_requests'] += 1
        else:
            self.service_metrics['failed_requests'] += 1
        
        # Update average response time
        total_successful = self.service_metrics['successful_requests']
        if total_successful > 0:
            current_avg = self.service_metrics['average_response_time']
            self.service_metrics['average_response_time'] = (
                (current_avg * (total_successful - 1) + response_time) / total_successful
            )
        
        self.service_metrics['last_request_time'] = datetime.now().isoformat()
    
    def run(self, host: str = "0.0.0.0", port: int = 8001):
        """
        Run the service.
        
        Args:
            host: Host address
            port: Port number
        """
        logger.info(f"Starting Process Optimization Service on {host}:{port}")
        uvicorn.run(self.app, host=host, port=port)


# Global service instance
service = ProcessOptimizationService()


def create_app() -> FastAPI:
    """Create FastAPI application."""
    return service.app


if __name__ == "__main__":
    service.run()
