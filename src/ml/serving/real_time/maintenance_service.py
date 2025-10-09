"""
Predictive Maintenance Real-time Service

This module implements the real-time service for predictive maintenance in PBF-LB/M processes.
It provides REST API endpoints for real-time equipment health monitoring, failure prediction,
maintenance scheduling, and cost optimization.
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
from ...models.predictive_maintenance import EquipmentHealthMonitor, FailurePredictor, MaintenanceScheduler, CostOptimizer
from ...config.config_manager import ConfigManager

logger = logging.getLogger(__name__)


# Pydantic models for API requests and responses
class HealthMonitoringRequest(BaseModel):
    """Request model for equipment health monitoring."""
    equipment_id: str = Field(..., description="Equipment identifier")
    sensor_data: Dict[str, float] = Field(..., description="Sensor data (temperature, vibration, pressure)")
    operational_data: Dict[str, Any] = Field(..., description="Operational data")
    maintenance_history: Optional[List[Dict[str, Any]]] = Field(None, description="Maintenance history")
    timestamp: Optional[str] = Field(None, description="Monitoring timestamp")


class HealthMonitoringResponse(BaseModel):
    """Response model for equipment health monitoring."""
    equipment_id: str = Field(..., description="Equipment identifier")
    health_score: float = Field(..., description="Health score (0-1)")
    health_status: str = Field(..., description="Health status (excellent, good, fair, poor, critical)")
    predicted_rul: float = Field(..., description="Predicted remaining useful life (hours)")
    risk_factors: List[str] = Field(..., description="Identified risk factors")
    confidence: float = Field(..., description="Prediction confidence")
    recommendations: List[str] = Field(..., description="Health improvement recommendations")
    processing_time: float = Field(..., description="Processing time in seconds")
    timestamp: str = Field(..., description="Response timestamp")


class FailurePredictionRequest(BaseModel):
    """Request model for failure prediction."""
    equipment_id: str = Field(..., description="Equipment identifier")
    sensor_data: Dict[str, float] = Field(..., description="Sensor data")
    operational_conditions: Dict[str, Any] = Field(..., description="Operational conditions")
    failure_history: Optional[List[Dict[str, Any]]] = Field(None, description="Historical failure data")
    prediction_horizon: int = Field(24, description="Prediction horizon in hours")


class FailurePredictionResponse(BaseModel):
    """Response model for failure prediction."""
    equipment_id: str = Field(..., description="Equipment identifier")
    failure_probability: float = Field(..., description="Failure probability (0-1)")
    predicted_failure_time: Optional[str] = Field(None, description="Predicted failure time")
    failure_type: str = Field(..., description="Predicted failure type")
    risk_level: str = Field(..., description="Risk level (low, medium, high, critical)")
    confidence: float = Field(..., description="Prediction confidence")
    preventive_actions: List[str] = Field(..., description="Preventive actions")
    processing_time: float = Field(..., description="Processing time in seconds")
    timestamp: str = Field(..., description="Response timestamp")


class MaintenanceSchedulingRequest(BaseModel):
    """Request model for maintenance scheduling."""
    equipment_id: str = Field(..., description="Equipment identifier")
    current_health: float = Field(..., description="Current health score")
    maintenance_types: List[str] = Field(..., description="Available maintenance types")
    constraints: Dict[str, Any] = Field(..., description="Scheduling constraints")
    cost_priorities: Optional[Dict[str, float]] = Field(None, description="Cost priorities")


class MaintenanceSchedulingResponse(BaseModel):
    """Response model for maintenance scheduling."""
    equipment_id: str = Field(..., description="Equipment identifier")
    recommended_schedule: Dict[str, Any] = Field(..., description="Recommended maintenance schedule")
    maintenance_priority: str = Field(..., description="Maintenance priority (low, medium, high, urgent)")
    estimated_duration: float = Field(..., description="Estimated maintenance duration (hours)")
    estimated_cost: float = Field(..., description="Estimated maintenance cost")
    expected_improvement: float = Field(..., description="Expected health improvement")
    confidence: float = Field(..., description="Scheduling confidence")
    processing_time: float = Field(..., description="Processing time in seconds")
    timestamp: str = Field(..., description="Response timestamp")


class CostOptimizationRequest(BaseModel):
    """Request model for cost optimization."""
    equipment_id: str = Field(..., description="Equipment identifier")
    maintenance_options: List[Dict[str, Any]] = Field(..., description="Available maintenance options")
    budget_constraints: Dict[str, float] = Field(..., description="Budget constraints")
    performance_requirements: Dict[str, float] = Field(..., description="Performance requirements")
    time_horizon: int = Field(30, description="Optimization time horizon (days)")


class CostOptimizationResponse(BaseModel):
    """Response model for cost optimization."""
    equipment_id: str = Field(..., description="Equipment identifier")
    optimized_plan: Dict[str, Any] = Field(..., description="Optimized maintenance plan")
    total_cost: float = Field(..., description="Total optimized cost")
    cost_savings: float = Field(..., description="Cost savings compared to baseline")
    performance_improvement: float = Field(..., description="Expected performance improvement")
    confidence: float = Field(..., description="Optimization confidence")
    processing_time: float = Field(..., description="Processing time in seconds")
    timestamp: str = Field(..., description="Response timestamp")


class MaintenanceService:
    """
    Real-time predictive maintenance service for PBF-LB/M processes.
    
    This service provides real-time maintenance support for:
    - Equipment health monitoring
    - Failure prediction
    - Maintenance scheduling
    - Cost optimization
    """
    
    def __init__(self, config_manager: Optional[ConfigManager] = None):
        """
        Initialize the maintenance service.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config_manager = config_manager or ConfigManager()
        self.app = FastAPI(
            title="Predictive Maintenance Service",
            description="Real-time predictive maintenance for PBF-LB/M manufacturing",
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
            'equipment_health_monitor': EquipmentHealthMonitor(self.config_manager),
            'failure_predictor': FailurePredictor(self.config_manager),
            'maintenance_scheduler': MaintenanceScheduler(self.config_manager),
            'cost_optimizer': CostOptimizer(self.config_manager)
        }
        
        # Service metrics
        self.service_metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'average_response_time': 0.0,
            'last_request_time': None,
            'health_assessments': 0,
            'failure_predictions': 0,
            'maintenance_schedules': 0,
            'cost_optimizations': 0
        }
        
        # Setup routes
        self._setup_routes()
        
        logger.info("Initialized MaintenanceService")
    
    def _setup_routes(self):
        """Setup API routes."""
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {
                "status": "healthy",
                "service": "predictive_maintenance",
                "timestamp": datetime.now().isoformat(),
                "metrics": self.service_metrics
            }
        
        @self.app.post("/monitor-health", response_model=HealthMonitoringResponse)
        async def monitor_health(request: HealthMonitoringRequest):
            """Monitor equipment health in real-time."""
            return await self._monitor_health(request)
        
        @self.app.post("/predict-failure", response_model=FailurePredictionResponse)
        async def predict_failure(request: FailurePredictionRequest):
            """Predict equipment failure."""
            return await self._predict_failure(request)
        
        @self.app.post("/schedule-maintenance", response_model=MaintenanceSchedulingResponse)
        async def schedule_maintenance(request: MaintenanceSchedulingRequest):
            """Schedule maintenance activities."""
            return await self._schedule_maintenance(request)
        
        @self.app.post("/optimize-costs", response_model=CostOptimizationResponse)
        async def optimize_costs(request: CostOptimizationRequest):
            """Optimize maintenance costs."""
            return await self._optimize_costs(request)
        
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
    
    async def _monitor_health(self, request: HealthMonitoringRequest) -> HealthMonitoringResponse:
        """
        Monitor equipment health in real-time.
        
        Args:
            request: Health monitoring request
            
        Returns:
            Health monitoring response
        """
        start_time = time.time()
        
        try:
            # Prepare input data
            input_data = {
                'timestamp': request.timestamp or datetime.now().isoformat(),
                'sensor_data': request.sensor_data,
                'process_data': {
                    'equipment_id': request.equipment_id,
                    'operational_data': request.operational_data,
                    'maintenance_history': request.maintenance_history or []
                }
            }
            
            # Get model configurations
            model_configs = [
                {
                    'model_name': 'equipment_health_monitor',
                    'version': 'latest'
                }
            ]
            
            # Generate predictions
            predictions = await self.inference_pipeline.process_real_time_data(input_data, model_configs)
            
            if predictions['status'] == 'error':
                raise HTTPException(status_code=500, detail=predictions.get('error', 'Prediction failed'))
            
            # Extract health monitoring results
            health_results = predictions['predictions'].get('equipment_health_monitor', {})
            
            # Process health monitoring results
            health_score = health_results.get('prediction', [0.8])[0] if health_results.get('prediction') else 0.8
            confidence = health_results.get('confidence', [0.8])[0] if health_results.get('confidence') else 0.8
            
            # Determine health status
            health_status = self._determine_health_status(health_score)
            
            # Predict remaining useful life
            predicted_rul = self._predict_remaining_useful_life(health_score, request.sensor_data)
            
            # Identify risk factors
            risk_factors = self._identify_risk_factors(request.sensor_data, request.operational_data)
            
            # Generate recommendations
            recommendations = self._generate_health_recommendations(health_score, risk_factors)
            
            # Update metrics
            self.service_metrics['health_assessments'] += 1
            
            self._update_service_metrics(start_time, True)
            
            return HealthMonitoringResponse(
                equipment_id=request.equipment_id,
                health_score=health_score,
                health_status=health_status,
                predicted_rul=predicted_rul,
                risk_factors=risk_factors,
                confidence=confidence,
                recommendations=recommendations,
                processing_time=time.time() - start_time,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"Health monitoring failed: {e}")
            self._update_service_metrics(start_time, False)
            raise HTTPException(status_code=500, detail=str(e))
    
    async def _predict_failure(self, request: FailurePredictionRequest) -> FailurePredictionResponse:
        """
        Predict equipment failure.
        
        Args:
            request: Failure prediction request
            
        Returns:
            Failure prediction response
        """
        start_time = time.time()
        
        try:
            # Prepare input data
            input_data = {
                'timestamp': datetime.now().isoformat(),
                'sensor_data': request.sensor_data,
                'process_data': {
                    'equipment_id': request.equipment_id,
                    'operational_conditions': request.operational_conditions,
                    'failure_history': request.failure_history or [],
                    'prediction_horizon': request.prediction_horizon
                }
            }
            
            # Get model configurations
            model_configs = [
                {
                    'model_name': 'failure_predictor',
                    'version': 'latest'
                }
            ]
            
            # Generate predictions
            predictions = await self.inference_pipeline.process_real_time_data(input_data, model_configs)
            
            if predictions['status'] == 'error':
                raise HTTPException(status_code=500, detail=predictions.get('error', 'Prediction failed'))
            
            # Extract failure prediction results
            failure_results = predictions['predictions'].get('failure_predictor', {})
            
            # Process failure prediction results
            failure_probability = failure_results.get('prediction', [0.1])[0] if failure_results.get('prediction') else 0.1
            confidence = failure_results.get('confidence', [0.8])[0] if failure_results.get('confidence') else 0.8
            
            # Predict failure time
            predicted_failure_time = self._predict_failure_time(failure_probability, request.prediction_horizon)
            
            # Determine failure type
            failure_type = self._determine_failure_type(request.sensor_data, request.operational_conditions)
            
            # Determine risk level
            risk_level = self._determine_risk_level(failure_probability)
            
            # Generate preventive actions
            preventive_actions = self._generate_preventive_actions(failure_type, risk_level, request)
            
            # Update metrics
            self.service_metrics['failure_predictions'] += 1
            
            self._update_service_metrics(start_time, True)
            
            return FailurePredictionResponse(
                equipment_id=request.equipment_id,
                failure_probability=failure_probability,
                predicted_failure_time=predicted_failure_time,
                failure_type=failure_type,
                risk_level=risk_level,
                confidence=confidence,
                preventive_actions=preventive_actions,
                processing_time=time.time() - start_time,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"Failure prediction failed: {e}")
            self._update_service_metrics(start_time, False)
            raise HTTPException(status_code=500, detail=str(e))
    
    async def _schedule_maintenance(self, request: MaintenanceSchedulingRequest) -> MaintenanceSchedulingResponse:
        """
        Schedule maintenance activities.
        
        Args:
            request: Maintenance scheduling request
            
        Returns:
            Maintenance scheduling response
        """
        start_time = time.time()
        
        try:
            # Prepare input data
            input_data = {
                'timestamp': datetime.now().isoformat(),
                'sensor_data': {
                    'current_health': request.current_health
                },
                'process_data': {
                    'equipment_id': request.equipment_id,
                    'maintenance_types': request.maintenance_types,
                    'constraints': request.constraints,
                    'cost_priorities': request.cost_priorities or {}
                }
            }
            
            # Get model configurations
            model_configs = [
                {
                    'model_name': 'maintenance_scheduler',
                    'version': 'latest'
                }
            ]
            
            # Generate predictions
            predictions = await self.inference_pipeline.process_real_time_data(input_data, model_configs)
            
            if predictions['status'] == 'error':
                raise HTTPException(status_code=500, detail=predictions.get('error', 'Prediction failed'))
            
            # Extract maintenance scheduling results
            scheduling_results = predictions['predictions'].get('maintenance_scheduler', {})
            
            # Process maintenance scheduling results
            confidence = scheduling_results.get('confidence', [0.8])[0] if scheduling_results.get('confidence') else 0.8
            
            # Generate maintenance schedule
            recommended_schedule = self._generate_maintenance_schedule(request, scheduling_results)
            
            # Determine maintenance priority
            maintenance_priority = self._determine_maintenance_priority(request.current_health, request.constraints)
            
            # Estimate duration and cost
            estimated_duration = self._estimate_maintenance_duration(request.maintenance_types, request.current_health)
            estimated_cost = self._estimate_maintenance_cost(request.maintenance_types, estimated_duration)
            
            # Calculate expected improvement
            expected_improvement = self._calculate_expected_improvement(request.current_health, request.maintenance_types)
            
            # Update metrics
            self.service_metrics['maintenance_schedules'] += 1
            
            self._update_service_metrics(start_time, True)
            
            return MaintenanceSchedulingResponse(
                equipment_id=request.equipment_id,
                recommended_schedule=recommended_schedule,
                maintenance_priority=maintenance_priority,
                estimated_duration=estimated_duration,
                estimated_cost=estimated_cost,
                expected_improvement=expected_improvement,
                confidence=confidence,
                processing_time=time.time() - start_time,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"Maintenance scheduling failed: {e}")
            self._update_service_metrics(start_time, False)
            raise HTTPException(status_code=500, detail=str(e))
    
    async def _optimize_costs(self, request: CostOptimizationRequest) -> CostOptimizationResponse:
        """
        Optimize maintenance costs.
        
        Args:
            request: Cost optimization request
            
        Returns:
            Cost optimization response
        """
        start_time = time.time()
        
        try:
            # Prepare input data
            input_data = {
                'timestamp': datetime.now().isoformat(),
                'sensor_data': {},
                'process_data': {
                    'equipment_id': request.equipment_id,
                    'maintenance_options': request.maintenance_options,
                    'budget_constraints': request.budget_constraints,
                    'performance_requirements': request.performance_requirements,
                    'time_horizon': request.time_horizon
                }
            }
            
            # Get model configurations
            model_configs = [
                {
                    'model_name': 'cost_optimizer',
                    'version': 'latest'
                }
            ]
            
            # Generate predictions
            predictions = await self.inference_pipeline.process_real_time_data(input_data, model_configs)
            
            if predictions['status'] == 'error':
                raise HTTPException(status_code=500, detail=predictions.get('error', 'Prediction failed'))
            
            # Extract cost optimization results
            optimization_results = predictions['predictions'].get('cost_optimizer', {})
            
            # Process cost optimization results
            confidence = optimization_results.get('confidence', [0.8])[0] if optimization_results.get('confidence') else 0.8
            
            # Generate optimized plan
            optimized_plan = self._generate_optimized_plan(request, optimization_results)
            
            # Calculate costs and savings
            total_cost = self._calculate_total_cost(optimized_plan)
            cost_savings = self._calculate_cost_savings(optimized_plan, request.maintenance_options)
            
            # Calculate performance improvement
            performance_improvement = self._calculate_performance_improvement(optimized_plan, request.performance_requirements)
            
            # Update metrics
            self.service_metrics['cost_optimizations'] += 1
            
            self._update_service_metrics(start_time, True)
            
            return CostOptimizationResponse(
                equipment_id=request.equipment_id,
                optimized_plan=optimized_plan,
                total_cost=total_cost,
                cost_savings=cost_savings,
                performance_improvement=performance_improvement,
                confidence=confidence,
                processing_time=time.time() - start_time,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"Cost optimization failed: {e}")
            self._update_service_metrics(start_time, False)
            raise HTTPException(status_code=500, detail=str(e))
    
    def _determine_health_status(self, health_score: float) -> str:
        """Determine health status based on score."""
        if health_score >= 0.9:
            return "excellent"
        elif health_score >= 0.8:
            return "good"
        elif health_score >= 0.6:
            return "fair"
        elif health_score >= 0.4:
            return "poor"
        else:
            return "critical"
    
    def _predict_remaining_useful_life(self, health_score: float, sensor_data: Dict[str, float]) -> float:
        """Predict remaining useful life in hours."""
        # Mock calculation based on health score and sensor data
        base_rul = health_score * 1000  # Base RUL in hours
        
        # Adjust based on sensor data
        if 'temperature' in sensor_data and sensor_data['temperature'] > 80:
            base_rul *= 0.8
        if 'vibration' in sensor_data and sensor_data['vibration'] > 5:
            base_rul *= 0.7
        
        return max(base_rul, 24)  # Minimum 24 hours
    
    def _identify_risk_factors(self, sensor_data: Dict[str, float], operational_data: Dict[str, Any]) -> List[str]:
        """Identify risk factors from sensor and operational data."""
        risk_factors = []
        
        if 'temperature' in sensor_data and sensor_data['temperature'] > 80:
            risk_factors.append("High operating temperature")
        if 'vibration' in sensor_data and sensor_data['vibration'] > 5:
            risk_factors.append("Excessive vibration")
        if 'pressure' in sensor_data and sensor_data['pressure'] > 10:
            risk_factors.append("High pressure conditions")
        
        if 'operating_hours' in operational_data and operational_data['operating_hours'] > 8000:
            risk_factors.append("High operating hours")
        if 'maintenance_interval' in operational_data and operational_data['maintenance_interval'] > 500:
            risk_factors.append("Overdue maintenance")
        
        return risk_factors
    
    def _generate_health_recommendations(self, health_score: float, risk_factors: List[str]) -> List[str]:
        """Generate health improvement recommendations."""
        recommendations = []
        
        if health_score < 0.6:
            recommendations.append("Schedule immediate maintenance inspection")
            recommendations.append("Reduce operating load to prevent further degradation")
        
        if "High operating temperature" in risk_factors:
            recommendations.append("Check cooling system and thermal management")
        if "Excessive vibration" in risk_factors:
            recommendations.append("Inspect bearings and mechanical components")
        if "High pressure conditions" in risk_factors:
            recommendations.append("Verify pressure relief systems")
        
        if health_score < 0.8:
            recommendations.append("Increase monitoring frequency")
            recommendations.append("Prepare maintenance plan for next scheduled downtime")
        
        return recommendations
    
    def _predict_failure_time(self, failure_probability: float, prediction_horizon: int) -> Optional[str]:
        """Predict failure time based on probability."""
        if failure_probability < 0.3:
            return None
        
        # Mock calculation
        hours_to_failure = prediction_horizon * (1 - failure_probability)
        failure_time = datetime.now() + timedelta(hours=hours_to_failure)
        return failure_time.isoformat()
    
    def _determine_failure_type(self, sensor_data: Dict[str, float], operational_conditions: Dict[str, Any]) -> str:
        """Determine most likely failure type."""
        if 'temperature' in sensor_data and sensor_data['temperature'] > 90:
            return "thermal_failure"
        elif 'vibration' in sensor_data and sensor_data['vibration'] > 8:
            return "mechanical_failure"
        elif 'pressure' in sensor_data and sensor_data['pressure'] > 15:
            return "pressure_failure"
        else:
            return "general_wear"
    
    def _determine_risk_level(self, failure_probability: float) -> str:
        """Determine risk level based on failure probability."""
        if failure_probability >= 0.8:
            return "critical"
        elif failure_probability >= 0.6:
            return "high"
        elif failure_probability >= 0.4:
            return "medium"
        else:
            return "low"
    
    def _generate_preventive_actions(self, failure_type: str, risk_level: str, request: FailurePredictionRequest) -> List[str]:
        """Generate preventive actions based on failure type and risk level."""
        actions = []
        
        if risk_level in ["critical", "high"]:
            actions.append("Schedule immediate maintenance")
            actions.append("Reduce operating load")
        
        if failure_type == "thermal_failure":
            actions.append("Check cooling system")
            actions.append("Monitor temperature closely")
        elif failure_type == "mechanical_failure":
            actions.append("Inspect mechanical components")
            actions.append("Check alignment and balance")
        elif failure_type == "pressure_failure":
            actions.append("Verify pressure relief systems")
            actions.append("Check for leaks")
        
        if risk_level == "medium":
            actions.append("Increase monitoring frequency")
            actions.append("Prepare maintenance plan")
        
        return actions
    
    def _generate_maintenance_schedule(self, request: MaintenanceSchedulingRequest, scheduling_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate maintenance schedule."""
        return {
            "next_maintenance": (datetime.now() + timedelta(days=7)).isoformat(),
            "maintenance_type": request.maintenance_types[0] if request.maintenance_types else "routine_inspection",
            "estimated_duration_hours": 4,
            "required_resources": ["technician", "spare_parts"],
            "priority": "medium"
        }
    
    def _determine_maintenance_priority(self, current_health: float, constraints: Dict[str, Any]) -> str:
        """Determine maintenance priority."""
        if current_health < 0.4:
            return "urgent"
        elif current_health < 0.6:
            return "high"
        elif current_health < 0.8:
            return "medium"
        else:
            return "low"
    
    def _estimate_maintenance_duration(self, maintenance_types: List[str], current_health: float) -> float:
        """Estimate maintenance duration in hours."""
        base_duration = 2.0  # Base duration in hours
        
        # Adjust based on maintenance types
        if "major_overhaul" in maintenance_types:
            base_duration += 8.0
        elif "minor_repair" in maintenance_types:
            base_duration += 2.0
        
        # Adjust based on health
        if current_health < 0.5:
            base_duration *= 1.5
        
        return base_duration
    
    def _estimate_maintenance_cost(self, maintenance_types: List[str], duration: float) -> float:
        """Estimate maintenance cost."""
        base_cost = 100.0  # Base cost
        
        # Adjust based on maintenance types
        if "major_overhaul" in maintenance_types:
            base_cost += 500.0
        elif "minor_repair" in maintenance_types:
            base_cost += 200.0
        
        # Adjust based on duration
        base_cost += duration * 50.0  # $50 per hour
        
        return base_cost
    
    def _calculate_expected_improvement(self, current_health: float, maintenance_types: List[str]) -> float:
        """Calculate expected health improvement."""
        base_improvement = 0.1  # Base improvement
        
        # Adjust based on maintenance types
        if "major_overhaul" in maintenance_types:
            base_improvement += 0.3
        elif "minor_repair" in maintenance_types:
            base_improvement += 0.1
        
        # Adjust based on current health
        if current_health < 0.5:
            base_improvement += 0.2
        
        return min(base_improvement, 1.0 - current_health)
    
    def _generate_optimized_plan(self, request: CostOptimizationRequest, optimization_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate optimized maintenance plan."""
        return {
            "schedule": [
                {
                    "date": (datetime.now() + timedelta(days=7)).isoformat(),
                    "maintenance_type": "routine_inspection",
                    "cost": 200.0,
                    "duration": 2.0
                },
                {
                    "date": (datetime.now() + timedelta(days=30)).isoformat(),
                    "maintenance_type": "preventive_maintenance",
                    "cost": 500.0,
                    "duration": 4.0
                }
            ],
            "total_duration": 6.0,
            "risk_reduction": 0.3
        }
    
    def _calculate_total_cost(self, optimized_plan: Dict[str, Any]) -> float:
        """Calculate total cost of optimized plan."""
        total_cost = 0.0
        for item in optimized_plan.get("schedule", []):
            total_cost += item.get("cost", 0.0)
        return total_cost
    
    def _calculate_cost_savings(self, optimized_plan: Dict[str, Any], maintenance_options: List[Dict[str, Any]]) -> float:
        """Calculate cost savings compared to baseline."""
        # Mock calculation
        baseline_cost = 1000.0
        optimized_cost = self._calculate_total_cost(optimized_plan)
        return max(0, baseline_cost - optimized_cost)
    
    def _calculate_performance_improvement(self, optimized_plan: Dict[str, Any], performance_requirements: Dict[str, float]) -> float:
        """Calculate expected performance improvement."""
        # Mock calculation
        return optimized_plan.get("risk_reduction", 0.3)
    
    def _update_service_metrics(self, start_time: float, success: bool):
        """Update service metrics."""
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
    
    def run(self, host: str = "0.0.0.0", port: int = 8004):
        """Run the service."""
        logger.info(f"Starting Predictive Maintenance Service on {host}:{port}")
        uvicorn.run(self.app, host=host, port=port)


# Global service instance
service = MaintenanceService()


def create_app() -> FastAPI:
    """Create FastAPI application."""
    return service.app


if __name__ == "__main__":
    service.run()
