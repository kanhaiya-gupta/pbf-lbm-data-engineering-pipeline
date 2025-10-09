"""
Performance Tracker

This module implements the performance tracker for PBF-LB/M processes.
It provides comprehensive performance monitoring, metrics collection,
and performance analytics for ML models and systems.
"""

import logging
import time
import asyncio
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import pickle
from collections import defaultdict, deque
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import mlflow
import mlflow.tensorflow
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from plotly.utils import PlotlyJSONEncoder

from ...config.config_manager import ConfigManager

logger = logging.getLogger(__name__)


# Pydantic models for API requests and responses
class PerformanceMetrics(BaseModel):
    """Model for performance metrics."""
    model_id: str = Field(..., description="Model ID")
    timestamp: str = Field(..., description="Metrics timestamp")
    accuracy: Optional[float] = Field(None, description="Accuracy score")
    precision: Optional[float] = Field(None, description="Precision score")
    recall: Optional[float] = Field(None, description="Recall score")
    f1_score: Optional[float] = Field(None, description="F1 score")
    mse: Optional[float] = Field(None, description="Mean squared error")
    mae: Optional[float] = Field(None, description="Mean absolute error")
    r2_score: Optional[float] = Field(None, description="R-squared score")
    inference_time: Optional[float] = Field(None, description="Inference time in ms")
    throughput: Optional[float] = Field(None, description="Throughput (requests/second)")
    memory_usage: Optional[float] = Field(None, description="Memory usage in MB")
    cpu_usage: Optional[float] = Field(None, description="CPU usage percentage")


class PerformanceTrackingRequest(BaseModel):
    """Request model for performance tracking."""
    tracker_id: str = Field(..., description="Tracker ID")
    model_id: str = Field(..., description="Model ID to track")
    metrics: PerformanceMetrics = Field(..., description="Performance metrics")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class PerformanceTrackingResponse(BaseModel):
    """Response model for performance tracking."""
    tracker_id: str = Field(..., description="Tracker ID")
    model_id: str = Field(..., description="Model ID")
    status: str = Field(..., description="Tracking status")
    metrics_stored: bool = Field(..., description="Whether metrics were stored")
    performance_trend: Optional[str] = Field(None, description="Performance trend")
    alerts_triggered: List[str] = Field(..., description="Alerts triggered")
    timestamp: str = Field(..., description="Response timestamp")


class PerformanceAnalytics(BaseModel):
    """Model for performance analytics."""
    model_id: str = Field(..., description="Model ID")
    time_period: str = Field(..., description="Time period for analytics")
    total_measurements: int = Field(..., description="Total number of measurements")
    average_accuracy: Optional[float] = Field(None, description="Average accuracy")
    accuracy_trend: str = Field(..., description="Accuracy trend")
    average_inference_time: Optional[float] = Field(None, description="Average inference time")
    throughput_trend: str = Field(..., description="Throughput trend")
    performance_score: float = Field(..., description="Overall performance score")
    recommendations: List[str] = Field(..., description="Performance recommendations")


class PerformanceTracker:
    """
    Performance tracker for PBF-LB/M processes.
    
    This tracker provides comprehensive performance monitoring capabilities for:
    - Model performance tracking
    - System performance monitoring
    - Performance analytics
    - Performance alerts
    """
    
    def __init__(self, config_manager: Optional[ConfigManager] = None):
        """
        Initialize the performance tracker.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config_manager = config_manager or ConfigManager()
        self.app = FastAPI(
            title="Performance Tracker",
            description="Performance tracking for PBF-LB/M manufacturing",
            version="1.0.0"
        )
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Performance tracking
        self.trackers = {}  # Store tracker information
        self.performance_history = defaultdict(list)  # Store performance history
        self.performance_alerts = {}  # Store performance alerts
        self.tracker_counter = 0
        
        # Service metrics
        self.service_metrics = {
            'total_trackers': 0,
            'active_trackers': 0,
            'total_measurements': 0,
            'models_tracked': 0,
            'total_alerts': 0,
            'last_measurement_time': None
        }
        
        # Performance thresholds
        self.performance_thresholds = {
            'accuracy_min': 0.8,
            'precision_min': 0.7,
            'recall_min': 0.7,
            'f1_score_min': 0.7,
            'inference_time_max': 1000,  # ms
            'memory_usage_max': 2048,  # MB
            'cpu_usage_max': 80  # percentage
        }
        
        # Setup routes
        self._setup_routes()
        
        logger.info("Initialized PerformanceTracker")
    
    def _setup_routes(self):
        """Setup API routes."""
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {
                "status": "healthy",
                "service": "performance_tracker",
                "timestamp": datetime.now().isoformat(),
                "metrics": self.service_metrics
            }
        
        @self.app.post("/track", response_model=PerformanceTrackingResponse)
        async def track_performance(request: PerformanceTrackingRequest):
            """Track model performance."""
            return await self._track_performance(request)
        
        @self.app.post("/trackers", response_model=Dict[str, Any])
        async def create_tracker(request: PerformanceTrackingRequest):
            """Create a new performance tracker."""
            return await self._create_tracker(request)
        
        @self.app.get("/trackers")
        async def list_trackers():
            """List all trackers."""
            return await self._list_trackers()
        
        @self.app.get("/trackers/{tracker_id}/status")
        async def get_tracker_status(tracker_id: str):
            """Get tracker status."""
            return await self._get_tracker_status(tracker_id)
        
        @self.app.get("/models/{model_id}/performance")
        async def get_model_performance(model_id: str, limit: int = Query(100, ge=1, le=1000)):
            """Get model performance history."""
            return await self._get_model_performance(model_id, limit)
        
        @self.app.get("/models/{model_id}/analytics", response_model=PerformanceAnalytics)
        async def get_performance_analytics(model_id: str, days: int = Query(7, ge=1, le=30)):
            """Get performance analytics for a model."""
            return await self._get_performance_analytics(model_id, days)
        
        @self.app.get("/models/{model_id}/trends")
        async def get_performance_trends(model_id: str, days: int = Query(7, ge=1, le=30)):
            """Get performance trends for a model."""
            return await self._get_performance_trends(model_id, days)
        
        @self.app.get("/alerts")
        async def list_performance_alerts(limit: int = Query(10, ge=1, le=100)):
            """List performance alerts."""
            return await self._list_performance_alerts(limit)
        
        @self.app.post("/alerts/{alert_id}/resolve")
        async def resolve_alert(alert_id: str):
            """Resolve a performance alert."""
            return await self._resolve_alert(alert_id)
        
        @self.app.get("/metrics")
        async def get_metrics():
            """Get service metrics."""
            return self.service_metrics
    
    async def _track_performance(self, request: PerformanceTrackingRequest) -> PerformanceTrackingResponse:
        """
        Track model performance.
        
        Args:
            request: Performance tracking request
            
        Returns:
            Performance tracking response
        """
        try:
            # Store performance metrics
            await self._store_performance_metrics(request.model_id, request.metrics)
            
            # Analyze performance trend
            performance_trend = await self._analyze_performance_trend(request.model_id)
            
            # Check for performance alerts
            alerts_triggered = await self._check_performance_alerts(request.model_id, request.metrics)
            
            # Update metrics
            self.service_metrics['total_measurements'] += 1
            self.service_metrics['last_measurement_time'] = datetime.now().isoformat()
            
            return PerformanceTrackingResponse(
                tracker_id=request.tracker_id,
                model_id=request.model_id,
                status="tracked",
                metrics_stored=True,
                performance_trend=performance_trend,
                alerts_triggered=alerts_triggered,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"Error tracking performance: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def _store_performance_metrics(self, model_id: str, metrics: PerformanceMetrics):
        """Store performance metrics."""
        # Convert metrics to dictionary
        metrics_dict = metrics.dict()
        metrics_dict['timestamp'] = datetime.now().isoformat()
        
        # Store in history
        self.performance_history[model_id].append(metrics_dict)
        
        # Keep only recent history (last 10000 measurements)
        if len(self.performance_history[model_id]) > 10000:
            self.performance_history[model_id] = self.performance_history[model_id][-10000:]
    
    async def _analyze_performance_trend(self, model_id: str) -> str:
        """Analyze performance trend for a model."""
        if model_id not in self.performance_history or len(self.performance_history[model_id]) < 2:
            return "insufficient_data"
        
        history = self.performance_history[model_id]
        
        # Get recent metrics (last 10 measurements)
        recent_metrics = history[-10:]
        
        # Calculate trend for accuracy if available
        if all('accuracy' in m and m['accuracy'] is not None for m in recent_metrics):
            accuracies = [m['accuracy'] for m in recent_metrics]
            if len(accuracies) >= 2:
                # Simple trend analysis
                first_half = np.mean(accuracies[:len(accuracies)//2])
                second_half = np.mean(accuracies[len(accuracies)//2:])
                
                if second_half > first_half + 0.01:
                    return "improving"
                elif second_half < first_half - 0.01:
                    return "declining"
                else:
                    return "stable"
        
        return "stable"
    
    async def _check_performance_alerts(self, model_id: str, metrics: PerformanceMetrics) -> List[str]:
        """Check for performance alerts."""
        alerts_triggered = []
        
        try:
            # Check accuracy threshold
            if metrics.accuracy is not None and metrics.accuracy < self.performance_thresholds['accuracy_min']:
                await self._create_performance_alert(
                    model_id, "low_accuracy", 
                    f"Accuracy {metrics.accuracy:.3f} below threshold {self.performance_thresholds['accuracy_min']}"
                )
                alerts_triggered.append("low_accuracy")
            
            # Check precision threshold
            if metrics.precision is not None and metrics.precision < self.performance_thresholds['precision_min']:
                await self._create_performance_alert(
                    model_id, "low_precision",
                    f"Precision {metrics.precision:.3f} below threshold {self.performance_thresholds['precision_min']}"
                )
                alerts_triggered.append("low_precision")
            
            # Check recall threshold
            if metrics.recall is not None and metrics.recall < self.performance_thresholds['recall_min']:
                await self._create_performance_alert(
                    model_id, "low_recall",
                    f"Recall {metrics.recall:.3f} below threshold {self.performance_thresholds['recall_min']}"
                )
                alerts_triggered.append("low_recall")
            
            # Check F1 score threshold
            if metrics.f1_score is not None and metrics.f1_score < self.performance_thresholds['f1_score_min']:
                await self._create_performance_alert(
                    model_id, "low_f1_score",
                    f"F1 score {metrics.f1_score:.3f} below threshold {self.performance_thresholds['f1_score_min']}"
                )
                alerts_triggered.append("low_f1_score")
            
            # Check inference time threshold
            if metrics.inference_time is not None and metrics.inference_time > self.performance_thresholds['inference_time_max']:
                await self._create_performance_alert(
                    model_id, "high_inference_time",
                    f"Inference time {metrics.inference_time:.1f}ms above threshold {self.performance_thresholds['inference_time_max']}ms"
                )
                alerts_triggered.append("high_inference_time")
            
            # Check memory usage threshold
            if metrics.memory_usage is not None and metrics.memory_usage > self.performance_thresholds['memory_usage_max']:
                await self._create_performance_alert(
                    model_id, "high_memory_usage",
                    f"Memory usage {metrics.memory_usage:.1f}MB above threshold {self.performance_thresholds['memory_usage_max']}MB"
                )
                alerts_triggered.append("high_memory_usage")
            
            # Check CPU usage threshold
            if metrics.cpu_usage is not None and metrics.cpu_usage > self.performance_thresholds['cpu_usage_max']:
                await self._create_performance_alert(
                    model_id, "high_cpu_usage",
                    f"CPU usage {metrics.cpu_usage:.1f}% above threshold {self.performance_thresholds['cpu_usage_max']}%"
                )
                alerts_triggered.append("high_cpu_usage")
        
        except Exception as e:
            logger.error(f"Error checking performance alerts: {e}")
        
        return alerts_triggered
    
    async def _create_performance_alert(self, model_id: str, alert_type: str, message: str):
        """Create a performance alert."""
        alert_id = f"perf_alert_{int(time.time())}"
        
        alert = {
            'alert_id': alert_id,
            'model_id': model_id,
            'alert_type': alert_type,
            'message': message,
            'severity': 'warning',
            'timestamp': datetime.now().isoformat(),
            'resolved': False
        }
        
        self.performance_alerts[alert_id] = alert
        self.service_metrics['total_alerts'] += 1
        
        logger.warning(f"Performance alert created: {message}")
    
    async def _create_tracker(self, request: PerformanceTrackingRequest) -> Dict[str, Any]:
        """Create a new performance tracker."""
        self.tracker_counter += 1
        tracker_id = f"tracker_{self.tracker_counter}_{int(time.time())}"
        
        tracker_info = {
            'tracker_id': tracker_id,
            'model_id': request.model_id,
            'created_at': datetime.now().isoformat(),
            'status': 'active',
            'total_measurements': 0
        }
        
        self.trackers[tracker_id] = tracker_info
        
        self.service_metrics['total_trackers'] += 1
        self.service_metrics['active_trackers'] += 1
        self.service_metrics['models_tracked'] += 1
        
        return {
            'tracker_id': tracker_id,
            'model_id': request.model_id,
            'status': 'created',
            'message': 'Performance tracker created successfully',
            'timestamp': datetime.now().isoformat()
        }
    
    async def _list_trackers(self) -> Dict[str, Any]:
        """List all trackers."""
        tracker_list = []
        for tracker_id, tracker_info in self.trackers.items():
            tracker_list.append({
                'tracker_id': tracker_id,
                'model_id': tracker_info['model_id'],
                'status': tracker_info['status'],
                'total_measurements': tracker_info['total_measurements'],
                'created_at': tracker_info['created_at']
            })
        
        return {
            'trackers': tracker_list,
            'total_trackers': len(tracker_list),
            'active_trackers': sum(1 for t in tracker_list if t['status'] == 'active'),
            'timestamp': datetime.now().isoformat()
        }
    
    async def _get_tracker_status(self, tracker_id: str) -> Dict[str, Any]:
        """Get tracker status."""
        if tracker_id not in self.trackers:
            raise HTTPException(status_code=404, detail=f"Tracker {tracker_id} not found")
        
        tracker_info = self.trackers[tracker_id]
        
        return {
            'tracker_id': tracker_id,
            'model_id': tracker_info['model_id'],
            'status': tracker_info['status'],
            'total_measurements': tracker_info['total_measurements'],
            'created_at': tracker_info['created_at']
        }
    
    async def _get_model_performance(self, model_id: str, limit: int) -> Dict[str, Any]:
        """Get model performance history."""
        if model_id not in self.performance_history:
            raise HTTPException(status_code=404, detail=f"No performance data found for model {model_id}")
        
        history = self.performance_history[model_id]
        limited_history = history[-limit:] if limit > 0 else history
        
        return {
            'model_id': model_id,
            'performance_history': limited_history,
            'total_measurements': len(history),
            'timestamp': datetime.now().isoformat()
        }
    
    async def _get_performance_analytics(self, model_id: str, days: int) -> PerformanceAnalytics:
        """Get performance analytics for a model."""
        if model_id not in self.performance_history:
            raise HTTPException(status_code=404, detail=f"No performance data found for model {model_id}")
        
        history = self.performance_history[model_id]
        
        # Filter by time period
        cutoff_time = datetime.now() - timedelta(days=days)
        recent_history = [
            h for h in history 
            if datetime.fromisoformat(h['timestamp']) > cutoff_time
        ]
        
        if not recent_history:
            raise HTTPException(status_code=404, detail=f"No performance data found for the last {days} days")
        
        # Calculate analytics
        total_measurements = len(recent_history)
        
        # Calculate average accuracy
        accuracies = [h['accuracy'] for h in recent_history if h.get('accuracy') is not None]
        average_accuracy = np.mean(accuracies) if accuracies else None
        
        # Calculate accuracy trend
        accuracy_trend = await self._calculate_trend(accuracies)
        
        # Calculate average inference time
        inference_times = [h['inference_time'] for h in recent_history if h.get('inference_time') is not None]
        average_inference_time = np.mean(inference_times) if inference_times else None
        
        # Calculate throughput trend
        throughputs = [h['throughput'] for h in recent_history if h.get('throughput') is not None]
        throughput_trend = await self._calculate_trend(throughputs)
        
        # Calculate overall performance score
        performance_score = await self._calculate_performance_score(recent_history)
        
        # Generate recommendations
        recommendations = await self._generate_performance_recommendations(recent_history)
        
        return PerformanceAnalytics(
            model_id=model_id,
            time_period=f"{days} days",
            total_measurements=total_measurements,
            average_accuracy=average_accuracy,
            accuracy_trend=accuracy_trend,
            average_inference_time=average_inference_time,
            throughput_trend=throughput_trend,
            performance_score=performance_score,
            recommendations=recommendations
        )
    
    async def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend for a list of values."""
        if len(values) < 2:
            return "insufficient_data"
        
        # Simple linear trend
        x = np.arange(len(values))
        y = np.array(values)
        
        # Calculate slope
        slope = np.polyfit(x, y, 1)[0]
        
        if slope > 0.01:
            return "improving"
        elif slope < -0.01:
            return "declining"
        else:
            return "stable"
    
    async def _calculate_performance_score(self, history: List[Dict[str, Any]]) -> float:
        """Calculate overall performance score."""
        try:
            scores = []
            
            # Accuracy score
            accuracies = [h['accuracy'] for h in history if h.get('accuracy') is not None]
            if accuracies:
                scores.append(np.mean(accuracies))
            
            # F1 score
            f1_scores = [h['f1_score'] for h in history if h.get('f1_score') is not None]
            if f1_scores:
                scores.append(np.mean(f1_scores))
            
            # Inference time score (inverted - lower is better)
            inference_times = [h['inference_time'] for h in history if h.get('inference_time') is not None]
            if inference_times:
                avg_inference_time = np.mean(inference_times)
                # Normalize to 0-1 scale (assuming max acceptable time is 2000ms)
                inference_score = max(0, 1 - (avg_inference_time / 2000))
                scores.append(inference_score)
            
            return float(np.mean(scores)) if scores else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating performance score: {e}")
            return 0.0
    
    async def _generate_performance_recommendations(self, history: List[Dict[str, Any]]) -> List[str]:
        """Generate performance recommendations."""
        recommendations = []
        
        try:
            # Check accuracy
            accuracies = [h['accuracy'] for h in history if h.get('accuracy') is not None]
            if accuracies and np.mean(accuracies) < 0.8:
                recommendations.append("Consider model retraining or hyperparameter tuning to improve accuracy")
            
            # Check inference time
            inference_times = [h['inference_time'] for h in history if h.get('inference_time') is not None]
            if inference_times and np.mean(inference_times) > 1000:
                recommendations.append("Consider model optimization or hardware upgrade to reduce inference time")
            
            # Check memory usage
            memory_usage = [h['memory_usage'] for h in history if h.get('memory_usage') is not None]
            if memory_usage and np.mean(memory_usage) > 1500:
                recommendations.append("Monitor memory usage and consider model optimization")
            
            # Check CPU usage
            cpu_usage = [h['cpu_usage'] for h in history if h.get('cpu_usage') is not None]
            if cpu_usage and np.mean(cpu_usage) > 70:
                recommendations.append("Consider load balancing or hardware scaling for CPU usage")
            
            if not recommendations:
                recommendations.append("Performance metrics are within acceptable ranges")
        
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            recommendations.append("Unable to generate recommendations due to data issues")
        
        return recommendations
    
    async def _get_performance_trends(self, model_id: str, days: int) -> Dict[str, Any]:
        """Get performance trends for a model."""
        if model_id not in self.performance_history:
            raise HTTPException(status_code=404, detail=f"No performance data found for model {model_id}")
        
        history = self.performance_history[model_id]
        
        # Filter by time period
        cutoff_time = datetime.now() - timedelta(days=days)
        recent_history = [
            h for h in history 
            if datetime.fromisoformat(h['timestamp']) > cutoff_time
        ]
        
        if not recent_history:
            raise HTTPException(status_code=404, detail=f"No performance data found for the last {days} days")
        
        # Extract trends data
        timestamps = [h['timestamp'] for h in recent_history]
        accuracies = [h.get('accuracy') for h in recent_history]
        inference_times = [h.get('inference_time') for h in recent_history]
        throughputs = [h.get('throughput') for h in recent_history]
        
        return {
            'model_id': model_id,
            'time_period': f"{days} days",
            'timestamps': timestamps,
            'accuracies': accuracies,
            'inference_times': inference_times,
            'throughputs': throughputs,
            'total_measurements': len(recent_history),
            'timestamp': datetime.now().isoformat()
        }
    
    async def _list_performance_alerts(self, limit: int) -> Dict[str, Any]:
        """List performance alerts."""
        alert_list = []
        for alert_id, alert in self.performance_alerts.items():
            alert_list.append(alert)
        
        # Sort by timestamp (newest first)
        alert_list.sort(key=lambda x: x['timestamp'], reverse=True)
        limited_alerts = alert_list[:limit] if limit > 0 else alert_list
        
        return {
            'alerts': limited_alerts,
            'total_alerts': len(alert_list),
            'unresolved_alerts': sum(1 for a in alert_list if not a['resolved']),
            'timestamp': datetime.now().isoformat()
        }
    
    async def _resolve_alert(self, alert_id: str) -> Dict[str, Any]:
        """Resolve a performance alert."""
        if alert_id not in self.performance_alerts:
            raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found")
        
        self.performance_alerts[alert_id]['resolved'] = True
        
        return {
            'alert_id': alert_id,
            'status': 'resolved',
            'message': 'Alert resolved successfully',
            'timestamp': datetime.now().isoformat()
        }
    
    def run(self, host: str = "0.0.0.0", port: int = 8017):
        """Run the service."""
        logger.info(f"Starting Performance Tracker on {host}:{port}")
        uvicorn.run(self.app, host=host, port=port)


# Global service instance
service = PerformanceTracker()


def create_app() -> FastAPI:
    """Create FastAPI application."""
    return service.app


if __name__ == "__main__":
    service.run()
