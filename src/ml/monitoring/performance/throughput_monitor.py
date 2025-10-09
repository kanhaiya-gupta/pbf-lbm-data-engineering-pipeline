"""
Throughput Monitor

This module implements the throughput monitor for PBF-LB/M processes.
It provides throughput tracking, throughput analysis,
and throughput-based alerts for ML models and systems.
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
class ThroughputMetrics(BaseModel):
    """Model for throughput metrics."""
    model_id: str = Field(..., description="Model ID")
    timestamp: str = Field(..., description="Metrics timestamp")
    requests_per_second: float = Field(..., description="Requests per second")
    requests_per_minute: Optional[float] = Field(None, description="Requests per minute")
    requests_per_hour: Optional[float] = Field(None, description="Requests per hour")
    total_requests: int = Field(..., description="Total requests processed")
    successful_requests: int = Field(..., description="Successful requests")
    failed_requests: int = Field(..., description="Failed requests")
    average_response_time: Optional[float] = Field(None, description="Average response time in ms")
    concurrent_requests: Optional[int] = Field(None, description="Concurrent requests")
    batch_size: Optional[int] = Field(None, description="Average batch size")
    model_version: Optional[str] = Field(None, description="Model version")
    hardware_info: Optional[Dict[str, Any]] = Field(None, description="Hardware information")


class ThroughputMonitoringRequest(BaseModel):
    """Request model for throughput monitoring."""
    monitor_id: str = Field(..., description="Monitor ID")
    model_id: str = Field(..., description="Model ID to monitor")
    metrics: ThroughputMetrics = Field(..., description="Throughput metrics")
    baseline_throughput: Optional[float] = Field(None, description="Baseline throughput for comparison")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class ThroughputMonitoringResponse(BaseModel):
    """Response model for throughput monitoring."""
    monitor_id: str = Field(..., description="Monitor ID")
    model_id: str = Field(..., description="Model ID")
    status: str = Field(..., description="Monitoring status")
    throughput_trend: str = Field(..., description="Throughput trend")
    performance_efficiency: str = Field(..., description="Performance efficiency assessment")
    alerts_triggered: List[str] = Field(..., description="Alerts triggered")
    recommendations: List[str] = Field(..., description="Recommendations")
    timestamp: str = Field(..., description="Response timestamp")


class ThroughputAnalytics(BaseModel):
    """Model for throughput analytics."""
    model_id: str = Field(..., description="Model ID")
    time_period: str = Field(..., description="Time period for analytics")
    total_measurements: int = Field(..., description="Total number of measurements")
    average_throughput: float = Field(..., description="Average throughput (RPS)")
    peak_throughput: float = Field(..., description="Peak throughput (RPS)")
    min_throughput: float = Field(..., description="Minimum throughput (RPS)")
    throughput_std: float = Field(..., description="Throughput standard deviation")
    throughput_trend: str = Field(..., description="Throughput trend")
    success_rate: float = Field(..., description="Success rate percentage")
    total_requests: int = Field(..., description="Total requests processed")
    total_successful: int = Field(..., description="Total successful requests")
    total_failed: int = Field(..., description="Total failed requests")
    efficiency_score: float = Field(..., description="Overall efficiency score")


class ThroughputMonitor:
    """
    Throughput monitor for PBF-LB/M processes.
    
    This monitor provides comprehensive throughput monitoring capabilities for:
    - Throughput tracking and analysis
    - Performance efficiency assessment
    - Throughput-based alerts
    - Throughput analytics
    """
    
    def __init__(self, config_manager: Optional[ConfigManager] = None):
        """
        Initialize the throughput monitor.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config_manager = config_manager or ConfigManager()
        self.app = FastAPI(
            title="Throughput Monitor",
            description="Throughput monitoring for PBF-LB/M manufacturing",
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
        
        # Throughput monitoring
        self.monitors = {}  # Store monitor information
        self.throughput_history = defaultdict(list)  # Store throughput history
        self.throughput_alerts = {}  # Store throughput alerts
        self.monitor_counter = 0
        
        # Service metrics
        self.service_metrics = {
            'total_monitors': 0,
            'active_monitors': 0,
            'total_measurements': 0,
            'models_monitored': 0,
            'total_alerts': 0,
            'last_measurement_time': None
        }
        
        # Throughput thresholds
        self.throughput_thresholds = {
            'min_throughput': 1.0,  # requests per second
            'warning_throughput': 5.0,  # requests per second
            'target_throughput': 10.0,  # requests per second
            'max_throughput': 50.0,  # requests per second
            'success_rate_min': 95.0,  # percentage
            'efficiency_threshold': 0.8  # efficiency score
        }
        
        # Setup routes
        self._setup_routes()
        
        logger.info("Initialized ThroughputMonitor")
    
    def _setup_routes(self):
        """Setup API routes."""
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {
                "status": "healthy",
                "service": "throughput_monitor",
                "timestamp": datetime.now().isoformat(),
                "metrics": self.service_metrics
            }
        
        @self.app.post("/monitor", response_model=ThroughputMonitoringResponse)
        async def monitor_throughput(request: ThroughputMonitoringRequest):
            """Monitor model throughput."""
            return await self._monitor_throughput(request)
        
        @self.app.post("/monitors", response_model=Dict[str, Any])
        async def create_monitor(request: ThroughputMonitoringRequest):
            """Create a new throughput monitor."""
            return await self._create_monitor(request)
        
        @self.app.get("/monitors")
        async def list_monitors():
            """List all monitors."""
            return await self._list_monitors()
        
        @self.app.get("/monitors/{monitor_id}/status")
        async def get_monitor_status(monitor_id: str):
            """Get monitor status."""
            return await self._get_monitor_status(monitor_id)
        
        @self.app.get("/models/{model_id}/throughput")
        async def get_model_throughput(model_id: str, limit: int = Query(100, ge=1, le=1000)):
            """Get model throughput history."""
            return await self._get_model_throughput(model_id, limit)
        
        @self.app.get("/models/{model_id}/throughput-analytics", response_model=ThroughputAnalytics)
        async def get_throughput_analytics(model_id: str, days: int = Query(7, ge=1, le=30)):
            """Get throughput analytics for a model."""
            return await self._get_throughput_analytics(model_id, days)
        
        @self.app.get("/models/{model_id}/throughput-trends")
        async def get_throughput_trends(model_id: str, days: int = Query(7, ge=1, le=30)):
            """Get throughput trends for a model."""
            return await self._get_throughput_trends(model_id, days)
        
        @self.app.get("/models/{model_id}/efficiency-analysis")
        async def get_efficiency_analysis(model_id: str, days: int = Query(7, ge=1, le=30)):
            """Get efficiency analysis for a model."""
            return await self._get_efficiency_analysis(model_id, days)
        
        @self.app.get("/alerts")
        async def list_throughput_alerts(limit: int = Query(10, ge=1, le=100)):
            """List throughput alerts."""
            return await self._list_throughput_alerts(limit)
        
        @self.app.post("/alerts/{alert_id}/resolve")
        async def resolve_alert(alert_id: str):
            """Resolve a throughput alert."""
            return await self._resolve_alert(alert_id)
        
        @self.app.get("/metrics")
        async def get_metrics():
            """Get service metrics."""
            return self.service_metrics
    
    async def _monitor_throughput(self, request: ThroughputMonitoringRequest) -> ThroughputMonitoringResponse:
        """
        Monitor model throughput.
        
        Args:
            request: Throughput monitoring request
            
        Returns:
            Throughput monitoring response
        """
        try:
            # Store throughput metrics
            await self._store_throughput_metrics(request.model_id, request.metrics)
            
            # Analyze throughput trend
            throughput_trend = await self._analyze_throughput_trend(request.model_id)
            
            # Assess performance efficiency
            performance_efficiency = await self._assess_performance_efficiency(
                request.model_id, request.metrics
            )
            
            # Check for throughput alerts
            alerts_triggered = await self._check_throughput_alerts(
                request.model_id, request.metrics
            )
            
            # Generate recommendations
            recommendations = await self._generate_throughput_recommendations(
                request.model_id, request.metrics, throughput_trend, performance_efficiency
            )
            
            # Update metrics
            self.service_metrics['total_measurements'] += 1
            self.service_metrics['last_measurement_time'] = datetime.now().isoformat()
            
            return ThroughputMonitoringResponse(
                monitor_id=request.monitor_id,
                model_id=request.model_id,
                status="monitored",
                throughput_trend=throughput_trend,
                performance_efficiency=performance_efficiency,
                alerts_triggered=alerts_triggered,
                recommendations=recommendations,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"Error monitoring throughput: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def _store_throughput_metrics(self, model_id: str, metrics: ThroughputMetrics):
        """Store throughput metrics."""
        # Convert metrics to dictionary
        metrics_dict = metrics.dict()
        metrics_dict['timestamp'] = datetime.now().isoformat()
        
        # Store in history
        self.throughput_history[model_id].append(metrics_dict)
        
        # Keep only recent history (last 10000 measurements)
        if len(self.throughput_history[model_id]) > 10000:
            self.throughput_history[model_id] = self.throughput_history[model_id][-10000:]
    
    async def _analyze_throughput_trend(self, model_id: str) -> str:
        """Analyze throughput trend for a model."""
        if model_id not in self.throughput_history or len(self.throughput_history[model_id]) < 2:
            return "insufficient_data"
        
        history = self.throughput_history[model_id]
        
        # Get recent metrics (last 20 measurements)
        recent_metrics = history[-20:]
        throughputs = [m['requests_per_second'] for m in recent_metrics]
        
        if len(throughputs) < 2:
            return "insufficient_data"
        
        # Calculate trend using linear regression
        x = np.arange(len(throughputs))
        y = np.array(throughputs)
        
        # Calculate slope
        slope = np.polyfit(x, y, 1)[0]
        
        # Determine trend
        if slope > 0.5:  # 0.5 RPS increase per measurement
            return "increasing"
        elif slope < -0.5:  # 0.5 RPS decrease per measurement
            return "decreasing"
        else:
            return "stable"
    
    async def _assess_performance_efficiency(self, model_id: str, metrics: ThroughputMetrics) -> str:
        """Assess performance efficiency based on throughput."""
        throughput = metrics.requests_per_second
        success_rate = (metrics.successful_requests / max(metrics.total_requests, 1)) * 100
        
        # Calculate efficiency score
        throughput_score = min(throughput / self.throughput_thresholds['target_throughput'], 1.0)
        success_score = success_rate / 100.0
        efficiency_score = (throughput_score + success_score) / 2.0
        
        if efficiency_score >= 0.9:
            return "excellent"
        elif efficiency_score >= 0.7:
            return "good"
        elif efficiency_score >= 0.5:
            return "fair"
        else:
            return "poor"
    
    async def _check_throughput_alerts(self, model_id: str, metrics: ThroughputMetrics) -> List[str]:
        """Check for throughput alerts."""
        alerts_triggered = []
        
        try:
            throughput = metrics.requests_per_second
            success_rate = (metrics.successful_requests / max(metrics.total_requests, 1)) * 100
            
            # Check minimum throughput threshold
            if throughput < self.throughput_thresholds['min_throughput']:
                await self._create_throughput_alert(
                    model_id, "low_throughput",
                    f"Low throughput detected: {throughput:.2f} RPS (threshold: {self.throughput_thresholds['min_throughput']} RPS)"
                )
                alerts_triggered.append("low_throughput")
            
            # Check warning throughput threshold
            elif throughput < self.throughput_thresholds['warning_throughput']:
                await self._create_throughput_alert(
                    model_id, "warning_throughput",
                    f"Warning throughput detected: {throughput:.2f} RPS (threshold: {self.throughput_thresholds['warning_throughput']} RPS)"
                )
                alerts_triggered.append("warning_throughput")
            
            # Check success rate threshold
            if success_rate < self.throughput_thresholds['success_rate_min']:
                await self._create_throughput_alert(
                    model_id, "low_success_rate",
                    f"Low success rate detected: {success_rate:.1f}% (threshold: {self.throughput_thresholds['success_rate_min']}%)"
                )
                alerts_triggered.append("low_success_rate")
            
            # Check for throughput spikes
            if model_id in self.throughput_history and len(self.throughput_history[model_id]) >= 10:
                recent_throughputs = [h['requests_per_second'] for h in self.throughput_history[model_id][-10:]]
                avg_throughput = np.mean(recent_throughputs[:-1])  # Exclude current measurement
                
                if avg_throughput > 0 and throughput > avg_throughput * 2.0:
                    await self._create_throughput_alert(
                        model_id, "throughput_spike",
                        f"Throughput spike detected: {throughput:.2f} RPS (average: {avg_throughput:.2f} RPS)"
                    )
                    alerts_triggered.append("throughput_spike")
            
            # Check for throughput drops
            if model_id in self.throughput_history and len(self.throughput_history[model_id]) >= 10:
                recent_throughputs = [h['requests_per_second'] for h in self.throughput_history[model_id][-10:]]
                avg_throughput = np.mean(recent_throughputs[:-1])  # Exclude current measurement
                
                if avg_throughput > 0 and throughput < avg_throughput * 0.5:
                    await self._create_throughput_alert(
                        model_id, "throughput_drop",
                        f"Throughput drop detected: {throughput:.2f} RPS (average: {avg_throughput:.2f} RPS)"
                    )
                    alerts_triggered.append("throughput_drop")
            
            # Check efficiency threshold
            efficiency_score = await self._calculate_efficiency_score(metrics)
            if efficiency_score < self.throughput_thresholds['efficiency_threshold']:
                await self._create_throughput_alert(
                    model_id, "low_efficiency",
                    f"Low efficiency detected: {efficiency_score:.2f} (threshold: {self.throughput_thresholds['efficiency_threshold']})"
                )
                alerts_triggered.append("low_efficiency")
        
        except Exception as e:
            logger.error(f"Error checking throughput alerts: {e}")
        
        return alerts_triggered
    
    async def _calculate_efficiency_score(self, metrics: ThroughputMetrics) -> float:
        """Calculate efficiency score."""
        throughput_score = min(metrics.requests_per_second / self.throughput_thresholds['target_throughput'], 1.0)
        success_rate = (metrics.successful_requests / max(metrics.total_requests, 1))
        
        # Consider response time if available
        response_time_score = 1.0
        if metrics.average_response_time is not None:
            # Assume optimal response time is 100ms
            response_time_score = max(0, 1 - (metrics.average_response_time / 1000))
        
        efficiency_score = (throughput_score + success_rate + response_time_score) / 3.0
        return efficiency_score
    
    async def _generate_throughput_recommendations(self, model_id: str, metrics: ThroughputMetrics,
                                                 throughput_trend: str, performance_efficiency: str) -> List[str]:
        """Generate throughput recommendations."""
        recommendations = []
        
        try:
            throughput = metrics.requests_per_second
            success_rate = (metrics.successful_requests / max(metrics.total_requests, 1)) * 100
            
            # Low throughput recommendations
            if throughput < self.throughput_thresholds['warning_throughput']:
                recommendations.append("Low throughput detected - consider optimization")
                recommendations.append("Review model architecture and inference pipeline")
                recommendations.append("Consider batch processing or parallel processing")
                recommendations.append("Evaluate hardware resources and scaling")
            
            # Low success rate recommendations
            if success_rate < self.throughput_thresholds['success_rate_min']:
                recommendations.append("Low success rate detected - investigate failures")
                recommendations.append("Review error handling and input validation")
                recommendations.append("Check for resource constraints or timeouts")
            
            # Trend-based recommendations
            if throughput_trend == "decreasing":
                recommendations.append("Throughput is decreasing - investigate root cause")
                recommendations.append("Check for resource constraints or bottlenecks")
                recommendations.append("Monitor system health and performance")
            elif throughput_trend == "increasing":
                recommendations.append("Throughput is improving - continue monitoring")
            
            # Efficiency-based recommendations
            if performance_efficiency == "poor":
                recommendations.append("Poor performance efficiency - comprehensive optimization needed")
                recommendations.append("Consider model optimization, hardware upgrade, or architecture changes")
            elif performance_efficiency == "fair":
                recommendations.append("Fair performance efficiency - optimization recommended")
                recommendations.append("Review and optimize bottlenecks in the pipeline")
            
            # Batch size recommendations
            if metrics.batch_size is not None:
                if metrics.batch_size == 1 and throughput < 5:
                    recommendations.append("Consider batch processing to improve throughput")
                elif metrics.batch_size > 1 and success_rate < 90:
                    recommendations.append("Consider reducing batch size to improve success rate")
            
            # Hardware recommendations
            if metrics.hardware_info:
                cpu_usage = metrics.hardware_info.get('cpu_usage')
                memory_usage = metrics.hardware_info.get('memory_usage')
                
                if cpu_usage and cpu_usage > 80:
                    recommendations.append("High CPU usage detected - consider CPU optimization or scaling")
                
                if memory_usage and memory_usage > 80:
                    recommendations.append("High memory usage detected - consider memory optimization")
            
            # General recommendations
            if not recommendations:
                recommendations.append("Throughput is within acceptable ranges")
                recommendations.append("Continue monitoring for performance changes")
        
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            recommendations.append("Unable to generate recommendations due to data issues")
        
        return recommendations
    
    async def _create_throughput_alert(self, model_id: str, alert_type: str, message: str):
        """Create a throughput alert."""
        alert_id = f"tput_alert_{int(time.time())}"
        
        # Determine severity based on alert type
        severity = "critical" if alert_type in ["low_throughput", "low_success_rate"] else "warning"
        
        alert = {
            'alert_id': alert_id,
            'model_id': model_id,
            'alert_type': alert_type,
            'message': message,
            'severity': severity,
            'timestamp': datetime.now().isoformat(),
            'resolved': False
        }
        
        self.throughput_alerts[alert_id] = alert
        self.service_metrics['total_alerts'] += 1
        
        logger.warning(f"Throughput alert created: {message}")
    
    async def _create_monitor(self, request: ThroughputMonitoringRequest) -> Dict[str, Any]:
        """Create a new throughput monitor."""
        self.monitor_counter += 1
        monitor_id = f"monitor_{self.monitor_counter}_{int(time.time())}"
        
        monitor_info = {
            'monitor_id': monitor_id,
            'model_id': request.model_id,
            'baseline_throughput': request.baseline_throughput,
            'created_at': datetime.now().isoformat(),
            'status': 'active',
            'total_measurements': 0
        }
        
        self.monitors[monitor_id] = monitor_info
        
        self.service_metrics['total_monitors'] += 1
        self.service_metrics['active_monitors'] += 1
        self.service_metrics['models_monitored'] += 1
        
        return {
            'monitor_id': monitor_id,
            'model_id': request.model_id,
            'status': 'created',
            'message': 'Throughput monitor created successfully',
            'timestamp': datetime.now().isoformat()
        }
    
    async def _list_monitors(self) -> Dict[str, Any]:
        """List all monitors."""
        monitor_list = []
        for monitor_id, monitor_info in self.monitors.items():
            monitor_list.append({
                'monitor_id': monitor_id,
                'model_id': monitor_info['model_id'],
                'baseline_throughput': monitor_info['baseline_throughput'],
                'status': monitor_info['status'],
                'total_measurements': monitor_info['total_measurements'],
                'created_at': monitor_info['created_at']
            })
        
        return {
            'monitors': monitor_list,
            'total_monitors': len(monitor_list),
            'active_monitors': sum(1 for m in monitor_list if m['status'] == 'active'),
            'timestamp': datetime.now().isoformat()
        }
    
    async def _get_monitor_status(self, monitor_id: str) -> Dict[str, Any]:
        """Get monitor status."""
        if monitor_id not in self.monitors:
            raise HTTPException(status_code=404, detail=f"Monitor {monitor_id} not found")
        
        monitor_info = self.monitors[monitor_id]
        
        return {
            'monitor_id': monitor_id,
            'model_id': monitor_info['model_id'],
            'baseline_throughput': monitor_info['baseline_throughput'],
            'status': monitor_info['status'],
            'total_measurements': monitor_info['total_measurements'],
            'created_at': monitor_info['created_at']
        }
    
    async def _get_model_throughput(self, model_id: str, limit: int) -> Dict[str, Any]:
        """Get model throughput history."""
        if model_id not in self.throughput_history:
            raise HTTPException(status_code=404, detail=f"No throughput data found for model {model_id}")
        
        history = self.throughput_history[model_id]
        limited_history = history[-limit:] if limit > 0 else history
        
        return {
            'model_id': model_id,
            'throughput_history': limited_history,
            'total_measurements': len(history),
            'timestamp': datetime.now().isoformat()
        }
    
    async def _get_throughput_analytics(self, model_id: str, days: int) -> ThroughputAnalytics:
        """Get throughput analytics for a model."""
        if model_id not in self.throughput_history:
            raise HTTPException(status_code=404, detail=f"No throughput data found for model {model_id}")
        
        history = self.throughput_history[model_id]
        
        # Filter by time period
        cutoff_time = datetime.now() - timedelta(days=days)
        recent_history = [
            h for h in history 
            if datetime.fromisoformat(h['timestamp']) > cutoff_time
        ]
        
        if not recent_history:
            raise HTTPException(status_code=404, detail=f"No throughput data found for the last {days} days")
        
        # Calculate analytics
        total_measurements = len(recent_history)
        throughputs = [h['requests_per_second'] for h in recent_history]
        
        average_throughput = np.mean(throughputs)
        peak_throughput = np.max(throughputs)
        min_throughput = np.min(throughputs)
        throughput_std = np.std(throughputs)
        
        # Calculate throughput trend
        throughput_trend = await self._analyze_throughput_trend(model_id)
        
        # Calculate success metrics
        total_requests = sum(h['total_requests'] for h in recent_history)
        total_successful = sum(h['successful_requests'] for h in recent_history)
        total_failed = sum(h['failed_requests'] for h in recent_history)
        success_rate = (total_successful / max(total_requests, 1)) * 100
        
        # Calculate efficiency score
        efficiency_scores = [await self._calculate_efficiency_score(ThroughputMetrics(**h)) for h in recent_history]
        efficiency_score = np.mean(efficiency_scores)
        
        return ThroughputAnalytics(
            model_id=model_id,
            time_period=f"{days} days",
            total_measurements=total_measurements,
            average_throughput=average_throughput,
            peak_throughput=peak_throughput,
            min_throughput=min_throughput,
            throughput_std=throughput_std,
            throughput_trend=throughput_trend,
            success_rate=success_rate,
            total_requests=total_requests,
            total_successful=total_successful,
            total_failed=total_failed,
            efficiency_score=efficiency_score
        )
    
    async def _get_throughput_trends(self, model_id: str, days: int) -> Dict[str, Any]:
        """Get throughput trends for a model."""
        if model_id not in self.throughput_history:
            raise HTTPException(status_code=404, detail=f"No throughput data found for model {model_id}")
        
        history = self.throughput_history[model_id]
        
        # Filter by time period
        cutoff_time = datetime.now() - timedelta(days=days)
        recent_history = [
            h for h in history 
            if datetime.fromisoformat(h['timestamp']) > cutoff_time
        ]
        
        if not recent_history:
            raise HTTPException(status_code=404, detail=f"No throughput data found for the last {days} days")
        
        # Extract trends data
        timestamps = [h['timestamp'] for h in recent_history]
        rps = [h['requests_per_second'] for h in recent_history]
        rpm = [h.get('requests_per_minute') for h in recent_history]
        rph = [h.get('requests_per_hour') for h in recent_history]
        total_requests = [h['total_requests'] for h in recent_history]
        successful_requests = [h['successful_requests'] for h in recent_history]
        failed_requests = [h['failed_requests'] for h in recent_history]
        response_times = [h.get('average_response_time') for h in recent_history]
        concurrent_requests = [h.get('concurrent_requests') for h in recent_history]
        batch_sizes = [h.get('batch_size') for h in recent_history]
        
        return {
            'model_id': model_id,
            'time_period': f"{days} days",
            'timestamps': timestamps,
            'requests_per_second': rps,
            'requests_per_minute': rpm,
            'requests_per_hour': rph,
            'total_requests': total_requests,
            'successful_requests': successful_requests,
            'failed_requests': failed_requests,
            'response_times': response_times,
            'concurrent_requests': concurrent_requests,
            'batch_sizes': batch_sizes,
            'total_measurements': len(recent_history),
            'timestamp': datetime.now().isoformat()
        }
    
    async def _get_efficiency_analysis(self, model_id: str, days: int) -> Dict[str, Any]:
        """Get efficiency analysis for a model."""
        if model_id not in self.throughput_history:
            raise HTTPException(status_code=404, detail=f"No throughput data found for model {model_id}")
        
        history = self.throughput_history[model_id]
        
        # Filter by time period
        cutoff_time = datetime.now() - timedelta(days=days)
        recent_history = [
            h for h in history 
            if datetime.fromisoformat(h['timestamp']) > cutoff_time
        ]
        
        if not recent_history:
            raise HTTPException(status_code=404, detail=f"No throughput data found for the last {days} days")
        
        # Calculate efficiency metrics
        efficiency_scores = []
        throughput_scores = []
        success_scores = []
        response_time_scores = []
        
        for h in recent_history:
            metrics = ThroughputMetrics(**h)
            efficiency_score = await self._calculate_efficiency_score(metrics)
            efficiency_scores.append(efficiency_score)
            
            # Individual component scores
            throughput_score = min(metrics.requests_per_second / self.throughput_thresholds['target_throughput'], 1.0)
            throughput_scores.append(throughput_score)
            
            success_score = metrics.successful_requests / max(metrics.total_requests, 1)
            success_scores.append(success_score)
            
            if metrics.average_response_time is not None:
                response_time_score = max(0, 1 - (metrics.average_response_time / 1000))
                response_time_scores.append(response_time_score)
        
        # Calculate efficiency categories
        efficiency_categories = {
            'excellent': 0,
            'good': 0,
            'fair': 0,
            'poor': 0
        }
        
        for score in efficiency_scores:
            if score >= 0.9:
                efficiency_categories['excellent'] += 1
            elif score >= 0.7:
                efficiency_categories['good'] += 1
            elif score >= 0.5:
                efficiency_categories['fair'] += 1
            else:
                efficiency_categories['poor'] += 1
        
        # Calculate percentages
        total_measurements = len(efficiency_scores)
        efficiency_percentages = {
            category: (count / total_measurements) * 100 
            for category, count in efficiency_categories.items()
        }
        
        return {
            'model_id': model_id,
            'time_period': f"{days} days",
            'total_measurements': total_measurements,
            'average_efficiency_score': np.mean(efficiency_scores),
            'efficiency_categories': efficiency_categories,
            'efficiency_percentages': efficiency_percentages,
            'component_scores': {
                'average_throughput_score': np.mean(throughput_scores),
                'average_success_score': np.mean(success_scores),
                'average_response_time_score': np.mean(response_time_scores) if response_time_scores else None
            },
            'timestamp': datetime.now().isoformat()
        }
    
    async def _list_throughput_alerts(self, limit: int) -> Dict[str, Any]:
        """List throughput alerts."""
        alert_list = []
        for alert_id, alert in self.throughput_alerts.items():
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
        """Resolve a throughput alert."""
        if alert_id not in self.throughput_alerts:
            raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found")
        
        self.throughput_alerts[alert_id]['resolved'] = True
        
        return {
            'alert_id': alert_id,
            'status': 'resolved',
            'message': 'Alert resolved successfully',
            'timestamp': datetime.now().isoformat()
        }
    
    def run(self, host: str = "0.0.0.0", port: int = 8020):
        """Run the service."""
        logger.info(f"Starting Throughput Monitor on {host}:{port}")
        uvicorn.run(self.app, host=host, port=port)


# Global service instance
service = ThroughputMonitor()


def create_app() -> FastAPI:
    """Create FastAPI application."""
    return service.app


if __name__ == "__main__":
    service.run()
