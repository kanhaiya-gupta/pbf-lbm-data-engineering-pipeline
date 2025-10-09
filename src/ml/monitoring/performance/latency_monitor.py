"""
Latency Monitor

This module implements the latency monitor for PBF-LB/M processes.
It provides latency tracking, latency analysis,
and latency-based alerts for ML models and systems.
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
class LatencyMetrics(BaseModel):
    """Model for latency metrics."""
    model_id: str = Field(..., description="Model ID")
    timestamp: str = Field(..., description="Metrics timestamp")
    inference_time: float = Field(..., description="Inference time in milliseconds")
    preprocessing_time: Optional[float] = Field(None, description="Preprocessing time in milliseconds")
    postprocessing_time: Optional[float] = Field(None, description="Postprocessing time in milliseconds")
    total_time: Optional[float] = Field(None, description="Total processing time in milliseconds")
    request_size: Optional[int] = Field(None, description="Request size in bytes")
    response_size: Optional[int] = Field(None, description="Response size in bytes")
    batch_size: Optional[int] = Field(None, description="Batch size")
    model_version: Optional[str] = Field(None, description="Model version")
    hardware_info: Optional[Dict[str, Any]] = Field(None, description="Hardware information")


class LatencyMonitoringRequest(BaseModel):
    """Request model for latency monitoring."""
    monitor_id: str = Field(..., description="Monitor ID")
    model_id: str = Field(..., description="Model ID to monitor")
    metrics: LatencyMetrics = Field(..., description="Latency metrics")
    baseline_latency: Optional[float] = Field(None, description="Baseline latency for comparison")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class LatencyMonitoringResponse(BaseModel):
    """Response model for latency monitoring."""
    monitor_id: str = Field(..., description="Monitor ID")
    model_id: str = Field(..., description="Model ID")
    status: str = Field(..., description="Monitoring status")
    latency_trend: str = Field(..., description="Latency trend")
    performance_impact: str = Field(..., description="Performance impact assessment")
    alerts_triggered: List[str] = Field(..., description="Alerts triggered")
    recommendations: List[str] = Field(..., description="Recommendations")
    timestamp: str = Field(..., description="Response timestamp")


class LatencyAnalytics(BaseModel):
    """Model for latency analytics."""
    model_id: str = Field(..., description="Model ID")
    time_period: str = Field(..., description="Time period for analytics")
    total_measurements: int = Field(..., description="Total number of measurements")
    average_latency: float = Field(..., description="Average latency in milliseconds")
    median_latency: float = Field(..., description="Median latency in milliseconds")
    p95_latency: float = Field(..., description="95th percentile latency")
    p99_latency: float = Field(..., description="99th percentile latency")
    min_latency: float = Field(..., description="Minimum latency")
    max_latency: float = Field(..., description="Maximum latency")
    latency_std: float = Field(..., description="Latency standard deviation")
    latency_trend: str = Field(..., description="Latency trend")
    throughput: float = Field(..., description="Average throughput (requests/second)")


class LatencyMonitor:
    """
    Latency monitor for PBF-LB/M processes.
    
    This monitor provides comprehensive latency monitoring capabilities for:
    - Latency tracking and analysis
    - Performance impact assessment
    - Latency-based alerts
    - Latency analytics
    """
    
    def __init__(self, config_manager: Optional[ConfigManager] = None):
        """
        Initialize the latency monitor.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config_manager = config_manager or ConfigManager()
        self.app = FastAPI(
            title="Latency Monitor",
            description="Latency monitoring for PBF-LB/M manufacturing",
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
        
        # Latency monitoring
        self.monitors = {}  # Store monitor information
        self.latency_history = defaultdict(list)  # Store latency history
        self.latency_alerts = {}  # Store latency alerts
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
        
        # Latency thresholds
        self.latency_thresholds = {
            'max_latency': 1000,  # milliseconds
            'warning_latency': 500,  # milliseconds
            'critical_latency': 2000,  # milliseconds
            'p95_threshold': 800,  # milliseconds
            'p99_threshold': 1500,  # milliseconds
            'latency_spike_threshold': 2.0  # multiplier for spike detection
        }
        
        # Setup routes
        self._setup_routes()
        
        logger.info("Initialized LatencyMonitor")
    
    def _setup_routes(self):
        """Setup API routes."""
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {
                "status": "healthy",
                "service": "latency_monitor",
                "timestamp": datetime.now().isoformat(),
                "metrics": self.service_metrics
            }
        
        @self.app.post("/monitor", response_model=LatencyMonitoringResponse)
        async def monitor_latency(request: LatencyMonitoringRequest):
            """Monitor model latency."""
            return await self._monitor_latency(request)
        
        @self.app.post("/monitors", response_model=Dict[str, Any])
        async def create_monitor(request: LatencyMonitoringRequest):
            """Create a new latency monitor."""
            return await self._create_monitor(request)
        
        @self.app.get("/monitors")
        async def list_monitors():
            """List all monitors."""
            return await self._list_monitors()
        
        @self.app.get("/monitors/{monitor_id}/status")
        async def get_monitor_status(monitor_id: str):
            """Get monitor status."""
            return await self._get_monitor_status(monitor_id)
        
        @self.app.get("/models/{model_id}/latency")
        async def get_model_latency(model_id: str, limit: int = Query(100, ge=1, le=1000)):
            """Get model latency history."""
            return await self._get_model_latency(model_id, limit)
        
        @self.app.get("/models/{model_id}/latency-analytics", response_model=LatencyAnalytics)
        async def get_latency_analytics(model_id: str, days: int = Query(7, ge=1, le=30)):
            """Get latency analytics for a model."""
            return await self._get_latency_analytics(model_id, days)
        
        @self.app.get("/models/{model_id}/latency-trends")
        async def get_latency_trends(model_id: str, days: int = Query(7, ge=1, le=30)):
            """Get latency trends for a model."""
            return await self._get_latency_trends(model_id, days)
        
        @self.app.get("/models/{model_id}/performance-impact")
        async def get_performance_impact(model_id: str, days: int = Query(7, ge=1, le=30)):
            """Get performance impact analysis for a model."""
            return await self._get_performance_impact(model_id, days)
        
        @self.app.get("/alerts")
        async def list_latency_alerts(limit: int = Query(10, ge=1, le=100)):
            """List latency alerts."""
            return await self._list_latency_alerts(limit)
        
        @self.app.post("/alerts/{alert_id}/resolve")
        async def resolve_alert(alert_id: str):
            """Resolve a latency alert."""
            return await self._resolve_alert(alert_id)
        
        @self.app.get("/metrics")
        async def get_metrics():
            """Get service metrics."""
            return self.service_metrics
    
    async def _monitor_latency(self, request: LatencyMonitoringRequest) -> LatencyMonitoringResponse:
        """
        Monitor model latency.
        
        Args:
            request: Latency monitoring request
            
        Returns:
            Latency monitoring response
        """
        try:
            # Store latency metrics
            await self._store_latency_metrics(request.model_id, request.metrics)
            
            # Analyze latency trend
            latency_trend = await self._analyze_latency_trend(request.model_id)
            
            # Assess performance impact
            performance_impact = await self._assess_performance_impact(
                request.model_id, request.metrics
            )
            
            # Check for latency alerts
            alerts_triggered = await self._check_latency_alerts(
                request.model_id, request.metrics
            )
            
            # Generate recommendations
            recommendations = await self._generate_latency_recommendations(
                request.model_id, request.metrics, latency_trend, performance_impact
            )
            
            # Update metrics
            self.service_metrics['total_measurements'] += 1
            self.service_metrics['last_measurement_time'] = datetime.now().isoformat()
            
            return LatencyMonitoringResponse(
                monitor_id=request.monitor_id,
                model_id=request.model_id,
                status="monitored",
                latency_trend=latency_trend,
                performance_impact=performance_impact,
                alerts_triggered=alerts_triggered,
                recommendations=recommendations,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"Error monitoring latency: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def _store_latency_metrics(self, model_id: str, metrics: LatencyMetrics):
        """Store latency metrics."""
        # Convert metrics to dictionary
        metrics_dict = metrics.dict()
        metrics_dict['timestamp'] = datetime.now().isoformat()
        
        # Store in history
        self.latency_history[model_id].append(metrics_dict)
        
        # Keep only recent history (last 10000 measurements)
        if len(self.latency_history[model_id]) > 10000:
            self.latency_history[model_id] = self.latency_history[model_id][-10000:]
    
    async def _analyze_latency_trend(self, model_id: str) -> str:
        """Analyze latency trend for a model."""
        if model_id not in self.latency_history or len(self.latency_history[model_id]) < 2:
            return "insufficient_data"
        
        history = self.latency_history[model_id]
        
        # Get recent metrics (last 20 measurements)
        recent_metrics = history[-20:]
        latencies = [m['inference_time'] for m in recent_metrics]
        
        if len(latencies) < 2:
            return "insufficient_data"
        
        # Calculate trend using linear regression
        x = np.arange(len(latencies))
        y = np.array(latencies)
        
        # Calculate slope
        slope = np.polyfit(x, y, 1)[0]
        
        # Determine trend
        if slope > 10:  # 10ms increase per measurement
            return "increasing"
        elif slope < -10:  # 10ms decrease per measurement
            return "decreasing"
        else:
            return "stable"
    
    async def _assess_performance_impact(self, model_id: str, metrics: LatencyMetrics) -> str:
        """Assess performance impact based on latency."""
        latency = metrics.inference_time
        
        if latency <= self.latency_thresholds['warning_latency']:
            return "minimal"
        elif latency <= self.latency_thresholds['max_latency']:
            return "moderate"
        elif latency <= self.latency_thresholds['critical_latency']:
            return "significant"
        else:
            return "critical"
    
    async def _check_latency_alerts(self, model_id: str, metrics: LatencyMetrics) -> List[str]:
        """Check for latency alerts."""
        alerts_triggered = []
        
        try:
            latency = metrics.inference_time
            
            # Check critical latency threshold
            if latency > self.latency_thresholds['critical_latency']:
                await self._create_latency_alert(
                    model_id, "critical_latency",
                    f"Critical latency detected: {latency:.1f}ms (threshold: {self.latency_thresholds['critical_latency']}ms)"
                )
                alerts_triggered.append("critical_latency")
            
            # Check maximum latency threshold
            elif latency > self.latency_thresholds['max_latency']:
                await self._create_latency_alert(
                    model_id, "high_latency",
                    f"High latency detected: {latency:.1f}ms (threshold: {self.latency_thresholds['max_latency']}ms)"
                )
                alerts_triggered.append("high_latency")
            
            # Check warning latency threshold
            elif latency > self.latency_thresholds['warning_latency']:
                await self._create_latency_alert(
                    model_id, "warning_latency",
                    f"Warning latency detected: {latency:.1f}ms (threshold: {self.latency_thresholds['warning_latency']}ms)"
                )
                alerts_triggered.append("warning_latency")
            
            # Check for latency spikes
            if model_id in self.latency_history and len(self.latency_history[model_id]) >= 10:
                recent_latencies = [h['inference_time'] for h in self.latency_history[model_id][-10:]]
                avg_latency = np.mean(recent_latencies[:-1])  # Exclude current measurement
                
                if avg_latency > 0 and latency > avg_latency * self.latency_thresholds['latency_spike_threshold']:
                    await self._create_latency_alert(
                        model_id, "latency_spike",
                        f"Latency spike detected: {latency:.1f}ms (average: {avg_latency:.1f}ms)"
                    )
                    alerts_triggered.append("latency_spike")
            
            # Check P95 and P99 thresholds if we have enough data
            if model_id in self.latency_history and len(self.latency_history[model_id]) >= 100:
                recent_latencies = [h['inference_time'] for h in self.latency_history[model_id][-100:]]
                p95_latency = np.percentile(recent_latencies, 95)
                p99_latency = np.percentile(recent_latencies, 99)
                
                if p95_latency > self.latency_thresholds['p95_threshold']:
                    await self._create_latency_alert(
                        model_id, "high_p95_latency",
                        f"High P95 latency: {p95_latency:.1f}ms (threshold: {self.latency_thresholds['p95_threshold']}ms)"
                    )
                    alerts_triggered.append("high_p95_latency")
                
                if p99_latency > self.latency_thresholds['p99_threshold']:
                    await self._create_latency_alert(
                        model_id, "high_p99_latency",
                        f"High P99 latency: {p99_latency:.1f}ms (threshold: {self.latency_thresholds['p99_threshold']}ms)"
                    )
                    alerts_triggered.append("high_p99_latency")
        
        except Exception as e:
            logger.error(f"Error checking latency alerts: {e}")
        
        return alerts_triggered
    
    async def _generate_latency_recommendations(self, model_id: str, metrics: LatencyMetrics,
                                             latency_trend: str, performance_impact: str) -> List[str]:
        """Generate latency recommendations."""
        recommendations = []
        
        try:
            latency = metrics.inference_time
            
            # Critical latency recommendations
            if performance_impact == "critical":
                recommendations.append("Critical latency detected - immediate optimization required")
                recommendations.append("Consider model quantization or pruning")
                recommendations.append("Evaluate hardware upgrade or load balancing")
            
            # High latency recommendations
            elif performance_impact == "significant":
                recommendations.append("Significant latency impact - optimization recommended")
                recommendations.append("Consider batch processing or model optimization")
                recommendations.append("Review preprocessing and postprocessing steps")
            
            # Moderate latency recommendations
            elif performance_impact == "moderate":
                recommendations.append("Monitor latency closely and consider optimization")
                recommendations.append("Review model architecture and hyperparameters")
            
            # Trend-based recommendations
            if latency_trend == "increasing":
                recommendations.append("Latency is increasing - investigate root cause")
                recommendations.append("Check for resource constraints or data drift")
            elif latency_trend == "decreasing":
                recommendations.append("Latency is improving - continue monitoring")
            
            # Batch size recommendations
            if metrics.batch_size is not None:
                if metrics.batch_size == 1 and latency > 100:
                    recommendations.append("Consider batch processing to improve throughput")
                elif metrics.batch_size > 1 and latency > 500:
                    recommendations.append("Consider reducing batch size to improve latency")
            
            # Hardware recommendations
            if metrics.hardware_info:
                cpu_usage = metrics.hardware_info.get('cpu_usage')
                memory_usage = metrics.hardware_info.get('memory_usage')
                
                if cpu_usage and cpu_usage > 80:
                    recommendations.append("High CPU usage detected - consider CPU optimization or upgrade")
                
                if memory_usage and memory_usage > 80:
                    recommendations.append("High memory usage detected - consider memory optimization")
            
            # General recommendations
            if not recommendations:
                recommendations.append("Latency is within acceptable ranges")
                recommendations.append("Continue monitoring for performance changes")
        
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            recommendations.append("Unable to generate recommendations due to data issues")
        
        return recommendations
    
    async def _create_latency_alert(self, model_id: str, alert_type: str, message: str):
        """Create a latency alert."""
        alert_id = f"lat_alert_{int(time.time())}"
        
        # Determine severity based on alert type
        severity = "critical" if alert_type in ["critical_latency", "high_p99_latency"] else "warning"
        
        alert = {
            'alert_id': alert_id,
            'model_id': model_id,
            'alert_type': alert_type,
            'message': message,
            'severity': severity,
            'timestamp': datetime.now().isoformat(),
            'resolved': False
        }
        
        self.latency_alerts[alert_id] = alert
        self.service_metrics['total_alerts'] += 1
        
        logger.warning(f"Latency alert created: {message}")
    
    async def _create_monitor(self, request: LatencyMonitoringRequest) -> Dict[str, Any]:
        """Create a new latency monitor."""
        self.monitor_counter += 1
        monitor_id = f"monitor_{self.monitor_counter}_{int(time.time())}"
        
        monitor_info = {
            'monitor_id': monitor_id,
            'model_id': request.model_id,
            'baseline_latency': request.baseline_latency,
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
            'message': 'Latency monitor created successfully',
            'timestamp': datetime.now().isoformat()
        }
    
    async def _list_monitors(self) -> Dict[str, Any]:
        """List all monitors."""
        monitor_list = []
        for monitor_id, monitor_info in self.monitors.items():
            monitor_list.append({
                'monitor_id': monitor_id,
                'model_id': monitor_info['model_id'],
                'baseline_latency': monitor_info['baseline_latency'],
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
            'baseline_latency': monitor_info['baseline_latency'],
            'status': monitor_info['status'],
            'total_measurements': monitor_info['total_measurements'],
            'created_at': monitor_info['created_at']
        }
    
    async def _get_model_latency(self, model_id: str, limit: int) -> Dict[str, Any]:
        """Get model latency history."""
        if model_id not in self.latency_history:
            raise HTTPException(status_code=404, detail=f"No latency data found for model {model_id}")
        
        history = self.latency_history[model_id]
        limited_history = history[-limit:] if limit > 0 else history
        
        return {
            'model_id': model_id,
            'latency_history': limited_history,
            'total_measurements': len(history),
            'timestamp': datetime.now().isoformat()
        }
    
    async def _get_latency_analytics(self, model_id: str, days: int) -> LatencyAnalytics:
        """Get latency analytics for a model."""
        if model_id not in self.latency_history:
            raise HTTPException(status_code=404, detail=f"No latency data found for model {model_id}")
        
        history = self.latency_history[model_id]
        
        # Filter by time period
        cutoff_time = datetime.now() - timedelta(days=days)
        recent_history = [
            h for h in history 
            if datetime.fromisoformat(h['timestamp']) > cutoff_time
        ]
        
        if not recent_history:
            raise HTTPException(status_code=404, detail=f"No latency data found for the last {days} days")
        
        # Calculate analytics
        total_measurements = len(recent_history)
        latencies = [h['inference_time'] for h in recent_history]
        
        average_latency = np.mean(latencies)
        median_latency = np.median(latencies)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)
        min_latency = np.min(latencies)
        max_latency = np.max(latencies)
        latency_std = np.std(latencies)
        
        # Calculate latency trend
        latency_trend = await self._analyze_latency_trend(model_id)
        
        # Calculate throughput (requests per second)
        if total_measurements > 1:
            time_span = (datetime.fromisoformat(recent_history[-1]['timestamp']) - 
                        datetime.fromisoformat(recent_history[0]['timestamp'])).total_seconds()
            throughput = total_measurements / max(time_span, 1)
        else:
            throughput = 0.0
        
        return LatencyAnalytics(
            model_id=model_id,
            time_period=f"{days} days",
            total_measurements=total_measurements,
            average_latency=average_latency,
            median_latency=median_latency,
            p95_latency=p95_latency,
            p99_latency=p99_latency,
            min_latency=min_latency,
            max_latency=max_latency,
            latency_std=latency_std,
            latency_trend=latency_trend,
            throughput=throughput
        )
    
    async def _get_latency_trends(self, model_id: str, days: int) -> Dict[str, Any]:
        """Get latency trends for a model."""
        if model_id not in self.latency_history:
            raise HTTPException(status_code=404, detail=f"No latency data found for model {model_id}")
        
        history = self.latency_history[model_id]
        
        # Filter by time period
        cutoff_time = datetime.now() - timedelta(days=days)
        recent_history = [
            h for h in history 
            if datetime.fromisoformat(h['timestamp']) > cutoff_time
        ]
        
        if not recent_history:
            raise HTTPException(status_code=404, detail=f"No latency data found for the last {days} days")
        
        # Extract trends data
        timestamps = [h['timestamp'] for h in recent_history]
        inference_times = [h['inference_time'] for h in recent_history]
        preprocessing_times = [h.get('preprocessing_time') for h in recent_history]
        postprocessing_times = [h.get('postprocessing_time') for h in recent_history]
        total_times = [h.get('total_time') for h in recent_history]
        batch_sizes = [h.get('batch_size') for h in recent_history]
        
        return {
            'model_id': model_id,
            'time_period': f"{days} days",
            'timestamps': timestamps,
            'inference_times': inference_times,
            'preprocessing_times': preprocessing_times,
            'postprocessing_times': postprocessing_times,
            'total_times': total_times,
            'batch_sizes': batch_sizes,
            'total_measurements': len(recent_history),
            'timestamp': datetime.now().isoformat()
        }
    
    async def _get_performance_impact(self, model_id: str, days: int) -> Dict[str, Any]:
        """Get performance impact analysis for a model."""
        if model_id not in self.latency_history:
            raise HTTPException(status_code=404, detail=f"No latency data found for model {model_id}")
        
        history = self.latency_history[model_id]
        
        # Filter by time period
        cutoff_time = datetime.now() - timedelta(days=days)
        recent_history = [
            h for h in history 
            if datetime.fromisoformat(h['timestamp']) > cutoff_time
        ]
        
        if not recent_history:
            raise HTTPException(status_code=404, detail=f"No latency data found for the last {days} days")
        
        latencies = [h['inference_time'] for h in recent_history]
        
        # Categorize performance impact
        impact_categories = {
            'minimal': 0,
            'moderate': 0,
            'significant': 0,
            'critical': 0
        }
        
        for latency in latencies:
            if latency <= self.latency_thresholds['warning_latency']:
                impact_categories['minimal'] += 1
            elif latency <= self.latency_thresholds['max_latency']:
                impact_categories['moderate'] += 1
            elif latency <= self.latency_thresholds['critical_latency']:
                impact_categories['significant'] += 1
            else:
                impact_categories['critical'] += 1
        
        # Calculate percentages
        total_measurements = len(latencies)
        impact_percentages = {
            category: (count / total_measurements) * 100 
            for category, count in impact_categories.items()
        }
        
        # Calculate SLA compliance
        sla_threshold = self.latency_thresholds['max_latency']
        sla_compliant = sum(1 for l in latencies if l <= sla_threshold)
        sla_compliance_rate = (sla_compliant / total_measurements) * 100
        
        return {
            'model_id': model_id,
            'time_period': f"{days} days",
            'total_measurements': total_measurements,
            'impact_categories': impact_categories,
            'impact_percentages': impact_percentages,
            'sla_compliance_rate': sla_compliance_rate,
            'sla_threshold': sla_threshold,
            'timestamp': datetime.now().isoformat()
        }
    
    async def _list_latency_alerts(self, limit: int) -> Dict[str, Any]:
        """List latency alerts."""
        alert_list = []
        for alert_id, alert in self.latency_alerts.items():
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
        """Resolve a latency alert."""
        if alert_id not in self.latency_alerts:
            raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found")
        
        self.latency_alerts[alert_id]['resolved'] = True
        
        return {
            'alert_id': alert_id,
            'status': 'resolved',
            'message': 'Alert resolved successfully',
            'timestamp': datetime.now().isoformat()
        }
    
    def run(self, host: str = "0.0.0.0", port: int = 8019):
        """Run the service."""
        logger.info(f"Starting Latency Monitor on {host}:{port}")
        uvicorn.run(self.app, host=host, port=port)


# Global service instance
service = LatencyMonitor()


def create_app() -> FastAPI:
    """Create FastAPI application."""
    return service.app


if __name__ == "__main__":
    service.run()
