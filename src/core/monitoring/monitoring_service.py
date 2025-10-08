"""
Central monitoring service for PBF-LB/M operations.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from datetime import datetime
import asyncio
import logging

from .metrics import PrometheusClient, CustomMetrics, MetricsRegistry
from .tracing import JaegerClient, TraceDecorator, SpanManager
from .dashboards import GrafanaClient, DashboardManager, AlertManager
from .apm import DataDogClient, PerformanceTracker, HealthChecker


class MonitoringService:
    """
    Central monitoring service that coordinates all monitoring activities.
    
    This service provides a unified interface for metrics collection,
    distributed tracing, dashboard management, and APM integration.
    """
    
    def __init__(
        self,
        prometheus_client: Optional[PrometheusClient] = None,
        custom_metrics: Optional[CustomMetrics] = None,
        metrics_registry: Optional[MetricsRegistry] = None,
        jaeger_client: Optional[JaegerClient] = None,
        trace_decorator: Optional[TraceDecorator] = None,
        span_manager: Optional[SpanManager] = None,
        grafana_client: Optional[GrafanaClient] = None,
        dashboard_manager: Optional[DashboardManager] = None,
        alert_manager: Optional[AlertManager] = None,
        datadog_client: Optional[DataDogClient] = None,
        performance_tracker: Optional[PerformanceTracker] = None,
        health_checker: Optional[HealthChecker] = None
    ):
        """Initialize the monitoring service with optional components."""
        self.prometheus_client = prometheus_client
        self.custom_metrics = custom_metrics
        self.metrics_registry = metrics_registry
        self.jaeger_client = jaeger_client
        self.trace_decorator = trace_decorator
        self.span_manager = span_manager
        self.grafana_client = grafana_client
        self.dashboard_manager = dashboard_manager
        self.alert_manager = alert_manager
        self.datadog_client = datadog_client
        self.performance_tracker = performance_tracker
        self.health_checker = health_checker
        
        self.logger = logging.getLogger(__name__)
        self._is_initialized = False
    
    async def initialize(self) -> None:
        """Initialize all monitoring components."""
        try:
            # Initialize metrics components
            if self.prometheus_client:
                await self.prometheus_client.initialize()
            
            if self.custom_metrics:
                await self.custom_metrics.initialize()
            
            if self.metrics_registry:
                await self.metrics_registry.initialize()
            
            # Initialize tracing components
            if self.jaeger_client:
                await self.jaeger_client.initialize()
            
            if self.span_manager:
                await self.span_manager.initialize()
            
            # Initialize dashboard components
            if self.grafana_client:
                await self.grafana_client.initialize()
            
            if self.dashboard_manager:
                await self.dashboard_manager.initialize()
            
            if self.alert_manager:
                await self.alert_manager.initialize()
            
            # Initialize APM components
            if self.datadog_client:
                await self.datadog_client.initialize()
            
            if self.performance_tracker:
                await self.performance_tracker.initialize()
            
            if self.health_checker:
                await self.health_checker.initialize()
            
            self._is_initialized = True
            self.logger.info("Monitoring service initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize monitoring service: {e}")
            raise
    
    async def shutdown(self) -> None:
        """Shutdown all monitoring components."""
        try:
            # Shutdown all components in reverse order
            components = [
                self.health_checker,
                self.performance_tracker,
                self.datadog_client,
                self.alert_manager,
                self.dashboard_manager,
                self.grafana_client,
                self.span_manager,
                self.jaeger_client,
                self.metrics_registry,
                self.custom_metrics,
                self.prometheus_client
            ]
            
            for component in components:
                if component and hasattr(component, 'shutdown'):
                    await component.shutdown()
            
            self._is_initialized = False
            self.logger.info("Monitoring service shutdown successfully")
            
        except Exception as e:
            self.logger.error(f"Error during monitoring service shutdown: {e}")
    
    def is_initialized(self) -> bool:
        """Check if the monitoring service is initialized."""
        return self._is_initialized
    
    # Metrics methods
    async def record_metric(self, metric_name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a metric value."""
        if not self._is_initialized:
            raise RuntimeError("Monitoring service not initialized")
        
        try:
            if self.custom_metrics:
                await self.custom_metrics.record_metric(metric_name, value, tags)
            
            if self.prometheus_client:
                await self.prometheus_client.record_metric(metric_name, value, tags)
            
            if self.datadog_client:
                await self.datadog_client.record_metric(metric_name, value, tags)
                
        except Exception as e:
            self.logger.error(f"Failed to record metric {metric_name}: {e}")
    
    async def increment_counter(self, counter_name: str, value: float = 1.0, tags: Optional[Dict[str, str]] = None) -> None:
        """Increment a counter metric."""
        if not self._is_initialized:
            raise RuntimeError("Monitoring service not initialized")
        
        try:
            if self.custom_metrics:
                await self.custom_metrics.increment_counter(counter_name, value, tags)
            
            if self.prometheus_client:
                await self.prometheus_client.increment_counter(counter_name, value, tags)
            
            if self.datadog_client:
                await self.datadog_client.increment_counter(counter_name, value, tags)
                
        except Exception as e:
            self.logger.error(f"Failed to increment counter {counter_name}: {e}")
    
    async def record_histogram(self, histogram_name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a histogram value."""
        if not self._is_initialized:
            raise RuntimeError("Monitoring service not initialized")
        
        try:
            if self.custom_metrics:
                await self.custom_metrics.record_histogram(histogram_name, value, tags)
            
            if self.prometheus_client:
                await self.prometheus_client.record_histogram(histogram_name, value, tags)
            
            if self.datadog_client:
                await self.datadog_client.record_histogram(histogram_name, value, tags)
                
        except Exception as e:
            self.logger.error(f"Failed to record histogram {histogram_name}: {e}")
    
    # Tracing methods
    async def start_trace(self, operation_name: str, tags: Optional[Dict[str, str]] = None) -> str:
        """Start a new trace."""
        if not self._is_initialized:
            raise RuntimeError("Monitoring service not initialized")
        
        try:
            if self.span_manager:
                return await self.span_manager.start_span(operation_name, tags)
            return None
        except Exception as e:
            self.logger.error(f"Failed to start trace {operation_name}: {e}")
            return None
    
    async def finish_trace(self, span_id: str, tags: Optional[Dict[str, str]] = None) -> None:
        """Finish a trace."""
        if not self._is_initialized:
            raise RuntimeError("Monitoring service not initialized")
        
        try:
            if self.span_manager:
                await self.span_manager.finish_span(span_id, tags)
        except Exception as e:
            self.logger.error(f"Failed to finish trace {span_id}: {e}")
    
    async def add_trace_tag(self, span_id: str, key: str, value: str) -> None:
        """Add a tag to a trace."""
        if not self._is_initialized:
            raise RuntimeError("Monitoring service not initialized")
        
        try:
            if self.span_manager:
                await self.span_manager.add_tag(span_id, key, value)
        except Exception as e:
            self.logger.error(f"Failed to add tag to trace {span_id}: {e}")
    
    # Alert methods
    async def create_alert(self, alert_data: Dict[str, Any]) -> str:
        """Create an alert."""
        if not self._is_initialized:
            raise RuntimeError("Monitoring service not initialized")
        
        try:
            if self.alert_manager:
                return await self.alert_manager.create_alert(alert_data)
            return None
        except Exception as e:
            self.logger.error(f"Failed to create alert: {e}")
            return None
    
    async def resolve_alert(self, alert_id: str, resolution_data: Optional[Dict[str, Any]] = None) -> bool:
        """Resolve an alert."""
        if not self._is_initialized:
            raise RuntimeError("Monitoring service not initialized")
        
        try:
            if self.alert_manager:
                return await self.alert_manager.resolve_alert(alert_id, resolution_data)
            return False
        except Exception as e:
            self.logger.error(f"Failed to resolve alert {alert_id}: {e}")
            return False
    
    # Health check methods
    async def perform_health_check(self, component: Optional[str] = None) -> Dict[str, Any]:
        """Perform a health check."""
        if not self._is_initialized:
            raise RuntimeError("Monitoring service not initialized")
        
        try:
            if self.health_checker:
                return await self.health_checker.check_health(component)
            return {"status": "unknown", "message": "Health checker not available"}
        except Exception as e:
            self.logger.error(f"Failed to perform health check: {e}")
            return {"status": "error", "message": str(e)}
    
    # Performance tracking methods
    async def start_performance_tracking(self, operation_name: str) -> str:
        """Start performance tracking for an operation."""
        if not self._is_initialized:
            raise RuntimeError("Monitoring service not initialized")
        
        try:
            if self.performance_tracker:
                return await self.performance_tracker.start_tracking(operation_name)
            return None
        except Exception as e:
            self.logger.error(f"Failed to start performance tracking for {operation_name}: {e}")
            return None
    
    async def stop_performance_tracking(self, tracking_id: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Stop performance tracking and get results."""
        if not self._is_initialized:
            raise RuntimeError("Monitoring service not initialized")
        
        try:
            if self.performance_tracker:
                return await self.performance_tracker.stop_tracking(tracking_id, metadata)
            return {}
        except Exception as e:
            self.logger.error(f"Failed to stop performance tracking {tracking_id}: {e}")
            return {}
    
    # Dashboard methods
    async def get_dashboard_data(self, dashboard_name: str, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get dashboard data."""
        if not self._is_initialized:
            raise RuntimeError("Monitoring service not initialized")
        
        try:
            if self.dashboard_manager:
                return await self.dashboard_manager.get_dashboard_data(dashboard_name, filters)
            return {}
        except Exception as e:
            self.logger.error(f"Failed to get dashboard data for {dashboard_name}: {e}")
            return {}
    
    # Utility methods
    async def get_monitoring_status(self) -> Dict[str, Any]:
        """Get overall monitoring status."""
        if not self._is_initialized:
            return {"status": "not_initialized"}
        
        try:
            status = {
                "status": "initialized",
                "components": {
                    "prometheus": self.prometheus_client is not None,
                    "custom_metrics": self.custom_metrics is not None,
                    "metrics_registry": self.metrics_registry is not None,
                    "jaeger": self.jaeger_client is not None,
                    "span_manager": self.span_manager is not None,
                    "grafana": self.grafana_client is not None,
                    "dashboard_manager": self.dashboard_manager is not None,
                    "alert_manager": self.alert_manager is not None,
                    "datadog": self.datadog_client is not None,
                    "performance_tracker": self.performance_tracker is not None,
                    "health_checker": self.health_checker is not None
                }
            }
            
            # Get health status for each component
            if self.health_checker:
                health_status = await self.health_checker.check_health()
                status["health"] = health_status
            
            return status
            
        except Exception as e:
            self.logger.error(f"Failed to get monitoring status: {e}")
            return {"status": "error", "message": str(e)}
    
    async def get_metrics_summary(self) -> Dict[str, Any]:
        """Get a summary of all metrics."""
        if not self._is_initialized:
            raise RuntimeError("Monitoring service not initialized")
        
        try:
            summary = {}
            
            if self.metrics_registry:
                summary["registry"] = await self.metrics_registry.get_metrics_summary()
            
            if self.prometheus_client:
                summary["prometheus"] = await self.prometheus_client.get_metrics_summary()
            
            if self.custom_metrics:
                summary["custom"] = await self.custom_metrics.get_metrics_summary()
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Failed to get metrics summary: {e}")
            return {}
    
    def get_trace_decorator(self) -> Optional[TraceDecorator]:
        """Get the trace decorator for function decoration."""
        return self.trace_decorator
