"""
Model Drift Detector

This module implements model drift detection for PBF-LB/M processes.
It provides model performance monitoring, prediction drift detection,
and model degradation alerts.
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
from scipy import stats
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
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
class ModelDriftConfig(BaseModel):
    """Configuration for model drift detection."""
    model_id: str = Field(..., description="Model ID to monitor")
    baseline_performance: Dict[str, float] = Field(..., description="Baseline performance metrics")
    performance_threshold: float = Field(0.1, description="Performance degradation threshold")
    prediction_drift_threshold: float = Field(0.15, description="Prediction drift threshold")
    window_size: int = Field(100, description="Window size for performance calculation")
    min_samples: int = Field(50, description="Minimum samples for drift detection")
    significance_level: float = Field(0.05, description="Significance level for statistical tests")
    enable_prediction_drift: bool = Field(True, description="Enable prediction drift detection")
    enable_performance_drift: bool = Field(True, description="Enable performance drift detection")


class ModelDriftRequest(BaseModel):
    """Request model for model drift detection."""
    detector_id: str = Field(..., description="Detector ID")
    model_id: str = Field(..., description="Model ID")
    predictions: List[Any] = Field(..., description="Model predictions")
    ground_truth: Optional[List[Any]] = Field(None, description="Ground truth labels")
    input_features: List[Dict[str, Any]] = Field(..., description="Input features")
    config: ModelDriftConfig = Field(..., description="Detection configuration")
    enable_alerts: bool = Field(True, description="Enable drift alerts")


class ModelDriftResponse(BaseModel):
    """Response model for model drift detection."""
    detector_id: str = Field(..., description="Detector ID")
    model_id: str = Field(..., description="Model ID")
    drift_detected: bool = Field(..., description="Whether drift was detected")
    drift_type: str = Field(..., description="Type of drift detected")
    drift_score: float = Field(..., description="Overall drift score")
    performance_metrics: Dict[str, float] = Field(..., description="Current performance metrics")
    performance_drift: Optional[Dict[str, Any]] = Field(None, description="Performance drift analysis")
    prediction_drift: Optional[Dict[str, Any]] = Field(None, description="Prediction drift analysis")
    recommendations: List[str] = Field(..., description="Recommendations based on drift")
    timestamp: str = Field(..., description="Detection timestamp")


class ModelPerformanceMetrics(BaseModel):
    """Model for model performance metrics."""
    model_id: str = Field(..., description="Model ID")
    accuracy: Optional[float] = Field(None, description="Accuracy score")
    precision: Optional[float] = Field(None, description="Precision score")
    recall: Optional[float] = Field(None, description="Recall score")
    f1_score: Optional[float] = Field(None, description="F1 score")
    mse: Optional[float] = Field(None, description="Mean squared error")
    mae: Optional[float] = Field(None, description="Mean absolute error")
    r2_score: Optional[float] = Field(None, description="R-squared score")
    timestamp: str = Field(..., description="Metrics timestamp")


class ModelDriftDetector:
    """
    Model drift detector for PBF-LB/M processes.
    
    This detector provides comprehensive model drift detection capabilities for:
    - Performance drift detection
    - Prediction drift detection
    - Model degradation monitoring
    - Performance trend analysis
    """
    
    def __init__(self, config_manager: Optional[ConfigManager] = None):
        """
        Initialize the model drift detector.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config_manager = config_manager or ConfigManager()
        self.app = FastAPI(
            title="Model Drift Detector",
            description="Model drift detection for PBF-LB/M manufacturing",
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
        
        # Detector management
        self.detectors = {}  # Store detector information
        self.model_performance_history = {}  # Store performance history
        self.prediction_history = {}  # Store prediction history
        self.drift_alerts = {}  # Store drift alerts
        self.detector_counter = 0
        
        # Service metrics
        self.service_metrics = {
            'total_detectors': 0,
            'active_detectors': 0,
            'total_detections': 0,
            'drift_detected_count': 0,
            'performance_drift_count': 0,
            'prediction_drift_count': 0,
            'total_alerts': 0,
            'last_detection_time': None
        }
        
        # Setup routes
        self._setup_routes()
        
        logger.info("Initialized ModelDriftDetector")
    
    def _setup_routes(self):
        """Setup API routes."""
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {
                "status": "healthy",
                "service": "model_drift_detector",
                "timestamp": datetime.now().isoformat(),
                "metrics": self.service_metrics
            }
        
        @self.app.post("/detect", response_model=ModelDriftResponse)
        async def detect_model_drift(request: ModelDriftRequest):
            """Detect model drift."""
            return await self._detect_model_drift(request)
        
        @self.app.post("/detectors", response_model=Dict[str, Any])
        async def create_detector(request: ModelDriftRequest):
            """Create a new model drift detector."""
            return await self._create_detector(request)
        
        @self.app.get("/detectors")
        async def list_detectors():
            """List all detectors."""
            return await self._list_detectors()
        
        @self.app.get("/detectors/{detector_id}/status")
        async def get_detector_status(detector_id: str):
            """Get detector status."""
            return await self._get_detector_status(detector_id)
        
        @self.app.get("/detectors/{detector_id}/performance-history")
        async def get_performance_history(detector_id: str, limit: int = Query(10, ge=1, le=100)):
            """Get model performance history."""
            return await self._get_performance_history(detector_id, limit)
        
        @self.app.get("/models/{model_id}/performance")
        async def get_model_performance(model_id: str):
            """Get current model performance."""
            return await self._get_model_performance(model_id)
        
        @self.app.get("/alerts")
        async def list_alerts(limit: int = Query(10, ge=1, le=100)):
            """List model drift alerts."""
            return await self._list_alerts(limit)
        
        @self.app.post("/alerts/{alert_id}/resolve")
        async def resolve_alert(alert_id: str):
            """Resolve a model drift alert."""
            return await self._resolve_alert(alert_id)
        
        @self.app.get("/metrics")
        async def get_metrics():
            """Get service metrics."""
            return self.service_metrics
    
    async def _detect_model_drift(self, request: ModelDriftRequest) -> ModelDriftResponse:
        """
        Detect model drift.
        
        Args:
            request: Model drift detection request
            
        Returns:
            Model drift detection response
        """
        try:
            # Convert predictions to numpy array
            predictions = np.array(request.predictions)
            
            # Initialize results
            drift_detected = False
            drift_type = "none"
            drift_score = 0.0
            performance_metrics = {}
            performance_drift = None
            prediction_drift = None
            
            # Calculate current performance metrics if ground truth is available
            if request.ground_truth is not None:
                ground_truth = np.array(request.ground_truth)
                performance_metrics = await self._calculate_performance_metrics(
                    predictions, ground_truth
                )
                
                # Detect performance drift
                if request.config.enable_performance_drift:
                    performance_drift = await self._detect_performance_drift(
                        performance_metrics, request.config
                    )
                    
                    if performance_drift['drift_detected']:
                        drift_detected = True
                        drift_type = "performance"
                        drift_score = performance_drift['drift_score']
            
            # Detect prediction drift
            if request.config.enable_prediction_drift:
                prediction_drift = await self._detect_prediction_drift(
                    predictions, request.detector_id, request.config
                )
                
                if prediction_drift['drift_detected']:
                    if not drift_detected:
                        drift_detected = True
                        drift_type = "prediction"
                        drift_score = prediction_drift['drift_score']
                    else:
                        # Combine drift scores
                        drift_score = max(drift_score, prediction_drift['drift_score'])
                        drift_type = "combined"
            
            # Store performance history
            if performance_metrics:
                await self._store_performance_history(
                    request.detector_id, performance_metrics
                )
            
            # Store prediction history
            await self._store_prediction_history(
                request.detector_id, predictions
            )
            
            # Generate recommendations
            recommendations = await self._generate_recommendations(
                drift_detected, drift_type, drift_score, performance_drift, prediction_drift
            )
            
            # Create alert if drift detected and alerts enabled
            if drift_detected and request.enable_alerts:
                await self._create_model_drift_alert(
                    request.detector_id, request.model_id, drift_type, drift_score
                )
            
            # Update metrics
            self.service_metrics['total_detections'] += 1
            if drift_detected:
                self.service_metrics['drift_detected_count'] += 1
                if drift_type in ['performance', 'combined']:
                    self.service_metrics['performance_drift_count'] += 1
                if drift_type in ['prediction', 'combined']:
                    self.service_metrics['prediction_drift_count'] += 1
            self.service_metrics['last_detection_time'] = datetime.now().isoformat()
            
            return ModelDriftResponse(
                detector_id=request.detector_id,
                model_id=request.model_id,
                drift_detected=drift_detected,
                drift_type=drift_type,
                drift_score=drift_score,
                performance_metrics=performance_metrics,
                performance_drift=performance_drift,
                prediction_drift=prediction_drift,
                recommendations=recommendations,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"Error in model drift detection: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def _calculate_performance_metrics(self, predictions: np.ndarray, 
                                           ground_truth: np.ndarray) -> Dict[str, float]:
        """Calculate performance metrics."""
        metrics = {}
        
        try:
            # Determine if it's classification or regression
            if len(np.unique(ground_truth)) <= 10 and np.all(np.isfinite(ground_truth)):
                # Classification metrics
                metrics['accuracy'] = float(accuracy_score(ground_truth, predictions))
                metrics['precision'] = float(precision_score(ground_truth, predictions, average='weighted', zero_division=0))
                metrics['recall'] = float(recall_score(ground_truth, predictions, average='weighted', zero_division=0))
                metrics['f1_score'] = float(f1_score(ground_truth, predictions, average='weighted', zero_division=0))
            else:
                # Regression metrics
                metrics['mse'] = float(mean_squared_error(ground_truth, predictions))
                metrics['mae'] = float(mean_absolute_error(ground_truth, predictions))
                metrics['r2_score'] = float(r2_score(ground_truth, predictions))
        
        except Exception as e:
            logger.warning(f"Error calculating performance metrics: {e}")
        
        return metrics
    
    async def _detect_performance_drift(self, current_metrics: Dict[str, float], 
                                      config: ModelDriftConfig) -> Dict[str, Any]:
        """Detect performance drift."""
        drift_detected = False
        drift_score = 0.0
        metric_drifts = {}
        
        for metric_name, current_value in current_metrics.items():
            if metric_name in config.baseline_performance:
                baseline_value = config.baseline_performance[metric_name]
                
                # Calculate performance degradation
                if metric_name in ['accuracy', 'precision', 'recall', 'f1_score', 'r2_score']:
                    # Higher is better
                    degradation = baseline_value - current_value
                    degradation_pct = degradation / baseline_value if baseline_value != 0 else 0
                else:
                    # Lower is better (MSE, MAE)
                    degradation = current_value - baseline_value
                    degradation_pct = degradation / baseline_value if baseline_value != 0 else 0
                
                metric_drifts[metric_name] = {
                    'baseline': baseline_value,
                    'current': current_value,
                    'degradation': degradation,
                    'degradation_pct': degradation_pct,
                    'drift_detected': degradation_pct > config.performance_threshold
                }
                
                if degradation_pct > config.performance_threshold:
                    drift_detected = True
                    drift_score = max(drift_score, degradation_pct)
        
        return {
            'drift_detected': drift_detected,
            'drift_score': drift_score,
            'metric_drifts': metric_drifts,
            'performance_threshold': config.performance_threshold
        }
    
    async def _detect_prediction_drift(self, current_predictions: np.ndarray, 
                                     detector_id: str, config: ModelDriftConfig) -> Dict[str, Any]:
        """Detect prediction drift."""
        drift_detected = False
        drift_score = 0.0
        
        # Get historical predictions
        if detector_id in self.prediction_history:
            historical_predictions = self.prediction_history[detector_id]
            
            if len(historical_predictions) >= config.min_samples:
                # Calculate prediction distribution drift
                try:
                    # Kolmogorov-Smirnov test
                    ks_statistic, ks_p_value = stats.ks_2samp(
                        historical_predictions, current_predictions
                    )
                    
                    # Calculate drift score based on KS statistic
                    drift_score = ks_statistic
                    drift_detected = drift_score > config.prediction_drift_threshold
                    
                    return {
                        'drift_detected': drift_detected,
                        'drift_score': drift_score,
                        'ks_statistic': float(ks_statistic),
                        'ks_p_value': float(ks_p_value),
                        'prediction_drift_threshold': config.prediction_drift_threshold,
                        'historical_samples': len(historical_predictions),
                        'current_samples': len(current_predictions)
                    }
                
                except Exception as e:
                    logger.warning(f"Error in prediction drift detection: {e}")
        
        return {
            'drift_detected': False,
            'drift_score': 0.0,
            'reason': 'Insufficient historical data' if detector_id not in self.prediction_history else 'Statistical test failed',
            'historical_samples': len(self.prediction_history.get(detector_id, [])),
            'current_samples': len(current_predictions)
        }
    
    async def _store_performance_history(self, detector_id: str, metrics: Dict[str, float]):
        """Store performance history."""
        if detector_id not in self.model_performance_history:
            self.model_performance_history[detector_id] = []
        
        performance_record = {
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics
        }
        
        self.model_performance_history[detector_id].append(performance_record)
        
        # Keep only recent history
        if len(self.model_performance_history[detector_id]) > 1000:
            self.model_performance_history[detector_id] = self.model_performance_history[detector_id][-1000:]
    
    async def _store_prediction_history(self, detector_id: str, predictions: np.ndarray):
        """Store prediction history."""
        if detector_id not in self.prediction_history:
            self.prediction_history[detector_id] = []
        
        # Store recent predictions
        self.prediction_history[detector_id].extend(predictions.tolist())
        
        # Keep only recent history
        if len(self.prediction_history[detector_id]) > 10000:
            self.prediction_history[detector_id] = self.prediction_history[detector_id][-10000:]
    
    async def _generate_recommendations(self, drift_detected: bool, drift_type: str, 
                                      drift_score: float, performance_drift: Optional[Dict[str, Any]], 
                                      prediction_drift: Optional[Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on drift detection results."""
        recommendations = []
        
        if not drift_detected:
            recommendations.append("No model drift detected - continue monitoring")
            return recommendations
        
        if drift_type in ['performance', 'combined'] and performance_drift:
            if drift_score > 0.2:
                recommendations.append("Critical performance degradation detected - immediate model retraining recommended")
            elif drift_score > 0.1:
                recommendations.append("Significant performance degradation - consider model retraining")
            else:
                recommendations.append("Minor performance degradation - monitor closely")
        
        if drift_type in ['prediction', 'combined'] and prediction_drift:
            if drift_score > 0.2:
                recommendations.append("Critical prediction drift detected - investigate data quality and model performance")
            elif drift_score > 0.15:
                recommendations.append("Significant prediction drift - review model inputs and retrain if necessary")
            else:
                recommendations.append("Minor prediction drift - continue monitoring")
        
        if drift_type == 'combined':
            recommendations.append("Multiple types of drift detected - comprehensive model review recommended")
        
        return recommendations
    
    async def _create_model_drift_alert(self, detector_id: str, model_id: str, 
                                      drift_type: str, drift_score: float):
        """Create a model drift alert."""
        alert_id = f"model_drift_alert_{int(time.time())}"
        
        # Determine alert severity
        if drift_score > 0.2:
            severity = "critical"
        elif drift_score > 0.1:
            severity = "warning"
        else:
            severity = "info"
        
        alert = {
            'alert_id': alert_id,
            'detector_id': detector_id,
            'model_id': model_id,
            'alert_type': 'model_drift',
            'severity': severity,
            'drift_type': drift_type,
            'drift_score': drift_score,
            'message': f"Model drift detected: {drift_type} drift with score {drift_score:.3f}",
            'timestamp': datetime.now().isoformat(),
            'resolved': False
        }
        
        self.drift_alerts[alert_id] = alert
        self.service_metrics['total_alerts'] += 1
        
        logger.warning(f"Model drift alert created: {alert['message']}")
    
    async def _create_detector(self, request: ModelDriftRequest) -> Dict[str, Any]:
        """Create a new model drift detector."""
        self.detector_counter += 1
        detector_id = f"model_detector_{self.detector_counter}_{int(time.time())}"
        
        detector_info = {
            'detector_id': detector_id,
            'model_id': request.model_id,
            'config': request.config.dict(),
            'created_at': datetime.now().isoformat(),
            'status': 'active',
            'total_detections': 0,
            'drift_detected_count': 0
        }
        
        self.detectors[detector_id] = detector_info
        
        self.service_metrics['total_detectors'] += 1
        self.service_metrics['active_detectors'] += 1
        
        return {
            'detector_id': detector_id,
            'model_id': request.model_id,
            'status': 'created',
            'message': 'Model drift detector created successfully',
            'timestamp': datetime.now().isoformat()
        }
    
    async def _list_detectors(self) -> Dict[str, Any]:
        """List all detectors."""
        detector_list = []
        for detector_id, detector_info in self.detectors.items():
            detector_list.append({
                'detector_id': detector_id,
                'model_id': detector_info['model_id'],
                'status': detector_info['status'],
                'total_detections': detector_info['total_detections'],
                'drift_detected_count': detector_info['drift_detected_count'],
                'created_at': detector_info['created_at']
            })
        
        return {
            'detectors': detector_list,
            'total_detectors': len(detector_list),
            'active_detectors': sum(1 for d in detector_list if d['status'] == 'active'),
            'timestamp': datetime.now().isoformat()
        }
    
    async def _get_detector_status(self, detector_id: str) -> Dict[str, Any]:
        """Get detector status."""
        if detector_id not in self.detectors:
            raise HTTPException(status_code=404, detail=f"Detector {detector_id} not found")
        
        detector_info = self.detectors[detector_id]
        
        return {
            'detector_id': detector_id,
            'model_id': detector_info['model_id'],
            'status': detector_info['status'],
            'total_detections': detector_info['total_detections'],
            'drift_detected_count': detector_info['drift_detected_count'],
            'created_at': detector_info['created_at']
        }
    
    async def _get_performance_history(self, detector_id: str, limit: int) -> Dict[str, Any]:
        """Get model performance history."""
        if detector_id not in self.model_performance_history:
            raise HTTPException(status_code=404, detail=f"Detector {detector_id} not found")
        
        history = self.model_performance_history[detector_id]
        limited_history = history[-limit:] if limit > 0 else history
        
        return {
            'detector_id': detector_id,
            'performance_history': limited_history,
            'total_records': len(history),
            'timestamp': datetime.now().isoformat()
        }
    
    async def _get_model_performance(self, model_id: str) -> Dict[str, Any]:
        """Get current model performance."""
        # Find detectors for this model
        model_detectors = [d for d in self.detectors.values() if d['model_id'] == model_id]
        
        if not model_detectors:
            raise HTTPException(status_code=404, detail=f"No detectors found for model {model_id}")
        
        # Get latest performance from all detectors
        latest_performance = {}
        for detector in model_detectors:
            detector_id = detector['detector_id']
            if detector_id in self.model_performance_history:
                history = self.model_performance_history[detector_id]
                if history:
                    latest_performance[detector_id] = history[-1]
        
        return {
            'model_id': model_id,
            'detectors': len(model_detectors),
            'latest_performance': latest_performance,
            'timestamp': datetime.now().isoformat()
        }
    
    async def _list_alerts(self, limit: int) -> Dict[str, Any]:
        """List model drift alerts."""
        alert_list = []
        for alert_id, alert in self.drift_alerts.items():
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
        """Resolve a model drift alert."""
        if alert_id not in self.drift_alerts:
            raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found")
        
        self.drift_alerts[alert_id]['resolved'] = True
        
        return {
            'alert_id': alert_id,
            'status': 'resolved',
            'message': 'Alert resolved successfully',
            'timestamp': datetime.now().isoformat()
        }
    
    def run(self, host: str = "0.0.0.0", port: int = 8014):
        """Run the service."""
        logger.info(f"Starting Model Drift Detector on {host}:{port}")
        uvicorn.run(self.app, host=host, port=port)


# Global service instance
service = ModelDriftDetector()


def create_app() -> FastAPI:
    """Create FastAPI application."""
    return service.app


if __name__ == "__main__":
    service.run()
