"""
Data Drift Detector

This module implements data drift detection for PBF-LB/M processes.
It provides statistical tests, distribution comparisons, and drift alerts
for monitoring data quality and consistency over time.
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
from scipy.stats import ks_2samp, chi2_contingency, anderson_ksamp
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import LabelEncoder
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
class DriftDetectionConfig(BaseModel):
    """Configuration for drift detection."""
    detection_method: str = Field("ks_test", description="Drift detection method")
    significance_level: float = Field(0.05, description="Significance level for statistical tests")
    window_size: int = Field(1000, description="Window size for drift detection")
    min_samples: int = Field(100, description="Minimum samples required for detection")
    drift_threshold: float = Field(0.1, description="Drift threshold")
    alert_threshold: float = Field(0.2, description="Alert threshold")
    enable_visualization: bool = Field(True, description="Enable drift visualization")


class DriftDetectionRequest(BaseModel):
    """Request model for drift detection."""
    detector_id: str = Field(..., description="Detector ID")
    reference_data: Dict[str, Any] = Field(..., description="Reference dataset")
    current_data: Dict[str, Any] = Field(..., description="Current dataset")
    features: List[str] = Field(..., description="Features to monitor")
    config: DriftDetectionConfig = Field(..., description="Detection configuration")
    enable_alerts: bool = Field(True, description="Enable drift alerts")


class DriftDetectionResponse(BaseModel):
    """Response model for drift detection."""
    detector_id: str = Field(..., description="Detector ID")
    drift_detected: bool = Field(..., description="Whether drift was detected")
    drift_score: float = Field(..., description="Overall drift score")
    feature_drifts: Dict[str, Dict[str, Any]] = Field(..., description="Feature-level drift results")
    statistical_tests: Dict[str, Dict[str, Any]] = Field(..., description="Statistical test results")
    visualizations: Optional[Dict[str, Any]] = Field(None, description="Drift visualizations")
    recommendations: List[str] = Field(..., description="Recommendations based on drift")
    timestamp: str = Field(..., description="Detection timestamp")


class DriftAlert(BaseModel):
    """Model for drift alerts."""
    alert_id: str = Field(..., description="Alert ID")
    detector_id: str = Field(..., description="Detector ID")
    alert_type: str = Field(..., description="Alert type (drift, warning, critical)")
    severity: str = Field(..., description="Alert severity")
    message: str = Field(..., description="Alert message")
    drift_score: float = Field(..., description="Drift score")
    affected_features: List[str] = Field(..., description="Affected features")
    timestamp: str = Field(..., description="Alert timestamp")
    resolved: bool = Field(False, description="Whether alert is resolved")


class DataDriftDetector:
    """
    Data drift detector for PBF-LB/M processes.
    
    This detector provides comprehensive drift detection capabilities for:
    - Statistical drift detection
    - Distribution comparison
    - Feature-level drift analysis
    - Drift visualization
    - Alert management
    """
    
    def __init__(self, config_manager: Optional[ConfigManager] = None):
        """
        Initialize the data drift detector.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config_manager = config_manager or ConfigManager()
        self.app = FastAPI(
            title="Data Drift Detector",
            description="Data drift detection for PBF-LB/M manufacturing",
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
        self.drift_history = {}  # Store drift detection history
        self.alerts = {}  # Store drift alerts
        self.detector_counter = 0
        
        # Service metrics
        self.service_metrics = {
            'total_detectors': 0,
            'active_detectors': 0,
            'total_detections': 0,
            'drift_detected_count': 0,
            'total_alerts': 0,
            'last_detection_time': None
        }
        
        # Setup routes
        self._setup_routes()
        
        logger.info("Initialized DataDriftDetector")
    
    def _setup_routes(self):
        """Setup API routes."""
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {
                "status": "healthy",
                "service": "data_drift_detector",
                "timestamp": datetime.now().isoformat(),
                "metrics": self.service_metrics
            }
        
        @self.app.post("/detect", response_model=DriftDetectionResponse)
        async def detect_drift(request: DriftDetectionRequest):
            """Detect data drift."""
            return await self._detect_drift(request)
        
        @self.app.post("/detectors", response_model=Dict[str, Any])
        async def create_detector(request: DriftDetectionRequest):
            """Create a new drift detector."""
            return await self._create_detector(request)
        
        @self.app.get("/detectors")
        async def list_detectors():
            """List all detectors."""
            return await self._list_detectors()
        
        @self.app.get("/detectors/{detector_id}/status")
        async def get_detector_status(detector_id: str):
            """Get detector status."""
            return await self._get_detector_status(detector_id)
        
        @self.app.get("/detectors/{detector_id}/history")
        async def get_drift_history(detector_id: str, limit: int = Query(10, ge=1, le=100)):
            """Get drift detection history."""
            return await self._get_drift_history(detector_id, limit)
        
        @self.app.get("/alerts")
        async def list_alerts(limit: int = Query(10, ge=1, le=100)):
            """List drift alerts."""
            return await self._list_alerts(limit)
        
        @self.app.post("/alerts/{alert_id}/resolve")
        async def resolve_alert(alert_id: str):
            """Resolve a drift alert."""
            return await self._resolve_alert(alert_id)
        
        @self.app.get("/metrics")
        async def get_metrics():
            """Get service metrics."""
            return self.service_metrics
    
    async def _detect_drift(self, request: DriftDetectionRequest) -> DriftDetectionResponse:
        """
        Detect data drift between reference and current data.
        
        Args:
            request: Drift detection request
            
        Returns:
            Drift detection response
        """
        try:
            # Convert data to DataFrames
            reference_df = pd.DataFrame(request.reference_data)
            current_df = pd.DataFrame(request.current_data)
            
            # Initialize results
            feature_drifts = {}
            statistical_tests = {}
            drift_scores = []
            
            # Detect drift for each feature
            for feature in request.features:
                if feature in reference_df.columns and feature in current_df.columns:
                    drift_result = await self._detect_feature_drift(
                        reference_df[feature],
                        current_df[feature],
                        feature,
                        request.config
                    )
                    
                    feature_drifts[feature] = drift_result['drift_info']
                    statistical_tests[feature] = drift_result['statistical_tests']
                    drift_scores.append(drift_result['drift_score'])
            
            # Calculate overall drift score
            overall_drift_score = np.mean(drift_scores) if drift_scores else 0.0
            drift_detected = overall_drift_score > request.config.drift_threshold
            
            # Generate visualizations if enabled
            visualizations = None
            if request.config.enable_visualization:
                visualizations = await self._generate_drift_visualizations(
                    reference_df, current_df, request.features
                )
            
            # Generate recommendations
            recommendations = await self._generate_recommendations(
                feature_drifts, overall_drift_score, request.config
            )
            
            # Create alert if drift detected and alerts enabled
            if drift_detected and request.enable_alerts:
                await self._create_drift_alert(
                    request.detector_id,
                    overall_drift_score,
                    feature_drifts,
                    request.config
                )
            
            # Update metrics
            self.service_metrics['total_detections'] += 1
            if drift_detected:
                self.service_metrics['drift_detected_count'] += 1
            self.service_metrics['last_detection_time'] = datetime.now().isoformat()
            
            return DriftDetectionResponse(
                detector_id=request.detector_id,
                drift_detected=drift_detected,
                drift_score=overall_drift_score,
                feature_drifts=feature_drifts,
                statistical_tests=statistical_tests,
                visualizations=visualizations,
                recommendations=recommendations,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"Error in drift detection: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def _detect_feature_drift(self, reference_data: pd.Series, current_data: pd.Series,
                                  feature_name: str, config: DriftDetectionConfig) -> Dict[str, Any]:
        """Detect drift for a specific feature."""
        # Ensure minimum samples
        if len(reference_data) < config.min_samples or len(current_data) < config.min_samples:
            return {
                'drift_info': {
                    'drift_detected': False,
                    'drift_score': 0.0,
                    'reason': 'Insufficient samples'
                },
                'statistical_tests': {},
                'drift_score': 0.0
            }
        
        # Remove missing values
        reference_clean = reference_data.dropna()
        current_clean = current_data.dropna()
        
        if len(reference_clean) < config.min_samples or len(current_clean) < config.min_samples:
            return {
                'drift_info': {
                    'drift_detected': False,
                    'drift_score': 0.0,
                    'reason': 'Insufficient samples after cleaning'
                },
                'statistical_tests': {},
                'drift_score': 0.0
            }
        
        # Perform statistical tests based on data type
        statistical_tests = {}
        drift_scores = []
        
        if pd.api.types.is_numeric_dtype(reference_clean):
            # Numeric features
            drift_scores.append(await self._detect_numeric_drift(
                reference_clean, current_clean, config, statistical_tests
            ))
        else:
            # Categorical features
            drift_scores.append(await self._detect_categorical_drift(
                reference_clean, current_clean, config, statistical_tests
            ))
        
        # Calculate overall drift score for this feature
        feature_drift_score = np.mean(drift_scores) if drift_scores else 0.0
        drift_detected = feature_drift_score > config.drift_threshold
        
        return {
            'drift_info': {
                'drift_detected': drift_detected,
                'drift_score': feature_drift_score,
                'reference_stats': {
                    'mean': float(reference_clean.mean()) if pd.api.types.is_numeric_dtype(reference_clean) else None,
                    'std': float(reference_clean.std()) if pd.api.types.is_numeric_dtype(reference_clean) else None,
                    'count': len(reference_clean)
                },
                'current_stats': {
                    'mean': float(current_clean.mean()) if pd.api.types.is_numeric_dtype(current_clean) else None,
                    'std': float(current_clean.std()) if pd.api.types.is_numeric_dtype(current_clean) else None,
                    'count': len(current_clean)
                }
            },
            'statistical_tests': statistical_tests,
            'drift_score': feature_drift_score
        }
    
    async def _detect_numeric_drift(self, reference_data: pd.Series, current_data: pd.Series,
                                  config: DriftDetectionConfig, statistical_tests: Dict[str, Any]) -> float:
        """Detect drift for numeric features."""
        drift_scores = []
        
        # Kolmogorov-Smirnov test
        try:
            ks_statistic, ks_p_value = ks_2samp(reference_data, current_data)
            statistical_tests['ks_test'] = {
                'statistic': float(ks_statistic),
                'p_value': float(ks_p_value),
                'significant': ks_p_value < config.significance_level
            }
            drift_scores.append(ks_statistic)
        except Exception as e:
            logger.warning(f"KS test failed: {e}")
            statistical_tests['ks_test'] = {'error': str(e)}
        
        # Anderson-Darling test
        try:
            ad_statistic, ad_critical_values, ad_significance_level = anderson_ksamp(
                [reference_data.values, current_data.values]
            )
            statistical_tests['anderson_darling'] = {
                'statistic': float(ad_statistic),
                'critical_values': [float(cv) for cv in ad_critical_values],
                'significance_level': float(ad_significance_level)
            }
            drift_scores.append(ad_statistic / max(ad_critical_values))
        except Exception as e:
            logger.warning(f"Anderson-Darling test failed: {e}")
            statistical_tests['anderson_darling'] = {'error': str(e)}
        
        # Mean difference test
        try:
            mean_diff = abs(reference_data.mean() - current_data.mean())
            mean_diff_normalized = mean_diff / reference_data.std() if reference_data.std() > 0 else 0
            statistical_tests['mean_difference'] = {
                'difference': float(mean_diff),
                'normalized_difference': float(mean_diff_normalized)
            }
            drift_scores.append(mean_diff_normalized)
        except Exception as e:
            logger.warning(f"Mean difference test failed: {e}")
            statistical_tests['mean_difference'] = {'error': str(e)}
        
        return np.mean(drift_scores) if drift_scores else 0.0
    
    async def _detect_categorical_drift(self, reference_data: pd.Series, current_data: pd.Series,
                                      config: DriftDetectionConfig, statistical_tests: Dict[str, Any]) -> float:
        """Detect drift for categorical features."""
        drift_scores = []
        
        # Chi-square test
        try:
            # Get unique values
            all_values = set(reference_data.unique()) | set(current_data.unique())
            
            # Create contingency table
            ref_counts = reference_data.value_counts()
            curr_counts = current_data.value_counts()
            
            contingency_table = []
            for value in all_values:
                contingency_table.append([
                    ref_counts.get(value, 0),
                    curr_counts.get(value, 0)
                ])
            
            if len(contingency_table) > 1:
                chi2_statistic, chi2_p_value, dof, expected = chi2_contingency(contingency_table)
                statistical_tests['chi_square'] = {
                    'statistic': float(chi2_statistic),
                    'p_value': float(chi2_p_value),
                    'degrees_of_freedom': int(dof),
                    'significant': chi2_p_value < config.significance_level
                }
                drift_scores.append(chi2_statistic / (len(reference_data) + len(current_data)))
        except Exception as e:
            logger.warning(f"Chi-square test failed: {e}")
            statistical_tests['chi_square'] = {'error': str(e)}
        
        # Mutual information
        try:
            # Encode categorical data
            le = LabelEncoder()
            all_data = pd.concat([reference_data, current_data])
            le.fit(all_data)
            
            ref_encoded = le.transform(reference_data)
            curr_encoded = le.transform(current_data)
            
            # Create binary labels (0 for reference, 1 for current)
            labels = np.concatenate([np.zeros(len(ref_encoded)), np.ones(len(curr_encoded))])
            features = np.concatenate([ref_encoded, curr_encoded])
            
            mi_score = mutual_info_score(labels, features)
            statistical_tests['mutual_information'] = {
                'score': float(mi_score)
            }
            drift_scores.append(mi_score)
        except Exception as e:
            logger.warning(f"Mutual information test failed: {e}")
            statistical_tests['mutual_information'] = {'error': str(e)}
        
        return np.mean(drift_scores) if drift_scores else 0.0
    
    async def _generate_drift_visualizations(self, reference_df: pd.DataFrame, 
                                           current_df: pd.DataFrame, 
                                           features: List[str]) -> Dict[str, Any]:
        """Generate drift visualizations."""
        visualizations = {}
        
        try:
            for feature in features:
                if feature in reference_df.columns and feature in current_df.columns:
                    if pd.api.types.is_numeric_dtype(reference_df[feature]):
                        # Numeric feature visualization
                        fig = go.Figure()
                        
                        # Add reference distribution
                        fig.add_trace(go.Histogram(
                            x=reference_df[feature].dropna(),
                            name='Reference',
                            opacity=0.7,
                            nbinsx=30
                        ))
                        
                        # Add current distribution
                        fig.add_trace(go.Histogram(
                            x=current_df[feature].dropna(),
                            name='Current',
                            opacity=0.7,
                            nbinsx=30
                        ))
                        
                        fig.update_layout(
                            title=f"Distribution Comparison - {feature}",
                            xaxis_title=feature,
                            yaxis_title="Frequency",
                            barmode='overlay'
                        )
                        
                        visualizations[f'{feature}_distribution'] = json.loads(fig.to_json())
                    else:
                        # Categorical feature visualization
                        ref_counts = reference_df[feature].value_counts()
                        curr_counts = current_df[feature].value_counts()
                        
                        fig = go.Figure()
                        
                        fig.add_trace(go.Bar(
                            x=ref_counts.index,
                            y=ref_counts.values,
                            name='Reference',
                            opacity=0.7
                        ))
                        
                        fig.add_trace(go.Bar(
                            x=curr_counts.index,
                            y=curr_counts.values,
                            name='Current',
                            opacity=0.7
                        ))
                        
                        fig.update_layout(
                            title=f"Category Comparison - {feature}",
                            xaxis_title=feature,
                            yaxis_title="Count",
                            barmode='group'
                        )
                        
                        visualizations[f'{feature}_categories'] = json.loads(fig.to_json())
        
        except Exception as e:
            logger.warning(f"Could not generate visualizations: {e}")
        
        return visualizations
    
    async def _generate_recommendations(self, feature_drifts: Dict[str, Any], 
                                      overall_drift_score: float, 
                                      config: DriftDetectionConfig) -> List[str]:
        """Generate recommendations based on drift detection results."""
        recommendations = []
        
        if overall_drift_score > config.alert_threshold:
            recommendations.append("Critical drift detected - immediate investigation required")
        elif overall_drift_score > config.drift_threshold:
            recommendations.append("Significant drift detected - monitor closely")
        
        # Feature-specific recommendations
        for feature, drift_info in feature_drifts.items():
            if drift_info['drift_detected']:
                if drift_info['drift_score'] > config.alert_threshold:
                    recommendations.append(f"Critical drift in feature '{feature}' - consider retraining model")
                else:
                    recommendations.append(f"Monitor feature '{feature}' for continued drift")
        
        if not recommendations:
            recommendations.append("No significant drift detected - continue monitoring")
        
        return recommendations
    
    async def _create_drift_alert(self, detector_id: str, drift_score: float, 
                                feature_drifts: Dict[str, Any], config: DriftDetectionConfig):
        """Create a drift alert."""
        alert_id = f"drift_alert_{int(time.time())}"
        
        # Determine alert severity
        if drift_score > config.alert_threshold:
            severity = "critical"
            alert_type = "drift"
        else:
            severity = "warning"
            alert_type = "drift"
        
        # Get affected features
        affected_features = [feature for feature, drift_info in feature_drifts.items() 
                           if drift_info['drift_detected']]
        
        alert = DriftAlert(
            alert_id=alert_id,
            detector_id=detector_id,
            alert_type=alert_type,
            severity=severity,
            message=f"Data drift detected with score {drift_score:.3f}",
            drift_score=drift_score,
            affected_features=affected_features,
            timestamp=datetime.now().isoformat(),
            resolved=False
        )
        
        self.alerts[alert_id] = alert
        self.service_metrics['total_alerts'] += 1
        
        logger.warning(f"Drift alert created: {alert.message}")
    
    async def _create_detector(self, request: DriftDetectionRequest) -> Dict[str, Any]:
        """Create a new drift detector."""
        self.detector_counter += 1
        detector_id = f"detector_{self.detector_counter}_{int(time.time())}"
        
        detector_info = {
            'detector_id': detector_id,
            'features': request.features,
            'config': request.config.dict(),
            'created_at': datetime.now().isoformat(),
            'status': 'active',
            'total_detections': 0,
            'drift_detected_count': 0
        }
        
        self.detectors[detector_id] = detector_info
        self.drift_history[detector_id] = []
        
        self.service_metrics['total_detectors'] += 1
        self.service_metrics['active_detectors'] += 1
        
        return {
            'detector_id': detector_id,
            'status': 'created',
            'message': 'Drift detector created successfully',
            'timestamp': datetime.now().isoformat()
        }
    
    async def _list_detectors(self) -> Dict[str, Any]:
        """List all detectors."""
        detector_list = []
        for detector_id, detector_info in self.detectors.items():
            detector_list.append({
                'detector_id': detector_id,
                'features': detector_info['features'],
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
            'status': detector_info['status'],
            'features': detector_info['features'],
            'total_detections': detector_info['total_detections'],
            'drift_detected_count': detector_info['drift_detected_count'],
            'created_at': detector_info['created_at']
        }
    
    async def _get_drift_history(self, detector_id: str, limit: int) -> Dict[str, Any]:
        """Get drift detection history."""
        if detector_id not in self.drift_history:
            raise HTTPException(status_code=404, detail=f"Detector {detector_id} not found")
        
        history = self.drift_history[detector_id]
        limited_history = history[-limit:] if limit > 0 else history
        
        return {
            'detector_id': detector_id,
            'history': limited_history,
            'total_detections': len(history),
            'timestamp': datetime.now().isoformat()
        }
    
    async def _list_alerts(self, limit: int) -> Dict[str, Any]:
        """List drift alerts."""
        alert_list = []
        for alert_id, alert in self.alerts.items():
            alert_list.append({
                'alert_id': alert_id,
                'detector_id': alert.detector_id,
                'alert_type': alert.alert_type,
                'severity': alert.severity,
                'message': alert.message,
                'drift_score': alert.drift_score,
                'affected_features': alert.affected_features,
                'timestamp': alert.timestamp,
                'resolved': alert.resolved
            })
        
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
        """Resolve a drift alert."""
        if alert_id not in self.alerts:
            raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found")
        
        self.alerts[alert_id].resolved = True
        
        return {
            'alert_id': alert_id,
            'status': 'resolved',
            'message': 'Alert resolved successfully',
            'timestamp': datetime.now().isoformat()
        }
    
    def run(self, host: str = "0.0.0.0", port: int = 8013):
        """Run the service."""
        logger.info(f"Starting Data Drift Detector on {host}:{port}")
        uvicorn.run(self.app, host=host, port=port)


# Global service instance
service = DataDriftDetector()


def create_app() -> FastAPI:
    """Create FastAPI application."""
    return service.app


if __name__ == "__main__":
    service.run()
