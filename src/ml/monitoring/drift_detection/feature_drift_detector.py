"""
Feature Drift Detector

This module implements feature drift detection for PBF-LB/M processes.
It provides feature-level drift monitoring, feature importance tracking,
and feature stability analysis.
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
from scipy.stats import ks_2samp, chi2_contingency
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
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
class FeatureDriftConfig(BaseModel):
    """Configuration for feature drift detection."""
    features: List[str] = Field(..., description="Features to monitor")
    detection_methods: List[str] = Field(["ks_test", "psi", "chi_square"], description="Detection methods to use")
    significance_level: float = Field(0.05, description="Significance level for statistical tests")
    psi_threshold: float = Field(0.2, description="PSI threshold for drift detection")
    ks_threshold: float = Field(0.1, description="KS test threshold")
    window_size: int = Field(1000, description="Window size for drift detection")
    min_samples: int = Field(100, description="Minimum samples required")
    enable_feature_importance: bool = Field(True, description="Enable feature importance tracking")
    enable_correlation_analysis: bool = Field(True, description="Enable correlation analysis")


class FeatureDriftRequest(BaseModel):
    """Request model for feature drift detection."""
    detector_id: str = Field(..., description="Detector ID")
    reference_data: Dict[str, Any] = Field(..., description="Reference dataset")
    current_data: Dict[str, Any] = Field(..., description="Current dataset")
    target_variable: Optional[str] = Field(None, description="Target variable for feature importance")
    config: FeatureDriftConfig = Field(..., description="Detection configuration")
    enable_alerts: bool = Field(True, description="Enable drift alerts")


class FeatureDriftResponse(BaseModel):
    """Response model for feature drift detection."""
    detector_id: str = Field(..., description="Detector ID")
    drift_detected: bool = Field(..., description="Whether drift was detected")
    drift_score: float = Field(..., description="Overall drift score")
    feature_drifts: Dict[str, Dict[str, Any]] = Field(..., description="Feature-level drift results")
    feature_importance: Optional[Dict[str, float]] = Field(None, description="Feature importance scores")
    correlation_analysis: Optional[Dict[str, Any]] = Field(None, description="Correlation analysis results")
    recommendations: List[str] = Field(..., description="Recommendations based on drift")
    timestamp: str = Field(..., description="Detection timestamp")


class FeatureStabilityMetrics(BaseModel):
    """Model for feature stability metrics."""
    feature_name: str = Field(..., description="Feature name")
    stability_score: float = Field(..., description="Feature stability score")
    drift_frequency: float = Field(..., description="Frequency of drift detection")
    importance_score: float = Field(..., description="Feature importance score")
    correlation_stability: float = Field(..., description="Correlation stability score")
    last_drift_time: Optional[str] = Field(None, description="Last drift detection time")


class FeatureDriftDetector:
    """
    Feature drift detector for PBF-LB/M processes.
    
    This detector provides comprehensive feature drift detection capabilities for:
    - Feature-level drift detection
    - Feature importance tracking
    - Feature stability analysis
    - Correlation drift monitoring
    """
    
    def __init__(self, config_manager: Optional[ConfigManager] = None):
        """
        Initialize the feature drift detector.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config_manager = config_manager or ConfigManager()
        self.app = FastAPI(
            title="Feature Drift Detector",
            description="Feature drift detection for PBF-LB/M manufacturing",
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
        self.feature_history = {}  # Store feature history
        self.feature_stability = {}  # Store feature stability metrics
        self.drift_alerts = {}  # Store drift alerts
        self.detector_counter = 0
        
        # Service metrics
        self.service_metrics = {
            'total_detectors': 0,
            'active_detectors': 0,
            'total_detections': 0,
            'drift_detected_count': 0,
            'features_monitored': 0,
            'total_alerts': 0,
            'last_detection_time': None
        }
        
        # Setup routes
        self._setup_routes()
        
        logger.info("Initialized FeatureDriftDetector")
    
    def _setup_routes(self):
        """Setup API routes."""
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {
                "status": "healthy",
                "service": "feature_drift_detector",
                "timestamp": datetime.now().isoformat(),
                "metrics": self.service_metrics
            }
        
        @self.app.post("/detect", response_model=FeatureDriftResponse)
        async def detect_feature_drift(request: FeatureDriftRequest):
            """Detect feature drift."""
            return await self._detect_feature_drift(request)
        
        @self.app.post("/detectors", response_model=Dict[str, Any])
        async def create_detector(request: FeatureDriftRequest):
            """Create a new feature drift detector."""
            return await self._create_detector(request)
        
        @self.app.get("/detectors")
        async def list_detectors():
            """List all detectors."""
            return await self._list_detectors()
        
        @self.app.get("/detectors/{detector_id}/status")
        async def get_detector_status(detector_id: str):
            """Get detector status."""
            return await self._get_detector_status(detector_id)
        
        @self.app.get("/detectors/{detector_id}/feature-stability")
        async def get_feature_stability(detector_id: str):
            """Get feature stability metrics."""
            return await self._get_feature_stability(detector_id)
        
        @self.app.get("/features/{feature_name}/history")
        async def get_feature_history(feature_name: str, limit: int = Query(10, ge=1, le=100)):
            """Get feature drift history."""
            return await self._get_feature_history(feature_name, limit)
        
        @self.app.get("/alerts")
        async def list_alerts(limit: int = Query(10, ge=1, le=100)):
            """List feature drift alerts."""
            return await self._list_alerts(limit)
        
        @self.app.post("/alerts/{alert_id}/resolve")
        async def resolve_alert(alert_id: str):
            """Resolve a feature drift alert."""
            return await self._resolve_alert(alert_id)
        
        @self.app.get("/metrics")
        async def get_metrics():
            """Get service metrics."""
            return self.service_metrics
    
    async def _detect_feature_drift(self, request: FeatureDriftRequest) -> FeatureDriftResponse:
        """
        Detect feature drift.
        
        Args:
            request: Feature drift detection request
            
        Returns:
            Feature drift detection response
        """
        try:
            # Convert data to DataFrames
            reference_df = pd.DataFrame(request.reference_data)
            current_df = pd.DataFrame(request.current_data)
            
            # Initialize results
            feature_drifts = {}
            drift_scores = []
            
            # Detect drift for each feature
            for feature in request.config.features:
                if feature in reference_df.columns and feature in current_df.columns:
                    drift_result = await self._detect_single_feature_drift(
                        reference_df[feature],
                        current_df[feature],
                        feature,
                        request.config
                    )
                    
                    feature_drifts[feature] = drift_result
                    drift_scores.append(drift_result['drift_score'])
            
            # Calculate overall drift score
            overall_drift_score = np.mean(drift_scores) if drift_scores else 0.0
            drift_detected = overall_drift_score > 0.1  # Default threshold
            
            # Calculate feature importance if target is provided
            feature_importance = None
            if request.target_variable and request.target_variable in current_df.columns:
                feature_importance = await self._calculate_feature_importance(
                    current_df, request.target_variable, request.config.features
                )
            
            # Perform correlation analysis
            correlation_analysis = None
            if request.config.enable_correlation_analysis:
                correlation_analysis = await self._analyze_correlation_drift(
                    reference_df, current_df, request.config.features
                )
            
            # Update feature stability metrics
            await self._update_feature_stability(request.detector_id, feature_drifts)
            
            # Generate recommendations
            recommendations = await self._generate_recommendations(
                feature_drifts, overall_drift_score, feature_importance
            )
            
            # Create alert if drift detected and alerts enabled
            if drift_detected and request.enable_alerts:
                await self._create_feature_drift_alert(
                    request.detector_id, overall_drift_score, feature_drifts
                )
            
            # Update metrics
            self.service_metrics['total_detections'] += 1
            if drift_detected:
                self.service_metrics['drift_detected_count'] += 1
            self.service_metrics['last_detection_time'] = datetime.now().isoformat()
            
            return FeatureDriftResponse(
                detector_id=request.detector_id,
                drift_detected=drift_detected,
                drift_score=overall_drift_score,
                feature_drifts=feature_drifts,
                feature_importance=feature_importance,
                correlation_analysis=correlation_analysis,
                recommendations=recommendations,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"Error in feature drift detection: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def _detect_single_feature_drift(self, reference_data: pd.Series, 
                                         current_data: pd.Series, feature_name: str,
                                         config: FeatureDriftConfig) -> Dict[str, Any]:
        """Detect drift for a single feature."""
        # Ensure minimum samples
        if len(reference_data) < config.min_samples or len(current_data) < config.min_samples:
            return {
                'feature_name': feature_name,
                'drift_detected': False,
                'drift_score': 0.0,
                'reason': 'Insufficient samples',
                'detection_methods': {}
            }
        
        # Remove missing values
        reference_clean = reference_data.dropna()
        current_clean = current_data.dropna()
        
        if len(reference_clean) < config.min_samples or len(current_clean) < config.min_samples:
            return {
                'feature_name': feature_name,
                'drift_detected': False,
                'drift_score': 0.0,
                'reason': 'Insufficient samples after cleaning',
                'detection_methods': {}
            }
        
        # Perform drift detection using specified methods
        detection_results = {}
        drift_scores = []
        
        for method in config.detection_methods:
            try:
                if method == "ks_test":
                    result = await self._ks_test_drift(reference_clean, current_clean)
                elif method == "psi":
                    result = await self._psi_drift(reference_clean, current_clean)
                elif method == "chi_square":
                    result = await self._chi_square_drift(reference_clean, current_clean)
                else:
                    continue
                
                detection_results[method] = result
                drift_scores.append(result['drift_score'])
                
            except Exception as e:
                logger.warning(f"Error in {method} for feature {feature_name}: {e}")
                detection_results[method] = {'error': str(e), 'drift_score': 0.0}
        
        # Calculate overall drift score
        overall_drift_score = np.mean(drift_scores) if drift_scores else 0.0
        drift_detected = overall_drift_score > 0.1  # Default threshold
        
        return {
            'feature_name': feature_name,
            'drift_detected': drift_detected,
            'drift_score': overall_drift_score,
            'detection_methods': detection_results,
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
        }
    
    async def _ks_test_drift(self, reference_data: pd.Series, current_data: pd.Series) -> Dict[str, Any]:
        """Perform Kolmogorov-Smirnov test for drift detection."""
        try:
            ks_statistic, ks_p_value = ks_2samp(reference_data, current_data)
            
            return {
                'method': 'ks_test',
                'statistic': float(ks_statistic),
                'p_value': float(ks_p_value),
                'drift_detected': ks_p_value < 0.05,
                'drift_score': float(ks_statistic)
            }
        except Exception as e:
            return {
                'method': 'ks_test',
                'error': str(e),
                'drift_score': 0.0
            }
    
    async def _psi_drift(self, reference_data: pd.Series, current_data: pd.Series) -> Dict[str, Any]:
        """Calculate Population Stability Index (PSI) for drift detection."""
        try:
            # Create bins for PSI calculation
            if pd.api.types.is_numeric_dtype(reference_data):
                # For numeric data, create bins
                bins = np.linspace(
                    min(reference_data.min(), current_data.min()),
                    max(reference_data.max(), current_data.max()),
                    11
                )
                reference_binned = pd.cut(reference_data, bins=bins, include_lowest=True)
                current_binned = pd.cut(current_data, bins=bins, include_lowest=True)
            else:
                # For categorical data, use unique values
                reference_binned = reference_data
                current_binned = current_data
            
            # Calculate PSI
            ref_counts = reference_binned.value_counts(normalize=True)
            curr_counts = current_binned.value_counts(normalize=True)
            
            # Align counts
            all_categories = set(ref_counts.index) | set(curr_counts.index)
            psi_score = 0.0
            
            for category in all_categories:
                ref_pct = ref_counts.get(category, 0.0001)  # Avoid division by zero
                curr_pct = curr_counts.get(category, 0.0001)
                
                if ref_pct > 0 and curr_pct > 0:
                    psi_score += (curr_pct - ref_pct) * np.log(curr_pct / ref_pct)
            
            return {
                'method': 'psi',
                'psi_score': float(psi_score),
                'drift_detected': psi_score > 0.2,
                'drift_score': float(psi_score)
            }
        except Exception as e:
            return {
                'method': 'psi',
                'error': str(e),
                'drift_score': 0.0
            }
    
    async def _chi_square_drift(self, reference_data: pd.Series, current_data: pd.Series) -> Dict[str, Any]:
        """Perform Chi-square test for categorical drift detection."""
        try:
            if pd.api.types.is_numeric_dtype(reference_data):
                # For numeric data, create bins
                bins = np.linspace(
                    min(reference_data.min(), current_data.min()),
                    max(reference_data.max(), current_data.max()),
                    11
                )
                reference_binned = pd.cut(reference_data, bins=bins, include_lowest=True)
                current_binned = pd.cut(current_data, bins=bins, include_lowest=True)
            else:
                reference_binned = reference_data
                current_binned = current_data
            
            # Create contingency table
            ref_counts = reference_binned.value_counts()
            curr_counts = current_binned.value_counts()
            
            all_categories = set(ref_counts.index) | set(curr_counts.index)
            contingency_table = []
            
            for category in all_categories:
                contingency_table.append([
                    ref_counts.get(category, 0),
                    curr_counts.get(category, 0)
                ])
            
            if len(contingency_table) > 1:
                chi2_statistic, chi2_p_value, dof, expected = chi2_contingency(contingency_table)
                
                return {
                    'method': 'chi_square',
                    'statistic': float(chi2_statistic),
                    'p_value': float(chi2_p_value),
                    'degrees_of_freedom': int(dof),
                    'drift_detected': chi2_p_value < 0.05,
                    'drift_score': float(chi2_statistic / (len(reference_data) + len(current_data)))
                }
            else:
                return {
                    'method': 'chi_square',
                    'error': 'Insufficient categories for chi-square test',
                    'drift_score': 0.0
                }
        except Exception as e:
            return {
                'method': 'chi_square',
                'error': str(e),
                'drift_score': 0.0
            }
    
    async def _calculate_feature_importance(self, data: pd.DataFrame, target: str, 
                                          features: List[str]) -> Dict[str, float]:
        """Calculate feature importance scores."""
        try:
            # Prepare data
            X = data[features].fillna(0)
            y = data[target].fillna(0)
            
            # Determine if it's classification or regression
            if len(y.unique()) <= 10 and np.all(np.isfinite(y)):
                # Classification
                importance_scores = mutual_info_classif(X, y, random_state=42)
            else:
                # Regression
                importance_scores = mutual_info_regression(X, y, random_state=42)
            
            # Create feature importance dictionary
            feature_importance = {}
            for i, feature in enumerate(features):
                feature_importance[feature] = float(importance_scores[i])
            
            return feature_importance
            
        except Exception as e:
            logger.warning(f"Error calculating feature importance: {e}")
            return {}
    
    async def _analyze_correlation_drift(self, reference_df: pd.DataFrame, 
                                       current_df: pd.DataFrame, 
                                       features: List[str]) -> Dict[str, Any]:
        """Analyze correlation drift between features."""
        try:
            # Calculate correlation matrices
            ref_corr = reference_df[features].corr()
            curr_corr = current_df[features].corr()
            
            # Calculate correlation drift
            correlation_drift = {}
            for feature1 in features:
                for feature2 in features:
                    if feature1 != feature2:
                        ref_corr_val = ref_corr.loc[feature1, feature2]
                        curr_corr_val = curr_corr.loc[feature1, feature2]
                        
                        if not np.isnan(ref_corr_val) and not np.isnan(curr_corr_val):
                            drift = abs(ref_corr_val - curr_corr_val)
                            correlation_drift[f"{feature1}_{feature2}"] = {
                                'reference_correlation': float(ref_corr_val),
                                'current_correlation': float(curr_corr_val),
                                'drift': float(drift)
                            }
            
            return {
                'correlation_drift': correlation_drift,
                'reference_correlation_matrix': ref_corr.to_dict(),
                'current_correlation_matrix': curr_corr.to_dict()
            }
            
        except Exception as e:
            logger.warning(f"Error in correlation analysis: {e}")
            return {'error': str(e)}
    
    async def _update_feature_stability(self, detector_id: str, feature_drifts: Dict[str, Any]):
        """Update feature stability metrics."""
        if detector_id not in self.feature_stability:
            self.feature_stability[detector_id] = {}
        
        for feature_name, drift_info in feature_drifts.items():
            if feature_name not in self.feature_stability[detector_id]:
                self.feature_stability[detector_id][feature_name] = {
                    'drift_count': 0,
                    'total_checks': 0,
                    'last_drift_time': None,
                    'stability_score': 1.0
                }
            
            stability_info = self.feature_stability[detector_id][feature_name]
            stability_info['total_checks'] += 1
            
            if drift_info['drift_detected']:
                stability_info['drift_count'] += 1
                stability_info['last_drift_time'] = datetime.now().isoformat()
            
            # Calculate stability score
            stability_info['stability_score'] = 1.0 - (stability_info['drift_count'] / stability_info['total_checks'])
    
    async def _generate_recommendations(self, feature_drifts: Dict[str, Any], 
                                      overall_drift_score: float, 
                                      feature_importance: Optional[Dict[str, float]]) -> List[str]:
        """Generate recommendations based on feature drift detection results."""
        recommendations = []
        
        if overall_drift_score > 0.2:
            recommendations.append("Critical feature drift detected - immediate investigation required")
        elif overall_drift_score > 0.1:
            recommendations.append("Significant feature drift detected - monitor closely")
        
        # Feature-specific recommendations
        high_importance_features = []
        if feature_importance:
            high_importance_features = [f for f, score in feature_importance.items() if score > 0.1]
        
        for feature, drift_info in feature_drifts.items():
            if drift_info['drift_detected']:
                if feature in high_importance_features:
                    recommendations.append(f"High-importance feature '{feature}' shows drift - consider feature engineering or model retraining")
                else:
                    recommendations.append(f"Feature '{feature}' shows drift - monitor for impact on model performance")
        
        if not recommendations:
            recommendations.append("No significant feature drift detected - continue monitoring")
        
        return recommendations
    
    async def _create_feature_drift_alert(self, detector_id: str, drift_score: float, 
                                        feature_drifts: Dict[str, Any]):
        """Create a feature drift alert."""
        alert_id = f"feature_drift_alert_{int(time.time())}"
        
        # Determine alert severity
        if drift_score > 0.2:
            severity = "critical"
        elif drift_score > 0.1:
            severity = "warning"
        else:
            severity = "info"
        
        # Get affected features
        affected_features = [feature for feature, drift_info in feature_drifts.items() 
                           if drift_info['drift_detected']]
        
        alert = {
            'alert_id': alert_id,
            'detector_id': detector_id,
            'alert_type': 'feature_drift',
            'severity': severity,
            'drift_score': drift_score,
            'affected_features': affected_features,
            'message': f"Feature drift detected with score {drift_score:.3f} affecting {len(affected_features)} features",
            'timestamp': datetime.now().isoformat(),
            'resolved': False
        }
        
        self.drift_alerts[alert_id] = alert
        self.service_metrics['total_alerts'] += 1
        
        logger.warning(f"Feature drift alert created: {alert['message']}")
    
    async def _create_detector(self, request: FeatureDriftRequest) -> Dict[str, Any]:
        """Create a new feature drift detector."""
        self.detector_counter += 1
        detector_id = f"feature_detector_{self.detector_counter}_{int(time.time())}"
        
        detector_info = {
            'detector_id': detector_id,
            'features': request.config.features,
            'config': request.config.dict(),
            'created_at': datetime.now().isoformat(),
            'status': 'active',
            'total_detections': 0,
            'drift_detected_count': 0
        }
        
        self.detectors[detector_id] = detector_info
        
        self.service_metrics['total_detectors'] += 1
        self.service_metrics['active_detectors'] += 1
        self.service_metrics['features_monitored'] += len(request.config.features)
        
        return {
            'detector_id': detector_id,
            'status': 'created',
            'features_monitored': len(request.config.features),
            'message': 'Feature drift detector created successfully',
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
    
    async def _get_feature_stability(self, detector_id: str) -> Dict[str, Any]:
        """Get feature stability metrics."""
        if detector_id not in self.feature_stability:
            raise HTTPException(status_code=404, detail=f"Detector {detector_id} not found")
        
        stability_metrics = []
        for feature_name, stability_info in self.feature_stability[detector_id].items():
            stability_metrics.append(FeatureStabilityMetrics(
                feature_name=feature_name,
                stability_score=stability_info['stability_score'],
                drift_frequency=stability_info['drift_count'] / max(stability_info['total_checks'], 1),
                importance_score=0.0,  # Would be calculated separately
                correlation_stability=0.0,  # Would be calculated separately
                last_drift_time=stability_info['last_drift_time']
            ))
        
        return {
            'detector_id': detector_id,
            'feature_stability': [metric.dict() for metric in stability_metrics],
            'timestamp': datetime.now().isoformat()
        }
    
    async def _get_feature_history(self, feature_name: str, limit: int) -> Dict[str, Any]:
        """Get feature drift history."""
        # This would implement actual history retrieval
        # For now, return mock data
        return {
            'feature_name': feature_name,
            'history': [],
            'total_records': 0,
            'timestamp': datetime.now().isoformat()
        }
    
    async def _list_alerts(self, limit: int) -> Dict[str, Any]:
        """List feature drift alerts."""
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
        """Resolve a feature drift alert."""
        if alert_id not in self.drift_alerts:
            raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found")
        
        self.drift_alerts[alert_id]['resolved'] = True
        
        return {
            'alert_id': alert_id,
            'status': 'resolved',
            'message': 'Alert resolved successfully',
            'timestamp': datetime.now().isoformat()
        }
    
    def run(self, host: str = "0.0.0.0", port: int = 8015):
        """Run the service."""
        logger.info(f"Starting Feature Drift Detector on {host}:{port}")
        uvicorn.run(self.app, host=host, port=port)


# Global service instance
service = FeatureDriftDetector()


def create_app() -> FastAPI:
    """Create FastAPI application."""
    return service.app


if __name__ == "__main__":
    service.run()
