"""
SHAP Explainer

This module implements the SHAP explainer for PBF-LB/M processes.
It provides SHAP-based model explainability, feature importance analysis,
and interpretable AI insights for ML models.
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

# SHAP imports (would be installed as dependency)
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    shap = None

from ...config.config_manager import ConfigManager

logger = logging.getLogger(__name__)


# Pydantic models for API requests and responses
class SHAPExplanationRequest(BaseModel):
    """Request model for SHAP explanation."""
    explainer_id: str = Field(..., description="Explainer ID")
    model_id: str = Field(..., description="Model ID to explain")
    input_data: Dict[str, Any] = Field(..., description="Input data for explanation")
    explanation_type: str = Field("local", description="Type of explanation (local, global)")
    feature_names: Optional[List[str]] = Field(None, description="Feature names")
    background_data: Optional[Dict[str, Any]] = Field(None, description="Background data for SHAP")
    max_features: Optional[int] = Field(10, description="Maximum number of features to explain")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class SHAPExplanationResponse(BaseModel):
    """Response model for SHAP explanation."""
    explainer_id: str = Field(..., description="Explainer ID")
    model_id: str = Field(..., description="Model ID")
    explanation_type: str = Field(..., description="Type of explanation")
    shap_values: Dict[str, Any] = Field(..., description="SHAP values")
    feature_importance: Dict[str, float] = Field(..., description="Feature importance scores")
    base_value: float = Field(..., description="Base value (expected output)")
    prediction: Any = Field(..., description="Model prediction")
    explanation_summary: str = Field(..., description="Explanation summary")
    visualizations: Optional[Dict[str, Any]] = Field(None, description="SHAP visualizations")
    timestamp: str = Field(..., description="Explanation timestamp")


class SHAPExplainerConfig(BaseModel):
    """Configuration for SHAP explainer."""
    explainer_type: str = Field("TreeExplainer", description="Type of SHAP explainer")
    background_samples: int = Field(100, description="Number of background samples")
    max_evals: int = Field(1000, description="Maximum evaluations for KernelExplainer")
    feature_perturbation: str = Field("interventional", description="Feature perturbation method")
    model_output: str = Field("raw", description="Model output type")
    link: str = Field("identity", description="Link function")


class SHAPExplainer:
    """
    SHAP explainer for PBF-LB/M processes.
    
    This explainer provides comprehensive SHAP-based explainability capabilities for:
    - Local and global model explanations
    - Feature importance analysis
    - SHAP value computation
    - Model interpretability insights
    """
    
    def __init__(self, config_manager: Optional[ConfigManager] = None):
        """
        Initialize the SHAP explainer.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config_manager = config_manager or ConfigManager()
        self.app = FastAPI(
            title="SHAP Explainer",
            description="SHAP-based model explainability for PBF-LB/M manufacturing",
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
        
        # SHAP explainer management
        self.explainers = {}  # Store explainer information
        self.explanation_history = {}  # Store explanation history
        self.explainer_counter = 0
        
        # Service metrics
        self.service_metrics = {
            'total_explainers': 0,
            'active_explainers': 0,
            'total_explanations': 0,
            'models_explained': 0,
            'last_explanation_time': None
        }
        
        # Check SHAP availability
        if not SHAP_AVAILABLE:
            logger.warning("SHAP library not available. Install with: pip install shap")
        
        # Setup routes
        self._setup_routes()
        
        logger.info("Initialized SHAPExplainer")
    
    def _setup_routes(self):
        """Setup API routes."""
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {
                "status": "healthy",
                "service": "shap_explainer",
                "shap_available": SHAP_AVAILABLE,
                "timestamp": datetime.now().isoformat(),
                "metrics": self.service_metrics
            }
        
        @self.app.post("/explain", response_model=SHAPExplanationResponse)
        async def explain_model(request: SHAPExplanationRequest):
            """Explain model predictions using SHAP."""
            return await self._explain_model(request)
        
        @self.app.post("/explainers", response_model=Dict[str, Any])
        async def create_explainer(request: SHAPExplanationRequest):
            """Create a new SHAP explainer."""
            return await self._create_explainer(request)
        
        @self.app.get("/explainers")
        async def list_explainers():
            """List all explainers."""
            return await self._list_explainers()
        
        @self.app.get("/explainers/{explainer_id}/status")
        async def get_explainer_status(explainer_id: str):
            """Get explainer status."""
            return await self._get_explainer_status(explainer_id)
        
        @self.app.get("/models/{model_id}/explanations")
        async def get_model_explanations(model_id: str, limit: int = Query(10, ge=1, le=100)):
            """Get model explanation history."""
            return await self._get_model_explanations(model_id, limit)
        
        @self.app.get("/models/{model_id}/feature-importance")
        async def get_feature_importance(model_id: str, days: int = Query(7, ge=1, le=30)):
            """Get feature importance analysis for a model."""
            return await self._get_feature_importance(model_id, days)
        
        @self.app.get("/models/{model_id}/global-explanation")
        async def get_global_explanation(model_id: str):
            """Get global model explanation."""
            return await self._get_global_explanation(model_id)
        
        @self.app.get("/metrics")
        async def get_metrics():
            """Get service metrics."""
            return self.service_metrics
    
    async def _explain_model(self, request: SHAPExplanationRequest) -> SHAPExplanationResponse:
        """
        Explain model predictions using SHAP.
        
        Args:
            request: SHAP explanation request
            
        Returns:
            SHAP explanation response
        """
        if not SHAP_AVAILABLE:
            raise HTTPException(status_code=500, detail="SHAP library not available")
        
        try:
            # Get or create explainer
            explainer = await self._get_or_create_explainer(request.model_id, request.explainer_id)
            
            # Prepare input data
            input_data = self._prepare_input_data(request.input_data, request.feature_names)
            
            # Generate SHAP explanation
            if request.explanation_type == "local":
                explanation = await self._generate_local_explanation(
                    explainer, input_data, request
                )
            else:
                explanation = await self._generate_global_explanation(
                    explainer, input_data, request
                )
            
            # Store explanation history
            await self._store_explanation_history(request.model_id, explanation)
            
            # Update metrics
            self.service_metrics['total_explanations'] += 1
            self.service_metrics['last_explanation_time'] = datetime.now().isoformat()
            
            return explanation
            
        except Exception as e:
            logger.error(f"Error explaining model: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def _get_or_create_explainer(self, model_id: str, explainer_id: str):
        """Get or create SHAP explainer for a model."""
        if explainer_id in self.explainers:
            return self.explainers[explainer_id]['explainer']
        
        # Create new explainer
        # This would implement actual SHAP explainer creation
        # For now, return a mock explainer
        mock_explainer = f"shap_explainer_{model_id}"
        
        explainer_info = {
            'explainer_id': explainer_id,
            'model_id': model_id,
            'explainer': mock_explainer,
            'created_at': datetime.now().isoformat(),
            'status': 'active'
        }
        
        self.explainers[explainer_id] = explainer_info
        
        return mock_explainer
    
    def _prepare_input_data(self, input_data: Dict[str, Any], feature_names: Optional[List[str]]) -> np.ndarray:
        """Prepare input data for SHAP explanation."""
        # Convert input data to numpy array
        if isinstance(input_data, dict):
            if feature_names:
                # Use specified feature order
                data_array = np.array([input_data.get(feature, 0) for feature in feature_names])
            else:
                # Use dictionary values in order
                data_array = np.array(list(input_data.values()))
        else:
            data_array = np.array(input_data)
        
        # Ensure 2D array for SHAP
        if data_array.ndim == 1:
            data_array = data_array.reshape(1, -1)
        
        return data_array
    
    async def _generate_local_explanation(self, explainer: Any, input_data: np.ndarray, 
                                        request: SHAPExplanationRequest) -> SHAPExplanationResponse:
        """Generate local SHAP explanation."""
        try:
            # This would implement actual SHAP explanation
            # For now, return mock explanation
            n_features = input_data.shape[1]
            feature_names = request.feature_names or [f"feature_{i}" for i in range(n_features)]
            
            # Mock SHAP values
            shap_values = np.random.normal(0, 1, n_features)
            base_value = np.random.normal(0, 0.5)
            prediction = base_value + np.sum(shap_values)
            
            # Create feature importance dictionary
            feature_importance = {
                feature_names[i]: float(shap_values[i]) 
                for i in range(n_features)
            }
            
            # Sort by absolute importance
            sorted_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)
            
            # Generate explanation summary
            explanation_summary = self._generate_explanation_summary(sorted_features, prediction)
            
            # Generate visualizations
            visualizations = await self._generate_shap_visualizations(
                shap_values, feature_names, base_value, prediction
            )
            
            return SHAPExplanationResponse(
                explainer_id=request.explainer_id,
                model_id=request.model_id,
                explanation_type="local",
                shap_values={
                    'values': shap_values.tolist(),
                    'feature_names': feature_names,
                    'data': input_data.tolist()
                },
                feature_importance=dict(sorted_features[:request.max_features or 10]),
                base_value=float(base_value),
                prediction=float(prediction),
                explanation_summary=explanation_summary,
                visualizations=visualizations,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"Error generating local explanation: {e}")
            raise
    
    async def _generate_global_explanation(self, explainer: Any, input_data: np.ndarray, 
                                         request: SHAPExplanationRequest) -> SHAPExplanationResponse:
        """Generate global SHAP explanation."""
        try:
            # This would implement actual global SHAP explanation
            # For now, return mock explanation
            n_features = input_data.shape[1]
            feature_names = request.feature_names or [f"feature_{i}" for i in range(n_features)]
            
            # Mock global SHAP values (average across multiple samples)
            n_samples = 100
            global_shap_values = np.random.normal(0, 0.5, n_features)
            base_value = np.random.normal(0, 0.3)
            
            # Create feature importance dictionary
            feature_importance = {
                feature_names[i]: float(global_shap_values[i]) 
                for i in range(n_features)
            }
            
            # Sort by absolute importance
            sorted_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)
            
            # Generate explanation summary
            explanation_summary = self._generate_global_explanation_summary(sorted_features)
            
            # Generate visualizations
            visualizations = await self._generate_global_shap_visualizations(
                global_shap_values, feature_names, base_value
            )
            
            return SHAPExplanationResponse(
                explainer_id=request.explainer_id,
                model_id=request.model_id,
                explanation_type="global",
                shap_values={
                    'values': global_shap_values.tolist(),
                    'feature_names': feature_names,
                    'n_samples': n_samples
                },
                feature_importance=dict(sorted_features[:request.max_features or 10]),
                base_value=float(base_value),
                prediction=None,  # Global explanation doesn't have single prediction
                explanation_summary=explanation_summary,
                visualizations=visualizations,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"Error generating global explanation: {e}")
            raise
    
    def _generate_explanation_summary(self, sorted_features: List[Tuple[str, float]], 
                                    prediction: float) -> str:
        """Generate explanation summary for local explanation."""
        try:
            # Get top positive and negative features
            positive_features = [f for f, v in sorted_features if v > 0][:3]
            negative_features = [f for f, v in sorted_features if v < 0][:3]
            
            summary_parts = []
            
            if positive_features:
                summary_parts.append(f"Features increasing prediction: {', '.join(positive_features)}")
            
            if negative_features:
                summary_parts.append(f"Features decreasing prediction: {', '.join(negative_features)}")
            
            # Add prediction context
            if prediction > 0.5:
                summary_parts.append("Overall prediction is positive")
            else:
                summary_parts.append("Overall prediction is negative")
            
            return ". ".join(summary_parts) + "."
            
        except Exception as e:
            logger.error(f"Error generating explanation summary: {e}")
            return "Unable to generate explanation summary"
    
    def _generate_global_explanation_summary(self, sorted_features: List[Tuple[str, float]]) -> str:
        """Generate explanation summary for global explanation."""
        try:
            # Get top features by importance
            top_features = sorted_features[:5]
            
            summary_parts = []
            summary_parts.append("Global feature importance ranking:")
            
            for i, (feature, importance) in enumerate(top_features, 1):
                direction = "increases" if importance > 0 else "decreases"
                summary_parts.append(f"{i}. {feature} ({direction} predictions)")
            
            return ". ".join(summary_parts) + "."
            
        except Exception as e:
            logger.error(f"Error generating global explanation summary: {e}")
            return "Unable to generate global explanation summary"
    
    async def _generate_shap_visualizations(self, shap_values: np.ndarray, feature_names: List[str],
                                          base_value: float, prediction: float) -> Dict[str, Any]:
        """Generate SHAP visualizations."""
        try:
            visualizations = {}
            
            # Waterfall plot
            fig = go.Figure()
            
            # Add base value
            fig.add_trace(go.Bar(
                x=['Base Value'],
                y=[base_value],
                name='Base Value',
                marker_color='lightblue'
            ))
            
            # Add feature contributions
            for i, (feature, value) in enumerate(zip(feature_names, shap_values)):
                color = 'green' if value > 0 else 'red'
                fig.add_trace(go.Bar(
                    x=[feature],
                    y=[value],
                    name=feature,
                    marker_color=color
                ))
            
            # Add prediction
            fig.add_trace(go.Bar(
                x=['Prediction'],
                y=[prediction],
                name='Prediction',
                marker_color='darkblue'
            ))
            
            fig.update_layout(
                title="SHAP Waterfall Plot",
                xaxis_title="Features",
                yaxis_title="SHAP Value",
                barmode='relative'
            )
            
            visualizations['waterfall'] = json.loads(fig.to_json())
            
            # Feature importance plot
            feature_importance_data = list(zip(feature_names, np.abs(shap_values)))
            feature_importance_data.sort(key=lambda x: x[1], reverse=True)
            
            features, importances = zip(*feature_importance_data)
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=list(importances),
                y=list(features),
                orientation='h',
                marker_color='lightcoral'
            ))
            
            fig.update_layout(
                title="SHAP Feature Importance",
                xaxis_title="Absolute SHAP Value",
                yaxis_title="Features"
            )
            
            visualizations['feature_importance'] = json.loads(fig.to_json())
            
            return visualizations
            
        except Exception as e:
            logger.error(f"Error generating SHAP visualizations: {e}")
            return {}
    
    async def _generate_global_shap_visualizations(self, shap_values: np.ndarray, 
                                                 feature_names: List[str], 
                                                 base_value: float) -> Dict[str, Any]:
        """Generate global SHAP visualizations."""
        try:
            visualizations = {}
            
            # Global feature importance plot
            feature_importance_data = list(zip(feature_names, np.abs(shap_values)))
            feature_importance_data.sort(key=lambda x: x[1], reverse=True)
            
            features, importances = zip(*feature_importance_data)
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=list(importances),
                y=list(features),
                orientation='h',
                marker_color='lightgreen'
            ))
            
            fig.update_layout(
                title="Global SHAP Feature Importance",
                xaxis_title="Average Absolute SHAP Value",
                yaxis_title="Features"
            )
            
            visualizations['global_importance'] = json.loads(fig.to_json())
            
            # SHAP values distribution
            fig = go.Figure()
            fig.add_trace(go.Box(
                y=shap_values,
                name='SHAP Values Distribution',
                marker_color='lightblue'
            ))
            
            fig.update_layout(
                title="SHAP Values Distribution",
                yaxis_title="SHAP Value"
            )
            
            visualizations['distribution'] = json.loads(fig.to_json())
            
            return visualizations
            
        except Exception as e:
            logger.error(f"Error generating global SHAP visualizations: {e}")
            return {}
    
    async def _store_explanation_history(self, model_id: str, explanation: SHAPExplanationResponse):
        """Store explanation history."""
        if model_id not in self.explanation_history:
            self.explanation_history[model_id] = []
        
        explanation_record = {
            'timestamp': explanation.timestamp,
            'explanation_type': explanation.explanation_type,
            'feature_importance': explanation.feature_importance,
            'base_value': explanation.base_value,
            'prediction': explanation.prediction
        }
        
        self.explanation_history[model_id].append(explanation_record)
        
        # Keep only recent history (last 1000 explanations)
        if len(self.explanation_history[model_id]) > 1000:
            self.explanation_history[model_id] = self.explanation_history[model_id][-1000:]
    
    async def _create_explainer(self, request: SHAPExplanationRequest) -> Dict[str, Any]:
        """Create a new SHAP explainer."""
        self.explainer_counter += 1
        explainer_id = f"explainer_{self.explainer_counter}_{int(time.time())}"
        
        explainer_info = {
            'explainer_id': explainer_id,
            'model_id': request.model_id,
            'explanation_type': request.explanation_type,
            'created_at': datetime.now().isoformat(),
            'status': 'active'
        }
        
        self.explainers[explainer_id] = explainer_info
        
        self.service_metrics['total_explainers'] += 1
        self.service_metrics['active_explainers'] += 1
        self.service_metrics['models_explained'] += 1
        
        return {
            'explainer_id': explainer_id,
            'model_id': request.model_id,
            'status': 'created',
            'message': 'SHAP explainer created successfully',
            'timestamp': datetime.now().isoformat()
        }
    
    async def _list_explainers(self) -> Dict[str, Any]:
        """List all explainers."""
        explainer_list = []
        for explainer_id, explainer_info in self.explainers.items():
            explainer_list.append({
                'explainer_id': explainer_id,
                'model_id': explainer_info['model_id'],
                'explanation_type': explainer_info['explanation_type'],
                'status': explainer_info['status'],
                'created_at': explainer_info['created_at']
            })
        
        return {
            'explainers': explainer_list,
            'total_explainers': len(explainer_list),
            'active_explainers': sum(1 for e in explainer_list if e['status'] == 'active'),
            'timestamp': datetime.now().isoformat()
        }
    
    async def _get_explainer_status(self, explainer_id: str) -> Dict[str, Any]:
        """Get explainer status."""
        if explainer_id not in self.explainers:
            raise HTTPException(status_code=404, detail=f"Explainer {explainer_id} not found")
        
        explainer_info = self.explainers[explainer_id]
        
        return {
            'explainer_id': explainer_id,
            'model_id': explainer_info['model_id'],
            'explanation_type': explainer_info['explanation_type'],
            'status': explainer_info['status'],
            'created_at': explainer_info['created_at']
        }
    
    async def _get_model_explanations(self, model_id: str, limit: int) -> Dict[str, Any]:
        """Get model explanation history."""
        if model_id not in self.explanation_history:
            raise HTTPException(status_code=404, detail=f"No explanations found for model {model_id}")
        
        history = self.explanation_history[model_id]
        limited_history = history[-limit:] if limit > 0 else history
        
        return {
            'model_id': model_id,
            'explanation_history': limited_history,
            'total_explanations': len(history),
            'timestamp': datetime.now().isoformat()
        }
    
    async def _get_feature_importance(self, model_id: str, days: int) -> Dict[str, Any]:
        """Get feature importance analysis for a model."""
        if model_id not in self.explanation_history:
            raise HTTPException(status_code=404, detail=f"No explanations found for model {model_id}")
        
        history = self.explanation_history[model_id]
        
        # Filter by time period
        cutoff_time = datetime.now() - timedelta(days=days)
        recent_history = [
            h for h in history 
            if datetime.fromisoformat(h['timestamp']) > cutoff_time
        ]
        
        if not recent_history:
            raise HTTPException(status_code=404, detail=f"No explanations found for the last {days} days")
        
        # Aggregate feature importance across explanations
        feature_importance_aggregated = {}
        feature_counts = {}
        
        for explanation in recent_history:
            for feature, importance in explanation['feature_importance'].items():
                if feature not in feature_importance_aggregated:
                    feature_importance_aggregated[feature] = 0
                    feature_counts[feature] = 0
                
                feature_importance_aggregated[feature] += abs(importance)
                feature_counts[feature] += 1
        
        # Calculate average importance
        average_importance = {
            feature: importance / feature_counts[feature]
            for feature, importance in feature_importance_aggregated.items()
        }
        
        # Sort by importance
        sorted_importance = sorted(average_importance.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'model_id': model_id,
            'time_period': f"{days} days",
            'total_explanations': len(recent_history),
            'feature_importance': dict(sorted_importance),
            'top_features': [f[0] for f in sorted_importance[:10]],
            'timestamp': datetime.now().isoformat()
        }
    
    async def _get_global_explanation(self, model_id: str) -> Dict[str, Any]:
        """Get global model explanation."""
        if model_id not in self.explanation_history:
            raise HTTPException(status_code=404, detail=f"No explanations found for model {model_id}")
        
        history = self.explanation_history[model_id]
        
        if not history:
            raise HTTPException(status_code=404, detail=f"No explanations found for model {model_id}")
        
        # Aggregate all feature importance
        all_feature_importance = {}
        explanation_count = 0
        
        for explanation in history:
            if explanation['explanation_type'] == 'local':
                for feature, importance in explanation['feature_importance'].items():
                    if feature not in all_feature_importance:
                        all_feature_importance[feature] = []
                    all_feature_importance[feature].append(abs(importance))
                explanation_count += 1
        
        # Calculate global importance
        global_importance = {}
        for feature, importances in all_feature_importance.items():
            global_importance[feature] = {
                'mean_importance': np.mean(importances),
                'std_importance': np.std(importances),
                'count': len(importances)
            }
        
        # Sort by mean importance
        sorted_global = sorted(global_importance.items(), key=lambda x: x[1]['mean_importance'], reverse=True)
        
        return {
            'model_id': model_id,
            'explanation_count': explanation_count,
            'global_feature_importance': dict(sorted_global),
            'top_features': [f[0] for f in sorted_global[:10]],
            'timestamp': datetime.now().isoformat()
        }
    
    def run(self, host: str = "0.0.0.0", port: int = 8021):
        """Run the service."""
        logger.info(f"Starting SHAP Explainer on {host}:{port}")
        uvicorn.run(self.app, host=host, port=port)


# Global service instance
service = SHAPExplainer()


def create_app() -> FastAPI:
    """Create FastAPI application."""
    return service.app


if __name__ == "__main__":
    service.run()
