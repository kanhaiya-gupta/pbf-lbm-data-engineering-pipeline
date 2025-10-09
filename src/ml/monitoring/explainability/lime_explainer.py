"""
LIME Explainer

This module implements the LIME explainer for PBF-LB/M processes.
It provides LIME-based model explainability, local interpretability,
and explainable AI insights for ML models.
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

# LIME imports (would be installed as dependency)
try:
    import lime
    import lime.lime_tabular
    import lime.lime_text
    import lime.lime_image
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    lime = None

from ...config.config_manager import ConfigManager

logger = logging.getLogger(__name__)


# Pydantic models for API requests and responses
class LIMEExplanationRequest(BaseModel):
    """Request model for LIME explanation."""
    explainer_id: str = Field(..., description="Explainer ID")
    model_id: str = Field(..., description="Model ID to explain")
    input_data: Dict[str, Any] = Field(..., description="Input data for explanation")
    data_type: str = Field("tabular", description="Type of data (tabular, text, image)")
    feature_names: Optional[List[str]] = Field(None, description="Feature names")
    class_names: Optional[List[str]] = Field(None, description="Class names for classification")
    num_features: int = Field(10, description="Number of features to explain")
    num_samples: int = Field(5000, description="Number of samples for LIME")
    distance_metric: str = Field("euclidean", description="Distance metric for LIME")
    kernel_width: Optional[float] = Field(None, description="Kernel width for LIME")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class LIMEExplanationResponse(BaseModel):
    """Response model for LIME explanation."""
    explainer_id: str = Field(..., description="Explainer ID")
    model_id: str = Field(..., description="Model ID")
    data_type: str = Field(..., description="Type of data")
    explanation: Dict[str, Any] = Field(..., description="LIME explanation")
    feature_importance: Dict[str, float] = Field(..., description="Feature importance scores")
    prediction: Any = Field(..., description="Model prediction")
    explanation_summary: str = Field(..., description="Explanation summary")
    confidence_score: float = Field(..., description="Explanation confidence score")
    visualizations: Optional[Dict[str, Any]] = Field(None, description="LIME visualizations")
    timestamp: str = Field(..., description="Explanation timestamp")


class LIMEExplainerConfig(BaseModel):
    """Configuration for LIME explainer."""
    explainer_type: str = Field("LimeTabularExplainer", description="Type of LIME explainer")
    num_samples: int = Field(5000, description="Number of samples for LIME")
    distance_metric: str = Field("euclidean", description="Distance metric")
    kernel_width: float = Field(0.75, description="Kernel width")
    feature_selection: str = Field("auto", description="Feature selection method")
    discretize_continuous: bool = Field(True, description="Whether to discretize continuous features")


class LIMEExplainer:
    """
    LIME explainer for PBF-LB/M processes.
    
    This explainer provides comprehensive LIME-based explainability capabilities for:
    - Local model explanations
    - Tabular, text, and image data support
    - Feature importance analysis
    - Model interpretability insights
    """
    
    def __init__(self, config_manager: Optional[ConfigManager] = None):
        """
        Initialize the LIME explainer.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config_manager = config_manager or ConfigManager()
        self.app = FastAPI(
            title="LIME Explainer",
            description="LIME-based model explainability for PBF-LB/M manufacturing",
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
        
        # LIME explainer management
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
        
        # Check LIME availability
        if not LIME_AVAILABLE:
            logger.warning("LIME library not available. Install with: pip install lime")
        
        # Setup routes
        self._setup_routes()
        
        logger.info("Initialized LIMEExplainer")
    
    def _setup_routes(self):
        """Setup API routes."""
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {
                "status": "healthy",
                "service": "lime_explainer",
                "lime_available": LIME_AVAILABLE,
                "timestamp": datetime.now().isoformat(),
                "metrics": self.service_metrics
            }
        
        @self.app.post("/explain", response_model=LIMEExplanationResponse)
        async def explain_model(request: LIMEExplanationRequest):
            """Explain model predictions using LIME."""
            return await self._explain_model(request)
        
        @self.app.post("/explainers", response_model=Dict[str, Any])
        async def create_explainer(request: LIMEExplanationRequest):
            """Create a new LIME explainer."""
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
        
        @self.app.get("/models/{model_id}/explanation-stability")
        async def get_explanation_stability(model_id: str, days: int = Query(7, ge=1, le=30)):
            """Get explanation stability analysis for a model."""
            return await self._get_explanation_stability(model_id, days)
        
        @self.app.get("/metrics")
        async def get_metrics():
            """Get service metrics."""
            return self.service_metrics
    
    async def _explain_model(self, request: LIMEExplanationRequest) -> LIMEExplanationResponse:
        """
        Explain model predictions using LIME.
        
        Args:
            request: LIME explanation request
            
        Returns:
            LIME explanation response
        """
        if not LIME_AVAILABLE:
            raise HTTPException(status_code=500, detail="LIME library not available")
        
        try:
            # Get or create explainer
            explainer = await self._get_or_create_explainer(request.model_id, request.explainer_id, request)
            
            # Prepare input data
            input_data = self._prepare_input_data(request.input_data, request.feature_names, request.data_type)
            
            # Generate LIME explanation
            explanation = await self._generate_lime_explanation(
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
    
    async def _get_or_create_explainer(self, model_id: str, explainer_id: str, 
                                     request: LIMEExplanationRequest):
        """Get or create LIME explainer for a model."""
        if explainer_id in self.explainers:
            return self.explainers[explainer_id]['explainer']
        
        # Create new explainer based on data type
        if request.data_type == "tabular":
            explainer = await self._create_tabular_explainer(request)
        elif request.data_type == "text":
            explainer = await self._create_text_explainer(request)
        elif request.data_type == "image":
            explainer = await self._create_image_explainer(request)
        else:
            raise ValueError(f"Unsupported data type: {request.data_type}")
        
        explainer_info = {
            'explainer_id': explainer_id,
            'model_id': model_id,
            'data_type': request.data_type,
            'explainer': explainer,
            'created_at': datetime.now().isoformat(),
            'status': 'active'
        }
        
        self.explainers[explainer_id] = explainer_info
        
        return explainer
    
    async def _create_tabular_explainer(self, request: LIMEExplanationRequest):
        """Create LIME tabular explainer."""
        # This would implement actual LIME tabular explainer creation
        # For now, return a mock explainer
        return f"lime_tabular_explainer_{request.model_id}"
    
    async def _create_text_explainer(self, request: LIMEExplanationRequest):
        """Create LIME text explainer."""
        # This would implement actual LIME text explainer creation
        # For now, return a mock explainer
        return f"lime_text_explainer_{request.model_id}"
    
    async def _create_image_explainer(self, request: LIMEExplanationRequest):
        """Create LIME image explainer."""
        # This would implement actual LIME image explainer creation
        # For now, return a mock explainer
        return f"lime_image_explainer_{request.model_id}"
    
    def _prepare_input_data(self, input_data: Dict[str, Any], feature_names: Optional[List[str]], 
                           data_type: str) -> Any:
        """Prepare input data for LIME explanation."""
        if data_type == "tabular":
            # Convert to numpy array for tabular data
            if isinstance(input_data, dict):
                if feature_names:
                    data_array = np.array([input_data.get(feature, 0) for feature in feature_names])
                else:
                    data_array = np.array(list(input_data.values()))
            else:
                data_array = np.array(input_data)
            
            # Ensure 2D array
            if data_array.ndim == 1:
                data_array = data_array.reshape(1, -1)
            
            return data_array
        
        elif data_type == "text":
            # Return text string
            if isinstance(input_data, dict):
                return input_data.get('text', '')
            else:
                return str(input_data)
        
        elif data_type == "image":
            # Return image data
            return input_data
        
        else:
            raise ValueError(f"Unsupported data type: {data_type}")
    
    async def _generate_lime_explanation(self, explainer: Any, input_data: Any, 
                                       request: LIMEExplanationRequest) -> LIMEExplanationResponse:
        """Generate LIME explanation."""
        try:
            # This would implement actual LIME explanation
            # For now, return mock explanation
            n_features = request.num_features
            feature_names = request.feature_names or [f"feature_{i}" for i in range(n_features)]
            
            # Mock LIME explanation
            explanation_data = {
                'feature_names': feature_names,
                'feature_values': np.random.normal(0, 1, n_features).tolist(),
                'feature_weights': np.random.normal(0, 0.5, n_features).tolist(),
                'intercept': np.random.normal(0, 0.2),
                'score': np.random.uniform(0.7, 0.95)
            }
            
            # Create feature importance dictionary
            feature_importance = {
                feature_names[i]: float(explanation_data['feature_weights'][i]) 
                for i in range(n_features)
            }
            
            # Sort by absolute importance
            sorted_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)
            
            # Generate prediction
            prediction = explanation_data['intercept'] + sum(explanation_data['feature_weights'])
            
            # Generate explanation summary
            explanation_summary = self._generate_lime_explanation_summary(sorted_features, prediction)
            
            # Calculate confidence score
            confidence_score = explanation_data['score']
            
            # Generate visualizations
            visualizations = await self._generate_lime_visualizations(
                explanation_data, feature_names, prediction
            )
            
            return LIMEExplanationResponse(
                explainer_id=request.explainer_id,
                model_id=request.model_id,
                data_type=request.data_type,
                explanation=explanation_data,
                feature_importance=dict(sorted_features),
                prediction=float(prediction),
                explanation_summary=explanation_summary,
                confidence_score=confidence_score,
                visualizations=visualizations,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"Error generating LIME explanation: {e}")
            raise
    
    def _generate_lime_explanation_summary(self, sorted_features: List[Tuple[str, float]], 
                                         prediction: float) -> str:
        """Generate LIME explanation summary."""
        try:
            # Get top positive and negative features
            positive_features = [f for f, v in sorted_features if v > 0][:3]
            negative_features = [f for f, v in sorted_features if v < 0][:3]
            
            summary_parts = []
            
            if positive_features:
                summary_parts.append(f"Features supporting prediction: {', '.join(positive_features)}")
            
            if negative_features:
                summary_parts.append(f"Features opposing prediction: {', '.join(negative_features)}")
            
            # Add prediction context
            if prediction > 0:
                summary_parts.append("Overall prediction is positive")
            else:
                summary_parts.append("Overall prediction is negative")
            
            return ". ".join(summary_parts) + "."
            
        except Exception as e:
            logger.error(f"Error generating LIME explanation summary: {e}")
            return "Unable to generate explanation summary"
    
    async def _generate_lime_visualizations(self, explanation_data: Dict[str, Any], 
                                          feature_names: List[str], prediction: float) -> Dict[str, Any]:
        """Generate LIME visualizations."""
        try:
            visualizations = {}
            
            # Feature weights plot
            weights = explanation_data['feature_weights']
            colors = ['green' if w > 0 else 'red' for w in weights]
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=feature_names,
                y=weights,
                marker_color=colors,
                name='Feature Weights'
            ))
            
            fig.update_layout(
                title="LIME Feature Weights",
                xaxis_title="Features",
                yaxis_title="Weight",
                xaxis_tickangle=-45
            )
            
            visualizations['feature_weights'] = json.loads(fig.to_json())
            
            # Feature importance plot
            abs_weights = [abs(w) for w in weights]
            feature_importance_data = list(zip(feature_names, abs_weights))
            feature_importance_data.sort(key=lambda x: x[1], reverse=True)
            
            features, importances = zip(*feature_importance_data)
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=list(importances),
                y=list(features),
                orientation='h',
                marker_color='lightblue'
            ))
            
            fig.update_layout(
                title="LIME Feature Importance",
                xaxis_title="Absolute Weight",
                yaxis_title="Features"
            )
            
            visualizations['feature_importance'] = json.loads(fig.to_json())
            
            # Prediction breakdown
            intercept = explanation_data['intercept']
            cumulative_weights = [intercept]
            for weight in weights:
                cumulative_weights.append(cumulative_weights[-1] + weight)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=list(range(len(cumulative_weights))),
                y=cumulative_weights,
                mode='lines+markers',
                name='Cumulative Prediction',
                line=dict(color='blue', width=2)
            ))
            
            fig.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Zero Line")
            
            fig.update_layout(
                title="LIME Prediction Breakdown",
                xaxis_title="Feature Addition",
                yaxis_title="Cumulative Prediction"
            )
            
            visualizations['prediction_breakdown'] = json.loads(fig.to_json())
            
            return visualizations
            
        except Exception as e:
            logger.error(f"Error generating LIME visualizations: {e}")
            return {}
    
    async def _store_explanation_history(self, model_id: str, explanation: LIMEExplanationResponse):
        """Store explanation history."""
        if model_id not in self.explanation_history:
            self.explanation_history[model_id] = []
        
        explanation_record = {
            'timestamp': explanation.timestamp,
            'data_type': explanation.data_type,
            'feature_importance': explanation.feature_importance,
            'prediction': explanation.prediction,
            'confidence_score': explanation.confidence_score
        }
        
        self.explanation_history[model_id].append(explanation_record)
        
        # Keep only recent history (last 1000 explanations)
        if len(self.explanation_history[model_id]) > 1000:
            self.explanation_history[model_id] = self.explanation_history[model_id][-1000:]
    
    async def _create_explainer(self, request: LIMEExplanationRequest) -> Dict[str, Any]:
        """Create a new LIME explainer."""
        self.explainer_counter += 1
        explainer_id = f"explainer_{self.explainer_counter}_{int(time.time())}"
        
        explainer_info = {
            'explainer_id': explainer_id,
            'model_id': request.model_id,
            'data_type': request.data_type,
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
            'data_type': request.data_type,
            'status': 'created',
            'message': 'LIME explainer created successfully',
            'timestamp': datetime.now().isoformat()
        }
    
    async def _list_explainers(self) -> Dict[str, Any]:
        """List all explainers."""
        explainer_list = []
        for explainer_id, explainer_info in self.explainers.items():
            explainer_list.append({
                'explainer_id': explainer_id,
                'model_id': explainer_info['model_id'],
                'data_type': explainer_info['data_type'],
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
            'data_type': explainer_info['data_type'],
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
    
    async def _get_explanation_stability(self, model_id: str, days: int) -> Dict[str, Any]:
        """Get explanation stability analysis for a model."""
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
        
        # Calculate stability metrics
        feature_importance_series = {}
        
        for explanation in recent_history:
            for feature, importance in explanation['feature_importance'].items():
                if feature not in feature_importance_series:
                    feature_importance_series[feature] = []
                feature_importance_series[feature].append(importance)
        
        # Calculate stability for each feature
        feature_stability = {}
        for feature, importance_series in feature_importance_series.items():
            if len(importance_series) > 1:
                stability = 1 - (np.std(importance_series) / (np.mean(np.abs(importance_series)) + 1e-8))
                feature_stability[feature] = {
                    'stability_score': float(stability),
                    'mean_importance': float(np.mean(importance_series)),
                    'std_importance': float(np.std(importance_series)),
                    'count': len(importance_series)
                }
        
        # Sort by stability
        sorted_stability = sorted(feature_stability.items(), key=lambda x: x[1]['stability_score'], reverse=True)
        
        # Calculate overall stability
        overall_stability = np.mean([f[1]['stability_score'] for f in sorted_stability])
        
        return {
            'model_id': model_id,
            'time_period': f"{days} days",
            'total_explanations': len(recent_history),
            'overall_stability': float(overall_stability),
            'feature_stability': dict(sorted_stability),
            'most_stable_features': [f[0] for f in sorted_stability[:5]],
            'least_stable_features': [f[0] for f in sorted_stability[-5:]],
            'timestamp': datetime.now().isoformat()
        }
    
    def run(self, host: str = "0.0.0.0", port: int = 8022):
        """Run the service."""
        logger.info(f"Starting LIME Explainer on {host}:{port}")
        uvicorn.run(self.app, host=host, port=port)


# Global service instance
service = LIMEExplainer()


def create_app() -> FastAPI:
    """Create FastAPI application."""
    return service.app


if __name__ == "__main__":
    service.run()
