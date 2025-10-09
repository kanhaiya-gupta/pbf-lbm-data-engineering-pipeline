"""
Model Interpretation Service

This module implements model interpretation for PBF-LB/M processes.
It provides comprehensive model interpretation, model understanding,
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

# Model interpretation libraries (would be installed as dependencies)
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    from lime import lime_tabular, lime_text, lime_image
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False

try:
    from sklearn.inspection import partial_dependence, PartialDependenceDisplay
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.linear_model import LinearRegression, LogisticRegression
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from ...config.config_manager import ConfigManager

logger = logging.getLogger(__name__)


# Pydantic models for API requests and responses
class ModelInterpretationRequest(BaseModel):
    """Request model for model interpretation."""
    model_id: str = Field(..., description="Model ID to interpret")
    model_type: str = Field(..., description="Model type (regression, classification)")
    feature_names: List[str] = Field(..., description="Feature names")
    feature_data: List[List[float]] = Field(..., description="Feature data")
    target_data: List[float] = Field(..., description="Target data")
    interpretation_methods: List[str] = Field(["shap", "lime", "pdp"], description="Interpretation methods to use")
    sample_size: int = Field(1000, description="Sample size for interpretation")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class ModelInterpretationResponse(BaseModel):
    """Response model for model interpretation."""
    model_id: str = Field(..., description="Model ID")
    model_type: str = Field(..., description="Model type")
    interpretation_methods: List[str] = Field(..., description="Methods used for interpretation")
    model_insights: Dict[str, Any] = Field(..., description="Model insights and interpretations")
    feature_contributions: Dict[str, Dict[str, float]] = Field(..., description="Feature contributions by method")
    model_complexity: Dict[str, Any] = Field(..., description="Model complexity analysis")
    interpretation_summary: str = Field(..., description="Interpretation summary")
    visualizations: Optional[Dict[str, Any]] = Field(None, description="Model interpretation visualizations")
    timestamp: str = Field(..., description="Interpretation timestamp")


class ModelUnderstandingRequest(BaseModel):
    """Request model for model understanding analysis."""
    model_id: str = Field(..., description="Model ID to understand")
    model_type: str = Field(..., description="Model type")
    feature_names: List[str] = Field(..., description="Feature names")
    feature_data: List[List[float]] = Field(..., description="Feature data")
    target_data: List[float] = Field(..., description="Target data")
    understanding_metrics: List[str] = Field(["complexity", "stability", "robustness"], description="Understanding metrics")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class ModelUnderstandingResponse(BaseModel):
    """Response model for model understanding analysis."""
    model_id: str = Field(..., description="Model ID")
    model_type: str = Field(..., description="Model type")
    understanding_metrics: List[str] = Field(..., description="Understanding metrics used")
    model_characteristics: Dict[str, Any] = Field(..., description="Model characteristics")
    complexity_analysis: Dict[str, Any] = Field(..., description="Model complexity analysis")
    stability_analysis: Dict[str, Any] = Field(..., description="Model stability analysis")
    robustness_analysis: Dict[str, Any] = Field(..., description="Model robustness analysis")
    understanding_summary: str = Field(..., description="Understanding summary")
    recommendations: List[str] = Field(..., description="Model improvement recommendations")
    timestamp: str = Field(..., description="Understanding timestamp")


class ModelInterpretationService:
    """
    Model interpretation service for PBF-LB/M processes.
    
    This service provides comprehensive model interpretation capabilities for:
    - Model understanding and insights
    - Feature contribution analysis
    - Model complexity assessment
    - Model stability and robustness analysis
    - Explainable AI insights
    """
    
    def __init__(self, config_manager: Optional[ConfigManager] = None):
        """
        Initialize the model interpretation service.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config_manager = config_manager or ConfigManager()
        self.app = FastAPI(
            title="Model Interpretation Service",
            description="Model interpretation and understanding for PBF-LB/M manufacturing",
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
        
        # Interpretation history
        self.interpretation_history = {}  # Store interpretation history
        self.model_insights_cache = {}  # Cache model insights
        
        # Service metrics
        self.service_metrics = {
            'total_interpretations': 0,
            'models_interpreted': 0,
            'interpretation_methods_used': 0,
            'last_interpretation_time': None
        }
        
        # Check library availability
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not available. Install with: pip install shap")
        
        if not LIME_AVAILABLE:
            logger.warning("LIME not available. Install with: pip install lime")
        
        if not SKLEARN_AVAILABLE:
            logger.warning("scikit-learn not available. Install with: pip install scikit-learn")
        
        # Setup routes
        self._setup_routes()
        
        logger.info("Initialized ModelInterpretationService")
    
    def _setup_routes(self):
        """Setup API routes."""
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {
                "status": "healthy",
                "service": "model_interpretation_service",
                "shap_available": SHAP_AVAILABLE,
                "lime_available": LIME_AVAILABLE,
                "sklearn_available": SKLEARN_AVAILABLE,
                "timestamp": datetime.now().isoformat(),
                "metrics": self.service_metrics
            }
        
        @self.app.post("/interpret", response_model=ModelInterpretationResponse)
        async def interpret_model(request: ModelInterpretationRequest):
            """Interpret model predictions and behavior."""
            return await self._interpret_model(request)
        
        @self.app.post("/understand", response_model=ModelUnderstandingResponse)
        async def understand_model(request: ModelUnderstandingRequest):
            """Analyze model understanding and characteristics."""
            return await self._understand_model(request)
        
        @self.app.get("/models/{model_id}/interpretations")
        async def get_model_interpretations(model_id: str, days: int = Query(7, ge=1, le=30)):
            """Get model interpretation history."""
            return await self._get_model_interpretations(model_id, days)
        
        @self.app.get("/models/{model_id}/insights")
        async def get_model_insights(model_id: str):
            """Get cached model insights."""
            return await self._get_model_insights(model_id)
        
        @self.app.get("/models/{model_id}/complexity")
        async def get_model_complexity(model_id: str):
            """Get model complexity analysis."""
            return await self._get_model_complexity(model_id)
        
        @self.app.get("/models/{model_id}/stability")
        async def get_model_stability(model_id: str, days: int = Query(7, ge=1, le=30)):
            """Get model stability analysis."""
            return await self._get_model_stability(model_id, days)
        
        @self.app.get("/models/{model_id}/robustness")
        async def get_model_robustness(model_id: str):
            """Get model robustness analysis."""
            return await self._get_model_robustness(model_id)
        
        @self.app.get("/metrics")
        async def get_metrics():
            """Get service metrics."""
            return self.service_metrics
    
    async def _interpret_model(self, request: ModelInterpretationRequest) -> ModelInterpretationResponse:
        """
        Interpret model predictions and behavior.
        
        Args:
            request: Model interpretation request
            
        Returns:
            Model interpretation response
        """
        try:
            # Convert data to numpy arrays
            X = np.array(request.feature_data)
            y = np.array(request.target_data)
            feature_names = request.feature_names
            
            # Validate data
            if X.shape[0] != len(y):
                raise ValueError("Feature data and target data must have the same number of samples")
            
            if X.shape[1] != len(feature_names):
                raise ValueError("Feature data must have the same number of features as feature names")
            
            # Sample data if needed
            if len(X) > request.sample_size:
                indices = np.random.choice(len(X), request.sample_size, replace=False)
                X = X[indices]
                y = y[indices]
            
            # Interpret model using multiple methods
            interpretation_results = {}
            feature_contributions = {}
            
            for method in request.interpretation_methods:
                if method == "shap" and SHAP_AVAILABLE:
                    interpretation_results[method], feature_contributions[method] = await self._shap_interpretation(X, y, feature_names, request.model_type)
                elif method == "lime" and LIME_AVAILABLE:
                    interpretation_results[method], feature_contributions[method] = await self._lime_interpretation(X, y, feature_names, request.model_type)
                elif method == "pdp" and SKLEARN_AVAILABLE:
                    interpretation_results[method], feature_contributions[method] = await self._pdp_interpretation(X, y, feature_names, request.model_type)
                else:
                    logger.warning(f"Unknown or unavailable interpretation method: {method}")
            
            # Analyze model complexity
            model_complexity = await self._analyze_model_complexity(X, y, feature_names, request.model_type)
            
            # Generate model insights
            model_insights = await self._generate_model_insights(interpretation_results, feature_contributions, model_complexity)
            
            # Generate interpretation summary
            interpretation_summary = await self._generate_interpretation_summary(
                interpretation_results, feature_contributions, model_complexity
            )
            
            # Generate visualizations
            visualizations = await self._generate_interpretation_visualizations(
                interpretation_results, feature_contributions, model_complexity
            )
            
            # Store interpretation history
            await self._store_interpretation_history(request.model_id, interpretation_results, model_insights)
            
            # Update metrics
            self.service_metrics['total_interpretations'] += 1
            self.service_metrics['models_interpreted'] += 1
            self.service_metrics['interpretation_methods_used'] += len(interpretation_results)
            self.service_metrics['last_interpretation_time'] = datetime.now().isoformat()
            
            return ModelInterpretationResponse(
                model_id=request.model_id,
                model_type=request.model_type,
                interpretation_methods=request.interpretation_methods,
                model_insights=model_insights,
                feature_contributions=feature_contributions,
                model_complexity=model_complexity,
                interpretation_summary=interpretation_summary,
                visualizations=visualizations,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"Error interpreting model: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def _shap_interpretation(self, X: np.ndarray, y: np.ndarray, 
                                 feature_names: List[str], model_type: str) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """Perform SHAP interpretation."""
        try:
            # Create a simple model for SHAP interpretation
            if model_type == "regression":
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            else:
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            
            # Fit the model
            model.fit(X, y)
            
            # Create SHAP explainer
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            
            # Handle classification case
            if model_type == "classification" and len(shap_values) > 1:
                shap_values = shap_values[1]  # Use positive class
            
            # Calculate feature contributions
            feature_contributions = {}
            for i, feature_name in enumerate(feature_names):
                feature_contributions[feature_name] = float(np.mean(np.abs(shap_values[:, i])))
            
            # Create interpretation results
            interpretation_results = {
                'shap_values': shap_values.tolist(),
                'expected_value': float(explainer.expected_value),
                'feature_importance': feature_contributions,
                'method': 'SHAP'
            }
            
            return interpretation_results, feature_contributions
            
        except Exception as e:
            logger.error(f"Error in SHAP interpretation: {e}")
            return {}, {}
    
    async def _lime_interpretation(self, X: np.ndarray, y: np.ndarray, 
                                 feature_names: List[str], model_type: str) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """Perform LIME interpretation."""
        try:
            # Create a simple model for LIME interpretation
            if model_type == "regression":
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            else:
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            
            # Fit the model
            model.fit(X, y)
            
            # Create LIME explainer
            explainer = lime_tabular.LimeTabularExplainer(X, feature_names=feature_names, mode='regression' if model_type == "regression" else 'classification')
            
            # Explain a sample instance
            sample_idx = 0
            explanation = explainer.explain_instance(X[sample_idx], model.predict, num_features=len(feature_names))
            
            # Extract feature contributions
            feature_contributions = {}
            for feature, weight in explanation.as_list():
                feature_contributions[feature] = float(abs(weight))
            
            # Create interpretation results
            interpretation_results = {
                'explanation': explanation.as_list(),
                'feature_importance': feature_contributions,
                'method': 'LIME'
            }
            
            return interpretation_results, feature_contributions
            
        except Exception as e:
            logger.error(f"Error in LIME interpretation: {e}")
            return {}, {}
    
    async def _pdp_interpretation(self, X: np.ndarray, y: np.ndarray, 
                                feature_names: List[str], model_type: str) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """Perform Partial Dependence Plot interpretation."""
        try:
            # Create a simple model for PDP interpretation
            if model_type == "regression":
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            else:
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            
            # Fit the model
            model.fit(X, y)
            
            # Calculate partial dependence for each feature
            feature_contributions = {}
            pdp_results = {}
            
            for i, feature_name in enumerate(feature_names):
                try:
                    # Calculate partial dependence
                    pdp = partial_dependence(model, X, [i])
                    
                    # Calculate feature contribution (variance of PDP)
                    feature_contributions[feature_name] = float(np.var(pdp['average'][0]))
                    
                    # Store PDP results
                    pdp_results[feature_name] = {
                        'values': pdp['values'][0].tolist(),
                        'average': pdp['average'][0].tolist()
                    }
                    
                except Exception as e:
                    logger.warning(f"Error calculating PDP for feature {feature_name}: {e}")
                    feature_contributions[feature_name] = 0.0
                    pdp_results[feature_name] = {'values': [], 'average': []}
            
            # Create interpretation results
            interpretation_results = {
                'partial_dependence': pdp_results,
                'feature_importance': feature_contributions,
                'method': 'PDP'
            }
            
            return interpretation_results, feature_contributions
            
        except Exception as e:
            logger.error(f"Error in PDP interpretation: {e}")
            return {}, {}
    
    async def _analyze_model_complexity(self, X: np.ndarray, y: np.ndarray, 
                                      feature_names: List[str], model_type: str) -> Dict[str, Any]:
        """Analyze model complexity."""
        try:
            # Create a simple model for complexity analysis
            if model_type == "regression":
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            else:
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            
            # Fit the model
            model.fit(X, y)
            
            # Calculate complexity metrics
            complexity_metrics = {
                'n_features': len(feature_names),
                'n_samples': len(X),
                'feature_to_sample_ratio': len(feature_names) / len(X),
                'model_type': model_type
            }
            
            # Add model-specific complexity metrics
            if hasattr(model, 'n_estimators'):
                complexity_metrics['n_estimators'] = model.n_estimators
            
            if hasattr(model, 'max_depth'):
                complexity_metrics['max_depth'] = model.max_depth
            
            if hasattr(model, 'feature_importances_'):
                complexity_metrics['feature_importance_entropy'] = float(-np.sum(model.feature_importances_ * np.log(model.feature_importances_ + 1e-8)))
            
            # Calculate model performance
            y_pred = model.predict(X)
            if model_type == "regression":
                from sklearn.metrics import mean_squared_error, r2_score
                complexity_metrics['mse'] = float(mean_squared_error(y, y_pred))
                complexity_metrics['r2_score'] = float(r2_score(y, y_pred))
            else:
                from sklearn.metrics import accuracy_score, f1_score
                complexity_metrics['accuracy'] = float(accuracy_score(y, y_pred))
                complexity_metrics['f1_score'] = float(f1_score(y, y_pred, average='weighted'))
            
            return complexity_metrics
            
        except Exception as e:
            logger.error(f"Error analyzing model complexity: {e}")
            return {}
    
    async def _generate_model_insights(self, interpretation_results: Dict[str, Any], 
                                     feature_contributions: Dict[str, Dict[str, float]], 
                                     model_complexity: Dict[str, Any]) -> Dict[str, Any]:
        """Generate model insights from interpretation results."""
        try:
            insights = {
                'interpretation_methods': list(interpretation_results.keys()),
                'model_complexity': model_complexity,
                'feature_contributions': feature_contributions
            }
            
            # Analyze feature contributions across methods
            if feature_contributions:
                # Get common features across methods
                common_features = set(feature_contributions[list(feature_contributions.keys())[0]].keys())
                for method_contributions in feature_contributions.values():
                    common_features = common_features.intersection(set(method_contributions.keys()))
                
                # Calculate average feature importance across methods
                if common_features:
                    average_importance = {}
                    for feature in common_features:
                        importance_scores = [method_contributions[feature] for method_contributions in feature_contributions.values()]
                        average_importance[feature] = float(np.mean(importance_scores))
                    
                    # Sort by importance
                    sorted_importance = sorted(average_importance.items(), key=lambda x: x[1], reverse=True)
                    insights['top_features'] = [f[0] for f in sorted_importance[:5]]
                    insights['average_feature_importance'] = dict(sorted_importance)
            
            # Add complexity insights
            if model_complexity:
                if model_complexity.get('feature_to_sample_ratio', 0) > 0.1:
                    insights['complexity_warning'] = "High feature-to-sample ratio may indicate overfitting risk"
                
                if model_complexity.get('feature_importance_entropy', 0) < 1.0:
                    insights['feature_distribution_warning'] = "Low feature importance entropy indicates imbalanced feature usage"
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating model insights: {e}")
            return {}
    
    async def _generate_interpretation_summary(self, interpretation_results: Dict[str, Any], 
                                             feature_contributions: Dict[str, Dict[str, float]], 
                                             model_complexity: Dict[str, Any]) -> str:
        """Generate interpretation summary."""
        try:
            summary_parts = []
            
            # Add interpretation methods used
            if interpretation_results:
                methods = list(interpretation_results.keys())
                summary_parts.append(f"Model interpreted using: {', '.join(methods)}")
            
            # Add top features
            if feature_contributions:
                # Get average importance across methods
                all_features = set()
                for method_contributions in feature_contributions.values():
                    all_features.update(method_contributions.keys())
                
                if all_features:
                    average_importance = {}
                    for feature in all_features:
                        importance_scores = [method_contributions.get(feature, 0) for method_contributions in feature_contributions.values()]
                        average_importance[feature] = np.mean(importance_scores)
                    
                    sorted_features = sorted(average_importance.items(), key=lambda x: x[1], reverse=True)
                    top_features = [f[0] for f in sorted_features[:3]]
                    summary_parts.append(f"Top contributing features: {', '.join(top_features)}")
            
            # Add complexity insights
            if model_complexity:
                n_features = model_complexity.get('n_features', 0)
                n_samples = model_complexity.get('n_samples', 0)
                summary_parts.append(f"Model complexity: {n_features} features, {n_samples} samples")
                
                if model_complexity.get('feature_to_sample_ratio', 0) > 0.1:
                    summary_parts.append("Warning: High feature-to-sample ratio detected")
            
            return ". ".join(summary_parts) + "."
            
        except Exception as e:
            logger.error(f"Error generating interpretation summary: {e}")
            return "Unable to generate interpretation summary"
    
    async def _generate_interpretation_visualizations(self, interpretation_results: Dict[str, Any], 
                                                    feature_contributions: Dict[str, Dict[str, float]], 
                                                    model_complexity: Dict[str, Any]) -> Dict[str, Any]:
        """Generate interpretation visualizations."""
        try:
            visualizations = {}
            
            # Feature importance comparison plot
            if feature_contributions:
                methods = list(feature_contributions.keys())
                if methods:
                    # Get common features
                    common_features = set(feature_contributions[methods[0]].keys())
                    for method_contributions in feature_contributions.values():
                        common_features = common_features.intersection(set(method_contributions.keys()))
                    
                    if common_features:
                        feature_names = list(common_features)
                        
                        fig = go.Figure()
                        
                        for method in methods:
                            method_scores = [feature_contributions[method].get(feature, 0) for feature in feature_names]
                            fig.add_trace(go.Bar(
                                x=feature_names,
                                y=method_scores,
                                name=method.upper(),
                                opacity=0.7
                            ))
                        
                        fig.update_layout(
                            title="Feature Importance by Interpretation Method",
                            xaxis_title="Features",
                            yaxis_title="Importance Score",
                            xaxis_tickangle=-45,
                            barmode='group'
                        )
                        
                        visualizations['feature_importance_comparison'] = json.loads(fig.to_json())
            
            # Model complexity plot
            if model_complexity:
                complexity_metrics = ['n_features', 'n_samples', 'feature_to_sample_ratio']
                complexity_values = [model_complexity.get(metric, 0) for metric in complexity_metrics]
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=complexity_metrics,
                    y=complexity_values,
                    marker_color='lightcoral',
                    name='Complexity Metrics'
                ))
                
                fig.update_layout(
                    title="Model Complexity Metrics",
                    xaxis_title="Metrics",
                    yaxis_title="Values"
                )
                
                visualizations['model_complexity'] = json.loads(fig.to_json())
            
            return visualizations
            
        except Exception as e:
            logger.error(f"Error generating interpretation visualizations: {e}")
            return {}
    
    async def _store_interpretation_history(self, model_id: str, interpretation_results: Dict[str, Any], 
                                          model_insights: Dict[str, Any]):
        """Store interpretation history."""
        if model_id not in self.interpretation_history:
            self.interpretation_history[model_id] = []
        
        interpretation_record = {
            'timestamp': datetime.now().isoformat(),
            'interpretation_results': interpretation_results,
            'model_insights': model_insights
        }
        
        self.interpretation_history[model_id].append(interpretation_record)
        
        # Keep only recent history (last 100 interpretations)
        if len(self.interpretation_history[model_id]) > 100:
            self.interpretation_history[model_id] = self.interpretation_history[model_id][-100:]
    
    async def _understand_model(self, request: ModelUnderstandingRequest) -> ModelUnderstandingResponse:
        """
        Analyze model understanding and characteristics.
        
        Args:
            request: Model understanding request
            
        Returns:
            Model understanding response
        """
        try:
            # Convert data to numpy arrays
            X = np.array(request.feature_data)
            y = np.array(request.target_data)
            feature_names = request.feature_names
            
            # Validate data
            if X.shape[0] != len(y):
                raise ValueError("Feature data and target data must have the same number of samples")
            
            if X.shape[1] != len(feature_names):
                raise ValueError("Feature data must have the same number of features as feature names")
            
            # Analyze model characteristics
            model_characteristics = await self._analyze_model_characteristics(X, y, feature_names, request.model_type)
            
            # Analyze model complexity
            complexity_analysis = await self._analyze_model_complexity(X, y, feature_names, request.model_type)
            
            # Analyze model stability
            stability_analysis = await self._analyze_model_stability(X, y, feature_names, request.model_type)
            
            # Analyze model robustness
            robustness_analysis = await self._analyze_model_robustness(X, y, feature_names, request.model_type)
            
            # Generate understanding summary
            understanding_summary = await self._generate_understanding_summary(
                model_characteristics, complexity_analysis, stability_analysis, robustness_analysis
            )
            
            # Generate recommendations
            recommendations = await self._generate_model_recommendations(
                model_characteristics, complexity_analysis, stability_analysis, robustness_analysis
            )
            
            return ModelUnderstandingResponse(
                model_id=request.model_id,
                model_type=request.model_type,
                understanding_metrics=request.understanding_metrics,
                model_characteristics=model_characteristics,
                complexity_analysis=complexity_analysis,
                stability_analysis=stability_analysis,
                robustness_analysis=robustness_analysis,
                understanding_summary=understanding_summary,
                recommendations=recommendations,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"Error understanding model: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def _analyze_model_characteristics(self, X: np.ndarray, y: np.ndarray, 
                                           feature_names: List[str], model_type: str) -> Dict[str, Any]:
        """Analyze model characteristics."""
        try:
            characteristics = {
                'n_features': len(feature_names),
                'n_samples': len(X),
                'feature_to_sample_ratio': len(feature_names) / len(X),
                'model_type': model_type,
                'feature_types': self._analyze_feature_types(X, feature_names)
            }
            
            # Add data quality metrics
            characteristics['data_quality'] = {
                'missing_values': float(np.isnan(X).sum()),
                'feature_variance': [float(np.var(X[:, i])) for i in range(X.shape[1])],
                'feature_correlation': self._analyze_feature_correlation(X, feature_names)
            }
            
            return characteristics
            
        except Exception as e:
            logger.error(f"Error analyzing model characteristics: {e}")
            return {}
    
    def _analyze_feature_types(self, X: np.ndarray, feature_names: List[str]) -> Dict[str, List[str]]:
        """Analyze feature types."""
        try:
            feature_types = {
                'continuous': [],
                'categorical': [],
                'binary': []
            }
            
            for i, feature_name in enumerate(feature_names):
                feature_values = X[:, i]
                unique_values = len(np.unique(feature_values))
                
                if unique_values == 2:
                    feature_types['binary'].append(feature_name)
                elif unique_values <= 10:
                    feature_types['categorical'].append(feature_name)
                else:
                    feature_types['continuous'].append(feature_name)
            
            return feature_types
            
        except Exception as e:
            logger.error(f"Error analyzing feature types: {e}")
            return {'continuous': [], 'categorical': [], 'binary': []}
    
    def _analyze_feature_correlation(self, X: np.ndarray, feature_names: List[str]) -> Dict[str, float]:
        """Analyze feature correlation."""
        try:
            # Calculate correlation matrix
            corr_matrix = np.corrcoef(X.T)
            
            # Find high correlations (above 0.8)
            high_correlations = {}
            for i in range(len(feature_names)):
                for j in range(i+1, len(feature_names)):
                    correlation = abs(corr_matrix[i, j])
                    if correlation > 0.8:
                        feature_pair = f"{feature_names[i]}-{feature_names[j]}"
                        high_correlations[feature_pair] = float(correlation)
            
            return high_correlations
            
        except Exception as e:
            logger.error(f"Error analyzing feature correlation: {e}")
            return {}
    
    async def _analyze_model_stability(self, X: np.ndarray, y: np.ndarray, 
                                     feature_names: List[str], model_type: str) -> Dict[str, Any]:
        """Analyze model stability."""
        try:
            # Create multiple models with different random seeds
            stability_scores = []
            
            for seed in range(5):
                if model_type == "regression":
                    model = RandomForestRegressor(n_estimators=100, random_state=seed)
                else:
                    model = RandomForestClassifier(n_estimators=100, random_state=seed)
                
                model.fit(X, y)
                y_pred = model.predict(X)
                
                if model_type == "regression":
                    from sklearn.metrics import mean_squared_error
                    score = mean_squared_error(y, y_pred)
                else:
                    from sklearn.metrics import accuracy_score
                    score = accuracy_score(y, y_pred)
                
                stability_scores.append(score)
            
            # Calculate stability metrics
            stability_analysis = {
                'score_variance': float(np.var(stability_scores)),
                'score_std': float(np.std(stability_scores)),
                'score_range': float(np.max(stability_scores) - np.min(stability_scores)),
                'stability_score': float(1 - (np.std(stability_scores) / (np.mean(stability_scores) + 1e-8)))
            }
            
            return stability_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing model stability: {e}")
            return {}
    
    async def _analyze_model_robustness(self, X: np.ndarray, y: np.ndarray, 
                                      feature_names: List[str], model_type: str) -> Dict[str, Any]:
        """Analyze model robustness."""
        try:
            # Create model
            if model_type == "regression":
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            else:
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            
            model.fit(X, y)
            
            # Test robustness with noise
            noise_levels = [0.01, 0.05, 0.1, 0.2]
            robustness_scores = []
            
            for noise_level in noise_levels:
                # Add noise to features
                X_noisy = X + np.random.normal(0, noise_level, X.shape)
                
                # Predict with noisy data
                y_pred_noisy = model.predict(X_noisy)
                
                # Calculate performance degradation
                if model_type == "regression":
                    from sklearn.metrics import mean_squared_error
                    original_score = mean_squared_error(y, model.predict(X))
                    noisy_score = mean_squared_error(y, y_pred_noisy)
                else:
                    from sklearn.metrics import accuracy_score
                    original_score = accuracy_score(y, model.predict(X))
                    noisy_score = accuracy_score(y, y_pred_noisy)
                
                # Calculate robustness score (1 - degradation)
                degradation = abs(noisy_score - original_score) / (original_score + 1e-8)
                robustness_score = max(0, 1 - degradation)
                robustness_scores.append(robustness_score)
            
            # Calculate overall robustness
            robustness_analysis = {
                'noise_levels': noise_levels,
                'robustness_scores': robustness_scores,
                'overall_robustness': float(np.mean(robustness_scores)),
                'robustness_variance': float(np.var(robustness_scores))
            }
            
            return robustness_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing model robustness: {e}")
            return {}
    
    async def _generate_understanding_summary(self, model_characteristics: Dict[str, Any], 
                                            complexity_analysis: Dict[str, Any], 
                                            stability_analysis: Dict[str, Any], 
                                            robustness_analysis: Dict[str, Any]) -> str:
        """Generate model understanding summary."""
        try:
            summary_parts = []
            
            # Add model characteristics
            if model_characteristics:
                n_features = model_characteristics.get('n_features', 0)
                n_samples = model_characteristics.get('n_samples', 0)
                summary_parts.append(f"Model has {n_features} features and {n_samples} samples")
                
                feature_types = model_characteristics.get('feature_types', {})
                if feature_types:
                    continuous_count = len(feature_types.get('continuous', []))
                    categorical_count = len(feature_types.get('categorical', []))
                    binary_count = len(feature_types.get('binary', []))
                    summary_parts.append(f"Feature types: {continuous_count} continuous, {categorical_count} categorical, {binary_count} binary")
            
            # Add complexity insights
            if complexity_analysis:
                feature_to_sample_ratio = complexity_analysis.get('feature_to_sample_ratio', 0)
                if feature_to_sample_ratio > 0.1:
                    summary_parts.append("Warning: High feature-to-sample ratio may indicate overfitting risk")
            
            # Add stability insights
            if stability_analysis:
                stability_score = stability_analysis.get('stability_score', 0)
                if stability_score > 0.9:
                    summary_parts.append("Model shows high stability across different random seeds")
                elif stability_score < 0.7:
                    summary_parts.append("Warning: Model shows low stability across different random seeds")
            
            # Add robustness insights
            if robustness_analysis:
                overall_robustness = robustness_analysis.get('overall_robustness', 0)
                if overall_robustness > 0.8:
                    summary_parts.append("Model shows high robustness to noise")
                elif overall_robustness < 0.6:
                    summary_parts.append("Warning: Model shows low robustness to noise")
            
            return ". ".join(summary_parts) + "."
            
        except Exception as e:
            logger.error(f"Error generating understanding summary: {e}")
            return "Unable to generate understanding summary"
    
    async def _generate_model_recommendations(self, model_characteristics: Dict[str, Any], 
                                            complexity_analysis: Dict[str, Any], 
                                            stability_analysis: Dict[str, Any], 
                                            robustness_analysis: Dict[str, Any]) -> List[str]:
        """Generate model improvement recommendations."""
        try:
            recommendations = []
            
            # Complexity recommendations
            if complexity_analysis:
                feature_to_sample_ratio = complexity_analysis.get('feature_to_sample_ratio', 0)
                if feature_to_sample_ratio > 0.1:
                    recommendations.append("Consider feature selection to reduce overfitting risk")
                    recommendations.append("Increase training data size if possible")
                
                if complexity_analysis.get('feature_importance_entropy', 0) < 1.0:
                    recommendations.append("Consider feature engineering to improve feature balance")
            
            # Stability recommendations
            if stability_analysis:
                stability_score = stability_analysis.get('stability_score', 0)
                if stability_score < 0.7:
                    recommendations.append("Improve model stability by increasing regularization")
                    recommendations.append("Consider ensemble methods for better stability")
            
            # Robustness recommendations
            if robustness_analysis:
                overall_robustness = robustness_analysis.get('overall_robustness', 0)
                if overall_robustness < 0.6:
                    recommendations.append("Improve model robustness by adding noise to training data")
                    recommendations.append("Consider robust loss functions")
            
            # Data quality recommendations
            if model_characteristics:
                data_quality = model_characteristics.get('data_quality', {})
                if data_quality.get('missing_values', 0) > 0:
                    recommendations.append("Address missing values in the dataset")
                
                high_correlations = data_quality.get('feature_correlation', {})
                if high_correlations:
                    recommendations.append("Consider removing highly correlated features")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating model recommendations: {e}")
            return []
    
    async def _get_model_interpretations(self, model_id: str, days: int) -> Dict[str, Any]:
        """Get model interpretation history."""
        if model_id not in self.interpretation_history:
            raise HTTPException(status_code=404, detail=f"No interpretation history found for model {model_id}")
        
        history = self.interpretation_history[model_id]
        
        # Filter by time period
        cutoff_time = datetime.now() - timedelta(days=days)
        recent_history = [
            h for h in history 
            if datetime.fromisoformat(h['timestamp']) > cutoff_time
        ]
        
        if not recent_history:
            raise HTTPException(status_code=404, detail=f"No interpretation history found for the last {days} days")
        
        return {
            'model_id': model_id,
            'time_period': f"{days} days",
            'total_interpretations': len(recent_history),
            'interpretation_history': recent_history,
            'timestamp': datetime.now().isoformat()
        }
    
    async def _get_model_insights(self, model_id: str) -> Dict[str, Any]:
        """Get cached model insights."""
        if model_id not in self.interpretation_history:
            raise HTTPException(status_code=404, detail=f"No insights found for model {model_id}")
        
        # Get the most recent insights
        history = self.interpretation_history[model_id]
        if not history:
            raise HTTPException(status_code=404, detail=f"No insights found for model {model_id}")
        
        latest_insights = history[-1]['model_insights']
        
        return {
            'model_id': model_id,
            'insights': latest_insights,
            'timestamp': datetime.now().isoformat()
        }
    
    async def _get_model_complexity(self, model_id: str) -> Dict[str, Any]:
        """Get model complexity analysis."""
        if model_id not in self.interpretation_history:
            raise HTTPException(status_code=404, detail=f"No complexity analysis found for model {model_id}")
        
        # Get the most recent complexity analysis
        history = self.interpretation_history[model_id]
        if not history:
            raise HTTPException(status_code=404, detail=f"No complexity analysis found for model {model_id}")
        
        latest_interpretation = history[-1]
        complexity_analysis = latest_interpretation.get('model_insights', {}).get('model_complexity', {})
        
        return {
            'model_id': model_id,
            'complexity_analysis': complexity_analysis,
            'timestamp': datetime.now().isoformat()
        }
    
    async def _get_model_stability(self, model_id: str, days: int) -> Dict[str, Any]:
        """Get model stability analysis."""
        if model_id not in self.interpretation_history:
            raise HTTPException(status_code=404, detail=f"No stability analysis found for model {model_id}")
        
        history = self.interpretation_history[model_id]
        
        # Filter by time period
        cutoff_time = datetime.now() - timedelta(days=days)
        recent_history = [
            h for h in history 
            if datetime.fromisoformat(h['timestamp']) > cutoff_time
        ]
        
        if not recent_history:
            raise HTTPException(status_code=404, detail=f"No stability analysis found for the last {days} days")
        
        # Aggregate stability metrics
        stability_scores = []
        for interpretation in recent_history:
            model_insights = interpretation.get('model_insights', {})
            complexity_analysis = model_insights.get('model_complexity', {})
            if complexity_analysis:
                stability_scores.append(complexity_analysis.get('stability_score', 0))
        
        if not stability_scores:
            raise HTTPException(status_code=404, detail="No stability scores found")
        
        return {
            'model_id': model_id,
            'time_period': f"{days} days",
            'total_analyses': len(recent_history),
            'average_stability': float(np.mean(stability_scores)),
            'stability_variance': float(np.var(stability_scores)),
            'stability_trend': stability_scores,
            'timestamp': datetime.now().isoformat()
        }
    
    async def _get_model_robustness(self, model_id: str) -> Dict[str, Any]:
        """Get model robustness analysis."""
        if model_id not in self.interpretation_history:
            raise HTTPException(status_code=404, detail=f"No robustness analysis found for model {model_id}")
        
        # Get the most recent robustness analysis
        history = self.interpretation_history[model_id]
        if not history:
            raise HTTPException(status_code=404, detail=f"No robustness analysis found for model {model_id}")
        
        latest_interpretation = history[-1]
        model_insights = latest_interpretation.get('model_insights', {})
        robustness_analysis = model_insights.get('robustness_analysis', {})
        
        return {
            'model_id': model_id,
            'robustness_analysis': robustness_analysis,
            'timestamp': datetime.now().isoformat()
        }
    
    def run(self, host: str = "0.0.0.0", port: int = 8024):
        """Run the service."""
        logger.info(f"Starting Model Interpretation Service on {host}:{port}")
        uvicorn.run(self.app, host=host, port=port)


# Global service instance
service = ModelInterpretationService()


def create_app() -> FastAPI:
    """Create FastAPI application."""
    return service.app


if __name__ == "__main__":
    service.run()
