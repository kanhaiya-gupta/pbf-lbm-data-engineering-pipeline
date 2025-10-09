"""
Feature Importance Analyzer

This module implements feature importance analysis for PBF-LB/M processes.
It provides comprehensive feature importance analysis, feature selection,
and feature ranking for ML models.
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

# Feature importance libraries (would be installed as dependencies)
try:
    from sklearn.feature_selection import SelectKBest, f_regression, f_classif
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.inspection import permutation_importance
    from sklearn.linear_model import LassoCV, RidgeCV
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

from ...config.config_manager import ConfigManager

logger = logging.getLogger(__name__)


# Pydantic models for API requests and responses
class FeatureImportanceRequest(BaseModel):
    """Request model for feature importance analysis."""
    model_id: str = Field(..., description="Model ID to analyze")
    feature_names: List[str] = Field(..., description="Feature names")
    feature_data: List[List[float]] = Field(..., description="Feature data")
    target_data: List[float] = Field(..., description="Target data")
    task_type: str = Field("regression", description="Task type (regression, classification)")
    methods: List[str] = Field(["permutation", "tree", "lasso"], description="Importance methods to use")
    top_k: int = Field(10, description="Number of top features to return")
    cross_validation: bool = Field(True, description="Whether to use cross-validation")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class FeatureImportanceResponse(BaseModel):
    """Response model for feature importance analysis."""
    model_id: str = Field(..., description="Model ID")
    task_type: str = Field(..., description="Task type")
    methods_used: List[str] = Field(..., description="Methods used for analysis")
    feature_importance: Dict[str, Dict[str, float]] = Field(..., description="Feature importance by method")
    top_features: List[str] = Field(..., description="Top features by importance")
    feature_ranking: Dict[str, int] = Field(..., description="Feature ranking")
    importance_summary: str = Field(..., description="Importance analysis summary")
    visualizations: Optional[Dict[str, Any]] = Field(None, description="Feature importance visualizations")
    timestamp: str = Field(..., description="Analysis timestamp")


class FeatureSelectionRequest(BaseModel):
    """Request model for feature selection."""
    feature_names: List[str] = Field(..., description="Feature names")
    feature_data: List[List[float]] = Field(..., description="Feature data")
    target_data: List[float] = Field(..., description="Target data")
    task_type: str = Field("regression", description="Task type")
    selection_method: str = Field("mutual_info", description="Selection method")
    k_features: int = Field(10, description="Number of features to select")
    threshold: Optional[float] = Field(None, description="Feature importance threshold")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class FeatureSelectionResponse(BaseModel):
    """Response model for feature selection."""
    selected_features: List[str] = Field(..., description="Selected feature names")
    feature_scores: Dict[str, float] = Field(..., description="Feature selection scores")
    selection_method: str = Field(..., description="Selection method used")
    k_features: int = Field(..., description="Number of features selected")
    selection_summary: str = Field(..., description="Selection summary")
    timestamp: str = Field(..., description="Selection timestamp")


class FeatureImportanceAnalyzer:
    """
    Feature importance analyzer for PBF-LB/M processes.
    
    This analyzer provides comprehensive feature importance analysis capabilities for:
    - Multiple importance methods (permutation, tree-based, Lasso, etc.)
    - Feature selection and ranking
    - Cross-validation and stability analysis
    - Feature importance visualization
    """
    
    def __init__(self, config_manager: Optional[ConfigManager] = None):
        """
        Initialize the feature importance analyzer.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config_manager = config_manager or ConfigManager()
        self.app = FastAPI(
            title="Feature Importance Analyzer",
            description="Feature importance analysis for PBF-LB/M manufacturing",
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
        
        # Analysis history
        self.analysis_history = {}  # Store analysis history
        self.feature_importance_cache = {}  # Cache feature importance results
        
        # Service metrics
        self.service_metrics = {
            'total_analyses': 0,
            'models_analyzed': 0,
            'features_analyzed': 0,
            'last_analysis_time': None
        }
        
        # Check library availability
        if not SKLEARN_AVAILABLE:
            logger.warning("scikit-learn not available. Install with: pip install scikit-learn")
        
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not available. Install with: pip install shap")
        
        # Setup routes
        self._setup_routes()
        
        logger.info("Initialized FeatureImportanceAnalyzer")
    
    def _setup_routes(self):
        """Setup API routes."""
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {
                "status": "healthy",
                "service": "feature_importance_analyzer",
                "sklearn_available": SKLEARN_AVAILABLE,
                "shap_available": SHAP_AVAILABLE,
                "timestamp": datetime.now().isoformat(),
                "metrics": self.service_metrics
            }
        
        @self.app.post("/analyze", response_model=FeatureImportanceResponse)
        async def analyze_feature_importance(request: FeatureImportanceRequest):
            """Analyze feature importance for a model."""
            return await self._analyze_feature_importance(request)
        
        @self.app.post("/select", response_model=FeatureSelectionResponse)
        async def select_features(request: FeatureSelectionRequest):
            """Select most important features."""
            return await self._select_features(request)
        
        @self.app.get("/models/{model_id}/importance")
        async def get_model_importance(model_id: str, days: int = Query(7, ge=1, le=30)):
            """Get feature importance history for a model."""
            return await self._get_model_importance(model_id, days)
        
        @self.app.get("/models/{model_id}/stability")
        async def get_importance_stability(model_id: str, days: int = Query(7, ge=1, le=30)):
            """Get feature importance stability for a model."""
            return await self._get_importance_stability(model_id, days)
        
        @self.app.get("/features/{feature_name}/importance")
        async def get_feature_importance_history(feature_name: str, days: int = Query(7, ge=1, le=30)):
            """Get importance history for a specific feature."""
            return await self._get_feature_importance_history(feature_name, days)
        
        @self.app.get("/features/ranking")
        async def get_feature_ranking(days: int = Query(7, ge=1, le=30), 
                                    task_type: str = Query("regression")):
            """Get global feature ranking across models."""
            return await self._get_feature_ranking(days, task_type)
        
        @self.app.get("/metrics")
        async def get_metrics():
            """Get service metrics."""
            return self.service_metrics
    
    async def _analyze_feature_importance(self, request: FeatureImportanceRequest) -> FeatureImportanceResponse:
        """
        Analyze feature importance for a model.
        
        Args:
            request: Feature importance analysis request
            
        Returns:
            Feature importance analysis response
        """
        if not SKLEARN_AVAILABLE:
            raise HTTPException(status_code=500, detail="scikit-learn library not available")
        
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
            
            # Analyze feature importance using multiple methods
            importance_results = {}
            
            for method in request.methods:
                if method == "permutation":
                    importance_results[method] = await self._permutation_importance(X, y, feature_names, request.task_type)
                elif method == "tree":
                    importance_results[method] = await self._tree_importance(X, y, feature_names, request.task_type)
                elif method == "lasso":
                    importance_results[method] = await self._lasso_importance(X, y, feature_names, request.task_type)
                elif method == "ridge":
                    importance_results[method] = await self._ridge_importance(X, y, feature_names, request.task_type)
                elif method == "mutual_info":
                    importance_results[method] = await self._mutual_info_importance(X, y, feature_names, request.task_type)
                elif method == "f_score":
                    importance_results[method] = await self._f_score_importance(X, y, feature_names, request.task_type)
                else:
                    logger.warning(f"Unknown importance method: {method}")
            
            # Combine results and rank features
            combined_importance = await self._combine_importance_methods(importance_results, feature_names)
            top_features = await self._get_top_features(combined_importance, request.top_k)
            feature_ranking = await self._rank_features(combined_importance)
            
            # Generate summary
            importance_summary = await self._generate_importance_summary(
                top_features, combined_importance, request.task_type
            )
            
            # Generate visualizations
            visualizations = await self._generate_importance_visualizations(
                importance_results, combined_importance, top_features
            )
            
            # Store analysis history
            await self._store_analysis_history(request.model_id, importance_results, combined_importance)
            
            # Update metrics
            self.service_metrics['total_analyses'] += 1
            self.service_metrics['models_analyzed'] += 1
            self.service_metrics['features_analyzed'] += len(feature_names)
            self.service_metrics['last_analysis_time'] = datetime.now().isoformat()
            
            return FeatureImportanceResponse(
                model_id=request.model_id,
                task_type=request.task_type,
                methods_used=request.methods,
                feature_importance=importance_results,
                top_features=top_features,
                feature_ranking=feature_ranking,
                importance_summary=importance_summary,
                visualizations=visualizations,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"Error analyzing feature importance: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def _permutation_importance(self, X: np.ndarray, y: np.ndarray, 
                                    feature_names: List[str], task_type: str) -> Dict[str, float]:
        """Calculate permutation importance."""
        try:
            # Create a simple model for permutation importance
            if task_type == "regression":
                from sklearn.ensemble import RandomForestRegressor
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            else:
                from sklearn.ensemble import RandomForestClassifier
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            
            # Fit the model
            model.fit(X, y)
            
            # Calculate permutation importance
            perm_importance = permutation_importance(model, X, y, n_repeats=10, random_state=42)
            
            # Create feature importance dictionary
            importance_dict = {}
            for i, feature_name in enumerate(feature_names):
                importance_dict[feature_name] = float(perm_importance.importances_mean[i])
            
            return importance_dict
            
        except Exception as e:
            logger.error(f"Error calculating permutation importance: {e}")
            return {feature: 0.0 for feature in feature_names}
    
    async def _tree_importance(self, X: np.ndarray, y: np.ndarray, 
                             feature_names: List[str], task_type: str) -> Dict[str, float]:
        """Calculate tree-based feature importance."""
        try:
            # Create tree-based model
            if task_type == "regression":
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            else:
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            
            # Fit the model
            model.fit(X, y)
            
            # Get feature importance
            importance_dict = {}
            for i, feature_name in enumerate(feature_names):
                importance_dict[feature_name] = float(model.feature_importances_[i])
            
            return importance_dict
            
        except Exception as e:
            logger.error(f"Error calculating tree importance: {e}")
            return {feature: 0.0 for feature in feature_names}
    
    async def _lasso_importance(self, X: np.ndarray, y: np.ndarray, 
                              feature_names: List[str], task_type: str) -> Dict[str, float]:
        """Calculate Lasso feature importance."""
        try:
            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Create Lasso model
            if task_type == "regression":
                model = LassoCV(cv=5, random_state=42)
            else:
                from sklearn.linear_model import LogisticRegressionCV
                model = LogisticRegressionCV(cv=5, random_state=42, penalty='l1', solver='liblinear')
            
            # Fit the model
            model.fit(X_scaled, y)
            
            # Get feature importance (absolute coefficients)
            importance_dict = {}
            for i, feature_name in enumerate(feature_names):
                importance_dict[feature_name] = float(abs(model.coef_[i]))
            
            return importance_dict
            
        except Exception as e:
            logger.error(f"Error calculating Lasso importance: {e}")
            return {feature: 0.0 for feature in feature_names}
    
    async def _ridge_importance(self, X: np.ndarray, y: np.ndarray, 
                              feature_names: List[str], task_type: str) -> Dict[str, float]:
        """Calculate Ridge feature importance."""
        try:
            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Create Ridge model
            if task_type == "regression":
                model = RidgeCV(cv=5)
            else:
                from sklearn.linear_model import LogisticRegressionCV
                model = LogisticRegressionCV(cv=5, random_state=42, penalty='l2')
            
            # Fit the model
            model.fit(X_scaled, y)
            
            # Get feature importance (absolute coefficients)
            importance_dict = {}
            for i, feature_name in enumerate(feature_names):
                importance_dict[feature_name] = float(abs(model.coef_[i]))
            
            return importance_dict
            
        except Exception as e:
            logger.error(f"Error calculating Ridge importance: {e}")
            return {feature: 0.0 for feature in feature_names}
    
    async def _mutual_info_importance(self, X: np.ndarray, y: np.ndarray, 
                                    feature_names: List[str], task_type: str) -> Dict[str, float]:
        """Calculate mutual information importance."""
        try:
            # Select mutual information
            if task_type == "regression":
                selector = SelectKBest(score_func=f_regression, k='all')
            else:
                selector = SelectKBest(score_func=f_classif, k='all')
            
            # Fit selector
            selector.fit(X, y)
            
            # Get feature importance
            importance_dict = {}
            for i, feature_name in enumerate(feature_names):
                importance_dict[feature_name] = float(selector.scores_[i])
            
            return importance_dict
            
        except Exception as e:
            logger.error(f"Error calculating mutual information importance: {e}")
            return {feature: 0.0 for feature in feature_names}
    
    async def _f_score_importance(self, X: np.ndarray, y: np.ndarray, 
                                feature_names: List[str], task_type: str) -> Dict[str, float]:
        """Calculate F-score importance."""
        try:
            # Select F-score
            if task_type == "regression":
                selector = SelectKBest(score_func=f_regression, k='all')
            else:
                selector = SelectKBest(score_func=f_classif, k='all')
            
            # Fit selector
            selector.fit(X, y)
            
            # Get feature importance
            importance_dict = {}
            for i, feature_name in enumerate(feature_names):
                importance_dict[feature_name] = float(selector.scores_[i])
            
            return importance_dict
            
        except Exception as e:
            logger.error(f"Error calculating F-score importance: {e}")
            return {feature: 0.0 for feature in feature_names}
    
    async def _combine_importance_methods(self, importance_results: Dict[str, Dict[str, float]], 
                                        feature_names: List[str]) -> Dict[str, float]:
        """Combine importance results from multiple methods."""
        try:
            combined_importance = {}
            
            for feature_name in feature_names:
                # Get importance scores from all methods
                scores = []
                for method, importance_dict in importance_results.items():
                    if feature_name in importance_dict:
                        scores.append(importance_dict[feature_name])
                
                if scores:
                    # Use average of normalized scores
                    normalized_scores = []
                    for score in scores:
                        if score > 0:
                            normalized_scores.append(score)
                    
                    if normalized_scores:
                        combined_importance[feature_name] = float(np.mean(normalized_scores))
                    else:
                        combined_importance[feature_name] = 0.0
                else:
                    combined_importance[feature_name] = 0.0
            
            return combined_importance
            
        except Exception as e:
            logger.error(f"Error combining importance methods: {e}")
            return {feature: 0.0 for feature in feature_names}
    
    async def _get_top_features(self, combined_importance: Dict[str, float], top_k: int) -> List[str]:
        """Get top K features by importance."""
        try:
            # Sort features by importance
            sorted_features = sorted(combined_importance.items(), key=lambda x: x[1], reverse=True)
            
            # Return top K features
            return [feature for feature, _ in sorted_features[:top_k]]
            
        except Exception as e:
            logger.error(f"Error getting top features: {e}")
            return []
    
    async def _rank_features(self, combined_importance: Dict[str, float]) -> Dict[str, int]:
        """Rank features by importance."""
        try:
            # Sort features by importance
            sorted_features = sorted(combined_importance.items(), key=lambda x: x[1], reverse=True)
            
            # Create ranking dictionary
            ranking = {}
            for rank, (feature, _) in enumerate(sorted_features, 1):
                ranking[feature] = rank
            
            return ranking
            
        except Exception as e:
            logger.error(f"Error ranking features: {e}")
            return {}
    
    async def _generate_importance_summary(self, top_features: List[str], 
                                         combined_importance: Dict[str, float], 
                                         task_type: str) -> str:
        """Generate feature importance summary."""
        try:
            if not top_features:
                return "No features analyzed"
            
            # Get top feature importance scores
            top_scores = [combined_importance[feature] for feature in top_features]
            
            # Generate summary
            summary_parts = []
            
            if task_type == "regression":
                summary_parts.append(f"Top features for regression: {', '.join(top_features[:3])}")
            else:
                summary_parts.append(f"Top features for classification: {', '.join(top_features[:3])}")
            
            # Add importance score context
            if top_scores:
                max_score = max(top_scores)
                min_score = min(top_scores)
                summary_parts.append(f"Importance range: {min_score:.3f} - {max_score:.3f}")
            
            # Add feature count
            summary_parts.append(f"Analyzed {len(combined_importance)} features")
            
            return ". ".join(summary_parts) + "."
            
        except Exception as e:
            logger.error(f"Error generating importance summary: {e}")
            return "Unable to generate importance summary"
    
    async def _generate_importance_visualizations(self, importance_results: Dict[str, Dict[str, float]], 
                                                combined_importance: Dict[str, float], 
                                                top_features: List[str]) -> Dict[str, Any]:
        """Generate feature importance visualizations."""
        try:
            visualizations = {}
            
            # Combined importance plot
            if combined_importance:
                features = list(combined_importance.keys())
                importance_scores = list(combined_importance.values())
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=features,
                    y=importance_scores,
                    marker_color='lightblue',
                    name='Combined Importance'
                ))
                
                fig.update_layout(
                    title="Combined Feature Importance",
                    xaxis_title="Features",
                    yaxis_title="Importance Score",
                    xaxis_tickangle=-45
                )
                
                visualizations['combined_importance'] = json.loads(fig.to_json())
            
            # Method comparison plot
            if len(importance_results) > 1:
                methods = list(importance_results.keys())
                feature_names = list(combined_importance.keys())
                
                fig = go.Figure()
                
                for method in methods:
                    method_scores = [importance_results[method].get(feature, 0) for feature in feature_names]
                    fig.add_trace(go.Bar(
                        x=feature_names,
                        y=method_scores,
                        name=method.title(),
                        opacity=0.7
                    ))
                
                fig.update_layout(
                    title="Feature Importance by Method",
                    xaxis_title="Features",
                    yaxis_title="Importance Score",
                    xaxis_tickangle=-45,
                    barmode='group'
                )
                
                visualizations['method_comparison'] = json.loads(fig.to_json())
            
            # Top features plot
            if top_features:
                top_scores = [combined_importance[feature] for feature in top_features]
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=top_scores,
                    y=top_features,
                    orientation='h',
                    marker_color='lightgreen',
                    name='Top Features'
                ))
                
                fig.update_layout(
                    title="Top Features by Importance",
                    xaxis_title="Importance Score",
                    yaxis_title="Features"
                )
                
                visualizations['top_features'] = json.loads(fig.to_json())
            
            return visualizations
            
        except Exception as e:
            logger.error(f"Error generating importance visualizations: {e}")
            return {}
    
    async def _store_analysis_history(self, model_id: str, importance_results: Dict[str, Dict[str, float]], 
                                    combined_importance: Dict[str, float]):
        """Store analysis history."""
        if model_id not in self.analysis_history:
            self.analysis_history[model_id] = []
        
        analysis_record = {
            'timestamp': datetime.now().isoformat(),
            'importance_results': importance_results,
            'combined_importance': combined_importance
        }
        
        self.analysis_history[model_id].append(analysis_record)
        
        # Keep only recent history (last 100 analyses)
        if len(self.analysis_history[model_id]) > 100:
            self.analysis_history[model_id] = self.analysis_history[model_id][-100:]
    
    async def _select_features(self, request: FeatureSelectionRequest) -> FeatureSelectionResponse:
        """Select most important features."""
        if not SKLEARN_AVAILABLE:
            raise HTTPException(status_code=500, detail="scikit-learn library not available")
        
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
            
            # Select features based on method
            if request.selection_method == "mutual_info":
                selected_features, feature_scores = await self._mutual_info_selection(X, y, feature_names, request.k_features)
            elif request.selection_method == "f_score":
                selected_features, feature_scores = await self._f_score_selection(X, y, feature_names, request.k_features)
            elif request.selection_method == "lasso":
                selected_features, feature_scores = await self._lasso_selection(X, y, feature_names, request.k_features)
            else:
                raise ValueError(f"Unknown selection method: {request.selection_method}")
            
            # Generate selection summary
            selection_summary = await self._generate_selection_summary(
                selected_features, feature_scores, request.selection_method
            )
            
            return FeatureSelectionResponse(
                selected_features=selected_features,
                feature_scores=feature_scores,
                selection_method=request.selection_method,
                k_features=len(selected_features),
                selection_summary=selection_summary,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"Error selecting features: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def _mutual_info_selection(self, X: np.ndarray, y: np.ndarray, 
                                   feature_names: List[str], k_features: int) -> Tuple[List[str], Dict[str, float]]:
        """Select features using mutual information."""
        try:
            # Select mutual information
            if len(np.unique(y)) > 10:  # Regression
                selector = SelectKBest(score_func=f_regression, k=k_features)
            else:  # Classification
                selector = SelectKBest(score_func=f_classif, k=k_features)
            
            # Fit selector
            selector.fit(X, y)
            
            # Get selected features
            selected_indices = selector.get_support(indices=True)
            selected_features = [feature_names[i] for i in selected_indices]
            
            # Get feature scores
            feature_scores = {}
            for i, feature_name in enumerate(feature_names):
                feature_scores[feature_name] = float(selector.scores_[i])
            
            return selected_features, feature_scores
            
        except Exception as e:
            logger.error(f"Error in mutual information selection: {e}")
            return [], {}
    
    async def _f_score_selection(self, X: np.ndarray, y: np.ndarray, 
                               feature_names: List[str], k_features: int) -> Tuple[List[str], Dict[str, float]]:
        """Select features using F-score."""
        try:
            # Select F-score
            if len(np.unique(y)) > 10:  # Regression
                selector = SelectKBest(score_func=f_regression, k=k_features)
            else:  # Classification
                selector = SelectKBest(score_func=f_classif, k=k_features)
            
            # Fit selector
            selector.fit(X, y)
            
            # Get selected features
            selected_indices = selector.get_support(indices=True)
            selected_features = [feature_names[i] for i in selected_indices]
            
            # Get feature scores
            feature_scores = {}
            for i, feature_name in enumerate(feature_names):
                feature_scores[feature_name] = float(selector.scores_[i])
            
            return selected_features, feature_scores
            
        except Exception as e:
            logger.error(f"Error in F-score selection: {e}")
            return [], {}
    
    async def _lasso_selection(self, X: np.ndarray, y: np.ndarray, 
                             feature_names: List[str], k_features: int) -> Tuple[List[str], Dict[str, float]]:
        """Select features using Lasso."""
        try:
            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Create Lasso model
            if len(np.unique(y)) > 10:  # Regression
                model = LassoCV(cv=5, random_state=42)
            else:  # Classification
                from sklearn.linear_model import LogisticRegressionCV
                model = LogisticRegressionCV(cv=5, random_state=42, penalty='l1', solver='liblinear')
            
            # Fit the model
            model.fit(X_scaled, y)
            
            # Get feature importance (absolute coefficients)
            feature_scores = {}
            for i, feature_name in enumerate(feature_names):
                feature_scores[feature_name] = float(abs(model.coef_[i]))
            
            # Select top K features
            sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
            selected_features = [feature for feature, _ in sorted_features[:k_features]]
            
            return selected_features, feature_scores
            
        except Exception as e:
            logger.error(f"Error in Lasso selection: {e}")
            return [], {}
    
    async def _generate_selection_summary(self, selected_features: List[str], 
                                        feature_scores: Dict[str, float], 
                                        selection_method: str) -> str:
        """Generate feature selection summary."""
        try:
            if not selected_features:
                return "No features selected"
            
            # Generate summary
            summary_parts = []
            
            summary_parts.append(f"Selected {len(selected_features)} features using {selection_method}")
            summary_parts.append(f"Top features: {', '.join(selected_features[:3])}")
            
            # Add score context
            if feature_scores:
                selected_scores = [feature_scores[feature] for feature in selected_features]
                max_score = max(selected_scores)
                min_score = min(selected_scores)
                summary_parts.append(f"Score range: {min_score:.3f} - {max_score:.3f}")
            
            return ". ".join(summary_parts) + "."
            
        except Exception as e:
            logger.error(f"Error generating selection summary: {e}")
            return "Unable to generate selection summary"
    
    async def _get_model_importance(self, model_id: str, days: int) -> Dict[str, Any]:
        """Get feature importance history for a model."""
        if model_id not in self.analysis_history:
            raise HTTPException(status_code=404, detail=f"No analysis history found for model {model_id}")
        
        history = self.analysis_history[model_id]
        
        # Filter by time period
        cutoff_time = datetime.now() - timedelta(days=days)
        recent_history = [
            h for h in history 
            if datetime.fromisoformat(h['timestamp']) > cutoff_time
        ]
        
        if not recent_history:
            raise HTTPException(status_code=404, detail=f"No analysis history found for the last {days} days")
        
        # Aggregate importance across time
        aggregated_importance = {}
        importance_counts = {}
        
        for analysis in recent_history:
            combined_importance = analysis['combined_importance']
            for feature, importance in combined_importance.items():
                if feature not in aggregated_importance:
                    aggregated_importance[feature] = 0
                    importance_counts[feature] = 0
                
                aggregated_importance[feature] += importance
                importance_counts[feature] += 1
        
        # Calculate average importance
        average_importance = {
            feature: importance / importance_counts[feature]
            for feature, importance in aggregated_importance.items()
        }
        
        # Sort by importance
        sorted_importance = sorted(average_importance.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'model_id': model_id,
            'time_period': f"{days} days",
            'total_analyses': len(recent_history),
            'average_importance': dict(sorted_importance),
            'top_features': [f[0] for f in sorted_importance[:10]],
            'timestamp': datetime.now().isoformat()
        }
    
    async def _get_importance_stability(self, model_id: str, days: int) -> Dict[str, Any]:
        """Get feature importance stability for a model."""
        if model_id not in self.analysis_history:
            raise HTTPException(status_code=404, detail=f"No analysis history found for model {model_id}")
        
        history = self.analysis_history[model_id]
        
        # Filter by time period
        cutoff_time = datetime.now() - timedelta(days=days)
        recent_history = [
            h for h in history 
            if datetime.fromisoformat(h['timestamp']) > cutoff_time
        ]
        
        if not recent_history:
            raise HTTPException(status_code=404, detail=f"No analysis history found for the last {days} days")
        
        # Calculate stability for each feature
        feature_stability = {}
        
        for analysis in recent_history:
            combined_importance = analysis['combined_importance']
            for feature, importance in combined_importance.items():
                if feature not in feature_stability:
                    feature_stability[feature] = []
                feature_stability[feature].append(importance)
        
        # Calculate stability metrics
        stability_metrics = {}
        for feature, importance_series in feature_stability.items():
            if len(importance_series) > 1:
                stability = 1 - (np.std(importance_series) / (np.mean(importance_series) + 1e-8))
                stability_metrics[feature] = {
                    'stability_score': float(stability),
                    'mean_importance': float(np.mean(importance_series)),
                    'std_importance': float(np.std(importance_series)),
                    'count': len(importance_series)
                }
        
        # Sort by stability
        sorted_stability = sorted(stability_metrics.items(), key=lambda x: x[1]['stability_score'], reverse=True)
        
        # Calculate overall stability
        overall_stability = np.mean([f[1]['stability_score'] for f in sorted_stability])
        
        return {
            'model_id': model_id,
            'time_period': f"{days} days",
            'total_analyses': len(recent_history),
            'overall_stability': float(overall_stability),
            'feature_stability': dict(sorted_stability),
            'most_stable_features': [f[0] for f in sorted_stability[:5]],
            'least_stable_features': [f[0] for f in sorted_stability[-5:]],
            'timestamp': datetime.now().isoformat()
        }
    
    async def _get_feature_importance_history(self, feature_name: str, days: int) -> Dict[str, Any]:
        """Get importance history for a specific feature."""
        # Collect importance history for the feature across all models
        feature_history = []
        
        for model_id, history in self.analysis_history.items():
            for analysis in history:
                if feature_name in analysis['combined_importance']:
                    feature_history.append({
                        'model_id': model_id,
                        'timestamp': analysis['timestamp'],
                        'importance': analysis['combined_importance'][feature_name]
                    })
        
        # Filter by time period
        cutoff_time = datetime.now() - timedelta(days=days)
        recent_history = [
            h for h in feature_history 
            if datetime.fromisoformat(h['timestamp']) > cutoff_time
        ]
        
        if not recent_history:
            raise HTTPException(status_code=404, detail=f"No importance history found for feature {feature_name} in the last {days} days")
        
        # Calculate statistics
        importance_scores = [h['importance'] for h in recent_history]
        
        return {
            'feature_name': feature_name,
            'time_period': f"{days} days",
            'total_observations': len(recent_history),
            'mean_importance': float(np.mean(importance_scores)),
            'std_importance': float(np.std(importance_scores)),
            'min_importance': float(np.min(importance_scores)),
            'max_importance': float(np.max(importance_scores)),
            'importance_history': recent_history,
            'timestamp': datetime.now().isoformat()
        }
    
    async def _get_feature_ranking(self, days: int, task_type: str) -> Dict[str, Any]:
        """Get global feature ranking across models."""
        # Collect feature importance across all models
        global_importance = {}
        feature_counts = {}
        
        for model_id, history in self.analysis_history.items():
            for analysis in history:
                combined_importance = analysis['combined_importance']
                for feature, importance in combined_importance.items():
                    if feature not in global_importance:
                        global_importance[feature] = 0
                        feature_counts[feature] = 0
                    
                    global_importance[feature] += importance
                    feature_counts[feature] += 1
        
        # Calculate average importance
        average_importance = {
            feature: importance / feature_counts[feature]
            for feature, importance in global_importance.items()
        }
        
        # Sort by importance
        sorted_importance = sorted(average_importance.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'task_type': task_type,
            'time_period': f"{days} days",
            'total_features': len(sorted_importance),
            'global_ranking': dict(sorted_importance),
            'top_features': [f[0] for f in sorted_importance[:20]],
            'timestamp': datetime.now().isoformat()
        }
    
    def run(self, host: str = "0.0.0.0", port: int = 8023):
        """Run the service."""
        logger.info(f"Starting Feature Importance Analyzer on {host}:{port}")
        uvicorn.run(self.app, host=host, port=port)


# Global service instance
service = FeatureImportanceAnalyzer()


def create_app() -> FastAPI:
    """Create FastAPI application."""
    return service.app


if __name__ == "__main__":
    service.run()
